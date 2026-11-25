# Capture golden inputs/outputs for the 4 ONNX graphs from a real propagation run.
# Runs PyTorch in pure fp32 (no autocast) = the reference numerics for parity checks.
import os, shutil
import numpy as np
import torch

# reference numerics: disable TF32 so golden is true fp32
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

REPO = os.path.join(os.path.dirname(__file__), "..", "vendor", "Efficient-SAM2")
CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "sam2.1_hiera_tiny.pt")
GOLD = os.path.join(os.path.dirname(__file__), "..", "golden18")
os.makedirs(GOLD, exist_ok=True)

from sam2.build_sam import build_sam2_video_predictor

# 18-frame video from the bedroom demo sequence (covers memattn variants up to steady state)
frame_dir = os.path.join(GOLD, "video18")
os.makedirs(frame_dir, exist_ok=True)
bedroom = os.path.join(REPO, "notebooks", "videos", "bedroom")
for i, f in enumerate(sorted(os.listdir(bedroom))[:18]):
    shutil.copy(os.path.join(bedroom, f), os.path.join(frame_dir, f"{i:05d}.jpg"))

predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", CKPT, device="cuda")
predictor.init_memory_info(enable_MeP_info=False)
predictor.Mem_Frame_Prune = False
predictor.time_log = {}
for m in (predictor.image_encoder, predictor.memory_attention, predictor.memory_encoder, predictor.sam_mask_decoder):
    m.FW_time_log = []

records = []

# ---- recording patches -------------------------------------------------
model = predictor  # SAM2VideoPredictor (subclass of SAM2Base)

orig_fwd_image = model.forward_image
def rec_fwd_image(img):
    out = orig_fwd_image(img)
    records.append(("image_encoder", {"image": img}, {
        "fpn0": out["backbone_fpn"][0], "fpn1": out["backbone_fpn"][1], "fpn2": out["backbone_fpn"][2],
        "pos0": out["vision_pos_enc"][0], "pos1": out["vision_pos_enc"][1], "pos2": out["vision_pos_enc"][2]}))
    return out
model.forward_image = rec_fwd_image

orig_ma_fwd = model.memory_attention.forward
def rec_ma_fwd(curr, curr_pos=None, memory=None, memory_pos=None, num_obj_ptr_tokens=0):
    out = orig_ma_fwd(curr=curr, curr_pos=curr_pos, memory=memory, memory_pos=memory_pos,
                      num_obj_ptr_tokens=num_obj_ptr_tokens)
    c = curr[0] if isinstance(curr, list) else curr
    cp = curr_pos[0] if isinstance(curr_pos, list) else curr_pos
    # record in the same BCHW layout the ONNX graph outputs
    out_bchw = out.permute(1, 2, 0).view(1, 256, 64, 64)
    records.append(("memattn", {"curr": c, "curr_pos": cp, "memory": memory, "memory_pos": memory_pos,
                                "num_obj_ptr_tokens": num_obj_ptr_tokens}, {"pix_feat_with_mem": out_bchw}))
    return out
model.memory_attention.forward = rec_ma_fwd

import sam2.modeling.sam2_base as sb
orig_enc_new_mem = sb.SAM2Base._encode_new_memory
def rec_enc_new_mem(self, current_vision_feats, feat_sizes, pred_masks_high_res, object_score_logits, is_mask_from_pts):
    mfeat, mpos = orig_enc_new_mem(self, current_vision_feats, feat_sizes, pred_masks_high_res,
                                   object_score_logits, is_mask_from_pts)
    records.append(("memenc", {"pix_feat": current_vision_feats[-1].permute(1, 2, 0).view(
                                    1, self.hidden_dim, *feat_sizes[-1]),
                               "pred_masks_high_res": pred_masks_high_res,
                               "object_score_logits": object_score_logits,
                               "is_mask_from_pts": bool(is_mask_from_pts)},
                    {"maskmem_features": mfeat, "maskmem_pos_enc": mpos[0] if isinstance(mpos, list) else mpos}))
    return mfeat, mpos
sb.SAM2Base._encode_new_memory = rec_enc_new_mem

orig_sam_heads = sb.SAM2Base._forward_sam_heads
def rec_sam_heads(self, backbone_features, point_inputs=None, mask_inputs=None, high_res_features=None, multimask_output=False):
    outs = orig_sam_heads(self, backbone_features=backbone_features, point_inputs=point_inputs,
                          mask_inputs=mask_inputs, high_res_features=high_res_features,
                          multimask_output=multimask_output)
    low_res_multimasks, high_res_multimasks, ious, low_res_masks, high_res_masks, obj_ptr, obj_score = outs
    records.append(("samheads", {"backbone_features": backbone_features,
                                 "point_coords": None if point_inputs is None else point_inputs["point_coords"],
                                 "point_labels": None if point_inputs is None else point_inputs["point_labels"],
                                 "fpn0": high_res_features[0], "fpn1": high_res_features[1],
                                 "multimask_output": bool(multimask_output)},
                    {"low_res_multimasks": low_res_multimasks, "high_res_multimasks": high_res_multimasks,
                     "ious": ious, "low_res_masks": low_res_masks, "high_res_masks": high_res_masks,
                     "obj_ptr": obj_ptr, "object_score_logits": obj_score}))
    return outs
sb.SAM2Base._forward_sam_heads = rec_sam_heads

# ---- run: frame 0 box prompt + propagate --------------------------------
with torch.inference_mode():  # NOTE: pure fp32, no autocast = reference numerics
    state = predictor.init_state(video_path=frame_dir)
    print('NUM_FRAMES =', state['num_frames'])
    box = np.array([210, 350, 400, 550], dtype=np.float32)  # child in bedroom demo
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, box=box)
    n_fr = 0
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        n_fr += 1
print(f'propagated {n_fr} frames, captured {len(records)} graph calls')
import collections
print(collections.Counter(r[0] for r in records))

def to_np(t):
    return t.detach().float().cpu().numpy() if torch.is_tensor(t) else t

# ---- save golden npz per call -------------------------------------------
summary = []
for i, (graph, ins, outs) in enumerate(records):
    data = {}
    meta = {}
    for k, v in ins.items():
        if torch.is_tensor(v):
            data["in_" + k] = to_np(v)
        elif v is not None:
            meta[k] = v
    for k, v in outs.items():
        if torch.is_tensor(v):
            data["out_" + k] = to_np(v)
    path = os.path.join(GOLD, f"{graph}_{i:03d}.npz")
    np.savez(path, **data, **{("meta_" + k): np.array(v) for k, v in meta.items()})
    summary.append((os.path.basename(path), {k: tuple(v.shape) for k, v in data.items() if k.startswith("in_")}, meta))

for name, ishapes, meta in summary:
    print(name, {k.split("in_")[1]: v for k, v in ishapes.items()}, meta if meta else "")
print("GOLDEN SAVED")
