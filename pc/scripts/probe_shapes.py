# Probe concrete tensor shapes across a 3-frame propagation (baseline config)
import os, shutil, sys
import numpy as np
import torch

REPO = os.path.join(os.path.dirname(__file__), "..", "vendor", "Efficient-SAM2")
CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "sam2.1_hiera_tiny.pt")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")

from sam2.build_sam import build_sam2_video_predictor

# 3-frame video dir: truck + 2 bedroom frames for variety
frame_dir = os.path.join(OUT, "probe_video")
os.makedirs(frame_dir, exist_ok=True)
bedroom = os.path.join(REPO, "notebooks", "videos", "bedroom")
frames = sorted(os.listdir(bedroom))[:2]
shutil.copy(os.path.join(REPO, "notebooks", "images", "truck.jpg"), os.path.join(frame_dir, "00000.jpg"))
for i, f in enumerate(frames, start=1):
    shutil.copy(os.path.join(bedroom, f), os.path.join(frame_dir, f"{i:05d}.jpg"))

predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", CKPT, device="cuda")
predictor.init_memory_info(enable_MeP_info=False)
predictor.Mem_Frame_Prune = False  # SMR off for baseline
predictor.time_log = {}
predictor.image_encoder.FW_time_log = []
predictor.memory_attention.FW_time_log = []
predictor.memory_encoder.FW_time_log = []
predictor.sam_mask_decoder.FW_time_log = []

def show(name, t):
    if torch.is_tensor(t):
        print(f"  {name}: shape={tuple(t.shape)} dtype={t.dtype}")
    elif isinstance(t, (list, tuple)):
        for i, x in enumerate(t):
            show(f"{name}[{i}]", x)
    elif isinstance(t, dict):
        for k, v in t.items():
            show(f"{name}.{k}", v)
    else:
        print(f"  {name}: {type(t).__name__} = {t if not callable(t) else 'callable'}")

with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=frame_dir)
    # raw image encoder output shape (bypass predictor cache wrapper)
    img = state["images"][0].to("cuda").float().unsqueeze(0)
    backbone_out = predictor.forward_image(img)
    print("== forward_image output ==")
    show("backbone_out", backbone_out)

    box = np.array([500, 382, 1100, 720], dtype=np.float32)
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, box=box)
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        print(f"== frame {frame_idx} propagated ==")

    obj_idx = state["obj_id_to_idx"][1]
    od = state["output_dict_per_obj"][obj_idx]
    print("== cond_frame_outputs ==")
    for t, out in od["cond_frame_outputs"].items():
        print(f" frame {t}:"); show("out", out)
    print("== non_cond_frame_outputs ==")
    for t, out in od["non_cond_frame_outputs"].items():
        print(f" frame {t}:"); show("out", out)
    print("== constants ==")
    show("constants", state["constants"])
    print("== time_log keys ==", {k: list(v.keys()) for k, v in predictor.time_log.items()})
print("PROBE DONE")
