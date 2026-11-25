# Probe: why does propagate_in_video yield 11 frames on an 18-frame dir?
import os
import torch
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False

REPO = os.path.join(os.path.dirname(__file__), "..", "vendor", "Efficient-SAM2")
CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "sam2.1_hiera_tiny.pt")
GOLD = os.path.join(os.path.dirname(__file__), "..", "golden")
frame_dir = os.path.join(GOLD, "video18")
print("jpgs:", sorted(os.listdir(frame_dir)))

from sam2.build_sam import build_sam2_video_predictor
import numpy as np

predictor = build_sam2_video_predictor("configs/sam2.1/sam2.1_hiera_t.yaml", CKPT, device="cuda")
predictor.init_memory_info(enable_MeP_info=False)
predictor.Mem_Frame_Prune = False
predictor.time_log = {}
for m in (predictor.image_encoder, predictor.memory_attention, predictor.memory_encoder, predictor.sam_mask_decoder):
    m.FW_time_log = []

with torch.inference_mode():
    state = predictor.init_state(video_path=frame_dir)
    print("num_frames =", state["num_frames"])
    print("images shape =", state["images"].shape)
    box = np.array([210, 350, 400, 550], dtype=np.float32)
    predictor.add_new_points_or_box(inference_state=state, frame_idx=0, obj_id=1, box=box)
    n = 0
    for frame_idx, obj_ids, mask_logits in predictor.propagate_in_video(state):
        print("yield frame", frame_idx)
        n += 1
    print("total yields:", n)
