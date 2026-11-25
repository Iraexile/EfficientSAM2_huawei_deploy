# M1 step 3: single-frame GPU forward smoke test
# Video predictor: 1 frame (truck.jpg) + box prompt -> mask overlay
import os
import shutil
import sys

import numpy as np
import torch
from PIL import Image

REPO = os.path.join(os.path.dirname(__file__), "..", "vendor", "Efficient-SAM2")
CKPT = os.path.join(os.path.dirname(__file__), "..", "checkpoints", "sam2.1_hiera_tiny.pt")
OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")
CFG = "configs/sam2.1/sam2.1_hiera_t.yaml"

os.makedirs(OUT, exist_ok=True)

# prepare a 1-frame "video" directory
frame_dir = os.path.join(OUT, "single_frame_video")
os.makedirs(frame_dir, exist_ok=True)
src_img = os.path.join(REPO, "notebooks", "images", "truck.jpg")
shutil.copy(src_img, os.path.join(frame_dir, "00000.jpg"))

from sam2.build_sam import build_sam2_video_predictor

device = "cuda"
print("building predictor ...")
predictor = build_sam2_video_predictor(CFG, CKPT, device=device)
# Efficient-SAM2 fork: must init SWR/SMR memory info (baseline = MeP off)
predictor.init_memory_info(enable_MeP_info=False)
# fork expects external profiling dicts (used in tools/vos_inference.py)
predictor.time_log = {}
predictor.image_encoder.FW_time_log = []
predictor.memory_attention.FW_time_log = []
predictor.memory_encoder.FW_time_log = []
predictor.sam_mask_decoder.FW_time_log = []

# bf16 autocast is the repo's standard inference path on modern GPUs
with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
    state = predictor.init_state(video_path=frame_dir)
    # box around the truck (from the official image_predictor notebook demo)
    box = np.array([500, 382, 1100, 720], dtype=np.float32)
    _, obj_ids, mask_logits = predictor.add_new_points_or_box(
        inference_state=state, frame_idx=0, obj_id=1, box=box
    )
    print("propagating ...")
    results = list(predictor.propagate_in_video(state))

print("propagation results:", [(f, list(o)) for f, o, _ in results])
frame_idx, obj_ids, mask_logits = results[0]
mask = (mask_logits[0] > 0.0).cpu().numpy().squeeze()
print("mask shape:", mask.shape, "mask area(px):", int(mask.sum()))
assert mask.sum() > 1000, "mask suspiciously small - something went wrong"

# save overlay
img = np.array(Image.open(src_img).convert("RGB")).astype(np.float32)
overlay = img.copy()
overlay[mask] = 0.35 * img[mask] + 0.65 * np.array([30, 144, 255], dtype=np.float32)
out_path = os.path.join(OUT, "single_frame_overlay.png")
Image.fromarray(overlay.astype(np.uint8)).save(out_path)
print("overlay saved:", out_path)
print("SMOKE TEST PASSED")
