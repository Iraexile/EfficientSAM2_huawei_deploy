# Bisect the image_encoder graph for the device-side divergence.
# For each cut: extract sub-graph onnx, compute golden intermediate with
# onnxruntime (fp32, TF32 off), export a device-vector case dir.
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx.utils import extract_model

PC = os.path.join(os.path.dirname(__file__), "..")
ONNX = os.path.join(PC, "outputs", "onnx", "image_encoder.onnx")
GOLD_IN = os.path.join(PC, "device_vectors", "image_encoder_000", "in_image.bin")
OUT = os.path.join(PC, "device_vectors")

# cut name -> intermediate tensor name (output of the sub-graph)
CUTS = {
    # patch embed conv (4x4s4) + transpose + add pos
    "cut_embed": "/enc/trunk/Add_output_0",
    # after block 0
    "cut_blk0": "/enc/trunk/blocks.0/Add_2_output_0",
}

m = onnx.load(ONNX)
names = {n.name for n in m.graph.node}
# find real tensor names near the desired cut points
all_tensors = set()
for n in m.graph.node:
    all_tensors.update(n.output)

img = np.fromfile(GOLD_IN, dtype=np.float32).reshape(1, 3, 1024, 1024)

for cut, tensor in CUTS.items():
    if tensor not in all_tensors:
        # fallback: find closest prefix match
        cands = sorted(t for t in all_tensors if t.startswith(tensor.rsplit("/", 1)[0]))[:10]
        print(f"{cut}: {tensor} NOT FOUND, candidates: {cands}")
        continue
    sub_path = os.path.join(PC, "outputs", "onnx", f"enc_{cut}.onnx")
    extract_model(ONNX, sub_path, ["image"], [tensor])
    sess = ort.InferenceSession(sub_path, providers=["CPUExecutionProvider"])
    out = sess.run(None, {"image": img})[0]
    out = np.ascontiguousarray(out.astype(np.float32))
    cdir = os.path.join(OUT, f"enc_{cut}_000")
    os.makedirs(cdir, exist_ok=True)
    img.tofile(os.path.join(cdir, "in_image.bin"))
    out.tofile(os.path.join(cdir, "out_feat.bin"))
    with open(os.path.join(cdir, "case.txt"), "w", newline="\n") as f:
        f.write(f"tensor in_image float32 4 1 3 1024 1024 in_image.bin\n")
        f.write(f"tensor out_feat float32 {out.ndim} {' '.join(map(str, out.shape))} out_feat.bin\n")
    print(f"{cut}: sub-graph + golden saved, out shape {out.shape}")
