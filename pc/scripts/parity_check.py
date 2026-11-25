# Parity check: onnxruntime (CPU, fp32) vs PyTorch golden tensors (atol/rtol 1e-4)
import glob
import os
import re

import numpy as np
import onnxruntime as ort

GOLD = os.path.join(os.path.dirname(__file__), "..", "golden")
ONNX_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "onnx")

ATOL = 1e-4
RTOL = 1e-4

sessions = {}


def get_session(name):
    if name not in sessions:
        sessions[name] = ort.InferenceSession(
            os.path.join(ONNX_DIR, name + ".onnx"), providers=["CPUExecutionProvider"]
        )
    return sessions[name]


def memattn_variant(d):
    N = d["in_memory"].shape[0]
    n_ptr = int(d["meta_num_obj_ptr_tokens"])
    m = (N - n_ptr) // 4096
    p = n_ptr // 4
    if p >= 16:
        return "memattn_steady"
    return f"memattn_k{max(m, p):02d}"


def graph_for(name, d):
    if name == "image_encoder":
        return "image_encoder"
    if name == "memattn":
        return memattn_variant(d)
    if name == "memenc":
        return "memenc_binarize" if bool(d["meta_is_mask_from_pts"]) else "memenc_sigmoid"
    if name == "samheads":
        return "samheads_cond" if "in_point_coords" in d.files else "samheads_track"
    raise ValueError(name)


results = []
for path in sorted(glob.glob(os.path.join(GOLD, "*.npz"))):
    base = os.path.basename(path)
    graph_type = base.rsplit("_", 1)[0]
    d = np.load(path)
    onnx_name = graph_for(graph_type, d)
    sess = get_session(onnx_name)

    feeds = {}
    for inp in sess.get_inputs():
        key = "in_" + inp.name
        if key not in d.files:
            print(f"!! {base}: missing input {inp.name}")
            continue
        arr = d[key]
        if inp.type == "tensor(int32)":
            arr = arr.astype(np.int32)
        elif inp.type == "tensor(int64)":
            arr = arr.astype(np.int64)
        feeds[inp.name] = arr

    outs = sess.run(None, feeds)
    out_names = [o.name for o in sess.get_outputs()]

    worst = 0.0
    worst_key = ""
    detail = []
    hard_fail = False
    for oname, oval in zip(out_names, outs):
        key = "out_" + oname
        if key not in d.files:
            detail.append(f"{oname}: NO GOLDEN")
            hard_fail = True
            continue
        ref = d[key].astype(np.float64)
        val = oval.astype(np.float64)
        if ref.shape != val.shape:
            detail.append(f"{oname}: SHAPE MISMATCH onnx{val.shape} vs golden{ref.shape}")
            hard_fail = True
            continue
        diff = np.abs(val - ref)
        tol = ATOL + RTOL * np.abs(ref)
        max_abs = float(diff.max()) if diff.size else 0.0
        max_rel = float((diff / (np.abs(ref) + 1e-12)).max()) if diff.size else 0.0
        ok = bool(np.all(diff <= tol))
        if max_abs > worst:
            worst, worst_key = max_abs, oname
        detail.append(f"{oname}: max|Δ|={max_abs:.2e} max_rel={max_rel:.2e} {'OK' if ok else 'FAIL'}")
        if not ok:
            hard_fail = True
    status = "CHECK" if hard_fail else "PASS"
    results.append((base, onnx_name, worst, worst_key, status, detail))
    wstr = f"{worst:.2e}" if np.isfinite(worst) else "inf"
    print(f"{base} -> {onnx_name}: worst={wstr} ({worst_key}) {status}")

n_fail = sum(1 for r in results if r[4] != "PASS")
print(f"\n{len(results) - n_fail}/{len(results)} calls within tolerance")
for base, onnx_name, worst, wk, status, detail in results:
    if status != "PASS":
        print(f"\n--- {base} -> {onnx_name} ---")
        for x in detail:
            print("   ", x)
