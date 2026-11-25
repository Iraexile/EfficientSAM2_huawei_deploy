# Parity check for the MSLite-target ONNX variants (post surgery/simplification).
# Validates that the export patches + tail cut + onnxsim + fixup chain is
# numerically identical to the golden PyTorch run.
#  - samheads_cond_fix: obj_ptr is RAW (pre no_obj combine); combine in numpy.
#    low/high_res_masks dropped (exact aliases of *_multimasks for M=1).
#  - samheads_track_fix: direct comparison.
import glob
import os

import numpy as np
import onnxruntime as ort

GOLD = os.path.join(os.path.dirname(__file__), "..", "golden")
ONNX_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "onnx")
NOBJ = os.path.join(os.path.dirname(__file__), "..", "outputs", "ms", "no_obj_ptr.npy")
ATOL = RTOL = 1e-4

no_obj_ptr = np.load(NOBJ)  # [1,256]

sess = {
    "cond": ort.InferenceSession(os.path.join(ONNX_DIR, "samheads_cond_fix.onnx"),
                                 providers=["CPUExecutionProvider"]),
    "track": ort.InferenceSession(os.path.join(ONNX_DIR, "samheads_track_fix.onnx"),
                                  providers=["CPUExecutionProvider"]),
}

results = []
for path in sorted(glob.glob(os.path.join(GOLD, "samheads_*.npz"))):
    base = os.path.basename(path)
    d = np.load(path)
    kind = "cond" if "in_point_coords" in d.files else "track"
    s = sess[kind]
    feeds = {}
    for inp in s.get_inputs():
        arr = d["in_" + inp.name]
        if inp.type == "tensor(int32)":
            arr = arr.astype(np.int32)
        elif inp.type == "tensor(int64)":
            arr = arr.astype(np.int64)
        feeds[inp.name] = arr
    outs = dict(zip([o.name for o in s.get_outputs()], s.run(None, feeds)))

    # cond: compose obj_ptr exactly as _forward_sam_heads does (fp32)
    if kind == "cond":
        lam = (outs["object_score_logits"] > 0).astype(np.float32)  # [1,1]
        outs["obj_ptr"] = lam * outs["obj_ptr"] + (1 - lam) * no_obj_ptr
        # aliases dropped in the graph: low/high_res_masks == *_multimasks (M=1)
        outs["low_res_masks"] = outs["low_res_multimasks"]
        outs["high_res_masks"] = outs["high_res_multimasks"]

    worst, worst_key, fails = 0.0, "", []
    for key in [k for k in d.files if k.startswith("out_")]:
        name = key[4:]
        if name not in outs:
            fails.append(f"{name}: MISSING IN ONNX OUTPUTS")
            continue
        ref = d[key].astype(np.float64)
        val = outs[name].astype(np.float64)
        if ref.shape != val.shape:
            fails.append(f"{name}: SHAPE {val.shape} vs {ref.shape}")
            continue
        diff = np.abs(val - ref)
        tol = ATOL + RTOL * np.abs(ref)
        m = float(diff.max()) if diff.size else 0.0
        if m > worst:
            worst, worst_key = m, name
        if not np.all(diff <= tol):
            fails.append(f"{name}: max|d|={m:.2e}")
    status = "PASS" if not fails else "FAIL"
    results.append((base, kind, worst, worst_key, status, fails))
    print(f"{base} ({kind}_fix): worst={worst:.2e} ({worst_key}) {status}")
    for f in fails:
        print("   ", f)

n_fail = sum(1 for r in results if r[4] != "PASS")
print(f"\n{len(results) - n_fail}/{len(results)} mslite-variant calls within tolerance")
