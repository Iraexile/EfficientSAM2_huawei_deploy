# 验证 samheads_001 用例自洽性：NHWC 数据 -> 转回 NCHW -> 跑 ONNX -> 与 golden(NHWC) 比
import json, numpy as np, onnxruntime as ort, os, sys

case_dir = r"C:\path\to\EfficientSAM2\pc\device_vectors\samheads_001"
onnx = r"C:\path\to\EfficientSAM2\pc\outputs\onnx\samheads_cond_fix.onnx"
if not os.path.exists(onnx):
    import glob
    cands = glob.glob(r"C:\path\to\EfficientSAM2\pc\outputs\onnx\*cond*")
    print("onnx candidates:", cands); sys.exit(1)

meta = json.load(open(os.path.join(case_dir, "case.json")))
def load(name):
    t = meta["tensors"][name]
    a = np.fromfile(os.path.join(case_dir, t["file"]), dtype=np.float32).reshape(t["shape"])
    return a

# 输入:NHWC 存储 -> NCHW 喂 ONNX(4D 的)
ins = {}
for n in ["in_backbone_features", "in_fpn0", "in_fpn1"]:
    a = load(n)
    ins[n.replace("in_", "")] = np.ascontiguousarray(a.transpose(0, 3, 1, 2))
ins["point_coords"] = load("in_point_coords")
ins["point_labels"] = load("in_point_labels").astype(np.int32)

sess = ort.InferenceSession(onnx, providers=["CPUExecutionProvider"])
input_names = [i.name for i in sess.get_inputs()]
print("onnx inputs:", input_names)
feed = {}
for k, v in ins.items():
    for cand in [k, "in_" + k]:
        if cand in input_names:
            feed[cand] = v
missing = [n for n in input_names if n not in feed]
if missing:
    print("MISSING inputs:", missing); sys.exit(1)
outs = sess.run(None, feed)
out_names = [o.name for o in sess.get_outputs()]
print("onnx outputs:", out_names)

for name, val in zip(out_names, outs):
    key = "out_" + name
    if key not in meta["tensors"]:
        print(f"{name}: (no golden, skipped) shape={val.shape}")
        continue
    g = load(key)
    v = val
    if v.ndim == 4:  # NCHW -> NHWC 与存储对齐
        v = np.ascontiguousarray(v.transpose(0, 2, 3, 1))
    if v.shape != g.shape:
        print(f"{name}: SHAPE MISMATCH onnx{v.shape} golden{g.shape}")
        continue
    d = np.abs(v - g).max()
    cos = float(np.dot(v.ravel(), g.ravel()) / (np.linalg.norm(v) * np.linalg.norm(g) + 1e-30))
    print(f"{name}: maxdiff={d:.3e} cos={cos:.6f}")
