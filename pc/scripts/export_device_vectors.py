# Export golden npz cases to device-friendly raw .bin + JSON manifests.
# Layout:  device_vectors/<case_name>/<tensor_key>.bin  + case.json
#          device_vectors/manifest.json  (all cases)
# Model mapping:
#   image_encoder_* -> image_encoder
#   memattn_*       -> memattn_kXX / memattn_steady, via N=in_memory.shape[0]:
#                      N=4096*(1+min(k-1,6)) + 4*(1+min(k-1,15))
#                      k = N/4100 (N<=24600) else k = (N-28672)/4 ; k>=16 -> steady
#   memenc_*        -> memenc_binarize if meta_is_mask_from_pts else memenc_sigmoid
#   samheads_*      -> samheads_cond if in_point_coords present else samheads_track
import glob
import json
import os

import numpy as np

GOLD = os.path.join(os.path.dirname(__file__), "..", "golden")
OUT = os.path.join(os.path.dirname(__file__), "..", "device_vectors")
os.makedirs(OUT, exist_ok=True)


def memattn_variant(n):
    if n <= 24600:
        k = n // 4100
        assert n == 4100 * k, f"bad N {n}"
    else:
        k = (n - 28672) // 4
        assert 28672 + 4 * k == n, f"bad N {n}"
    return "memattn_steady" if k >= 16 else f"memattn_k{k:02d}"


cases = []
for path in sorted(glob.glob(os.path.join(GOLD, "*.npz"))):
    base = os.path.splitext(os.path.basename(path))[0]
    graph = base.rsplit("_", 1)[0]
    d = np.load(path)

    if graph == "image_encoder":
        model = "image_encoder"
    elif graph == "memattn":
        model = memattn_variant(int(d["in_memory"].shape[0]))
    elif graph == "memenc":
        model = "memenc_binarize" if bool(d["meta_is_mask_from_pts"]) else "memenc_sigmoid"
    elif graph == "samheads":
        model = "samheads_cond" if "in_point_coords" in d.files else "samheads_track"
    else:
        raise ValueError(f"unknown graph {graph}")

    cdir = os.path.join(OUT, base)
    os.makedirs(cdir, exist_ok=True)
    tensors, metas = {}, {}
    lines = []
    for key in d.files:
        # memenc 的 maskmem_pos_enc 跨帧恒定，已从模型裁掉、烘焙为常量
        # (outputs/ms/maskmem_pos_enc.bin)，不再作为用例输出对拍。
        if graph == "memenc" and key == "out_maskmem_pos_enc":
            continue
        # samheads_cond 模型做了切尾（M2），不输出 low/high_res_masks；
        # cond 帧为单 mask，multimasks 即 masks，由 C++ 胶水取用。
        if model == "samheads_cond" and key in ("out_low_res_masks", "out_high_res_masks"):
            continue
        arr = np.ascontiguousarray(d[key])
        # point_labels 在 ONNX/.ms 里是 int32；golden npz 里存的是 float32 值，
        # 直接 tofile 会把 0x3F800000 当 int 读 → 标签全乱 → prompt 路径算错。
        if key.endswith("point_labels"):
            arr = arr.astype(np.int32)
        # samheads_cond 用默认转换（NHWC I/O）：NCHW 转换会算错 prompt 路径。
        # 4D 张量一律按 NHWC 排布导出（bin 数据与 shape 同步转置）。
        if model == "samheads_cond" and arr.ndim == 4:
            arr = np.ascontiguousarray(arr.transpose(0, 2, 3, 1))
        if key.startswith("meta_"):
            metas[key[5:]] = arr.item() if arr.ndim == 0 else arr.tolist()
            continue
        fn = f"{key}.bin"
        arr.tofile(os.path.join(cdir, fn))
        tensors[key] = {"file": fn, "shape": list(arr.shape), "dtype": str(arr.dtype)}
        # C++-friendly flat line: tensor <key> <dtype> <ndim> <d0..dn> <file>
        lines.append(f"tensor {key} {arr.dtype} {arr.ndim} "
                     f"{' '.join(str(s) for s in arr.shape)} {fn}")
    with open(os.path.join(cdir, "case.json"), "w") as f:
        json.dump({"case": base, "model": model, "tensors": tensors, "meta": metas}, f, indent=1)
    for mk, mv in metas.items():
        lines.append(f"meta {mk} {mv}")
    with open(os.path.join(cdir, "case.txt"), "w", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
    cases.append({"case": base, "model": model, "dir": base})
    print(f"{base:24s} -> {model}")

with open(os.path.join(OUT, "manifest.json"), "w") as f:
    json.dump({"cases": cases}, f, indent=1)
with open(os.path.join(OUT, "manifest.txt"), "w", newline="\n") as f:
    for c in cases:
        f.write(f"case {c['dir']} {c['model']}\n")
print(f"\n{len(cases)} cases exported to {os.path.abspath(OUT)}")

# quick sanity: which memattn variants are covered
cov = sorted({c["model"] for c in cases if c["model"].startswith("memattn")})
print("memattn variants covered:", cov)
