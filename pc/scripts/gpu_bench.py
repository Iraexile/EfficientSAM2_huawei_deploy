# PC GPU (CUDA) 逐图计时：与手机 CPU 基线同口径（71 例 + memattn 逐档）
# 用法: python gpu_bench.py
import json, os, sys, time
import numpy as np

# onnxruntime-gpu 1.22 需要 CUDA 12 + cuDNN 9 的 DLL，借用 torch 自带的那套
_TORCH_LIB = os.path.join(os.path.dirname(sys.executable), "..", "Lib",
                          "site-packages", "torch", "lib")
if os.path.isdir(_TORCH_LIB):
    os.add_dll_directory(os.path.abspath(_TORCH_LIB))
import onnxruntime as ort

ROOT = r"C:\path\to\EfficientSAM2\pc"
ONNX = os.path.join(ROOT, "outputs", "onnx")          # 21 个 onnx 所在目录（含子目录也行，递归找）
VECTORS = os.path.join(ROOT, "device_vectors")        # 71 例设备用例（与手机同一份输入）
OUT = os.path.join(ROOT, "outputs", "gpu_bench.json")

WARMUP = 3
REPEAT = 10

def find_onnx(name):
    for dp, _, fns in os.walk(ONNX):
        for fn in fns:
            if fn == name + ".onnx":
                return os.path.join(dp, fn)
    return None

def load_case(case_dir):
    """解析 case.txt（tensor <key> <dtype> <ndim> <dims...> <file>）"""
    tensors = {}
    with open(os.path.join(case_dir, "case.txt"), encoding="utf-8") as f:
        for line in f:
            parts = line.split()
            if len(parts) < 4 or parts[0] != "tensor":
                continue
            key, dtype = parts[1], parts[2]
            ndim = int(parts[3])
            shape = [int(x) for x in parts[4:4 + ndim]]
            fname = parts[4 + ndim]
            dt = {"float32": np.float32, "float16": np.float16,
                  "int32": np.int32, "int64": np.int64}[dtype]
            raw = np.fromfile(os.path.join(case_dir, fname), dtype=dt).reshape(shape)
            tensors[key] = raw
    ins = {k[3:]: v for k, v in tensors.items() if k.startswith("in_")}
    outs = {k[4:]: v for k, v in tensors.items() if k.startswith("out_")}
    return ins, outs

def main():
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    results = []
    cases = []
    with open(os.path.join(VECTORS, "manifest.txt"), encoding="utf-8") as f:
        for line in f:
            p = line.split()
            if len(p) == 3 and p[0] == "case":
                cases.append((p[1], p[2]))   # (用例目录, 模型名)
    print(f"cases: {len(cases)}")
    for case, model in cases:
        cdir = os.path.join(VECTORS, case)
        onnx_path = find_onnx(model)
        if not onnx_path:
            print(f"  {case}: model {model} onnx not found, skip")
            continue
        try:
            ins, _ = load_case(cdir)
        except Exception as e:
            print(f"  {case}: load fail {e}")
            continue
        so = ort.SessionOptions()
        so.log_severity_level = 3
        try:
            sess = ort.InferenceSession(onnx_path, so, providers=providers)
        except Exception as e:
            print(f"  {case}: session fail {e}")
            results.append({"case": case, "model": model, "error": str(e)})
            continue
        feed = {}
        ok = True
        for inp in sess.get_inputs():
            if inp.name in ins:
                arr = ins[inp.name]
                want = inp.type
                if "float16" in want and arr.dtype != np.float16:
                    arr = arr.astype(np.float16)
                elif "float" in want and "float16" not in want and arr.dtype != np.float32:
                    arr = arr.astype(np.float32)
                feed[inp.name] = arr
            else:
                print(f"  {case}: missing input {inp.name}")
                ok = False
        if not ok:
            continue
        try:
            for _ in range(WARMUP):
                sess.run(None, feed)
            ts = []
            for _ in range(REPEAT):
                t0 = time.perf_counter()
                sess.run(None, feed)
                ts.append((time.perf_counter() - t0) * 1000)
            ms = float(np.median(ts))
            print(f"  {case}: {model} {ms:.1f}ms (min {min(ts):.1f})")
            results.append({"case": case, "model": model, "ms_gpu": ms,
                            "ms_gpu_min": float(min(ts))})
        except Exception as e:
            print(f"  {case}: run fail {e}")
            results.append({"case": case, "model": model, "error": str(e)})
        del sess
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=1)
    print("saved:", OUT)

if __name__ == "__main__":
    main()
