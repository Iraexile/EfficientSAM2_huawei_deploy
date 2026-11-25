# Generate push_list.txt for the device-side HTTP sync.
# Lines: <path relative to pc/> <path relative to app filesDir/>
# Default: smoke set (2 models + 2 cases). --full: all 21 models + all 43 cases.
import hashlib
import json
import os
import sys

PC = os.path.join(os.path.dirname(__file__), "..")
FULL = "--full" in sys.argv
MD5_CACHE = os.path.join(PC, ".push_md5.json")

lines = []

try:
    _cache = json.load(open(MD5_CACHE))
except Exception:
    _cache = {}


def md5_of(path, size, mtime_ns):
    """带缓存的 md5：同 (size, mtime) 直接命中。设备端增量同步第 4 列。"""
    ent = _cache.get(path)
    if ent and ent[0] == size and ent[1] == mtime_ns:
        return ent[2]
    h = hashlib.md5()
    with open(os.path.join(PC, path), "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    _cache[path] = [size, mtime_ns, digest]
    return digest


def add(remote, local):
    # 第 3 列 = 大小，第 4 列 = md5；设备端 size+md5 双一致才跳过
    st = os.stat(os.path.join(PC, remote))
    lines.append(f"{remote} {local} {st.st_size} {md5_of(remote, st.st_size, st.st_mtime_ns)}")

# NOTE: device 上 .ms 后缀的文件会被系统组件秒删，本地一律用 .dat 后缀。
OFFICIAL_MODELS = (
    ["image_encoder"]
    + [f"memattn_k{i:02d}" for i in range(1, 16)]
    + ["memattn_steady", "memenc_binarize", "memenc_sigmoid",
       "samheads_cond", "samheads_track"]
)

if FULL:
    for m in OFFICIAL_MODELS:
        add(f"outputs/ms/{m}.ms", f"models/{m}.dat")
    # 烘焙常量：memenc 的 maskmem_pos_enc（跨帧恒定，已从模型中裁剪）
    add("outputs/ms/maskmem_pos_enc.bin", "models/maskmem_pos_enc.bin")
    # pipeline 胶水常量（bank 组装 / cond 帧补齐）
    for c in ("const_maskmem_tpos_enc.bin", "const_no_mem_embed.bin",
              "const_no_mem_pos_enc.bin", "const_obj_ptr_tpos_proj_w.bin",
              "const_obj_ptr_tpos_proj_b.bin", "no_obj_ptr.bin"):
        add(f"outputs/ms/{c}", f"models/{c}")
    vdir = os.path.join(PC, "device_vectors")
    add("device_vectors/manifest.txt", "vectors/manifest.txt")
    for case in sorted(os.listdir(vdir)):
        cdir = os.path.join(vdir, case)
        if not os.path.isdir(cdir):
            continue
        for f in sorted(os.listdir(cdir)):
            add(f"device_vectors/{case}/{f}", f"vectors/{case}/{f}")
else:
    add("outputs/ms/image_encoder.ms", "models/image_encoder.dat")
    add("outputs/ms/memattn_k01.ms", "models/memattn_k01.dat")
    for case in ("image_encoder_000", "memattn_004"):
        cdir = os.path.join(PC, "device_vectors", case)
        for f in sorted(os.listdir(cdir)):
            add(f"device_vectors/{case}/{f}", f"vectors/{case}/{f}")

out = os.path.join(PC, "push_list.txt")
# 必须 LF：Windows 文本模式会写出 CRLF，\r 会混进设备端文件名
with open(out, "w", newline="\n") as f:
    f.write("\n".join(lines) + "\n")
with open(MD5_CACHE, "w") as f:
    json.dump(_cache, f)
total = 0
for ln in lines:
    r = ln.split(" ")[0]
    total += os.path.getsize(os.path.join(PC, r))
print(f"{len(lines)} files, {total / 1e6:.0f} MB -> {out}")
