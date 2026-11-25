# PC-side dry-run of the C++ memory-bank glue: rebuild each track frame's
# memattn inputs (memory / memory_pos / num_obj_ptr_tokens) purely from
# per-frame golden outputs + exported constants, and compare against the
# golden memattn inputs. If this matches bitwise-closely, the C++ glue
# spec is proven correct before writing any C++.
#
# Bank rules (sam2_base._prepare_memory_conditioned_features, baseline):
#   maskmem: cond frame (tpos idx 6) + prev 6 non-cond frames (tpos idx (k-p)-1)
#   ptrs:    cond ptr (pos=k) + up to 15 prev non-cond ptrs (pos=t_diff, newest first)
#   ptr token: [256] -> 4 tokens of 64; pos enc: tpos_proj(sine_pe(pos/15, dim=256)) x4
import glob
import os

import numpy as np

GOLD = os.path.join(os.path.dirname(__file__), "..", "golden")
MS = os.path.join(os.path.dirname(__file__), "..", "outputs", "ms")

# ---- constants ----
tpos_enc = np.load(os.path.join(MS, "const_maskmem_tpos_enc.npy"))  # [7,1,1,64]
proj_w = np.load(os.path.join(MS, "const_obj_ptr_tpos_proj_w.npy"))  # [64,256]
proj_b = np.load(os.path.join(MS, "const_obj_ptr_tpos_proj_b.npy"))  # [64]
maskmem_pos_enc = np.load(os.path.join(MS, "maskmem_pos_enc.npy"))  # [1,64,64,64]
no_mem_embed = np.load(os.path.join(MS, "const_no_mem_embed.npy"))  # [1,1,256]


def sine_pe_1d(pos, dim, temperature=10000.0):
    # sam2_utils.get_1d_sine_pe 精确复刻：dim_t=temp^(2*(i//2)/pe_dim)，cat([sin,cos])
    pe_dim = dim // 2
    i = np.arange(pe_dim, dtype=np.float64)
    dim_t = temperature ** (2 * (i // 2) / pe_dim)
    pe = np.asarray(pos, dtype=np.float64)[:, None] / dim_t[None, :]
    return np.concatenate([np.sin(pe), np.cos(pe)], axis=-1)


def check(cond):
    print(("  OK  " if cond else "  FAIL"), end=" ")
    return cond


# ---- load per-frame goldens ----
# frame k (0-based video frame). cond = frame 0.
# records: frame 0 -> encoder_000/samheads_001/memenc_002
#          frame k -> encoder_(4k-1)/memattn_(4k)/samheads_(4k+1)/memenc_(4k+2)
def case_idx(graph, frame):
    if frame == 0:
        return {"image_encoder": 0, "samheads": 1, "memenc": 2}[graph]
    base = 4 * frame - 1
    return {"image_encoder": base, "memattn": base + 1, "samheads": base + 2, "memenc": base + 3}[graph]


def tok64(arr):
    # [1,64,64,64] -> flatten(2).permute(2,0,1) == [4096,64]
    return arr.reshape(64, 4096).T


def bf16_round(arr):
    # 官方流程 bank 内 maskmem_features 以 bf16 存储
    import torch
    return torch.from_numpy(np.ascontiguousarray(arr)).to(torch.bfloat16).to(torch.float32).numpy()


def load(graph, frame):
    p = os.path.join(GOLD, f"{graph}_{case_idx(graph, frame):03d}.npz")
    return np.load(p)


N_FRAMES = 18
n_fail = 0

# verify cond-frame glue: samheads_001 in_backbone_features == fpn2 + no_mem_embed
enc0 = load("image_encoder", 0)
sh0 = load("samheads", 0)
add = enc0["out_fpn2"] + no_mem_embed.reshape(1, 256, 1, 1)
d = np.abs(add - sh0["in_backbone_features"]).max()
print(f"cond glue: fpn2+no_mem_embed vs samheads_001 backbone  maxdiff={d:.3e}", check(d < 1e-5))

# rebuild bank incrementally and verify each track frame's memattn inputs
bank = []  # list of dict(frame, maskmem_features [64x64x64 flat], obj_ptr [256], is_cond)
for frame in range(N_FRAMES):
    me = load("memenc", frame)
    sh = load("samheads", frame)
    if frame == 0:
        bank.append({"frame": 0, "is_cond": True,
                     "feat": bf16_round(tok64(me["out_maskmem_features"])),
                     "ptr": sh["out_obj_ptr"].reshape(-1)})
        continue

    k = frame  # 1-based track frame index == frame number here
    # ---- build memory / memory_pos from bank ----
    mem_parts, pos_parts = [], []
    # 1) cond frame first, tpos idx 6
    cond = bank[0]
    mem_parts.append(cond["feat"])
    pos_parts.append(tok64(maskmem_pos_enc) + tpos_enc[6].reshape(64))
    # 2) prev 6 non-cond frames, oldest first; tpos idx (k-p)-1
    prev_all = [e for e in bank if not e["is_cond"]]
    prev_frames = prev_all[-6:]  # maskmem 窗口：最近 6 帧，最旧在前
    for e in prev_frames:
        p = e["frame"]
        mem_parts.append(e["feat"])
        pos_parts.append(tok64(maskmem_pos_enc) + tpos_enc[(k - p) - 1].reshape(64))
    # 3) obj ptrs: cond first (pos=k), then newest->oldest non-cond (pos=t_diff)
    ptr_list = [(k, bank[0]["ptr"])]
    for e in reversed(prev_all[-15:]):
        ptr_list.append((k - e["frame"], e["ptr"]))
    ptrs = np.stack([p for _, p in ptr_list])  # [nptr,256]
    pos_list = [d for d, _ in ptr_list]
    nptr = len(ptr_list)
    obj_pos = sine_pe_1d(np.array(pos_list) / 15.0, 256)  # [nptr,256]
    obj_pos = (obj_pos @ proj_w.T + proj_b)  # [nptr,64]
    obj_pos = np.repeat(obj_pos, 4, axis=0)  # repeat_interleave 4
    ptr_tokens = ptrs.reshape(nptr, 4, 64).reshape(nptr * 4, 64)
    mem_parts.append(ptr_tokens)  # (nptr*4, 64)
    pos_parts.append(obj_pos)     # (nptr*4, 64)

    memory = np.concatenate(mem_parts, axis=0).reshape(-1, 1, 64).astype(np.float32)
    memory_pos = np.concatenate(pos_parts, axis=0).reshape(-1, 1, 64).astype(np.float32)
    num_ptr_tokens = nptr * 4

    # ---- compare against golden memattn inputs ----
    ma = load("memattn", frame)
    gm, gp = ma["in_memory"], ma["in_memory_pos"]
    gpt = int(ma["meta_num_obj_ptr_tokens"])
    dm = np.abs(memory - gm).max() if memory.shape == gm.shape else float("inf")
    dp = np.abs(memory_pos - gp).max() if memory_pos.shape == gp.shape else float("inf")
    ok = check(dm < 1e-4 and dp < 1e-4 and gpt == num_ptr_tokens)
    if not (dm < 1e-4 and dp < 1e-4 and gpt == num_ptr_tokens):
        n_fail += 1
    print(f"frame {k:2d}: N={memory.shape[0]} (gold {gm.shape[0]}) ptr_tok={num_ptr_tokens} (gold {gpt})"
          f"  d_mem={dm:.2e} d_pos={dp:.2e}")

    # append this frame to bank (non-cond)
    bank.append({"frame": frame, "is_cond": False,
                 "feat": bf16_round(tok64(me["out_maskmem_features"])),
                 "ptr": sh["out_obj_ptr"].reshape(-1)})

print("\nFAIL count:", n_fail)
