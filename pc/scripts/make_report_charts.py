# 从设备报告提取完整数据 + 生成报告图表
# SAC 拦截 pyexpat 的临时绕过：font_manager 无条件 import plistlib，但 Windows 上不会真用
import sys, types
for name in ("pyexpat", "xml.parsers.expat"):
    m = types.ModuleType(name)
    m.ParserCreate = lambda *a, **k: None
    m.ExpatError = Exception
    sys.modules[name] = m
import json, os, re
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = r"C:\path\to\EfficientSAM2"
OUT = os.path.join(ROOT, "pc", "report_assets")
os.makedirs(OUT, exist_ok=True)

# ---------- 解析 71 例报告 ----------
cases = []
for line in open(os.path.join(ROOT, "pc/device_results/report.txt"), encoding="utf-8"):
    m = re.match(r"(\S+)\s+(\S+)\s+(OK|FAIL|ERR).*?maxdiff=([\d.e+-]+).*?\]\s*([\d.]+)ms", line)
    if m:
        cases.append({"case": m.group(1), "model": m.group(2), "result": m.group(3),
                      "maxdiff": float(m.group(4)), "ms": float(m.group(5))})
print("cases:", len(cases))
json.dump(cases, open(os.path.join(OUT, "cases.json"), "w"), ensure_ascii=False, indent=1)

# ---------- 解析流水线逐帧报告 ----------
frames = []
for line in open(os.path.join(ROOT, "pc/device_results/pipeline_report.txt"), encoding="utf-8"):
    m = re.match(r"f(\d+) enc \S+ [\d.e-]+ ([\d.]+)ms \| (?:cond \S+ [\d.e-]+ ([\d.]+)ms \| memencB \S+ [\d.e-]+ ([\d.]+)ms|(\S+) \S+ [\d.e-]+ ([\d.]+)ms .*? \| track \S+ [\d.e-]+ ([\d.]+)ms .*?memencS \S+ [\d.e-]+ ([\d.]+)ms)", line)
    if m:
        if m.group(3):
            frames.append({"f": int(m.group(1)), "enc": float(m.group(2)), "memattn": 0.0, "sam": float(m.group(3)), "memenc": float(m.group(4))})
        else:
            frames.append({"f": int(m.group(1)), "enc": float(m.group(2)), "memattn": float(m.group(6)), "sam": float(m.group(7)), "memenc": float(m.group(8))})
print("frames:", len(frames), frames[0], frames[-1])
json.dump(frames, open(os.path.join(OUT, "frames.json"), "w"), indent=1)

# ---------- 图 1：memattn 耗时 vs 记忆 token 数 ----------
toks = [4100, 8200, 12300, 16400, 20500, 24600, 28700]
tms = [2130, 2971, 4153, 4684, 5496, 6269, 7273]
fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=150)
ax.plot([t/1000 for t in toks], [t/1000 for t in tms], "o-", color="#37474F", lw=2, ms=5)
ax.axhline(7.05, color="#B0BEC5", ls="--", lw=1.2)
ax.text(3.2, 7.25, "平台期 ≈7.05 s（记忆窗口已打满）", fontsize=9, color="#78909C")
ax.set_xlabel("参与注意力的记忆 token 数（千）")
ax.set_ylabel("耗时（秒）")
ax.set_title("记忆注意力耗时随记忆长度线性增长 → token 越少越快")
for x, y in [(4.1, 2.13), (28.7, 7.27)]:
    ax.annotate(f"{y:.1f}s", (x, y), textcoords="offset points", xytext=(0, 8), fontsize=9)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_memattn_scaling.png"))
plt.close(fig)

# ---------- 图 2：track 帧耗时构成 ----------
fig, ax = plt.subplots(figsize=(7.6, 3.4), dpi=150)
parts = [("记忆注意力 memattn", 7.05, "#37474F"), ("图像编码器 encoder", 3.74, "#78909C"),
         ("掩码解码 samheads", 0.13, "#B0BEC5"), ("记忆编码 memenc", 0.21, "#CFD8DC")]
total = sum(p[1] for p in parts)
x = 0
for name, v, c in parts:
    ax.barh([""], [v], left=[x], color=c, height=0.55)
    if v > 1.2:
        ax.text(x + v/2, 0, f"{name}\n{v:.2f} 秒（{v/total*100:.0f}%）", ha="center", va="center",
                fontsize=10, color="white")
    x += v
ax.text(total - 0.1, -0.52, "掩码解码 0.13 秒 + 记忆编码 0.21 秒（合计约 3%）", ha="right",
        fontsize=9, color="#78909C")
ax.set_xlim(0, total)
ax.set_ylim(-0.7, 0.55)
ax.set_yticks([])
ax.set_xlabel("单帧耗时（秒）")
ax.set_title("一帧约 11.1 秒花在哪里：记忆注意力占六成以上")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_frame_breakdown.png"))
plt.close(fig)

# ---------- 图 3：18 帧逐帧耗时堆叠 ----------
fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=150)
fs_ = [f["f"] for f in frames]
enc = [f["enc"]/1000 for f in frames]
ma = [f["memattn"]/1000 for f in frames]
sam = [f["sam"]/1000 for f in frames]
me = [f["memenc"]/1000 for f in frames]
b0 = [0]*18
ax.bar(fs_, enc, 0.7, label="图像编码器", color="#78909C")
ax.bar(fs_, ma, 0.7, bottom=enc, label="记忆注意力", color="#37474F")
b1 = [e+m for e, m in zip(enc, ma)]
ax.bar(fs_, sam, 0.7, bottom=b1, label="掩码解码", color="#B0BEC5")
b2 = [a+s for a, s in zip(b1, sam)]
ax.bar(fs_, me, 0.7, bottom=b2, label="记忆编码", color="#CFD8DC")
ax.set_xlabel("帧序号（0 为标注帧）")
ax.set_xticks(range(0, 18, 2))
ax.set_ylabel("耗时（秒）")
ax.set_title("逐帧耗时：记忆随帧数累积，注意力越来越贵")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_frames.png"))
plt.close(fig)

# ---------- 图 4：PC(x86) vs Mate 80 同模型对比 ----------
fig, ax = plt.subplots(figsize=(7.0, 3.8), dpi=150)
names = ["图像编码器", "记忆注意力(稳态)", "记忆编码", "掩码解码"]
x86 = [1.058, 2.241, 0.044, 0.024]
m80 = [3.743, 7.05, 0.21, 0.134]
x = range(4)
ax.bar([i-0.18 for i in x], x86, 0.36, label="PC（x86 CPU）", color="#B0BEC5")
ax.bar([i+0.18 for i in x], m80, 0.36, label="Mate 80（麒麟 9020）", color="#37474F")
for i, (a, b) in enumerate(zip(x86, m80)):
    ax.text(i-0.18, a+0.08, f"{a:.2f}", ha="center", fontsize=8)
    ax.text(i+0.18, b+0.08, f"{b:.2f}", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(names, fontsize=9)
ax.set_ylabel("耗时（秒）")
ax.set_title("同一模型，手机 CPU 约为 PC 的 3~5 倍耗时")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_x86_vs_m80.png"))
plt.close(fig)

# ---------- 图 5：NPU 算子支持度探测 ----------
fig, ax = plt.subplots(figsize=(7.2, 3.2), dpi=150)
ops = [("MatMul（矩阵乘）", True), ("Conv（卷积）", True), ("Reshape（改形状）", True),
       ("Transpose（转置）", False), ("Softmax", False),
       ("LayerNorm（层归一化）", False), ("Split（切分）", False)]
ys = range(len(ops))[::-1]
for y, (name, ok) in zip(ys, ops):
    c = "#4C7A5A" if ok else "#B0574A"
    ax.barh(y, 1, color=c, height=0.6, alpha=0.9)
    ax.text(0.5, y, f"{name}：{'支持' if ok else '不支持（构建失败）'}",
            ha="center", va="center", fontsize=10, color="white")
ax.set_xlim(0, 1); ax.set_xticks([]); ax.set_yticks([])
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("麒麟 9020 NPU 单算子探测：Transformer 的关键零件大半缺席", pad=12)
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_npu_ops.png"))
plt.close(fig)

# ---------- 图 6：三平台耗时对比（PC GPU / PC CPU / 手机 CPU，对数轴） ----------
gpu_bench = json.load(open(os.path.join(ROOT, "pc", "outputs", "gpu_bench.json")))
gpu_med = {}
for r in gpu_bench:
    if "ms_gpu" in r:
        gpu_med.setdefault(r["model"], []).append(r["ms_gpu"])
gpu_med = {k: sorted(v)[len(v)//2] for k, v in gpu_med.items()}

names6 = ["图像编码器", "记忆注意力(稳态)", "记忆编码", "掩码解码"]
pc_gpu = [gpu_med["image_encoder"], gpu_med["memattn_steady"],
          gpu_med["memenc_sigmoid"], gpu_med["samheads_track"]]
pc_cpu = [1058, 2241, 44, 24]
m80    = [3743, 7273, 210, 134]
fig, ax = plt.subplots(figsize=(7.6, 4.0), dpi=150)
x = range(4)
b1 = ax.bar([i-0.26 for i in x], pc_gpu, 0.24, label="PC GPU（RTX 5070 Ti）", color="#4C7A5A")
b2 = ax.bar([i for i in x], pc_cpu, 0.24, label="PC CPU（x86）", color="#B0BEC5")
b3 = ax.bar([i+0.26 for i in x], m80, 0.24, label="Mate 80 CPU（麒麟 9020）", color="#37474F")
ax.set_yscale("log")
for bars in (b1, b2, b3):
    for r in bars:
        v = r.get_height()
        ax.text(r.get_x()+r.get_width()/2, v*1.12,
                f"{v:.0f}" if v >= 100 else f"{v:.1f}", ha="center", fontsize=8)
ax.set_xticks(list(x)); ax.set_xticklabels(names6, fontsize=9)
ax.set_ylabel("耗时（毫秒，对数轴）")
ax.set_title("同一模型三平台对比：PC GPU 比手机 CPU 快 26~145 倍")
ax.legend(fontsize=9)
ax.grid(axis="y", alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(os.path.join(OUT, "chart_gpu_compare.png"))
plt.close(fig)

print("charts done:", os.listdir(OUT))
