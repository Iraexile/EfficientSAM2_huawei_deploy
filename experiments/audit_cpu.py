"""Audit saved device report without changing its acceptance rule or rerunning it."""
import csv
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / 'pc/device_results/report.txt'
rows = []
for line in source.read_text(encoding='utf-8').splitlines():
    m = re.match(r'(\S+) (\S+) (OK|FAIL|ERR) .*?maxdiff=([\deE.+-]+)\(([^)]+)\)', line)
    if m:
        case, graph, status, error, output = m.groups()
        rows.append(dict(case=case, graph=graph, status=status,
                         max_abs_error=float(error), worst_output=output, raw=line))
assert len(rows) == 71, f'Expected 71 cases, found {len(rows)}'
out = ROOT / 'experiments/logs'
out.mkdir(parents=True, exist_ok=True)
(out / 'cpu_audit.json').write_text(json.dumps(dict(
    source=str(source.relative_to(ROOT)), sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    acceptance='abs(actual-golden) <= 1e-4 + 1e-4*abs(golden)', cases=rows), indent=2), encoding='utf-8')
with (out / 'cpu_audit.csv').open('w', newline='', encoding='utf-8-sig') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
failed = [r for r in rows if r['status'] == 'FAIL']
text = ['# CPU 71 例误差审计', '',
        '来源：pc/device_results/report.txt（历史真机运行，本次未重跑）。', '',
        f'共 {len(rows)} 例；日志 OK {sum(r["status"] == "OK" for r in rows)}；FAIL {len(failed)}。', '',
        '源码阈值为 atol=rtol=1e-4，即每元素 |actual-golden| ≤ 1e-4+1e-4×|golden|。',
        'FAIL 表示至少一个元素违反该公式；最大误差的位置未必就是违反公式的位置。', '',
        '|案例|计算图|最大绝对误差|最大误差输出|', '|---|---|---:|---|']
for r in failed:
    text.append(f'|{r["case"]}|{r["graph"]}|{r["max_abs_error"]:.6g}|{r["worst_output"]}|')
text += ['', '## 实际影响与证据局限', '',
         '- 12 个 Memory Attention 案例、1 个条件分割头案例违反原有严格阈值，不能声称 71 例全部严格通过。',
         '- samheads_001 的 low_res_multimasks=2.861e-4、high_res_multimasks=2.561e-4 均超限；ious=5.960e-8、obj_ptr=1.680e-5、object_score_logits=1.907e-6。',
         '- FAIL 输出日志余弦均打印为 1.000000、scale 为 1.0000，只能说明没有观察到明显整体方向/尺度偏离，舍入值不能证明数值完全相同或分割完全一致。',
         '- report.txt 不含逐元素预测值、超限数量、超限位置对应 golden、NaN/Inf 检查、mask IoU/边界翻转。device_results 未找到预测 .bin，当前无法重算这些指标。',
         '- 18 帧反馈流水线有单独报告，误差口径不同。历史记录的“无发散”和排序一致不能代替独立分割质量验证。',
         '- 浮点累积差是待验证解释，现有日志不足以排除所有实现差异。需要真机保存预测张量后离线逐元素、mask IoU 和 near-zero logits 翻转率验证。',
         '- 当前比较器非 float32 输出只检查字节数；未显式拒绝 NaN/Inf。下一次复测需加强诊断，保持原阈值。', '']
(ROOT / 'experiments/CPU_AUDIT.md').write_text('\n'.join(text), encoding='utf-8')
print(f'cases={len(rows)} OK={len(rows)-len(failed)} FAIL={len(failed)} max_abs={max(r["max_abs_error"] for r in rows)}')
