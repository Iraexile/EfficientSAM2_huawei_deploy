from pathlib import Path
import json,csv,collections
root=Path(__file__).resolve().parents[1];logs=root/'experiments/logs'
a=json.loads((logs/'k01_device_audit.json').read_text());pc=json.loads((logs/'k01_pc_checkpoints.json').read_text())
lookup={(x['blob'],x['reference']):x for x in a['checkpoints']}
text=['# k01 数值与性能详表','','数据来自完整GPU二进制的独立PC复核；阈值始终为 atol=rtol=1e-4。','',
      f'PC独立核查：{len(a["checkpoints"])}项比较，严格超限比较数{a["strict_failed_comparisons"]}；另有最终PyTorch golden校验。',
      f'全层最大绝对误差：{max(x["maxabs"] for x in a["checkpoints"]):.9g}；最大容差归一化比值：{max(x["max_tolerance_ratio"] for x in a["checkpoints"]):.9g}（≤1为通过）。','',
      '| ONNX关键节点 | GPU层 | 最大绝对误差 | 超限元素 |','|---|---|---:|---:|']
for p in pc:
    row=lookup[p['blob'],'ONNX'];text.append(f'|{p["onnx_node"]}|{p["layer"]}|{row["maxabs"]:.9g}|{row["violations"]}/{row["total"]}|')
row=lookup['out0','ONNX'];text.append(f'|最终输出|out0|{row["maxabs"]:.9g}|{row["violations"]}/{row["total"]}|')
text+=['','## 全层诊断计时（第一轮完整审计）','','submit/wait包括排队、执行和同步，不是纯GPU timestamp。形状别名记0表示不提交数据运算，不表示主机管理时间绝对为0。','',
       '| 类型 | 层数 | CPU编码合计ms | 提交等待合计ms |','|---|---:|---:|---:|']
groups=collections.defaultdict(list)
for r in csv.DictReader((logs/'k01_operator_times.csv').open()):groups[r['type']].append(r)
for name,rs in sorted(groups.items(),key=lambda x:-sum(float(r['submit_wait_ms']) for r in x[1])):
    text.append(f'|{name}|{len(rs)}|{sum(float(r["encode_ms"]) for r in rs):.3f}|{sum(float(r["submit_wait_ms"]) for r in rs):.3f}|')
text+=['','## 重复运行','','| 运行 | 张量检查点 | 编码ms | 提交等待ms | 读回ms | 比较保存ms | 总诊断ms | 结果 |',
       '|---|---:|---:|---:|---:|---:|---:|---:|']
for t in a['trials']:
    text.append('|'+ '|'.join(str(t[k]) for k in ['trial','checkpoints','encode_ms','submit_wait_ms','readback_ms','compare_save_ms','diagnostic_wall_ms','result'])+'|')
text+=['','原始张量/运行日志在experiments/artifacts/k01-device；逐层逐张量完整记录见同名CSV和JSON。','']
(root/'experiments/K01_NUMERIC_TABLES.md').write_text('\n'.join(text),encoding='utf-8')
