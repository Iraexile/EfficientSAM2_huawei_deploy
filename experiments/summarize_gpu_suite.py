"""Summarize existing device evidence without treating preparation as execution."""
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
SUITE = ROOT / 'artifacts' / 'gpu-suite'
RESULTS = SUITE / 'device-results'
OUT = ROOT / 'logs'
COMPARE = re.compile(r'GRAPH_COMPARE .*?maxabs=([^ ]+) .*?violations=(\d+) total=(\d+) nonfinite=(\d+)')
SUMMARY = re.compile(r'(\w+)=([^ ]+)')
OPS = re.compile(r'GRAPH_OP .*?type=(\S+) encode_ms=([^ ]+) submit_wait_ms=([^ ]+).*?path=(\S+)')


def main():
    prepared = json.loads((SUITE / 'preparation.json').read_text(encoding='utf-8'))
    rows = []
    op_rows = []
    for item in prepared:
        name = item['name']
        logs = sorted(RESULTS.glob(f'*/{name}/*graph-probe.txt'), key=lambda p:p.stat().st_mtime)
        complete = [p for p in logs if 'GRAPH_SUMMARY' in p.read_text(errors='replace')]
        log = complete[-1] if complete else (logs[-1] if logs else None)
        blocked = ('BLOCKED_CONVERSION' if name in ('image_encoder','samheads_cond')
                   else 'BLOCKED_LOADING' if name == 'samheads_track' else 'PARTIAL')
        row = dict(graph=name, preparation=item['status'], case=item.get('case',''),
                   inputs=json.dumps(item.get('inputs',{}),separators=(',',':')),
                   outputs=json.dumps(item.get('outputs',{}),separators=(',',':')),
                   ncnn_ops=json.dumps(item.get('ncnn_counts',{}),separators=(',',':')),
                   pc_ncnn_onnx_maxabs=max((x['maxabs'] for x in item.get('pc_ncnn_vs_onnx',[])),default=''),
                   device_log=str(log.relative_to(ROOT.parent)) if log else '',
                   device_status=blocked, reason=item.get('reason',''), independent_onnx_checks='',
                   gpu='', cpu_forward_calls='', maxabs='', violations='', elements='', nonfinite='',
                   encode_ms='', submit_wait_ms='', readback_ms='', compare_save_ms='',
                   diagnostic_wall_ms='', peak_requested_bytes='', process_hwm_kb='')
        if log:
            s = log.read_text(errors='replace')
            sm = next((line for line in s.splitlines() if line.startswith('GRAPH_SUMMARY ')), '')
            fields = dict(SUMMARY.findall(sm))
            for key in ('encode_ms','submit_wait_ms','readback_ms','compare_save_ms',
                        'diagnostic_wall_ms','peak_requested_bytes','cpu_forward_calls'):
                row[key] = fields.get(key,'')
            row['gpu'] = 'Maleoon 920' if 'GRAPH_DEVICE name=Maleoon 920' in s else ''
            m = COMPARE.search(s)
            if m:
                row['maxabs'],row['violations'],row['elements'],row['nonfinite']=m.groups()
            hwm = re.search(r'GRAPH_PROCESS VmHWM:\s*(\d+) kB',s)
            if hwm: row['process_hwm_kb']=hwm.group(1)
            explicit = (row['gpu'] and row['cpu_forward_calls']=='0' and
                        all('vulkan=1' in x for x in s.splitlines() if x.startswith('GRAPH_MAP ')) and
                        'path=VULKAN_FORWARD' in s)
            row['device_status'] = ('GPU_EXECUTED' if int(row['violations'] or -1)==0 else
                                    'GPU_EXECUTED_WITH_ERROR') if explicit and m and fields.get('checkpoints') == str(len(item.get('outputs',{}))) else (
                                        'PARTIAL' if row['gpu'] and 'path=VULKAN_FORWARD' in s
                                        else blocked)
            groups=defaultdict(lambda:[0,0,0])
            for typ,enc,wait,path in OPS.findall(s):
                entry=groups[(typ,path)];entry[0]+=1;entry[1]+=float(enc);entry[2]+=float(wait)
            for (typ,path),(count,enc,wait) in groups.items():
                op_rows.append(dict(graph=name,type=typ,path=path,count=count,encode_ms=round(enc,6),submit_wait_ms=round(wait,6)))
            outputs=sorted(log.parent.glob('gpu_out*.bin'))
            if outputs:
                checks=[]
                for p in outputs:
                    idx=int(p.stem.removeprefix('gpu_out'))
                    ref=SUITE/name/f'ref_{idx}.bin'
                    if not ref.exists(): continue
                    a=np.fromfile(p,dtype='<f4');b=np.fromfile(ref,dtype='<f4')
                    if a.size!=b.size: checks.append(dict(output=idx,error='size_mismatch'));continue
                    diff=np.abs(a.astype('float64')-b.astype('float64'))
                    bad=~np.isfinite(a)|~np.isfinite(b)
                    violations=int(np.count_nonzero((diff>1e-4+1e-4*np.abs(b))|bad))
                    checks.append(dict(output=idx,elements=int(a.size),maxabs=float(np.max(diff)),violations=violations,nonfinite=int(np.count_nonzero(bad))))
                row['independent_onnx_checks']=json.dumps(checks,separators=(',',':'))
        rows.append(row)
    OUT.mkdir(exist_ok=True)
    with (OUT/'gpu_suite_summary.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with (OUT/'gpu_suite_op_profile.csv').open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=['graph','type','path','count','encode_ms','submit_wait_ms']);w.writeheader();w.writerows(op_rows)
    print(json.dumps(dict(graphs=len(rows),statuses=dict((s,sum(r['device_status']==s for r in rows)) for s in sorted(set(r['device_status'] for r in rows))),rows=[{k:r[k] for k in ('graph','device_status','maxabs','violations','submit_wait_ms','reason')} for r in rows]),indent=2))


if __name__=='__main__': main()
