"""Independent PC verification of raw GPU checkpoint bytes and per-layer audit."""
from pathlib import Path
import json,re,csv,collections
import numpy as np
root=Path(__file__).resolve().parents[1];data=root/'experiments/artifacts/k01-device';refs=root/'experiments/artifacts/k01-strict';logs=root/'experiments/logs'
comparisons=[]
manifest=refs/('refs-full.txt' if (data/'fulltrial1.txt').exists() else 'refs.txt')
for line in manifest.read_text().splitlines():
    blob,n,pc,onnx=line.split();got=np.fromfile(data/f'gpu_{blob}.bin',np.float32)
    assert got.size==int(n) and np.isfinite(got).all()
    pairs=[('PC_NCNN',pc)]+([('ONNX',onnx)] if pc!=onnx else [])
    for kind,file in pairs:
        ref=np.fromfile(refs/file,np.float32);diff=np.abs(got.astype(np.float64)-ref)
        bad=int(np.count_nonzero(diff>1e-4+1e-4*np.abs(ref.astype(np.float64))))
        index=int(np.argmax(diff/(1e-4+1e-4*np.abs(ref.astype(np.float64)))))
        comparisons.append(dict(blob=blob,reference=kind,maxabs=float(diff.max()),rmse=float(np.sqrt(np.mean(diff**2))),violations=bad,total=got.size,worst_normalized_index=index,worst_actual=float(got[index]),worst_reference=float(ref[index]),max_tolerance_ratio=float((diff/(1e-4+1e-4*np.abs(ref.astype(np.float64)))).max())))
gold=np.fromfile(refs/'golden_final.bin',np.float32);got=np.fromfile(data/'gpu_out0.bin',np.float32);diff=np.abs(got.astype(np.float64)-gold)
final=dict(maxabs=float(diff.max()),violations=int(np.count_nonzero(diff>1e-4+1e-4*np.abs(gold.astype(np.float64)))),total=got.size)
assert final['violations']==0
trials=[];ops=[];mapping=[]
for file in sorted(data.glob('*trial[123].txt')):
    text=file.read_text(encoding='utf-8');assert 'K01_EXIT=0' in text and 'APP_PROBE_EXIT=0' in text
    summary=next(x for x in text.splitlines() if x.startswith('K01_SUMMARY'))
    values=dict(re.findall(r'(\w+)=([\w.+-]+)',summary));values['trial']=file.name;trials.append(values)
    assert values['layers']=='475' and int(values['checkpoints']) in (34,612)
    if file.name==('fulltrial1.txt' if (data/'fulltrial1.txt').exists() else 'trial1.txt'):
        for line in text.splitlines():
            if line.startswith('K01_OP '):ops.append(dict(re.findall(r'(\w+)=([\w.+-]+)',line)))
            if line.startswith('K01_MAP '):mapping.append(dict(re.findall(r'(\w+)=([\w.+-]+)',line)))
assert len(mapping)==475 and len(ops)==471
assert all(x['path'] in ('VULKAN_FORWARD','GPU_BUFFER_METADATA_ALIAS') for x in ops)
audit=dict(checkpoints=comparisons,pytorch_final=final,trials=trials,
    mapping_counts=dict(collections.Counter(x['type'] for x in mapping)),
    paths=dict(collections.Counter(x['path'] for x in ops)),pure_kernel_timestamps_available=False,
    strict_failed_comparisons=sum(x['violations']>0 for x in comparisons),
    note='submit_wait_ms includes GPU queue, execution, synchronization; diagnostic per-layer submits and readbacks are not production latency')
(logs/'k01_device_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
for filename,rows in [('k01_operator_times.csv',ops),('k01_gpu_mapping.csv',mapping),('k01_device_checkpoints.csv',comparisons)]:
    with (logs/filename).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps(dict(pytorch_final=final,trials=trials,paths=audit['paths']),indent=2))
