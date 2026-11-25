from pathlib import Path
import re,json
import numpy as np
root=Path(__file__).resolve().parents[1]
raw=root/'experiments/logs/gpu-app-hilog-final.txt'
lines=raw.read_text(encoding='utf-8-sig').splitlines()
probe=[x.split('SAM2GPU: ',1)[1] for x in lines if 'SAM2GPU: ' in x]
probe=probe[max(i for i,x in enumerate(probe) if x.startswith('probe=')):]
assert 'APP_PROBE_EXIT=0' in probe
result={}
for name,prefix in [('matmul','GPU_MATMUL_ROW'),('attention','GPU_ATTENTION_ROW')]:
    rows={}
    for line in probe:
        if line.startswith(prefix+' '):
            parts=line.split();index=int(parts[1]);assert index not in rows
            rows[index]=[float(x) for x in parts[2:]]
    assert sorted(rows)==list(range(16))
    got=np.array([rows[i] for i in range(16)],np.float32);assert got.shape==(16,16)
    golden=np.load(root/f'experiments/artifacts/models/{name}.npz')['expected']
    diff=np.abs(got-golden)
    assert np.isfinite(got).all() and np.all(diff<=1e-4+1e-4*np.abs(golden))
    result[name]=dict(max_abs_error=float(diff.max()),violations=int(np.count_nonzero(diff>1e-4+1e-4*np.abs(golden))),elements=256,output=got.tolist(),reference=golden.tolist())
    print(name,result[name]['max_abs_error'],'violations=',result[name]['violations'])
result['device']=next(x for x in probe if x.startswith('device='))
result['execution']=next(x for x in probe if x.startswith('stage=completed'))
(root/'experiments/logs/gpu_numeric_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
(root/'experiments/logs/gpu_public_probe.txt').write_text('\n'.join(probe)+'\n',encoding='utf-8')
