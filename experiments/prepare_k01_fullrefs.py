"""Add PC-ncnn oracle for every non-input blob; ONNX key oracles retained."""
import runpy
from pathlib import Path
import numpy as np
root=Path(__file__).resolve().parents[1]
ctx=runpy.run_path(str(root/'experiments/prepare_k01_strict.py'))
out=ctx['out'];setup=ctx['setup'];net,ex=setup(out/'k01.param')
old={line.split()[0]:line for line in (out/'refs.txt').read_text().splitlines()}
dest=out/'fullrefs';dest.mkdir(exist_ok=True)
lines=[];total=0;seen=set()
for p in ctx['parsed']:
    if p['type']=='Input':continue
    for blob in p['top']:
        if blob in seen:continue
        seen.add(blob)
        if blob in old:lines.append(old[blob]);continue
        rc,m=ex.extract(blob);assert rc==0,(p['name'],blob)
        a=m.numpy(batch_index=233).copy().reshape(-1)
        assert np.isfinite(a).all()
        path=f'fullrefs/pc_{blob}.bin';a.tofile(out/path);total+=a.nbytes
        lines.append(f'{blob} {a.size} {path} {path}')
(out/'refs-full.txt').write_text('\n'.join(lines)+'\n')
print('FULL_REFERENCE_BLOBS',len(lines),'additional_bytes',total)
