"""Deduplicate exact PC oracle bytes for reproducible raw-resource staging."""
from pathlib import Path
import hashlib,shutil,json
root=Path(__file__).resolve().parents[1];base=root/'experiments/artifacts/k01-strict';raw=root/'entry/src/main/resources/rawfile'
seen={};lines=[];total=0
for line in (base/'refs-full.txt').read_text().splitlines():
    blob,count,pc,onnx=line.split();names=[]
    for name in (pc,onnx):
        path=base/name
        with path.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        dst='oracle_'+digest+'.bin'
        if digest not in seen:
            shutil.copyfile(path,raw/dst);seen[digest]=name;total+=path.stat().st_size
        names.append(dst)
    lines.append(f'{blob} {count} {names[0]} {names[1]}')
(raw/'refs-full.txt').write_text('\n'.join(lines)+'\n')
(root/'experiments/logs/k01_oracle_package.json').write_text(json.dumps(dict(unique_files=len(seen),bytes=total,hash_to_original=seen),indent=2))
print('deduplicated_oracle_files',len(seen),'bytes',total)
