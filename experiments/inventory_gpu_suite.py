from pathlib import Path
import json,collections,hashlib
import onnx
root=Path(__file__).resolve().parents[1];base=root/'experiments/artifacts/gpu-suite'
names=['image_encoder']+[f'memattn_k{i:02d}' for i in range(1,16)]+['memattn_steady','memenc_binarize','memenc_sigmoid','samheads_cond','samheads_track']
rows=[]
for name in names:
    path=root/f'pc/outputs/onnx/{name}.onnx';m=onnx.load(path,load_external_data=False)
    def desc(x):return dict(name=x.name,shape=[d.dim_param or d.dim_value for d in x.type.tensor_type.shape.dim],dtype=onnx.TensorProto.DataType.Name(x.type.tensor_type.elem_type))
    cases=[x.split()[1] for x in (root/'pc/device_vectors/manifest.txt').read_text().splitlines() if x.startswith('case ') and x.split()[2]==name]
    row=dict(name=name,case=cases[0],source=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),inputs=[desc(x) for x in m.graph.input],outputs=[desc(x) for x in m.graph.output],onnx_counts=dict(collections.Counter(n.op_type for n in m.graph.node)))
    status=json.loads((base/name/'status.json').read_text());row['preparation']=status
    rows.append(row)
(root/'experiments/logs/suite_inventory.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
(base/'preparation-latest.json').write_text(json.dumps([x['preparation'] for x in rows],indent=2),encoding='utf-8')
print('inventoried',len(rows),'official graphs')
