"""Deterministic minimal ONNX and ORT reference; inventory actual model shapes/types."""
from pathlib import Path
import json
import hashlib
from collections import Counter
import numpy as np
import onnx
from onnx import helper as h, TensorProto as T
import onnxruntime as ort

root = Path(__file__).resolve().parents[1]
out = root / 'experiments/artifacts/models'
out.mkdir(parents=True, exist_ok=True)
q = np.array([[((i*7+j*3)%19-9)/32 for j in range(16)] for i in range(16)], np.float32)
k = np.array([[((i*5+j*2)%17-8)/32 for i in range(16)] for j in range(16)], np.float32)
v = np.array([[((i*3+j*7)%23-11)/32 for j in range(16)] for i in range(16)], np.float32)
def value(n): return h.make_tensor_value_info(n, T.FLOAT, [16,16])
nodes=[h.make_node('MatMul',['q','kt'],['scores']), h.make_node('Softmax',['scores'],['probs'],axis=-1), h.make_node('MatMul',['probs','v'],['y'])]
for name, ns, ins in [('matmul',nodes[:1],['q','kt']),('attention',nodes,['q','kt','v'])]:
    final='scores' if name=='matmul' else 'y'
    m=h.make_model(h.make_graph(ns,name,[value(n) for n in ins],[value(final)]),opset_imports=[h.make_opsetid('',13)])
    m.ir_version=8
    onnx.checker.check_model(m)
    onnx.save(m,out/f'{name}.onnx')
    feed=dict(q=q,kt=k,v=v)
    got=ort.InferenceSession(str(out/f'{name}.onnx'),providers=['CPUExecutionProvider']).run(None,{n:feed[n] for n in ins})[0]
    scores=q.astype(np.float64)@k.astype(np.float64)
    probs=np.exp(scores-scores.max(axis=-1,keepdims=True)); probs/=probs.sum(axis=-1,keepdims=True)
    expected=scores if name=='matmul' else probs@v
    error=float(np.max(np.abs(got-expected)))
    assert np.isfinite(got).all() and error<1e-5
    np.savez(out/f'{name}.npz',**{n:feed[n] for n in ins},expected=expected.astype(np.float32))
    case=out/f'{name}_case';case.mkdir(exist_ok=True)
    manifest=[]
    for key in ins:
        feed[key].tofile(case/f'in_{key}.bin')
        manifest.append(f'tensor in_{key} float32 2 16 16 in_{key}.bin')
    expected.astype(np.float32).tofile(case/f'out_{final}.bin')
    manifest.append(f'tensor out_{final} float32 2 16 16 out_{final}.bin')
    (case/'case.txt').write_text('\n'.join(manifest)+'\n',encoding='utf-8')
    print(f'{name}: ORT_CPU_vs_numpy_fp64 maxabs={error:.9g}')
inventory=[]
for base in ['pc/outputs/onnx_opprobe','pc/outputs/onnx']:
    for path in sorted((root/base).glob('*.onnx')):
        m=onnx.load(path,load_external_data=False)
        def desc(x):
            t=x.type.tensor_type
            return dict(name=x.name,dtype=T.DataType.Name(t.elem_type),shape=[d.dim_param or d.dim_value for d in t.shape.dim])
        inventory.append(dict(path=str(path.relative_to(root)),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            opsets=[dict(domain=o.domain,version=o.version) for o in m.opset_import],
            ops=dict(Counter(n.op_type for n in m.graph.node)),inputs=[desc(x) for x in m.graph.input],outputs=[desc(x) for x in m.graph.output]))
(root/'experiments/logs/model_inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
print(f'model_inventory={len(inventory)}; ncnn_parity_results=experiments/logs/ncnn-pc-correctness.txt')
