"""Fixed-shape, auditable k01 GPU input/reference bundle. Never changes source models."""
from pathlib import Path
import sys,json,re,collections
import numpy as np
import onnx
import onnxruntime as ort
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'dependencies/python'))
import ncnn
out=root/'experiments/artifacts/k01-strict';out.mkdir(exist_ok=True)
src=root/'experiments/artifacts/models/memattn_k01_fixed.ncnn.param'
weights=root/'experiments/artifacts/models/memattn_k01_fixed.ncnn.bin'
lines=src.read_text().splitlines();parsed=[]
for line in lines[2:]:
    t=line.split();ni,no=map(int,t[2:4]);parsed.append(dict(type=t[0],name=t[1],bottom=t[4:4+ni],top=t[4+ni:4+ni+no],line=line))
case=root/'pc/device_vectors/memattn_004'
feed={name:np.fromfile(case/f'in_{name}.bin',np.float32).reshape(shape) for name,shape in [('curr',(4096,1,256)),('curr_pos',(4096,1,256)),('memory',(4100,1,64)),('memory_pos',(4100,1,64))]}
def setup(path):
    net=ncnn.Net();net.opt.num_threads=4;net.opt.use_vulkan_compute=False;net.opt.use_packing_layout=False
    net.opt.use_fp16_packed=net.opt.use_fp16_storage=net.opt.use_fp16_arithmetic=False
    assert net.load_param(str(path))==0;assert net.load_model(str(weights))==0
    ex=net.create_extractor();ex.set_light_mode(False)
    for i,a in enumerate(feed.values()):assert ex.input(f'in{i}',ncnn.Mat(a,batch_index=233).clone())==0
    return net,ex
net,ex=setup(src);rewrites=[];newlines=lines[:2]
for p in parsed:
    line=p['line']
    if p['type']=='ExpandDims':
        rc,m=ex.extract(p['top'][0]);assert rc==0
        opts=f'0={m.w}'
        if m.dims>=2:opts+=f' 1={m.h}'
        if m.dims>=3:opts+=f' 2={m.c}'
        if m.dims==4:opts+=f' 11={m.d}'
        line=' '.join(['Reshape',p['name'],'1','1',p['bottom'][0],p['top'][0],opts])
        rewrites.append(dict(name=p['name'],before=p['line'],after=line,dims=m.dims,w=m.w,h=m.h,d=m.d,c=m.c))
    newlines.append(line)
rc,before=ex.extract('out0');assert rc==0;before=before.numpy(batch_index=233).copy()
del ex,net
fixed=out/'k01.param';fixed.write_text('\n'.join(newlines)+'\n')
net,ex=setup(fixed);rc,after=ex.extract('out0');assert rc==0;after=after.numpy(batch_index=233).copy()
assert np.array_equal(before,after),np.max(np.abs(before-after))
print('ExpandDims_to_Reshape=',len(rewrites),'final_bitwise_equal=True')
original=onnx.load(root/'pc/outputs/onnx/memattn_k01.onnx')
ln=[n for n in original.graph.node if n.op_type=='LayerNormalization']
sdpa=[n for n in original.graph.node if n.op_type=='MatMul' and re.search(r'/(self_attn|cross_attn_image)/MatMul_1$',n.name)]
assert len(ln)==13 and len(sdpa)==8
key=[p for p in parsed if p['type'] in ('LayerNorm','SDPA')];counts=collections.Counter();pairs=[]
for p in key:
    kind=p['type'];j=counts[kind];counts[kind]+=1
    node=(ln if kind=='LayerNorm' else sdpa)[j];pairs.append((p,node.output[0],node.name))
shape_model=onnx.shape_inference.infer_shapes(original)
info={x.name:x for x in list(shape_model.graph.value_info)+list(shape_model.graph.output)}
for _,name,_ in pairs:original.graph.output.append(info[name])
op=ort.SessionOptions();op.intra_op_num_threads=4
session=ort.InferenceSession(original.SerializeToString(),op,providers=['CPUExecutionProvider'])
values=session.run(None,feed);refs={x.name:a for x,a in zip(session.get_outputs(),values)}
manifest=[];audit=[]
for i,(p,oname,node_name) in enumerate(pairs):
    rc,m=ex.extract(p['top'][0]);assert rc==0
    a=m.numpy(batch_index=233).copy().reshape(-1);b=refs[oname].reshape(-1)
    assert a.size==b.size
    d=np.abs(a-b);bad=int(np.count_nonzero(d>1e-4+1e-4*np.abs(b)))
    # Keep both oracles; a PC conversion discrepancy must not be hidden.
    a.tofile(out/f'pc_{i}.bin');b.tofile(out/f'onnx_{i}.bin')
    manifest.append(f'{p["top"][0]} {a.size} pc_{i}.bin onnx_{i}.bin')
    audit.append(dict(layer=p['name'],type=p['type'],blob=p['top'][0],onnx_node=node_name,maxabs=float(d.max()),violations=bad,elements=a.size))
    print(p['name'],node_name,'PC_ncnn_vs_ONNX',float(d.max()),'violations',bad)
gold=np.fromfile(case/'out_pix_feat_with_mem.bin',np.float32)
after.reshape(-1).tofile(out/'pc_final.bin');refs[session.get_outputs()[0].name].reshape(-1).tofile(out/'onnx_final.bin');gold.tofile(out/'golden_final.bin')
manifest.append(f'out0 {after.size} pc_final.bin onnx_final.bin')
first=next(p for p in parsed if p['type']=='SDPA')
aux=list(first['bottom'])
for p in parsed:
    if p['name'] in ('gemm_4','reshape_110','permute_63','reshape_111','unbind_4','add_29','sub_32','cat_8'):
        aux.extend(p['top'])
for blob in dict.fromkeys(aux):
    rc,m=ex.extract(blob);assert rc==0
    a=m.numpy(batch_index=233).copy().reshape(-1);name=f'aux_{blob}.bin';a.tofile(out/name)
    manifest.append(f'{blob} {a.size} {name} {name}')
(out/'refs.txt').write_text('\n'.join(manifest)+'\n')
shapes=[]
for p in parsed:
    if p['type'] in ('Reshape','ExpandDims'):
        rc,m=ex.extract(p['top'][0]);assert rc==0
        shapes.append(f'{p["name"]} {m.dims} {m.w} {m.h} {m.d} {m.c}')
(out/'shapes.txt').write_text('\n'.join(shapes)+'\n')
for name,a in feed.items():a.tofile(out/f'in_{name}.bin')
import shutil
shutil.copyfile(weights,out/'k01.bin')
log=root/'experiments/logs'
(log/'k01_rewrites.json').write_text(json.dumps(rewrites,indent=2))
(log/'k01_pc_checkpoints.json').write_text(json.dumps(audit,indent=2))
(log/'k01_structure.json').write_text(json.dumps(dict(inputs={k:dict(shape=list(v.shape),dtype=str(v.dtype)) for k,v in feed.items()},output=[1,256,64,64],ncnn_counts=dict(collections.Counter(p['type'] for p in parsed)),onnx_counts=dict(collections.Counter(n.op_type for n in original.graph.node))),indent=2))
print('bundle_ready',out,'PC_checkpoint_violations=',sum(x['violations'] for x in audit))
