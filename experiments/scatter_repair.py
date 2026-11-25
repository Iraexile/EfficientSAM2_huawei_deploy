"""Inspect statically folded ScatterND indices before attempting a semantics-preserving fix."""
from pathlib import Path
import onnx
from onnx import numpy_helper
import onnxruntime as ort
import numpy as np
root=Path(__file__).resolve().parents[1]
src=root/'pc/outputs/onnx/memattn_k01.onnx'
dst=root/'experiments/artifacts/models/memattn_k01.ort.onnx'
opt=ort.SessionOptions();opt.optimized_model_filepath=str(dst)
opt.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
session=ort.InferenceSession(str(src),opt,providers=['CPUExecutionProvider'])
m=onnx.load(dst);constants={x.name:numpy_helper.to_array(x) for x in m.graph.initializer}
m=onnx.shape_inference.infer_shapes(m)
shapes={x.name:[d.dim_value for d in x.type.tensor_type.shape.dim] for x in list(m.graph.value_info)+list(m.graph.input)+list(m.graph.output)}
nodes=[];count=0
for n in m.graph.node:
    if n.op_type=='ScatterND':
        a=constants.get(n.input[1])
        data,indices,updates=n.input
        ds,us=shapes[data],shapes[updates]
        expected=np.zeros((1,1,4096,3),np.int64);expected[0,0,:,2]=np.arange(4096)
        assert a is not None and np.array_equal(a,expected), n.name
        assert ds[:2]==[1,1] and ds[3]==256 and us==[1,1,4096,256] and ds[2] in (4096,4100)
        assert not n.attribute or all(x.name=='reduction' and x.s==b'none' for x in n.attribute)
        if ds[2]==4096:
            nodes.append(onnx.helper.make_node('Identity',[updates],list(n.output),name=n.name+'_full_replace'))
        else:
            names=[]
            for suffix,array in [('starts',[4096]),('ends',[4100]),('axes',[2]),('steps',[1])]:
                key=n.name+'_'+suffix;names.append(key)
                m.graph.initializer.append(numpy_helper.from_array(np.array(array,np.int64),key))
            tail=n.name+'_tail'
            nodes.append(onnx.helper.make_node('Slice',[data]+names,[tail],name=n.name+'_preserve_ptr_tail'))
            nodes.append(onnx.helper.make_node('Concat',[updates,tail],list(n.output),axis=2,name=n.name+'_prefix_replace'))
        count+=1;print('repaired',n.name,'data_shape=',ds,'update_shape=',us)
    else: nodes.append(n)
assert count==8
m.graph.ClearField('node');m.graph.node.extend(nodes)
fixed=root/'experiments/artifacts/models/memattn_k01.repaired.onnx'
onnx.checker.check_model(m);onnx.save(m,fixed)
opts=ort.SessionOptions();opts.intra_op_num_threads=4
original=ort.InferenceSession(str(src),opts,providers=['CPUExecutionProvider'])
repaired=ort.InferenceSession(str(fixed),opts,providers=['CPUExecutionProvider'])
case=root/'pc/device_vectors/memattn_004'
feed={i.name:np.fromfile(case/f'in_{i.name}.bin',np.float32).reshape(i.shape) for i in original.get_inputs()}
for label,inputs in [('golden_input',feed),('scaled_input',{k:v*np.float32(.9) for k,v in feed.items()})]:
    a=original.run(None,inputs)[0];b=repaired.run(None,inputs)[0]
    error=float(np.max(np.abs(a-b)))
    assert np.isfinite(b).all() and np.array_equal(a,b), (label,error)
    print(f'{label}: original_vs_repaired maxabs={error} bitwise_equal=True')
print('scope=memattn_k01_fixed_shape_8_ScatterND_only')
