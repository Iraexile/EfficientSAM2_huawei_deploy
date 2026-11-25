from pathlib import Path
import onnx
from onnx import numpy_helper
root=Path(__file__).resolve().parents[1]
m=onnx.load(root/'experiments/artifacts/models/memattn_k01.pnnx.onnx')
constants={x.name:numpy_helper.to_array(x) for x in m.graph.initializer}
producers={o:n for n in m.graph.node for o in n.output}
for n in m.graph.node:
    if n.op_type=='Constant' and n.attribute and n.attribute[0].type==onnx.AttributeProto.TENSOR:
        constants[n.output[0]]=numpy_helper.to_array(n.attribute[0].t)
for n in m.graph.node:
    if n.op_type=='ScatterND':
        print(n.name,list(n.input),list(n.output))
        for x in n.input:
            if x in constants:
                a=constants[x];print('constant',x,a.shape,a.flatten()[:16])
            elif x in producers: print('producer',producers[x])
        break
