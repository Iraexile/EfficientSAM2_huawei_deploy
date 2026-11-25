# 生成单算子微型 ONNX，用于在设备上逐一探测 NNRT 对 transformer 核心算子的支持度。
import os
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs", "onnx_opprobe")
os.makedirs(OUT, exist_ok=True)

def save(name, nodes, in_shape, out_shape, inits=()):
    inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, in_shape)
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, out_shape)
    g = helper.make_graph(list(nodes), name, [inp], [out], list(inits))
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, os.path.join(OUT, name + ".onnx"))
    print("saved", name)

# 注意力块里的形状：LN 在 channel 维，attention 是 4D matmul + softmax
save("op_ln", [helper.make_node("LayerNormalization", ["x", "s", "b"], ["y"], axis=-1)],
     [1, 256, 256, 96], [1, 256, 256, 96],
     [numpy_helper.from_array(np.ones(96, np.float32), "s"),
      numpy_helper.from_array(np.zeros(96, np.float32), "b")])

save("op_matmul4d", [helper.make_node("MatMul", ["x", "w"], ["y"])],
     [1, 64, 256, 64], [1, 64, 256, 64],
     [numpy_helper.from_array(np.random.randn(1, 64, 64, 64).astype(np.float32), "w")])

save("op_matmul2d", [helper.make_node("MatMul", ["x", "w"], ["y"])],
     [256, 96], [256, 96],
     [numpy_helper.from_array(np.random.randn(96, 96).astype(np.float32), "w")])

save("op_softmax", [helper.make_node("Softmax", ["x"], ["y"], axis=-1)],
     [1, 64, 256, 256], [1, 64, 256, 256])

save("op_transpose", [helper.make_node("Transpose", ["x"], ["y"], perm=[0, 3, 1, 2])],
     [1, 256, 256, 96], [1, 96, 256, 256])

save("op_split", [helper.make_node("Split", ["x"], ["y", "y2", "y3"], axis=-1)],
     [1, 256, 256, 288], [1, 256, 256, 96])
# Split 有 3 个输出，需要单独声明
inp = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 256, 256, 288])
outs = [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 256, 256, 96]) for n in ("y", "y2", "y3")]
g = helper.make_graph([helper.make_node("Split", ["x"], ["y", "y2", "y3"], axis=-1)], "op_split", [inp], outs)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 8
onnx.checker.check_model(m)
onnx.save(m, os.path.join(OUT, "op_split.onnx"))
print("saved op_split (fixed)")

save("op_reshape", [helper.make_node("Reshape", ["x", "sh"], ["y"])],
     [1, 256, 256, 96], [1, 256, 64, 256 * 96 // 64],
     [numpy_helper.from_array(np.array([1, 256, 64, -1], np.int64), "sh")])
