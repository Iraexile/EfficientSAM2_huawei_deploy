# Minimal ONNX surgery for MSLite: pin object_score_logits to static [1,1]
# via a constant-target Reshape right before the /Greater that feeds the
# no_obj_ptr combine. Numerics unchanged (runtime shape is already [1,1]).
import onnx
from onnx import helper, numpy_helper
import numpy as np

SRC = "outputs/onnx/samheads_cond.onnx"
DST = "outputs/onnx/samheads_cond_mslite.onnx"

m = onnx.load(SRC)
g = m.graph

shape_init = numpy_helper.from_array(np.array([1, 1], dtype=np.int64), name="_pin_objscore_shape")
g.initializer.append(shape_init)

rs = helper.make_node("Reshape", ["object_score_logits", "_pin_objscore_shape"],
                      ["_objscore_pinned"], name="/pin_objscore")

# insert right before the Greater consuming object_score_logits
for i, n in enumerate(g.node):
    if "object_score_logits" in list(n.input) and n.op_type == "Greater":
        n.input[list(n.input).index("object_score_logits")] = "_objscore_pinned"
        g.node.insert(i, rs)
        break
else:
    raise RuntimeError("Greater consuming object_score_logits not found")

onnx.save(m, DST)
print("saved", DST)
