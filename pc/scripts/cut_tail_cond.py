# samheads_cond surgery for MindSpore Lite:
# cut the no_obj_ptr combine tail (Greater/Cast/Mul/Sub/Mul/Add) out of the
# graph. obj_ptr output becomes the RAW obj_ptr_proj output; the combine
#   obj_ptr = lam*raw + (1-lam)*no_obj_ptr,  lam = (object_score_logits > 0)
# is done in fp32 by the C++ glue on device (no_obj_ptr saved to npz).
# Numerics: identical fp32 elementwise math, just moved out of the graph.
import numpy as np
import onnx
from onnx import helper, numpy_helper

SRC = "outputs/onnx/samheads_cond.onnx"
DST = "outputs/onnx/samheads_cond_mslite.onnx"
NOBJ_OUT = "outputs/ms/no_obj_ptr.npy"

m = onnx.load(SRC)
graph = m.graph

TAIL_OPS = {"/Cast_2", "/Mul", "/Sub", "/Mul_1", "/Add"}  # keep /Greater: it also feeds mask occlusion Where
# constants only used by the tail: /Constant_1 (0), /Constant_8 (1.0)
prod_count = {}
for n in graph.node:
    for inp in n.input:
        prod_count[inp] = prod_count.get(inp, 0) + 1

# 1) extract no_obj_ptr initializer for the C++ glue
for init in graph.initializer:
    if init.name == "m.no_obj_ptr":
        np.save(NOBJ_OUT, numpy_helper.to_array(init))
        print("saved no_obj_ptr ->", NOBJ_OUT)

# 2) find the raw obj_ptr source (input of /Mul that is not the cast output)
raw = None
for n in graph.node:
    if n.name == "/Mul":
        raw = [x for x in n.input if x != "/Cast_2_output_0"][0]
assert raw, "raw obj_ptr source not found"
print("raw obj_ptr source:", raw)

# 3) drop tail nodes only (constants may be shared elsewhere; keep them)
keep = [n for n in graph.node if n.name not in TAIL_OPS]

# 4) rename the raw tensor to the graph output name (only the removed /Mul
#    consumed it, verified below), no Identity bridge needed
consumers = [n.name for n in keep if raw in n.input]
assert not consumers, f"{raw} still consumed by {consumers}"
for n in keep:
    if raw in n.output:
        n.output[list(n.output).index(raw)] = "obj_ptr"
        print(f"renamed output of {n.name} -> obj_ptr")
        break

# 5) topological re-sort (MSLite parser requires topo-ordered nodes)
init_names = {i.name for i in graph.initializer}
in_names = {i.name for i in graph.input}
produced = set(init_names) | set(in_names)
ordered = []
remaining = list(keep)
while remaining:
    progressed = False
    for n in list(remaining):
        if all((x in produced or x == "") for x in n.input):
            ordered.append(n)
            produced.update(n.output)
            remaining.remove(n)
            progressed = True
    if not progressed:
        raise RuntimeError(f"topo sort stuck: {[n.name for n in remaining][:5]}")

del graph.node[:]
graph.node.extend(ordered)
onnx.save(m, DST)
print("saved", DST)
