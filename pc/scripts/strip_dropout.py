# Post-process ONNX for MindSpore Lite:
#  - strip eval-mode Dropout (identity) nodes
#  - replace degenerate Flatten (axis >= input rank) with constant-target Reshape
#    (NNACL Flatten infershape fails when axis == rank)
# Graph outputs that pointed at a dropped tensor are rewired through an explicit
# Identity node so output names stay stable.
import sys

import numpy as np
import onnx
from onnx import helper, numpy_helper


def fix_for_mslite(src, dst):
    m = onnx.load(src)
    m = onnx.shape_inference.infer_shapes(m)
    g = m.graph
    vi = {v.name: v for v in list(g.value_info) + list(g.input) + list(g.output)}

    # --- replace degenerate Flatten with Reshape --------------------------
    n_flat = 0
    new_nodes = []
    for n in g.node:
        if n.op_type == "Flatten":
            axis = next((a.i for a in n.attribute if a.name == "axis"), 1)
            v = vi.get(n.input[0])
            dims = ([d.dim_value for d in v.type.tensor_type.shape.dim]
                    if v is not None and v.type.tensor_type.shape.dim else None)
            if dims and all(d > 0 for d in dims) and axis >= len(dims):
                d0 = int(np.prod(dims[:axis]))
                d1 = int(np.prod(dims[axis:]))
                tgt = numpy_helper.from_array(np.array([d0, d1], dtype=np.int64),
                                              name=f"_flatten_fix_{n_flat}")
                g.initializer.append(tgt)
                new_nodes.append(helper.make_node(
                    "Reshape", [n.input[0], tgt.name], list(n.output),
                    name=n.name + "_asreshape"))
                n_flat += 1
                continue
        new_nodes.append(n)

    # --- strip Dropout (identity in eval) -----------------------------------
    remap = {}
    keep2 = []
    for n in new_nodes:
        if n.op_type == "Dropout":
            remap[n.output[0]] = n.input[0]
            continue
        keep2.append(n)

    def fix(nm):
        while nm in remap:
            nm = remap[nm]
        return nm

    for n in keep2:
        for i, inp in enumerate(n.input):
            n.input[i] = fix(inp)
    for o in g.output:
        if o.name in remap:
            keep2.append(helper.make_node("Identity", [fix(o.name)], [o.name],
                                          name="/dropout_bridge" + o.name))

    # --- eliminate Identity nodes (MSLite drops them, breaking outputs) -----
    # rule: if Y is a graph output and X is also a graph output -> the output
    # is a redundant alias; drop Y and the Identity. Otherwise rewire.
    n_id = 0
    out_names = {o.name for o in g.output}
    use_count = {}
    for n in keep2:
        for x in n.input:
            use_count[x] = use_count.get(x, 0) + 1
    alias_dropped = []
    keep3 = []
    for n in keep2:
        if n.op_type != "Identity":
            keep3.append(n)
            continue
        x, y = n.input[0], n.output[0]
        if y in out_names and x in out_names:
            alias_dropped.append(y)  # Y is an exact alias of output X
            n_id += 1
            continue
        if y in out_names and use_count.get(x, 0) == 1:
            # rename producer output X -> Y
            for p in keep3 + keep2:
                if x in p.output:
                    p.output[list(p.output).index(x)] = y
                    break
            n_id += 1
            continue
        # internal identity: rewire consumers
        for c in keep2:
            for i, inp in enumerate(c.input):
                if inp == y:
                    c.input[i] = x
        n_id += 1
    if alias_dropped:
        new_outs = [o for o in g.output if o.name not in alias_dropped]
        del g.output[:]
        g.output.extend(new_outs)
        print("  dropped redundant alias outputs:", alias_dropped)

    del g.node[:]
    g.node.extend(keep3)
    onnx.save(m, dst)
    print(f"{src} -> {dst}: {n_flat} Flatten->Reshape, stripped {len(remap)} Dropout, {n_id} Identity removed")


if __name__ == "__main__":
    fix_for_mslite(sys.argv[1], sys.argv[2])

