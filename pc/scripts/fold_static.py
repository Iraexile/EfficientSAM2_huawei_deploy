# 迷你静态常量折叠：onnxsim 的 DLL 被本机 SAC 拦截，手写一个针对 shape-glue 的折叠器。
# 对所有"输入全部是常量"的节点用 numpy 求值，把结果固化成 initializer，反复迭代直到收敛。
import sys
import collections
import numpy as np
import onnx
from onnx import helper, numpy_helper, TensorProto

FOLDABLE = {"Shape", "Gather", "Unsqueeze", "Concat", "Cast", "Mod", "Sub", "Div", "Sqrt",
            "Mul", "Add", "Squeeze", "Slice", "Reshape", "Transpose", "ConstantOfShape",
            "Expand", "Equal", "Where", "Range", "Tile", "Neg", "Abs", "Floor", "Ceil",
            "Pow", "Clip", "Min", "Max", "ReduceMean", "ReduceSum", "MatMul", "Gemm",
            "Softmax", "LayerNormalization", "Conv", "Split"}

TO_NP = {TensorProto.INT64: np.int64, TensorProto.INT32: np.int32,
         TensorProto.FLOAT: np.float32, TensorProto.BOOL: np.bool_}

def np_dtype(t):
    return {TensorProto.INT64: np.int64, TensorProto.INT32: np.int32,
            TensorProto.FLOAT: np.float32, TensorProto.BOOL: np.bool_}.get(t, np.float32)

def eval_node(node, a):
    """a = 与 node.input（去空）对应的 numpy 数组列表。"""
    at = {attr.name: helper.get_attribute_value(attr) for attr in node.attribute}
    op = node.op_type
    if op == "Conv":
        # 只做 shape 推断：N,C 取权重，空间维按 pads/strides 公式
        x, w = a[0], a[1]
        pads = list(at.get("pads", [0] * (x.ndim - 2) * 2))
        strides = list(at.get("strides", [1] * (x.ndim - 2)))
        dil = list(at.get("dilations", [1] * (x.ndim - 2)))
        out = [x.shape[0], w.shape[0]]
        for i in range(x.ndim - 2):
            k = w.shape[2 + i]
            eff = dil[i] * (k - 1) + 1
            out.append((x.shape[2 + i] + pads[i] + pads[i + x.ndim - 2] - eff) // strides[i] + 1)
        return [np.zeros(out, dtype=x.dtype)]  # 仅 shape 有意义
    if op == "Split":
        axis = int(at.get("axis", 0))
        if len(a) > 1:
            sections = [int(s) for s in a[1]]
            return list(np.split(a[0], np.cumsum(sections)[:-1], axis=axis))
        num = int(at.get("num_outputs", 2))
        return list(np.array_split(a[0], num, axis=axis))
    if op == "Shape": return [np.array(a[0].shape, dtype=np.int64)]
    if op == "Gather": return [np.take(a[0], a[1], axis=int(at.get("axis", 0)))]
    if op == "Unsqueeze": return [np.expand_dims(a[0], tuple(int(x) for x in np.atleast_1d(a[1])))]
    if op == "Concat":
        ranks = {x.ndim for x in a}
        if len(ranks) > 1:
            if all(x.size == 1 for x in a):
                a = [x.reshape(1) for x in a]  # shape 标量链：统一摊平成 (1,)
            else:
                raise ValueError("concat rank mismatch")
        return [np.concatenate(a, axis=int(at.get("axis", 0)))]
    if op == "Cast": return [a[0].astype(np_dtype(int(at["to"])))]
    if op == "Mod": return [np.mod(a[0], a[1])]
    if op == "Sub": return [a[0] - a[1]]
    if op == "Div": return [a[0] / a[1]]
    if op == "Sqrt": return [np.sqrt(a[0])]
    if op == "Mul": return [a[0] * a[1]]
    if op == "Add": return [a[0] + a[1]]
    if op == "Neg": return [-a[0]]
    if op == "Abs": return [np.abs(a[0])]
    if op == "Floor": return [np.floor(a[0])]
    if op == "Squeeze": return [np.squeeze(a[0], axis=tuple(int(x) for x in np.atleast_1d(a[1])))]
    if op == "Slice":
        starts, ends = a[1], a[2]
        axes = a[3] if len(a) > 3 else np.arange(len(starts))
        steps = a[4] if len(a) > 4 else np.ones(len(starts), dtype=np.int64)
        sl = [slice(None)] * a[0].ndim
        for s, e, ax, st in zip(starts, ends, axes, steps):
            sl[int(ax)] = slice(int(s), int(e), int(st))
        return [a[0][tuple(sl)]]
    if op == "Reshape":
        tgt = [int(x) for x in a[1]]
        if not int(at.get("allowzero", 0)):
            tgt = [a[0].shape[k] if t == 0 and k < a[0].ndim else t for k, t in enumerate(tgt)]
        return [np.reshape(a[0], tuple(tgt))]
    if op == "Transpose": return [np.transpose(a[0], axes=at.get("perm"))]
    if op == "Range": return [np.arange(a[0], a[1], a[2], dtype=a[0].dtype)]
    if op == "Tile": return [np.tile(a[0], tuple(int(x) for x in a[1]))]
    if op == "ConstantOfShape":
        v = at.get("value")
        fill = numpy_helper.to_array(v)[0] if v else 0
        return [np.full(tuple(int(x) for x in a[0]), fill)]
    if op == "Expand": return [np.broadcast_to(a[0], tuple(int(x) for x in a[1])).copy()]
    if op == "Equal": return [a[0] == a[1]]
    if op == "Where": return [np.where(a[0], a[1], a[2])]
    if op == "Clip": return [np.clip(a[0], a[1], a[2])]
    if op == "Min": return [np.minimum.reduce(a)]
    if op == "Max": return [np.maximum.reduce(a)]
    if op == "Pow": return [np.power(a[0], a[1])]
    if op == "ReduceMean": return [a[0].mean(axis=tuple(int(x) for x in np.atleast_1d(a[1])), keepdims=bool(at.get("keepdims", 1)))]
    if op == "ReduceSum": return [a[0].sum(axis=tuple(int(x) for x in np.atleast_1d(a[1])), keepdims=bool(at.get("keepdims", 1)))]
    if op == "MatMul": return [a[0] @ a[1]]
    if op == "Gemm":
        y = a[0] @ (a[1].T if at.get("transB", 0) else a[1])
        if at.get("transA", 0): y = (a[0].T) @ (a[1].T if at.get("transB", 0) else a[1])
        alpha, beta = at.get("alpha", 1.0), at.get("beta", 1.0)
        return [alpha * y + (beta * a[2] if len(a) > 2 else 0)]
    if op == "Softmax":
        ax = int(at.get("axis", -1))
        x = a[0] - a[0].max(axis=ax, keepdims=True)
        e = np.exp(x)
        return [e / e.sum(axis=ax, keepdims=True)]
    if op == "LayerNormalization":
        ax = int(at.get("axis", -1)); eps = float(at.get("epsilon", 1e-5))
        x = a[0]; mu = x.mean(axis=ax, keepdims=True)
        var = ((x - mu) ** 2).mean(axis=ax, keepdims=True)
        y = (x - mu) / np.sqrt(var + eps) * a[1] + a[2]
        return [y]
    if op == "Conv":
        raise ValueError("conv not worth folding")
    raise ValueError(op)

def probe_shapes(src):
    """用 onnxruntime 跑一遍随机输入，记录所有中间张量的真实 shape（静态模型，shape 与数值无关）。"""
    import onnxruntime as ort
    m = onnx.shape_inference.infer_shapes(onnx.load(src))
    g = m.graph
    # 收集所有中间张量名
    produced = {o for n in g.node for o in n.output if o}
    existing_out = {o.name for o in g.output}
    for vi in list(g.value_info) + list(g.output):
        t = vi.type.tensor_type
        if vi.name in produced and vi.name not in existing_out and t.elem_type:
            g.output.append(helper.make_tensor_value_info(vi.name, t.elem_type, None))
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(m.SerializeToString(), sess_options=so,
                                providers=["CPUExecutionProvider"])
    feeds = {}
    for i in sess.get_inputs():
        dt = {1: np.float32, 7: np.int64, 6: np.int32, 9: np.bool_}[i.type and
              {"tensor(float)": 1, "tensor(int64)": 7, "tensor(int32)": 6, "tensor(bool)": 9}[i.type]]
        feeds[i.name] = np.random.randn(*[d if isinstance(d, int) and d > 0 else 1
                                          for d in i.shape]).astype(dt) \
            if dt == np.float32 else np.zeros([d if isinstance(d, int) and d > 0 else 1
                                               for d in i.shape], dtype=dt)
    outs = sess.run(None, feeds)
    return {o.name: list(v.shape) for o, v in zip(sess.get_outputs(), outs)}


def fold(src, dst):
    m = onnx.load(src)
    m = onnx.shape_inference.infer_shapes(m)
    g = m.graph
    vals = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    inputs = {i.name for i in g.input}
    # Constant 节点没有输入，先把它们的值收进来
    for n in g.node:
        if n.op_type == "Constant":
            for attr in n.attribute:
                if attr.name == "value":
                    vals[n.output[0]] = numpy_helper.to_array(attr.t)
    # 静态 shape 表：推理推导 + onnxruntime 实测
    shapes = {}
    def reg_shapes(vi_list):
        for vi in vi_list:
            t = vi.type.tensor_type
            if not t.HasField("shape"):
                continue
            dims = [d.dim_value for d in t.shape.dim]
            if all(d > 0 for d in dims):
                shapes[vi.name] = dims
    reg_shapes(g.input)
    reg_shapes(g.value_info)
    reg_shapes(g.output)
    try:
        for name, sh in probe_shapes(src).items():
            shapes.setdefault(name, sh)
    except Exception as e:
        print("probe_shapes failed:", e)
    # 元素类型表（用于哑数据 shape 传播）
    dtypes = {}
    def reg_types(vi_list):
        for vi in vi_list:
            t = vi.type.tensor_type
            if t.elem_type:
                dtypes[vi.name] = t.elem_type
    reg_types(g.input)
    reg_shapes(g.input)
    reg_shapes(g.value_info)
    reg_types(g.value_info)
    reg_shapes(g.output)
    print("static shapes:", len(shapes))

    def shape_of(name):
        if name in shapes:
            return shapes[name]
        if name in vals:
            return list(vals[name].shape)
        return None

    def propagate(node):
        """输入 shape 全已知时用哑数据推输出 shape。"""
        ins = [i for i in node.input if i]
        if not ins or not all(shape_of(i) is not None for i in ins):
            return
        if node.op_type not in FOLDABLE:
            return
        args = []
        for i in ins:
            if i in vals:
                args.append(vals[i])
            else:
                sh = shape_of(i)
                if int(np.prod(sh)) > (1 << 26):
                    return
                args.append(np.zeros(sh, dtype=np_dtype(dtypes.get(i, TensorProto.FLOAT))))
        try:
            with np.errstate(all="ignore"):
                outs = eval_node(node, args)
        except Exception as e:
            if os.environ.get("FOLD_DEBUG"):
                print("propfail", node.op_type, node.name.split("/")[-1], repr(e)[:80])
            return
        added = False
        for name, arr in zip(node.output, outs):
            if name and name not in shapes:
                shapes[name] = list(arr.shape)
                added = True
        return added
    changed = True
    import os
    dbg = os.environ.get("FOLD_DEBUG")
    rnd = 0
    while changed:
        changed = False
        nadd = 0
        keep = []
        for node in g.node:
            ins = [i for i in node.input if i]
            # 已固化的 Constant 节点直接移除（否则与 initializer 重名违反 SSA）
            if node.op_type == "Constant" and node.output[0] in vals:
                changed = True
                continue
            # Shape(静态张量) 直接固化为常量
            if node.op_type == "Shape" and ins and ins[0] in shapes:
                vals[node.output[0]] = np.array(shapes[ins[0]], dtype=np.int64)
                changed = True
                continue
            if (node.op_type in FOLDABLE and ins and all(i in vals for i in ins)
                    and node.op_type != "Conv"):
                try:
                    outs = eval_node(node, [vals[i] for i in ins])
                except Exception:
                    keep.append(node)
                    continue
                # 防止把巨大的激活也固化进来（>50MB 不折）
                if sum(o.nbytes for o in outs) > 50 << 20:
                    keep.append(node)
                    continue
                for name, arr in zip(node.output, outs):
                    if name:
                        vals[name] = np.ascontiguousarray(arr)
                        shapes[name] = list(arr.shape)
                changed = True
                continue
            keep.append(node)
            if propagate(node):  # 保留的节点也用哑数据把输出 shape 传播下去
                changed = True
                nadd += 1
        if dbg:
            print(f"round {rnd}: kept={len(keep)} new_shapes={nadd} vals={len(vals)}")
        rnd += 1
        del g.node[:]
        g.node.extend(keep)
    # 只保留仍被引用的 initializer
    used = {i for n in g.node for i in n.input if i}
    used |= {o for o in [o.name for o in g.output]}
    del g.initializer[:]
    for name, arr in vals.items():
        if name in used and name not in inputs:
            g.initializer.append(numpy_helper.from_array(arr, name))
    onnx.checker.check_model(m)
    del m.graph.value_info[:]  # 清掉折叠前的旧 value_info，避免与新形状冲突
    m = onnx.shape_inference.infer_shapes(m)
    onnx.save(m, dst)
    ops = collections.Counter(n.op_type for n in g.node)
    print(f"{dst}: nodes={len(g.node)} ops={dict(ops)}")

if __name__ == "__main__":
    fold(sys.argv[1], sys.argv[2])
