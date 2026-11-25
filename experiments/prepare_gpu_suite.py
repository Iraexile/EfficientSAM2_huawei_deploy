"""Independent conversion/oracle jobs for the 21 official graphs, never editing pc/."""
from pathlib import Path
import os,sys,json,subprocess,shutil,re,collections,traceback
root=Path(__file__).resolve().parents[1]
dest=root/'experiments/artifacts/gpu-suite';dest.mkdir(exist_ok=True)
names=['image_encoder']+[f'memattn_k{i:02d}' for i in range(1,16)]+['memattn_steady','memenc_binarize','memenc_sigmoid','samheads_cond','samheads_track']
if len(sys.argv)==1:
    results=[]
    for name in names:
        folder=dest/name;folder.mkdir(exist_ok=True)
        with (folder/'prepare.log').open('w',encoding='utf-8') as f:
            try:p=subprocess.run([sys.executable,__file__,name],env=os.environ,stdout=f,stderr=subprocess.STDOUT,timeout=240)
            except subprocess.TimeoutExpired:p=None
        if (folder/'status.json').exists():status=json.loads((folder/'status.json').read_text())
        else:status=dict(name=name,status='PREPARE_PROCESS_FAILED',exitcode=None if p is None else p.returncode)
        results.append(status);print(name,status['status'],flush=True)
        (dest/'preparation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    sys.exit(0)
name=sys.argv[1];assert name in names
folder=dest/name;folder.mkdir(exist_ok=True)
status=dict(name=name,status='PREPARING')
try:
    import numpy as np,onnx,onnxruntime as ort
    from onnx import numpy_helper,helper
    sys.path.insert(0,str(root/'dependencies/python'));import ncnn
    case=next(line.split()[1] for line in (root/'pc/device_vectors/manifest.txt').read_text().splitlines() if line.startswith('case ') and line.split()[2]==name)
    cdir=root/'pc/device_vectors'/case;spec={}
    for line in (cdir/'case.txt').read_text().splitlines():
        t=line.split()
        if t and t[0]=='tensor':rank=int(t[3]);spec[t[1]]=(t[2],tuple(map(int,t[4:4+rank])),t[4+rank])
    original=root/'pc/outputs/onnx'/f'{name}.onnx'
    adapted=original
    so=ort.SessionOptions();so.intra_op_num_threads=4;so.graph_optimization_level=ort.GraphOptimizationLevel.ORT_ENABLE_BASIC;so.optimized_model_filepath=str(folder/'basic.onnx')
    session=ort.InferenceSession(str(adapted),so,providers=['CPUExecutionProvider']);feed={};layouts=[]
    dtype={'tensor(float)':np.float32,'tensor(int64)':np.int64,'tensor(int32)':np.int32,'tensor(bool)':np.bool_}
    for inp in session.get_inputs():
        typ,shape,file=spec['in_'+inp.name];a=np.fromfile(cdir/file,np.dtype(typ)).reshape(shape)
        if tuple(a.shape)!=tuple(inp.shape):
            for axes in [(0,3,1,2),(0,2,3,1)]:
                if a.ndim==4 and tuple(a.transpose(axes).shape)==tuple(inp.shape):a=a.transpose(axes);layouts.append(dict(input=inp.name,axes=axes));break
            else:raise RuntimeError(f'INPUT_SHAPE {inp.name}: {a.shape} vs {inp.shape}')
        feed[inp.name]=np.ascontiguousarray(a,dtype=dtype[inp.type])
    reference=session.run(None,feed)
    model=onnx.shape_inference.infer_shapes(onnx.load(folder/'basic.onnx'))
    constants={x.name:numpy_helper.to_array(x) for x in model.graph.initializer};shapes={x.name:[d.dim_value for d in x.type.tensor_type.shape.dim] for x in list(model.graph.value_info)+list(model.graph.input)+list(model.graph.output)}
    nodes=[];fixes=[]
    # Propagate float masks through a closed island of boolean shape/cast/Where
    # operations; preserve other boolean islands rather than guessing semantics.
    masks=set();island_seeds=[]
    for gt in model.graph.node:
        if gt.op_type!='Greater' or gt.input[1] not in constants or not np.all(constants[gt.input[1]]==0):continue
        flow={gt.output[0]};valid=True;changed=True
        while changed:
            changed=False
            for n in model.graph.node:
                if not any(x in flow for x in n.input):continue
                if n.op_type=='Where' and n.input[0] in flow:continue
                if n.op_type=='Cast' and n.input[0] in flow and next((a.i for a in n.attribute if a.name=='to'),0)==onnx.TensorProto.FLOAT:continue
                if n.op_type in ('Unsqueeze','Squeeze','Reshape','Expand','Identity','Cast') and n.input[0] in flow:
                    for x in n.output:
                        if x not in flow:flow.add(x);changed=True
                else:valid=False
        if valid:masks.update(flow);island_seeds.append(gt.name)
    if masks:
        island=[]
        for n in model.graph.node:
            if n.name in island_seeds:
                z=n.name+'_z';one=n.name+'_one';mid=n.name+'_clip';model.graph.initializer.extend([numpy_helper.from_array(np.array(0,np.float32),z),numpy_helper.from_array(np.array(1,np.float32),one)]);island.extend([helper.make_node('Clip',[n.input[0],z,one],[mid]),helper.make_node('Ceil',[mid],list(n.output))]);fixes.append(dict(node=n.name,rewrite='finite_float_mask_island'));continue
            if n.op_type=='Cast' and n.input[0] in masks:island.append(helper.make_node('Identity',[n.input[0]],list(n.output)));continue
            if n.op_type=='Where' and n.input[0] in masks:
                one=n.name+'_one';inv=n.name+'_inv';left=n.name+'_left';right=n.name+'_right';model.graph.initializer.append(numpy_helper.from_array(np.array(1,np.float32),one));island.extend([helper.make_node('Sub',[one,n.input[0]],[inv]),helper.make_node('Mul',[n.input[0],n.input[1]],[left]),helper.make_node('Mul',[inv,n.input[2]],[right]),helper.make_node('Add',[left,right],list(n.output))]);continue
            island.append(n)
        model.graph.ClearField('node');model.graph.node.extend(island)
        for value in model.graph.value_info:
            if value.name in masks:value.type.tensor_type.elem_type=onnx.TensorProto.FLOAT
    producers={x:n for n in model.graph.node for x in n.output};consumers=collections.Counter(x for n in model.graph.node for x in n.input)
    fused={}
    for node in model.graph.node:
        if node.op_type=='Cast' and node.input and node.input[0] in producers:
            gt=producers[node.input[0]]
            to=next((a.i for a in node.attribute if a.name=='to'),None)
            if gt.op_type=='Greater' and to==onnx.TensorProto.FLOAT and consumers[gt.output[0]]==1:
                threshold=constants.get(gt.input[1])
                if threshold is not None and np.all(threshold==0):fused[node.name]=gt
    removed={gt.name for gt in fused.values()}
    where_masks={}
    for gt in model.graph.node:
        if gt.op_type!='Greater' or gt.input[1] not in constants or not np.all(constants[gt.input[1]]==0):continue
        uses=[n for n in model.graph.node if gt.output[0] in n.input]
        if uses and all((n.op_type=='Where' and n.input[0]==gt.output[0]) or (n.op_type=='Cast' and next((a.i for a in n.attribute if a.name=='to'),0)==onnx.TensorProto.FLOAT) for n in uses):
            where_masks[gt.output[0]]=gt;removed.add(gt.name)
    for node in model.graph.node:
        if node.op_type=='Greater' and node.output[0] in where_masks:
            zero=node.name+'_zero';one=node.name+'_one';clip=node.name+'_clip';mask=node.output[0]+'_floatmask'
            model.graph.initializer.extend([numpy_helper.from_array(np.array(0,np.float32),zero),numpy_helper.from_array(np.array(1,np.float32),one)])
            nodes.extend([helper.make_node('Clip',[node.input[0],zero,one],[clip]),helper.make_node('Ceil',[clip],[mask])]);fixes.append(dict(node=node.name,rewrite='finite_zero_threshold_mask_and_blend'));continue
        if node.op_type=='Cast' and node.input[0] in where_masks:
            nodes.append(helper.make_node('Identity',[node.input[0]+'_floatmask'],list(node.output)));continue
        if node.op_type=='Where' and node.input[0] in where_masks:
            mask=node.input[0]+'_floatmask';one=node.name+'_one';inv=node.name+'_inv';left=node.name+'_a';right=node.name+'_b'
            model.graph.initializer.append(numpy_helper.from_array(np.array(1,np.float32),one))
            nodes.extend([helper.make_node('Sub',[one,mask],[inv]),helper.make_node('Mul',[mask,node.input[1]],[left]),helper.make_node('Mul',[inv,node.input[2]],[right]),helper.make_node('Add',[left,right],list(node.output))]);continue
        if node.name in removed:continue
        if node.name in fused:
            gt=fused[node.name];zero=node.name+'_zero';one=node.name+'_one';clip=node.name+'_clip'
            model.graph.initializer.extend([numpy_helper.from_array(np.array(0,np.float32),zero),numpy_helper.from_array(np.array(1,np.float32),one)])
            nodes.extend([helper.make_node('Clip',[gt.input[0],zero,one],[clip]),helper.make_node('Ceil',[clip],list(node.output))])
            fixes.append(dict(node=node.name,rewrite='FP32 finite Greater(x,0)+Cast -> Ceil(Clip(x,0,1))'));continue
        if node.op_type=='ScatterND' and name.startswith('memattn_'):
            data,ind,updates=node.input;indices=constants.get(ind);ds,us=shapes[data],shapes[updates]
            length=us[2];expect=np.zeros((1,1,length,3),np.int64);expect[0,0,:,2]=np.arange(length)
            if indices is None or not np.array_equal(indices,expect) or ds[:2]!=[1,1] or us[:2]!=[1,1] or ds[3]!=256:raise RuntimeError('SCATTER_NOT_FIXED_PREFIX '+node.name)
            if ds[2]==length:nodes.append(helper.make_node('Identity',[updates],list(node.output),name=node.name+'_replace'))
            else:
                keys=[]
                for suffix,value in [('starts',[length]),('ends',[ds[2]]),('axes',[2]),('steps',[1])]:
                    key=node.name+'_'+suffix;keys.append(key);model.graph.initializer.append(numpy_helper.from_array(np.array(value,np.int64),key))
                tail=node.name+'_tail';nodes.append(helper.make_node('Slice',[data]+keys,[tail]));nodes.append(helper.make_node('Concat',[updates,tail],list(node.output),axis=2))
            fixes.append(dict(node=node.name,prefix=length,total=ds[2]))
        else:nodes.append(node)
    model.graph.ClearField('node');model.graph.node.extend(nodes);onnx.checker.check_model(model);onnx.save(model,folder/'source.onnx')
    verify=ort.SessionOptions();verify.intra_op_num_threads=4;verify.log_severity_level=3
    reference=ort.InferenceSession(str(original),verify,providers=['CPUExecutionProvider']).run(None,feed)
    converted_reference=ort.InferenceSession(str(folder/'source.onnx'),verify,providers=['CPUExecutionProvider']).run(None,feed)
    output_layouts=[]
    for i,(a,b) in enumerate(zip(reference,converted_reference)):
        if a.shape!=b.shape:
            for axes in [(0,3,1,2),(0,2,3,1)]:
                if a.ndim==4 and a.transpose(axes).shape==b.shape:reference[i]=a.transpose(axes);output_layouts.append(dict(output=i,axes=axes));break
            else:raise RuntimeError('ADAPTED_OUTPUT_SHAPE')
    source_checks=[dict(maxabs=float(np.max(np.abs(a.astype(np.float64)-b))),violations=int(np.count_nonzero(np.abs(a.astype(np.float64)-b)>1e-4+1e-4*np.abs(a)))) for a,b in zip(reference,converted_reference)]
    status.update(source_rewrite_checks=source_checks,output_layout_changes=output_layouts)
    assert all(x['violations']==0 for x in source_checks),'SOURCE_REWRITE_CHANGED_VALUES'
    status['source_rewrite_checks']=source_checks
    shape_arg=','.join('['+','.join(map(str,a.shape))+']'+('i64' if a.dtype==np.int64 else 'i32' if a.dtype==np.int32 else 'f32') for a in feed.values())
    args=[str(root/'dependencies/pnnx/pnnx-20261020-windows/pnnx.exe'),str(folder/'source.onnx'),'inputshape='+shape_arg,'fp16=0']
    actual_files=[]
    for i,a in enumerate(feed.values()):np.save(folder/f'trace_input_{i}.npy',a);actual_files.append(str(folder/f'trace_input_{i}.npy'))
    args.append('input='+','.join(actual_files))
    args=[x for x in args if not x.startswith('inputshape=')]
    for key,file in [('pnnxparam','model.pnnx.param'),('pnnxbin','model.pnnx.bin'),('pnnxpy','model_pnnx.py'),('pnnxonnx','model.pnnx.onnx'),('ncnnparam','model.param'),('ncnnbin','model.bin'),('ncnnpy','model_ncnn.py')]:args.append(key+'='+str(folder/file))
    with (folder/'conversion.log').open('w',encoding='utf-8') as f:p=subprocess.run(args,stdout=f,stderr=subprocess.STDOUT,timeout=180)
    if p.returncode:raise RuntimeError('PNNX_EXIT '+str(p.returncode))
    lines=(folder/'model.param').read_text().splitlines();ops=[x.split()[0] for x in lines[2:]]
    existing_tops=set()
    for line in lines[2:]:
        t=line.split();ni,no=map(int,t[2:4]);existing_tops.update(t[4+ni:4+ni+no])
    added_constants=[]
    for i,(desc,a) in enumerate(zip(session.get_outputs(),reference)):
        if f'out{i}' not in existing_tops and desc.name in constants:
            assert a.dtype==np.float32 and a.ndim==4 and a.shape[0]==1
            _,c,h,w=a.shape;lines.append(f'MemoryData const_output_{i} 0 1 out{i} 0={w} 1={h} 2={c} 21=1')
            with (folder/'model.bin').open('ab') as f:a.tofile(f)
            layer_count,blob_count=map(int,lines[1].split());lines[1]=f'{layer_count+1} {blob_count+1}';added_constants.append(desc.name)
    (folder/'model.param').write_text('\n'.join(lines)+'\n');ops=[x.split()[0] for x in lines[2:]];status['constant_outputs_materialized']=added_constants
    status.update(case=case,input_layout_changes=layouts,inputs={n:dict(shape=list(a.shape),dtype=str(a.dtype)) for n,a in feed.items()},outputs={o.name:dict(shape=list(a.shape),dtype=str(a.dtype)) for o,a in zip(session.get_outputs(),reference)},ncnn_counts=dict(collections.Counter(ops)),prefix_rewrites=fixes)
    unresolved=[x for x in set(ops) if x.startswith(('onnx.','pnnx.','torch.','Tensor.')) or x=='ScatterND']
    if unresolved:raise RuntimeError('UNRESOLVED_OPERATORS '+','.join(sorted(unresolved)))
    def make_net(param):
        net=ncnn.Net();net.opt.use_vulkan_compute=False;net.opt.use_packing_layout=False;net.opt.num_threads=4;net.opt.use_fp16_packed=net.opt.use_fp16_storage=net.opt.use_fp16_arithmetic=False
        assert net.load_param(str(param))==0,'NCNN_CPU_PARAM_LOAD';assert net.load_model(str(folder/'model.bin'))==0,'NCNN_CPU_WEIGHT_LOAD'
        ex=net.create_extractor();ex.set_light_mode(False);return net,ex
    net,ex=make_net(folder/'model.param');input_lines=[]
    generated=(folder/'model_ncnn.py').read_text()
    batch_indices=re.findall(r'batch_index=(\d+)',generated)
    for i,(key,a) in enumerate(feed.items()):
        batch=int(batch_indices[i]) if i<len(batch_indices) else 233
        typed=a.astype(np.int32) if a.dtype in (np.int64,np.int32) else a
        m=ncnn.Mat(typed,batch_index=batch).clone();assert ex.input(f'in{i}',m)==0
        file=f'input_{i}.bin';typed.tofile(folder/file)
        input_lines.append(f'in{i} {m.dims} {m.w} {m.h} {m.d} {m.c} {m.n} {typed.dtype} {file}')
    parsed=[];shape_lines=[];rewrite=[];new=lines[:2]
    for line in lines[2:]:
        t=line.split();ni,no=map(int,t[2:4]);tops=t[4+ni:4+ni+no];kind,lname=t[:2];parsed.append((kind,lname,tops))
        if kind in ('Reshape','ExpandDims'):
            rc,m=ex.extract(tops[0]);assert rc==0
            shape_lines.append(f'{lname} {m.dims} {m.w} {m.h} {m.d} {m.c}')
            if kind=='ExpandDims':
                opt=f'0={m.w}'+(f' 1={m.h}' if m.dims>=2 else '')+(f' 2={m.c}' if m.dims>=3 else '')+(f' 11={m.d}' if m.dims==4 else '')
                line=f'Reshape {lname} 1 1 {t[4]} {tops[0]} {opt}';rewrite.append(lname)
        new.append(line)
    before=[];outputs=[];checks=[]
    for i,(desc,expected) in enumerate(zip(session.get_outputs(),reference)):
        rc,m=ex.extract('out'+str(i));assert rc==0;actual=m.numpy(batch_index=233).copy();before.append(actual)
        assert actual.size==expected.size
        expected.astype(np.float32).reshape(-1).tofile(folder/f'ref_{i}.bin')
        d=np.abs(actual.reshape(-1).astype(np.float64)-expected.reshape(-1));bad=int(np.count_nonzero(d>1e-4+1e-4*np.abs(expected.reshape(-1))))
        checks.append(dict(output=desc.name,maxabs=float(d.max()),violations=bad,elements=expected.size));outputs.append(f'out{i} {expected.size} ref_{i}.bin ref_{i}.bin')
    (folder/'model.param').write_text('\n'.join(new)+'\n');(folder/'inputs.txt').write_text('\n'.join(input_lines)+'\n');(folder/'shapes.txt').write_text('\n'.join(shape_lines)+'\n');(folder/'refs.txt').write_text('\n'.join(outputs)+'\n')
    status.update(status='PREPARED',shape_aliases=len(shape_lines),expanddims_rewrites=rewrite,pc_ncnn_vs_onnx=checks)
    (folder/'files.json').write_text(json.dumps(['model.param','model.bin','inputs.txt','shapes.txt','refs.txt']+[f'input_{i}.bin' for i in range(len(feed))]+[f'ref_{i}.bin' for i in range(len(reference))]))
except Exception as e:
    status.update(status='OFFLINE_BLOCKED',reason=str(e));traceback.print_exc()
finally:(folder/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')
