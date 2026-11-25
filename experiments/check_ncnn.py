from pathlib import Path
import sys
import numpy as np
root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root/'dependencies/python'))
import ncnn
for name,ins in [('matmul',['q','kt']),('attention',['q','kt','v'])]:
    base=root/'experiments/artifacts/models'/name
    data=np.load(str(base)+'.npz')
    with ncnn.Net() as net:
        net.opt.use_vulkan_compute=False
        net.opt.use_fp16_packed=False;net.opt.use_fp16_storage=False;net.opt.use_fp16_arithmetic=False
        assert net.load_param(str(base)+'.ncnn.param')==0
        assert net.load_model(str(base)+'.ncnn.bin')==0
        with net.create_extractor() as ex:
            for i,key in enumerate(ins): assert ex.input(f'in{i}',ncnn.Mat(data[key],batch_index=233).clone())==0
            rc,got=ex.extract('out0');assert rc==0
            got=got.numpy(batch_index=233)
            err=float(np.max(np.abs(got-data['expected'])))
            assert np.isfinite(got).all() and err<1e-4
            print(f'{name}: NCNN_PC_CPU maxabs={err:.9g} shape={got.shape} vulkan=OFF')

base=root/'experiments/artifacts/models/memattn_k01_fixed'
case=root/'pc/device_vectors/memattn_004'
import onnxruntime as ort
session=ort.InferenceSession(str(root/'pc/outputs/onnx/memattn_k01.onnx'),providers=['CPUExecutionProvider'])
feed={i.name:np.fromfile(case/f'in_{i.name}.bin',np.float32).reshape(i.shape) for i in session.get_inputs()}
reference=session.run(None,feed)[0]
with ncnn.Net() as net:
    net.opt.num_threads=4
    net.opt.use_vulkan_compute=False
    net.opt.use_fp16_packed=False;net.opt.use_fp16_storage=False;net.opt.use_fp16_arithmetic=False
    assert net.load_param(str(base)+'.ncnn.param')==0, 'ncnn model load failed'
    assert net.load_model(str(base)+'.ncnn.bin')==0
    with net.create_extractor() as ex:
        for i,key in enumerate(feed): assert ex.input(f'in{i}',ncnn.Mat(feed[key],batch_index=233).clone())==0
        rc,got=ex.extract('out0');assert rc==0
        print(f'memattn_ncnn_layout dims={got.dims} w={got.w} h={got.h} d={got.d} c={got.c} elempack={got.elempack}')
        got=got.numpy(batch_index=233).reshape(reference.shape)
        assert np.isfinite(got).all()
        diff=np.abs(got-reference)
        bad=int(np.count_nonzero(diff>1e-4+1e-4*np.abs(reference)))
        print(f'memattn_k01_fixed: NCNN_PC_CPU_vs_ORT maxabs={diff.max():.9g} violations={bad}/{diff.size} shape={got.shape} vulkan=OFF')
        np.save(root/'experiments/artifacts/models/memattn_k01_fixed.pc_prediction.npy',got)
        if bad: print('status=STRICT_PARITY_FAIL requires_diagnosis=1')
