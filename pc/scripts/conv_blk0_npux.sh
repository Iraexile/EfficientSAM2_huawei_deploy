#!/bin/bash
M=$HOME/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$M/tools/converter/lib:$M/runtime/lib:$M/runtime/third_party/glog
C=$M/tools/converter/converter/converter_lite
D=/mnt/c/Users/hp/EfficientSAM2/pc/outputs
$C --fmk=ONNX --modelFile=$D/onnx/enc_cut_blk0.onnx --outputFile=$D/ms_npux/blk0_nchw_fp16 --inputDataFormat=NCHW --outputDataFormat=NCHW --fp16=on > /tmp/c2.log 2>&1 && echo OK nchw_fp16 || tail -3 /tmp/c2.log
$C --fmk=ONNX --modelFile=$D/onnx/enc_cut_blk0.onnx --outputFile=$D/ms_npux/blk0_nhwc_fp16 --fp16=on > /tmp/c3.log 2>&1 && echo OK nhwc_fp16 || tail -3 /tmp/c3.log
$C --fmk=ONNX --modelFile=$D/onnx/enc_cut_blk0.onnx --outputFile=$D/ms_npux/blk0_nhwc_transf --optimizeTransformer=true > /tmp/c4.log 2>&1 && echo OK nhwc_transf || tail -3 /tmp/c4.log
ls -la $D/ms_npux/
