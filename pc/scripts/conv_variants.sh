#!/bin/bash
set -x
M=$HOME/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$M/tools/converter/lib:$M/runtime/lib:$M/runtime/third_party/glog
C=$M/tools/converter/converter/converter_lite
D=/mnt/c/Users/hp/EfficientSAM2/pc/outputs
$C --fmk=ONNX --modelFile=$D/onnx/enc_cut_conv.onnx --outputFile=$D/ms/enc_cut_conv_nchw --inputDataFormat=NCHW --outputDataFormat=NCHW 2>&1 | tail -2
ls -la $D/ms/ | grep nchw
