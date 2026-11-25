#!/bin/bash
M=$HOME/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$M/tools/converter/lib:$M/runtime/lib:$M/runtime/third_party/glog
C=$M/tools/converter/converter/converter_lite
D=/mnt/c/Users/hp/EfficientSAM2/pc/outputs
$C --fmk=ONNX --modelFile=$D/onnx/enc_cut_blk0_fold.onnx --outputFile=$D/ms_npux/blk0_fold --inputDataFormat=NCHW --outputDataFormat=NCHW > /tmp/c5.log 2>&1 && echo OK blk0_fold || tail -3 /tmp/c5.log
