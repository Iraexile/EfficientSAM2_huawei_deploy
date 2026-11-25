#!/bin/bash
M=$HOME/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$M/tools/converter/lib:$M/runtime/lib:$M/runtime/third_party/glog
C=$M/tools/converter/converter/converter_lite
D=/mnt/c/Users/hp/EfficientSAM2/pc/outputs
for b in memenc_binarize memenc_sigmoid; do
  echo "=== $b fonly + both flags ==="
  $C --fmk=ONNX --modelFile=$D/onnx/${b}_fonly.onnx --outputFile=$D/ms/$b \
     --inputDataFormat=NCHW --outputDataFormat=NCHW 2>&1 | grep -iE "error|CONVERT RESULT" | head -3
done
ls -la $D/ms/ | grep memenc
