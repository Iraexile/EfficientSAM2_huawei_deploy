#!/bin/bash
M=$HOME/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$M/tools/converter/lib:$M/runtime/lib:$M/runtime/third_party/glog
C=$M/tools/converter/converter/converter_lite
D=/mnt/c/Users/hp/EfficientSAM2/pc/outputs
mkdir -p $D/ms_opprobe 2>/dev/null || true
for f in $D/onnx_opprobe/*.onnx; do
  b=$(basename "$f" .onnx)
  $C --fmk=ONNX --modelFile="$f" --outputFile="$D/ms_opprobe/$b" > /tmp/conv_op_$b.log 2>&1 && echo "OK $b" || { echo "FAIL $b"; tail -2 /tmp/conv_op_$b.log; }
done
