#!/bin/bash
# Re-convert samheads from the *_fix.onnx (onnxsim-folded) variants with NCHW I/O.
# The plain samheads_*.onnx still carries the symbolic-shape MulFusion that the
# runtime cannot schedule (M2 已知问题, fix=常量折叠).
set -e
M=$HOME/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$M/tools/converter/lib:$M/runtime/lib:$M/runtime/third_party/glog
C=$M/tools/converter/converter/converter_lite
D=/mnt/c/Users/hp/EfficientSAM2/pc/outputs
for b in samheads_cond samheads_track; do
  echo "=== $b from ${b}_fix.onnx ==="
  $C --fmk=ONNX --modelFile=$D/onnx/${b}_fix.onnx --outputFile=$D/ms/$b \
     --inputDataFormat=NCHW --outputDataFormat=NCHW 2>&1 | grep -iE "error|CONVERT RESULT" | head -3
done
# x86 smoke-validate immediately
export LD_LIBRARY_PATH=$M/tools/benchmark/lib:$M/runtime/lib:$M/runtime/third_party/glog
BM=$M/tools/benchmark/benchmark
for b in samheads_cond samheads_track; do
  echo "=== bench $b ==="
  $BM --modelFile=$D/ms/$b.ms --device=CPU --loopCount=1 --warmUpLoopCount=0 --numThreads=4 2>&1 | grep -E "AvgRunTime|ERROR|Failed" | head -3
done
