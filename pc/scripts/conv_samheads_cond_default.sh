#!/bin/bash
# samheads_cond: fall back to DEFAULT conversion (NHWC I/O at 4D tensors).
# The NCHW conversion corrupts the point-prompt path (cond-only failure;
# track has no point inputs and passes with NCHW).
set -e
M=$HOME/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$M/tools/converter/lib:$M/runtime/lib:$M/runtime/third_party/glog
C=$M/tools/converter/converter/converter_lite
D=/mnt/c/Users/hp/EfficientSAM2/pc/outputs
$C --fmk=ONNX --modelFile=$D/onnx/samheads_cond_fix.onnx --outputFile=$D/ms/samheads_cond 2>&1 | grep -iE "error|CONVERT RESULT" | head -3
export LD_LIBRARY_PATH=$M/tools/benchmark/lib:$M/runtime/lib:$M/runtime/third_party/glog
$M/tools/benchmark/benchmark --modelFile=$D/ms/samheads_cond.ms --device=CPU --loopCount=1 --warmUpLoopCount=0 --numThreads=4 2>&1 | grep -E "AvgRunTime|ERROR|Failed" | head -3
