#!/bin/bash
set -u
MSL=~/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$MSL/tools/benchmark/lib:$MSL/runtime/lib:$MSL/runtime/third_party/glog:${LD_LIBRARY_PATH:-}
BM=$MSL/tools/benchmark/benchmark
DST=/mnt/c/Users/hp/EfficientSAM2/pc/outputs/ms
"$BM" --modelFile="$DST/samheads_cond.ms" --device=CPU --loopCount=1 --warmUpLoopCount=0 --numThreads=4 2>&1 | grep -iE "error|fail|kernel|support|node" | head -25
