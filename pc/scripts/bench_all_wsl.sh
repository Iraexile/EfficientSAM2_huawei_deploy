#!/bin/bash
# Benchmark-validate every .ms: proves the MindSpore Lite runtime can load
# and execute all ops (CPU backend, random inputs, 1 warm-up + 2 iters).
set -u
MSL=~/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$MSL/tools/benchmark/lib:$MSL/runtime/lib:$MSL/runtime/third_party/glog:${LD_LIBRARY_PATH:-}
BM=$MSL/tools/benchmark/benchmark

DST=/mnt/c/Users/hp/EfficientSAM2/pc/outputs/ms
LOG=/mnt/c/Users/hp/EfficientSAM2/pc/outputs/ms/benchmark_log.txt
: > "$LOG"

for f in "$DST"/*.ms; do
  name=$(basename "$f" .ms)
  echo "=== $name ===" | tee -a "$LOG"
  if out=$("$BM" --modelFile="$f" --device=CPU --loopCount=2 --warmUpLoopCount=1 --numThreads=4 2>&1); then
    avg=$(echo "$out" | grep -oP 'AvgRunTime = \K[0-9.]+' | tail -1)
    echo "OK  $name  avg=${avg}ms" | tee -a "$LOG"
  else
    echo "FAIL $name" | tee -a "$LOG"
    echo "$out" | tail -20 >> "$LOG"
  fi
done
echo "BENCH DONE"
