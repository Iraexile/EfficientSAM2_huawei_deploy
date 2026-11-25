#!/bin/bash
# Convert all 21 graphs to .ms with MindSpore Lite 2.11.0 converter (in WSL).
# samheads use the *_fix.onnx variants produced by prepare_mslite.py.
set -u
MSL=~/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$MSL/tools/converter/lib:${LD_LIBRARY_PATH:-}
CONV=$MSL/tools/converter/converter/converter_lite

SRC=/mnt/c/Users/hp/EfficientSAM2/pc/outputs/onnx
DST=/mnt/c/Users/hp/EfficientSAM2/pc/outputs/ms
LOG=/mnt/c/Users/hp/EfficientSAM2/pc/outputs/ms/convert_log.txt
mkdir -p "$DST"
: > "$LOG"

ok=0; fail=0
for f in "$SRC"/*.onnx; do
  name=$(basename "$f" .onnx)
  case "$name" in
    samheads_cond)  f="$SRC/samheads_cond_fix.onnx" ;;
    samheads_track) f="$SRC/samheads_track_fix.onnx" ;;
    *_sim*|*_fix|*_mslite|_*) continue ;;  # intermediates / scratch
  esac
  echo "=== $name ($(basename "$f")) ===" | tee -a "$LOG"
  if "$CONV" --fmk=ONNX --modelFile="$f" --outputFile="$DST/$name" >>"$LOG" 2>&1; then
    echo "OK   $name" | tee -a "$LOG"; ok=$((ok+1))
  else
    echo "FAIL $name" | tee -a "$LOG"; fail=$((fail+1))
  fi
done
echo "DONE: $ok ok, $fail fail"
