#!/bin/bash
# Re-convert ALL ONNX models with explicit NCHW input/output data format.
# Root cause fix: default conversion exports 4D-input models expecting NHWC
# data layout; our harness feeds NCHW -> silent garbage. The flags make the
# converter insert pre/post permute nodes so model I/O stays NCHW.
set -e
M=$HOME/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$M/tools/converter/lib:$M/runtime/lib:$M/runtime/third_party/glog
C=$M/tools/converter/converter/converter_lite
D=/mnt/c/Users/hp/EfficientSAM2/pc/outputs

mkdir -p $D/ms_nhwc_default
for f in $D/ms/*.ms; do
  b=$(basename "$f")
  [ -f "$D/ms_nhwc_default/$b" ] || cp "$f" "$D/ms_nhwc_default/$b"
done

cd $D/onnx
ok=0; fail=0
for f in *.onnx; do
  b="${f%.onnx}"
  if $C --fmk=ONNX --modelFile="$f" --outputFile="$D/ms/$b" \
        --inputDataFormat=NCHW --outputDataFormat=NCHW > /tmp/conv_$b.log 2>&1; then
    ok=$((ok+1)); echo "OK   $b"
  else
    fail=$((fail+1)); echo "FAIL $b"; tail -3 /tmp/conv_$b.log
  fi
done
echo "== done: $ok ok, $fail fail =="
ls $D/ms/*.ms | wc -l
