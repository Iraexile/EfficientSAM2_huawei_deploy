#!/bin/bash
M=$HOME/mslite/mindspore-lite-2.11.0-linux-x64
export LD_LIBRARY_PATH=$M/tools/converter/lib:$M/runtime/lib:$M/runtime/third_party/glog
$M/tools/converter/converter/converter_lite --help 2>&1
