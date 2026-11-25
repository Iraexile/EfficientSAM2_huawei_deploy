# Mate 80 GPU 21-graph execution and profiling baseline

Date: 2026-11-25. Device: Mate 80, Vulkan device `Maleoon 920`, driver `3104880233`, Vulkan API `4206901` as printed by native device logs. Backend: ncnn + Vulkan, FP32. The CPU and K01 baseline commits remain unchanged.

All 21 graphs have been evaluated for conversion/loading and 18 produced complete device GPU outputs. Status: 2 `GPU_EXECUTED`, 16 `GPU_EXECUTED_WITH_ERROR`, 2 `BLOCKED_CONVERSION`, 1 `BLOCKED_LOADING`. Machine-readable per-graph and per-operator tables are `experiments/logs/gpu_suite_summary.csv` and `experiments/logs/gpu_suite_op_profile.csv`. Raw logs and GPU output binaries remain under ignored `experiments/artifacts/gpu-suite/device-results/`.

## Execution and correctness

The 16 Memory Attention graphs (`k01`–`k15`, `steady`) produced full outputs on the device through explicit Vulkan layer forwards. Their logs identify Maleoon 920, list layer mappings with `vulkan=1`, and report `cpu_forward_calls=0`. Static reshapes and splits use GPU buffer metadata aliases, not CPU numerical execution. Each of the 16 saved outputs was independently re-read on PC and compared with its original ONNX FP32 reference: `abs(gpu-reference) <= 1e-4 + 1e-4*abs(reference)` elementwise. Each output has 1,048,576 elements and no nonfinite values.

| Graph | Device execution | Strict violations | Max abs error | Submit/wait ms | Requested peak MiB |
|---|---|---:|---:|---:|---:|
| memattn_k01 | GPU, pass | 0 | 0.00003767 | 8771.8 | 115.0 |
| memattn_k02 | GPU, pass | 0 | 0.00009811 | 12860.0 | 198.2 |
| memattn_k03 | GPU, tolerance exceeded | 8 | 0.00023282 | 15772.5 | 285.3 |
| memattn_k04 | GPU, tolerance exceeded | 13 | 0.00019932 | 19689.5 | 372.3 |
| memattn_k05 | GPU, tolerance exceeded | 75 | 0.00020576 | 22900.6 | 459.4 |
| memattn_k06 | GPU, tolerance exceeded | 602 | 0.00027251 | 26681.0 | 546.5 |
| memattn_k07 | GPU, tolerance exceeded | 2543 | 0.00040960 | 30327.4 | 633.6 |
| memattn_k08 | GPU, tolerance exceeded | 4152 | 0.00046498 | 30731.1 | 633.7 |
| memattn_k09 | GPU, tolerance exceeded | 3257 | 0.00043422 | 30488.1 | 633.8 |
| memattn_k10 | GPU, tolerance exceeded | 4584 | 0.00044012 | 30506.5 | 633.8 |
| memattn_k11 | GPU, tolerance exceeded | 5676 | 0.00046098 | 30527.2 | 633.9 |
| memattn_k12 | GPU, tolerance exceeded | 6986 | 0.00050819 | 30815.9 | 634.0 |
| memattn_k13 | GPU, tolerance exceeded | 6566 | 0.00043702 | 30459.5 | 634.1 |
| memattn_k14 | GPU, tolerance exceeded | 6375 | 0.00042188 | 30530.7 | 634.2 |
| memattn_k15 | GPU, tolerance exceeded | 5486 | 0.00039482 | 30592.2 | 634.3 |
| memattn_steady | GPU, tolerance exceeded | 5221 | 0.00040710 | 30262.0 | 634.4 |
| memenc_binarize | GPU, tolerance exceeded | 262110 / 262144 in output 0 | 5.21269 | 715.1 | 49.0 |
| memenc_sigmoid | GPU, tolerance exceeded | 262110 / 262144 in output 0 | 5.40505 | 635.9 | 49.0 |

These are single diagnostic runs, not steady-state throughput measurements. K01's earlier dedicated strict report remains the stronger repeated-run evidence. The old native log labels its ONNX reference `PC_NCNN`; the reference file provenance and independent comparison in the new summary clarify the actual comparator. The K03–steady tolerance failures are numerical discrepancies, not evidence that their GPU execution failed. Accumulation/rounding in longer attention histories is a plausible hypothesis; the first diverging operator has not yet been isolated, so no root cause is claimed.

| Graph | Current boundary | Device execution |
|---|---|---|
| image_encoder | Converted representation contains unresolved `Tensor.reshape` window partition operations (rank-six intermediate). | Not executed. |
| memenc_binarize | Initial device attempt crashed near `ConvolutionDepthWise`. Packing compatibility and readback layout fixes let all 135 layers finish and both outputs be read back. Output 0 differs substantially; output 1 matches exactly. | `GPU_EXECUTED_WITH_ERROR`. |
| memenc_sigmoid | Initial device attempt stalled near `ConvolutionDepthWise`. The same two fixes let all 134 layers finish and both outputs be read back. Output 0 differs substantially; output 1 matches exactly. | `GPU_EXECUTED_WITH_ERROR`. |
| samheads_cond | Original PNNX conversion exits `3221225477`; an altered-output variant does not preserve the seven-output contract. Integer point labels require explicit support. | Not executed. |
| samheads_track | Conversion/load does not yet yield a complete runnable ncnn graph; `ArgMax` and other unresolved operations appear in generated graph. | Not executed. |

No NPU tests were launched for this phase. These conversion/framework limitations are not chip capability conclusions.

## Profiling interpretation

The device runner records each layer's CPU encode time and blocking submit/wait interval, plus output readback, comparison/save wall time, process high-water RSS and the peak sum of requested live GPU buffers. The latter reached about 634.4 MiB on `steady`; it excludes weights, allocator pools, driver allocations and all process resources. Process HWM also contains unrelated inference and diagnostic allocations, so neither is a physical GPU memory peak.

Across the 18 completed diagnostic runs, `SDPA` contributed about 374,673 ms of 413,267 ms measured submit/wait time (90.7%). `Gemm` contributed about 25,644 ms (6.2%). These intervals include GPU work, queueing and CPU synchronization; hardware kernel-only timestamps were not collected. Every operator is submitted and waited separately in this diagnostic runner, so these numbers overstate optimized pipeline latency. Longer attention history grows requested memory to a plateau around 634 MiB and drives SDPA time near 30 s per graph. Future optimization should batch submissions and eliminate per-layer waits, then profile SDPA internals with reliable GPU timestamps.

## Reproduction and remaining work

Run `experiments/summarize_gpu_suite.py` with the bundled Python/numpy environment to regenerate both CSVs from the local device result directories. The generated CSV reports every graph's IO shapes, dtype, converted operator counts, status, raw log path, exact error count, timing and memory fields. The script independently reads `gpu_out*.bin` and `ref_*.bin` for correctness. No input/output raw tensors are candidates for the public snapshot.

Device command for the signed diagnostic HAP: `hdc install entry/build/default/outputs/default/entry-default-signed.hap`, then `hdc shell aa start -a GpuSuiteAbility -b com.example.efficientsam2 -m entry`. The 16 Attention runs used the USB suite payloads; the two Memory Encoder runs used bundled raw resources after reverse HTTP returned `CURLcode 7`. Their final native logs are `experiments/artifacts/gpu-suite/device-results/isolated/memenc_{binarize,sigmoid}/retry2-graph-probe.txt`; original failed attempts and first retries remain beside them. The complete Attention logs and outputs are under `device-results/run1/<graph>/`. The HAP includes generated model assets and is local-only.

GPU kernel-only time is unavailable in this run. Per-operator `submit_wait_ms` is a host interval, not a hardware timestamp. `peak_requested_bytes` is the sum of requested live Vulkan buffers, not device physical allocation; process `VmHWM` is reported separately in the CSV. Upload, readback, encode, compare/save and total diagnostic wall intervals are in the same CSV. The status checker verified 21 rows, 18 complete GPU runs, zero CPU forward calls in those runs, and independently compared every saved output binary.

The three blocked graphs are `image_encoder` (`Tensor.reshape` rank-six window operation not represented by ncnn conversion), `samheads_cond` (PNNX access violation `3221225477` with original seven-output contract), and `samheads_track` (generated `ArgMax`/`Gather`/Torch operations cannot load as a complete ncnn model). These are backend/conversion boundaries, not Maleoon hardware impossibility claims. They have no device execution time. K03 and longer histories have small numerical tolerance exceedances, which were recorded without precision tuning. The two Memory Encoder outputs show large error and need a later correctness investigation before use in a full pipeline. The USB reverse HTTP path failed with device `CURLcode 7`; a temporary self-contained HAP provided an offline device path. No full 18-frame pipeline result is asserted.
