# Project status — 2026-11-25

| Area | Observed status |
|---|---|
| CPU | Original MindSpore Lite implementation preserved locally. Of 71 strict parity cases, 58 pass and 13 exceed the original tolerance; maximum absolute error `2.861e-4`. See `experiments/CPU_AUDIT.md`. |
| Minimal Vulkan | Maleoon 920 GPU compute submission, readback and PC reference comparison verified on Mate 80. |
| K01 strict GPU | Original tagged local baseline passed two complete runs with all 612 intermediate checks and final reference checks. |
| 21 GPU graphs | 18 completed on GPU with outputs and profiling; 2 within strict tolerance, 16 exceeded it. Two conversion blockers and one load blocker remain. See `experiments/GPU_21_GRAPH_REPORT.md`. |
| 18-frame GPU pipeline | Not yet executed end to end on GPU. |
| NPU | Existing NNRT + CPU diagnostic work exists; MNN + HiAI/CANN route is exploratory. No NPU execution/profiling success is claimed. |

This branch is a reviewed public derivative snapshot. Raw device evidence, generated models and original local Git objects are retained separately; missing raw data is not evidence that an experiment was not performed.
