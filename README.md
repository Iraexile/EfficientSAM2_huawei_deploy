# EfficientSAM2_huawei_deploy
EfficientSAM2模型端侧部署到huawei mate80(Kirin 9020) 的cpu/gpu/npu性能测试。
cpu后端采用官方mindsporelite推理。

GPU 实验后端采用 ncnn + Vulkan。Mate 80 的 Maleoon 920 上，21 张计算图已完成状态评估：18 张完整 GPU 执行并产生输出，其中 2 张满足原数值容差，16 张数值超限；另外 2 张转换阻塞、1 张加载阻塞。逐图证据见 [GPU 报告](experiments/GPU_21_GRAPH_REPORT.md)及 [逐图 CSV](experiments/logs/gpu_suite_summary.csv)、[算子 Profiling CSV](experiments/logs/gpu_suite_op_profile.csv)。完整 18 帧 GPU 流水线尚未验证。

CPU 基线为 MindSpore Lite；NPU 的 NNRT 异构和 MNN + HiAI/CANN 仍处于诊断及依赖研究阶段，未完成 NPU 部署或 Profiling。

此公开分支是经过隐私审查的衍生快照，保留了本仓库原有 README 提交历史；原始科研 Git 历史、原始设备日志、模型权重、签名材料和大型张量仅保存在本地。原始 CPU 基线 SHA：`5d3d33de0d157ca4587f99bfacf06735bb109afc`；K01 严格 GPU 基线 SHA：`1307c725674171579ef45dee0ff87b2fe2d2c18a`，本地 Tag：`k01-strict-gpu-verified-20261124`。这些 SHA 是溯源标识，本公开提交不是原始基线提交。

工程结构：`entry/src/main/cpp` 为 HarmonyOS CPU/GPU/NPU 原生代码，`entry/src/main/ets` 为应用入口；`pc/scripts` 为模型转换与参考对拍工具；`experiments/native` 与 `experiments/*.py`、`*.ps1` 为硬件实验代码。运行模型需要自行按许可取得权重并生成模型文件；仓库不提供权重、ONNX、MindSpore `.ms` 或 ncnn `.bin`。
