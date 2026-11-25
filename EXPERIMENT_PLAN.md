# Mate 80 Efficient-SAM2 实验计划

## GPU 阶段结论（2026-11-25）

21 张图均已完成状态评估：18 张在 Mate 80 Maleoon 920 上通过显式 Vulkan 路径完整执行并保存输出，其中 2 张满足原严格阈值，16 张记录为 `GPU_EXECUTED_WITH_ERROR`。另 2 张在模型转换阶段受阻、1 张在 ncnn 加载阶段受阻；不得称为 21 张全部执行成功。逐图状态、形状、耗时、Buffer 请求峰值、算子统计和日志索引见 `experiments/GPU_21_GRAPH_REPORT.md`、`experiments/logs/gpu_suite_summary.csv`、`experiments/logs/gpu_suite_op_profile.csv`。Memory Encoder 两图第一输出误差严重，需后续核查；K03 至 steady 的数值超限暂不深调。GPU 阶段的执行覆盖率与 Profiling 摸底已收口，完整 18 帧 GPU 流水线未验证。NPU 阶段尚未开始。

更新：2026-11-24。关联 ChatGPT 中给 Kimi 的部署任务作为总体参考；本文件按当前源码和保存的实验数据修正，不把历史建议当作已执行结果。

## 状态与证据

| 项目 | 状态 | 证据与边界 |
|---|---|---|
| CPU 工程、71 例、18 帧 | 已完成历史运行；本轮离线审计 | pc/device_results/report.txt、pipeline_report.txt；严格容差需逐案例报告 |
| NNRT 与 CPU 同时注册 | 已完成源码核查 | mslite_runner.cpp ModelGuard::Load；“未添加 CPU”假设被源码否定 |
| NNRT 构图失败 | 历史真机证据 | report_npu.txt、probe_npu.txt；status=-1 不唯一定位根因 |
| 单算子 NPU 支持 | 待验证实际执行 | BUILD-OK 仅为构图成功，不能证明无 CPU 回退 |
| 系统 MSLite GPU | 历史构图失败 | report_gpu.txt status=-2；不代表其他 GPU 后端失败 |
| ncnn Vulkan GPU | 待验证；等待真机 | 准备独立测试程序，不修改 CPU 流水线 |
| Mobiinfer hiai_npu | 待验证 | 核查固定提交、MNN 后端、SDK/HiAI/CANN 依赖；不大规模移植 |
| “芯片仅可用 CPU” | 证据不支持的外推 | 当前失败仅限定测试的软件路径 |

## 执行顺序

1. 保存 Git 基线 cpu-baseline-20261124。排除模型/数据/依赖/缓存/签名；保留原始结果。根 build-profile.json5 含签名配置，仅留本机；config/build-profile.unsigned.json5 为无签名模板。
2. 按源码 atol=rtol=1e-4 审计 71 例。列出 FAIL、逐输出最大误差、日志的舍入余弦值；不将舍入为 1 的余弦视为任务正确性证明。缺失预测张量则标记无法重算逐元素超限率和分割影响。
3. GPU：ncnn 官方 HarmonyOS Vulkan 固定版本 → 设备和 Compute 队列 → 显式 VkMat MatMul dispatch/submit/download → CPU 参考误差 → QK/Softmax/AV 最小 Attention → memattn_k01 转换与验证 → 18 帧。任何前置失败先最小修复，不直接迁移全部图。
4. NPU：逐图保存输入输出 Shape/dtype 与算子；检查现有构图日志的信息缺口。比较 NNRT 通用设备创建与按 accelerator 类型创建；CPU 对照、NNRT+CPU、可证明执行的 NPU 最小图。明确 BUILD 与 PREDICT 阶段，记录设备分区或执行证据。
5. Mobiinfer：检查 hiai_npu 分支的在线/离线编译、HiAI DDK/CANN Kit 版本和 HarmonyOS 库，依赖可用后先最小 MatMul，再 Attention。

## 每阶段交付与判据

- 代码、固定依赖版本/hash、构建命令及原始日志入 experiments；大型二进制留 experiments/artifacts 或 dependencies，不入 Git。
- 正确性同时报告绝对误差与相对容差；FP16 单独标识，不能改变 CPU 基线验收口径。
- 显式 Vulkan 命令缓冲提交和读回为 GPU 计算证据；网络启用 use_vulkan_compute 本身不足以证明无回退。
- NPU 设备枚举、构图成功、延迟下降均不足以单独证明 NPU 执行；需调度/分区/运行时证据。
- 性能区分冷启动/构图、预热、推理中位数及端到端含传输延迟；PC CUDA 只作参考。

## 离线结束条件

保存可构建测试工程、可用产物、CPU 审计、依赖缺口和真机步骤。设备暂不连接，不安装、不冒充真机结果。完成可行的离线准备后暂停，待用户连接 Mate 80。

## 本轮离线结果

2026-11-24 k01更新：独立MemattnK01Ability完成475层显式路径映射、612个非输入张量与PC逐点比较、21个关键中间结果和最终输出ONNX对照、最终PyTorch golden对照。两次完整运行635项比较均符合原容差；具体PC独立复核、计时边界和风险见experiments/K01_STRICT_GPU_REPORT.md及logs/k01_device_audit.json。禁止把旧Net入口的Vulkan选项当作无回退证明；该入口现在拒绝k01执行。本轮仅k01，等待审核再扩展。

2026-11-24 真机第一阶段更新：Maleoon 920显式Vulkan Gemm/Softmax/Gemm提交、读回和PC对齐已成功，详见experiments/GPU_DEVICE_RESULT.md。原shell入口退出126；改用独立签名GpuProbeAbility。真实memattn、NPU及完整模型仍待用户审核后验证。

- 已完成：Git 基线；71 例审计（58 OK/13 FAIL）；42 个原始 ONNX 的 Shape/dtype/opset/hash 清单；原 HAP 构建；3 个独立 ARM64 程序构建。
- 已完成：ncnn/pnnx 20261020 的最小模型转换和 PC CPU 验证；最小 Attention 对照 maxabs=1.1176e-8。
- 已完成局部修复：原 memattn_k01 转换残留 8 个 ScatterND，尽管转换器 exit=0。按固定索引语义替换为 Identity 或 Slice+Concat，保留 4 个 pointer token。原图/改写图在两组输入逐位一致；改写后 ncnn PC CPU 对照 ORT maxabs=2.9564e-5，0/1048576 超限。
- 已完成：WSL MSLite 转换最小 MatMul/Attention 为 .ms。Windows 转换器此次无输出退出 1，采用已有 WSL 转换器，不推断模型失败。
- 待验证/等待真机：Vulkan 设备/驱动/实际计算；NNRT 三配置构图与预测；NPU 分区执行证明；真实 memattn Vulkan 数值与延迟；18 帧硬件路径。
- 依赖缺口：Mobiinfer HiAI 后端需要独立 CANN Kit 的头文件和 libhiai/libhiai_ir/libhiai_ir_build，本机已知 SDK 尚不具备完整包；Kirin9020 离线插件未取得。完整仓库 clone 网络中断，固定提交的关键原始文档已取回，不据此宣称后端可构建。
- 真机执行说明：experiments/DEVICE_RUNBOOK.md；离线阶段在此暂停。
