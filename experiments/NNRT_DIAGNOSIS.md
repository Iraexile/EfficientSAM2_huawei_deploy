# NNRT 失败诊断与 Mobiinfer 依赖核查

日期：2026-11-24。本轮无真机，以下区分源码/历史数据与待验证解释。

## 已确认事实

- `ModelGuard::Load` 创建通用 `OH_AI_DEVICETYPE_NNRT`，然后添加 FP32 CPU。已有异构上下文，“未添加 CPU”已被源码否定。
- 历史 `report_npu.txt` 中 image_encoder、memattn、samheads 在 `OH_AI_ModelBuildFromFile` 返回 -1。失败阶段是构图，未进入这些案例的预测，不能归咎于结果校验。
- `probe_npu.txt` 中 op_ln、op_softmax、op_transpose、op_split 在 plain/fp16 配置均 BUILD-FAIL；MatMul2D/4D、Reshape 为 BUILD-OK。探针只测 Build，不能把后者视为 NPU 计算成功。
- `gen_opprobe.py` 中 op_ln 输入 FP32 [1,256,256,96]，axis=-1；Softmax FP32 [1,64,256,256]，axis=-1；Transpose FP32 [1,256,256,96]，perm=[0,3,1,2]；Split FP32 [1,256,256,288]，axis=-1，三输出。
- memattn_k01 用例 curr/curr_pos 为 FP32 [4096,1,256]，memory/memory_pos 为 FP32 [4100,1,64]，输出 [1,256,64,64]。全部 ONNX 输入输出、opset 和算子计数见 logs/model_inventory.json。
- `hilog_nnrt.txt` 的 MSLite 日志含 `<private>`；保存内容没有提供可唯一定位到失败节点的完整诊断。不能凭 -1 确定算子白名单、Shape/dtype、图分区失败或驱动限制中的唯一根因。
- PC 转换工具 2.11.0 与手机系统 NDK 运行时不是同一份库；系统运行时确切版本仍需真机采集，不能直接称为设备上 2.11.0。

## 最小修复/复测

独立 `native/nnrt_probe.cpp` 保持原工程不变，比较 cpu、通用 nnrt、`OH_AI_CreateNNRTDeviceInfoByType(OH_AI_NNRTDEVICE_ACCELERATOR)` 三种配置。类型选择是待验证变量，不宣称已修好。

先对已有 op_reshape/op_matmul2d/op_softmax 的完全相同模型做三配置对照，再缩小 shape、单一 dtype/axis 实验。失败必须保存原模型 hash 和完整 hilog，不沿用“某算子全部不支持”的无边界结论。

可选 case_dir 参数支持真实 golden 输入、输出二进制保存、原 atol/rtol 超限元素计数及 NaN/Inf 拒绝。硬件执行始终标为 UNPROVEN，直到取得设备分区或执行追踪证据。即使 accelerator 配置成功，也不自动判定 NPU 成功。

## Mobiinfer hiai_npu

来源：https://github.com/IPADS-SAI/Mobiinfer-HarmonyOS/tree/hiai_npu

本轮通过 GitHub 文件树固定提交 `da8d4d95baa132188d917028130d7d39af2b7ac0`，取得 README、HiAI CMake 和构建/算子说明的原始文件，保存在 dependencies/mobi-audit。完整 clone 的进度单独记录，不能将部分仓库视为可构建。

- CMake 的 OHOS 分支链接 `libhiai.so`、`libhiai_ir.so`、`libhiai_ir_build.so`，并使用 3rdParty/include 中的厂商头文件。
- sdk_config.sh 默认 CANN-Kit-next-6.0.1.0；离线编译另需 DDK-tools-next-6.1.1.0 和芯片插件。默认插件为 Kirin9030，不能用于证明 Kirin9020 兼容。
- 本机 SDK API26 已有 CANNKit 系统头文件、libhiai_foundation.so、neural_network_runtime 接口。这些不等于 MNN 后端所需独立 CANN Kit DDK 包。
- 对已知 SDK 目录检查未找到 libhiai_ir_build.so；当前工程也无该 DDK 包。因此暂不能完成 MNN HiAI 链接及离线芯片编译。不申请全盘搜索，不安装未知来源 SDK。
- 后续优先在线最小 MatMul；如需独立 CANN Kit，获取官方对应版本并先验证 ABI/API26 适配。离线方案还需明确 Kirin9020 的官方插件，禁止直接替换使用 9030 成品 OMC。

## 下一轮必须补齐

真机系统/驱动/运行时版本、相同模型 CPU/NNRT 对照、构图原始诊断、逐输出误差和实际调度证据。当前证据尚不能把失败扩大到整个 NPU 或所有 NNRT 组合。
