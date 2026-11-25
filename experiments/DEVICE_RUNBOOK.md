# 真机接入与离线交付

日期：2026-11-24。**当前全部手机 GPU/NPU 新实验均为“等待真机”**。已编译不等于已运行，不把 PC 数值通过等同于手机部署成功。

## 现在何时需要连接手机

离线代码、最小模型、参考验证和 ARM64 构建已准备好。下一阶段开始时连接 Mate 80 即可。手机启用开发者模式和 USB 调试（具体入口以当前系统界面为准），使用可传输数据的 USB 线，解锁并在手机上允许本机调试连接。无需 root 或管理员权限。

先在项目根目录运行：

```powershell
& 'D:/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe' list targets
./experiments/run_device.ps1 -Stage gpu
```

有多个设备时加 `-Target <序列号>`。测试脚本未在无设备状态下执行；本轮仅做 PowerShell 语法检查。手机端 shell 执行权限、目录传输和系统动态库加载仍需第一次连接核实；如被限制，保存原始报错后再将同一探针桥接进 HAP。

## 第一阶段 GPU

脚本将三个 ARM64 程序及匹配的 libc++_shared.so/libomp.so 推送到独立目录 `/data/local/tmp/sam2_probe_20261124`，不会覆盖应用数据或原始 CPU 模型。无需先安装新应用。

`vulkan_probe` 预期日志（不是本次实测值）：

```text
stage=enumerate gpu_count=<非零>
device=<真机设备> vendor=... device_id=... driver=... api=... compute_family=... queues=...
stage=completed explicit_vkmat_dispatch=1 submissions=2 matmul_maxabs=... attention_maxabs=... cold_with_transfers_ms=...
exit_code=0 scope=minimal_unscaled_attention_not_SAM2
```

程序使用 Vulkan-only 层与 VkMat、显式提交并读回；任一 Vulkan 层不可用就失败，不尝试 CPU 回退。数值参考是确定性的 FP64 CPU 公式。该最小 Attention 未包含 SAM2 的投影、RoPE、残差或记忆库。

先检查 `REMOTE_EXIT=0`、设备确为 Mate 80 GPU、数值有限且两项 maxabs ≤1e-4，再进入真实子图：

```powershell
./experiments/run_device.ps1 -Stage memattn
```

该阶段加载修复后 memattn_k01，分别运行 CPU 与 Vulkan 配置，使用现有 memattn_004 golden 输入，保存输出并按原容差比较。**Net 路径允许框架分区/回退，所以日志明确标为 MIXED_OR_GPU_UNPROVEN**；即使通过，也必须继续检查各层调度/Profiler，才可宣称 Memory Attention 完整 GPU 执行。未准备其余 20 图迁移。

## 第二阶段 NNRT

```powershell
./experiments/run_device.ps1 -Stage nnrt
```

同一个最小 MatMul/Attention .ms 模型比较 cpu、nnrt、accelerator 三配置，包含 golden 输入、Predict、输出保存和有限性/超限元素统计。已有 ms_opprobe 中的单算子模型仅做三配置 Build 对照，用于复核旧报告。

预期字段：`stage=build mode=... status=...`；构图成功且提供用例时再有 `stage=predict`、`output=... saved=...`、`compare=... maxabs=... violations=... nonfinite=...`。`hardware_execution=UNPROVEN` 保持到有真实 NPU 执行追踪证据。负面结果也是需保留的实验结果。

脚本保存 hilog、远端退出码、系统信息和预测张量到 `experiments/logs/device-时间-阶段/`。hilog 可能包含 `<private>`；不更改系统隐私设置、不用枚举或速度替代执行证明。

## 离线已执行命令及结果

| 命令 | 实际结果 | 原始记录 |
|---|---|---|
| `./experiments/run_python.ps1 experiments/audit_cpu.py` | 71例，58 OK/13 FAIL；maxabs=2.861e-4 | CPU_AUDIT.md、logs/cpu_audit.json |
| `./experiments/run_python.ps1 experiments/offline_models.py` | 最小 ORT 模型对照 FP64：MatMul 0，Attention 1.2549e-8 | logs/offline-models.txt |
| `./experiments/run_python.ps1 experiments/scatter_repair.py` | 8个 ScatterND 局部修复；两组输入原图/改写图逐位一致 | logs/scatter-repair.txt |
| `./experiments/convert_ncnn.ps1` | 原 memattn 残留 ScatterND；修复后已消除并运行验证 | logs/pnnx-*.txt |
| `./experiments/run_python.ps1 experiments/check_ncnn.py` | 最小 Attention maxabs=1.1176e-8；memattn maxabs=2.9564e-5，0/1048576 超限 | logs/ncnn-pc-correctness.txt |
| `./experiments/build_native.ps1` | 3个 ARM64 可执行程序构建成功 | logs/native-build.txt |
| `hvigorw.bat assembleHap --mode module -p product=default --no-daemon` | 原 HAP BUILD SUCCESSFUL，部分任务使用有效构建缓存 | logs/hap-build.txt |

`run_python.ps1` 使用捆绑 CPython3.12 和原虚拟环境的同版本库，未改写损坏的历史解释器路径。ncnn Python wheel 保存在 dependencies/python；本次沙箱无法直接读取由下载进程安装的 wheel 子目录，因此 PC ncnn 对照经按需授权执行。未申请全盘访问。

MSLite 最小 .ms 由现有 WSL 2.11.0 转换器生成，成功日志为 logs/mslite-*-wsl.txt。Windows 转换器本轮无输出退出1，未把该工具异常计为模型失败。

构建产物：`experiments/artifacts/device/`；最小和修复模型：`experiments/artifacts/models/`；原有已签名 HAP：`entry/build/default/outputs/default/entry-default-signed.hap`。上述二进制均不入 Git。

## 依赖与未完成事项

ncnn 官方 HarmonyOS Vulkan / Windows 包和 pnnx 固定为 20261020，下载 SHA256 在 logs/dependency-sha256.json。官方来源：https://github.com/Tencent/ncnn/releases/tag/20261020 与 https://github.com/pnnx/pnnx/releases/tag/20261020 。Hash 是本机下载身份记录，不是独立发布方签名验证。

Mobiinfer hiai_npu 固定提交与缺失 DDK/9020插件见 NNRT_DIAGNOSIS.md。完整 clone 网络中断，已保存固定提交的关键原文到 evidence/mobi；未完成 MNN HiAI 构建、手机 NPU 执行或离线 OMC。不能把面向 Kirin9030 的发布模型直接当作 9020 适配证据。

后续顺序：连接手机 → GPU 最小 Compute → NNRT 三配置 → memattn 真机数值与调度证据 → 稳定计时 → 完整18帧。当前在离线准备结束处暂停。
