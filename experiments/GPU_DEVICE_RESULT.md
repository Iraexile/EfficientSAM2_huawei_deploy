# Mate 80 GPU 第一阶段实测

日期：2026-11-24，成功运行约19:15（Asia/Shanghai）。本轮仅GPU，没有运行NPU、真实memattn或完整模型。

## 结论与身份边界

真机显式 Vulkan 矩阵计算、最小未缩放 Attention、结果读回和PC数值对齐成功。不是仅枚举成功，也不是 ncnn Net 的 CPU 回退成功。

hdc 查询：HUAWEI、VYG-AL00、VYG-AL00 7.0.0.111(SP12C00E111R3P4)，型号与交接文档中的 Mate80一致。SoC专用参数查询失败；“麒麟9020”目前继承交接设备记录，未从本次系统参数独立读出，不把它伪装成新测量值。

实际 Vulkan GPU 名为 **Maleoon 920**，不是此前假设的Mali。设备数1，vendor=6629，device_id=536875008，driver原始值3104880233，api原始值4206901（1.3.309），compute family=0，队列数2。驱动原始值未按厂商私有编码猜测版本。

## 数值与执行证据

- 固定 ncnn HarmonyOS Vulkan 20261020；FP32，禁用FP16存储/算术/统一常量和packing。
- 使用 Vulkan-only Gemm（两次矩阵乘法）及 Softmax，输入上传为VkMat；两次submit_and_wait完成，读回数据有限、形状符合预期。
- 真机矩阵乘法对探针内FP64参考 maxabs=0；最小Attention maxabs=1.03859203e-8。
- 独立PC脚本解析真机日志中的全部32行、512个输出数值，与离线标准npz比较：MatMul maxabs=0；Attention maxabs=1.0244548320770264e-8；各256个元素超限数均0（atol=rtol=1e-4）。PC标准为FP32，探针内参考为FP64，故两项Attention误差略不同。
- APP_PROBE_EXIT=0。显式VkMat路径没有CPU层回退入口；CPU仅生成输入和参考误差。
- 单次记录34.337ms包含传输、冷启动和CPU参考/校验开销，不能当作纯GPU算子延迟或模型性能。

原始读回值及PC标准逐项保存在 logs/gpu_numeric_audit.json；完整探针文本 logs/gpu_public_probe.txt。

## 失败、定位与最小修复

1. 原命令 `./experiments/run_device.ps1 -Stage gpu`：远端退出126，Permission denied，发生在程序启动前；文件755，/data挂载未显示noexec。执行策略的确切机制尚未确定，未关闭系统保护。改为正常签名HAP内独立GpuProbeAbility。
2. HAP加载日志明确缺失libomp.so；将匹配ABI的SDK OpenMP库打包后修复。
3. ncnn直接创建MatMul Vulkan层失败（APP_EXIT=5，MatMul Vulkan unavailable）。改用等价的Gemm Vulkan，参数由固定版本官方gemm.cpp核对。该失败是具体框架层路径，不是芯片矩阵计算失败。
4. Softmax缺少现代参数版本标志，load_param=-1且提示param is too old。补齐参数1=1，与此前pnnx产物一致后成功。
5. Windows资源构建沙箱内DLL加载126；按需授权在沙箱外构建成功。不是手机GPU问题。

这些也是先前最小探针代码的缺陷，不能归因于手机不支持。

## 修改和保护

新增 gpu_bridge.cpp、GpuProbeAbility.ets、GpuProbe.ets、libgpu_probe类型声明；模块/页面/依赖/CMake增加独立GPU入口。既有CPU runner、pipeline、Index页面未修改，历史结果未覆盖。原签名HAP和原探针二进制备份位于 artifacts/pre-device。

experiments/native/vulkan_probe.cpp增加逐行输出、Gemm替换和Softmax参数修复；verify_device_gpu.py独立数值核查；run_python.ps1更新本机已升级的捆绑解释器路径。

## 重现与文件

已安装的测试HAP只启动GPU入口：

```powershell
& 'D:/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe' shell aa start -a GpuProbeAbility -b com.example.efficientsam2
& 'D:/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe' shell hilog -x
./experiments/run_python.ps1 experiments/verify_device_gpu.py
```

完整原始系统日志、构建日志和HAP在 `experiments/artifacts/gpu-20261124/`，最终系统日志为gpu-app-hilog-final.txt。初次shell退出日志在 `experiments/logs/device-20261124-190704-gpu/`。所有这些本地原始记录保留但不加入Git；精简公开探针、数值审计和HAP hash入Git。原始完整系统日志可能包含其他应用信息，未外发。

## 审核后下一步（本轮不执行）

先验证非方阵Gemm、批量/多头、Softmax轴、RoPE等，再对修复后memattn_k01做逐层执行分配和真机golden验证。当前pnnx产物含MatMul，已知直接MatMul Vulkan不可用，必须验证Gemm改写/调度，不能只启用Net Vulkan后宣称完整GPU执行。

再按encoder、samheads、memenc及各记忆长度逐图核对21图的转换、精度、回退与含传输时延；最后运行18帧并评估误差反馈与端到端延迟。等待用户审核后再进入该阶段。
