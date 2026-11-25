# memattn_k01 真机严格 Vulkan 验证

日期：2026-11-24；设备沿用上一阶段 VYG-AL00、HarmonyOS 7.0.0.111、Maleoon 920（Vulkan API原始值4206901、驱动原始值3104880233）。仅测试真实 k01，未运行其余计算图、NPU或18帧流水线。

## 结果与验收口径

真实 k01 已在显式 Vulkan 路径完成。完整475层映射、612个非输入张量检查；关键21个中间张量及最终输出另与当轮ONNX运行对照，最终输出同时与既有PyTorch golden对照。保持 `abs(actual-reference) <= 1e-4 + 1e-4*abs(reference)`，拒绝NaN/Inf，不放宽门限。

最终数值、独立PC回收复核及每次运行结果见 `logs/k01_device_audit.json`，全张量表 `logs/k01_device_checkpoints.csv`，全层映射 `logs/k01_gpu_mapping.csv`，逐层计时 `logs/k01_operator_times.csv`。这些表区分PC ncnn桥接参考和直接ONNX参考，不能称612个张量均有独立PyTorch/ONNX oracle。

## 原始结构与精度

- curr/curr_pos：FP32 `[4096,1,256]`；memory/memory_pos：FP32 `[4100,1,64]`；输出：FP32 `[1,256,64,64]`。
- 4个记忆注意力block，每个包含单头256维自注意力、交叉注意力、LayerNorm与256→2048→256的ReLU前馈、残差；末尾再归一化和空间重排。
- 自注意力4096个token；交叉记忆为4096个空间token加4个pointer token，key的RoPE保留pointer尾部。初始curr还包含0.1×位置编码，memory key/value投影的输入路径保留原图定义。
- ONNX原图的常量/Shape/Gather/Unsqueeze等静态链在转换中折叠。原始8个ScatterND仅在固定索引已验证的范围内替换为Identity或Slice+Concat；该修复在前期与原ONNX逐位对拍。
- 本阶段继续将32个ExpandDims等价转成Reshape，PC最终输出逐位不变。图结构和原始算子计数见 `logs/k01_structure.json`；改写记录 `logs/k01_rewrites.json`。

## GPU映射及CPU路径审计

| 层类型 | 数量 | 执行方式 |
|---|---:|---|
| Input | 4 | CPU文件读取，显式Vulkan上传；不参与张量算术 |
| Split | 53 | 已在GPU的缓冲区共享引用，不回读数据 |
| Reshape（含改写） | 166 | 固定形状、连续性与元素数检查后仅修改GPU缓冲区元数据 |
| Gemm | 40 | Vulkan前向：Q/K/V、输出和FFN投影 |
| SDPA | 8 | Vulkan融合缩放点积注意力，包含Softmax |
| LayerNorm | 13 | Vulkan前向 |
| BinaryOp/Eltwise | 109/1 | Vulkan前向，包含旋转乘加、位置和残差 |
| ReLU | 4 | Vulkan前向 |
| Permute | 35 | Vulkan前向 |
| Slice/Concat | 20/20 | 各16个RoPE布局操作改用专用FP32 Vulkan搬运内核，其余Vulkan前向 |
| MemoryData | 2 | GPU常量上传/绑定 |

合计：4个输入、219个GPU缓冲区元数据操作、252个数据层Vulkan调用。每层可能包含多个shader以及显式packing转换，不将252宣称为精确dispatch数。

执行器使用Net仅加载参数和权重，不使用Extractor做设备推理；张量前向只调用 `forward(VkMat...)`，不调用 `forward(Mat...)`。不支持的层或布局直接报错退出；没有CPU替代分支。主机执行文件I/O、shader编译、调度、元数据和误差核查，这些均与模型算术分开记录。CPU转移/调度开销不等于CPU算子回退。

所有目标中间数据以FP32 pack1流转；上传和部分原生层即使禁用packing选项仍会生成pack4，因此在GPU上显式转换到pack1。FP16存储、算术与uniform均关闭。

## 数值对拍及误差来源

1. PC ncnn对所有中间张量导出参考，21个关键语义节点的桥接误差另与原ONNX逐点核对，均符合原阈值。PyTorch本轮使用已有同输入golden，没有重新导出新的PyTorch中间张量。
2. 真机逐层读回，不只校验最终输出；完整审计覆盖612个非输入blob，包含多输出Slice、别名Split等。对PC桥接、22个ONNX比较和最终PyTorch golden共635项检查。
3. 最终GPU对ONNX maxabs约3.7670e-5；对PyTorch golden约4.0054e-5、RMSE约1.4534e-6；对PC ncnn约5.5313e-5。最终1048576个元素原门限超限数0，无NaN/Inf。精确值以机器审计文件为准。
4. 剩余微小差异与FP32累积顺序及后端实现差异相容，这是解释性推断，未通过高精度单算子控制实验唯一归因。不能据此免除未测试输入的验收。当前通过范围仅是现有memattn_004输入和固定k01形状，不代表其他记忆长度、FP16或完整视频反馈链已经通过。

全层最大绝对误差约1.38283e-4，仍满足原始绝对/相对混合容差；不能把它误写成“所有误差均小于1e-4”。关键ONNX张量和按类型汇总的计时详表见K01_NUMERIC_TABLES.md。

## 性能和内存：必须区分诊断与部署

完整审计两次分别约83.724秒、90.450秒，其中模型算子逐层 `submit_and_wait` 合计约9.553秒、9.723秒；检查点读回约4.329秒、4.553秒；CPU比较与文件保存约62.235秒、68.282秒。完整数据I/O还包含在总墙钟时间中。上传10,487,808字节，首次约62.718ms，包含CPU文件读取和GPU unpack；模型加载/权重上传/管线建立首次约3.291秒。

逐层CSV将CPU编码与提交等待分开。**等待时间包含队列、GPU执行、同步和可能的运行时开销，不能称为纯kernel耗时**。本版ncnn `NCNN_BENCHMARK=0`，没有取得硬件timestamp；纯GPU timestamp及细粒度传输/同步分离仍待专门profiling。频繁同步和读回有意服务于正确性定位，这组数据不用于宣称加速比或生产FPS。

自定义blob/workspace分配器记录的请求峰值为120,605,696字节（约115.02MiB），不包括权重分配池、staging、驱动、管线和allocator池化余量，不能当成物理GPU内存总占用。模型权重文件27,883,840字节另计。

完整诊断进程首轮VmHWM=1,352,340kB、结束VmRSS=721,176kB；此值包含约1GiB参考资源解包、文件缓存和CPU核查。不是生产推理内存。确切权重/驱动物理占用与统一内存带宽未测到，保留为待验证项。完整参考与GPU输出额外占用数GB磁盘，均是测试证据，不是模型必要输入。

第二轮VmHWM=1,432,216kB、结束VmRSS=1,316,028kB。两轮RSS波动本身也表明不能把这类携带参考资源的诊断进程占用当作模型内存。

## 失败定位与修复历史

- 初始Vulkan预检发现32个ExpandDims无GPU实现，未允许框架自动回退；先作PC等价改写。
- 初次完整GPU结果最终误差约6.74，第一LayerNorm通过、首个自注意力偏差0.519。增加Q/K/V及RoPE检查点，Gemm正确、Q/K布局链路出错，V正确。
- 手写显式执行器遗漏ncnn自动packing的归一化。日志确认上传/层输出可为pack4；所有转换均在GPU完成，并对静态Reshape进行连续性和元素数检查，排除错误元数据解释。
- packing修正后，投影至四维RoPE前输入通过，四维宽度2→两路宽度1的原生Slice路径仍错。用两个最小Vulkan内核实现偶奇拆分及交织，保留完全相同数学定义。结论限定为本设备/固定版本/显式路径，不扩大为芯片或所有ncnn Slice均不支持。
- 第一版自定义内核重复声明workgroup specialization，管线创建失败；按ncnn编译约定去掉重复项。
- 随后虽数值通过，原始stderr出现push-constant反射数量告警；这三次临时运行归档为 provisional-pushconstant-warning，不作为最终稳定性证据。固定形状内核取消push constant，使用明确的524288元素边界；stdout与stderr分开后重新验证，最终stderr只包含设备能力信息。
- hdc目录传输的目标映射未让应用读取到全层清单，检查点数量仍34。改为精确SHA256去重的263个参考资源随HAP分发；仅当实际日志为612检查点才计为完整审计。

## 复现、日志与版本保护

在已具备前期模型/转换产物和固定依赖的本机：

```powershell
./experiments/run_k01_strict.ps1 -Prepare -Build -Install -Run
# 运行约1–2分钟，查看应用内结果；入口仅为k01。
& 'D:/DevEco Studio/sdk/default/openharmony/toolchains/hdc.exe' file recv -b com.example.efficientsam2 /data/storage/el2/base/haps/entry/files/k01-probe.txt <本地日志路径>
```

前置产物若需再生成，依次运行既有 `scatter_repair.py`、`convert_ncnn.ps1`，然后 `prepare_k01_fullrefs.py` 与 `package_k01_fullrefs.py`。前两步只针对k01，不运行其余21图。

`artifacts/k01-device/` 保留每次原始stdout/stderr、系统日志和全部GPU二进制；`artifacts/k01-strict/` 保留PC/ONNX oracle；原始失败日志保存在 `logs/k01-*-hilog.txt`。大文件和原始系统日志不进Git。原模型/输入指纹在 `logs/k01_provenance.json`，参考去重映射在 `logs/k01_oracle_package.json`。

新入口 `MemattnK01Ability` 与原最小 `GpuProbeAbility` 分开；未修改CPU runner、原Index页面或历史结果。基线可回滚至 `cpu-baseline-20261124`；上阶段GPU最小验证提交为716bed3。本轮最终提交见Git记录。

## 尚存风险与审核后计划

当前只证明固定形状k01、一个真实golden输入、FP32及所列后端版本的正确路径。未覆盖随机/极端输入、量化、不同token数量、长视频反馈、并发、温控或生产调度。专用RoPE内核严格限定4096×128复数对与单头，不作为通用算子对外使用。

下一阶段先增加同图输入扰动与边界测试、GPU timestamp和低干扰计时，再在审核后逐个扩展其他计算图。不要将本次正确性通过写成已完成21图部署或已证明端到端加速。本轮完成后暂停，等待用户审核。
