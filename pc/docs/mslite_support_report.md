# M2 算子支持度报告：Efficient-SAM2 (SAM2.1-tiny) → MindSpore Lite 2.11.0

日期：2026-11-13 ｜ 工具链：converter_lite 2.11.0（WSL2 Ubuntu，Windows 版 exe 被 Smart App Control 拦截）

## 结论

**21/21 个图全部转换成功并通过 benchmark 运行时验证**（x86 CPU 加载 + 推理，随机输入）。
无算子缺失：全部 45 种 ONNX 算子均有 CPU kernel 实现。需要绕行的不是"缺算子"，
而是 **动态形状推断** 和 **个别 kernel 的广播限制**，均已在导出/后处理阶段解决，
且每一版改动都经过 onnxruntime 对 golden 的数值对拍（≤2.1e-05，容差 1e-4）。

## 各图状态

| 图 | 大小 | 转换 | benchmark | 后处理 |
|---|---|---|---|---|
| image_encoder | 128 MB | ✅ | ✅ | pos embed 预计算为常量（消除 ONNX `If`） |
| memattn_k01…k15, steady | 135–232 MB | ✅ ×16 | ✅ ×16 | 无（含 ScatterND/Where/ConstantOfShape 但形状可静态推断） |
| memenc_binarize / sigmoid | 7 MB | ✅ | ✅ | 无 |
| samheads_cond | 16 MB | ✅ | ✅ | 切尾 + onnxsim + fixup（见下） |
| samheads_track | 16 MB | ✅ | ✅ | onnxsim + fixup |

## 遇到的四类问题与修法

### 1. ONNX `If` 节点（image_encoder）→ converter 直接不支持
- **来源**：Hiera `_get_pos_embed` 里 bicubic interpolate 的 scale 保护分支被导出成 `If`。
- **修法**：输入固定 1024×1024，pos embed 是确定常量 → 导出时预计算并烘焙成常量
  （`export_onnx.py` 的 `_get_pos_embed_cached` patch，数值逐位一致）。

### 2. 动态形状推断失败 → `FindBackendKernel nullptr`（MulFusion）
- **现象**：转换成功，但 benchmark 加载时调度器失败，报 `MulFusion` kernel 找不到。
- **根因**：prompt encoder 里 broadcast/expand 被导出成符号形状计算链
  （`Shape→Where(Equal)→ConstantOfShape`），MSLite 的形状推断无法静态求值，
  下游所有 Mul 形状未知 → kernel 选择失败。`--inputShape` 钉死输入也救不了中间张量。
- **修法**：`onnxsim` 用具体输入做常量折叠，把形状链全部算成常量（track 靠这个直接解决）。

### 3. 个别算子的 converter/kernel 限制（samheads_cond）
- `ScatterND`：`PositionEmbeddingRandom.forward_with_coords` 的原地索引赋值
  （`coords[:,:,0] = …`）导出为 ScatterND，converter 形状推断失败。
  → patch 成广播除法（`export_onnx.py`，逐位等价）。
- `Where`（prompt encoder 掩码版）：NNACL Where kernel 对 `[B,N,1] bool` 条件广播
  运行时报错（ret 137）。→ `_embed_points` patch 成 0/1 掩码算术
  （`where(l==-1,A,B) == B*(1-m)+A*m`，独热标签下逐位等价）。
- `Flatten(axis=2)` 作用于 2D 张量（退化情形）：NNACL 形状推断失败。
  → 后处理换成常量目标 Reshape。
- `Dropout`（eval 恒等）/`Identity`（MSLite 会丢弃导致输出失配）：
  → 后处理脚本 `strip_dropout.py` 统一剥除（Identity 别名输出直接删掉）。
- **no_obj_ptr 组合尾**（`λ·obj_ptr + (1−λ)·no_obj_ptr`）：动态形状污染的核心节点，
  → 从 cond 图切除，obj_ptr 输出原始投影值，组合放到端侧 C++ 胶水层
  （`no_obj_ptr` [1,256] 常量已导出 `outputs/ms/no_obj_ptr.npy`）。
  track 图经 onnxsim 后该尾部可留在图内。

### 4. Windows 版工具被系统策略拦截
- `converter_lite.exe`（未签名）被 Smart App Control + UMCI 拦截，DLL 也无法加载。
- **修法**：WSL2 + Linux 版 2.11.0 工具链（不动系统安全策略）。

## benchmark 耗时参考（x86-64 CPU，4 线程，随机输入，仅供参考）

| 图 | avg |
|---|---|
| image_encoder | 1058 ms |
| memattn_k01 → k15 | 626 → 2587 ms（随记忆帧数增长，k07 后饱和 ~2.4 s） |
| memattn_steady | 2241 ms |
| memenc_binarize / sigmoid | 43 / 45 ms |
| samheads_cond / track | 25 / 24 ms |

稳态单帧（encoder + memattn_steady + memenc + samheads）≈ **3.4 s**（x86 CPU）。
麒麟 9020 CPU 预计更慢——memattn 的 28K token 注意力是最大瓶颈，
M3 需要实测并考虑减帧记忆/降分辨率/算子级优化。

## 数值验证

| 验证 | 结果 |
|---|---|
| 全部 21 图 ONNX vs PyTorch golden（43 组，atol=rtol=1e-4） | 43/43 PASS，最大 5.29e-05 |
| samheads cond_fix/track_fix 变体 vs golden（11 组） | 11/11 PASS，最大 2.10e-05 |

## 端侧胶水层（C++）需要补的逻辑

1. **cond 帧 obj_ptr 组合**：`λ = (object_score_logits > 0)`；
   `obj_ptr = λ·raw + (1−λ)·no_obj_ptr`（256 维，fp32；常量见 `outputs/ms/no_obj_ptr.npy`）。
2. **cond 帧 mask 输出**：`low_res_masks`/`high_res_masks` 直接用
   `low_res_multimasks`/`high_res_multimasks`（M=1 时逐位相同）。
3. **track 帧多 mask 选择**：`argmax(ious)` 取三选一（track 图输出 3 个 multimask + ious）。

## 风险提示（M3 验证点）

- x86 benchmark 全绿 ≠ ARM/麒麟完全等价：NNACL ARM kernel 覆盖面通常 ≥ x86，
  但 Where/ScatterND 这类算子在 ARM 上的广播限制需上机复测。
- memattn_steady 232 MB、attention 序列 N=28,712：端侧 CPU 内存带宽和时延是主要风险，
  M3 profiling 重点关注。
- 45 种算子并集：Add, ArgMax, Cast, Concat, Constant, ConstantOfShape, Conv,
  ConvTranspose, Cos, Div, Equal, Erf, Expand, Flatten, Gather, Gemm, Greater,
  GreaterOrEqual, LayerNormalization, MatMul, MaxPool, Mod, Mul, Pad, Pow, Range,
  ReduceMean, ReduceSum, Relu, Reshape, Resize, ScatterND, Shape, Sigmoid, Sin,
  Slice, Softmax, Split, Sqrt, Squeeze, Sub, Tile, Transpose, Unsqueeze, Where。

## 复现流水线

```bash
# PC 侧（.venv）
python scripts/export_onnx.py            # 21 个 ONNX（含全部 patch）
python scripts/prepare_mslite.py         # samheads cond/track 的后处理
# WSL 侧
bash scripts/convert_all_wsl.sh          # 21 × converter_lite → outputs/ms/*.ms
bash scripts/bench_all_wsl.sh            # 21 × benchmark → outputs/ms/benchmark_log.txt
```
