# Efficient-SAM2 (SAM2.1-tiny) 推理状态清单 · 拆图依据

> 依据：`sam2/sam2_video_predictor.py`、`sam2/modeling/sam2_base.py` 通读 + 3 帧传播实测探针（`pc/scripts/probe_shapes.py`）。
> 基线配置：`sam2.1_hiera_t.yaml`，`enable_MeP_info=False`，`Mem_Frame_Prune=False`（SMR/SWR 全部关闭）。
> 形状均为实测值（探针在 autocast bf16 下跑，dtype 标注为实测；**ONNX 导出时全 fp32**）。

## 0. 关键配置常量（实测）

| 参数 | 值 | 含义 |
|---|---|---|
| num_maskmem | 7 | memory bank：1 条件帧 + 最近 6 帧 |
| hidden_dim / mem_dim | 256 / 64 | 主干特征维 / 记忆特征维 |
| image_size | 1024 | 输入分辨率（预处理直接 resize 到 1024×1024，不保持长宽比） |
| maskmem_tpos_enc | [7,1,1,64] fp32 | 时序位置编码（模型参数，图内常量） |
| no_mem_embed | [1,1,256] | 首帧（条件帧）直接相加的占位记忆 |
| no_obj_embed_spatial | [1,64] | 物体消失（occlusion）时叠加到 maskmem_features |
| sigmoid_scale/bias_for_mem_enc | 20.0 / -10.0 | mask 进记忆编码器前的 scaled sigmoid |
| binarize_mask_from_pts_for_mem_enc | True | 点/box prompt 帧的记忆用二值 mask |
| directly_add_no_mem_embed | **True** | **条件帧不过 memory attention**：`pix_feat = vision_feat + no_mem_embed` |
| max_cond_frames_in_attn | -1 | 全部条件帧参与 |
| max_obj_ptrs_in_encoder | 16 | obj_ptr token 上限（cond 全部 + 最近 15 非条件帧） |
| memory_temporal_stride_for_eval | 1 | 连续取帧（基线无跳帧） |
| 预处理 | mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225) | `x/255` 后标准化 |
| fill_hole_area | 8 | 后处理去小孔洞，**依赖未编译的 `_C` 扩展，基线跳过**（见 D3） |

## 1. 逐图 I/O 规格

### ① image_encoder（每帧一次）

| | 名称 | 形状 | 备注 |
|---|---|---|---|
| 输入 | image | [1,3,1024,1024] fp32 | 已 resize+normalize |
| 输出 | fpn0 | [1,32,256,256] | 高分辨率特征（decoder 用） |
| 输出 | fpn1 | [1,64,128,128] | 高分辨率特征（decoder 用） |
| 输出 | fpn2 | [1,256,64,64] | 主特征（memory attention / memory encoder 用） |
| 输出 | pos0 | [1,256,256,256] | 位置编码（decoder 用） |
| 输出 | pos1 | [1,256,128,128] | 同上 |
| 输出 | pos2 | [1,256,64,64] | 同上（memory attention 用） |

### ② memory_attention（仅非条件帧；条件帧跳过此图）

| | 名称 | 形状 | 备注 |
|---|---|---|---|
| 输入 | curr | [4096,1,256] | fpn2 flatten(HW)BC |
| 输入 | curr_pos | [4096,1,256] | pos2 flatten |
| 输入 | memory | [N,1,64] | 全部记忆 token 拼接（见 §2） |
| 输入 | memory_pos | [N,1,64] | maskmem_pos_enc flatten + tpos；ptr 段为 tpos 投影 |
| 输入 | num_obj_ptr_tokens | 标量 | 用于内部切片（ptr token 不加 RoPE） |
| 输出 | pix_feat_with_mem | [1,256,64,64] | 条件化特征，送 ④ |

- N = 4096×帧数 + 4×ptr数，**随 bank 充满过程变化 → 静态 shape 风险，见决策 D1**
- tpos 索引规则（fork 与官方一致）：t_pos = num_maskmem − t_rel；`maskmem_tpos_enc[7 − t_pos − 1]`，条件帧 t_pos=0 → 用 index 6，最近帧 t_rel=1 → index 0

### ③ memory_encoder（每帧一次，写 bank）

| | 名称 | 形状 | 备注 |
|---|---|---|---|
| 输入 | pix_feat | [1,256,64,64] | fpn2（**原始主干特征，不过 memory attention**） |
| 输入 | mask_for_mem | [1,1,1024,1024] | `sigmoid(high_res_mask)×20−10`；点/box prompt 帧为二值 mask |
| 输入 | is_obj_appearing | [1,1] | `(object_score_logits>0)`，控制 no_obj_embed_spatial 叠加 |
| 输出 | maskmem_features | [1,64,64,64] | 写入 bank（官方流程存 bf16，见 D2） |
| 输出 | maskmem_pos_enc | [1,64,64,64] | **跨帧恒定，只需算一次**，官方缓存在 `constants` |

### ④ prompt_encoder + mask_decoder

| | 名称 | 形状 | 备注 |
|---|---|---|---|
| 输入 | pix_feat | [1,256,64,64] | ② 的输出（条件帧：fpn2+no_mem_embed） |
| 输入 | fpn0 / fpn1 | [1,32,256,256] / [1,64,128,128] | 高分辨率特征 |
| 输入 | point_coords | [1,P,2] | 1024 坐标系（box 按 2 个角点，P=2；跟踪帧 P=0 走 placeholder） |
| 输入 | point_labels | [1,P] | 2/3=box 角点，1/0=正负点 |
| 输入 | mask_input | [1,1,256,256] | 可选（基线跟踪帧不用） |
| 输入 | has_mask_input / multimask | 标量 | 控制分支与输出头 |
| 输出 | low_res_multimasks | [1,M,256,256] | M=1（条件帧 box，2pt>multimask_max_pt_num=1→单 mask）；M=3（跟踪帧，0pt→multimask） |
| 输出 | high_res_multimasks | [1,M,1024,1024] | 双线性上采样后（确认点在导出时核实） |
| 输出 | ious | [1,M] | 跟踪帧取 argmax 选最佳（host 侧） |
| 输出 | obj_ptr | [1,256] | 跨帧状态，写 bank |
| 输出 | object_score_logits | [1,1] | occlusion 分数，写 bank |

## 2. 跨帧状态（C++ memory bank 必须显式管理的）

| 状态 | 形状 | 生命周期 | 说明 |
|---|---|---|---|
| maskmem_features | [1,64,64,64]×7 帧 | FIFO，容量 7 | 1 条件帧 + 最近 6 帧；官方存 bf16（0.5MB/帧） |
| maskmem_pos_enc | [1,64,64,64] | 全局一份 | 常量 |
| obj_ptr + frame_idx + is_cond | [1,256]×N | 全部条件帧 + 最近 15 帧 | 每个 ptr 拆成 4×64 token |
| object_score_logits / best_iou | 标量 | 逐帧 | occlusion 与质量监控 |
| pred_masks | [1,1,256,256] | 逐帧 | 显示用；基线不回灌（prev_sam_mask_logits 仅交互修正用） |

**每帧新增 bank 容量账**：maskmem_features 64×64×64 = 262,144 元素 → fp32 1.0MB / fp16 0.5MB（与简报 §7.5 的 ≈1MB/帧 对应 fp32 口径）。

## 3. 待决策项

- **D1 静态 shape**：bank 未充满时（帧 1~6）memory token 数 N 变化。方案：(a) 导出「满 bank」单版本 + C++ 侧对早期帧补占位 token（需验证占位 token 不破坏数值——占位进入 attention 会改变 softmax 归一化，**不可直接补零**，需 attention mask 或走短图）；(b) 导出 7 个 N 变体（onnx 支持同图多 N 不现实，变体太多）；(c) **推荐**：导出带动态 N 轴的图先看 converter_lite 是否吃 dynamic shape，不吃再退化到 (a)+mask。
- **D2 bank dtype**：官方 GPU 流程 bank 存 bf16（`_run_single_frame_inference` line 1890）。端侧 fp32 存会引入 ±1e-3 级数值差（bank 是 64 维特征，误差不致命但要在对拍基线里固定口径）。**建议：PC 侧对拍与端侧统一 fp32 存 bank**，精度差异在对拍误差表中量化记录。
- **D3 fill_holes**：`fill_hole_area=8` 依赖 `_C` CUDA 扩展，PC 基线与端侧都不启用，保持一致。
- **D4 mask prompt 路径**：`use_mask_input_as_output_without_sam=True`（mask prompt 直接输出不进 SAM heads）。基线只支持 box/点 prompt，此路径不进 ④。

## 4. 帧循环伪码（端侧 C++ 侧复现口径）

```
frame 0（条件帧）:  ① → (fpn2 + no_mem_embed) → ④(box) → ③ 写 bank(t_pos=0)
frame i (i≥1):      ① → ②(读 bank: cond + 最近6帧 + ptrs) → ④(无 prompt, multimask) → argmax iou → ③ 写 bank → FIFO 驱逐最老非条件帧
```
