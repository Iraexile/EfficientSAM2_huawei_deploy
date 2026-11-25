# CPU 71 例误差审计

来源：pc/device_results/report.txt（历史真机运行，本次未重跑）。

共 71 例；日志 OK 58；FAIL 13。

源码阈值为 atol=rtol=1e-4，即每元素 |actual-golden| ≤ 1e-4+1e-4×|golden|。
FAIL 表示至少一个元素违反该公式；最大误差的位置未必就是违反公式的位置。

|案例|计算图|最大绝对误差|最大误差输出|
|---|---|---:|---|
|memattn_024|memattn_k06|0.0001452|pix_feat_with_mem|
|memattn_028|memattn_k07|0.000162|pix_feat_with_mem|
|memattn_032|memattn_k08|0.0001423|pix_feat_with_mem|
|memattn_036|memattn_k09|0.0001467|pix_feat_with_mem|
|memattn_040|memattn_k10|0.0001723|pix_feat_with_mem|
|memattn_044|memattn_k11|0.000187|pix_feat_with_mem|
|memattn_048|memattn_k12|0.00021|pix_feat_with_mem|
|memattn_052|memattn_k13|0.0002242|pix_feat_with_mem|
|memattn_056|memattn_k14|0.0002122|pix_feat_with_mem|
|memattn_060|memattn_k15|0.000187|pix_feat_with_mem|
|memattn_064|memattn_steady|0.0001775|pix_feat_with_mem|
|memattn_068|memattn_steady|0.0001945|pix_feat_with_mem|
|samheads_001|samheads_cond|0.0002861|low_res_multimasks|

## 实际影响与证据局限

- 12 个 Memory Attention 案例、1 个条件分割头案例违反原有严格阈值，不能声称 71 例全部严格通过。
- samheads_001 的 low_res_multimasks=2.861e-4、high_res_multimasks=2.561e-4 均超限；ious=5.960e-8、obj_ptr=1.680e-5、object_score_logits=1.907e-6。
- FAIL 输出日志余弦均打印为 1.000000、scale 为 1.0000，只能说明没有观察到明显整体方向/尺度偏离，舍入值不能证明数值完全相同或分割完全一致。
- report.txt 不含逐元素预测值、超限数量、超限位置对应 golden、NaN/Inf 检查、mask IoU/边界翻转。device_results 未找到预测 .bin，当前无法重算这些指标。
- 18 帧反馈流水线有单独报告，误差口径不同。历史记录的“无发散”和排序一致不能代替独立分割质量验证。
- 浮点累积差是待验证解释，现有日志不足以排除所有实现差异。需要真机保存预测张量后离线逐元素、mask IoU 和 near-zero logits 翻转率验证。
- 当前比较器非 float32 输出只检查字节数；未显式拒绝 NaN/Inf。下一次复测需加强诊断，保持原阈值。
