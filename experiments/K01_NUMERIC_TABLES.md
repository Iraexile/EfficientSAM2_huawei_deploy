# k01 数值与性能详表

数据来自完整GPU二进制的独立PC复核；阈值始终为 atol=rtol=1e-4。

PC独立核查：634项比较，严格超限比较数0；另有最终PyTorch golden校验。
全层最大绝对误差：0.000138282776；最大容差归一化比值：0.741362666（≤1为通过）。

| ONNX关键节点 | GPU层 | 最大绝对误差 | 超限元素 |
|---|---|---:|---:|
|/ma/layers.0/norm1/LayerNormalization|ln_40|1.1920929e-06|0/1048576|
|/ma/layers.0/self_attn/MatMul_1|sdpa_227|1.17719173e-06|0/1048576|
|/ma/layers.0/norm2/LayerNormalization|ln_41|3.81469727e-06|0/1048576|
|/ma/layers.0/cross_attn_image/MatMul_1|sdpa_228|1.47521496e-05|0/1048576|
|/ma/layers.0/norm3/LayerNormalization|ln_42|3.98159027e-05|0/1048576|
|/ma/layers.1/norm1/LayerNormalization|ln_43|4.38094139e-05|0/1048576|
|/ma/layers.1/self_attn/MatMul_1|sdpa_229|4.38839197e-06|0/1048576|
|/ma/layers.1/norm2/LayerNormalization|ln_44|7.82012939e-05|0/1048576|
|/ma/layers.1/cross_attn_image/MatMul_1|sdpa_230|1.85966492e-05|0/1048576|
|/ma/layers.1/norm3/LayerNormalization|ln_45|4.48226929e-05|0/1048576|
|/ma/layers.2/norm1/LayerNormalization|ln_46|5.62369823e-05|0/1048576|
|/ma/layers.2/self_attn/MatMul_1|sdpa_231|1.82390213e-05|0/1048576|
|/ma/layers.2/norm2/LayerNormalization|ln_47|5.00679016e-05|0/1048576|
|/ma/layers.2/cross_attn_image/MatMul_1|sdpa_232|2.14576721e-06|0/1048576|
|/ma/layers.2/norm3/LayerNormalization|ln_48|5.89191914e-05|0/1048576|
|/ma/layers.3/norm1/LayerNormalization|ln_49|4.48226929e-05|0/1048576|
|/ma/layers.3/self_attn/MatMul_1|sdpa_233|1.22785568e-05|0/1048576|
|/ma/layers.3/norm2/LayerNormalization|ln_50|4.33921814e-05|0/1048576|
|/ma/layers.3/cross_attn_image/MatMul_1|sdpa_234|2.62260437e-06|0/1048576|
|/ma/layers.3/norm3/LayerNormalization|ln_51|3.99500132e-05|0/1048576|
|/ma/norm/LayerNormalization|ln_52|3.76701355e-05|0/1048576|
|最终输出|out0|3.76701355e-05|0/1048576|

## 全层诊断计时（第一轮完整审计）

submit/wait包括排队、执行和同步，不是纯GPU timestamp。形状别名记0表示不提交数据运算，不表示主机管理时间绝对为0。

| 类型 | 层数 | CPU编码合计ms | 提交等待合计ms |
|---|---:|---:|---:|
|SDPA|8|21.591|6898.586|
|Gemm|40|14.538|1816.089|
|BinaryOp|109|10.569|342.151|
|Permute|35|3.540|141.900|
|LayerNorm|13|1.893|118.070|
|ReLU|4|0.213|84.560|
|Slice|20|2.362|74.710|
|Concat|20|1.637|66.103|
|MemoryData|2|0.124|6.870|
|Eltwise|1|0.102|3.855|
|Split|53|0.000|0.000|
|Reshape|166|0.000|0.000|

## 重复运行

| 运行 | 张量检查点 | 编码ms | 提交等待ms | 读回ms | 比较保存ms | 总诊断ms | 结果 |
|---|---:|---:|---:|---:|---:|---:|---:|
|fulltrial1.txt|612|56.567817|9552.894694|4329.364703|62235.092496|83724.287188|0|
|fulltrial2.txt|612|58.504068|9722.640627|4553.274997|68281.575620|90449.643437|0|
|trial1.txt|34|62.114058|9245.556404|999.623591|6239.257343|16997.995000|0|

原始张量/运行日志在experiments/artifacts/k01-device；逐层逐张量完整记录见同名CSV和JSON。
