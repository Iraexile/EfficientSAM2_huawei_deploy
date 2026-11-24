# EfficientSAM2_huawei_deploy
EfficientSAM2模型端侧部署到huawei mate80(Kirin 9020) 的cpu/gpu/npu性能测试。
cpu后端采用官方mindsporelite推理。
gpu和npu尚在推理后端选取的探索中，目前gou后端采用腾讯开源的ncnn vulkan推理框架，npu后端采用MNN + HiAI/CANN推理框架。
