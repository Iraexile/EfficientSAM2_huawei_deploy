# 恢复与证据保护

Git 标签 `cpu-baseline-20261124` 指向原始 CPU 工程及小型实验结果基线。使用 `git show cpu-baseline-20261124:entry/src/main/cpp/mslite_runner.cpp` 可查看原件。需要独立恢复时，在新目录克隆本地仓库后检出该标签；避免在当前工作目录执行会覆盖未提交内容的恢复命令。

`.gitignore` 排除了模型、golden、device_vectors、工具包、依赖、构建缓存和签名材料。它们仍在原路径；Git 不能恢复被删除的忽略文件，应另行备份原目录。`experiments/logs/model_inventory.json` 记录 ONNX 的 SHA256，可用于核对模型身份。

根 build-profile.json5 含本机签名配置，不入 Git。无签名模板为 config/build-profile.unsigned.json5；新机器复制为根配置并通过 DevEco 配置自己的签名。local.properties 同样保留本机。

本轮实验为 experiments 下独立程序；未改写已通过历史验证的 CPU 模块。原始报告仍保存在 pc/device_results 和 handoff/device_results。
