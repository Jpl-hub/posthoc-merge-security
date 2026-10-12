# 冻结数据重算

只重用 `fda0a57` 的 block 5 `c_fc` 矩阵、两份目标 PCA 和四组训练激活统计。没有重新采集特征、生成保护权重、换位置或调参。

`results.json` 保存基座与专家任务更新在同一补空间中的响应 RMS，以及实际 FP32 回填与数学回填的差。独立核验用协方差平方根和均值响应计算，不调用重算脚本的 Gram trace 实现。RMS 不是分类准确率或防御成功指标。

两个 Python 文件是这一次重算的冻结源码，不是新实验框架。输入 `.pt`、原始预测、运行日志只保留在 3090；文件与 checkpoint 的 SHA-256 在结果中。DTD/native 是 DTD 输入经过 MNIST 专家，不是 DTD 独立专家。

复核须在服务器项目根目录使用原有数据和 Python 环境。原运行命令：

```bash
python outputs/complement_update_audit.py
OPENBLAS_NUM_THREADS=4 python outputs/verify_complement_update.py
```

本目录保存了这两个脚本的逐字副本。重算入口拒绝覆盖已有输出；结果、验证和脚本的指纹均已核对。独立效用和融合准确率沿用 `results/middle_completion/` 的完整测试，本次新增模型前向为零。
