# 执行记录

| 阶段 | 状态 | 实际记录 |
|---|---|---|
| 环境 | 完成 | 3090；现有 torch 2.3.0+cu121；GPU 算子可用 |
| 论文 | 完成 | 阅读本地 28 页 PDF；视觉核对第 7、13、19 页相关公式 |
| 源码 | 完成 | 两个官方仓库固定 commit；原文件未修改 |
| P0/P1/P3 | 完成 | 56 行；3 seeds；2.153 s |
| P2 | 完成 | 22 行；full MNIST/SVHN；546.816 s |
| LayerNorm 候选 | 完成 | 16 行；202.773 s |
| 结果验证 | 已通过基础检查 | 函数等价、投影恒等式、样本数及 accuracy 计数 |

服务器工作目录：`/root/autodl-tmp/merge_security_20261009/`。完整日志位于其 `outputs/2026-10-09/`；本地未下载 checkpoint、数据或依赖缓存。

实际运行入口：

```bash
python experiments/record_environment.py
python experiments/mechanisms.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/clip_benchmark.py --datasets MNIST SVHN --coefficients 0.2 0.5
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/layernorm_posthoc.py --strengths 1 10
python experiments/validate_results.py
python experiments/plot_results.py
python experiments/build_report.py
```

解释器实际为 `/root/miniconda3/bin/python`。网络代理只用于下载，无模型传入本地磁盘。

已处理问题：非交互 shell 未初始化 Python PATH；本地 GitHub 直连失败后使用现有代理；服务器 Drive/HF 直连受限后使用 SSH 代理转发；MNIST 原源返回 404 后由 torchvision 切换 S3；旧 checkpoint 缺少 `src.modeling` 模块路径，用新入口兼容注册解决。初次机制入口因 SciPy 尚未安装退出，安装完成后成功重跑。

保护耗时：1.214 s；作者恢复耗时：4.298 s。参数、feature error 与 accuracy 计量对象不同，报告分别列出。
