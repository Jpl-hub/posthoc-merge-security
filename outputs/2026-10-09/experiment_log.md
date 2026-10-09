# 执行记录

| 阶段 | 状态 | 实际记录 |
|---|---|---|
| 环境 | 完成 | 3090；现有 torch 2.3.0+cu121；GPU 算子可用 |
| 论文 | 完成 | 阅读本地 28 页 PDF；视觉核对第 7、13、19 页相关公式 |
| 源码 | 完成 | AGA、Task Arithmetic、PaRaMS、RotationSymmetry 固定 commit；原文件未修改 |
| P0/P1/P3 | 完成 | 56 行；3 seeds；2.153 s |
| P2 | 完成 | 34 行；full MNIST/SVHN；636.982 s |
| 官方 PaRaMS | 完成 | 34 行；1341.369 s |
| 配对保函数对齐 | 完成 | 12 行；272.956 s |
| LayerNorm 候选 | 完成 | 16 行；205.034 s |
| 低活跃方向候选 | 完成 | 7 行；76.216 s |
| 不可逆 V 秩编辑 | 完成 | 50 行；991.004 s；四配置均未保护 |
| 完整 FFN 重表示 | 完成的运行才计入 | 40 行；独立校准的功能重构与共同格式对照 |
| DTD / GTSRB 资源检查 | 完成 | 1,880 / 12,630 张全测试；不是八任务结果 |
| checkpoint 镜像 | 已排除 | MNIST 候选 LFS 指纹通过，但 152 个视觉参数与原权重均不相同 |
| 八任务资源 | 部分完成 | 全部 head；6 个专家权重受下载配额限制 |
| 结果验证 | 已通过基础检查 | 函数等价、投影恒等式、样本数及 accuracy 计数 |

服务器工作目录：`/root/autodl-tmp/merge_security_20261009/`。完整日志位于其 `outputs/2026-10-09/`；本地未下载 checkpoint、数据或依赖缓存。

实际运行入口：

```bash
python experiments/record_environment.py
python experiments/mechanisms.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/clip_benchmark.py --datasets MNIST SVHN --coefficients 0.2 0.5 --attack-all
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/layernorm_posthoc.py --strengths 1 10
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/low_activity_posthoc.py
JAX_PLATFORMS=cpu PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/clip_benchmark.py --protection params --attack-all --output outputs/2026-10-09/params
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/coupled_alignment.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/rank_probe.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/polynomial_probe.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/polynomial_probe.py --config configs/polynomial_multilayer.json --output outputs/2026-10-09/polynomial_multilayer
python experiments/prepare_eight_task_data.py --output outputs/2026-10-09/eight_task/data_1810
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/eight_task_benchmark.py --pretrained-checks DTD GTSRB --output outputs/2026-10-09/eight_task/pretrained_1828
python experiments/validate_results.py
python experiments/plot_results.py
python experiments/build_report.py
```

解释器实际为 `/root/miniconda3/bin/python`。网络代理只用于下载，无模型传入本地磁盘。

PaRaMS 使用 CPU JAX、4 核 affinity 和 DataLoader workers=0；其他真实基准使用 workers=2。首次 JAX 编译计入运行耗时，故这些 wall time 不作为跨方法的严格速度比较。

已处理问题：非交互 shell 未初始化 Python PATH；本地 GitHub 直连失败后使用现有代理；服务器 Drive/HF 直连受限后使用 SSH 代理转发；MNIST 原源返回 404 后由 torchvision 切换 S3；旧 checkpoint 缺少 `src.modeling` 模块路径，用新入口兼容注册解决。初次机制入口因 SciPy 尚未安装退出，安装完成后成功重跑。

保护耗时：1.006 s；作者恢复耗时：3.999 s。参数、feature error 与 accuracy 计量对象不同，报告分别列出。

FFN 原型各次 wall time：polynomial_probe: 355.346 s; polynomial_multilayer: 388.550 s。每次包括拟合和所有评价；仅四层扩展分别记录转换与重构耗时，单层未记录的攻击时间不填造。

初次镜像比较假定 checkpoint 是模型对象而报错；实际为 OrderedDict。修正加载并只规范 model. 前缀后完成比较，失败与成功日志均留服务器。后台启动 SSH 通道曾返回 TimeoutError，但作业继续完成；以完整结果及 completed 元数据核验，不按启动响应判定实验失败。
