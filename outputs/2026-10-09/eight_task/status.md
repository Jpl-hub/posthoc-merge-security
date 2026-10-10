# 八任务复现状态

10 月 10 日。八个 Task Arithmetic 官方专家及全部 head 已齐；八专家融合在七个完整测试集上完成。**Table 6 尚未对齐，不能报告八任务均值。**

| 任务 | 测试资源 | 原专家正确数 / 总数 | 八专家融合正确数 / 总数 | 融合准确率 | 论文 Table 6 基线 |
|---|---|---:|---:|---:|---:|
| SUN397 | 原第 1 划分，归档哈希与覆盖已核验 | 14,944 / 19,850 | 12,657 / 19,850 | 63.76% | 61.75% |
| Cars | 原 test 标注，镜像图片逐文件核验 | 6,245 / 8,041 | 4,990 / 8,041 | 62.06% | 55.52% |
| RESISC45 | 原 6,300 索引与图片已核验 | 6,055 / 6,300 | 4,536 / 6,300 | 72.00% | 60.24% |
| EuroSAT | 原随机索引缺 | — | — | — | 69.32% |
| SVHN | 官方 test | 25,371 / 26,032 | 19,374 / 26,032 | 74.42% | 83.25% |
| GTSRB | 官方 test 与 GT | 12,470 / 12,630 | 8,220 / 12,630 | 65.08% | 63.20% |
| MNIST | 官方 test | 9,969 / 10,000 | 9,400 / 10,000 | 94.00% | 97.55% |
| DTD | 官方 partition 1 test | 1,493 / 1,880 | 981 / 1,880 | 52.18% | 47.02% |

融合包含全部八个专家；这里只测试七个就绪任务，未减少融合模型数。固定系数 0.2 来自 [AGA 作者代码](https://github.com/krumpguo/secure-merge-attack/blob/fb61429f5e6f3517102d506697a150ac26abbdea/main_task_arithmetic.py#L78)，没有按测试结果扫系数。论文数字仅作对照，来自 [Table 6](https://arxiv.org/html/2606.30360v1)，不记作我们的结果。

[五项专家计数](../../2026-10-10/eight_task/expert_checks_1252/expert_checks.json)、[Cars 专家计数](../../2026-10-10/eight_task/cars_expert_1330/expert_checks.json)、[SUN397 专家计数](../../2026-10-10/eight_task/sun_expert_after_download/expert_checks.json)、[八专家融合的七项计数](../../2026-10-10/eight_task/merged_seven_after_download/merged_resource_checks.json)。[服务器原始预测核验](../../2026-10-10/eight_task/resource_validation.json)包含新增 SUN397 专家、七任务融合与之前结果；其余六项融合正确数与之前完全相同。预测文件仅存服务器。源码 commit 和 SHA256 随每个结果目录保存。

## 对齐问题

作者原始 `eval.eval_single_dataset` 在 DTD、RESISC45 上复算，得到 52.18%、72.00%，与封装计数完全相同。[原入口核验](../../2026-10-10/eight_task/author_eval_1350/validation.json)。

单专家也与论文 Table 8 有差异，例如 DTD 为 79.41% 对 73.99%，RESISC45 为 96.11% 对 91.67%。因此应继续核对作者实际 checkpoint、head 与划分，而不能靠调融合系数填近表格。已取得文件来自 Task Arithmetic 官方 Drive；它们尚未被独立确认为 AGA 作者表格使用的精确文件。AGA 公开 README 没有提供这些实验文件的指纹。当前证据不足以确定差异原因。

新增文献交叉核对：[TADrop v1 Table 1](https://arxiv.org/html/2508.06163v1) 的 ViT-B/32 Task Arithmetic 行，在七个就绪任务上为 SUN397 63.8、Cars 62.1、RESISC45 72.0、SVHN 74.4、GTSRB 65.1、MNIST 94.0、DTD 52.2，全部与本项目结果四舍五入到一位小数一致。它支持当前未保护融合计算的可信度，不证明两个项目用了相同文件，也不解决与 AGA Table 6 的差异。

## 资源来源与剩余阻塞

17 个权重文件通过指纹、有限数值、专家结构与 head 类别数检查。[核验结果](../../2026-10-10/eight_task/expert_checks_1252/checkpoint_validation.json)。[下载清单](../../2026-10-10/eight_task/retry_proxy_1235_4090/checkpoint_manifest.json)与 `configs/checkpoints.json` 固定 Drive ID；新增 SHA256 是首次成功下载后的固定指纹，不是作者另行公布的校验值。

Cars 固定镜像版本，核验 16,193 个 Git blob；devkit 验证原 MD5，原 train/test 标注与镜像内容相同，测试 8,041 张。[成功清单](../../2026-10-10/eight_task/cars_data_retry_1258/data_manifest.json)。[首次 devkit 请求失败](../../2026-10-10/eight_task/cars_data_1250/data_manifest.json)保留。

SUN397 原网站返回 404；固定公开镜像中的命名划分，第 1 划分 train/test 各 19,850、每类各 50、互不重叠。[划分核验](../../2026-10-10/eight_task/sun_partitions_1345/data_manifest.json)。服务器已完整流式读取 39,077,296,924 字节归档，全归档原 MD5、镜像 SHA256 与 39,700 张覆盖全部通过；只保留第 1 划分约 13.56 GiB 图片，未存完整归档，也未下载到本地。[完成清单](../../2026-10-10/eight_task/sun_stream_direct_1403/data_manifest.json)。自动队列完成了专家测试和八专家融合的七任务测试。[执行协议](../../2026-10-10/eight_task/sun_evaluation_queue/run_plan.json)。原分类列表与扁平目录顺序并非相同，标签按作者 ImageFolder 排序；[首次顺序检查失败](../../2026-10-10/eight_task/sun_stream_startup_failure/failure.json)保留。EuroSAT 作者说明为 21,600 / 2,700 / 2,700 随机划分，目前缺原索引，其他论文划分或重新抽样不能默认等价。[Task Arithmetic 作者说明](https://github.com/mlfoundations/task_vectors/issues/1#issuecomment-1419521679)。

此前 Drive 配额、代理、网络、数据准备失败的 JSON 与服务器日志全部保留；没有用其他微调配置专家补表。

```bash
python experiments/eight_task_benchmark.py --verify-checkpoints --check-only --output outputs/checkpoint_check
python experiments/eight_task_benchmark.py --expert-checks DTD GTSRB RESISC45 SVHN MNIST Cars --output outputs/expert_check
python experiments/eight_task_benchmark.py --ready-task-checks DTD GTSRB RESISC45 SVHN MNIST Cars --output outputs/merged_resource_check
python experiments/validate_results.py --eight-task-resources --output outputs/2026-10-10/eight_task
```

默认完整入口仍要求八任务齐备后执行。新增[官方 PaRaMS 与完整作者 AGA 的七任务对照](../../2026-10-10/eight_task/params_seven_protection/summary.md)已经完成；每次只保护与恢复一名专家，八专家融合固定系数 0.2。56 行结果、21 个原始文件哈希及逐图配对计数均通过核验。保护前后独立分类预测逐图相同；AGA 后六任务融合预测完全相同，MNIST 仅 1 张不同，保护组高 0.01 pp。本配置没有抗 AGA 效果。

PaRaMS 使用官方实现的一轮独立置换组 assignment 与既定 Attention 缩放。这是补充基线，PaRaMS 不属于 AGA Table 6 的原防御列表，不能把本次结果标为 Table 6 完整复现。

```bash
JAX_PLATFORMS=cpu PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/eight_task_benchmark.py --protection params --protection-checks MNIST DTD SVHN GTSRB RESISC45 Cars SUN397 --output outputs/params_seven
python experiments/validate_results.py --eight-protection --output outputs/params_seven
python experiments/plot_results.py --eight-protection --output outputs/params_seven
python experiments/build_report.py --eight-protection --output outputs/params_seven
```
