# 八任务复现状态

10 月 10 日。八个 Task Arithmetic 官方专家及全部 head 已齐；八专家融合在六个完整测试集上完成。**Table 6 尚未对齐，不能报告八任务均值。**

| 任务 | 测试资源 | 原专家正确数 / 总数 | 八专家融合正确数 / 总数 | 融合准确率 | 论文 Table 6 基线 |
|---|---|---:|---:|---:|---:|
| SUN397 | 第 1 划分已核验，图片缺 | — | — | — | 61.75% |
| Cars | 原 test 标注，镜像图片逐文件核验 | 6,245 / 8,041 | 4,990 / 8,041 | 62.06% | 55.52% |
| RESISC45 | 原 6,300 索引与图片已核验 | 6,055 / 6,300 | 4,536 / 6,300 | 72.00% | 60.24% |
| EuroSAT | 原随机索引缺 | — | — | — | 69.32% |
| SVHN | 官方 test | 25,371 / 26,032 | 19,374 / 26,032 | 74.42% | 83.25% |
| GTSRB | 官方 test 与 GT | 12,470 / 12,630 | 8,220 / 12,630 | 65.08% | 63.20% |
| MNIST | 官方 test | 9,969 / 10,000 | 9,400 / 10,000 | 94.00% | 97.55% |
| DTD | 官方 partition 1 test | 1,493 / 1,880 | 981 / 1,880 | 52.18% | 47.02% |

融合包含全部八个专家；这里只测试六个就绪任务，未减少融合模型数。固定系数 0.2 来自 [AGA 作者代码](https://github.com/krumpguo/secure-merge-attack/blob/fb61429f5e6f3517102d506697a150ac26abbdea/main_task_arithmetic.py#L78)，没有按测试结果扫系数。论文数字仅作对照，来自 [Table 6](https://arxiv.org/html/2606.30360v1)，不记作我们的结果。

[五项专家计数](../../2026-10-10/eight_task/expert_checks_1252/expert_checks.json)、[Cars 专家计数](../../2026-10-10/eight_task/cars_expert_1330/expert_checks.json)、[八专家融合的六项计数](../../2026-10-10/eight_task/merged_ready_1330/merged_resource_checks.json)。[服务器原始预测核验](../../2026-10-10/eight_task/resource_validation.json)覆盖全部 12 条记录；预测文件仅存服务器。源码 commit 和 SHA256 随每个结果目录保存。

## 对齐问题

作者原始 `eval.eval_single_dataset` 在 DTD、RESISC45 上复算，得到 52.18%、72.00%，与封装计数完全相同。[原入口核验](../../2026-10-10/eight_task/author_eval_1350/validation.json)。

单专家也与论文 Table 8 有差异，例如 DTD 为 79.41% 对 73.99%，RESISC45 为 96.11% 对 91.67%。因此应继续核对作者实际 checkpoint、head 与划分，而不能靠调融合系数填近表格。已取得文件来自 Task Arithmetic 官方 Drive；它们尚未被独立确认为 AGA 作者表格使用的精确文件。AGA 公开 README 没有提供这些实验文件的指纹。当前证据不足以确定差异原因。

## 资源来源与剩余阻塞

17 个权重文件通过指纹、有限数值、专家结构与 head 类别数检查。[核验结果](../../2026-10-10/eight_task/expert_checks_1252/checkpoint_validation.json)。[下载清单](../../2026-10-10/eight_task/retry_proxy_1235_4090/checkpoint_manifest.json)与 `configs/checkpoints.json` 固定 Drive ID；新增 SHA256 是首次成功下载后的固定指纹，不是作者另行公布的校验值。

Cars 固定镜像版本，核验 16,193 个 Git blob；devkit 验证原 MD5，原 train/test 标注与镜像内容相同，测试 8,041 张。[成功清单](../../2026-10-10/eight_task/cars_data_retry_1258/data_manifest.json)。[首次 devkit 请求失败](../../2026-10-10/eight_task/cars_data_1250/data_manifest.json)保留。

SUN397 原网站返回 404；固定公开镜像中的命名划分，第 1 划分 train/test 各 19,850、每类各 50、互不重叠。[划分核验](../../2026-10-10/eight_task/sun_partitions_1345/data_manifest.json)。不使用丢失原文件名的 parquet 替代。正在流式读取约 39 GB 原归档，只保留第 1 划分图片；需要全归档原 MD5、镜像 SHA256、39,700 张覆盖均通过才标记就绪。[启动协议](../../2026-10-10/eight_task/sun_stream_1357/launch.json)。原分类列表与扁平目录顺序并非相同，标签按作者 ImageFolder 排序；[首次顺序检查失败](../../2026-10-10/eight_task/sun_stream_startup_failure/failure.json)保留。EuroSAT 作者说明为 21,600 / 2,700 / 2,700 随机划分，目前缺原索引，其他论文划分或重新抽样不能默认等价。[Task Arithmetic 作者说明](https://github.com/mlfoundations/task_vectors/issues/1#issuecomment-1419521679)。

此前 Drive 配额、代理、网络、数据准备失败的 JSON 与服务器日志全部保留；没有用其他微调配置专家补表。

```bash
python experiments/eight_task_benchmark.py --verify-checkpoints --check-only --output outputs/checkpoint_check
python experiments/eight_task_benchmark.py --expert-checks DTD GTSRB RESISC45 SVHN MNIST Cars --output outputs/expert_check
python experiments/eight_task_benchmark.py --ready-task-checks DTD GTSRB RESISC45 SVHN MNIST Cars --output outputs/merged_resource_check
python experiments/validate_results.py --eight-task-resources --output outputs/2026-10-10/eight_task
```

默认完整入口仍要求八任务齐备后执行。保护和 AGA 的八任务比较尚未开始；可选基变换对照不等于 Table 6 原防御配置。
