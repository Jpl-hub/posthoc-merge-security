# 八任务资源状态

ViT-B/32；任务顺序、TA 系数 0.2 按论文设置。当前尚无八任务准确率。

| 任务 | 官方专家权重 | 官方 classification head | 测试划分 |
|---|---|---|---|
| SUN397 | Drive 配额限制 | 已取得 | 第 1 分割；19,850，图片待准备 |
| Cars | Drive 配额限制 | 已取得 | 官方 test；8,041，待准备 |
| RESISC45 | Drive 配额限制 | 已取得 | 6,300；图片与原索引已核验，作者入口已全量推理 |
| EuroSAT | Drive 配额限制 | 已取得 | 2,700；原随机索引待取得 |
| SVHN | 已验证 SHA256 | 已验证 SHA256 | 官方 test；26,032，已评价 |
| GTSRB | Drive 配额限制 | 已取得 | 官方 test；12,630，原作者入口已全量推理 |
| MNIST | 已验证 SHA256 | 已验证 SHA256 | 官方 test；10,000，已评价 |
| DTD | Drive 配额限制 | 已取得 | 第 1 分割；1,880，原作者入口已全量推理 |

官方 pretrained 权重已取得。来源、Drive ID、已有文件指纹在 `configs/checkpoints.json`。完整下载失败日志留在服务器；`preflight.json` 列出六个缺失专家。未用近似配置模型填补缺项。

Task Arithmetic 作者确认 DTD、SUN397 使用第 1 分割，EuroSAT 为 21,600/2,700/2,700 随机划分，但缺少可直接复用的原始索引。[作者说明](https://github.com/mlfoundations/task_vectors/issues/1#issuecomment-1419521679)

公开 [tanganke MNIST model card](https://huggingface.co/tanganke/clip-vit-base-patch32_mnist) 标明另一套微调配置，不能默认视为官方 Drive 权重的等价镜像。

```bash
python experiments/download_checkpoints.py --all-tasks --output outputs/2026-10-09/eight_task
python experiments/eight_task_benchmark.py --check-only
# 所有权重及数据齐备后，使用原 TaskVector.apply_to 和 dataset registry：
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/eight_task_benchmark.py
```

当前仅验证了资源检查入口；完整八任务执行尚未验证。默认先跑未保护基线，禁止缺少专家时悄悄缩减任务数。可选基变换保护仍是独立对照，不属于 Table 6 的原配置防御。

## 本轮实际推进

官方六个缺失 checkpoint 再次返回配额限制，记录见 [下载清单](retry_1805/checkpoint_manifest.json)。沿 TSV 作者引用的 TALL 镜像取得 MNIST 候选，LFS SHA256 验证通过；但规范 model. 前缀后，与原权重 158 个字段全部不一致，其中 152 个是视觉字段，因此拒绝作为原权重替代。[逐参数比较](../continuation/mirror_check.json)。没有继续下载其他候选权重。

DTD 使用 torchvision 官方 partition=1，train/val/test 各 1,880；在服务器建立作者 ImageFolder 入口。GTSRB 使用原作者 PyTorchGTSRB 和官方 GT。测试索引 SHA256 与原源码指纹见 [数据清单](data_1810/data_manifest.json)。

pretrained + 官方 head 的完整测试：DTD 835/1,880（44.41%），GTSRB 4,112/12,630（32.56%）。这些是输入、标签和 head 推理检查，不能单独证明与原训练划分完全一致，也不是八任务融合结果。[原始计数](pretrained_1828/pretrained_checks.json)。

```bash
python experiments/prepare_eight_task_data.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/eight_task_benchmark.py --pretrained-checks DTD GTSRB --output outputs/2026-10-09/eight_task/pretrained_1828
```

19:23 再次重试官方 checkpoint：已有 11 个文件通过检查，六个缺失专家仍受配额限制。[本次清单](retry_1923/checkpoint_manifest.json)。

RESISC45 三份索引已取得，train/val/test 为 18,900/6,300/6,300，覆盖 45 类且互不重叠。原 Google 地址返回 403，保留了[失败记录](splits_1926/data_manifest.json)；随后使用 [TorchGeo 官方代码引用的公开镜像](https://github.com/torchgeo/torchgeo/blob/main/torchgeo/datasets/resisc45.py)，固定镜像版本并按 AGA 原源码的 MD5 校验，三份均一致。[核验清单](splits_1927/data_manifest.json)。这里只补齐划分索引，图片尚未准备，不能算数据集就绪。索引仅保存在服务器。

```bash
python experiments/prepare_eight_task_data.py --tasks RESISC45 --output outputs/2026-10-09/eight_task/splits_1927
```

21:04 开始的图片准备已完成：31,500 张图片全部覆盖原 18,900 / 6,300 / 6,300 索引，并建立原作者的划分和分类目录。使用 TorchGeo 官方代码引用的公开镜像，固定 Hugging Face 版本 `883edc0eee77b2c84225472f10f126e3ed83fa6e`；427,389,445 字节压缩包 SHA256 为 `beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53`。图片与压缩包仅保存在服务器。[本次数据清单](data_2104/data_manifest.json)。

21:17 作者 RESISC45 入口 pretrained + 官方 head 完整测试：3,794 / 6,300（60.22%）。这是资源与推理检查，不是官方 RESISC45 专家准确率或八任务融合结果。[原始计数](pretrained_2117/pretrained_checks.json)。

```bash
python experiments/prepare_eight_task_data.py --tasks RESISC45 --resisc-images --output outputs/2026-10-09/eight_task/data_2104
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/eight_task_benchmark.py --pretrained-checks RESISC45 --output outputs/2026-10-09/eight_task/pretrained_2117
```

21:46 再次请求六个官方专家，仍全部受 Drive 下载限制；已有 11 个文件完成指纹检查。[本次清单](retry_2146/checkpoint_manifest.json)。保留原始失败原因，不使用其他微调配置替代。

22:50 重试仍缺同样六个官方专家，已有 11 个文件通过指纹检查。[本次清单](retry_2250/checkpoint_manifest.json)。本轮核对了[官方 README](https://github.com/mlfoundations/task_vectors#checkpoints) 的 Drive 入口；未找到可确认等价的官方替代来源，没有下载不同配置的专家填表。

后续顺序：补齐原专家权重与数据，先运行八任务未保护基线，再按论文配置运行保护与 AGA，对照 Table 6。新机制继续在双任务环境筛选，两条工作分别保留结果。

10 月 10 日 00:06 再次重试：已有 11 个文件通过指纹检查，Cars、DTD、EuroSAT、GTSRB、RESISC45、SUN397 六个官方专家仍受 Drive 限制。[本次清单](../../2026-10-10/eight_task/retry_0006/checkpoint_manifest.json)。

10 月 10 日 01:01 重试结果相同：六个缺失专家返回下载配额限制，11 个已有文件通过指纹检查。没有替换权重或缩减任务设置。[本次清单](../../2026-10-10/eight_task/retry_0101/checkpoint_manifest.json)。
