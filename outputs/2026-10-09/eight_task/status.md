# 八任务资源状态

ViT-B/32；任务顺序、TA 系数 0.2 按论文设置。当前尚无八任务准确率。

| 任务 | 官方专家权重 | 官方 classification head | 测试划分 |
|---|---|---|---|
| SUN397 | Drive 配额限制 | 已取得 | 第 1 分割；19,850，图片待准备 |
| Cars | Drive 配额限制 | 已取得 | 官方 test；8,041，待准备 |
| RESISC45 | Drive 配额限制 | 已取得 | 6,300；原索引待核对 |
| EuroSAT | Drive 配额限制 | 已取得 | 2,700；原随机索引待取得 |
| SVHN | 已验证 SHA256 | 已验证 SHA256 | 官方 test；26,032，已评价 |
| GTSRB | Drive 配额限制 | 已取得 | 官方 test；12,630，待准备 |
| MNIST | 已验证 SHA256 | 已验证 SHA256 | 官方 test；10,000，已评价 |
| DTD | Drive 配额限制 | 已取得 | 第 1 分割；1,880，待准备 |

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
