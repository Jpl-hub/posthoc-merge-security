# PaRaMS完整默认基线的三任务结果

PaRaMS官方完整默认调用：MLP重排及Attention对角缩放，seed0，未裁剪到项目原参数限制；TA系数0.5，原版AGA目标／双方分列。MNIST复用已核完整结果；DTD及SVHN为本次新批，伙伴固定MNIST。独立能力、原R0.5及改判数均保留，不因项目功能阈值筛去完整基线。构造不使用图片，评价数据已有历史暴露；不主张完整原论文表复现。

## 保护任务准确率（%）

| Protected task | Native original | Native protected | TA original | TA protected | AGA target original | AGA target protected | AGA both original | AGA both protected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 99.69 | 99.69 | 99.37 | 8.92 | 99.30 | 99.30 | 99.32 | 99.32 |
| DTD | 79.41 | 79.41 | 69.31 | 2.07 | 68.40 | 68.35 | 68.78 | 68.78 |
| SVHN | 97.46 | 97.46 | 92.50 | 9.09 | 91.56 | 91.56 | 91.77 | 91.77 |

## 伙伴任务准确率（%）

| Protected / partner | TA original | TA protected | AGA target original | AGA target protected | AGA both original | AGA both protected |
|---|---:|---:|---:|---:|---:|---:|
| MNIST / SVHN | 92.50 | 9.15 | 92.78 | 92.79 | 91.77 | 91.77 |
| DTD / MNIST | 99.45 | 8.92 | 99.45 | 99.45 | 99.42 | 99.42 |
| SVHN / MNIST | 99.37 | 8.95 | 99.39 | 99.39 | 99.32 | 99.32 |
