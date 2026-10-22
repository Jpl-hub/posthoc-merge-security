# 三任务：去除任务方向项

准确率为%，融合列按保护任务 / 伙伴排列；TA0.5。

| Protected / partner | Construction | Standalone | TA | AGA target | AGA both |
| --- | --- | --- | --- | --- | --- |
| MNIST / SVHN | Task-directed | 99.69 | 10.98 / 14.32 | 9.80 / 9.81 | 9.80 / 9.83 |
| MNIST / SVHN | Distance-only | 99.69 | 13.99 / 8.78 | 11.27 / 16.32 | 11.08 / 16.30 |
| DTD / MNIST | Task-directed | 79.41 | 2.39 / 10.29 | 2.39 / 10.09 | 2.39 / 10.09 |
| DTD / MNIST | Distance-only | 79.41 | 1.44 / 10.41 | 2.02 / 9.82 | 1.97 / 9.82 |
| SVHN / MNIST | Task-directed | 97.46 | 7.30 / 9.45 | 6.45 / 12.19 | 6.45 / 12.22 |
| SVHN / MNIST | Distance-only | 97.46 | 9.36 / 9.35 | 8.52 / 9.58 | 8.48 / 9.58 |

使用已固定的同一构造和对照顺序，不重新选择参数。
