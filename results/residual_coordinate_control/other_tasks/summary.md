# 固定全网规则的三任务结果

同一固定全网换序规则；TA系数0.5，未修改原版AGA，目标专家与双方作用范围分别报告。构造不使用图片，无优化或按测试选参。三个保护任务对应两种无序任务组合；MNIST/SVHN交换保护方后的无攻击模型在实数下为全网坐标等价，不能当三个独立伙伴。已知坐标对称结构，不主张新算法原创性或抗所有恢复攻击；原MNIST针对性公开匹配恢复的边界保留。原参数D0.5对照不变，本批D20事前单列，任务R0.5保持。

## 保护任务准确率（%）

| Protected task | Native original | Native protected | TA original | TA protected | AGA target original | AGA target protected | AGA both original | AGA both protected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 99.69 | 99.69 | 99.37 | 7.39 | 99.30 | 10.09 | 99.32 | 10.09 |
| DTD | 79.41 | 79.41 | 69.31 | 2.23 | 68.40 | 2.18 | 68.78 | 2.13 |
| SVHN | 97.46 | 97.46 | 92.50 | 7.74 | 91.56 | 9.74 | 91.77 | 9.63 |

## 伙伴任务准确率（%）

| Protected / partner | TA original | TA protected | AGA target original | AGA target protected | AGA both original | AGA both protected |
|---|---:|---:|---:|---:|---:|---:|
| MNIST / SVHN | 92.50 | 7.74 | 92.78 | 6.78 | 91.77 | 6.78 |
| DTD / MNIST | 99.45 | 8.68 | 99.45 | 9.89 | 99.42 | 9.85 |
| SVHN / MNIST | 99.37 | 7.39 | 99.39 | 10.28 | 99.32 | 10.28 |
