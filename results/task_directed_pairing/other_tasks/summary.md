# 同一任务方向构造的三任务结果

同一任务SVD有限构造：74矩阵、rank32、F=T−2I及相同贪心规则，目标／公开基座参数决定一次发布；无图片构造或专家训练。MNIST复用本批已核结果；DTD、SVHN新实测，伙伴固定MNIST。DTD沿用历史训练划分，不称新确认；SVHN排除9306个旧训练索引后一次抽1024。完整测试均历史暴露，原R0.5及零改判要求不变。TA0.5、原AGA仅目标／双方分列；三保护任务对应两个无序组合，不能当三个独立伙伴。已知结构、SVD和贪心不单独主张原创，已知家族恢复边界保留。

## 保护任务准确率（%）

| Protected task | Native original | Native protected | TA original | TA protected | AGA target original | AGA target protected | AGA both original | AGA both protected |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MNIST | 99.69 | 99.69 | 99.37 | 10.98 | 99.30 | 9.80 | 99.32 | 9.80 |
| DTD | 79.41 | 79.41 | 69.31 | 2.39 | 68.40 | 2.39 | 68.78 | 2.39 |
| SVHN | 97.46 | 97.46 | 92.50 | 7.30 | 91.56 | 6.45 | 91.77 | 6.45 |

## 伙伴任务准确率（%）

| Protected / partner | TA original | TA protected | AGA target original | AGA target protected | AGA both original | AGA both protected |
|---|---:|---:|---:|---:|---:|---:|
| MNIST / SVHN | 92.50 | 14.32 | 92.78 | 9.81 | 91.77 | 9.83 |
| DTD / MNIST | 99.45 | 10.29 | 99.45 | 10.09 | 99.42 | 10.09 |
| SVHN / MNIST | 99.37 | 9.45 | 99.39 | 12.19 | 99.32 | 12.22 |
