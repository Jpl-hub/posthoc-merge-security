# 融合伙伴对照

七个保护任务分别与其余六个任务融合，共42个有向组合。全部使用原来的七份 NBP 发布、TA 系数0.5与目标专家 AGA；没有按伙伴重新构造或选参。

本轮新增35个组合：350条完整数据集评价、5600个分数按原128张批上下文重放，最大误差0。另7个组合复用已核验的原固定配对。完整合并条件计数见 [comparison.csv](comparison.csv)；每个组合均保留无保护/保护、直接TA/AGA后TA、目标/伙伴两列，共336行。

下表对每个保护任务的六个伙伴取平均。准确率为百分比；范围来自六个实际伙伴，不是置信区间。

| 保护任务 | 无保护目标 | NBP目标 | NBP目标范围 | 无保护伙伴 | NBP伙伴 |
| --- | ---: | ---: | ---: | ---: | ---: |
| SUN397 | 71.94 | 0.34 | 0.25–0.75 | 88.38 | 4.28 |
| Cars | 72.58 | 0.51 | 0.44–0.56 | 88.34 | 4.27 |
| RESISC45 | 91.19 | 2.44 | 2.44–2.44 | 84.26 | 8.17 |
| SVHN | 93.65 | 6.70 | 6.70–6.70 | 83.50 | 5.31 |
| GTSRB | 94.95 | 1.19 | 1.19–1.19 | 83.70 | 9.37 |
| MNIST | 99.30 | 9.80 | 9.80–9.80 | 82.12 | 2.80 |
| DTD | 70.07 | 2.13 | 2.13–2.13 | 88.52 | 5.86 |

42个组合的目标平均准确率为84.81%→3.30%，伙伴为85.55%→5.72%。这说明影响会同时波及伙伴任务。此表只回答固定发布在不同伙伴下的表现，不覆盖修改最终读出的针对性恢复。

新增组合的完整计数、逐类和配对记录见 [completed/](completed/)。原始逐图分数保留在17897服务器 `outputs/nbp_partner_pairs_20261107/full/`；源码与输入指纹在每组 `execution.json` 中。`evaluation_protocol.json` 为执行前设置，`loader_change.json` 记录第3组完成后将读图并行数从2改到8的同像素核对。

`results.json` 是另一个只读参数分析：42个双专家条件与7个八专家条件、保护/无保护共98份最终读出。它与本目录的完整图像评价分开报告。

## Complete pair matrix and prediction counts

[All42 pairs](pair_matrix.md) report the target accuracy for each individual companion. [Prediction diagnostics](prediction_diagnostics.json) reuse complete prediction counts for the35 newly evaluated pairs, with unprotected controls. Under direct TA,30 of35 protected readouts have positive margin bounds and predict their certified class on every image. The five uncertified readouts still have dominant-class fractions from73.47% to100%; a negative bound is inconclusive. These counts are not inferred from accuracy or plotted as new forward passes.
