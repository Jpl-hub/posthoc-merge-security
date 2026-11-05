# 论文实验与结果索引

截至 2026 年 11 月 6 日，**已选主实验和两项代表性恢复均有完整结果，两种防御的结论还未全部成立。** 当前主表使用防御一的权重偏置版和 11 月 2 日冻结的防御二 A/B 版。加入读出补偿的三任务结果是独立对照。公开增益抵消版已经关闭，结果留在失败记录。

## 论文主表与边界

目标任务为 MNIST、SVHN、DTD，主骨干为 CLIP ViT-B/32。两专家 TA 系数 0.5，原作者 AGA 后再融合。防御与匹配无保护、官方 PaRaMS 在同条件下比较。目标和伙伴都列准确率。带位置知识的公开替换与使用额外图片的 ProDistill 分开报告，具体权限见[攻击清单与原论文](../attack_generalization_20261028/attack_inventory.md)。

下表的三项数字都按 MNIST、SVHN、DTD 排列，单位为准确率百分比。

| 问题 | 防御一 | 防御二（当前 A/B） |
|---|---|---|
| 独立使用 | 99.69 / 97.46 / 79.41，完整测试逐图预测保持 | 99.57 / 96.66 / 76.33，比原专家分别低 0.12 / 0.80 / 3.09 个百分点 |
| TA 与原版 AGA | TA 和 AGA 后均为 9.80 / 6.70 / 2.13 | TA 为 25.15 / 2.26 / 2.13，AGA 仅目标后为 2.99 / 1.86 / 2.07 |
| 公开参数恢复 | 换回末层 MLP 的 B 和偏置时保护仍在。直接清理最终 LayerNorm 偏置时，目标与匹配无保护逐图一致 | AGA 后换回末层 MLP 的 B 和偏置，目标回到 99.28 / 91.38 / 64.95 |
| 固定 ProDistill | 9.80 / 6.70 / 2.13。固定流程不优化最终 LayerNorm 和投影 | 99.43 / 96.18 / 66.70，说明此设定下仍能恢复较多目标能力 |
| 输出校正 | AGA 后用 64 张随机图校正，回到 99.30 / 91.25 / 57.07 | 同条件为 10.76 / 6.99 / 5.00。八专家下 DTD 恢复较多 |

A/B 删项与同范数随机扰动支持联合补偿的作用。移除 A 的任务投影约束后，三任务的收益方向不一致。论文可以描述 SVD 的构造作用和完整消融，证据还不足以把保护效果归因于 SVD。

八专家 TA 0.2 已完成三个保护方向、七个完整测试集的对比。这里的“八专家”是参与融合的模型数，不是八个数据集。当前防御二使用[版本表中的 Updated 行](defense2_eight_versions.md)；Earlier 行只作版本对照。EuroSAT 的原专家测试划分仍待核实，因此不写八任务平均数。另列的防御一 ViT-B/16 结果支持第二骨干上的直接融合和 AGA 比较。

## 写作与汇报用表

| 用途 | Markdown / TeX |
|---|---|
| 两专家正文主表 | [独立、目标与伙伴](paper_main.md) · [TeX](paper_main.tex) |
| 输出校正、匹配及硬标签蒸馏的逐项结果 | [目标](attacks.md) · [TeX](attacks.tex) · [伙伴](partners.md) · [TeX](partners.tex) |
| 当前发布在八专家下的 TA、AGA 和输出校正 | [三个目标](eight_calibration_target.md) · [TeX](eight_calibration_target.tex) · [七个测试集](eight_calibration.md) · [TeX](eight_calibration.tex) |
| 防御二旧版与当前版的八专家对照 | [版本表](defense2_eight_versions.md) · [TeX](defense2_eight_versions.tex) |
| 防御二的 A/B 删项和同范数随机对照 | [目标](defense2_components.md) · [TeX](defense2_components.tex) · [伙伴](defense2_component_partners.md) · [TeX](defense2_component_partners.tex) |
| 任务 SVD 投影约束的消融 | [目标](defense2_projection.md) · [TeX](defense2_projection.tex) · [伙伴](defense2_projection_partners.md) · [TeX](defense2_projection_partners.tex) |
| 两种防御在相同设置下的 ProDistill | [结果](feature_distillation.md) · [TeX](feature_distillation.tex) |
| 防御一的公开替换与已知位置恢复 | [B 替换](defense1_B_replacement.md) · [TeX](defense1_B_replacement.tex) · [最终偏置清理](../phase3/bias_weight_only/attacks/projection.md) · [TeX](../phase3/bias_weight_only/attacks/projection.tex) |
| 早期读出补偿对照（已关闭） | [结果](defense2_readout.md) · [TeX](defense2_readout.tex) |
| 防御二居中读出迁移 | [结果](centered_readout.md) · [TeX](centered_readout.tex) |

居中读出迁移的三任务结果已经完成。独立预测与原 A/B 版逐图一致。固定 ProDistill 目标准确率为 MNIST 10.32%、SVHN 15.94%、DTD 2.13%。对同一蒸馏分数做一次固定取反并按 64 张噪声图校正后，分别为 0.01%、56.73%、13.46%。这项结果只作为固定 ProDistill 的补充对照，公开 B 替换仍可恢复，不能替换主 A/B 发布。

早期读出补偿版的结果只作为关闭记录：它的独立预测与 A/B 版一致，但公开 B 替换和分数取反仍能恢复，不能作为当前主方法。原始分数与取反使用同一批样本，详见[逐项核对](readout_sign_recovery.json)。当前三任务读出迁移结果只看[居中读出表](centered_readout.md)。

## 参考论文和代码后的安排

| 参考 | 对本项目的用法 |
|---|---|
| [PaRaMS，ICCV 2025，表 1](https://arxiv.org/html/2503.07661v1#S5) · [官方代码](https://github.com/ISCT-W/PaRaMS) | 正文同一行并列独立准确率、融合后的目标和伙伴结果。无保护、基线及两种防御用本文固定的同条件数据。 |
| [MergeGuard，CVPR 2026，§6–7](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Defending_Unauthorized_Model_Merging_via_Dual-Stage_Weight_Protection_CVPR_2026_paper.html) | 将正常使用、融合和适应恢复分开交代。本文额外做了 ProDistill，单列其图片和训练预算。 |
| [Trap²，ICML 2026，表 5 与附录表 14](https://arxiv.org/html/2601.21898v2#S6.SS4) · [官方全参数评价入口](https://github.com/ml-postech/Trap2/blob/ef65a26590d92fe07955c5621c08bb315484ff29/eval_scripts/8vision_pertask_linearsearch_full.py) | 正文和补充分别回答主要设置与更强恢复。仓库沿用固定配置、冻结发布、评价入口分离的结构。本文的两专家 ProDistill 使用每任务 64 张训练图，结果按目标和伙伴列出。 |

方法图画独立使用时如何补偿，以及融合后补偿为何失配。准确率用表报告。系数曲线用同一冻结发布和同一对照的实测点绘制，曲线上的测试结果用于解释机制。

## 接下来按这个顺序

1. **先把主方法定清楚。** 正文使用原来的 A/B 防御二。另一个版本只在 A/B 之后改最后的 LayerNorm 和投影，三任务的固定 ProDistill 结果为 MNIST 10.32%、SVHN 15.94%、DTD 2.13%；公开 B 替换仍能恢复，因此它放在补充表里。
2. **把两种防御的结论写到各自证据为止。** 防御一保留原版 AGA 和固定 ProDistill 的正结果。如果投稿主张包括知道最终偏置位置的攻击，当前只改核空间偏置的机制可以被投影直接清理，需要解决结构问题。防御二的主对比使用同一冻结发布，读出补偿、取反及投影消融单列。
3. **方法选定后收齐稿件与复现材料。** 同一发布整理独立损失、目标和伙伴、两种 AGA 范围、选定恢复、关键消融、构造成本和复现命令。已有结果按指纹复用。当前 GitHub 同步状态以本地 `outputs/latest_github_sync_receipt.json` 为准。

## 补充与历史证据

- **现版补充**：[TIES/DARE](defense2_mergers.md) · [硬标签蒸馏](defense2_hard_label.md) · [Unmask](unmask.md) · [投影约束下的恢复](defense2_projection_recovery.md) · [ViT-B/16](../backbone_b16/summary.md)。TA、TIES、DARE 属于融合方法，单列各自的恢复设置。
- **版本与组合**：[两专家版本对比](defense2_versions.md) · [八专家版本对比](defense2_eight_versions.md) · [两防御合用](combined_bias_ab/summary.md) · [PaRaMS 组合](params_composition.md)。主文按表注明使用的发布版本。
- **关闭的探索**：[公开增益抵消与取反恢复](../data_free_compensation/readout_public_gain_20261106/result.json) · [PaRaMS 剪枝对照](params_pruning/summary.md) · [互补视图](defense2_native_views.md) · [逐类均值](defense2_native_means.md) · [防御二研究记录](../../notes/phase2_coupling.md)。
- **来源与复现**：[表格输入及指纹](sources.json) · [防御一固定记录](../phase3/bias_weight_only/other_tasks/summary.md) · [防御二冻结发布及成本](../data_free_compensation/finite_classification_result.json) · [工程说明](../../docs/engineering.md)。完整逐图分数、逐类和配对统计留在服务器，轻量记录注明路径和哈希。

从仓库根目录把当前 Markdown 和 TeX 重建到临时目录：

```bash
python -m experiment.report --defense-comparison --output /tmp/posthoc-paper-tables
```
