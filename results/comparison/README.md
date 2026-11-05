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

八专家 TA 0.2 已完成三个保护方向、七个完整测试集的对比。EuroSAT 的原专家测试划分仍待核实，现表按任务列结果。另列的防御一 ViT-B/16 结果支持第二骨干上的直接融合和 AGA 比较。

## 写作与汇报用表

| 用途 | Markdown / TeX |
|---|---|
| 两专家正文主表 | [独立、目标与伙伴](paper_main.md) · [TeX](paper_main.tex) |
| 输出校正、匹配及硬标签蒸馏的逐项结果 | [目标](attacks.md) · [TeX](attacks.tex) · [伙伴](partners.md) · [TeX](partners.tex) |
| 当前发布在八专家下的 TA、AGA 和输出校正 | [三个目标](eight_calibration_target.md) · [TeX](eight_calibration_target.tex) · [七个测试集](eight_calibration.md) · [TeX](eight_calibration.tex) |
| 防御二的 A/B 删项和同范数随机对照 | [目标](defense2_components.md) · [TeX](defense2_components.tex) · [伙伴](defense2_component_partners.md) · [TeX](defense2_component_partners.tex) |
| 任务 SVD 投影约束的消融 | [目标](defense2_projection.md) · [TeX](defense2_projection.tex) · [伙伴](defense2_projection_partners.md) · [TeX](defense2_projection_partners.tex) |
| 两种防御在相同设置下的 ProDistill | [结果](feature_distillation.md) · [TeX](feature_distillation.tex) |
| 防御一的公开替换与已知位置恢复 | [B 替换](defense1_B_replacement.md) · [TeX](defense1_B_replacement.tex) · [最终偏置清理](../phase3/bias_weight_only/attacks/projection.md) · [TeX](../phase3/bias_weight_only/attacks/projection.tex) |
| 防御二加入读出补偿、B 替换及取反对照 | [结果](defense2_readout.md) · [TeX](defense2_readout.tex) |

读出补偿版的独立预测与同一 A/B 版一致。AGA 后替换 B 的三个目标原始准确率为 0.07%、0.45%、0.59%，统一取反后为 61.78%、62.45%、11.86%。固定 ProDistill 为 98.63%、83.45%、30.85%。汇报时同时展示这两组恢复结果。原始分数和取反使用同一批样本，详见[逐项核对](readout_sign_recovery.json)。

## 参考论文和代码后的安排

| 参考 | 对本项目的用法 |
|---|---|
| [PaRaMS，ICCV 2025，表 1](https://arxiv.org/html/2503.07661v1#S5) · [官方代码](https://github.com/ISCT-W/PaRaMS) | 正文同一行并列独立准确率、融合后的目标和伙伴结果。无保护、基线及两种防御用本文固定的同条件数据。 |
| [MergeGuard，CVPR 2026，§6–7](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Defending_Unauthorized_Model_Merging_via_Dual-Stage_Weight_Protection_CVPR_2026_paper.html) | 将正常使用、融合和适应恢复分开交代。本文额外做了 ProDistill，单列其图片和训练预算。 |
| [Trap²，ICML 2026，表 5 与附录表 14](https://arxiv.org/html/2601.21898v2#S6.SS4) · [官方全参数评价入口](https://github.com/ml-postech/Trap2/blob/ef65a26590d92fe07955c5621c08bb315484ff29/eval_scripts/8vision_pertask_linearsearch_full.py) | 正文和补充分别回答主要设置与更强恢复。仓库沿用固定配置、冻结发布、评价入口分离的结构。本文的两专家 ProDistill 使用每任务 64 张训练图，结果按目标和伙伴列出。 |

方法图画独立使用时如何补偿，以及融合后补偿为何失配。准确率用表报告。系数曲线用同一冻结发布和同一对照的实测点绘制，曲线上的测试结果用于解释机制。

## 接下来按这个顺序

1. **先解决防御二的主要缺口。** 在现有无真实训练图、冻结专家的设置下，先用 MNIST 检验新补偿是否同时保持独立能力、压低固定 ProDistill 后的准确率，并且在分数统一取反后仍有收益。继续使用匹配无保护、原 A/B 和读出补偿对照。机制与参数先固定，结果出来后再决定是否按同一规则做 SVHN 和 DTD。DTD 的独立损失需要一起报告和改善。
2. **把两种防御的结论写到各自证据为止。** 防御一保留原版 AGA 和固定 ProDistill 的正结果。如果投稿主张包括知道最终偏置位置的攻击，当前只改核空间偏置的机制可以被投影直接清理，需要解决结构问题。防御二的主对比使用同一冻结发布，读出补偿、取反及投影消融单列。
3. **方法选定后收齐稿件与复现材料。** 同一发布整理独立损失、目标和伙伴、两种 AGA 范围、选定恢复、关键消融、构造成本和复现命令。已有结果按指纹复用。当前 GitHub 同步状态以本地 `outputs/latest_github_sync_receipt.json` 为准。

## 补充与历史证据

- **现版补充**：[TIES/DARE](defense2_mergers.md) · [硬标签蒸馏](defense2_hard_label.md) · [Unmask](unmask.md) · [投影约束下的恢复](defense2_projection_recovery.md) · [ViT-B/16](../backbone_b16/summary.md)。TA、TIES、DARE 属于融合方法，单列各自的恢复设置。
- **版本与组合**：[两专家版本对比](defense2_versions.md) · [八专家版本对比](defense2_eight_versions.md) · [两防御合用](combined_bias_ab/summary.md) · [PaRaMS 组合](params_composition.md)。主文按表注明使用的发布版本。
- **关闭的探索**：[公开增益抵消与取反恢复](../data_free_compensation/readout_public_gain_20261106/result.json) · [PaRaMS 剪枝对照](params_pruning/summary.md) · [互补视图](defense2_native_views.md) · [逐类均值](defense2_native_means.md) · [防御二研究记录](../../notes/phase2_coupling.md)。
- **来源与复现**：[表格输入及指纹](sources.json) · [防御一固定记录](../phase3/bias_weight_only/other_tasks/summary.md) · [防御二冻结发布及成本](../data_free_compensation/finite_classification_result.json) · [工程说明](../../docs/engineering.md)。完整逐图分数、逐类和配对统计留在服务器，轻量记录注明路径和哈希。

从仓库根目录重建当前 Markdown 和 TeX：

```bash
python -m experiment.report --defense-comparison --output results/comparison
```
