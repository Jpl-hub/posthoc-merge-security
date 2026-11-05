# 方法、实验与当前结论

截至2026年11月5日，两种防御均有明确有效结果，但还不能称核心方法和全部实验已经完成。主要恢复实验已经跑完，最近的构造调整没有改善抗蒸馏效果；不能把这两件事混为“实验都没跑”。

实验组织参照PaRaMS、Trap²和MergeBarrier，沿用已有表格入口：

- **主实验**：同一发布的独立能力、无保护融合、防御后融合及AGA恢复；保护任务与伙伴任务都给出。PaRaMS表1同时报告两方任务，MergeBarrier表2主要报告保护任务，本文明确采用前者的双任务呈现。
- **代表性恢复**：公开模块替换、固定ProDistill，分别对应MergeBarrier §4.1和Trap² §6.4。不同数据和位置知识分组；不把所有论文的攻击取并集作为必须全胜的要求。
- **消融**：已有A侧、B侧、联合补偿和任务投影对照。按实际结果解释贡献；当前未证明任务投影有一致收益。
- **补充实验**：已有融合方式、八专家、输出校正、硬标签蒸馏、Unmask和适用的第二骨干结果。方法构造成本与恢复成本分别汇总，最终同一发布仍缺完整统一成本表。SFT有先例但未跑，不声称抗带标签微调。

[PaRaMS §5—6](https://arxiv.org/html/2503.07661v1)、[Trap² §6及附录C/D](https://arxiv.org/html/2601.21898v2)、[MergeBarrier §4](https://arxiv.org/html/2511.10712v2)中的任务数、数据权限、融合系数和统计口径不同，本文不直接拼接原论文数字。Trap²官方README将恢复代码指向ProDistill原作者仓库；本项目采用固定原作者核并明确两专家适配。完整来源和取舍见[攻击清单](../attack_generalization_20261028/attack_inventory.md)。

目前主要恢复实验已经完成。防御二在代表性蒸馏和已知B替换下仍有明显恢复。最近一组有限A/B补偿对照中，蒸馏和B替换在合成输入上分别保持64/64和800/800个原专家判断。两组都结束于预设的源检查，没有新增完整测试结果。[对照记录](../data_free_compensation/finite_recovery_compensation.json) DTD去投影组在同一批蒸馏图上改用末层B最小二乘拟合后，完整测试恢复从20.53%升至66.12%。低恢复率依赖固定的100步求解条件。该最小二乘解的48.34%输出矩阵元素超出这100步Adam能达到的外界。[完整测试](../data_free_compensation/dtd_b_fit_transfer.json) · [参数范围复核](../data_free_compensation/dtd_adam_radius.json)

| 项目 | 当前情况 | 直接结论或缺口 |
|---|---|---|
| 防御一：三任务独立、TA、AGA | 完成 | 独立预测保持，融合准确率低；64张随机图输出校正可恢复至接近匹配无保护。 |
| 防御二：当前同一发布 | 完成独立、TA、AGA、校正及匹配 | 独立99.57/96.66/76.33%；AGA加校正10.76/6.99/5.00%。DTD独立损失3.09个百分点。 |
| 其他恢复方法 | 完成MLP匹配、已知模块替换、硬标签蒸馏、ProDistill | 修改位置、额外真实数据和训练分别声明；不能说所有恢复都防住。 |
| 防御基线 | 完成匹配无保护、PaRaMS及防御一比较 | PaRaMS是防御方法，TA/TIES/DARE是融合方法。 |
| 防御二组件贡献 | 当前任务投影对照已完成 | 三任务未证明投影约束的整体收益；九个同版本删项与随机对照已完成。只留A侧缺少保护，只留B侧会损伤独立任务，两侧合用有补偿作用。 |
| 八专家融合 | 三个保护任务，七个完整测试集已完成 | EuroSAT原专家划分未核实；不能写完整八任务平均，也不是保护八种任务。 |
| 第二种骨干 | 防御一ViT-B/16完整评价、原批重放和独立复核完成 | 不代替方法改进；防御二第二骨干暂不展开。 |
| Unmask及其他论文处理 | 当前冻结版Unmask已完成84条评价；GradErase缺少可复现细节，MergeLock官方实现未提供 | Unmask后三个保护任务11.35/12.58/6.12%，无保护同处理结果完全相同，不能据此声称防御优势。 |

三个数字依次为MNIST、SVHN、DTD，单位为准确率百分比。两个防御构造都不使用真实训练图；蒸馏攻击使用额外数据，属于不同条件。防御一在固定ProDistill下目标准确率为9.80/6.70/2.13%，仍有保护；防御二为99.43/96.18/66.70%，恢复弱点明确。当前方法问题包括防御一的输出校正弱点、防御二的DTD独立损失、任务投影的实际贡献，以及指定恢复条件下的能力；不以所有条件均不可恢复为总验收标准，也不能靠增加融合算法解决。

[第二骨干结果](../backbone_b16/summary.md) / [TeX](../backbone_b16/summary.tex)：防御一仍被输出校正恢复；本批不代表防御二有进展。

防御二加入读出补偿后，MNIST、SVHN、DTD 的独立预测与同一 A/B 版一致。AGA 后再替换末层 B，A/B 版的目标准确率为 99.28%、91.38%、64.95%，读出补偿版为 0.07%、0.45%、0.59%。这项处理下伙伴准确率也很低。普通 AGA 下，SVHN 的目标准确率由 1.86% 升至 7.76%，说明读出补偿没有在每项比较上改善保护。

固定 ProDistill 在读出补偿版上的目标准确率为 98.63%、83.45%、30.85%。SVHN 和 DTD 使用和 A/B 版相同的图片、系数核与训练设置。新批次的无保护对照复现了原准确率，权重指纹有差异，因此只比较完整测试结果。MNIST 的已知读出恢复另见[固定恢复检查](../data_free_compensation/readout_channels/summary.md)。

防御一的同条件 B 替换保留了最终偏置改动，三个任务经 AGA 再融合仍为 9.80%、6.70%、2.13%。针对最终偏置的恢复使用不同的位置信息，结果单列在[偏置投影表](../phase3/bias_weight_only/attacks/projection.md)。

## 结果表

| 表 | 内容 |
|---|---|
| [防御二读出补偿](defense2_readout.md) / [TeX](defense2_readout.tex) | 与同一 A/B 发布配对，独立、两种 AGA 范围、公开 B 替换、ProDistill，目标和伙伴同列。 |
| [防御一公开 B 替换](defense1_B_replacement.md) / [TeX](defense1_B_replacement.tex) | 三任务、匹配无保护，替换后独立、TA、两种 AGA 范围。 |
| [攻击结果](attacks.md) / [TeX](attacks.tex) | 无保护、PaRaMS、防御一、防御二；三个保护任务，同条件攻击 |
| [当前Unmask](unmask.md) / [TeX](unmask.tex) | 当前两种防御、PaRaMS及匹配无保护；八专家、七个完整测试集 |
| [防御一与防御二合用](combined_bias_ab/summary.md) / [TeX](combined_bias_ab/summary.tex) | 固定MNIST发布，独立、TA、AGA、偏置清理、B替换和ProDistill；原组件不重拟合。 |
| [PaRaMS组合对照](params_composition.md) / [TeX](params_composition.tex) | 固定MNIST组合的独立、TA、AGA和ProDistill；已知位置恢复另见[恢复表](params_composition_recovery.md)。 |
| [伙伴结果](partners.md) / [TeX](partners.tex) | 相同模型在伙伴任务上的表现 |
| [ProDistill恢复](feature_distillation.md) / [TeX](feature_distillation.tex) | 额外64图/任务的中间特征蒸馏；目标和伙伴同时报告 |
| [八专家融合](eight_expert.md) / [TeX](eight_expert.tex) | 早期版、三个保护任务、七个完整测试集；TA及AGA |
| [八专家恢复对比](eight_calibration_target.md) / [TeX](eight_calibration_target.tex) | 无保护、PaRaMS、防御一、更新版防御二；同一64随机输入校正 |
| [校正后的七任务结果](eight_calibration.md) / [TeX](eight_calibration.tex) | 相同模型在七个完整测试集上的表现 |
| [新版硬标签蒸馏](defense2_hard_label.md) / [TeX](defense2_hard_label.tex) | 相同图像与训练预算；学生独立、目标与伙伴结果 |
| [新版TIES/DARE](defense2_mergers.md) / [TeX](defense2_mergers.tex) | 同一新版防御二，目标与伙伴分别报告 |
| [逐类均值的补偿结果](defense2_native_means.md) / [TeX](defense2_native_means.tex) | 三个任务及伙伴；独立净增加4、58、2个正确预测，DTD输出校正后的保护减弱，未替换主方法。 |
| [互补视图的补偿结果](defense2_native_views.md) / [TeX](defense2_native_views.tex) | 两个通过源条件的任务及伙伴。SVHN 独立净多判对 149 张，DTD 净少判对 31 张。MNIST 源检查失败。 |
| [当前防御二组件](defense2_components.md) / [TeX](defense2_components.tex) | 独立、TA与两种AGA范围，另列[伙伴结果](defense2_component_partners.md)。 |
| [任务投影约束对照](defense2_projection.md) / [TeX](defense2_projection.tex) | 同一构造仅移除A的两侧SVD约束；[伙伴结果](defense2_projection_partners.md) |
| [投影对照的恢复比较](defense2_projection_recovery.md) / [TeX](defense2_projection_recovery.tex) | 同三份发布的MLP匹配、已知B及偏置替换、固定ProDistill；[伙伴结果](defense2_projection_recovery_partners.md) |
| [八专家投影对照](defense2_projection_eight.md) / [TeX](defense2_projection_eight.tex) | 三个保护任务的独立、TA与AGA准确率，另列[七个测试集](defense2_projection_eight_datasets.md)。 |

`attacks`、`partners`及`feature_distillation`现在统一使用当前冻结防御二；早期八专家表仍明确保留旧版。版本差异见[版本比较](defense2_versions.md)、[当前八专家](defense2_eight_versions.md)及对应校正表，不能挑选不同版本的最好列拼接。移除投影约束的三个固定对照独立准确率为99.68/97.41/77.87%，AGA加校正为10.23/12.32/0.90%；SVHN保护减弱，DTD改善。它仍保留原任务合成探针，不能称删除所有SVD用途，也不替换主发布。

[当前组件协议](../data_free_compensation/finite_components_protocol.json)只删除现有改动或构造同等逐张量范数的固定随机对照，不重新优化。A侧包含输入权重和输入偏置，B侧包含输出权重和输出偏置。此前[ProDistill组件表](prodistill_components.md)比较的是蒸馏后系数的作用，不等于防御构造的删项实验。旧版三张量随机和删项结果保留，但不作为当前四张量版的完成证明。

恢复方法的来源、论文代码和适配范围见[攻击清单](../attack_generalization_20261028/attack_inventory.md)。无数据输出校正不需要干净目标专家或修改位置。公开模块替换需要知道操作位置；硬标签蒸馏和ProDistill需要额外真实图片及训练。结果分开报告，既不要求所有权限下绝对不可恢复，也不忽略同一权限内已经发现的恢复。

从仓库根目录生成现有MD/TeX表，无需模型、数据集或GPU：

```bash
python -m experiment.report --defense-comparison --output results/comparison
```

[sources.json](sources.json)记录来源。原始逐图分数、逐类和配对统计保留在服务器。表格沿用[AGA表1与表4](https://arxiv.org/pdf/2606.30360)和[PaRaMS表4](https://openaccess.thecvf.com/content/ICCV2025/html/Junhao_Disrupting_Model_Merging_A_Parameter-Level_Defense_Without_Sacrificing_Accuracy_ICCV_2025_paper.html)的逐任务列和恢复分组：三线表、两位小数，伙伴任务与额外数据条件另列。不照搬其他论文的八任务平均或其未复现的范围。

11月4日，PaRaMS随机剪枝的固定适配检查已关闭：防御二独立99.57%降至40.73%，防御一99.69%降至95.60%，未继续融合或恢复评价。见[完整检查](params_pruning/summary.md)。该点不替换任何主方法。
