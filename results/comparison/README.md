# 当前方法与攻击比较

| 表 | 内容 |
|---|---|
| [攻击结果](attacks.md) / [TeX](attacks.tex) | 无保护、PaRaMS、防御一、防御二；三个保护任务，同条件攻击 |
| [伙伴结果](partners.md) / [TeX](partners.tex) | 相同模型在伙伴任务上的表现 |
| [ProDistill恢复](feature_distillation.md) / [TeX](feature_distillation.tex) | 额外64图/任务的中间特征蒸馏；目标和伙伴同时报告 |
| [八专家融合](eight_expert.md) / [TeX](eight_expert.tex) | 早期版、三个保护任务、七个完整测试集；TA及AGA |
| [八专家恢复对比](eight_calibration_target.md) / [TeX](eight_calibration_target.tex) | 无保护、PaRaMS、防御一、更新版防御二；同一64随机输入校正 |
| [校正后的七任务结果](eight_calibration.md) / [TeX](eight_calibration.tex) | 相同模型在七个完整测试集上的表现 |
| [新版TIES/DARE](defense2_mergers.md) / [TeX](defense2_mergers.tex) | 同一新版防御二，目标与伙伴分别报告 |
| [任务投影约束对照](defense2_projection.md) / [TeX](defense2_projection.tex) | 同一构造仅移除A的两侧SVD约束；[伙伴结果](defense2_projection_partners.md) |
| [投影对照的恢复比较](defense2_projection_recovery.md) / [TeX](defense2_projection_recovery.tex) | 同三份发布的MLP匹配、已知B及偏置替换、固定ProDistill；[伙伴结果](defense2_projection_recovery_partners.md) |
| [八专家投影对照](defense2_projection_eight.md) / [TeX](defense2_projection_eight.tex) | 三个保护任务的独立、TA与AGA准确率，另列[七个测试集](defense2_projection_eight_datasets.md)。 |

PaRaMS是已有防御方法；TA是融合算法。AGA、输出校正、MLP匹配和蒸馏是不同的恢复处理。输出校正采用固定64张随机图，不需要干净目标专家或修改位置；蒸馏额外使用无标签训练图和学生训练，单独说明权限。已知位置的公开模块替换仍见[防御二结果](../attack_generalization_20261028/attack_comparison.md)和[防御一结果](../phase3/bias_weight_only/attacks/projection.md)。这些实现的论文来源和适配范围见[攻击清单](../attack_generalization_20261028/attack_inventory.md)，不都称作官方CLIP攻击复现。

结果按版本分别阅读：

- `attacks` 和 `partners` 使用早期防御二：DTD独立准确率下降3.30个百分点。防御一独立预测保持。`defense2_versions` 单列更新版，DTD下降3.09个百分点，不能混用两版中的最佳列。
- PaRaMS在AGA下接近无保护水平，单独输出校正与AGA后再校正是不同处理。防御一被输出校正恢复；更新版防御二仍有明显保护。结果以对应表中同一处理的列为准。
- 同预算硬标签蒸馏后，三个方法均接近匹配的无保护学生。它与无数据恢复属于不同攻击条件。
- ProDistill恢复后，防御二目标MNIST/SVHN/DTD为99.45/95.78/63.83%，对应无保护99.45/96.73/73.14%；数字任务保护基本失效，DTD仍有9.31个百分点差距。防御一保持9.80/6.70/2.13%，但该流程不更新其最终偏置，且输出校正已有恢复反例，不能称普遍抗蒸馏。
- 防御二联合修改A、B及偏置的作用已有删项对照。最新移除A的任务投影约束后，三任务独立准确率更高，部分恢复后的保护减弱；没有证据称该约束对整体效果必不可少。固定的任务合成探针仍保留，不等于删除所有SVD用途。

八专家均参与融合，但只保护MNIST、SVHN、DTD，并评价七个完整测试集。EuroSAT的原专家训练／测试划分尚未核实，不计算八任务平均。防御一的MLP匹配已补齐；三任务所有MLP匹配均为恒等排列，未触及防御一修改的最终LayerNorm偏置。它不代表对任意全网对齐均安全。

接下来优先处理无需知道修改位置的输出校正，以及公开替换B和偏置后的恢复。只延长B的优化无法改变被替换后的模型，不能把这种重跑当成改进。保留当前固定结果，方法修改只依据合法构造输入及明确机制；不按已曝光测试集挑参数。补齐数据划分和必要比较与方法研究并行，已完成的实验不重复。

从仓库根目录重建以上表格，无需模型、数据集或GPU：

```bash
python -m experiment.report --defense-comparison --output results/comparison
```

所有输入均为仓库内的轻量结果；[sources.json](sources.json)记录来源及八专家表的逐项正确数、样本数和模型指纹。原始分数与独立逐类检查保存在实验服务器。

排版参考[AGA表1与表4](https://arxiv.org/pdf/2606.30360)的防御／攻击分组和逐任务列，以及[PaRaMS论文](https://arxiv.org/pdf/2503.07661)的独立与融合性能对照。使用三线表、任务间分隔和统一两位小数。额外真实数据蒸馏单独分组，伙伴表保留，不作不公平的统一排名；没有照搬八任务平均或他人的实验范围。TeX仅需 `booktabs`，表注随表显示，JSON、Markdown和TeX由现有入口生成。


后续方法对照见[最新三任务结果](../data_free_compensation/calibrated_objective.md#有限分类条件对照2026-11-02)：同一发布列出独立、TA、AGA、校正和参数匹配，已知位置替换另表。它们属于新的固定对照，不能替换本目录旧模型中的个别最佳列。新点的固定ProDistill比较已完成，见[新旧版本目标表](defense2_versions.md)与[伙伴表](defense2_version_partners.md)；[A/B组件表](prodistill_components.md)解释固定恢复模型中两侧的作用。同一新点八专家评价已完成，见[八专家版本对照](defense2_eight_versions.md)。固定64张随机输入校正也已完成，见[目标前后对照](defense2_eight_calibration_target.md)和[七任务完整结果](defense2_eight_calibration.md)，均使用同一冻结发布。

正式排版进一步参考[PaRaMS ICCV2025表4](https://openaccess.thecvf.com/content/ICCV2025/html/Junhao_Disrupting_Model_Merging_A_Parameter-Level_Defense_Without_Sacrificing_Accuracy_ICCV_2025_paper.html)的防御/恢复处理分行与表注。主表报告同一攻击条件下的结果，额外数据和已知位置的恢复单列；不以恢复是否完全成功代替具体准确率。


2026-11-02完成同条件八专家校正比较。AGA后再校正，无保护MNIST/SVHN/DTD为93.19/72.82/38.35%，PaRaMS为93.19/72.82/38.35%，防御一为93.13/72.58/39.36%，更新版防御二为22.99/17.16/28.46%。前两种防御在这套连续恢复处理下接近无保护；防御二的数字任务仍有明显保护，DTD下降9.89个百分点，但独立准确率也由79.41%降至76.33%。这些结果不能合并成“两种防御均抵抗所有恢复”。

这批只补相同条件下的必要基线：12个新校正模型、768次随机输入前向，所有完整真实测试分数复用。每种防御84条完整评价、42个匹配比较，独立逐类和配对检查通过；未新增防御点、训练或AGA求解。

现有实验足以写清已验证的结果，还不足以写“完整八任务有效”或“SVD贡献已证”。当前只保护三个任务；EuroSAT原划分和第二种骨干仍缺证据。较早期版的硬标签蒸馏和Unmask结果不冒充更新版结果；新旧版本的TA、AGA、校正、MLP匹配、ProDistill已另表对齐；新版TIES/DARE也已完整补齐。已知位置和额外数据恢复单列，不要求任何防御在所有权限下绝对不可恢复。


2026-11-03补齐同一新版的TIES、DARE：MNIST/SVHN/DTD的TIES准确率29.78/13.92/2.29%，DARE为26.57/2.28/2.82%。对应无保护TIES99.34/94.37/74.26%，DARE99.36/92.51/69.47%。这是固定融合算法的补充，不是新增恢复攻击；未加AGA前处理。独立准确率和之前相同，伙伴结果完整保留。每组试跑和完整评价均通过原128批上下文重放；独立NumPy配对复算通过。

任务投影对照的独立准确率为99.68/97.41/77.87%，原专家为99.69/97.46/79.41%；AGA后再校正为10.23/12.32/0.90%。三任务同规则、同输入，没有分任务择优。MNIST/SVHN未完全满足合成输入上的零分类违反条件，DTD满足；按执行前约定仍完整报告所有数值有效对照。这是组件作用的证据，不替换主方法发布，也不沿用主方法的TIES/DARE、八专家或蒸馏成绩。

去投影对照的恢复比较已完成。ProDistill后MNIST/SVHN/DTD为99.07/95.82/20.53%，保留约束的主发布为99.43/96.18/66.70%；输入图片、索引和训练步数相同。DTD有明显改善，但数字任务仍被恢复；已知B及偏置替换仍为99.34/92.26/65.96%。同三份发布的八专家校正评价已经完成。AGA加校正后，去投影的三个目标为22.31/29.48/2.39%，保留投影为22.99/17.16/28.46%。
