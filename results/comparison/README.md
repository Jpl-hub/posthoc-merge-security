# 当前方法与攻击比较

| 表 | 内容 |
|---|---|
| [攻击结果](attacks.md) / [TeX](attacks.tex) | 无保护、PaRaMS、防御一、防御二；三个保护任务，同条件攻击 |
| [伙伴结果](partners.md) / [TeX](partners.tex) | 相同模型在伙伴任务上的表现 |
| [ProDistill恢复](feature_distillation.md) / [TeX](feature_distillation.tex) | 额外64图/任务的中间特征蒸馏；目标和伙伴同时报告 |
| [八专家融合](eight_expert.md) / [TeX](eight_expert.tex) | 三个保护任务、七个完整测试集；TA及AGA |

PaRaMS是已有防御方法；TA是融合算法。AGA、输出校正、MLP匹配和蒸馏是不同的恢复处理。输出校正采用固定64张随机图，不需要干净目标专家或修改位置；蒸馏额外使用无标签训练图和学生训练，单独说明权限。已知位置的公开模块替换仍见[防御二结果](../attack_generalization_20261028/attack_comparison.md)和[防御一结果](../phase3/bias_weight_only/attacks/projection.md)。这些实现的论文来源和适配范围见[攻击清单](../attack_generalization_20261028/attack_inventory.md)，不都称作官方CLIP攻击复现。

当前结论：

- 两种防御在所测TA与AGA下均能降低保护任务准确率；防御一独立预测保持，防御二的DTD独立准确率下降3.30个百分点。伙伴任务也会受损。
- PaRaMS在AGA下接近无保护水平，但固定输出校正未恢复PaRaMS。防御一被输出校正恢复，防御二也有明显恢复。因此不能把某一种攻击的结果当作对所有攻击的结论。
- 同预算硬标签蒸馏后，三个方法均接近匹配的无保护学生。它与无数据恢复属于不同攻击条件。
- ProDistill恢复后，防御二目标MNIST/SVHN/DTD为99.45/95.78/63.83%，对应无保护99.45/96.73/73.14%；数字任务保护基本失效，DTD仍有9.31个百分点差距。防御一保持9.80/6.70/2.13%，但该流程不更新其最终偏置，且输出校正已有恢复反例，不能称普遍抗蒸馏。
- 防御二联合修改A、B及偏置的作用已有删项对照；SVD的分类收益尚未证明，不能写成已经成立的贡献。

八专家均参与融合，但只保护MNIST、SVHN、DTD，并评价七个完整测试集。EuroSAT的原专家训练／测试划分尚未核实，不计算八任务平均。防御一的MLP匹配已补齐；三任务所有MLP匹配均为恒等排列，未触及防御一修改的最终LayerNorm偏置。它不代表对任意全网对齐均安全。

接下来优先处理无需知道修改位置的输出校正，以及公开替换B和偏置后的恢复。只延长B的优化无法改变被替换后的模型，不能把这种重跑当成改进。保留当前固定结果，方法修改只依据合法构造输入及明确机制；不按已曝光测试集挑参数。补齐数据划分和必要比较与方法研究并行，已完成的实验不重复。

从仓库根目录重建四张表，无需模型、数据集或GPU：

```bash
python -m experiment.report --defense-comparison --output results/comparison
```

所有输入均为仓库内的轻量结果；[sources.json](sources.json)记录来源及八专家表的逐项正确数、样本数和模型指纹。原始分数与独立逐类检查保存在实验服务器。

排版参考[AGA表1与表4](https://arxiv.org/pdf/2606.30360)的防御／攻击分组和逐任务列，以及[PaRaMS论文](https://arxiv.org/pdf/2503.07661)的独立与融合性能对照。使用三线表、任务间分隔和统一两位小数。额外真实数据蒸馏单独分组，伙伴表保留，不作不公平的统一排名；没有照搬八任务平均或他人的实验范围。TeX仅需 `booktabs`，表注随表显示，JSON、Markdown和TeX由现有入口生成。


后续方法对照见[最新三任务结果](../data_free_compensation/calibrated_objective.md#有限分类条件对照2026-11-02)：同一发布列出独立、TA、AGA、校正和参数匹配，已知位置替换另表。它们属于新的固定对照，不能替换本目录旧模型中的个别最佳列。新点的额外数据蒸馏和八专家评价尚未补跑。

正式排版进一步参考[PaRaMS ICCV2025表4](https://openaccess.thecvf.com/content/ICCV2025/html/Junhao_Disrupting_Model_Merging_A_Parameter-Level_Defense_Without_Sacrificing_Accuracy_ICCV_2025_paper.html)的防御/恢复处理分行与表注。主表报告同一攻击条件下的结果，额外数据和已知位置的恢复单列；不以恢复是否完全成功代替具体准确率。
