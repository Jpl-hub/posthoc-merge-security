# 防御一实验与结果

11月6日讨论后，后续工作集中在防御一。防御二停止新增试验，已有代码和结果保留作历史对照。防御一当前发布不变，构造只读目标专家、公开基座和固定分类头，不用真实或合成图片。

## 已有结果与待补内容

| 内容 | 已完成 | 待补 |
|---|---|---|
| 独立使用与两专家 TA、AGA | MNIST、SVHN、DTD、RESISC45、Cars、GTSRB、SUN397 七个保护任务；有对应无保护和 PaRaMS 结果 | 按同一主方法整理统一表格 |
| 八专家融合 | 当前防御一专表覆盖三个保护方向、五个完整测试集 | 核对其他已归档分数，补齐缺少的保护方向和评价列 |
| 公开 B 替换 | MNIST、SVHN、DTD；同时替换最后一层 MLP 的 B 和偏置 | 按最终实验范围补齐缺项 |
| ProDistill | MNIST、SVHN、DTD；每任务64张训练图，逐层拟合，最终 LayerNorm 和投影固定 | 按最终实验范围补齐缺项 |
| 其他防御基线 | 官方 PaRaMS 同条件结果 | 重新核对 MergeGuard、MergeLock 的公开实现与本项目兼容性 |
| 其他融合方式 | 已有历史 TA、TIES、DARE 等结果 | 先核对防御发布和协议，只复用与当前主方法一致的结果 |
| 机制分析 | 42组最终层参数的界；两专家、八专家各5/7个任务满足常数分类充分条件 | 六个已固定的方向、幅度控制等待当前八专家作业完成后评价 |

七个保护任务的两专家结果，不等于七个保护方向的八专家结果。八个专家参与融合，也不等于八个数据集都已评价。EuroSAT 的原专家测试划分仍未核实，不列八任务平均数。

## 当前表格

| 内容 | Markdown | TeX |
|---|---|---|
| MNIST、SVHN、DTD 防御一 | [结果](../phase3/bias_weight_only/other_tasks/summary.md) | [表格](../phase3/bias_weight_only/other_tasks/table.tex) |
| RESISC45、Cars、GTSRB、SUN397 两专家比较 | [结果](defense1_additional_pairs.md) | [表格](defense1_additional_pairs.tex) |
| 八专家，三个保护方向、五个评价集 | [结果](../phase3/bias_weight_only/eight_expert/comparison.md) | [表格](../phase3/bias_weight_only/eight_expert/comparison.tex) |
| 公开 B 替换 | [结果](defense1_B_replacement.md) | [表格](defense1_B_replacement.tex) |
| 固定 ProDistill 同条件比较 | [结果](feature_distillation.md) | [表格](feature_distillation.tex) |
| 新补 SUN397/MNIST 的 TIES、DARE+TA | [结果](../../paper/tables/fusion.md) | [表格](../../paper/tables/fusion.tex) |
| 已知位置的最终偏置清理 | [结果](../phase3/bias_weight_only/attacks/projection.md) | [表格](../phase3/bias_weight_only/attacks/projection.tex) |
| 输出校正 | [结果](../phase3/bias_weight_only/attacks/calibration.md) | [表格](../phase3/bias_weight_only/attacks/calibration.tex) |

防御一在前三个任务的独立准确率为99.69%、97.46%、79.41%。在已测 TA、AGA、末层 B 替换和固定 ProDistill 下，目标结果均为9.80%、6.70%、2.13%。最后两项处理没有修改最终 LayerNorm 偏置；已知位置偏置清理和输出校正的恢复结果分别保留，不据此概括为抵抗所有替换或蒸馏。

后续主表按融合方式分组，在相同专家、任务、数据划分和恢复条件下比较无保护、PaRaMS、防御一及可复现的其他防御基线。独立能力、保护任务和其他参与任务同时报告。原始图片不用于防御构造。

## 方法来源与实现

- 防御一构造见 [construct_bias](../../_impl/defenses.py)；入口为 `python -m experiment.construct --method bias`。
- AGA、PaRaMS 和融合算法的作者仓库及固定版本见 [sources.json](../../configs/sources.json)。
- 公开替换和蒸馏的原论文、处理位置和预算见 [攻击清单](../attack_generalization_20261028/attack_inventory.md)。ProDistill 本身是融合算法，本项目用其固定流程作恢复评价。
- 表格布局参考 MergeBarrier，防御比较参考 PaRaMS、MergeGuard、MergeLock 等同类工作。不同架构、任务或权限下的论文数字不直接填入本项目表格。
- 模型、原始分数、失败日志和执行源保留在服务器，轻量记录注明原路径和哈希。当前工程说明见 [engineering.md](../../docs/engineering.md)。
- 当前文稿的表格、矢量机制图及数据图见 [paper](../../paper/README.md)。对近期原论文的实验形式与现有缺口的逐项对应也在该目录说明中。

## 历史对照

[原两方法主表](paper_main.md)、[防御二恢复结果](defense2_versions.md)和[原两方法八专家表](defense2_eight_versions.md)保留作已有结果。历史汇总可用 `python -m experiment.report --defense-comparison --output <临时目录>` 重建，不代表本次扩大后的防御一实验已经完成。
