# 恢复实验范围（更新至2026-11-04）

当前对象：防御一的纯权重偏置噪声、11月2日防御二的三份固定A/B发布，以及官方默认PaRaMS。保护任务为MNIST、SVHN、DTD。下表与主结果表统一使用当前发布，历史结果保留在各原目录。

| 恢复操作 | 攻击者所需信息或资源 | 防御一 | 防御二 | PaRaMS |
|---|---|---|---|---|
| AGA | 公开基座、发布权重 | 完成 | 完成 | 完成 |
| MLP权重匹配 | 公开基座、发布权重；只对齐MLP隐藏单元 | 完成 | 完成 | 完成 |
| 输出校正及AGA后校正 | 分类头、64张随机图；不需原专家或修改位置 | 完成 | 完成 | 完成 |
| 已知模块恢复 | 知道修改模块；公开参数或基座替换 | 偏置投影完成 | A、B+b、完整MLP替换完成 | 专门结构恢复见历史研究 |
| 硬标签蒸馏 | 额外无标签训练图、教师查询、学生训练 | 完成 | 完成 | 完成 |
| ProDistill | 每任务64张无标签训练图、任务归属、中间特征、系数训练 | 完成 | 完成 | 完成 |
| Unmask减法公式 | 合并权重、发布权重、融合系数 | 完成 | 完成 | 完成 |

主表及伙伴表：[results/comparison](../comparison/README.md)。已知模块恢复见[防御一投影](../phase3/bias_weight_only/attacks/projection.md)和[防御二明细](attack_comparison.md)。硬标签相同的教师明确共用同一学生，不声称重复训练。

## 文献依据和复现范围

| 相关防御论文 | 论文中的恢复检查 | 本项目采用情况 |
|---|---|---|
| [PaRaMS，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Junhao_Disrupting_Model_Merging_A_Parameter-Level_Defense_Without_Sacrificing_Accuracy_ICCV_2025_paper.html) §6、附录H.2 | 机制已知的结构恢复；无标签数据蒸馏 | 蒸馏流程作native适配，数据量和优化器明确固定；不是取得了作者完整KD入口。旧结构恢复不算当前防御二已跑。 |
| [MergeGuard，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Defending_Unauthorized_Model_Merging_via_Dual-Stage_Weight_Protection_CVPR_2026_paper.html) §4、§7 | Unmask、GradErase；主要威胁模型排除昂贵蒸馏 | 当前两种防御及PaRaMS的Unmask已完成；GradErase缺少可复现细节，未跑。不能把该论文当作抗蒸馏的证据。 |
| [MergeBarrier，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40433) §4.1、附录D | 公开基座层替换；针对PaRaMS的解码 | 同类公开替换已完成；我们的联合权重匹配不等于其附录解码器。 |
| [Trap²，ICML 2026](https://proceedings.mlr.press/v306/jang26e.html) §6.4 | SFT、ProDistill恢复 | 当前版本ProDistill已完成；SFT未跑。官方README明确恢复工具未随库提供，ProDistill转引原作者实现。 |
| [MergeLock作者仓库](https://github.com/hetailang/Merge-Lock) | 配套Attention矩阵对齐 | 当前仓库只有README、标注Coming Soon；未取得可执行核，不自行补造为官方实现。 |

输出校正来自[Calibrate Before Use，ICML 2021](https://proceedings.mlr.press/v139/zhao21c.html)，MLP匹配来自[Git Re-Basin，ICLR 2023](https://arxiv.org/abs/2209.04836)。二者是通用方法在本项目的适配，不是专门针对融合防御的官方攻击。Sensitivity as a Shield全文仍遇到网页验证，不从搜索摘要编造机制。FeatCal核已读但未运行，本轮不增加另一套数据辅助流程。

## 当前取舍

已有ProDistill和硬标签蒸馏均已完成。ProDistill使用原作者逐元素系数核，每任务64图、100轮/块、学习率0.01；它不是另一种无数据攻击。当前同版本结果见[目标/伙伴表](../comparison/feature_distillation.md)及[硬标签学生表](../comparison/defense2_hard_label.md)。

停止扩大输出校正实验，原结果保留。已知模块恢复和额外数据恢复单列。TA/TIES/DARE是融合算法，ProDistill是被用于恢复评价的融合算法，不按名称凑攻击数量。机制与源码核对见[研究记录](../../notes/phase3_attack_review.md)，其中标明哪些机制能直接用于当前结构、哪些还缺无数据构造依据。

## 完整任务范围

八个专家参与，三个受保护任务，七个完整测试集：MNIST、SVHN、DTD、GTSRB、RESISC45、Cars、SUN397。EuroSAT已取得第三方2700图，但与原专家训练划分的对应关系未核实，因此不报八任务平均。历史输入追溯、失败记录和旧版本实验见[研究记录](../../notes/phase3_attack_review.md)及各原结果目录。


## 11月2日更新版防御二

同一冻结发布已完成三任务独立、双专家TA/AGA两作用范围、固定64随机输入校正、MLP匹配、已知B及偏置替换、ProDistill，以及八专家TA/AGA仅目标与七数据集输出校正。八专家同条件防御一、PaRaMS及无保护校正对比已补齐，见[目标表](../comparison/eight_calibration_target.md)与[全部评价任务](../comparison/eight_calibration.md)。

该版TIES/DARE、硬标签蒸馏及Unmask同版本补充均已完成，见[融合结果](../comparison/defense2_mergers.md)、[学生恢复](../comparison/defense2_hard_label.md)、[Unmask](../comparison/unmask.md)。Unmask的84条完整结果及1344个原批分数重放已核，三种防御与同处理无保护逐图预测相同。该操作抵消被减专家，低准确率不是本方法相对优势。DTD独立损失及公开替换、额外数据恢复弱点仍在。
