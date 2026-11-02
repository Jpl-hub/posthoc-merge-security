# 恢复实验范围（更新至2026-11-02）

当前对象：防御一的纯权重偏置噪声、数值防御二的三份固定A/B发布，以及官方默认PaRaMS。保护任务为MNIST、SVHN、DTD。下表的防御二指10月28日早期三任务发布。11月2日更新版的已完成范围见文末；不同发布不互相替代。

| 恢复操作 | 攻击者所需信息或资源 | 防御一 | 防御二 | PaRaMS |
|---|---|---|---|---|
| AGA | 公开基座、发布权重 | 完成 | 完成 | 完成 |
| MLP权重匹配 | 公开基座、发布权重；只对齐MLP隐藏单元 | 完成 | 完成 | 完成 |
| 输出校正及AGA后校正 | 分类头、64张随机图；不需原专家或修改位置 | 完成 | 完成 | 完成 |
| 已知模块恢复 | 知道修改模块；公开参数或基座替换 | 偏置投影完成 | A、B+b、完整MLP替换完成 | 专门结构恢复见历史研究 |
| 硬标签蒸馏 | 额外无标签训练图、教师查询、学生训练 | 完成 | 完成 | 完成 |
| ProDistill | 每任务64张无标签训练图、任务归属、中间特征、系数训练 | 完成 | 完成 | 完成 |
| Unmask减法公式 | 合并权重、发布权重、融合系数 | 当前版本未跑 | 完成 | 未跑 |

主表及伙伴表：[results/comparison](../comparison/README.md)。已知模块恢复见[防御一投影](../phase3/bias_weight_only/attacks/projection.md)和[防御二明细](attack_comparison.md)。硬标签相同的教师明确共用同一学生，不声称重复训练。

## 文献依据和复现范围

| 相关防御论文 | 论文中的恢复检查 | 本项目采用情况 |
|---|---|---|
| [PaRaMS，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Junhao_Disrupting_Model_Merging_A_Parameter-Level_Defense_Without_Sacrificing_Accuracy_ICCV_2025_paper.html) §6、附录H.2 | 机制已知的结构恢复；无标签数据蒸馏 | 蒸馏流程作native适配，数据量和优化器明确固定；不是取得了作者完整KD入口。旧结构恢复不算当前防御二已跑。 |
| [MergeGuard，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Chen_Defending_Unauthorized_Model_Merging_via_Dual-Stage_Weight_Protection_CVPR_2026_paper.html) §7 | Unmask、GradErase | Unmask公式已适配到防御二；GradErase的估计器及训练配置未取得可复现实现，未跑。 |
| [MergeBarrier，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40433) §4.1、附录D | 公开基座层替换；针对PaRaMS的解码 | 同类公开替换已完成；我们的联合权重匹配不等于其附录解码器。 |
| [Trap²，ICML 2026](https://proceedings.mlr.press/v306/jang26e.html) §6.4 | SFT、ProDistill恢复 | 当前版本ProDistill已完成；SFT未跑。官方README明确恢复工具未随库提供，ProDistill转引原作者实现。 |
| [MergeLock作者仓库](https://github.com/hetailang/Merge-Lock) | 配套Attention矩阵对齐 | 当前仓库只有README、标注Coming Soon；未取得可执行核，不自行补造为官方实现。 |

输出校正来自[Calibrate Before Use，ICML 2021](https://proceedings.mlr.press/v139/zhao21c.html)，MLP匹配来自[Git Re-Basin，ICLR 2023](https://arxiv.org/abs/2209.04836)。二者是通用方法在本项目的适配，不是专门针对融合防御的官方攻击。Sensitivity as a Shield全文仍遇到网页验证，不从搜索摘要编造机制。FeatCal核已读但未运行，本轮不增加另一套数据辅助流程。

## 本轮取舍

只补ProDistill这一项：它直接出现在相关防御论文的恢复评价中，与已完成的硬标签蒸馏不同。复用Oct18作者系数核和64图、100epoch、lr0.01规则；当前三任务、两种防御和PaRaMS全部报告。无新AGA、无防御更新，不扫超参数。MNIST无保护模型按SHA复用；SVHN/DTD旧模型不在保留资产中，按旧规则重建匹配控制。防御一只改最终偏置，逐张量核对后复用相同训练块。定义见[protocol.json](prodistill/protocol.json)，已核结果见[summary.json](prodistill/summary.json)和[目标/伙伴表](../comparison/feature_distillation.md)。

主实验保留AGA、MLP匹配和输出校正；已知模块恢复和额外数据恢复单列。TA/TIES/DARE是融合算法，ProDistill也是被用于恢复评价的融合算法，不按名称凑攻击数量。当前证据已表明输出校正是实质弱点；继续补表不能代替改进核心方法。

## 完整任务范围

八个专家参与，三个受保护任务，七个完整测试集：MNIST、SVHN、DTD、GTSRB、RESISC45、Cars、SUN397。EuroSAT已取得第三方2700图，但与原专家训练划分的对应关系未核实，因此不报八任务平均。历史输入追溯、失败记录和旧版本实验见[研究记录](../../notes/phase3_attack_review.md)及各原结果目录。


## 11月2日更新版防御二

同一冻结发布已完成三任务独立、双专家TA/AGA两作用范围、固定64随机输入校正、MLP匹配、已知B及偏置替换、ProDistill，以及八专家TA/AGA仅目标与七数据集输出校正。八专家同条件防御一、PaRaMS及无保护校正对比已补齐，见[目标表](../comparison/eight_calibration_target.md)与[全部评价任务](../comparison/eight_calibration.md)。

该版未另跑TIES/DARE、硬标签蒸馏或Unmask；上面的早期版结果不计作其完成记录。该版独立DTD损失及公开替换、额外数据恢复弱点仍在，不为凑齐名称重复启动同类攻击。
