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
| [MergeBarrier，AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/40433) §4.1—4.2、附录D | 公开基座层替换；针对PaRaMS的解码；额外带标签数据微调 | 同类公开替换已完成；我们的联合权重匹配不等于其附录解码器。 |
| [Trap²，ICML 2026](https://proceedings.mlr.press/v306/jang26e.html) §6.4 | SFT、ProDistill恢复 | 当前版本ProDistill已完成；SFT未跑。官方README明确恢复工具未随库提供，ProDistill转引原作者实现。 |
| [MergeLock作者仓库](https://github.com/hetailang/Merge-Lock) | 配套Attention矩阵对齐 | 当前仓库只有README、标注Coming Soon；未取得可执行核，不自行补造为官方实现。 |

输出校正来自[Calibrate Before Use，ICML 2021](https://proceedings.mlr.press/v139/zhao21c.html)，MLP匹配来自[Git Re-Basin，ICLR 2023](https://arxiv.org/abs/2209.04836)。二者是通用方法在本项目的适配，不是专门针对融合防御的官方攻击。Sensitivity as a Shield全文仍遇到网页验证，不从搜索摘要编造机制。FeatCal核已读但未运行，本轮不增加另一套数据辅助流程。

## 当前取舍

11月4日按论文与本文问题确定优先级：**主实验为TA及AGA处理后的融合；当前方法改进优先检验公开参数替换与ProDistill。** 后两项分别对应MergeBarrier §4.1与Trap² §6.4，当前三任务均已完成。它们是现阶段的代表性检查，不是老师指定的全部清单，也不是所有同类论文的统一要求。ProDistill是用于恢复评价的蒸馏融合方法，不是为本项目新造的攻击。Trap²使用每任务64张留出图和逐元素系数；本项目使用每任务64张训练图及两个专家，明确属于适配，不称原表复现。

硬标签蒸馏、Unmask、MLP匹配、输出校正等已完成结果保留，说明来源和条件。SFT在Trap² §6.4及MergeBarrier §4.2确有先例，目前未跑；GradErase有论文描述但本项目未取得足够的可复现细节。当前先解决已复现的蒸馏弱点，不立即增加这些实验；若最终声称抵抗额外带标签微调，则必须补同条件SFT，或明确收窄结论。TA/TIES/DARE归入融合方式。新增实验需有相关原始来源、明确的攻击者资源及可重现设置，并回答现有结果未回答的问题；不要求合并所有论文的实验，也不因预计结果不好而省略直接相关的检查。

两种防御目前都保持了AGA下的保护；在固定ProDistill设置下，防御一三个目标为9.80/6.70/2.13%，防御二为99.43/96.18/66.70%。前者的最后LayerNorm与投影未被此恢复优化，不能外推为抵抗所有蒸馏；后者在此条件下确有恢复弱点。输出校正对防御一有效的既有结果也继续保留。后续按不同条件陈述效果，不设“所有恢复必须失败”的总门槛，也不通过改变归类消除已观察到的弱点。

11月4日按教师截图进一步收敛：先确认同类防御论文实际测过什么恢复、什么条件下仍有效，再判断其机制能否用于本项目。不是把所有恢复方法都补一遍，也不是将“测过恢复”写成“防住恢复”。现阶段不新增输出校正变体；保留已完成记录。

| 论文中的实际证据 | 起作用的设计与实现 | 本项目的取舍 |
|---|---|---|
| PaRaMS表4：无保护84.13%，剪枝后的PaRaMS经KD为82.27%；其自适应结构恢复为64.35% | 置换、缩放后再随机零化10%权重，主要干扰结构恢复。表4不支持它已经强力防住KD | 官方默认组合和剪枝已测；剪枝使当前防御二独立准确率降至40.73%，该点关闭，不继续把它当作蒸馏解决方案 |
| Trap²表14：ProDistill的平均准确率从73.823%降至49.153%，SFT从56.859%降至45.496% | 训练时保持正常任务损失，并提高缩放任务更新后的损失。官方脚本更新整个模型，使用真实带标签图；没有在防御训练中展开ProDistill | 有恢复后仍降低性能的证据，但并非完全阻止恢复。借鉴构造目标，不能直接复制其有数据训练、也不能把跨任务平均当成我们的目标任务准确率 |
| MergeBarrier §4.1、表2：改写层换回公开基座再融合，LM与Math的TA结果中GSM8K由62.17%降至0% | Attention投影与FFN结构改写同时存在；指定的层替换不等于删除全部保护。表4单独Attention或FFN结果均弱于组合 | 需要看替换后还保留哪部分作用。尚无已核官方执行版本；不自行写一个同名实现。不能将结果归因于单个多项式模块，也不重启已关闭的正交变换点 |

原表位置：[PaRaMS §6表4](https://arxiv.org/html/2503.07661v1#S6)、[Trap²附录表14](https://arxiv.org/html/2601.21898v2)、[MergeBarrier §4及表2/4](https://arxiv.org/html/2511.10712v2#S4)。Trap²各列指定被保护任务、单元格为融合模型的跨任务平均，不能与本项目单个目标列直接比较。数字均为原论文结果，不是本项目复现。

TAR仅作“恢复后再计算保护目标”的补充机制参考，研究对象是LLM抗微调，不是同类CLIP融合防御基线。固定恢复导数已完成核对；用其调整现有A/B构造的固定检查已结束：保持合成视图原生分类，但恢复后的64个预测没有改变。相关数值构造尚未有新成功结果，不能把导数正确或优化目标改善当作防御有效。

已有ProDistill和硬标签蒸馏均已完成。ProDistill使用原作者逐元素系数核，每任务64图、100轮/块、学习率0.01；它不是另一种无数据攻击。当前同版本结果见[目标/伙伴表](../comparison/feature_distillation.md)及[硬标签学生表](../comparison/defense2_hard_label.md)。

停止扩大输出校正实验，原结果保留。已知模块恢复和额外数据恢复单列。TA/TIES/DARE是融合算法，ProDistill是被用于恢复评价的融合算法，不按名称凑攻击数量。机制与源码核对见[研究记录](../../notes/phase3_attack_review.md)，其中标明哪些机制能直接用于当前结构、哪些还缺无数据构造依据。

## 完整任务范围

八个专家参与，三个受保护任务，七个完整测试集：MNIST、SVHN、DTD、GTSRB、RESISC45、Cars、SUN397。EuroSAT已取得第三方2700图，但与原专家训练划分的对应关系未核实，因此不报八任务平均。历史输入追溯、失败记录和旧版本实验见[研究记录](../../notes/phase3_attack_review.md)及各原结果目录。


## 11月2日更新版防御二

同一冻结发布已完成三任务独立、双专家TA/AGA两作用范围、固定64随机输入校正、MLP匹配、已知B及偏置替换、ProDistill，以及八专家TA/AGA仅目标与七数据集输出校正。八专家同条件防御一、PaRaMS及无保护校正对比已补齐，见[目标表](../comparison/eight_calibration_target.md)与[全部评价任务](../comparison/eight_calibration.md)。

该版TIES/DARE、硬标签蒸馏及Unmask同版本补充均已完成，见[融合结果](../comparison/defense2_mergers.md)、[学生恢复](../comparison/defense2_hard_label.md)、[Unmask](../comparison/unmask.md)。Unmask的84条完整结果及1344个原批分数重放已核，三种防御与同处理无保护逐图预测相同。该操作抵消被减专家，低准确率不是本方法相对优势。DTD独立损失及公开替换、额外数据恢复弱点仍在。
