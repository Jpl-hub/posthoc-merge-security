# 组会讲稿（约 4–5 分钟）

本文为 19:30 组会前讲稿。组会后的新增实验与结论见[研究简报](post_meeting/report.md)，不改写当时汇报的结果。

老师，今天我围绕您说的“不通过微调，加后处理能力”推进：对已经微调好的专家，在发布前增加保护，保住独立任务能力，同时抵抗未经授权融合和自适应恢复。本轮具体检验了两项候选，结果都是负面的，还没有得到有效的新防御。

先说文献。我重点核对了 PaRaMS、MergeLock、MergeBarrier，以及您发的《On the Vulnerability of Parameter-Level Defenses to Model Merging》，也就是提出 AGA 恢复攻击的那篇。PaRaMS 和 MergeLock 已经覆盖置换、缩放与可逆混合；AGA 说明公开预训练模型可以作为恢复锚点。MergeBarrier 则转向 FFN 的 Taylor 表示，所以换参数格式和多项式表示也已有前作。

我还补读了 TSV 和 SVD-LLM，确认低秩压缩有成熟依据，甚至可能改善融合，不能把“不可逆”直接当作安全。PLeaS 提醒我必须检查功能最小二乘恢复，而不仅是参数匹配。MergeGuard 完整方法包含额外微调，其免训练第二阶段也已有消融。这些文献主要帮助排除重复思路，并明确应该测哪些攻击；还没有形成新颖性证明。

实验沿用官方 CLIP ViT-B/32 权重、作者的融合和 AGA 代码。保护 MNIST 专家，与 SVHN 专家融合，评价全部一万张 MNIST 和 26,032 张 SVHN 图像。独立 MNIST 基线是 99.69%；系数 0.5 的未保护融合是 99.37% 和 92.50%。只有一个种子，用于初筛，不能当作八任务复现或统计显著性。

基础 PaRaMS、正交保护和此前的解析扰动都保留了完整对照，但未挡住相应恢复攻击。PaRaMS 的 dropout 加固还没有完整复现。本轮把重点转向不可逆编辑与完整 FFN 功能重表示，独立性能损失预算预先设为 0.5 个百分点。

第一项候选是不可逆秩收缩：用 32 张合成笔画图像的激活确定投影，把最后四层 V 输出压到 rank 16，删去每个 head 四分之三的维度。独立 MNIST 从 99.69% 降到 99.61%，只损失 0.08 个百分点，但直接融合已经有效。攻击者从发布参数估计保留空间，用公开预训练模型补全其余方向，不需要数据或秘密投影，融合达到 99.37% 和 92.60%。

第二项候选是完整 FFN 重表示：将最后一个 FFN 改写为二次映射，删除该块原 W1、W2。用 32 张无标签合成笔画图像闭式拟合，没有梯度或优化器，但存在新参数拟合，不能称无需数据。独立 MNIST 保持 99.69%；共同格式融合仍然有效。攻击者用另一批 32 张合成图像拟合回原架构后，融合为 99.36% 和 92.51%。

在单层失败后，我做了固定四层的覆盖消融，保持同一 rank、阶数和正则。独立 MNIST 降到 88.78%，损失 10.91 个百分点，远超预算；功能重构后融合又达到 99.00% 和 90.13%。这项配置因独立效用不达标而淘汰。

这轮实验说明，局部秩降低没有转化为融合保护；单层改写保住效用却可被恢复，四层改写则损伤了原有能力，直接融合下降不能计作成功。攻击者可以得到另一种可融合表示，无需找回原参数。这只是本次配置的结论，不证明所有后处理防御不可能，也不是对 MergeBarrier 原方法的攻击结论。

下一步停止扫描当前失败的配置族，先分析哪些任务特异信息能在小性能损失下删除，同时难以被公共预训练模型或功能拟合补回。有可检验构造后，再固定攻击样本和计算预算做真实实验；通过独立效用与恢复攻击两道检查后，才扩大任务和种子。

八任务复现也继续补资源：全部分类 head 已取得，DTD 第一划分和 GTSRB 全测试入口已验证。六个官方专家权重仍受下载配额限制，公开镜像逐参数不一致，已拒绝替代，当前没有八任务准确率。完整结果和失败记录都已保留。

## 论文出处备查

以下用于提问时查阅，不计入朗读部分；原文位置及版本见 [文献核对](literature_triage.md)。

- [AGA：On the Vulnerability of Parameter-Level Defenses to Model Merging](https://arxiv.org/abs/2606.30360v1)
- [PaRaMS：Disrupting Model Merging](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Junhao_Disrupting_Model_Merging_A_Parameter-Level_Defense_Without_Sacrificing_Accuracy_ICCV_2025_paper.pdf)
- [MergeLock](https://arxiv.org/abs/2509.01548)、[MergeBarrier（本次审读 v1）](https://arxiv.org/abs/2511.10712v1)
- [PLeaS](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Nasery_PLeaS_-_Merging_Models_with_Permutations_and_Least_Squares_CVPR_2025_paper.pdf)
- [Task Singular Vectors](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Gargiulo_Task_Singular_Vectors_Reducing_Task_Interference_in_Model_Merging_CVPR_2025_paper.pdf)、[SVD-LLM（v5）](https://arxiv.org/abs/2403.07378v5)
- [MergeGuard：Defending Unauthorized Model Merging via Dual-Stage Weight Protection](https://www.openaccess.thecvf.com/content/CVPR2026/papers/Chen_Defending_Unauthorized_Model_Merging_via_Dual-Stage_Weight_Protection_CVPR_2026_paper.pdf)
