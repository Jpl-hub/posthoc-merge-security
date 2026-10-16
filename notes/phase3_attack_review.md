# Phase 3: attack review and evaluation protocol

Phase 2 停止构造新防御。第一阶段、O/B、A/B、PaRaMS 及失败记录原样保留。所有攻击结果只用于评价，不更新保护权重。

## 方法与权限

| 方法 | 类别、原问题 | 攻击者所需信息 / 代价 | CLIP ViT-B/32 的使用范围 |
|---|---|---|---|
| [AGA](https://github.com/krumpguo/secure-merge-attack) | 公开基座锚点对齐攻击；Attention 最小二乘、MLP 匈牙利匹配 | 发布专家、公开基座；无需图片、标签或训练；逐头求解与逐层匹配 | 固定作者代码，复用有哈希和独立验证的攻击权重。分别报告仅处理保护专家、处理全部参与专家 |
| [PaRaMS §6 自适应恢复](https://arxiv.org/html/2503.07661v1#S6) | 已知防御的 MLP 重排与 Attention 缩放恢复 | 公开基座、方法知识；对齐求解 | 论文机制已核对；与 AGA 有重叠，不能将 AGA 重命名后计为第二个已复现攻击。独立官方恢复入口尚未确认 |
| PaRaMS §6 蒸馏恢复 | 用可查询的独立专家训练替代模型，再融合 | 同分布无标签图片、教师查询、额外训练 | 不属于无数据对齐。单列高成本权限层，本批不运行；不据此声称当前防御抗蒸馏 |
| [Git Re-Basin](https://github.com/samuela/git-re-basin) | 为融合寻找函数等价的神经元置换；不是针对本防御提出的攻击 | 待对齐模型及参照；weight matching 无标签，activation matching 需图片 | 官方实现主要为 MLP/VGG/ResNet，不能直接宣称完整 CLIP Transformer 复现。AGA 已包含不同目标的 MLP 匹配；暂不另写适配器凑数量 |
| [Task Arithmetic](https://github.com/mlfoundations/task_vectors) | 融合基准：基座加任务向量加权和 | 发布专家、伙伴、公开基座；无需数据或训练 | 固定系数 0.5；原作者差分与求和代码，另用显式公式复核 |
| [TIES-Merging](https://github.com/prateeky2806/ties-merging) | 融合算法：全局幅度裁剪、符号选择、同号聚合 | 同上；无需数据或训练 | 新增正式融合对照。固定作者函数、保留前 20%、mass 符号、disjoint mean、系数 1.0；不是论文最优系数表的复现 |
| [DARE / MergeLM](https://github.com/yule-BUAA/MergeLM) | 随机删除并重标定任务增量；融合组件，不是恢复攻击 | 同上；需记录删除率及随机种子 | 原论文以语言模型为主。待固定源码和 CLIP 适配检查；本批无 DARE 结果，不按结果挑种子 |
| [TSV-Merge](https://github.com/AntoAndGar/task_singular_vectors) | 分矩阵 SVD 压缩与去干扰融合 | 所有发布任务向量及基座；矩阵分解，无标签 | 与现有 checkpoint 兼容；须核对向量/偏置处理及秩规则。既往 SVD 诊断不算完整 TSV-Merge 复现 |
| [RegMean](https://github.com/bloomberg/dataless-model-merging) | 用输入二阶统计约束线性层重构的融合算法 | 专家、基座及校准输入，无需标签；收集激活 | 数据权限不同；需先固定校准来源和非线性层处理，本批不运行 |
| [AdaMerging](https://github.com/EnnengYang/AdaMerging) | 在无标签数据上最小化熵以学习融合系数 | 专家、基座、输入与系数梯度 | 属于数据辅助自适应融合。原文使用无标签测试数据，不能与当前无数据协议混表；后续须另设适配集与评价集 |
| [PaRaMS](https://github.com/ISCT-W/PaRaMS) | 纯后处理防御，MLP 置换与 Attention 配对缩放 | 防御构造无需伙伴/图片 | 已完成官方完整配置及 Attention 消融；完整配置超参数预算，分列参考 |
| [MergeLock](https://github.com/hetailang/Merge-Lock) | Attention 可逆保护防御 | 防御端变换；论文讨论对齐与逆变换风险 | 固定审计版本仅 README，未复现完整方法，不能填入自产数字 |
| [ARF](https://arxiv.org/html/2606.30360v1) | 排斥锚点微调后施加可逆保护 | 原任务训练和梯度 | 训练期防御；不作为纯后处理同约束基线，未复现完整训练/Table 6 |

AGA 与 PaRaMS 已固定版本见 `configs/sources.json`；TIES 本批版本为 `44e7891fc84f3de7e4caa52664cd864ca3715e91`。新增调研条目的链接不等于已固定源码或已复现。BadMerging 等污染伙伴的后门攻击针对模型完整性，不能替代这里的受保护能力恢复评价。

## 冻结对象与约束

权重路径、SHA-256、原 checkpoint、攻击权重和历史预测索引统一列于 `configs/phase3.json`。大文件只在 3090。发布权重不再生成、插值、裁剪或修补。

| 对照 | 当前证据边界 |
|---|---|
| 无保护 | 每个任务组合、融合算法、攻击范围分别提供匹配基线 |
| Phase 1 LN bias | MNIST、DTD、SVHN；理想输出核保持整个独立特征。实际参数距离另报，不能默认满足 Phase 2 的 0.5 预算 |
| O/B | 有真实融合干扰；训练索引 41538 已核验漂移 0.789627，超过 0.5。只列效果参考，不作严格可行方法 |
| A/B | 既往独立有限确认 0.463577；没有全输入保持证明，是本阶段重点评价对象 |
| PaRaMS 完整官方配置 | 矩阵距离 30.4251，全部变化张量距离 47.2573；超预算参考 |
| PaRaMS Attention 消融 | 官方组件消融，全部变化张量距离 0.264848；不是完整官方方法的同预算复现 |

## 统一协议

模型为相同公开 CLIP ViT-B/32 基座的冻结专家。FP32，关闭 TF32，只用 RTX3090。分类头独立保留，不参与融合。

设公开基座为 $\theta_0$，发布目标为 $\theta_d$，伙伴为 $\theta_p$，则

\[
\theta_{TA}=\theta_0+0.5[(\theta_d-\theta_0)+(\theta_p-\theta_0)].
\]

两模型下等于算术平均，但复核保留原公式。对齐函数 $A_{\theta_0}$ 使用固定原作者代码，三行分别为

\[
M(\theta_d,\theta_p),\quad
M(A_{\theta_0}(\theta_d),\theta_p),\quad
M(A_{\theta_0}(\theta_d),A_{\theta_0}(\theta_p)).
\]

英文表列名采用 **No attack / AGA (protected expert) / AGA (all experts)**。两种作用范围不能合并或取较低准确率。TIES 与 TA 分表；TIES 系数 1.0 乘的是同号非零均值，不能与 TA 的 0.5 向量和混称为相同系数。

TIES 直接执行固定文件中未修改的纯函数；AST 仅排除 T5 加载、绘图等无关导入。全模型浮点张量按键排序后统一展平，包含偏置与 LayerNorm，逐任务保留前 20%，不逐层设阈值，不因防御强度改变裁剪规则。用一个固定小矩阵的独立 NumPy 实现检查裁剪、符号与聚合，并重建完整模型重放图片；这不等于第二套完整规模 TIES 实现。

固定四组：MNIST→SVHN、MNIST→DTD、DTD→MNIST、SVHN→MNIST。前两组评价同一 MNIST 发布权重；后两组只评价已有 Phase 1 专家，不能冒充 A/B 跨被保护任务验证。

测试使用 MNIST 10,000、SVHN 26,032、DTD 官方 test1 1,880 张。历史测试已多次查看，本阶段是冻结方法的系统评价，**不是全新盲测**。不通过本阶段结果选择保护参数或攻击超参数。

每一行保存目标与伙伴的正确数/样本数、准确率、逐类正确数、相对匹配无保护控制的改判数、lost/gained/net loss、精确配对 McNemar 的 log10 p、配对准确率差的近似 95% 区间。大样本 p 值极小不代替效果量；图中点估计不是随机种子重复均值。当前 TA/TIES/AGA 为固定确定性配置，后续随机方法才需要预注册重复种子。

另外只读审计每个被保护任务的 512 张训练图，固定种子 20261016，不排除教师错误。对所有正教师间隔计算完整分类分数的相对变化，记录最大值和超过 0.5 的输入数。该批可能与历史训练索引重合，不叫全新独立确认，更不用于修复权重。参数距离同时报告逐矩阵归一化平方和与全部变化张量口径；零原范数单列，不偷偷加分母常数。任务向量 SVD 探针固定为最后一块 O、MLP A/B 矩阵的前 32 个右奇异方向，输入维度随矩阵变化；只衡量线性探针作用，不冒充完整任务功能证书。

优先导入带完整分数、哈希和一致协议的历史预测，并逐行重放；旧文件若只有预测标签、没有分数，则补跑完整分数评价，不伪造 logits。所有导入标注来源。

先以固定均匀索引 64 张/任务完成 MNIST/SVHN 端到端测试，核对官方标签、作用范围、TA 显式公式、TIES 全局向量处理及三类权重状态。独立复核通过后才启动完整四组。全部行都保存，不以结果优劣筛掉方法。正式输出在 `outputs/phase3/`，轻量归档在 `results/phase3/`。

## 安全结论的限制

原版 AGA 不修改最终 LayerNorm 偏置或视觉投影，Phase 1 效果包含明确的攻击覆盖缺口。不能由此宣称对任意恢复攻击鲁棒。已知位置的预训练参数替换是额外、带位置知识的诊断，不是原任务向量恢复，也不叫原版 AGA。此次主实验不围绕它调防御。

O/B 越界见证、A/B 的有限确认、PaRaMS 超预算和类别集中损失全部保留。无数据 AGA、数据辅助融合、蒸馏训练的攻击者能力分别列明，不能在一张表中省略权限差别并声称统一安全保证。
