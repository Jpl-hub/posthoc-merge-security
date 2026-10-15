# 后处理融合保护：相关工作与创新性审查

2026-10-16。**当前没有通过创新性审查的新算法。** O/B保留为有真实融合效果的研究基线，不能据此宣称原创性、跨任务泛化或普遍满足功能预算。此文是新实现之前的依据，不是论文创新性证明。

## 相关工作

| 工作 | 保护发生阶段 | 核心机制与已解决的问题 | 与当前研究的关系及证据边界 |
|---|---|---|---|
| [PaRaMS，ICCV 2025](https://arxiv.org/html/2503.07661v1) | 纯后处理 | MLP成对排列、Attention成对缩放；保持模块函数，破坏直接参数融合 | “功能不变但参数分离”已是原有设计。官方默认已在本框架运行；完整配置超预算，Attention消融单列，不能称同预算完整复现 |
| [MergeLock](https://arxiv.org/html/2509.01548v1) | 纯后处理 | 逐头Q/K、V/O一般可逆变换，严格抵消 | §4.2.2检验的是受保护／未保护专家间耦合的正交对齐，不是AGA的公开基座独立最小二乘。所核固定仓库只有README，未复现 |
| [AGA / ARF](https://arxiv.org/html/2606.30360v1) | AGA为攻击；ARF包含额外训练 | AGA对齐公开预训练锚点并匹配MLP；ARF先排斥锚点训练，再加Attention可逆保护 | AGA攻击代码已固定并运行；ARF不能作为冻结专家的纯后处理基线。固定代码未含ARF训练入口或权重；未复现完整Table 6 |
| [Task Singular Vectors，CVPR 2025](https://arxiv.org/html/2412.00081v2) | 融合／压缩，非防御 | 逐矩阵任务向量SVD，压缩及处理任务间方向干扰 | SVD是已有结构工具；截断参数能量不证明任务功能保持。已核源码定义，未声称完整TSVM复现 |
| [MergeBarrier，AAAI论文](https://ojs.aaai.org/index.php/AAAI/article/download/40433/44394)，[所读正文](https://arxiv.org/html/2511.10712v1) | 后处理；FFN改变表示及计算结构 | Attention正交变换与距离目标；FFN激活多项式展开重写参数 | 非线性重参数化和近似功能保持已有直接先例，不能以此声称首次。其计算结构／开销与原架构下的双0.5协议不同；未定位对应官方实现，未复现，也未核证其密码学主张 |
| [MergeGuard，CVPR 2026正式论文](https://openaccess.thecvf.com/content/CVPR2026/papers/Chen_Defending_Unauthorized_Model_Merging_via_Dual-Stage_Weight_Protection_CVPR_2026_paper.pdf)，[所读v1](https://arxiv.org/html/2511.11851v1) | 完整方法含任务微调；第二步免训练 | 密度感知CE＋L2训练，再掩码减去部分任务向量 | “任务子空间失配”并非空白。不能只截取第二步就把整篇性能当纯后处理结果；未定位可核官方实现，未复现 |
| [Trap²，ICML 2026](https://arxiv.org/html/2601.21898v2)，[官方仓库](https://github.com/ml-postech/Trap2) | 训练期 | 学习任务更新，压低原尺度损失、提高偏离尺度的任务损失 | 将缩放当融合代理并对抗优化也已有先例。官方完整微调入口直接更新模型参数；不得移植后称冻结专家后处理。仅论文及源码阅读，未运行 |
| [Generalized LMC，NeurIPS 2025](https://arxiv.org/html/2506.22712v1)，[官方实现](https://github.com/alexandertheus/Generalized-LMC-for-Transformers) | 对齐研究，非保护 | 统一排列、半排列、正交／可逆变换及近似对称性 | 换基、跨头混合与近似函数保持不能因换个名字而成为原创。此处只作机制边界对照，未复现 |
| [RegMean，ICLR 2023](https://arxiv.org/abs/2212.09849)，[官方示例](https://github.com/bloomberg/dataless-model-merging/blob/main/regmean_demo.ipynb) | 融合，非保护 | 用输入Gram统计最小化线性层预测差异，求闭式融合 | 激活重建／最小二乘补偿本身已有来源。与保护目标不同；这里只核对思想及示例，未运行基线 |
| [ESM，CVPR 2026](https://arxiv.org/html/2602.20208v1)，[官方实现](https://github.com/kiddo127/ESM) | 融合，非保护 | 对真实输入下的任务输出位移做SVD，选择最小化线性输出重建误差的子空间 | 激活加权的任务SVD已有直接先例；平均线性重建误差不保证最坏归一化分类间隔保持。仅阅读，未复现 |

“未定位”限定于此次检查的版本和可访问来源，不表示代码永远不存在。所读版本不自动等同最新修订；后续若出现改变机制或实验结论的版本，需重新审查。搜索不是原创性的完备证明。

## 官方代码依据

固定文件、Git blob及SHA256见[源码核对记录](../results/author_protocol_audit/source_review.json)，不能把网页摘要当实现复现。

| 仓库固定版本 | 核对位置 | 核对事实 |
|---|---|---|
| [PaRaMS `4159a9a`](https://github.com/ISCT-W/PaRaMS/tree/4159a9a8bf253088694154d944f4583db8eca60e) | `PaRaMS_algorithm.py`、`apply_PaRaMS.py`、TA入口 | 默认排列上限200、容差1，Attention区间[0.05,20]；当前对角实现的QK与VO共用alpha；不是论文文字中两个独立采样组 |
| [AGA `fb61429`](https://github.com/krumpguo/secure-merge-attack/tree/fb61429f5e6f3517102d506697a150ac26abbdea) | `utils_attack_B32.py`、`main_task_arithmetic_attack_B32.py` | 逐头锚点最小二乘，正则1e-8、float64求解再FP32；MLP余弦Hungarian匹配。公开主程序只处理当前保护专家；论文§4.1另述无先验时处理全部专家，必须分开记录 |
| [TSV `a6c9718`](https://github.com/AntoAndGar/task_singular_vectors/tree/a6c97188f7aa0bb20753de4f2939be89dd64f1c3) | `task_vectors.py`、`TSVC_utils.py`、`TSVM_utils.py` | 先专家减基座，逐二维矩阵SVD；压缩和多任务正交化是融合操作，不是功能蒸馏证明 |
| [MergeLock `f01aea9`](https://github.com/hetailang/Merge-Lock/tree/f01aea9d7dddf626b6b6536d2a73894feb833fb2) | 完整树检查 | 仅README；不能声称作者算法已运行 |

Trap²的[固定完整微调入口](https://github.com/ml-postech/Trap2/blob/ef65a26590d92fe07955c5621c08bb315484ff29/training_scripts/8vision_training_full_trap2.py)中，AdamW接收`model.parameters()`；`make_scaled_params`计算`w0 + alpha*(p-w0)`，随后对原尺度CE减去缩放CE反传。固定提交`ef65a26`、Git blob为`6e3817b3c6d3e5c1457fc432c837cc7677d39916`，仅核对源码，未运行训练。此公式也不能仅加上硬约束就宣称全新思想。

ESM固定提交[`d60b506`](https://github.com/kiddo127/ESM/blob/d60b50683fd5089578ad69b7c847e2c47025f73c/ESM-ViT/essential_subspace_decomposition.py)，文件Git blob为`bea60fa42a06408b8e8b6ec7291cbaee9cee9c13`。它对$X\Delta W^\top$直接做未减均值的SVD；当前Attention O投影代码被注释，不能把说明文字当全部矩阵均已实现。当前仓库含后续ESM++，此固定文件核对不冒充原论文完整复现。其`Val`数据注册的`test_dataset`字段须先核对划分含义；本项目不能照抄该字段作为合法训练数据证明。

## 损失界与保持条件的核验

Trap²附录A.2的正曲率条件给出损失下界；A.4还需要真实伙伴差向量沿整段路径的积分曲率条件。源专家的尺度敏感性不自动验证未知伙伴的这些条件。这里不否定原文的条件定理，也不将其改写为准确率保证。

预先固定的[解析核验](../experiments/verify_prior_conditions.py)与[结果](../results/author_protocol_audit/prior_conditions.json)给出两个边界：

- 取二分类间隔$m(u)=2-(1-u)^2$，$u\in[1/2,1]$。CE在$u=1$导数为零，整段曲率至少$2/(1+e^2)>0$；从1到1/2损失增加0.033296，大于0.029801下界，但间隔始终不小于7/4，零改判。单位特征和固定线性头可以实现这些分数。这是“损失界不推出改判”的反例，不是CLIP参数候选。
- 取原间隔$g(x,y)=x$、扰动$\delta g=\frac1{100}\prod_{i=1}^{20}((y-i)/(-i))^2$。在20条完整直线$y=i$上，值、法向导数和一阶切向导数均保持；未见点$(-1/200,0)$却由负间隔变为正间隔。这否定有限路径检验的普遍充分性，不声称该多项式已被预算内CLIP参数实现。

保持预算在低间隔区域尤其严格。若连续邻域内处处要求$|g_r-g_e|\le\varepsilon|g_e|$、$\varepsilon<1$，由连续性，原决策边界上必须有$g_r-g_e=0$。在可微且$\nabla g_e\ne0$的正则边界附近，用$g_e$作局部坐标，Hadamard引理给出$g_r-g_e=g_e h$，且$|h|\le\varepsilon$。有限样本、路径或SVD探针不证明此因式关系。此结论解释低间隔泛化的困难；不把连续域证书追加入现有有限验收协议，也不允许事后删去教师误判。

两项核验只用符号代数和精确有理数，无模型、图片或梯度，服务器耗时0.671秒。源码先于执行冻结在`a73032c`；结果不是新防御效果。下一候选仍必须直接检验完整有符号间隔，而不是以曲率、平均重建或有限路径保持替代。

## 非可逆变换与正则化对齐：执行前审查

Generalized LMC官方代码固定在[`5e343c0`](https://github.com/alexandertheus/Generalized-LMC-for-Transformers/tree/5e343c05683ce1f2972c7ca3f25111e82d3f00e5)：`merger.py`将原参数存为buffer并优化匹配矩阵；GPT路径先吸收LN缩放、换成RMSNorm并对残差输出去均值。学习变换系数、残差旋转和头匹配已有先例；该实现也不能直接称为原架构CLIP后处理复现。它需要两个待融合模型，不允许将未来伙伴移入我们的保护构造。

另一个待审设想是：非可逆地删除源任务不使用的方向，能否破坏AGA？这与PaRaMS／MergeLock的可逆变换假设不同，但截断SVD、压缩或零空间本身均不构成创新。必须同时解释有限任务保持、完整分类方向和融合行为；当前不批准该设想为新算法。

对实际正则强度$\lambda=10^{-8}$，AGA列对齐的精确算术输出为

$$A_\lambda(X)=X(X^\top X+\lambda I)^{-1}X^\top X_0
=U\operatorname{diag}\!\left(\frac{\sigma_i^2}{\sigma_i^2+\lambda}\right)U^\top X_0.$$

它对$X$连续。删去一个奇异分量后，对齐差为$-\frac{\sigma_j^2}{\sigma_j^2+\lambda}u_ju_j^\top X_0$，而原任务层响应差为$-\sigma_j u_jv_j^\top z$。两个作用可以不同，但均须经过完整分类读出。原AGA定理的满列秩和可逆保护假设不能自动用于此处；有限正则也不能被偷换成不连续的伪逆极限。

本轮只固定一个解析反例，先冻结再调用官方列求解器核验：$X=\operatorname{diag}(1,1,1/10)$、$X_r=\operatorname{diag}(1,1,0)$，源输入为$e_1,e_2$；公开锚点列依次为$(-1,0,1)^\top,(0,-1,-1)^\top,(0,0,1/10)^\top$。固定二分类头差向量为$g=(1,-1,2)$，保留最终L2归一化。参数距离为$1/\sqrt{201}<0.5$；源输出和任务向量右奇异探针均严格保持。

预期可直接验证：AGA列对齐后两个源预测均翻转，但与同一公开锚点按0.5均值融合后，两个预测又均正确。发布层直接融合也不改判。所有系数事先列出，不搜索。若官方数值实现因FP32舍入改变极小正间隔的符号，分别记录精确有理数、float64公式和作者FP32结果，不扩大容差或把临界数值当鲁棒效果。该反例只针对“对齐后独立任务损害足以保证融合损害”的错误蕴含；不是完整Attention、CLIP防御、新候选或未知伙伴保证，不用于修改历史O/B。

固定解析实验已完成，独立有理数矩阵计算与NumPy求解相符。作者列求解后独立预测为0/2正确；精确算术下与锚点融合恢复2/2，作者FP32结果则为两张平局、1/2正确。该临界平局不能称为稳健分类破坏。结果见[原始核验](../results/author_protocol_audit/anchor_rank_condition.json)及[独立复核](../results/author_protocol_audit/anchor_rank_validation.json)。进一步令锚点第三坐标为$\pm h$，同一解析公式给出融合间隔$h-(1+a)/2$，$h>1$时有严格正余量；此为数学推论，未追加扫描或CLIP实验。

下一项仅为**固定权重的理论条件核对**：读取原MNIST专家与公开基座的最后一块O矩阵，逐一报告全部12头的最小奇异值、$\lambda/(\sigma_{\min}^2+\lambda)$、实际列恢复误差与任务向量误差界。无数据、无方向选择、无截断发布、无新候选。对满列秩$X$，有限正则补充项满足

$$\|A_\lambda(X)-X\|_F\le\|X-X_0\|_F+\frac{\lambda}{\sigma_{\min}^2+\lambda}\|X_0\|_F.$$

只核对该界及达到秩亏的必要距离$\sigma_{\min}/\|X\|_F$；不能由这些谱量推断任务保持或分类保护。全部头一起报告，不按照效果选择头，不比较其他层，不以这一诊断重启已关闭的Q/K、零空间或截断优化。若数值正则影响很小，就排除“原AGA在当前原专家上因近奇异而不稳定”的解释；任意发布后的条件仍须另行验证。

固定真实权重核对已完成并经NumPy独立验证：最后O的12头最小奇异值0.30137—0.37258，条件数1.532—2.586，正则滤波偏差最大1.102e-7。原作者FP32与公式最大差2.854e-8；全部误差界通过。当前原专家没有“接近正则阈值而失稳”的数值前提。逐头秩亏距离0.08473—0.10355以各头范数归一化，**不是全O矩阵的原0.5距离预算**。这不证明任意发布后的稳定性，也不证明秩删除防御可行；没有选择头或生成权重。[全部计数](../results/author_protocol_audit/anchor_spectrum.json)、[独立验证](../results/author_protocol_audit/anchor_spectrum_validation.json)。本轮不据此进入新算法。

## 具体研究缺口

本项目可以研究的**候选缺口**是：公开预训练锚点可用、允许AGA对齐、保持原架构和冻结专家、限制发布变换双0.5预算时，能否利用源任务有限补偿，留下足以跨越完整分类间隔的融合计算差异，并迁移到不同被保护任务？

这不是“此前无人研究过融合安全”。已读工作分别涉及对称性、非线性重写、子空间扰动及训练期对抗保护。尚未在这些已核资料中找到对上述**全部条件同时成立**的对应证据；这只界定待检问题，不能据此宣称首创。仅约束更严也不自动构成方法贡献。

三项困难仍然分离：

1. 全输入函数对称性常与参数对齐存在可逆关系；源任务有限补偿不保证未见输入保持。
2. AGA后非零算子差异不保证分类损害，归一化和正向缩放可能保序。
3. 源端可观察的表示集合不等同未知伙伴分布；必须说明覆盖条件，不得用历史伙伴结果选择新参数。

## O/B的实质区别与不足

O/B在固定架构的最后Attention输出矩阵O与MLP矩阵B之间，优化发布变换系数，使完整有限响应在源任务上相互补偿。它没有全输入可逆恒等式，也没有改变FFN为多项式计算。这区别于所读PaRaMS／MergeLock的坐标对称和MergeBarrier的结构重写；**区别不等于已证明原创或优越**。

冻结效果仍为MNIST独立99.71%、TA78.77%；AGA仅处理保护专家时MNIST79.53%、SVHN65.82%，处理全部参与专家时MNIST77.85%、SVHN61.42%。完整PaRaMS不同预算参考的AGA恢复结果不能当严格同预算优劣证明。匹配协议与统计见[固定源码与协议核对](../results/author_protocol_audit/source_review.json)、[PaRaMS基线](phase2_params_and_structure.md)。

新[独立审计](../results/frozen_signed_condition/OB_task_budget_witness.json)改变了保持结论的边界：训练索引41538上，教师预测5、真值6，保护前后均预测5，但完整归一化分数的最大相对间隔变化为**0.789627＞0.5**。即使原本误判，该输入也没有被原预算排除，不能事后剔除。原确认样本通过和历史准确率均保留；O/B不能再标为已满足全任务最坏0.5约束的候选。该样本仅用于审计，禁止据此修补或选参。

此外，当前仅证明MNIST保护；类别1与7在已测AGA净损失中集中；未知伙伴覆盖、跨任务方法优势、全部攻击鲁棒性均未证明。已知位置预训练参数替换仅是补充诊断，不是AGA或任务向量恢复，也不取消真实TA/AGA效果。

## 统一数学问题

冻结专家$\theta_e$、公开基座$\theta_0$，发布$\theta_r=T_\psi(\theta_e;\theta_0,\mathcal X)$。仅$\psi$可优化，$\theta_e$不更新。沿用原距离

$$D^2(\psi)=\sum_{W\in\mathcal M}\frac{\|W_r-W_e\|_F^2}{\|W_e\|_F^2}\le0.25.$$

矩阵集合与偏置计数必须预先固定，不通过更换分母制造预算。对每个$\Delta W\in\mathbb R^{m\times n}$作$U\Sigma V^\top$：右奇异向量是$n$维模块输入探针，左奇异向量是$m$维输出方向；用真实激活检验响应，不按奇异值大小直接认定功能重要性。

对完整归一化教师分数$s_e$、教师类别$y$，沿用有限任务约束

$$R_{\mathcal X}(\psi)=\max_{x\in\mathcal X,c\ne y}\left|\frac{(s_r-s_e)_y-(s_r-s_e)_c}{s_{e,y}-s_{e,c}}\right|\le0.5.$$

任务SVD蒸馏约束与该完整功能约束共同限制同一个发布变换，不能互相替代；有限$\mathcal X$通过不推出所有输入通过。新独立确认仍是拒绝门槛，不得用于修复。

令$a$为明确攻击操作、$e$为有合法参数实现的环境、$s_m^0,s_m^\psi$为同环境未保护／保护融合分数。真正有关的损害为

$$H(x,c;a,e,\psi)=1-\frac{s_{m,y}^\psi-s_{m,c}^\psi}{s_{m,y}^0-s_{m,c}^0},\qquad H>1\ \Longleftrightarrow\ \text{严格跨过该分类间隔}.$$

一个有实际对手含义的**未解决问题**是：在原双预算及蒸馏约束下，最大化参数分离，同时使攻击者选择$a,e$后的有限分类损害仍在多个类别中成立。对手应最小化损害，竞争类别应检验实际最容易跨越的间隔，再由类别均衡统计检验覆盖；不能用任意分数差或“min-max”字样代替这些定义。没有可辩护的环境集合，就不能宣称完整鲁棒博弈已实现。

此次[精确反例](phase2_finite_signed_condition.md)表明：同一变换的源功能、SVD功能探针和对齐后距离全部相同，功能等价的不同伙伴仍可让融合分别保持或翻转。它否定源端三个指标的普遍充分性，不证明所有后处理防御不可能。

## 实现前的可证伪主张

**当前可检验的研究主张：** 纯坐标对称与源任务有限补偿对公开锚点对齐具有不同的计算后果；若一种非对称的有限补偿能在双0.5约束下保留分类方向上的攻击后响应，它可能在完全相同的TA／AGA协议下优于官方预算内消融，并在新被保护任务上产生效果。**这仍是有缺口的条件假设，尚无获准实现的新变换。**

否定该方法主张的结果包括：与已有机制仅名称不同；任一独立输入／探针超过预算；只有参数残差而无真实分类作用；公平真实评价不优于对照；优势仅依赖测试挑参或单个保护任务。未知环境覆盖无证明时，最多报告固定协议实证，不能声称通用保证。

新实现前须在一两句话中填写：**已有工作的具体局限、改变哪一个数学关系、严格相同的比较设置、什么结果会否定它。** 缺一项就不启动新算法。先完成这一步，再冻结唯一候选，依次独立确认、真实TA和原版AGA。当前不批准扩大任何已关闭优化。
