# 精读与机制边界

10 月 11 日。按用户最新要求，优先正文提供代码链接、近期发布或更新的工作。当前精读对象为 AGA、NUFILT 的 2026 更新版，以及 10 月发布的 SLOA；引用旧实验时保留原记录，不重新扩大旧文献清单。下列是原文方法、证明条件与评价协议的核对，复现状态分别标明。

## AGA：参数距离的上界不是防御成功条件

[论文 v1，2026-06-29](https://arxiv.org/html/2606.30360v1)，正文提供[作者仓库](https://github.com/krumpguo/secure-merge-attack)。本轮重读 §2–3、§5 与附录证明，结合既有原版函数调用记录。

Attention 求解针对公开 anchor 的线性对齐；附录满列秩、可逆右变换条件下，恢复结果是原模型列空间上的 anchor 投影，与秘密基变换无关。任务增量范数控制的是恢复误差上界。**扩大这个上界不能推出恢复误差必然增大，更不能推出融合准确率下降。**加入可被投影或替换删除的大噪声不能借此得到安全保证。

MLP 实现用行归一化后的 cosine assignment；附录的原始欧氏 margin 条件不能无条件套到该实现。已有固定反例与数值复核保留于[理论记录](../2026-10-09/theory_notes.md)。这属于证明条件核查，不足以推翻原版 AGA 的实测恢复。

§5 的 ARF 在任务微调中改变 Attention 与 anchor 的距离，随后施加可逆保护。它不能直接满足本项目“给既有专家发布前处理、不额外微调”的约束。已有作者 AGA 调用与七任务结果是真实实验；ARF 没有复现。下一步候选必须另证独立能力、恢复后效果和参数删除代价。

## NUFILT：对齐是有条件的，完整方法还有训练

[v2，2026-03-15 更新](https://arxiv.org/html/2509.21413v2)，正文提供[作者仓库](https://github.com/zihuanqiu/NUFILT)。本轮重读 Theorem 1、Corollary 1、§4 与附录 A。

定理需要任务更新包含与表征相连的低秩信号、误差界和正谱间隔，并要求代理维数覆盖表征秩；它约束子空间亲和度。推论控制局部线性响应误差，保留谱间隔误差项。**不能据此认定任意层固定低维，也不能把单个主奇异方向当成不可删除的分类信息。**本轮 MNIST 删除诊断正检验了这一差别。

完整流程包括过滤、训练低秩适配器与融合。只复用其 SVD 和对齐思想，不等于完整 NUFILT 复现，也不能将完整方法称为没有训练。当前 pin 的 SVD 复用及既有响应能量诊断均有源码指纹；本轮没有训练适配器。

## SLOA：攻击者可以处理计算算子

[v1，2026-10-05](https://arxiv.org/html/2610.06415v1)，正文链接至 [GitHub 项目页](https://akansh12.github.io/SLOA-Merge/)，其中提供[作者代码](https://github.com/akansh12/SLOA-Merge)。本轮精读 §3、附录 D/E，核对原秩与扩秩评价。

它对组合 QK／VO、LayerNorm 仿射与线性模块建立局部行为目标，并使用部分融合模型实际产生的上游输入，顺序求解带数据几何正则的二次问题。QK／VO 最后分解为可执行参数；保留原 head 秩和增加容量分别报告。视觉校准来自训练集，每任务 256 张。

由其算子不依赖内部因子基这一性质，可推断单纯改变坐标仍可能被功能融合绕过；**这是对本项目的攻击启示，没有测得 SLOA 对我们的候选恢复成功。**当前未运行完整作者 SLOA，既有 MLP 重建不能改称其复现。

## 对下一步的约束

本轮还核查了[EMNLP Findings 2025 头对齐／合并](https://aclanthology.org/2025.findings-emnlp.467/)及其[作者仓库](https://github.com/fpcsong/mha2gqa)：Procrustes、冗余头压缩已有先例，不能作为本项目新机制的创新依据。[2026 CAHP](https://arxiv.org/html/2606.19150v1)正文提供[研究代码](https://github.com/yanivlivert/cahp)，但完整流程含微调，不符合我们的保护约束。这两项是相关工作核查，未进行完整复现。联合头有损表示只作[固定机制检验](head_collapse_protocol.md)，不从头冗余或截断 SVD 推出安全保证。

先明确后处理改变了哪项不可廉价删除的任务计算，再说明独立输入上的保持条件，以及融合或恢复为何破坏该条件。每项结论分别列假设和反例。对有代数删除规则且实测删除代价小的方向族停止调参；当前没有已证明可行的新构造。后续只对机制通过者运行官方测试、原版 AGA 和同条件恢复对照。

导师要求依据此前保存的[录音反馈整理](../2026-10-09/post_meeting/review.md)：任务子空间及补空间、保持原任务、检验融合干扰、核对评价目标。这份整理标明转写错词与例举维数；本轮没有找到原 Downloads 录音转写文件，不能声称重新逐字核对了原录音。抗替换要求来自用户后续明确指令，不归作导师逐字原话。

## 本轮新增核查

[Model Fusion via Retrofitting，ICML 2026](https://proceedings.mlr.press/v306/luenam26a.html)，[v2 正文](https://arxiv.org/html/2507.00037v2)提供[作者代码](https://github.com/AndrewSpano/model-fusion-via-retrofitting)。核对 §4、Theorem 2 与 §6：先匹配／聚类，再让当前层逼近目标表征，输入来自正在构建的融合模型。仿射层用闭式最小二乘；一般非线性层用 SGD，不能将后者称为无训练。定理要求先前层固定；表示匹配代价的保证不等于分类或安全保证。尚未运行该方法；攻击启示是顺序重建实际计算，而不是只恢复坐标。当前最终投影攻击不等于完整 Retrofitting。

[CASS，2026-09-28](https://arxiv.org/html/2609.34184v1)的 HTML 中仅发现 arXiv 页尾维护仓库，没有核实正文研究代码入口；不纳入本轮用户要求的精读清单。不能把页尾 GitHub 链接当作者代码。

再次查看 [Task Arithmetic 原始划分讨论](https://github.com/mlfoundations/task_vectors/issues/1)：作者确认 EuroSAT 数量，但未提供原始索引或随机种子；后续参与者的随机划分脚本不能补作原始划分。此复现阻塞仍在，暂不重复搜索同一讨论，也不生成新划分冒充论文协议。

[Q-realign，2026](https://arxiv.org/html/2601.08089v1)正文确实提供[作者代码](https://github.com/Skilteee/Q-Realign)，仓库标为 EMNLP 2026 Findings。核对 §2—3 与官方运行命令：它用于 LLM 微调后的安全对齐恢复，先训练冻结激活上的稀疏逻辑探针，再用重建与分离目标优化量化平滑／裁剪参数；代码示例明确运行 10 个 epoch。不能因为称为 post-training quantization，就把完整流程改称没有梯度优化；它也未评价未经授权的模型融合。不升级环境或将此方法直接当我们的保护机制。启示仅是需要明确哪些量被训练、在哪些输入上保持功能，当前未复现。

另外核查 MergeBarrier 与 MergeGuard 的 arXiv HTML 代码入口：尚未确认正文作者研究仓库，页尾维护链接不能算代码。保留已有相关工作范围核查，但本轮不将它们加入满足用户代码条件的精读成果。Sensitivity as a Shield 的直接 OpenReview 访问被验证页阻挡，未核实研究代码，不作方法成功或完整阅读的结论。

[MatGPTQ v2，2026-09-30](https://arxiv.org/html/2602.03537v2)正文提供[作者代码](https://github.com/IST-DASLab/MatGPTQ)。核对 §3.2 的多位宽格点目标与 OBS 误差补偿，以及固定源码的原 GPTQ 核心：单次顺序量化不使用梯度训练。它与 Q-realign 的学习型 PTQ 不能混为一谈；多精度格式、压缩精度和推理速度不提供抗融合保证。随后仅复用其中原 GPTQ 核心完成一次固定 CLIP 有损表示控制：独立性能小幅变化、真实融合没有下降，[原始结果与复核](quantization_probe/summary.md)。没有复现完整多精度方法，也没有新防御结论；入口的恢复反例与门槛见[机制检查](quantization_gate.md)。

[ESM，CVPR 2026 的 v2](https://arxiv.org/html/2602.20208v2)和[扩展 v3](https://arxiv.org/html/2606.19164v3)正文均提供[研究代码](https://github.com/kiddo127/ESM)。精读 §III-B、附录 A/B，并固定源码 fc82fe7，读取 3 文件共 15,232 字节。对输出增量 y=τx 的未中心化二阶矩取主方向 E，得到 τ̂=EEᵀτ；在同一分布二阶矩已知的条件下，平方输出误差等于被舍弃特征值之和。代理样本只给经验保证，不能推出测试分类保持或安全性。

源码对 activation_diff 做未中心化 SVD，特征值按 n−1 缩放。静态 ESM 的极分解合并与动态 ESM++ 的专家路由分别记录；QKV 钩子不等于完整 QK／VO 算子复现，out_proj 钩子在该实现中被注释。样本少于特征维数时零特征值补基不能自动视为正交。未安装入口依赖、未运行完整方法，[源码核查](esm_source_audit/source_audit.json)。主要启示是按实际响应衡量任务信息；低能尾部可被丢弃这一事实同时构成防御的恢复风险，不能将 ESD 本身当创新。

[Platonic Task Arithmetic v2，2026-10-02](https://arxiv.org/html/2610.00929v2)正文提供[研究代码](https://github.com/jhpark-ai/Platonic-Task-Arithmetic)。精读 §4、附录 C，固定源码 2011bad，读取 6 文件共 20,411 字节。描述符只观察图像表征变化在文本方向上的投影；主路线是双边岭回归并折入最终投影，另一个 LoRA 路线需要训练。固定源码实际均匀抽取类别数×30 张训练图，并非逐类配额；保留标签供诊断，求解不读标签。描述符残差不提供自动分类／校准外保证，不同任务算子相加也有真实伙伴代价。[源码核查](pta_source_audit/source_audit.json)；未运行完整作者方法、不下载新模型，不将本项目既有投影恢复改名为该论文复现。

[Theseus v3，2026-07-30](https://arxiv.org/html/2602.12952v3)正文提供[研究代码](https://github.com/apanariello4/merge-and-rebase)，ICML 2026 的[正式条目](https://proceedings.mlr.press/v306/rinaldi26a.html)可核对。阅读 §3、附录 A/B：以输入／输出激活的双线性效应和 Procrustes 共享子空间迁移更新；需要源基础模型、源专家和相应激活，不等于任意白盒发布都可保证恢复。异宽情形须区分嵌入与截断，截断不能保证任意更新零残差。HTML 中 Eq10/18 的转置方向与声明维数有冲突，使用前须核对原 PDF／源码；本轮尚未核对实现，也未运行完整方法。

Theseus 源码新增核查：commit563e3ee，6files143425bytes。`_shared.py` 实际权重运输为Tout.T @ delta @ Tin，维数正确，CLIP proj转置另处理。之前HTML不一致不能归咎实现，未核查PDF。两个矩阵 primitive 的 CPU形状／放宽功能签名检查通过；完整作者方法未运行，没安装依赖或下载checkpoint。见theseus_source_audit。

补充范围核查：[Merge++ v1，2026-09-19](https://arxiv.org/html/2609.22886v1)的正文HTML只发现页尾arXiv维护仓库，未核实作者研究代码入口。因此不计入用户要求的代码可复核精读成果，未运行或安装其方法。
