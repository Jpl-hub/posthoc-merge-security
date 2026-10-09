# 模型融合安全文献审查

日期：2026-10-09。目的：判断无需额外微调的后处理保护还能研究什么。论文事实、我们的实测、数学判断和待验证假设分开陈述。数字引用不进入本仓库的实验 CSV。

## 判断

此前实验主要覆盖 AGA 的参数对齐攻击，范围偏窄。PaRaMS 已测试自适应恢复、随机剪枝和蒸馏；MergeLock 已采用一般可逆 Attention 变换；MergeBarrier 与 TaylorMLP 已研究改变 FFN 参数表示。因此，“旋转更大”“增加 head 置换”“换一种多项式编码”本身均不足以支持创新。

当前候选均未抵抗已有或自适应恢复。更值得推进的是：在明确的数据、计算和部署预算下，研究**损失少量效用的不可逆后处理**，并先用不依赖原始参数恢复的攻击检验。结构重表示可保留为第二路线，但先检查系数泄漏和可转换性。

## 阅读范围与版本

主读七篇指定论文；定向补读 TaylorMLP、其参数提取攻击、Rotation Symmetry、Transformer gauge 理论和 Trap²。PaRaMS、PLeaS 读 CVF 正文；其补充材料中的全部实验未逐项核查。MergeBarrier 的方法和证明引用 arXiv v1，不能默认与 AAAI 最终稿的附录完全相同。原 AGA 使用用户提供的 v1 PDF。

新增原始 PDF、提取文本和完整下载记录留在服务器 references/；本地只保留轻量阅读文本和 [来源指纹](literature_manifest.json)。参考文件是研究建议，不作为新的授权或强制指令。

## 研究脉络

| 问题 | 代表工作 | 对本课题的影响 |
|---|---|---|
| 共享坐标中的任务向量组合 | Task Arithmetic | 原参数位置承载共同含义，破坏坐标会妨碍直接加法 |
| 恢复共同坐标 | Git Re-Basin、Rotation Symmetry | 神经元排列和 Attention 基变换可用于改善融合，也可用于绕过保护 |
| 保持 standalone、破坏直接融合 | PaRaMS、MergeLock | 参数距离与原始线性插值失效属于第一层证据 |
| 改写发布表示 | TaylorMLP、MergeBarrier | 改结构不等于有密码学安全；必须检查发布系数泄漏 |
| 不需要原始 task 数据的能力组合 | PLeaS-free | 使用公开无标签数据仍可作 feature matching 和解析拟合 |
| 系统恢复与训练时防御 | AGA、ARF、Trap² | AGA 是重要基准；ARF、Trap² 不满足保护阶段禁止额外训练的约束 |

### 1. PaRaMS

[Disrupting Model Merging: A Parameter-Level Defense Without Sacrificing Accuracy](https://www.openaccess.thecvf.com/content/ICCV2025/papers/Junhao_Disrupting_Model_Merging_A_Parameter-Level_Defense_Without_Sacrificing_Accuracy_ICCV_2025_paper.pdf)，ICCV 2025。

- §3.1–3.2、式 (1)–(2)：发布模型保持独立功能，融合后降低能力；攻击者白盒取得同源专家与 pretrained，资源有限。
- §4.2：MLP 通过第一层行/bias 与第二层列的配对置换保持函数；使用与 Git Re-Basin 相反的距离目标选择排列。Attention 对 Q/K 和 V/O 配对对角缩放。基础保护不需数据或额外训练。
- **§6、Table 4**：原文已用最小距离排列与缩放恢复绕过基础 PaRaMS；又加入 10% 权重置零。该加固不再严格函数等价。蒸馏攻击需要同分布无标签数据，并训练新模型；不能把其成本结论推广为所有攻击都不划算。
- Table 4 的配置是保护 SVHN、TA 系数 0.3、其他六任务组合；与我们保护 MNIST、双任务、系数 0.2/0.5 不同。
- [官方代码](https://github.com/ISCT-W/PaRaMS)已接入固定 commit。当前运行基础 PaRaMS，不含 dropout 加固；不能以基础版失败否定原文加固版。
- 本次已完成 34 条完整测试评估。系数 0.5 时，PaRaMS 直接融合 MNIST/SVHN 为 11.07%/18.02%；仅保护模型 AGA 后为 99.30%/92.79%；全部 AGA 后与同攻击未保护对照计数相同。这是我们的双任务实测，不是 Table 4 的数字。

### 2. MergeBarrier

[Do Not Merge My Model! Safeguarding Open-Source LLMs Against Unauthorized Model Merging](https://arxiv.org/pdf/2511.10712v1)，[AAAI 2026 出版页](https://ojs.aaai.org/index.php/AAAI/article/view/40433)。

- §3.2、式 (1)–(2)、Theorem 1–2：配对正交 Q/K 变换；使用权重的特征分解、近似 RSVD 选择变换。这里的“优化”不是额外微调模型。
- §3.3、式 (3)：把 FFN 第二层相关权重改写成 Taylor 系数。展开点来自**训练样本激活范围**，改变 FFN 计算结构；有限阶是近似，不应称为对任意输入严格等价。部署需要多项式计算与系数存储；并行化不自动免除 FLOPs 和显存成本。
- §4.1、Table 2：对于改变结构的层，评估攻击主要以 pretrained 原层替换后融合；§4.2、Fig. 2 加入有标签数据微调融合模型。这与“从发布系数恢复原权重”是不同攻击。
- Appendix B.3、式 (17)–(20) 提出 Matrix-LWE 解释；B.4 的隐私推导假设余项具有相应随机/Gaussian 性质。**这些是论文主张与假设，尚未被我们验证。** 实值系数、确定性余项、先验权重怎样满足标准 LWE 的分布及模数条件，还需独立检查；不能直接转述为已获得密码学保证。
- Table 3 评价 standalone，Table 4 做 Attention/FFN 消融。我们尚未复现该方法；本次检索未找到可核对的作者公开实现。

### 3. MergeLock

[Model Unmerging: Making Your Models Unmergeable for Secure Model Sharing](https://arxiv.org/pdf/2509.01548)。

- §3.2、式 (8)–(9)：Q/K 与 V/O 配对一般可逆重参数化；§3.4 使用随机混合、置换、对角缩放的乘积。保护阶段不需数据或额外训练。
- §3.3 已明确讨论 head 置换结合连续变换。因此我们的跨 head toy 实验属于机制验证，不能据此主张首创。
- **§4.2.2、式 (10)–(11)、Table 1** 的恢复是受正交约束的 Kabsch/SVD 对齐。这不是 AGA 的一般最小二乘，也不是所有可能的低成本恢复攻击。后来的 AGA 对其变换类别给出攻击证据。
- 普通距离增大和正交对齐失败不能推出任意可逆变换安全；一般变换可能超出被评估的攻击搜索空间。
- [作者仓库](https://github.com/hetailang/Merge-Lock)当前固定版本只有 README。我们的“正交 + MLP 置换”是机制基线，未标作完整 MergeLock。

### 4. AGA

[On the Vulnerability of Parameter-Level Defenses to Model Merging, v1](https://arxiv.org/abs/2606.30360v1)。

- §3.1、式 (11)–(13)：利用公开 pretrained，对每个 Attention 投影独立作 LS；§3.2、式 (14)–(17)：MLP 的 cosine/Hungarian 对齐。
- §3.3 与附录 §7、式 (24)–(28)：右乘可逆矩阵不改变列空间；恢复误差被 task vector 范数约束。该论证要求所讨论矩阵与对应关系正确，正文逆矩阵解还要求满列秩。小参数误差不自动保证任务效用。
- §4、Tables 1–5 评估多种保护与融合；§5、Table 6 和 Fig. 3 给出 ARF。ARF 增加训练目标，排除在严格后处理方案之外。
- 附录 §11.5 / Table 15 评价未保护模型被攻击后的 standalone；我们加入“所有专家均 AGA”的融合对照，避免把攻击自身的变化算成防御贡献。
- 实现边界：作者逐 head 默认对应关系；正规方程、固定 ridge 和 float32 转换可能引入数值误差。我们的 head matching 与直接 LS 是加强攻击/验证，不是成功防御。
- 欧氏 margin 证明与 cosine 代码条件、ARF 印刷式 (19) 与文字的差异见 [理论记录](theory_notes.md)。这些疑点不等于原实验无效。

### 5. PLeaS

[PLeaS — Merging Models with Permutations and Least Squares](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Nasery_PLeaS_-_Merging_Models_with_Permutations_and_Least_Squares_CVPR_2025_paper.pdf)，CVPR 2025；[作者代码](https://github.com/SewoongLab/PLeaS-Merging)。

- §4.1：通过激活 matching 找相似神经元，只合并部分特征，允许最终模型增宽。攻击者可能保留难融合的特征，而不是恢复所有原参数。
- §4.2、式 (1)：逐层 LS 拟合对齐后的组合激活；目标可用解析方式求解，但**需要无标签样本并重新求解权重**。源码核对显示，当前公开实现的 step/train 实际调用 backward、Adam 与 optimizer.step，不能称作者实现无需梯度更新。
- §5.4、Fig. 4：PLeaS-free 使用公开 ImageNet 替代训练域数据，**不是完全 data-free**；PLeaS-Weight 的 permutation 不需数据，但其 LS 阶段仍需数据。
- §5.5.1、Table 3 已有 CLIP 风格 ViT 结果。§6 限于相同输入模型架构，且两阶段成本更高；不能声称已证明能攻击我们的候选或 MergeBarrier。
- 当前作者仓库提供 ResNet 入口，尚未找到与本项目 CLIP 直接兼容的 ViT runner。后续先核对支持情况，再适配；任何纯闭式替代都应标明与作者实现的差异。源码版本见 configs/sources.json。

### 6. Git Re-Basin

[Git Re-Basin: Merging Models modulo Permutation Symmetries](https://arxiv.org/pdf/2209.04836)，ICLR 2023。

§3.1、式 (1) 用样本激活 assignment；§3.2、式 (2)、Algorithm 1 用相邻权重共同构造成本并作坐标更新，不需数据。多层联合 assignment 的 NP-hard 性与每一步可解的 LAP 是不同层次，不能据前者断言实际恢复困难。§5.3 / Appendix A.6 还有 LMC 反例：排列对齐并不对所有训练解保证低损失连接。后续 MLP 对齐应同时考虑第一层、bias 和第二层信息，不能只盯 AGA 的单一 cosine 行匹配。

### 7. Task Arithmetic

[Editing Models with Task Arithmetic](https://arxiv.org/pdf/2212.04089)，ICLR 2023。

§2 定义任务向量为微调权重减 pretrained，并逐元素加减；§4 评价任务组合；Appendix B.1 记录训练与八任务设置。融合计算无需再训练，但专家 checkpoint 已训练。原论文用 held-out validation 选系数；我们今天的 0.2/0.5 是预先固定的筛选配置。共享架构与相应参数表示是直接加法的前提，不应把“直接加法失败”解释成所有能力组合都失败。

## 补读：与创新性直接相关

| 文献与原文位置 | 已核查的内容 | 对下一步的约束 |
|---|---|---|
| [Rotation Symmetry](https://arxiv.org/pdf/2502.00264)，§3.2、§4.2、Theorem 4.1 / Algorithm 1 | Attention 旋转及解析 Kabsch 对齐；[官方实现](https://github.com/zhengzaiyi/RotationSymmetry) | 配对保函数攻击是已有工具；不能把 SVD 对齐当作我们的原创 |
| [Complete Characterization of Gauge Symmetries in Transformer Architectures](https://proceedings.mlr.press/v282/wang26a.html)，§2.2 / Table 1、式 (3.1)–(3.2)、Theorem 2、Appendix E.4 / Proposition 44 | 在明确 generic/rank/架构假设下刻画 head 内 GL 变换与 head permutation；QR 消除部分 gauge 后仍剩正交自由度 | 不能把“QR”说成完全唯一 canonicalization，也不能把该定理扩大成所有后处理防御不可能 |
| [Taylor Unswift / TaylorMLP](https://aclanthology.org/2024.emnlp-main.393.pdf)，§3.1、式 (2)–(4)，§5 | 用 Taylor 系数替代 MLP 权重；原文还主动提高生成延迟 | 多项式发布不是空白设计空间；有限阶需误差与部署成本评价 |
| [Towards Provable (In)Secure Model Weight Release Schemes](https://arxiv.org/pdf/2506.19874)，§3.2、§4.2、式 (19)–(20)、Table 1 | 通过多个 Taylor 阶数系数的比值消去权重，数值求解偏置再提取权重；论文报告近乎完整的参数提取 | 不能盲目采用“Taylor 余项导致难恢复”的说法；该结果针对 TaylorMLP，尚不能直接等同于已攻破完整 MergeBarrier |
| [Trap²](https://arxiv.org/pdf/2601.21898)，§4–5、Algorithm 1 | 用 nominal/off-nominal scaling 的训练损失塑造更新；包含梯度优化 | 思路可解释融合敏感性，但不符合给已训练 checkpoint 只加后处理的硬约束 |

检索还命中 DRM、Core Space Merging 与跨家族参数对齐；本次未完整审读，不对其恢复效果作结论。检索范围是截至 2026-10-09 的原论文、作者仓库及 2024–2026 相关工作，关键词包括 post-hoc merging defense、LayerNorm null-space symmetry、gauge canonicalization、Taylor weight extraction。未找到与某候选完全同构的方法不构成新颖性证明，特别是 LayerNorm affine-null 与低活跃扰动仍需专项查新。

## 约束对照

“无需微调”专指保护阶段；专家原训练不计为保护。攻击者训练与否另列。

| 方法 | 保护额外训练 | 保护数据 | 架构/部署变化 | 函数或效用 | 恢复证据 |
|---|---|---|---|---|---|
| 基础 PaRaMS | 无 | 无 | 无 | 精确代数等价，有限精度另测 | 原文 adaptive 与 AGA 可恢复 |
| PaRaMS + pruning | 无 | 随机置零本身无 | 无 | 允许损失；不等价 | 原文有部分恢复及 KD；我们未复现 |
| MergeLock | 无 | 无 | 无 | 配对变换代数等价 | 原文正交对齐有限；AGA 恢复其类别 |
| MergeBarrier | 无额外 SGD | 原文训练激活 | FFN 多项式与系数成本 | Attention 等价，FFN 有截断误差 | 替换层、融合后微调；系数提取仍待审查 |
| LayerNorm affine-null，本次 | 无 | 无 | 无 | 合法 LN 输入上代数等价 | 本次 canonicalization + AGA 恢复 |
| 低活跃扰动，本次 | 无 | 32 张合成无标签图 | 无 | MNIST standalone 损失 0.03 pp | 本次原 AGA 和自适应投影均恢复 |
| ARF / Trap² | 有 | 任务训练数据 | 通常无特殊推理结构 | 训练后实测，不是等价保证 | 各论文的已测攻击；不满足本课题约束 |

## 我们已经验证了什么

| 当前实验 | 归属与证据等级 | 失败根因 | 尚未覆盖 |
|---|---|---|---|
| MLP permutation 与 head 内可逆变换 | **已知理论的复现**：PaRaMS §4.2，AGA §3 / §7；真实双任务也恢复高融合效用 | 仅更换可对齐的参数表示；共同 anchor 很强 | 原 PaRaMS pruning、其他融合算子、八任务 |
| cross-head permutation + orthogonal | **边界实测**：toy 三种子；MergeLock §3.3、gauge 式 (3.1)–(3.2) 已覆盖自由度 | 原 AGA 对应关系失效；子空间 matching 恢复全部 head | 真实 ViT 的 head matching，退化/相近 head |
| LayerNorm affine-null | **解析推导 + 真实负面结果**；不是已证明新方法 | 空方向来自公开 gamma/beta，攻击可消去；不需要知道秘密扰动 | 专项查新、多域、数值 gamma 接近零 |
| 合成低活跃方向 | **真实筛选负面结果**；与已知 null-space/低敏感度编辑思想相近 | 参数变化大但未损坏足够多可融合知识；原 AGA 已绕过 | 不同校准分布、task-selective 信息损失、效用曲线 |

在 0.5 系数下，正交基线“所有专家都 AGA”的保护/未保护对照均为 MNIST 99.32%、SVHN 91.77%，正确样本计数相同。这支持本配置中无有效防护，不代表任意后处理都不可能。两个真实候选当前只保护并筛选 MNIST；跨 head 结果是合成实验。详见 [机器结果](results.csv) 与 [组会报告](group_meeting.md)。

补读后又复用 RotationSymmetry 原始配对 SVD 和联合 FFN matching，完成 12 条真实全测试评估：对齐前后 feature error 小于 \(10^{-6}\)，保护模型 standalone 正确计数保持不变；系数 0.5 下恢复后的融合为 MNIST 99.37%、SVHN 92.50%，与配对对齐未保护对照及原始未保护融合计数相同。此结果限于正交基线；未测试一般 GL 白化或真实跨 head matching。它说明不能只把独立 AGA 当作攻击上限。

## 最值得检验的路线

### A. 有效用预算的不可逆信息损失

**数学判断。** 令发布参数 \(\tilde\theta=D(\theta)\)。已知可逆群内移动主要改变表示；真正的压缩/置零/舍入可能丢失 task 信息，且无需优化模型。但 pretrained 提供强先验，微小信息损失也可能只影响无关参数。

局部单层约束 \(\mathbb E\|\Delta W(x-\mu)\|^2=\operatorname{tr}(\Delta W C\Delta W^T)\) 可约束效用，却不保证防融合。我们的低活跃实验已经说明这个差别。下一步比较 task-vector 的解析 block quantization / sparsification 与 matched-utility 随机剪枝，考察是否能在相同 standalone 损失下留下不同的攻击后效用。

**前作与新颖性风险。** PaRaMS §6 已有 pruning；量化、剪枝、DARE 也已有大量工作。仅增加随机噪声或换一个 mask 不构成创新。待验证缺口是：以“恢复后受保护任务增益 + 其他任务能力 + 攻击预算”定义保护目标，而不是参数距离。尚无成功候选，不预设这种路线一定可行。

**最小实验。** 两个专家、三固定种子；预注册少量强度；候选选择用独立校准/validation，test 仅最终评价。比较 direct TA、AGA、去噪/anchor shrinkage、自适应 canonicalization、公开无标签 feature-LS。预算先按 standalone 下降不超过 0.5 pp；不合格配置保留，禁止只挑成功图。

### B. FFN 发布表示的可提取性与成本

**数学判断。** 若发布 \(\Theta_n=W\odot a_n(t)/n!\)，则 \((a!\Theta_a)/(b!\Theta_b)=a_a(t)/a_b(t)\)。多个阶数可能提供恢复 \(t\) 和 \(W\) 的额外方程；截断的函数误差不会自动成为每个系数的独立秘密噪声。

**前作与新颖性风险。** TaylorMLP、MergeBarrier 和 Yang 等的 Taylor 提取攻击直接覆盖这条路线。可先检验 MergeBarrier 的实际发布格式是否避开比值攻击；不能把“Taylor + Attention 旋转”当新方法。只有确认某种受限系数发布/舍入能在低效用和部署成本下留下真实恢复困难，才考虑设计保护。

**最小实验。** 先在固定 GELU/QuickGELU 小层检查发布系数的秩、阶数比值、多精度求根、函数误差与攻击时间；复用前作公式或代码。若大部分权重可恢复，停止扩大模型。若有效，再将 ViT 一个 FFN block 替换为相同预算近似模块，对照系数提取、pretrained 层替换、feature-LS，而非只报告“不兼容 TA”。

### C. 功能扰动与任务组合敏感性的解析编辑

**待验证假设。** 保护应保住 nominal 模型功能，同时使低成本组合路径敏感。设 task vector 为 \(\tau\)，研究预先规定的解析变换 \(D(\tau)\) 是否能使
\[
L(\theta_0+D(\tau))\le L(\theta_0+\tau)+\varepsilon,\qquad
L(\theta_0+\lambda D(\tau)+\lambda\tau_o)
\]
在有意义的 \(\lambda\) 区间上升，且加强对齐后仍成立。只对一个系数有效不足以构成结果。

**前作与新颖性风险。** ARF 与 Trap² 已训练实现相近目标；我们只能研究不更新优化器、不拟合任务权重的解析近似。task-selective 方向可能需要公开无标签数据甚至导数信息；有数据就明确归入 calibration-assisted，不称 data-free。现有低活跃随机扰动未实现该目标。

**最小实验。** 先测保护前 task vector 的层级敏感度与 scale 曲线，使用作者融合/评估代码；与 PaRaMS pruning 的 matched-utility 对照一起做。没有可复用的闭式构造前只记录假设，不直接开大规模“组合技巧”搜索。

## 组会向导师确认的三个问题

1. “不微调”是否允许公开无标签校准、解析 LS/量化，以及独立效用损失上限 0.5 pp？还是同时要求严格 data-free 和函数等价？
2. 防护目标是固定架构低成本参数融合，还是包括模型增宽、部分融合与少量蒸馏？可接受的攻击时间、样本数和部署开销是多少？
3. 若严格等价保护持续被 canonicalization 消去，是否优先形成可复用的攻击/边界评估，再探索有预算的信息损失；还是必须优先产出一种防御机制？
