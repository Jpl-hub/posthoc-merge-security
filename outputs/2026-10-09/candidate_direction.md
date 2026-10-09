# 具体方向：删去可恢复子空间，或改写完整 FFN 功能表示

日期：2026-10-09。这是经过文献核查的研究假设与两项真实筛选，**不是已证明新颖或有效的防御**。原微调专家不使用梯度或优化器继续训练；结构候选需要公开合成校准，并对新映射作闭式拟合，不能称 data-free 或“完全不拟合参数”。

## 要回答的问题

可逆保护仍位于同一参数子空间，AGA 可对齐；结构保护即使隐藏原参数，攻击者也可能拟合另一种有用表示。我们需要知道：**在独立效用损失不超过 0.5 pp 的预设预算内，后处理能否删除共同 pretrained 无法廉价补回、功能拟合也难恢复的任务信息？**

## 前作与具体差别

- PaRaMS §6 已有随机剪枝；本次随机 V 权重置零是已有思路的对照，不是新机制。
- [TSV-C §4.1 式 (7)](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Gargiulo_Task_Singular_Vectors_Reducing_Task_Interference_in_Model_Merging_CVPR_2025_paper.pdf) 已做 task delta 低秩压缩，甚至改善融合。本次删的是 V 的完整仿射算子方向，并检查 pretrained 补全；压缩操作本身不新。
- [SVD-LLM v5 §3.1 / Theorem 3.2](https://arxiv.org/abs/2403.07378v5) 已研究激活加权低秩误差。本次输出空间投影使用等价的谱近似依据，避免在奇异校准协方差上求逆；不复现其 LoRA 后续更新。
- [MergeBarrier §3.3 式 (3)](https://arxiv.org/pdf/2511.10712v1) 改写第二线性层的逐神经元 Taylor 系数；TaylorMLP 亦有前作。本次小原型直接汇总整个 FFN 为低维二次映射，不保留该块 W1/W2。因此逐神经元系数比值提取不直接适用于此参数格式，但仍可能被功能拟合、张量分解或共享格式融合绕过。
- [PLeaS §4.2](https://www.openaccess.thecvf.com/content/CVPR2025/papers/Nasery_PLeaS_-_Merging_Models_with_Permutations_and_Least_Squares_CVPR_2025_paper.pdf) 已有 feature-LS。我们的闭式原架构重构是基于同类目标的薄适配，不是 PLeaS 原公开 Adam 实现的复现，也不把 LS 当创新。

潜在缺口是上述条件下的**功能恢复与参数表示之间的差别及成本**。本轮有限检索不足以认定该问题未被研究，也尚未发现有效保护构造。

## A：不可逆 Attention 秩收缩

单 head 令仿射 V 参数为 \(B=[W_V,b_V]\in\mathbb R^{64\times769}\)，校准输入为 \(\bar X=[X,1]\)。取 \(Y=\bar X B^T\)，以及 \(Y^TY\) 最大的 r 个特征向量 \(U_r\)。发布

\[
\tilde B=U_rU_r^T B.
\]

**已知近似定理的应用。** \(\tilde B\) 的秩不超过 r；在校准输入上达到 rank-r 输出近似，平方误差等于被丢弃的输出谱能量。这个局部结论不保证全网络测试性能，也不保证融合失败。

**AGA 的局部边界。** 若 \(A=\tilde B^T\)、\(A_0=B_{pre}^T\)，理想 LS 拟合为 \(AA^+A_0\)。对任意 rank-r 列空间，残差平方至少为 \(\sum_{i>r}\sigma_i(A_0)^2\)。此为线性代数下界，不是分类准确率或安全下界。作者实现的 weight/bias 分开拟合与理想仿射目标有差别，真实实验仍调用作者代码。

**自适应反例检查。** 攻击者从发布的 \(\tilde B\) 估计输出投影 \(P\)，构造

\[
B^{rec}=\tilde B+(I-P)B_{pre}.
\]

它不需要秘密投影、随机种子或数据。若删掉的主要是 pretrained 已有的公共功能，补全可能恢复融合能力；这正是必须排除的失败原因。

四个配置提前写入 configs/rank_probe.json：最后四层的激活 rank 32/16、无数据 weight rank 32、10% 随机 V 权重置零。只保护 MNIST 专家；第二专家是原始 SVHN。三个投影配置逐 head 验证实际数值秩。每个配置都做全测试 standalone、直接 TA、原 AGA、作者配对对齐、pretrained 补全，并使用相同攻击处理的未保护对照。随机剪枝是对照，不是完全复现 PaRaMS §6 的全参数设置。

## B：完整 FFN 的二次功能重表示

只替换 ViT 第 12 个 FFN。用 pretrained 的 32 张合成笔画激活得到固定中心、PCA 基和尺度，令 \(z=S^{-1}U^T(x-\mu)\)。构造包含常数、线性项及全部二次项的 \(\phi(z)\in\mathbb R^{561}\)，用同分布 32 张合成图像的原 FFN 输出做闭式 ridge：

\[
C=(\Phi^T\Phi+\lambda I)^{-1}\Phi^TF(X),\qquad
\hat F(x)=\phi(z)^TC.
\]

发布块含中心、基、尺度和汇总系数；不含该块原 W1/W2。采样、rank=32、degree=2、ridge、性能预算及系数均预设，不在 test 上选择。拟合误差只控制校准集上的块输出；必须单独检查完整测试和部署存储，不能声称严格函数等价。

**两个必须测的绕过。** 第一，把 pretrained 与攻击者自己的 SVHN 模型转换到同一个公共表示，再直接作作者 TaskVector 加法；结构不兼容不计作安全。第二，使用另一批 32 张公开合成图像，固定 pretrained 的 W1，解析拟合 W2 回原架构。令 \(H=[\mathrm{GELU}(W_{1,pre}X+b_{1,pre}),1]\)、\(T\) 为发布块的输出，\(D_0\) 为 pretrained 第二层，解

\[
D^*=D_0+H^T(HH^T+\lambda I)^{-1}(T-HD_0).
\]

该攻击只接收发布模型、pretrained 和公开样本，不接收原专家权重；同样攻击原始未保护模型作对照。无需精确恢复原 W1/W2，只要生成可融合功能表示就算绕过。

AGA 只作用于仍兼容的 12 个 Attention 和 11 个原 FFN，明确跳过替换块；这是适配攻击，不伪称 AGA 原论文能直接处理新架构。本轮不把改格式与精确保函数的置换旋转拼成一个“新方法”。

## 实测

两项实验运行后由机器结果生成表格。未完成的数据不计入结论。
