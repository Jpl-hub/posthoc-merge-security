# 理论记录：后处理保护与 AGA

日期：2026-10-09。以下为论文核对与独立推导，数值验证见 `results.csv`。核心参考：[论文 v1](https://arxiv.org/abs/2606.30360v1)，本地 PDF 附录第 19–21 页。

## 1. 最小二乘的投影结论

令 $W_{\rm ft},W_{\rm pre}\in\mathbb R^{N\times D}$，$P\in\mathbb R^{D\times D}$ 可逆，$W^p=W_{\rm ft}P$。攻击求解

$$
\min_{T\in\mathbb R^{D\times D}}\|W^pT-W_{\rm pre}\|_F^2.
$$

最小范数解是 $T^*=(W^p)^+W_{\rm pre}$。即使解不唯一，所有最小二乘解的拟合值相同：

$$
W^a=W^p(W^p)^+W_{\rm pre}
=\Pi_{\operatorname{col}(W_{\rm ft})}W_{\rm pre}.
$$

理由是右乘可逆矩阵保持列空间，$AA^+$ 是该列空间的正交投影。这个投影结论不要求满列秩；但论文使用的 $(A^TA)^{-1}A^T$ 公式要求满列秩，且必有 $N\ge D$。

**归属核对：上述核心结论及误差上界已由论文附录式 (24)–(28) 给出，不是我们的新定理。我们的工作是验证实现边界，并分析后处理设计的含义。**

设 $\tau=W_{\rm ft}-W_{\rm pre}$，则

$$
W^a-W_{\rm ft}=-\Pi_{\rm ft}\tau,\qquad
\|W^a-W_{\rm ft}\|_F\le\|\tau\|_F.
$$

因此理想攻击结果与保护矩阵 $P$ 无关，却不等于精确恢复 $W_{\rm ft}$。保留下来的 task vector 是 $(I-\Pi_{\rm ft})\tau$，位于拟合值允许的几何结构中。若 $W_{\rm ft}$ 是可逆方阵，$\Pi_{\rm ft}=I$，攻击直接得到 $W_{\rm pre}$，所有该矩阵上的任务变化均被消除。小参数误差也不自动意味着分类准确率保持；还需要网络敏感度和真实任务评价。

本结论只对攻击求解的同形状矩阵与正确 correspondence 成立。不能由整块方阵推出逐 head 攻击必定回到 pretrained：公开实现实际求解的是 $768\times64$ 的矩阵，分别对应 Q/K/V 的转置切片和 O 的列切片。

## 2. 数值实现的适用边界

作者 `solve_linear_transformation` 先转 float64，构造 $X^TX+10^{-8}I$，再对正规方程调用 `torch.linalg.lstsq`，最后把恢复矩阵转成 float32。它不是对原始 $X$ 直接求解的无正则最小二乘。

岭回归拟合值为

$$
X(X^TX+\lambda I)^{-1}X^TY.
$$

固定 $\lambda>0$ 时，它通常依赖 $P$；正规方程又将条件数平方。有限精度、秩阈值、float32 转换也会产生依赖。应以 CPU float64 `lstsq(..., driver='gelsd')` 的直接解作为理想参照，并单独报告正规方程及转换误差。不能把病态矩阵带来的数值失败宣称为安全性。

## 3. Attention 保护的矩阵方向

PyTorch Linear 存储 $W\in\mathbb R^{d_{out}\times d_{in}}$，前向为 $XW^T+b$。对每个 head，令输出特征 $Q'=QA$、$K'=KA^{-T}$、$V'=VB$。对应存储格式为

$$
W_Q'=A^TW_Q,\quad b_Q'=A^Tb_Q,
$$
$$
W_K'=A^{-1}W_K,\quad b_K'=A^{-1}b_K,
$$
$$
W_V'=B^TW_V,\quad b_V'=B^Tb_V,
\quad W_O'=W_OB^{-T}.
$$

于是 $Q'K'^T=QK^T$，且输出投影抵消 $B$。一般可逆 $A$ 必须使用逆转置配对；只有对角矩阵等特殊情形才可直接把 $A^{-T}$ 写成 $A^{-1}$。

MLP 的逐坐标 GELU/QuickGELU 与隐藏维置换可交换，故同步置换第一层行、bias、第二层列严格保持函数。一般隐藏层旋转不能穿过 GELU，不能套用 Attention 的任意可逆变换。

## 4. MLP 定理与 cosine 代码的差别

论文附录的 $\|\tau\|_F<\delta_{min}/2$ 唯一恢复证明使用未归一化的欧氏 Frobenius 目标。代码则按行归一化后使用负余弦匹配。仅当行归一化及扰动条件同时正确处理时，余弦距离与归一化行之间的欧氏距离才能相互转换；原始行范数不同的情况下，两种 assignment 不等价。

可构造原始范数间距大、方向间距小的两行权重，满足原始欧氏 margin 条件，但余弦 matching 选错。实验入口包含这一固定反例。正确推广需要对归一化行定义角度 margin，并约束扰动后的归一化误差，不能直接沿用原始权重的 $\delta_{min}$。

## 5. ARF 印刷公式检查

本地 PDF 第 13 页已视觉确认，式 (19) 写成 $\min(0,\rho\|\theta_{pre}\|_2-\|\theta-\theta_{pre}\|_2)$，并以正号加入最小化目标。这一表达式在距离小于 margin 时为零、没有排斥梯度；距离超过 margin 后持续奖励距离增大，与正文“未到 margin 时排斥、达到后停止”的描述不一致。

符合该段文字的常见 margin penalty 应为 $\max(0,\rho\|\theta_{pre}\|_2-\|\theta-\theta_{pre}\|_2)$。这里只指出印刷公式与文字不一致；公开仓库未找到 ARF 训练实现，无法确认作者真实训练究竟使用哪个符号，也不能据此推断其论文实验无效。

## 6. 对本研究方向的约束

在正确对应的同一 head 内扩大可逆变换的幅度、换成稠密矩阵、叠加多轮变换，都不能改变理想 LS 拟合的列空间投影。这些操作可以破坏直接 Task Arithmetic，但无法由上述机制带来对理想 AGA 的保护增益。

跨 head 置换改变公开代码的对应关系，值得作为实现边界检查；仍需加入自适应 head matching。投影子空间 affinity 可以抵消 head 内基变换，再用 Hungarian 恢复对应关系。因此单纯打乱 head 不应被描述为强防御。

后续研究应首先区分：严格函数等价且架构不变的对称变换、允许极小效用下降的参数扰动、需要改变部署或访问方式的保护。前者常可被 canonicalization 消去；后两者需新的约束和可观测评价，不能仅凭参数距离大来宣称成功。
