# 输入补空间后处理

**已验证：MNIST 独立准确率不变，原版 AGA 后伙伴 SVHN 降至 9.38%。公开投影清理能消除干扰，因此还不是可靠防御。**

这次按[老师的要求](teacher_feedback.md)，保留当前任务计算，只在补空间加扰动。只改 patch embedding 的 `conv1.weight`，不改中间层，不微调。构造不读取伙伴数据或权重。

## 怎么做

MNIST 预处理将灰度值 \(g\) 复制为 RGB，再按公开的均值、标准差归一化。每个像素的输入为

\[
x=ag-b,\qquad a=1/\sigma,\quad b=\mu/\sigma.
\]

取单位向量 \(v\)，使 \(v^\top a=v^\top b=0\)，令 \(C=vv^\top\)。于是所有这种灰度输入都满足 \(Cx=0\)。对卷积核的每个空间位置，沿 RGB 维加入

\[
W'=W+\Delta W,\qquad \Delta W=\alpha RC,
\quad \alpha=\frac{100\|WC\|_F}{\|RC\|_F}.
\]

\(R\) 为固定种子 20261012 的高斯噪声。只算一次，不扫幅度、秩或种子。这个范数与此前 \(100WC\) 的放大对照相同，用来比较放大已有颜色特征和加入随机颜色特征的差别。

因为 \(\Delta Wx=0\)，理想精度下 MNIST 的 patch 计算不变。彩色输入不一定满足这个约束，可能受到干扰；最终分类效果必须实测。这里利用的是输入端的准确约束，不是老师举例中的“16 维”估计。每个 32×32 patch 保留一个 2048 维线性空间，它包含全部预处理后的灰度输入。

## 真实结果

CLIP ViT-B/32，MNIST+SVHN，两任务 Task Arithmetic 系数 0.5。AGA 使用固定版本作者代码，对双方全部 12 层先恢复 Attention，再恢复 MLP。完整测试：MNIST 10,000 张，SVHN 26,032 张。

| 测试 | 未保护 | 补空间噪声 |
|---|---:|---:|
| MNIST 独立 | 99.69% | **99.69%** |
| 直接融合，MNIST | 99.37% | 99.37% |
| 直接融合，SVHN | 92.50% | **9.93%** |
| AGA 后融合，MNIST | 99.32% | 99.32% |
| AGA 后融合，SVHN | 91.77% | **9.38%** |
| 公开投影清理再 AGA，SVHN | 91.77% | 91.77% |

全部 18 项计数已从服务器原始预测、官方标签独立重算，并核对作者函数、代码与权重指纹。MNIST 独立预测逐样本相同。256 张无标签 MNIST 训练图仅用于检查数值保持条件，没有拟合；测试集不用于选择方向或幅度。此前 MNIST/SVHN 测试结果已经看过，不能称盲测。

[原始计数](../results/input_complement/noise/results.json)、[独立复核](../results/input_complement/noise/independent_validation.json)、[协议](../results/input_complement/noise/protocol.json)。冻结源码 `e211a1e`；编辑包 SHA-256 为 `4d49665f558501478cbedeb4885baeff3a384b06b5d05091eb789738af632e7c`。重文件只在 3090。

同范数的 \(100WC\) 放大对照，AGA 后 SVHN 为 58.39%；固定噪声为 9.38%。这是本次控制实验的差别，不是创新性证明。[放大对照](../results/input_complement/amplification/results.json)。

## 为什么仍不能称可靠防御

原版 AGA 不修改 `conv1`，因此这里的噪声未被它处理。结果说明该实现的覆盖范围有缺口，不能推断更广泛的恢复攻击无效。[固定版本源码](https://github.com/krumpguo/secure-merge-attack/blob/fb61429f5e6f3517102d506697a150ac26abbdea/utils_attack_B32.py)。

攻击者只需公开预处理和预训练权重 \(W_0\)，便能做

\[
F(W')=W'-(W'-W_0)C.
\]

因为 \(\Delta WC=\Delta W\)，精确计算下 \(F(W+\Delta W)=F(W)\)。实测清理后双方参数最大差为 \(1.51\times10^{-7}\)，融合预测逐样本一致。这次没有解决公开清理问题。

另一个边界：灰度输入在融合后仍满足 \(Cx=0\)，所以本构造不会破坏 MNIST 的融合性能。受损的是伙伴任务。它不能替代 [AGA 论文](https://arxiv.org/html/2606.30360v1)中保护任务融合准确率下降的主指标，也不是 Table 6 复现。对其他灰度伙伴无此干扰保证，其他目标专家尚未验证。

下一步只复用这份冻结编辑包验证另一个彩色伙伴 DTD，不重新选择参数。DTD 数据已齐，checkpoint 下载尚未完成，没有 DTD 结果。

## 已停止的方向

固定 Q/K 补空间候选：独立 MNIST 保持，直接融合有干扰；AGA 后 MNIST 99.32→99.28%、SVHN 91.77→91.89%，干扰消失，已停止。[记录](../results/qk_complement/results.json)。旧补空间与内部压缩证据保留在 Git 和服务器归档，不再增加运行入口。内部压缩不作为老师“只改补空间”的成果。

[NUFILT](https://arxiv.org/html/2509.21413v2)提供了子空间分析的依据，但其完整方法包含 LoRA 优化；本项目没有完整复现它，也不能把它当免训练防御的证明。
