# 2026-10-09 组会：无微调的模型融合保护

## 研究问题

对已经训练好的模型增加后处理保护：不再微调，尽量不使用原始训练数据，保持 standalone utility，阻止未经授权融合，并抵抗恢复攻击。

今天没有执行模型训练。完成 56 条机制验证、22 条真实模型评估、16 条 LayerNorm 候选评估。尚未得到可抵抗自适应攻击的强防御。

## 论文与实现

AGA 用 pretrained 权重作 anchor：Attention 对 Q/K/V/O 分别逐 head 最小二乘对齐；MLP 用 cosine + Hungarian 恢复隐藏神经元顺序。ARF 在微调时扩大 Attention 与 anchor 的距离，因此不满足本课题的“仅后处理”。

作者公开代码未包含完整保护、ARF 训练和 checkpoint。今天使用 Task Arithmetic 官方 B/32 checkpoint，自行实现函数等价保护，直接调用固定版本的作者 AGA 函数。属于独立复现，不是论文八任务表格的原样复现。

## 已完成：机制验证

3 个固定种子；GELU MLP 宽度 48、隐藏维 96；Attention 宽度 96、12 heads。参数变化是合成扰动，不是下游训练。

| 机制；合成 task change = 1% | 保护输出误差 | AGA 参数误差 | AGA 输出误差 |
|---|---|---|---|
| MLP 置换 | 1.32e-07 | 0 | 0 |
| Attention QK/VO | 3.17e-07 | 2.86e-03 | 5.05e-03 |

MLP 的 15 个测试配置均 100% 恢复真实置换；恢复后的参数误差为零。Attention 在不同扰动幅度下仍保持保护函数等价，但 AGA 独立估计投影会留下恢复误差。

![机制验证](figures/mechanisms.png)

## 已完成：真实 CLIP 对照

MNIST test：10,000；SVHN test：26,032。保护对象是 MNIST 微调模型，第二专家是 SVHN。两个融合系数预先固定，未使用测试集选择系数。测试仅评价，没有校准。

| 组别 | 系数 | MNIST | SVHN |
|---|---|---|---|
| A：未保护 MNIST 模型 | — | 99.69% | 49.23% |
| B：后处理保护 | — | 99.69% | 49.23% |
| C：B 经 AGA 恢复 | — | 99.68% | 49.08% |
| D：未保护融合 | 0.2 | 95.29% | 77.31% |
| E：保护后直接融合 | 0.2 | 11.31% | 15.58% |
| F：恢复后融合 | 0.2 | 94.79% | 76.83% |
| D：未保护融合 | 0.5 | 99.37% | 92.50% |
| E：保护后直接融合 | 0.5 | 10.28% | 9.35% |
| F：恢复后融合 | 0.5 | 99.30% | 92.78% |

保护前后 16 张真实图像上的特征相对误差为 9.66e-07。原模型与保护模型在两个完整测试集上的正确样本数相同。AGA 后全 checkpoint 参数相对误差为 4.34e-03；该指标含未参与视觉推理的遗留参数，不能替代 feature error 或 accuracy。AGA 特征误差为 0.0619。

![真实模型融合](figures/clip_merging.png)

## 已完成：后处理边界

对于 $W^p=W_{ft}P$、可逆 $P$，理想 LS 的拟合值为

$$
W^a=\Pi_{\operatorname{col}(W_{ft})}W_{pre},\qquad
W^a-W_{ft}=-\Pi_{ft}\tau.
$$

这一结论已在论文附录证明。我们的直接 LS 数值验证中，不同正交、对角及稠密保护矩阵与统一投影结果的最大相对差异为 2.40e-11。作者带正则的正规方程与 float32 转换在高条件数时偏离理想结果；这些数值误差不构成安全防御。

跨 head 置换保持函数，但会破坏原代码默认对应关系：

| 跨 head 置换 + 正交基变换 | 相对输出误差；3 seeds 平均 |
|---|---|
| 保护 standalone | 3.14e-07 |
| 原逐 head AGA | 0.9735 |
| 子空间 matching + AGA | 0.0049 |

自适应子空间 matching 在全部 3 个种子中 100% 恢复 head correspondence，故不把跨 head 置换当成强防御。

![跨 head 置换](figures/head_permutation.png)

## 后处理候选：LayerNorm 空方向

若 $y=\gamma\odot z+\beta$ 且 $\mathbf1^Tz=0$，取归一化 $r\propto1/\gamma$，则 $r^T(y-\beta)=0$。对紧随 LayerNorm 的 Linear 作

$$
W'=W+ur^T,\qquad b'=b-u(r^T\beta),
$$

可改变参数列空间而保持该层在合法输入上的输出。它超出了“仅右乘可逆矩阵”的限制，且无需训练数据。候选施加在 QKV 与 MLP 第一层，再与基变换保护组合。

但 $r$ 由公开 LayerNorm 参数确定。攻击者可解析消去该方向：

$$
W_c=W-(Wr)r^T,\qquad b_c=b+(Wr)(r^T\beta).
$$

因此必须评价“canonicalization + AGA”，不能只观察原 AGA 是否失效。

| MNIST 全测试集 | 扰动强度 1 | 扰动强度 10 |
|---|---|---|
| 空方向扰动 standalone | 99.69% | 99.69% |
| 空方向 + 基变换 standalone | 99.69% | 99.69% |
| 原 AGA standalone | 99.67% | 99.67% |
| 投影消除 + AGA standalone | 99.68% | 99.68% |
| 原 AGA 后融合 | 99.32% | 99.12% |
| 投影消除 + AGA 后融合 | 99.30% | 99.30% |
| canonical 未保护融合对照 | 99.37% | 99.37% |

所有参数修改均为解析后处理，无训练、无校准数据。保护误差须通过脚本阈值 $10^{-4}$。canonicalization 使用公开 LayerNorm 参数，不需要秘密或原始训练数据。比较原 AGA 与自适应攻击时，同时保留 canonical 未保护融合对照。

![LayerNorm 后处理探索](figures/layernorm_exploration.png)

## 结论与下一步

1. 小任务变化下，置换和 head 内可逆变换可被 AGA 绕过；扩大变换幅度不改变理想投影结果。
2. 精确函数等价后处理确有可用自由度，但已知对称性往往也给攻击者提供 canonicalization。
3. 下一步优先分析“不属于易消去对称性”的小效用损失扰动：用少量公开、未标注校准输入的低活跃方向构造解析扰动，并预先固定 standalone 损失预算。该方向尚未验证，是否允许校准数据需与导师确认。
4. 先保留 MNIST/SVHN 两任务对照，加入 head matching、LayerNorm 投影消除及直接 LS 攻击；候选通过后再扩展任务、种子和融合方法。

## 审读发现

MLP 附录的原始欧氏 margin 条件不能直接保证代码中的 cosine matching；固定反例满足条件却得到错误置换。ARF 印刷式 (19) 的 `min` 与正文 margin 行为不一致，已核对 PDF。由于 ARF 训练代码未公开，不能据此推断其实际实现或实验无效。详情见 `theory_notes.md`。

参考：[论文 v1](https://arxiv.org/abs/2606.30360v1)、[作者代码](https://github.com/krumpguo/secure-merge-attack)、[Task Arithmetic](https://github.com/mlfoundations/task_vectors)。数字均取自本次程序输出；论文数字未填入结果表。
