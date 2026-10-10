# LayerNorm 规范化后替换 FC

本轮检验一种可能的补救是否具有抗恢复依据：给相邻 LayerNorm／线性层加入精确的缩放补偿。没有发布新保护，也没有把已有参数对称性当创新；先给出公开消去操作，再在冻结旧候选上评价。

## 公开可消去的耦合

相邻计算为 $W[\operatorname{diag}(\gamma)z+\beta]+b$，$z$ 是 LayerNorm 的标准化结果。公开 anchor 的仿射项为 $\gamma_0,\beta_0$，各项 $\gamma_0$ 非零且标准化的 $\varepsilon$ 相同。设

\[
W_c=W\operatorname{diag}(\gamma/\gamma_0),\qquad
b_c=b+W\beta-W_c\beta_0.
\]

把 LN 仿射项改为公开值，后续线性层改成 $W_c,b_c$，原线性输入输出对任意 $z$ 都不变。这个操作适用于 Attention 的合并 QKV 和 MLP 的 FC；后续 softmax／QuickGELU 无需具有齐次性。

若发布者用任意可逆对角 $D$ 改写为 $W'=WD$、$\gamma'=D^{-1}\gamma$、$\beta'=D^{-1}\beta$，上述规范化会得到相同的 $W_c,b_c$。故这一精确耦合不能阻止恢复：攻击者只用发布模型和公开 anchor，无需识别 $D$、任务标签、校准图或保护前模型。结论仅覆盖这种仿射等价变换，不能推广到所有跨模块依赖。

## 真实检验

对旧候选四层的八个 LN／线性对执行规范化，然后把四层 FC 换回公开参数；两组同时接受这条攻击，伙伴 SVHN 保持原样。

| 官方完整测试 | 未保护，同样攻击 | 旧保护，同样攻击 |
|---|---:|---:|
| 独立 MNIST | 9,963 / 10,000 = 99.63% | 9,970 / 10,000 = 99.70% |
| 融合 MNIST | 9,904 / 10,000 = 99.04% | 9,925 / 10,000 = 99.25% |
| 融合 SVHN | 24,330 / 26,032 = 93.4627% | 23,833 / 26,032 = 91.5535% |

攻击仍将受保护 MNIST 的融合恢复至 99.25%。它不是作者 AGA；旧候选的 15.56% AGA 结果不能抵消这条公开恢复路径。旧保护曾使用已知伙伴，因此不作为本轮未知伙伴的新方案。

额外 CPU 代数控制使用固定二进制对角缩放 1/4、1/2、1、2、4，作用于两组各八个参数对。规范化后与未缩放控制的全部模型参数逐元素相同。真实参数的有效矩阵／偏置相对误差最大分别为 $2.53\times10^{-8}$、$2.81\times10^{-8}$；这是浮点实现核验，不是分类误差界，也没有验证任意极端缩放的数值稳定性。

## 核验与范围

源码 `6385705` 在运行前冻结，作业 70.77 秒。保护参数与原单专家 AGA 及 FC 替换实验是同一冻结版本，没有重算保护。六行逐图计数、八对算子核验和代数控制指纹全部保存。[协议](protocol.json)、[结果](results.json)、[核验](validation.json)、[参数对控制](compensated_100_affine_audit.json)、[源码对照](source_validation.json)。重型预测只在服务器。

```bash
python experiments/subspace_interference.py --evaluate-saved outputs/2026-10-10/subspace_compensation_mnist --recovery-groups canonical_fc --output outputs/canonical_fc
python experiments/validate_results.py --subspace-test --output outputs/canonical_fc
```

本轮得到的是机制性反例：**把易删除的扰动用 LN 仿射缩放绑定到相邻线性层，不能修补当前 FC 漏洞。**尚未找到保持原任务计算且抵抗所有已测恢复的新局部构造。
