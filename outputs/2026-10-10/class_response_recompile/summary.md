# 类别响应重编译：效用门槛未通过

这是任务相关响应可表示性的辅助检验，不替代导师的任务子空间／补空间后处理主线，也不是新防御结论。原十二层构造要求拟合完整 block 表示；本次只约束最终类别差，并检验把任务响应重新分配到一个输出投影是否可行。

## 一个固定构造

保留专家的 embedding、位置编码、LayerNorm、最终投影及 head；全部十二层 Attention 与 MLP 先替换为公开 anchor，然后只用最后 MLP 输出投影承载重新拟合的响应。同架构、同融合实现，不使用梯度或优化器，但有校准数据和参数拟合。

令最终 LayerNorm 前 CLS 为 $z$，$u(z)=(z-\bar z)/\sqrt{\operatorname{Var}(z)+\varepsilon}$。最终视觉投影为 $V$，类别 head 为 $h_j$，LayerNorm 的仿射参数为 $\gamma,\beta$。定义中心化列矩阵

\[
C=\operatorname{center}\left[\gamma\odot V(h_j-h_0)\right]_{j=1}^{9},\qquad P=CC^+.
\]

记原专家响应为 $u_t$、替换后响应为 $u_b$，设 $a=u_tP$、$q=u_b(I-P)$。在剩余范数为正的条件下，构造

\[
u_*=a+\sqrt{\frac{\|u_b\|^2-\|a\|^2}{\|q\|^2}}q,
\qquad z_*=\bar z_b+\sqrt{\operatorname{Var}(z_b)+\varepsilon}\,u_*.
\]

它保留 base 的均值与方差，并满足 $u_*C=u_tC$。$\beta$ 不变，head 偏置为零；CLIP 最后的正长度归一化不改变类别排序。因此**精确实现这个逐输入目标**会保留原类别排序。该条件仅在校准输入上构造，不保证拟合器或新图实现它。

以最后 MLP 的 GELU 隐藏特征和常数列组成 $H$，只解一次已有的岭回归更新：

\[
\Delta D=(H^\top H+\rho I)^{-1}H^\top(Z_*-Z_b),\qquad
\rho=10^{-4}\|H\|_F^2/N.
\]

本次类别约束秩为 9；方差补偿也改变类零空间成分，**参数更新不声称 rank-9**。闭式响应重建已有 [GRAIL](https://arxiv.org/html/2602.23795v2) 等前作，本轮未确立原创性。

## 预先约定的恢复检验

FC 与 Attention 已是公开参数，单独替换不会改动这些参数；完整 MLP 替换会删除重新承载的响应。完整 Attention＋MLP 替换后，两组模型参数必相同，其融合准确率是否仍低必须实测，不能从支持集合推断。

配置还冻结了原作者 AGA、两类输出替换及公开最后输出增量乘 2；每条均有匹配未保护对照。任何通过者还要做功能恢复与官方 test。不能用功能重新分配的代数解释，替代攻击结果。

## 真实结果：停止

用 1,024 张无标签 MNIST 训练图及原专家内部响应作防御校准。另抽 4,000 张训练确认图，与所有列出的旧子集及本次校准互斥；它们不是原专家训练之外的数据，不是官方 test。

| 独立 MNIST | 正确数 | 准确率 |
|---|---:|---:|
| 原专家 | 3,993 / 4,000 | 99.825% |
| 类别响应重编译 | 3,723 / 4,000 | 93.075% |

损失 **6.75 pp >0.5 pp**。配对计数：失去 273 张、新增正确 3 张。按冻结门槛淘汰，**没有运行 AGA、替换、功能恢复或官方 test**；这些攻击不能填写有效／无效结论，也不继续调岭系数或规模。

精确目标类别响应相对误差为 $1.43\times10^{-15}$；实际闭式拟合后的校准类别差误差为 2.736%，完整目标误差为 2.644%。正规方程残差为 $1.37\times10^{-11}$，求解器核验通过。这区分了代数目标与实际拟合；仍没有单独判定容量、正则或校准覆盖的因果贡献，不能推广为所有任务响应重表示均不可行。

源码 `12bb62f` 在运行前冻结，计算段 22.32 秒；额外核验源码 `ffc1974`。[协议](protocol.json)、[逐图结果](results.json)、[门槛判定](decision.json)、[原始参数及预测核验](validation.json)、[源码指纹](source_validation.json)完整保留，所有重型产物只在服务器。

```bash
python experiments/class_response_recompile.py --config configs/class_response_recompile.json --output outputs/class_recompile
python experiments/validate_results.py --class-recompile --output outputs/class_recompile
```

下一构造须保留足够的原任务计算依赖，并在运行前说明补空间干扰与补偿如何作用。不能因为类别约束低维，就假定一个公开 MLP 输出层足以重建原专家任务功能。
