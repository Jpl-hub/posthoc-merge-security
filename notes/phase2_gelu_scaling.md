# QuickGELU近齐次变换与有限融合响应

## 可证伪假设

温度路线的真实AGA结果99.32%没有下降，已关闭。新假设针对MLP的隐藏表示坐标：输入行与输出列的正缩放在ReLU下严格保持函数，在QuickGELU下具有显式、有限且可核验的误差；参数平均不保持这对缩放，可能把少量具有分类作用的通道系统性放大。

缩放对称性是已有知识，不是本工作的创新。[Git Re-Basin](https://arxiv.org/abs/2209.04836)、[Symmetry-Aware Graph Metanetwork Autoencoders](https://arxiv.org/html/2511.12601v1)、[神经元合并](https://arxiv.org/abs/2010.13160)已讨论相关参数表示问题。本轮只判断这种已知几何在冻结CLIP/QuickGELU、严格预算及原版AGA下是否提供真实保护机制。

## 发布变换与任务误差

对最后MLP的神经元j，令$u_j=A_jx+a_j$，$g(u)=u\operatorname{sigmoid}(1.702u)$。选择$s_j\ge1$，改变

$$A'_j=s_jA_j,\quad a'_j=s_ja_j,\quad B'_{:j}=B_{:j}/s_j.$$

它不是旧反射，也不是归一化共轭：QuickGELU输入确实改变。对无人工偏移的真实输入，端点隐藏变化精确为

$$\delta h_e(x)=\sum_j B_{:j} e_{s_j}(u_j),\qquad e_s(u)=u[\sigma(1.702su)-\sigma(1.702u)].$$

对于$s\ge1$，$0\le e_s(u)\le |u|/(1+\exp(1.702|u|))$，因此每个神经元的齐次误差有全局上界；这不代表多神经元相加后分类仍保持。自然输入及任务奇异探针必须按完整归一化CLIP分数验证原0.5约束。B输入探针还产生$(1/s-1)o$，不能漏掉这一项。

距离对三个改变的张量分别归一化，不能把bias藏入更大的矩阵分母：

$$D^2=\sum_j\left[\alpha_j(s_j-1)^2+\beta_j(s_j^{-1}-1)^2\right]\le0.5^2,$$
$$\alpha_j=\|A_j\|^2/\|A\|_F^2+a_j^2/\|a\|^2,\qquad\beta_j=\|B_{:j}\|^2/\|B\|_F^2.$$

不另扫缩放幅度。令C为类别数（不超过隐藏宽度），每个神经元的唯一可选增益是上述成本等于$0.5^2/C$的正根；不选时$s=1$，总共至多选C个。正根用单调方程求解，和准确率无关。这样排除无预算的尺度增大，并使同一规则适用于不同任务。

## 有限融合恒等式

令$\lambda=0.5$，伙伴未知，$u_0=\lambda u_e+(1-\lambda)u_p$，$B_0=\lambda B_e+(1-\lambda)B_p$。保护后两条路径在该MLP前的输入完全相同，因为只修改最后MLP。对每个神经元有

$$u_1=[\lambda s+(1-\lambda)]u_0+\lambda(1-\lambda)(s-1)(u_e-u_p),$$
$$B_1=[\lambda/s+(1-\lambda)]B_0+\lambda(1-\lambda)(s^{-1}-1)(B_e-B_p).$$

当伙伴局部算子恰与原专家一致时，ReLU通道放大系数为

$$1+\lambda(1-\lambda)(s+s^{-1}-2).$$

QuickGELU的对应有限响应直接计算，不能用ReLU公式冒充真实CLIP结果。未知伙伴与原专家的偏差构成上面两个显式余项；前者经QuickGELU Lipschitz常数、后者经有限激活范数传播到输出，再由分类间隔判定。没有伙伴信息便不能给出这些余项的覆盖保证。源参数中点只用于这个有条件代数模型的构造目标，不是所有保护方法的统一必要门槛。

## 唯一子集构造

沿用目标专家的4,096张校准、512张尺度图片，以及功能响应排序得到的32阶A/B任务奇异探针，不使用已观察的伙伴或测试来选神经元。对每个固定增益，计算完整有限端点和参数中点的每神经元隐藏变化，不用参数原点Jacobian代替这些响应。

CLIP排序等价于$N(h)=hM+\sigma(h)b$。设源教师类别y与固定次高类别k的原始分子间隔为$G_{yk}>0$，第j个神经元在中点的隐藏增量为$d_{ij}B_{:j}$。其保守间隔损害

$$L_{ij}=\frac{-d_{ij}B_{:j}^{\top}(M_y-M_k)-|d_{ij}|\|PB_{:j}\||b_y-b_k|/\sqrt d}{G_{yk}}$$

满足：任何选定子集的$\sum_jm_jL_{ij}$均不大于实际有限间隔损害。它使用全局方差Lipschitz界，不把多神经元的LayerNorm交互当作可加线性项。大于1才认证该固定类别对跨界；类平均为正不等于实际改判。

只解一次二元子集问题：在端点生成约束、最多C个神经元内，最大化$D^2/0.5^2+\min_c\operatorname{mean}_{i:y_i=c}\sum_jm_jL_{ij}$。最坏类损害变量$z\ge0$，允许空子集；空解不发布。内层对手为类别单纯形，外层变量为发布子集；不是未知伙伴的鲁棒博弈。端点生成约束用有限隐藏变化及源分数读出Jacobian，内部上限0.25；它只是候选生成器，最终全部输入仍须满足原始0.5非线性约束。MILP固定60秒、1,000节点、1%间隙，保留唯一整数可行点，报告实际间隙。

先冻结子集/增益哈希，再独立核验全部自然/探针和分类间隔下界。若原任务保持通过，则按预先冻结规则进行新训练确认及一次真实TA/AGA；不按源中点或基座代理改判数回选子集。失败不增加增益、迭代、秩或种子。

## AGA与可恢复性边界

原版AGA对MLP依据A行余弦做排列匹配。正缩放不改变理论余弦，但源码有数值epsilon，因此实际排列仍需核验；该入口没有求解MLP正缩放。它在Attention中的最小二乘不能直接消除这里的A/B成对尺度。必须经过真实原作者AGA验证，不能从处理位置推断成功。

已知增益可直接逆转此变换；公开锚点行范数也可能用于估计尺度。这是明确的潜在廉价恢复方式。若仅在原版AGA下有效，应报告其攻击范围限制，不能声称获得普遍抗恢复或论文创新性。第一阶段和旧O/B不改变，只用3090，不进行任务重新微调。

## 有限覆盖条件

上式令$a_s=\lambda s+1-\lambda$、$b_s=\lambda/s+1-\lambda$。真实局部保护输出与理想输出$\sum_j b_{s_j}B_{0j}g(a_{s_j}u_{0j})$之差，有显式上界

$$R_h\le\sum_j\bigl(|b_{s_j}|\|B_{0j}\|L_g|r_{uj}|+\|r_{Bj}\|\,|g(a_{s_j}u_{0j}+r_{uj})|\bigr),$$
$$r_{uj}=\lambda(1-\lambda)(s_j-1)(u_{ej}-u_{pj}),\quad r_{Bj}=\lambda(1-\lambda)(s_j^{-1}-1)(B_{ej}-B_{pj}).$$

QuickGELU可取保守全局常数$L_g=1+1/e$。分类分子间隔另有误差界$(\|M_y-M_k\|+|b_y-b_k|/\sqrt d)R_h$。只有理想负间隔超过这一余量，才能得到相应条件下的分类跨越结论。还需要控制真实融合输入、读出与源构造上下文的差异；此处没有这些未知伙伴的上界。因此类平均损害优化不等于具有未知伙伴覆盖的鲁棒博弈。真实TA只在系数0.5时等于两个专家平均，本轮恒等式用于该固定协议，不外推其他TA系数。

## 冻结验证结果

唯一整数解选择10个神经元，增益5.664–9.668；三张量联合距离0.5。源4,608张自然图片均保持预测，9,728个自然/奇异探针最坏间隔变化0.11742。独立NumPy从实际缩放后的A、bias、B重建全部响应并通过；两种SVD矩阵方程误差均低于$3\times10^{-13}$。

新512张训练确认与全部历史登记数据分离，原／保护均512/512正确，3,072个自然/探针最坏变化0.01197。独立NumPy、原／保护各512次完整FP32模型重放全部一致。参数在确认前冻结，确认与真实融合之后没有重选子集或增益。

| 指标 | 未保护 | 成对缩放 | 保留O/B |
|---|---:|---:|---:|
| MNIST独立 | 99.69% | 99.69% | 99.71% |
| TA MNIST | 99.37% | 99.36% | 78.77% |
| AGA后MNIST独立 | 99.68% | 99.68% | — |
| AGA融合MNIST | 99.32% | 99.30% | 77.85% |
| TA SVHN | 92.50% | 92.93% | 62.94% |
| AGA融合SVHN | 91.77% | 92.21% | 61.42% |

MNIST独立10,000张预测不变；TA改变5张（失去3、新增2），AGA改变6张（失去4、新增2，配对p=0.6875），没有实质防御效果。SVHN伙伴独立25,371/26,032；TA与AGA分别净增加112、114个正确预测，不能当作保护效果。完整逐类、配对与原始计数见[真实结果](../results/gelu_scaling_real_eval/results.json)。

17组官方标签重计、发布三张量重建、显式TA和272次完整模型重放通过；[独立核验](../results/gelu_scaling_real_eval/independent_validation.json)。FP32导出距离0.4999999969，没有超过原0.5预算。没有新公开清理、跨被保护任务或测试后调参。

## 机制判定

固定作者AGA的原／保护MLP排列相同，A、bias、B均逐位等于原发布缩放加同一排列，[核验](../results/gelu_scaling_real_eval/recovery_scope_audit.json)。因此失败不是AGA删除缩放；**任务保持、大参数距离和保留下来的表示失配仍不足以造成分类失效。**

训练内也没有多类别强破坏证书：最坏类别平均间隔损害下界只有0.000173，固定次高类别跨界证书仅1/512，集中在类别8。源参数中点仅改判1/4,608；它没有被用作拒绝真实评估的门槛，但也不能因为有一个跨界就宣称迁移性。真实结果否定了这个固定构造的防御作用，不证明所有函数保持变换或所有有限补偿都无效。

本轮关闭，不扩大增益、子集数量或预算，不按这些测试结果寻找另一缩放配置。下一项机制必须说明：有限补偿留下的变化为什么是**跨类别、非保序且能超过表示／读出失配余量**，而不是仅让参数距离或通道幅度变大。目前尚无满足这项条件的新冻结候选。

## 复现与成本

3090项目根目录运行；输出路径须不存在。源码、配置和所有重文件指纹见[构造冻结](../results/gelu_scaling/source_freeze.json)与[真实评估冻结](../results/gelu_scaling_real_eval/source_freeze.json)。历史精确重放使用相应冻结源码包。

```bash
python experiments/finite_gelu_scaling.py --protocol configs/finite_gelu_scaling.json
python experiments/verify_gelu_scaling.py --protocol configs/finite_gelu_scaling.json
python experiments/confirm_gelu_scaling.py --protocol configs/confirm_gelu_scaling.json
python experiments/evaluate_cross_release.py --target MNIST --partner SVHN --protocol configs/gelu_scaling_real_eval.json --output outputs/phase2_gelu_scaling_eval_mnist_svhn
python experiments/verify_cross_evaluation.py --output outputs/phase2_gelu_scaling_eval_mnist_svhn
python experiments/audit_gelu_recovery.py --output outputs/phase2_gelu_scaling_eval_mnist_svhn
```

一次整数规划、一次解析读出Jacobian、零反向梯度、零原专家更新。构造12.78秒、NumPy源核验7.51秒、新确认17.68秒；完整评估前向212,208例、分组前向合计197.88秒，另有272例独立重放。分组前向时间不含模型载入、AGA和未单独计时的部分核验，不能作为端到端GPU总时长。重文件仅服务器保存，第一阶段与旧最佳完整保留。
