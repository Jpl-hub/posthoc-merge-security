# 下游分类反向与上游有限补偿

## 假设与范围

O/B、A/B旧点包含被联合补偿掩盖的下游分类破坏，但类别覆盖不足。本轮反过来主动构造下游作用，再检查有限补偿是否可行。下游单分支改判仅是这条构造的设计目标，不是所有保护方法的必要条件；源内可行也不等于真实融合有效。

固定MNIST专家末块MLP的$A\in\mathbb R^{3072\times768}$、$B\in\mathbb R^{768\times3072}$。原专家和分类头不更新；只允许发布变换系数改变。沿用4,096张目标训练图片、512张既有尺度校准图片，以及40个低间隔训练上下文的A/B任务奇异探针，共9,728个有限输入。独立确认图片不参与构造。

## 分类作用与预算

最终CLIP排序等价于去中心化分类分子

$$N(h)=hM+\sigma(h)b,\quad \sigma(h)=\sqrt{\operatorname{mean}(h-\bar h)^2+\epsilon}.$$

令$Q$为$M$的完整数值列空间，$M=Qg$、$b=vg$、$w=vQ^\top$。给定目标$R=-N(h)$，构造

$$d=(R-hM)g^\top(gg^\top)^{-1},\quad s=h+dQ^\top,$$
$$a=1-\operatorname{mean}(w-\bar w)^2,\quad b_0=\operatorname{mean}[(s-\bar s)(w-\bar w)],$$
$$c_0=\operatorname{mean}(s-\bar s)^2+\epsilon,\quad
\rho=\frac{c_0}{\sqrt{b_0^2+ac_0}+b_0},\quad h_*=h+(d-\rho v)Q^\top.$$

在已核验的$\operatorname{rank}(g)=C-1$、$a>0$条件下，$\rho=\sigma(h_*)$且$N(h_*)=R$。这是非保序目标，区别于正比例缩小分类间隔；它不保证共享参数矩阵能够实现每张图片的目标。

只拟合一个共享下游变换$\delta B=\|B\|_FQK$：

$$\min_{\|K\|_F\le r_B}\sum_i\omega_i\left\|z_iK^\top-\frac{(h_{*,i}-h_i)Q}{\|B\|_F}\right\|^2,
\quad \omega_i=\frac1{C\,n_{y_i}},\quad r_A=r_B=\frac{0.5}{\sqrt2}.$$

类别由原教师预测确定；所有类别使用同一规则，不指定MNIST某个数字或人工类别循环。谱信赖域根求解只解同一个凸问题，不扫描保护幅度。不在无分类作用的方向填满剩余预算。逐类统计真实有限分数的改判，预设至少8/10类各有10%改判，才进入该构造的补偿检验。

## 上游恢复与任务对手

固定$K$后，仅优化$\delta A=\|A\|_FCV_A$，其中$V_A\in\mathbb R^{32\times768}$来自既有、按源任务功能响应选择的任务向量奇异方向。$C\in\mathbb R^{3072\times32}$；不把奇异值大小等同功能重要性，不把这32维称为完整任务空间。

联合有限计算为

$$h_i(C)=r_i+\left[\phi\!\left(x_iA^\top+a+\|A\|_Fx_iV_A^\top C^\top\right)+o_i\right](B+\delta B)^\top.$$

$r_i$保留原计算中的残差及浮点重建项，$o_i$保留显式B输入探针。任务对手从全部训练输入、探针和竞争类选择最大相对间隔变化：

$$\mathcal R(C)=\max_{i,k\ne y_i}
\left|\frac{[s_{y_i}(h_i(C))-s_k(h_i(C))]-[s_{y_i}(h_i)-s_k(h_i)]}{s_{y_i}(h_i)-s_k(h_i)}\right|.$$

只接受$D=\sqrt{\|C\|_F^2+\|K\|_F^2}\le0.5$、$\mathcal R\le0.5$且所有自然训练预测保持的点。保留原40个低间隔上下文，并累积每类最坏见证。每轮以精确非线性点的分类Jacobian解最小范数信赖域蒸馏修正，最多8轮、每轮最多8次回溯；接受条件是全部9,728输入上的最坏漂移严格下降。不是标签交叉熵微调；原模型冻结，但发布系数存在明确的导数优化成本。

Jacobian每行有Kronecker结构$J_\ell=a_\ell\otimes t_\ell$，因此

$$ (JJ^\top)_{\ell m}=(a_\ell^\top a_m)(t_\ell^\top t_m),\qquad
J^\top u=\sum_\ell u_\ell a_\ell t_\ell^\top.$$

避免显式存储98,304列的矩阵；每轮独立有限差分检查真实方向导数。有限分类和最坏约束仍用完整非线性计算验证。

## 极大极小含义与局限

理想外层可表述为在$D\le0.5$及$\mathcal R\le0.5$下，增大有分类作用的参数差异和最坏类别下游跨界率；内层类别对手选择受损最少的类别，任务对手选择漂移最大的输入／探针／竞争类。本轮只实现一个明确的两阶段可行性近似：类别均衡的凸反向拟合，再作最坏任务约束修复。**没有宣称解出全局极大极小或未知伙伴对手。** 参数距离作为预算和实测量，不能独立判定保护成功。

固定下游变换的融合残差仍含$\lambda(Z_f-Z_e)\delta B^\top M_e$、上游补偿失配、读出与归一化余项。只有这些项沿某竞争类合计跨越原间隔才改判。源内反向拟合不提供未知伙伴表示变化的方向和大小保证；不能以人为随机代理补齐覆盖假设。

## 验证顺序与停止规则

先冻结配置和源码，只运行一个源内构造。下游覆盖不足即关闭；补偿未通过原预算及最坏任务约束，也关闭，不缩小破坏强度重试。独立NumPy重建原始参数、谱拟合KKT条件、全部有限分数与逐类计数，保留失败点及中间系数。仅源内通过后才允许一次新训练确认，之后冻结导出，再真实TA与原作者AGA；伙伴／测试结果不反馈构造。第一阶段及所有历史权重不动。

与已有工作相比，[PaRaMS](https://arxiv.org/html/2503.07661v1)的排列／成对尺度变换、[AGA/ARF](https://arxiv.org/html/2606.30360v1)的锚点恢复／排斥锚点微调，均不直接证明本构造成立。[Task Singular Vectors](https://arxiv.org/html/2412.00081v2)说明任务矩阵奇异结构的经验价值，不提供此处非线性蒸馏或未知融合保持保证。线性最小二乘、谱信赖域和非线性蒸馏均是已有工具，本轮不主张它们本身具有新颖性。

```bash
python experiments/downstream_first_compensation.py --protocol configs/downstream_first_compensation.json
```
