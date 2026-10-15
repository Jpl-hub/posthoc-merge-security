# 融合联合响应的条件稳定性

## 研究假设

已有O/B的真实效果与最新联合信赖域反例共同说明：原任务保持、下游单分支改判、参数距离均不能给出融合后联合间隔损害的下界。本轮只检验一个更强、可证伪的假设：**在已经观测到补偿失配的位置，保持已知任务撤回分量不变的其他表示变化，不会轻易消除联合分类破坏。**

原专家、第一阶段和全部发布点冻结。只使用目标训练缓存中的原专家与公开基座插值状态，不读取真实伙伴缓存、测试或新确认数据。没有新的发布优化、保护权重或恢复攻击。

## 有限状态集合

把最后Transformer块之前的全部50个token记为$H$。从同一目标训练图片得到原专家状态$H_e$与固定公开基座插值状态$H_0$，定义$d=H_0-H_e$。末块及读出参数$p_0$来自这一个实际基座插值模型，没有拼接多个代理。

检验集合为

$$\mathcal U_i=\{H_0+\eta:\langle\eta,d\rangle_F=0,\ \|\eta\|_F\le\|d\|_F,\ R_i(s_0(H_0+\eta),s_0(H_0))\le0.5\}.$$

$R_i$是全部教师类别对的最大相对间隔漂移。零漂移方向之外还必须通过实际有限验收，并保持未保护预测。正交条件保留已观测任务撤回方向上的坐标；内层不能沿$d$简单返回原专家状态。半径以观测到的任务撤回长度为单位，不扫描增益或抽取随机代理。

这是**条件接口集合**。它给出一个明确的稳定性问题，既不证明未知伙伴属于集合，也不证明每个接口点都能由某个真实前缀或伙伴实现。真实融合还会改变局部参数、LayerNorm和读出；本检验将这些固定，不能据此认证整个融合过程。每张图片独立求见证也是对共同模型环境的放宽，不能把见证当作实际攻击模型。

## 联合间隔极小问题

令$y_i$为未保护中心类别，$g_i>0$为中心最强竞争间隔。对发布变化$\delta$，固定融合系数$\lambda=0.5$，完整后缀计算的间隔为

$$m_i(\delta,\eta)=\frac{s_{y_i}(H_0+\eta;p_0+\lambda\delta)-\max_{k\ne y_i}s_k(H_0+\eta;p_0+\lambda\delta)}{g_i}.$$

稳定失效要求$\sup_{\eta\in\mathcal U_i}m_i(\delta,\eta)<0$，而不是中心改判或下游单分支改判。若找到实际$m_i\ge0.01$的可行见证，就否定该样本在此集合内稳定失效；见证范数给出鲁棒半径的**上界**。未找到见证不构成下界或证书。

对应的候选构造问题可以写成

$$\max_{\delta}\ \alpha D(\delta)^2+
\min_{\pi\in\Delta_C}\min_{\eta_i\in\mathcal U_i}
\sum_c\pi_c\operatorname{mean}_{y_i=c}\sigma[-m_i(\delta,\eta_i)/\tau]
+\tau_c\mathrm{KL}(\pi\|\mathrm{Unif}),$$
$$D(\delta)\le0.5,\qquad\max_{x,\text{任务SVD探针},k,\pm}R_{xk}(\theta_e+\delta,\theta_e)\le0.5.$$

发布变量仍是结构化参数变换，原专家冻结；任务对手检查源功能，类别对手强调最弱类别，环境对手尽量保住发布后的正确分类。它直接作用于**有限联合排序**，区别于已关闭的源B分支目标。任务向量SVD只规定发布坐标与蒸馏探针，不能替代有限保持验收。

这只是本轮待证伪的构造方向，**尚未启动外层优化或得到新方法**。即使这个条件博弈可解，未知伙伴覆盖与AGA仍须另外验证；不能把加入内层环境对手本身当作创新或鲁棒性证明。

## 一次低成本检验

在读取保护点前，从源专家和未保护基座插值均正确的同一训练池中，每类按原索引次序固定4张，共40张。比较保留的O/B点与最新联合信赖域点，不按保护响应挑图片。

内层使用完整末块Attention、MLP、末LN及特征归一化。局部梯度只生成接口步：投影到$d$和全部未保护间隔Jacobian的正交空间；提议范数最多为$0.25\|d\|$。最多8步、每步6次减半；只接受实际间隔改善、实际／预测改善比至少0.1、完整未保护漂移不超过0.5、范数与正交条件均满足的步。不扩大步数、重启或搜索种子。梯度只用于接口变量，发布参数梯度为零；这是有限机制诊断，不是零计算方法。

保存全部提议分数、方向、接受记录和接口见证。独立NumPy重算全部token的Attention与有限非线性、所有提议和验收、逐类计数、方向有限差分和见证几何。只有真实冻结候选通过独立保持确认后才运行TA/AGA；本检验不产生候选，不能填入新的测试准确率。

```bash
python experiments/finite_environment_witness.py --protocol configs/finite_environment_witness.json
python experiments/verify_environment_witness.py --protocol configs/finite_environment_witness.json
```

## 相关工作与证据边界

[PaRaMS，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Junhao_Disrupting_Model_Merging_A_Parameter-Level_Defense_Without_Sacrificing_Accuracy_ICCV_2025_paper.html)已使用MLP排列、Attention配对缩放及距离目标；这些工具不能包装成新颖性。[TSV，CVPR 2025](https://arxiv.org/html/2412.00081v2)研究逐矩阵奇异方向压缩与白化，但不提供这里所需的未知伙伴联合间隔保证。[ESM，CVPR 2026](https://arxiv.org/html/2602.20208v1)根据任务更新引起的激活偏移估计功能相关子空间，说明不能仅从参数奇异值推断功能保持；它的目标是改善融合，不是本轮保护博弈。[作者代码](https://github.com/kiddo127/ESM)与论文正文均已核对。

[AGA／ARF](https://arxiv.org/html/2606.30360v1)分别是锚点恢复攻击与需要微调的Attention排斥防御；没有将ARF视作后处理或声称复现完整Table6。这里的条件集合、标准投影和有限验收并未完成新颖性证明。若本检验暴露方向反例，只能否定对应的稳定性假设，不能否定所有源端防御或把尚无证据的代理优化称为突破。
