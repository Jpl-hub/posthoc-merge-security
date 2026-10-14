# 分类边界与任务保持

旧联合发布点已关闭。保持失败见证不能再次参与构造。新诊断不加载旧候选，仅加载原专家、公开基座及任务奇异基。

## 冻结诊断

从MNIST允许的训练数据排除历史校准、确认及训练留出索引，固定抽取4,096张新校准图片。按原专家预测类别，每类选两个教师top-two间隔最小的上下文；另取相同类别数量的随机上下文，与旧探针上下文作匹配比较。规则适用于任意类别数，不定向选择5／6。此集合是校准诊断，不能再用作独立确认。

最后O／B的位置、原任务奇异基、分类读出空间和参数预算0.5不变。用原专家原始归一化分数间隔定义局部任务响应：

$$[J_{\rm task}]_{(x,c),:}=\frac{\nabla_\delta(s_y-s_c)}{s_y-s_c},\qquad y=\arg\max s_e(x).$$

同时计算公开基座0.5代理的去中心化归一化分数排序梯度$g$，不读取真实伙伴。数值SVD定义有限上下文任务响应矩阵的行空间$V$，其近零方向为

$$d=\frac{(I-V^\top V)g}{\|(I-V^\top V)g\|}.$$

检查保留下来的排序梯度、夹角、全部4,096个自然校准输入与新的任务奇异探针上的方向导数及非线性变化。有限路径仅作预算内的预定八步回溯，从0.5距离开始；可行点按原任务0.5约束判断，不按代理准确率选择。

这不是保护方法或参数更新。最坏保持误差来自有限任务输入，不能据$J_{\rm task}d\approx0$宣称样本外保持；大代理梯度也不能宣称真实融合失效。若可行方向的有限影响不足，就先收缩该参数化，不再直接运行64步旧方案。

## 与已有工作的关系

| 方法 | 核心机制 | 与当前边界的区别 |
|---|---|---|
| [PaRaMS](https://openaccess.thecvf.com/content/ICCV2025/papers/Junhao_Disrupting_Model_Merging_A_Parameter-Level_Defense_Without_Sacrificing_Accuracy_ICCV_2025_paper.pdf) | MLP排列、Attention成对缩放；dropout是额外变体 | 函数对称变换已有公开方案，不能当作新方法 |
| [MergeLock](https://arxiv.org/html/2509.01548v1) | Q/K、V/O成对可逆变换，独立函数保持 | [官方仓库](https://github.com/hetailang/Merge-Lock)当前仅README，不能声称完整代码复现 |
| [MergeGuard](https://arxiv.org/html/2511.11851v1) | 密度正则任务微调，再作权重方向保护 | 这里指双阶段防御，区别于同名水印审计、Trojan论文；微调阶段不符合当前冻结专家约束 |
| [MergeBarrier](https://arxiv.org/html/2511.10712v1) | Attention正交变换与FFN展开表示 | 改变FFN结构；低成本参数替换仍需评价，不直接移入当前O／B后处理 |
| [AGA／ARF](https://arxiv.org/html/2606.30360v1) | AGA锚点解析恢复；ARF排斥锚点微调后再施加Attention可逆保护 | 受保护任务失效是主指标；ARF不是后处理或攻击，当前没有复现Table6 |

原作者AGA源码固定在`configs/sources.json`；PaRaMS源码也已固定。论文中的多任务系数、受保护任务平均结果与当前双任务TA(0.5)不能直接横比。不把已有对称性、论文主张或代理损失写成我们的新安全结论。
