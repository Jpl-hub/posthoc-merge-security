# Neuron Identifiability：核查范围

[ICML2026论文](https://proceedings.mlr.press/v306/burgin26a.html)、[arXiv v2](https://arxiv.org/pdf/2606.04754v2)、[作者代码](https://github.com/Vuenc/neuron-identifiability)。PDF正文首页明确提供代码。下载原文在服务器，读§3–4及附录G.4，未声称完整复现或全部证明已核查；摘要不能代替假设。

固定代码c92eb826d68c22cbb66ed2f0f102f43d7db92f17，已读最小实现文件，哈希见source_pin.json。作者用输入支撑上的函数类和最小权重范数讨论辨识：M_i=UᵀDiag(d_i)，S_i=M_iM_iᵀ。代码确实按投影中心和Gram伪逆计算重分配范数。非单射激活的结论还要求支撑含对称生成集合；有限PCA样本不能自动满足。

其架构F+D⊙W在训练中优化W；不是现成的免微调防御。ViT扩展明确讨论QK/VO对称、LN和残差耦合。范数代价不能直接当成攻击者查询数、恢复时间或安全下界。另一评估文件还调用内部优化，未执行这部分，也未把它算免训练。

对本项目的启示仅为：观察参数身份与局部谱能量，还不够说明实际函数依赖。当前路由构造独立推导标准softmax链式导数，不引用这篇论文作为安全保证。
