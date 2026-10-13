# posthoc-merge-security

对已微调 CLIP ViT-B/32 专家做免额外微调后处理：保留任务输入子空间，只在补空间加噪声。目标是保持独立能力，使参数融合后的**被保护任务本身**下降；同一算法须在不同任务专家上验证。

当前只做第一阶段：一个冻结构造，分别保护 DTD 与 MNIST，TA(0.5)、原作者 AGA。暂不新增公开清理、其它攻击或参数距离方案。仅使用 3090。

**已有初步 MNIST 融合干扰，尚非完整防御。** 独立 99.69% → 99.64%；原版 AGA 后，SVHN 伙伴下 MNIST 99.32% → 91.15%，DTD 伙伴下 99.42% → 77.16%。两次使用同一保护权重，独立核验通过。

类间噪声未改善：独立 99.68%，AGA 后融合 MNIST 99.32%。它扩大了输出方向覆盖，却降低了同预算下的融合响应；该候选已停止。[构造、真实结果与原因](notes/phase1_class_noise.md)。

同一交叉协方差噪声已验证两种被保护任务：AGA 后 DTD 68.78% → 68.62%，MNIST 99.32% → 98.94%，独立性能基本保持。效果不足，停止该候选；不能把表示负相关当作分类失效。[数学构造与跨任务结果](notes/covariance_noise.md)。

旧 RGB 输入噪声只干扰伙伴；旧 Q/K 噪声的直接干扰被 AGA 消除。[旧计数](results/previous_noise.json)仅作历史记录。

- [构造与停止条件](notes/phase1.md)
- `experiments/covariance_noise.py`：同一补空间噪声规则的两种被保护任务验证
- `experiments/verify_covariance_noise.py`：独立数学重算、官方标签及预测复核
- `experiments/feature_noise.py`：末层 MLP 输入补空间噪声及冻结的类间结构对照
- `experiments/replay_feature_noise.py`：同一发布权重的 DTD 迁移及预测核验
- `experiments/clip_benchmark.py`：既有 checkpoint、评估、Task Arithmetic 和作者 AGA 入口
- `experiments/mechanisms.py`：固定版本作者函数加载
- `experiments/subspace_probe.py`：共享训练划分和数据加载
- `configs/`：权重、源码及冻结配置

旧入口、旧分析和重复副本已从当前目录删除；唯一证据在[服务器归档](results/archive.json)，原始预测和模型只存服务器。历史 Git 版本保留。服务器工程目录为 `/root/autodl-tmp/merge_security_20261009`。
