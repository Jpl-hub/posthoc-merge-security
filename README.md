# posthoc-merge-security

对已微调 CLIP ViT-B/32 的 MNIST 专家做免额外微调后处理：保留任务输入子空间，只在补空间加噪声。目标是保持独立 MNIST 能力，使参数融合后的 **MNIST 本身**下降。

当前只做第一阶段。一次候选、MNIST＋SVHN、TA(0.5)、原作者 AGA；通过后才用同一发布权重验证 DTD。暂不新增公开清理、其它攻击或参数放大方案。仅使用 3090。

**已有初步 MNIST 融合干扰，尚非完整防御。** 独立 99.69% → 99.64%；原版 AGA 后，SVHN 伙伴下 MNIST 99.32% → 91.15%，DTD 伙伴下 99.42% → 77.16%。两次使用同一保护权重，独立核验通过。

旧 RGB 输入噪声只干扰伙伴；旧 Q/K 噪声的直接干扰被 AGA 消除。[旧计数](results/previous_noise.json)仅作历史记录。

- [构造与停止条件](notes/phase1.md)
- `experiments/feature_noise.py`：一个末层 MLP 输入补空间噪声候选
- `experiments/replay_feature_noise.py`：同一发布权重的 DTD 迁移及预测核验
- `experiments/clip_benchmark.py`：既有 checkpoint、评估、Task Arithmetic 和作者 AGA 入口
- `experiments/mechanisms.py`：固定版本作者函数加载
- `experiments/subspace_probe.py`：共享训练划分和数据加载
- `configs/`：权重、源码及冻结配置

旧入口、旧分析和重复副本已从当前目录删除；唯一证据在[服务器归档](results/archive.json)，原始预测和模型只存服务器。历史 Git 版本保留。服务器工程目录为 `/root/autodl-tmp/merge_security_20261009`。
