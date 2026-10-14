# posthoc-merge-security

对已微调 CLIP ViT-B/32 专家做发布前参数后处理，研究如何保持独立能力、阻止融合利用被保护任务。第一阶段补空间加噪声已冻结；第二阶段冻结原专家，只优化发布变换的系数，在任务奇异向量探针上约束计算变化，并增大有预算的参数距离。当前使用 3090。

**已有经过独立复核的阶段性效果。** 同一投影核偏置噪声算法分别保护 MNIST、DTD、SVHN；独立预测逐张保持，原作者 AGA 后的目标分类输出塌缩为单一类别。

| 被保护专家 | 伙伴 | 独立：原 → 保护 | AGA 后目标：原 → 保护 |
|---|---|---:|---:|
| MNIST | SVHN | 99.69% → 99.69% | 99.32% → 9.80% |
| MNIST | DTD | 同一 MNIST 保护权重 | 99.42% → 9.80% |
| DTD | MNIST | 79.41% → 79.41% | 68.78% → 2.13% |
| SVHN | MNIST | 97.46% → 97.46% | 91.77% → 6.70% |

只改最终 LN 的一个偏置向量。噪声满足 $v^\top B_e=0$，原专家完整视觉特征保持；融合投影 $B_f$ 改变后，$v^\top B_f$ 可能非零，产生类别偏置。没有 PCA 样本外保持假设、额外微调或伙伴参与构造。[数学机制、完整指标和边界](notes/projection_bias_noise.md)。

**范围必须区分：** TA(0.5)、FP32、三个被保护专家。原版 AGA 不修改最后 LN 偏置与视觉投影，因此这是该原版攻击后的融合干扰；尚无抗公开清理／参数替换的结论，也未证明任务信息不可恢复，不是 ARF/Table 6 复现。第一阶段结果作为冻结对照保留。第二阶段首轮及任务向量蒸馏补偿均保持独立能力，但真实融合和 AGA 后没有有效的目标准确率下降；[补偿构造、真实结果和约束问题](notes/phase2_compensation.md)、[结果与机制结论](notes/phase2_results.md)、[数学建模](notes/phase2_distance.md)。随后有限任务探针候选在三个专家上也没有有效下降：MNIST、DTD、SVHN 的 AGA 后准确率分别为99.35%、68.83%、91.76%。[有限探针机制与结果](notes/phase2_margin_probes.md)。当前发布系数优化采用目标训练数据和公开基座，属于有梯度的后处理，不是零梯度构造；[优化变量、约束及求解诊断](notes/phase2_projected_game.md)。新的去中心化归一化分数优化与解析复合蒸馏，仍只改判1/512；独立确认最坏探针变化0.699/0.651超过冻结的0.5预算，没有新release或完整融合评估。[排序梯度、可行性、分类边界证据](notes/phase2_angular.md)。最后Attention与MLP的跨模块候选通过了独立训练确认，但真实MNIST＋SVHN的AGA目标准确率99.32%→99.30%，配对p=0.815，仍未构成有效防御。[跨模块推导、完整目标/伙伴指标与代价](notes/phase2_cross_module.md)。未开展全面攻击实验或新增恢复攻击。

- `experiments/projection_bias_noise.py`：当前保护构造
- `configs/projection_bias_noise_svhn.json`：当前直接采集训练图的冻结协议；[SVHN 迁移结果](notes/projection_bias_noise_svhn.md)
- `configs/projection_bias_noise.json`：MNIST/DTD 原始协议，重放须使用 `7949958` 源码或服务器冻结包
- `experiments/replay_projection_bias.py`：去缓存后精确重现 MNIST/DTD 编辑权重
- `experiments/covariance_noise.py`：共享数据与匹配评测入口，默认历史算法保留
- `experiments/verify_noise_release.py`：独立 NumPy 重建、官方标签及真实前向核验
- `experiments/transfer_noise_release.py`：同权重更换伙伴与显式融合公式重放
- `results/projection_bias_noise/`：完整轻量计数、逐类结果、协议与指纹

历史对照保留：原 MNIST 秩一噪声 AGA 后为 91.15%／77.16%（SVHN／DTD 伙伴）；[类间噪声](notes/phase1_class_noise.md)、[交叉协方差噪声](notes/covariance_noise.md)、[MLP 输出补空间噪声](notes/decision_noise.md)未改善。[判别与 PCA 稳定性分析](notes/decision_audit.md)解释此前限制。历史重放使用各自冻结源码包；不混用旧版本指纹与当前共享入口。

SVHN 运行命令（3090 项目根目录，输出目录须为新的路径）：

```bash
python experiments/projection_bias_noise.py --target SVHN --partner MNIST --protocol configs/projection_bias_noise_svhn.json --output outputs/my_svhn_run
python experiments/verify_noise_release.py --output outputs/my_svhn_run --probe-n 32
```

原始预测、保护权重、模型、数据及历史证据只存服务器 `/root/autodl-tmp/merge_security_20261009`；[归档索引](results/archive.json)。不写凭据，不大规模下载，不删除真实失败证据。
