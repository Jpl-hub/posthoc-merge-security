# posthoc-merge-security

研究已有 CLIP ViT-B/32 专家的发布前后处理：不增加梯度微调，不依赖未来融合伙伴，保留原任务计算，在补空间引入融合干扰。

**目前没有可靠防御：AGA 后 MNIST 融合准确率仍为 99.32%，公开清理还能消除伙伴干扰。** 输入补空间噪声保持独立 MNIST 99.69%，使 AGA 后 SVHN 从 91.77% 降至 9.38%；这只是伙伴干扰。当前只用 3090；[机制与结果](notes/research.md)说明保持条件及恢复边界。

同一份冻结权重迁移到 DTD：原版 AGA 后 68.78% → 12.55%；公开清理后恢复到 68.67%，与同条件对照逐样本相同。尚未阻止攻击者融合利用 MNIST 能力。

当前先做[机制与可行性分析](notes/mechanism_analysis.md)：固定公开投影的联合编辑仍可逐处清理，V/O 零空间噪声也有直接删除公式。尚无通过恢复代价检查的新候选，暂不生成保护权重。 [首层 MLP 检查](notes/mlp_functional_null.md)未发现跨输入共同保持局部计算的新方向；最弱近似方向仍落在旧低活跃神经元，已停止。

已有中间层数据的 CPU 复核进一步区分了基座计算与专家新增计算：后者在补空间的响应约为前者的 0.8%，而公开回填保留前者。这解释了该区域融合响应增加、清理代价却很小的现象；尚无抗恢复突破。

[末层 CLS Query 回填](notes/final_query_completion.md)也已停止：两份公开投影删除约 6% 的 Query 更新能量，MNIST 独立 99.69%、融合 99.45% 的预测全部不变，DTD 融合只少对 1 张。10 行计数及 1280 张复跑核验通过；这只是清理代价诊断，没有新保护模型或 AGA 结果。

[AGA／ARF 核对](results/arf_review/README.md)明确了主指标和复现缺项；[Key 差异空间核验](results/key_contrast_audit/README.md)只复用训练缓存，没有生成保护权重。

## 代码

| 文件 | 用途 |
|---|---|
| `experiments/subspace_probe.py` | 中心化任务子空间、补空间编辑与清理；无搜索入口 |
| `experiments/clip_benchmark.py` | checkpoint、作者分类器、Task Arithmetic、AGA 与配对计数 |
| `experiments/mechanisms.py` | 固定版本作者求解器加载与基础参数变换 |
| `experiments/replay_complement.py` | 冻结补空间结果的迁移核验，不搜索新参数 |
| `experiments/input_complement.py` | 灰度输入补空间后处理；一次固定噪声对照，含 AGA 与公开恢复 |
| `experiments/readout_completion.py` | 最后 MLP 分类读出补空间的公开回填代价；不生成保护候选 |
| `experiments/middle_completion.py` | 固定一个中间 MLP 输入补空间，测真实融合响应和两份公开 PCA 回填代价 |
| `experiments/eight_task_benchmark.py` | 原论文八任务基线评测 |
| `experiments/prepare_eight_task_data.py` | 官方数据与划分准备 |

源码和 checkpoint 指纹见 `configs/sources.json`、`configs/checkpoints.json`。只有三份配置。数据、模型、原始预测只保留在服务器；运行环境为 Python 3.12、Torch 2.3。

## 证据

- `results/complement/reference/`：旧补空间构造的原始协议、选择过程和计数；未知伙伴约束的限制见研究说明。
- `results/complement/replay/`：3090 迁移核验。
- `results/qk_complement/`：固定 Q/K 候选，独立效用保持、直接干扰有效，但 AGA 恢复，已停止。
- `results/input_complement/amplification/`：独立 MNIST 保持，AGA 后 SVHN 降 33.38 个百分点；公开补空间恢复完全消除效果。
- `results/input_complement/noise/`：固定种子、同范数噪声对照；独立 MNIST 保持，AGA 后 SVHN 降至 9.38%，18 项计数独立复核通过；公开恢复仍有效。
- `results/input_complement/dtd/`：冻结编辑包的迁移验证；20 项计数、官方测试划分和复用指纹核对通过，公开恢复仍有效。
- `results/input_complement/mechanism_review.json`：AGA／NUFILT 核查、公开清理等式及旧控制结果。
- `results/readout_completion/`：公开回填的必要条件诊断。移除大量任务更新能量仍保留融合能力，已停止该固定输出补空间方向；[推导与结果](notes/recovery.md)。
- `results/middle_completion/`：中间层补空间响应在融合后增大，但公开回填仍保留 MNIST 融合能力、DTD 只少判对 1 张；停止该固定区域，不生成噪声。[单次诊断](notes/middle_completion.md)。
- `results/complement_update_audit/`：上述冻结矩阵与激活统计的 CPU 重算及独立核验；不增加模型前向、候选或准确率实验。
- `results/mnist/` 与 [旧组会报告](reports/2026-10-12/组会汇报.md)：内部压缩历史证据，不作为当前补空间主线。
- `results/reproduction/`：已有 PaRaMS／AGA 基线记录；尚非完整 Table 6 复现。
- `results/index.json`：证据路径与指纹；`results/history.csv`：失败记录索引。

## 运行

服务器项目：`/root/autodl-tmp/merge_security_20261009`。

```bash
python experiments/replay_complement.py \
  --reference /root/autodl-tmp/research_archive/2026-10-12/3090/raw_outputs/2026-10-09/post_meeting/subspace_probe \
  --output outputs/complement_replay
```

历史入口已从当前目录删除。复核旧结果应使用其冻结 Git 版本，不能用精简后的源码冒充历史版本。清理前最后完整目录版本为 `75b61cf`，本次再次精简前为 `4e3c4f7`。原始结果归档在 GPU 服务器项目目录外；归档不等于释放磁盘空间。3090 数据盘目前约使用 9 GB / 50 GB。

AGA 原始权重身份和 EuroSAT 原划分仍未核实，因此不宣称精确复现 Table 6，也不填写缺项八任务均值。
