# posthoc-merge-security

研究已有 CLIP ViT-B/32 专家的发布前后处理：不增加梯度微调，不依赖未来融合伙伴，保留原任务计算，在补空间引入融合干扰。

**当前使用 3090。尚无确认的可靠防御。** [研究主线与失败原因](notes/research.md)说明已有结果、保持条件和下一步判断。

## 代码

| 文件 | 用途 |
|---|---|
| `experiments/subspace_probe.py` | 中心化任务子空间、补空间编辑与清理；无搜索入口 |
| `experiments/clip_benchmark.py` | checkpoint、作者分类器、Task Arithmetic、AGA 与配对计数 |
| `experiments/mechanisms.py` | 固定版本作者求解器加载与基础参数变换 |
| `experiments/replay_complement.py` | 冻结补空间结果的迁移核验，不搜索新参数 |
| `experiments/qk_complement.py` | 一个固定的 Q/K 补空间候选；结果待验证，无调参入口 |
| `experiments/eight_task_benchmark.py` | 原论文八任务基线评测 |
| `experiments/prepare_eight_task_data.py` | 官方数据与划分准备 |

源码和 checkpoint 指纹见 `configs/sources.json`、`configs/checkpoints.json`。只有三份配置。数据、模型、原始预测只保留在服务器；运行环境为 Python 3.12、Torch 2.3。

## 证据

- `results/complement/reference/`：旧补空间构造的原始协议、选择过程和计数；未知伙伴约束的限制见研究说明。
- `results/complement/replay/`：3090 迁移核验。
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
