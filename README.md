# posthoc-merge-security

已微调 CLIP ViT-B/32 专家的发布前后处理实验。目标是在不增加梯度微调、不依赖未来融合伙伴的条件下保持原任务能力，并干扰未经授权的参数融合。

**当前研究已按用户要求暂停。没有确认可靠防御或论文级突破。**

## 当前结果

只保留一个主要候选：[MNIST 局部 Attention／MLP 后处理组会汇报](reports/2026-10-12/组会汇报.md)。

| MNIST 准确率 | 原始专家对照 | 发布专家 |
|---|---:|---:|
| 独立运行 | 99.69% | 99.63% |
| 直接融合 | 94.00% | 78.33% |
| 作者原版 AGA 后融合 | 93.23% | 64.93% |
| 公开替换＋256 次类别标签查询恢复 | 94.93% | 94.93% |

融合包含八个专家，Task Arithmetic 系数为 0.2。候选只用目标训练校准数据，包含三次闭式响应拟合，没有额外梯度微调。结果说明存在真实融合干扰；公开替换和有限查询恢复仍可消除保护效果。完整协议、伙伴任务代价和适用边界见报告。

## 目录

- `reports/2026-10-12/`：唯一主报告、结果图及其绘图代码。
- `results/mnist/`：冻结构造、融合评测、类别标签恢复的轻量原始记录与复核证据。
- `results/reproduction/`：已有七个任务的 PaRaMS／AGA 基线证据，不作为完整 Table 6 复现。
- `experiments/`：三个候选与攻击入口，以及运行它们所需的共享模块。
- `configs/`：所选候选、评测协议、源码与 checkpoint 指纹。
- `results/index.json`：当前证据与原始路径的对应关系。
- `results/history.csv`：历史记录索引；旧报告和失败入口已移出当前目录。
- `results/cleanup.json`：清理范围与服务器归档位置。

## 复核与复现

主要入口分别为 `joint_lossy_recoding.py`、`local_release_audit.py`、`prototype_recovery_gate.py`。融合、数据集和 AGA 使用已有作者实现，源码版本固定在 `configs/sources.json`。

历史实验必须按各自 `launch.json` 的 `source_commit` 恢复代码与协议：

| 阶段 | 冻结代码版本 | 轻量证据 |
|---|---|---|
| 构造与独立效用 | `c67ceebce20fc98c747b13000cc5d1dee53d317e` | `results/mnist/construction/` |
| 融合与恢复评测 | `65be298bb40ebc5382530479ad167d8de8535b3c` | `results/mnist/evaluation/` |
| 类别标签查询恢复 | `8577d9a104b7cfe48cddf20f6113a41d4b514114` | `results/mnist/hard_label_recovery/` |

清理后的目录删去了旧指纹清单引用的无关历史文件，因此整份冻结清单的复核应在对应 Git 版本中进行。原始协议和结果没有重写。原目录最后完整版本为 `75b61cf`，普通 Git 历史保留。

模型、数据和原始预测只存放在已有 GPU 服务器。服务器项目目录为 `/root/autodl-tmp/merge_security_20261009`；旧输出已移至 `/root/autodl-tmp/research_archive/2026-10-12/raw_outputs`，所选结果及其必要父记录通过链接保留原路径。本地清理前快照及服务器代码快照也保存在该归档目录。

绘图只读取保存的计数，不运行模型：

```bash
python reports/2026-10-12/plot.py
```

使用已有 Python 3.12／Torch 2.3 服务器环境。未经重新授权，不启动探索或续跑。
