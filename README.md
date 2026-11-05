# Post-hoc model merge protection

研究冻结 CLIP 专家的发布后处理。目标是尽量保持独立任务能力，并降低未经授权融合得到的目标任务准确率。防御一的构造只读权重。防御二使用冻结权重、任务向量 SVD、固定分类头和合成输入。两者都不使用真实训练图构造防御。

**从[论文实验与结果索引](results/comparison/README.md)开始。** 这里区分当前主方法、补充对照和投稿前仍需解决的问题，并提供对应的 Markdown、TeX 和原始证据入口。

## 核心代码

| 内容 | 入口 |
|---|---|
| 防御一，构造视觉投影核空间的偏置扰动 | [experiment/construct.py](experiment/construct.py) 的 `--method bias` · [_impl/defenses.py](_impl/defenses.py) 的 `construct_bias` |
| 防御二，末层 MLP 的任务 SVD 约束与 A/B 补偿 | [experiment/defense2.py](experiment/defense2.py) · [构造配置](configs/defense2_inputs.json) · [固定协议](results/data_free_compensation/finite_classification.json) |
| 防御二的读出补偿对照 | [experiment/readout_channels.py](experiment/readout_channels.py) · [结果](results/comparison/defense2_readout.md) |
| 两专家评价、独立评价和八专家融合 | [experiment/main_table.py](experiment/main_table.py) · [experiment/native.py](experiment/native.py) · [experiment/eight_task.py](experiment/eight_task.py) |
| 原作者 AGA、公开恢复和额外数据蒸馏 | [experiment/recover.py](experiment/recover.py) · [experiment/distill.py](experiment/distill.py) |
| TA、TIES、DARE 融合 | [_impl/merge.py](_impl/merge.py) |
| 论文表格生成 | [experiment/report.py](experiment/report.py) · [_impl/report.py](_impl/report.py) |

`_impl/` 放公共实现，`experiment/` 放入口，`configs/` 放固定输入，`results/` 放结果及来源，`notes/` 放推导和研究记录。服务器的 `outputs/` 保存模型、逐图分数及执行源码。历史研究可从[归档索引](results/engineering_refactor/snapshot.json)查回。

## 复现

在仓库根目录使用 Python 3.12 和 [requirements.txt](requirements.txt)。完整 PaRaMS 的额外依赖见 [requirements-params.txt](requirements-params.txt)。模型清单与 SHA 见 [configs/checkpoints.json](configs/checkpoints.json)，原作者源码版本见 [configs/sources.json](configs/sources.json)。各历史批次按其执行版本和协议复现。

生成当前对比表无需模型、数据或 GPU：

```bash
python -m experiment.report --defense-comparison --output results/comparison
python -m pytest -q tests/test_paper_tables.py
```

运行模型实验前，先从[结果索引](results/comparison/README.md)找到对应的冻结发布和协议。安装、数据路径和复核方式见[工程说明](docs/engineering.md)。
