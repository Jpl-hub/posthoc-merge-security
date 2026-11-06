# Post-hoc model merge protection

研究冻结 CLIP 专家的发布后处理。目标是保持独立任务能力，并降低未经授权融合得到的目标任务准确率。当前集中完善防御一。构造只读模型权重和固定分类头，不用真实图片或合成图片，不训练专家。

防御一只在最终 LayerNorm 的偏置上加入扰动。扰动位于原视觉投影的左零空间，在原专家中被投影消去；融合后投影改变，扰动可能影响分类。方向和幅度由目标专家、公开基座及固定分类头计算。公开基座条件下的界与真实多专家融合结果分别报告。

**先看[防御一结果](results/phase3/bias_weight_only/other_tasks/summary.md)和[实验完成情况](results/comparison/README.md)。** 11月6日讨论后，后续工作集中在防御一的完整评价、其他防御基线、方法说明和必要消融。防御二停止新增试验，已有代码和结果保留为历史对照。

## 核心代码

| 内容 | 入口 |
|---|---|
| 防御一，构造视觉投影核空间的偏置扰动 | [experiment/construct.py](experiment/construct.py) 的 `--method bias` · [_impl/defenses.py](_impl/defenses.py) 的 `construct_bias` |
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
