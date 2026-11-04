# 工程与复现

所有命令从仓库根目录执行。当前算法、实验入口、固定配置及结果分开存放；历史资产在工程目录之外归档，不作为当前算法入口。

## 当前代码

| 职责 | 位置 |
|---|---|
| 共同路径 | `cfg.py` |
| 模型、数据、融合、攻击及统计 | `_impl/` |
| 构造、评价和生成表格 | `experiment/` |
| 固定模型和作者源码版本 | `configs/` |
| 结果、MD/TeX表格和来源 | `results/` |
| 模型、原始分数及执行快照 | 实验机器上的 `checkpoints/`、`data/`、`outputs/` |

模型、数据、认证信息不进入Git。当前代码只有一套加载和评价实现；具体实验选择放在协议文件，不再复制入口。

防御一入口是 `experiment.construct --method bias`，构造只读权重。防御二入口是 `experiment.defense2`，三个任务共用相同实现，固定合成输入由 `configs/defense2_inputs.json` 指定。当前主方法的完整选项和结果在README；真实图片用于冻结后的评价和单列的数据辅助恢复。

互补视图与逐类均值两项试验未带来整体改善，已移除43行试验分支，主构造恢复为Git `be57bec` 中的同一计算。试验源码可从 `a973133:experiment/defense2.py` 取回，原始权重与逐图分数在21516、41558两份SHA一致的归档中。历史配置及源码指纹保留原值。

`experiment.main_table` 负责固定模型评价，`experiment.native` 检查独立任务，`experiment.eight_task` 负责八专家融合。AGA和公开恢复在 `experiment.recover`，数据辅助蒸馏在 `experiment.distill`。TA、TIES和DARE是融合算法。

## 执行与检查

在仓库根目录安装 `requirements.txt`，作者仓库必须使用 `configs/sources.json` 中固定的提交。检查点逐文件SHA见 `configs/checkpoints.json`。完整PaRaMS的额外依赖见 `requirements-params.txt`。

先读对应结果的协议和执行版本，再运行入口。旧结果中的源码指纹属于当时版本，不应改成当前代码的指纹；已有输出不覆盖。主评价的 `--verify` 使用原128张批上下文、独立标签和固定误差限。

```bash
python -m experiment.main_table --help
python -m experiment.native --help
python -m experiment.eight_task --help
python -m experiment.report --output /tmp/posthoc-paper-tables
pytest -q
```

表格从同一份结果生成MD和TeX。新4090上的25项现有测试已通过，覆盖算法等价性、恢复输入边界、权重噪声构造和既有论文表格。完整科研评价仍以各批原始分数、逐类结果和执行记录为准。

## 历史记录

已结束的探索不保留为当前算法入口。移除路径、原SHA和Git版本见 `results/engineering_refactor/retired.json`、`snapshot.json`、`validation.json`；旧清理和迁移详情保存在这些记录及本文件的Git历史中。唯一模型、原始分数和失败证据保留在归档，缓存只在核实重复或可再生后清理。

11月4日移除7个已结束实验的一次性脚本，共591行：旧校正、匹配准备、对照准备、偏置跨任务驱动及两份特征均值检查脚本。当前入口没有导入这些文件；原源码可从 `a17a754` 的原路径取回，历史协议中的源码指纹保持原值。蒸馏导数检查仍供当前研究使用；没有将它加入主方法入口。

```bash
git show <执行提交>:<原路径>
git archive <执行提交> | tar -x -C <空目录>
```

GitHub发布对应关系见 `results/engineering_refactor/publication.json`，最近同步情况见本地 `outputs/latest_github_sync_receipt.json`。本地提交不等于已上传。
