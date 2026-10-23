# posthoc-merge-security

研究 CLIP ViT-B/32 专家的训练后参数保护：单独使用时保持任务能力，未经授权融合及原版 AGA 恢复后降低受保护任务准确率。专家不重新训练。

**最新数值版防御二：**在 MNIST / SVHN 固定组合上，独立 MNIST 准确率由99.69%到99.50%，TA融合后24.46%，原版AGA仅处理保护专家/双方后13.02%/13.09%。方法修改12层MLP的24个矩阵，用任务SVD约束和实际逐层输出重构保持能力；原专家冻结，使用1024张目标训练图校准、12次线性求解，无梯度微调。跨任务和已知模块恢复正在验证，[完整数据及限制](results/nonlinear_prefix/summary.md)。它不是下面的全网换序对照；原严格间隔检查未通过，不能称函数完全相同。

| 已有全网换序对照 | MNIST | DTD | SVHN |
|---|---:|---:|---:|
| 单独使用 | 99.69% | 79.41% | 97.46% |
| 原版 AGA 仅处理保护专家后，TA 融合 | 9.80% | 2.39% | 6.45% |

三份冻结发布的完整独立预测保持；伙伴成绩、AGA 两种作用范围及对照见[论文实验总表](results/paper_tables/README.md)。该对照使用任务 SVD 选择全网共同换序；知道变换家族后的公开 patch 匹配能够恢复，属于单列的针对性攻击，不能与原版 AGA 混淆。[方法](results/task_directed_pairing/method.md) · [恢复限制](results/task_directed_pairing/recovery/summary.md)。防御一、完整 PaRaMS、历史 O/B 和 A/B 以及所有失败结果继续保留。

## 代码结构

```text
cfg.py                       # 共同路径和配置检查
_impl/
  models.py                  # 冻结模型加载、发布参数应用
  data.py                    # 官方数据划分、独立标签读取
  defenses.py                # 防御一、数值逐层补偿、全网换序对照
  subspace.py                # 74 个任务矩阵 SVD、有限交换目标
  attacks.py                 # 原版 AGA、已知家族的公开匹配
  baselines.py               # 完整官方 PaRaMS
  merge.py                   # 原作者 TA、TIES、DARE、八专家顺序
  evaluate.py                # 双任务评价、配对统计、原批上下文重放
  metrics.py                 # 分类间隔、逻辑参数距离、McNemar
  io.py                      # JSON 与 SHA256
  report.py                  # 已验证结果生成同源 MD / TeX / 图
experiment/
  construct.py               # 防御构造及固定距离消融
  native.py                  # 冻结发布的独立任务检查
  main_table.py              # 主实验：TA / 原版 AGA / 匹配基线
  eight_task.py              # 八专家融合，当前仅评价三个数据集
  recover.py                 # 两种恢复权限分别执行
  report.py                  # 论文图表入口，无需模型或 GPU
configs/                     # 当前配置、模型和外部源码指纹
results/                     # 已验证轻量结果、表格、失败证据索引
notes/                       # 数学推导与论文核对记录
tests/                      # 算法等价性、有限目标、IO、表格回归
```

没有引入其他项目的模型、数据或算法。已结束的一次性探索代码和旧配置移出活动目录，通过[固定 Git 快照及哈希索引](results/engineering_refactor/snapshot.json)保留；恢复方式见[工程说明](docs/engineering.md)。结果文件中的历史路径和 SHA 不回写为当前代码。

## 环境

在仓库根目录运行 Python 3.12 和 `requirements.txt`。完整 PaRaMS 另需 `requirements-params.txt`；开发检查用 `requirements-dev.txt`。模型与数据只保存在实验机器，不进入 Git。

原作者 CLIP 检查点需要固定版本的 `external/secure-merge-attack`，其提交见 `configs/sources.json`；TIES 的固定源码与 SHA 在 `configs/evaluate.json`。不能用最新版替换固定版本。

```text
checkpoints/ViT-B-32/zeroshot.pt
checkpoints/ViT-B-32/{TASK}/finetuned.pt
checkpoints/ViT-B-32/head_{TASK}.pt
data/MNIST/...
data/dtd/...
data/{train,test}_32x32.mat
```

模型清单及 SHA：`configs/checkpoints.json`。历史原始分数、唯一发布权重和失败日志仍在 3090 的原路径；工程重构不移动或删除它们。

## 常用入口

仅重建论文表格，不运行新实验：

```bash
python -m experiment.report --output /tmp/posthoc-paper-tables
```

输出 MD 和 TeX 使用同一份数据；输入分数和来源记录只读。正式论文文件在 `results/paper_tables/`，不因重构覆盖。

固定模型的主实验及独立原批上下文重放：

```bash
python -m experiment.main_table --protocol configs/evaluate.json --case 0 --pilot
python -m experiment.main_table --verify outputs/refactored/main/pilot/00_MNIST_SVHN
python -m experiment.main_table --protocol configs/evaluate.json --case 0
python -m experiment.main_table --verify outputs/refactored/main/full/00_MNIST_SVHN
```

`configs/evaluate.json` 列出六个保护方向；每组同时报告目标与伙伴，原版 AGA 仅保护专家／全部专家分别列出。它们是**新入口的复现命令**，不是声称本次又跑了一遍这些实验。已完成结果无需重复运行。

防御二构造复用已冻结的任务基，不读图像或伙伴：

```bash
python -m experiment.construct --method task-directed --target MNIST --output outputs/refactored/construction/MNIST
```

`--variant distance_only` 是固定距离目标的消融，不作为另一个按成绩选择的发布。`--method basis` 从原专家减基座生成相同 rank32 任务基；`--method params` 运行完整 PaRaMS 默认配置；`--method bias` 使用原目标训练索引及其固定 SHA，具体参数见 `--help`。不自动抽取训练图或按成绩选参。

独立检查、八专家和恢复分别使用：

```bash
python -m experiment.native --help
python -m experiment.eight_task --help
python -m experiment.recover --help
```

八专家参与不等于八数据集已评价；当前配置仅评价 MNIST、DTD、SVHN。TA/TIES/DARE 是融合算法，AGA 是恢复攻击。已知家族匹配另外报告，不能把它写成原版 AGA。

## 检查

```bash
ruff check cfg.py _impl experiment tests
ruff format --check cfg.py _impl experiment tests
pytest -q
```

测试包含完整小型 Transformer 的配套换序、有限交换的穷举对照、原算法函数体记录、恢复输入边界、逻辑 Q/K/V 距离、输出防覆盖，以及论文 MD／TeX 与原结果一致性。测试不新增模型训练或改变科研结论。

本次仅做工程整理。最新读出补偿候选的完整任务失败另见[实验记录](results/centered_readout_game/summary.md)，不替代上面的有效冻结结果。其旧实现与数值失败仍可按快照恢复。其他补充实验的历史复现及当前验证范围见[工程说明](docs/engineering.md)。
