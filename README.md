# posthoc-merge-security

研究 CLIP ViT-B/32 专家的训练后参数保护：单独使用时保持任务能力，未经授权融合及 AGA 攻击后降低受保护任务准确率。专家不重新训练。

当前重点是无真实构造图片的数值防御二：使用冻结专家、公开基座、任务 SVD 和已声明的固定分类头，研究任务功能保持与跨模块补偿。生成输入用于构造，真实图像只用于冻结后的评价。

防御一和防御二的当前构造要求均为Data-Free。防御一已改为只由权重确定噪声方向和幅度，MNIST独立99.69%，融合及AGA后均为9.80%；构造没有使用真实或合成图片。[结果与方法](results/phase3/bias_weight_only/summary.md)。历史数据辅助结果单列保留。

当前数值防御二修改最后一个MLP的A、B及输出偏置。A在所选任务方向的补空间改变，B补偿独立模型输出；构造只使用冻结权重、固定分类头和合成输入。固定TA系数为0.5，完整测试准确率如下：

| 保护任务 | 原模型独立 | 防御二独立 | TA融合 | AGA仅目标 | AGA双方 |
|---|---:|---:|---:|---:|---:|
| MNIST | 99.69% | 99.61% | 9.80% | 9.80% | 9.80% |
| SVHN | 97.46% | 96.98% | 6.70% | 6.70% | 6.70% |
| DTD | 79.41% | 76.12% | 未运行 | 未运行 | 未运行 |

MNIST和SVHN满足事前独立准确率下降不超过0.5个百分点的条件，DTD未通过。A/B删项对照支持联合补偿，但两项SVD单删项尚未显示准确率收益。当前A由最小范数行正交条件确定，并未求解参数距离最大化。另列的无数据输出校正能部分恢复分类，原AGA结果与该额外处理分别报告。[方法、对照与完整结果](results/data_free_compensation/summary.md) · [核心方法审查](notes/phase2_coupling.md#oct26核心方法审查)。

| 通道置换对照 | MNIST | DTD | SVHN |
|---|---:|---:|---:|
| 单独使用 | 99.69% | 79.41% | 97.46% |
| AGA 仅处理保护专家后，TA 融合 | 9.80% | 2.39% | 6.45% |

该对照使用任务 SVD 选择共同通道顺序；三份冻结发布保持完整独立预测。知道变换家族后的公开匹配能够恢复，针对性恢复与原版 AGA 分列。[方法](results/task_directed_pairing/method.md) · [完整结果与恢复](results/task_directed_pairing/summary.md) · [论文表格](results/paper_tables/README.md)。防御一及完整 PaRaMS 对照继续保留。

## 代码结构

```text
cfg.py                       # 共同路径和配置检查
_impl/
  models.py                  # 冻结模型加载、发布参数应用
  data.py                    # 官方数据划分、独立标签读取
  defenses.py                # 防御一、通道置换对照
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

模型清单及 SHA：`configs/checkpoints.json`。当前数值实验的原始分数、发布权重及源码保存在4090，3090保留历史模型和记录；旧离线4090的唯一记录尚未全部迁回。具体机器和路径以`research_state.json`及各批结果索引为准。已归档的诊断文件采用无损压缩，路径和还原方法见工程说明。

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

通道置换对照复用已冻结的任务基，不读图像或伙伴：

```bash
python -m experiment.construct --method task-directed --target MNIST --output outputs/refactored/construction/MNIST
```

`--variant distance_only` 是固定距离目标的消融，不作为另一个按成绩选择的发布。`--method basis` 从原专家减基座生成相同 rank32 任务基；`--method params` 运行完整 PaRaMS 默认配置；`--method bias` 仅由权重计算噪声方向和幅度，不读取训练图或合成图。具体参数见 `--help`。

当前数值防御二尚未合入该构造入口。已执行源码、固定配置和模型指纹见`results/data_free_compensation/population_compensation_result.json`，服务器各批输出目录的`source_archive/`保留对应源码；不能把上述置换命令当作数值方法的复现入口。

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

其他补充实验的历史复现及当前验证范围见[工程说明](docs/engineering.md)。
