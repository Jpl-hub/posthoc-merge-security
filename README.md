# posthoc-merge-security

研究 CLIP ViT-B/32 专家的训练后参数保护：单独使用时保持任务能力，未经授权融合及 AGA 攻击后降低受保护任务准确率。专家不重新训练。

当前主线为防御一和数值防御二。防御一由权重构造偏置噪声；防御二不使用真实构造图片：使用冻结专家、公开基座、任务 SVD 和已声明的固定分类头，研究任务功能保持与跨模块补偿。生成输入用于构造，真实图像只用于冻结后的评价。

## 从哪里看代码

当前计算机：3090，SSH 端口 **55548**。在编辑器中打开 **`/root/autodl-tmp/posthoc-merge-security`**，先看下表；`merge_security_20261009` 是历史资产目录。

| 内容 | 文件 | 看什么 |
|---|---|---|
| **数值防御二** | [experiment/defense2.py](experiment/defense2.py) | A 的任务 SVD 约束、B 与偏置补偿、融合分类目标 |
| 防御一 | [_impl/defenses.py](_impl/defenses.py) | `construct_bias`：视觉投影核空间中的偏置扰动 |
| AGA 与公开恢复 | [_impl/attacks.py](_impl/attacks.py) | 作者 AGA 调用、公开模块替换和权重匹配 |
| 融合 | [_impl/merge.py](_impl/merge.py) | TA、TIES、DARE |
| 实验评价 | [experiment/main_table.py](experiment/main_table.py)、[_impl/evaluate.py](_impl/evaluate.py) | 固定发布、目标与伙伴评价 |
| 最新对比结果 | [results/attack_generalization_20261028](results/attack_generalization_20261028) | `attack_comparison.md` / `.tex`，攻击权限及完整数值 |

`experiment/defense2.py` 从产生当前 MNIST 发布的原始源码整理而来，只调整排版、注释和程序入口，计算语句保持相同。它是固定 MNIST 构造，不是三个任务通用命令：运行需要原合成缓存 `outputs/native_ellipse_classification_20261026/source/` 和 `outputs/task_shaped_compensation_20261026/layer_inputs.pt`，输出目录必须尚不存在。当前评价直接加载已冻结权重，无需重新构造。SVHN、DTD 的实际执行源码及哈希仍在 `results/data_free_compensation/population_compensation_result.json` 的对应任务定义中。

防御一和防御二的当前构造均不使用真实图片。防御一的方向和幅度完全由权重确定，也不使用合成图片；三个保护任务的完整独立预测均保持不变，TA及AGA后MNIST/SVHN/DTD准确率分别为9.80%/6.70%/2.13%，伙伴准确率也明显降低。[三任务完整表](results/phase3/bias_weight_only/other_tasks/summary.md)。历史图片辅助幅度单独保留。

当前数值防御二修改最后一个MLP的A、B及输出偏置。A在所选任务方向的补空间改变，B补偿独立模型输出；构造只使用冻结权重、固定分类头和合成输入。固定TA系数为0.5，完整测试准确率如下：

| 保护任务 | 原模型独立 | 防御二独立 | TA融合 | AGA仅目标 | AGA双方 |
|---|---:|---:|---:|---:|---:|
| MNIST | 99.69% | 99.61% | 9.80% | 9.80% | 9.80% |
| SVHN | 97.46% | 96.98% | 6.70% | 6.70% | 6.70% |
| DTD | 79.41% | 76.12% | 2.13% | 2.13% | 2.13% |

独立准确率分别下降0.08、0.48、3.30个百分点。DTD曾因项目自设的0.5个百分点条件停止，本次补齐同一固定权重的融合评价；旧失败记录保留。相关论文没有统一的0.5个百分点标准，后续直接比较独立能力损失与保护效果。[尺度依据](notes/parameter_scale_and_threat_model.md)。A/B删项支持联合补偿，两项SVD单删项尚未显示准确率收益；当前A是最小范数行正交解，并未求解参数距离最大化。另列的无数据输出校正能部分恢复分类，与仅运行AGA分开报告。[方法、对照与完整结果](results/data_free_compensation/summary.md) · [核心方法审查](notes/phase2_coupling.md#oct26核心方法审查)。

最新三任务攻击对照已完成：TIES、DARE下仍有融合保护；已知末层MLP位置、用公开基座替换B及偏置后，融合准确率恢复至99.35%、92.39%、66.81%。两种结果的恢复操作和权限不同，详见[完整对照表](results/attack_generalization_20261028/attack_comparison.md)。

同一数值防御二的[八专家融合评价](results/attack_generalization_20261028/eight_expert/comparison.md)已完成：固定TA0.2，MNIST/SVHN/DTD保护任务准确率为9.81%/6.70%/2.39%，AGA仅处理保护专家后为9.80%/6.70%/2.13%。表中同时保留无保护和伙伴结果。已补齐GTSRB和RESISC45，现有五个完整数据集；Cars、EuroSAT、SUN397仍缺失，不报八任务平均。

参数置换退出后续研究，不再追加实验。既有结果只作[历史对照](results/task_directed_pairing/summary.md)，保留恢复实验所需的正对照。

[Unmask补充](results/attack_generalization_20261028/unmask/comparison.md)也已完成。它对保护和匹配无保护模型产生相同预测，不能据此再算一次防御成功。公开模块替换仍能恢复，攻击泛化问题尚未解决。

[数据辅助恢复](results/attack_generalization_20261028/distillation/comparison.md)已完成MNIST、SVHN、DTD：攻击者增加无标签训练图和50轮学生训练后，TA融合准确率分别为95.53%、85.84%、58.56%；同预算无保护对照为95.53%、86.22%、58.19%。MNIST标签完全相同并复用同一学生；SVHN/DTD各实际训练两个学生。该结果只针对额外数据与训练权限，不与无数据AGA混列，也未完全恢复原专家融合性能。

## 代码结构

```text
cfg.py                       # 共同路径和配置检查
_impl/
  models.py                  # 冻结模型加载、发布参数应用
  data.py                    # 官方数据划分、独立标签读取
  defenses.py                # 防御一；保留历史置换正对照
  subspace.py                # 74 个任务矩阵 SVD、有限交换目标
  attacks.py                 # 原版 AGA、已知家族的公开匹配
  baselines.py               # 完整官方 PaRaMS
  merge.py                   # 原作者 TA、TIES、DARE、八专家顺序
  evaluate.py                # 双任务评价、配对统计、原批上下文重放
  metrics.py                 # 分类间隔、逻辑参数距离、McNemar
  io.py                      # JSON 与 SHA256
  report.py                  # 已验证结果生成同源 MD / TeX / 图
experiment/
  construct.py               # 防御一及已有基线构造
  defense2.py                # 当前数值防御二的固定 MNIST 构造源码
  native.py                  # 冻结发布的独立任务检查
  main_table.py              # 主实验：TA / 原版 AGA / 匹配基线
  eight_task.py              # 八专家融合及Unmask，当前评价五个数据集
  recover.py                 # 权重恢复与公开匹配
  distill.py                 # 固定数据预算的教师标签恢复，攻击方训练
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

模型清单及 SHA：`configs/checkpoints.json`。当前计算使用55548的3090；12997为CPU存储，保留上一批原始分数。55548上的八专家及分类头已通过原清单核对，完整八数据集仍未齐全。具体资产与备份以`research_state.json`及各批结果索引为准。

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

防御一只由权重构造，不读真实图片或合成图片。SVHN和DTD已完成，固定定义见[两任务实验](results/phase3/bias_weight_only/other_tasks/protocol.json)。

```bash
python -m experiment.construct --method bias --target SVHN --protocol results/phase3/bias_weight_only/other_tasks/protocol.json --output outputs/new_bias_SVHN
```

数值防御二见`experiment/defense2.py`；原执行源码、固定配置和模型指纹见`results/data_free_compensation/population_compensation_result.json`。`construct`中的置换入口仅用于历史复现和恢复正对照。

独立检查、八专家和恢复分别使用：

```bash
python -m experiment.native --help
python -m experiment.eight_task --help
python -m experiment.recover --help
```

八专家参与不等于八数据集已评价；当前数值发布已评价 MNIST、SVHN、DTD、GTSRB、RESISC45，旧配置的三数据集范围保留。TA/TIES/DARE 是融合算法，AGA 是恢复攻击。已知家族匹配另外报告，不能把它写成原版 AGA。

## 检查

```bash
ruff check cfg.py _impl experiment tests
ruff format --check cfg.py _impl experiment tests
pytest -q
```

测试包含完整小型 Transformer 的配套换序、有限交换的穷举对照、原算法函数体记录、恢复输入边界、逻辑 Q/K/V 距离、输出防覆盖，以及论文 MD／TeX 与原结果一致性。测试不新增模型训练或改变科研结论。

其他补充实验的历史复现及当前验证范围见[工程说明](docs/engineering.md)。
