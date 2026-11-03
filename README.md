# posthoc-merge-security

研究 CLIP ViT-B/32 专家的训练后参数保护：单独使用时保持任务能力，未经授权融合及 AGA 攻击后降低受保护任务准确率。专家不重新训练。

当前主线为防御一和数值防御二。防御一由权重构造偏置噪声；防御二不使用真实构造图片：使用冻结专家、公开基座、任务 SVD 和已声明的固定分类头，研究任务功能保持与跨模块补偿。生成输入用于构造，真实图像只用于冻结后的评价。

## 从哪里看代码

在仓库根目录查看以下入口。算法实现、实验入口和结果来源分开存放。

| 内容 | 文件 | 看什么 |
|---|---|---|
| **数值防御二** | [experiment/defense2.py](experiment/defense2.py) | A 的任务 SVD 约束、B 与偏置补偿、融合分类目标 |
| 防御一 | [_impl/defenses.py](_impl/defenses.py) | `construct_bias`：视觉投影核空间中的偏置扰动 |
| AGA 与公开恢复 | [_impl/attacks.py](_impl/attacks.py) | 作者 AGA 调用、公开模块替换和权重匹配 |
| 融合 | [_impl/merge.py](_impl/merge.py) | TA、TIES、DARE |
| 实验评价 | [experiment/main_table.py](experiment/main_table.py)、[_impl/evaluate.py](_impl/evaluate.py) | 固定发布、目标与伙伴评价 |
| 最新对比结果 | [results/comparison](results/comparison) | 防御一、二与PaRaMS的攻击、伙伴和八专家结果；MD / TeX |

当前防御二由同一入口构造 MNIST、SVHN、DTD。用 `--target` 指定任务，固定输入及参数见 `configs/defense2_inputs.json`。当前三任务主方法使用 `--calibration-aware --task-reference --input-bias --noise-calibration --finite-classification`，定义和发布指纹见 [固定协议](results/data_free_compensation/finite_classification.json) 及 [结果](results/data_free_compensation/finite_classification_result.json)。已有发布直接用于评价，无需重新构造。原合成缓存路径写在配置中，新输出目录必须不存在。

互补视图、逐类均值两项试验已结束，未替换主方法，其分支已从活动入口移除。执行源码保留于 Git `a973133` 和两台服务器的原始结果，分别见 [视图对照](results/comparison/defense2_native_views.md) 与 [均值对照](results/comparison/defense2_native_means.md)。旧实验应检出其执行版本，不能用当前源码冒充复现。

防御一和防御二的当前构造均不使用真实图片。防御一的方向和幅度完全由权重确定，也不使用合成图片；三个保护任务的完整独立预测均保持不变，TA及AGA后MNIST/SVHN/DTD准确率分别为9.80%/6.70%/2.13%，伙伴准确率也明显降低。[三任务完整表](results/phase3/bias_weight_only/other_tasks/summary.md)。历史图片辅助幅度单独保留。

早期数值防御二修改最后一个MLP的A、B及输出偏置。A在所选任务方向的补空间改变，B补偿独立模型输出；构造只使用冻结权重、固定分类头和合成输入。固定TA系数为0.5，完整测试准确率如下：

| 保护任务 | 原模型独立 | 防御二独立 | TA融合 | AGA仅目标 | AGA双方 |
|---|---:|---:|---:|---:|---:|
| MNIST | 99.69% | 99.61% | 9.80% | 9.80% | 9.80% |
| SVHN | 97.46% | 96.98% | 6.70% | 6.70% | 6.70% |
| DTD | 79.41% | 76.12% | 2.13% | 2.13% | 2.13% |

独立准确率分别下降0.08、0.48、3.30个百分点。DTD曾因项目自设的0.5个百分点条件停止，本次补齐同一固定权重的融合评价；旧失败记录保留。相关论文没有统一的0.5个百分点标准，后续直接比较独立能力损失与保护效果。[尺度依据](notes/parameter_scale_and_threat_model.md)。A/B删项支持联合补偿，两项SVD单删项尚未显示准确率收益；当前A是最小范数行正交解，并未求解参数距离最大化。另列的无数据输出校正能部分恢复分类，与仅运行AGA分开报告。[方法、对照与完整结果](results/data_free_compensation/summary.md) · [核心方法审查](notes/phase2_coupling.md#oct26核心方法审查)。

随机输入校正版（4854eb28）将独立随机输入的输出校正纳入求解：独立99.54%，TA16.61%，AGA仅目标后0.26%，再校正后0.48%；联合MLP匹配后2.78%。相比上一版，校正后的保护明显改善，直接TA则由8.04%升到16.61%；已知位置公开B替换后仍为99.36%。同规则SVHN独立97.46%→95.67%，AGA后2.85%、再校正后2.36%；DTD独立79.41%→65.96%，源条件失败，未运行其融合评价。该版本尚未实现跨任务稳定保持，不替换上面的三任务方法。[完整构造对照](results/data_free_compensation/calibrated_objective.md)。

最新三任务攻击对照已完成：TIES、DARE下仍有融合保护；已知末层MLP位置、用公开基座替换B及偏置后，融合准确率恢复至99.35%、92.39%、66.81%。两种结果的恢复操作和权限不同，详见[完整对照表](results/attack_generalization_20261028/attack_comparison.md)。

同一数值防御二的[八专家融合评价](results/attack_generalization_20261028/eight_expert/comparison.md)已完成：固定TA0.2，MNIST/SVHN/DTD保护任务准确率为9.81%/6.70%/2.39%，AGA仅处理保护专家后为9.80%/6.70%/2.13%。表中同时保留无保护和伙伴结果。已补齐GTSRB、RESISC45、Cars和SUN397，现有七个完整评价数据集；EuroSAT的原始划分尚未核实，不报八任务平均。[七数据集表](results/attack_generalization_20261028/sun397/seven_dataset.md)。

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
  defense2.py                # 数值防御二的三任务构造
  native.py                  # 冻结发布的独立任务检查
  main_table.py              # 主实验：TA / 原版 AGA / 匹配基线
  eight_task.py              # 八专家融合及Unmask；当前主表已有七个数据集
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

模型清单及SHA见`configs/checkpoints.json`。模型、数据和原始分数保存在实验机器，运行时按固定协议加载；各批结果记录输入版本和来源。

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

八专家参与不等于八数据集已评价；当前数值发布已评价 MNIST、SVHN、DTD、GTSRB、RESISC45、Cars、SUN397，旧配置的三数据集范围保留。TA/TIES/DARE 是融合算法，AGA 是恢复攻击。已知家族匹配另外报告，不能把它写成原版 AGA。

## 检查

```bash
ruff check cfg.py _impl experiment tests
ruff format --check cfg.py _impl experiment tests
pytest -q
```

测试包含完整小型 Transformer 的配套换序、有限交换的穷举对照、原算法函数体记录、恢复输入边界、逻辑 Q/K/V 距离、输出防覆盖，以及论文 MD／TeX 与原结果一致性。测试不新增模型训练或改变科研结论。

其他补充实验的历史复现及当前验证范围见[工程说明](docs/engineering.md)。

[PaRaMS同条件MLP匹配](results/params_reference/other_tasks/matching.md)已完成：独立预测保持，融合目标准确率为10.72%、9.55%、2.93%，该固定匹配没有恢复PaRaMS。只匹配MLP，与另处理注意力的AGA分开报告。

[同条件蒸馏对比](results/params_reference/other_tasks/distillation.md)已完成标签核对：防御一、PaRaMS与无保护模型在固定三组训练图片上的硬标签完全相同，因此明确复用同一批既有学生。此攻击额外需要无标签图片和训练，不能与无数据AGA混列。

本次SVHN/DTD迁移只替换任务权重、分类头和该任务原有合成输入；同一数值补偿与随机输入校正目标保持不变。原任务各自的源正则和功能误差界由历史规则确定，见 `configs/defense2_inputs.json`。该批已完成。固定组件对照显示，初始补偿后的独立准确率为SVHN97.45%、DTD79.20%，主要损失出现在后续公开分类优化。`--finite-classification` 的三任务固定对照已完成独立、TA、AGA和输出校正评价：独立99.57%／96.66%／76.33%，AGA仅目标后2.99%／1.86%／2.07%，再校正10.76%／6.99%／5.00%。SVHN、DTD独立保持改善，但前两任务的校正保护弱于上一轮。每行使用同一冻结模型；完整伙伴与配对结果见 [目标调整对照](results/data_free_compensation/calibrated_objective.md#有限分类条件对照2026-11-02)，定义见 `results/data_free_compensation/finite_classification.json`。

同一新版防御二的八专家TA0.2评价已完成：MNIST/SVHN/DTD目标为24.71%／15.25%／4.84%，AGA仅目标后23.35%／11.03%／3.24%，相同AGA无保护为93.23%／72.53%／51.38%。新版目标保护弱于早期版，完整伙伴损失另列；独立专家保持与融合破坏分别评价。七个完整数据集及早期版、无保护均列在[同版本八专家表](results/comparison/defense2_eight_versions.md)（[TeX](results/comparison/defense2_eight_versions.tex)）。84行中42行复用相同无保护模型，未重训或改参数。EuroSAT原划分仍未核实，不报八任务平均。ProDistill仍可恢复较高准确率，见[同版本攻击表](results/comparison/defense2_versions.md)。

同一新版八专家模型经固定64随机输入输出校正后，AGA目标MNIST/SVHN/DTD为22.99%／17.16%／28.46%，对应无保护93.19%／72.82%／38.35%。[校正前后表](results/comparison/defense2_eight_calibration_target.md)与[七任务全表](results/comparison/defense2_eight_calibration.md)均有TeX版本。DTD恢复较多，不能把两专家下的校正抵抗直接外推到八专家；无需新增攻击类型即可检验这一范围。

[同条件八专家恢复对比](results/comparison/eight_calibration_target.md)（[TeX](results/comparison/eight_calibration_target.tex)）已补齐防御一和PaRaMS，更新版防御二使用同一冻结发布。AGA后再校正的MNIST/SVHN/DTD准确率：无保护93.19/72.82/38.35%，PaRaMS93.19/72.82/38.35%，防御一93.13/72.58/39.36%，防御二22.99/17.16/28.46%。[七任务明细](results/comparison/eight_calibration.md)保留全部伙伴结果。这是已有模型的公平比较，不是新的防御方法；防御一的校正弱点和防御二的DTD独立能力损失仍需解决。

同一新版防御二的[TIES/DARE补充](results/comparison/defense2_mergers.md)（[TeX](results/comparison/defense2_mergers.tex)）已完成：TIES下MNIST/SVHN/DTD为29.78/13.92/2.29%，DARE下26.57/2.28/2.82%，均低于匹配无保护。所有模型沿用同一冻结发布，无参数搜索；两种融合算法不算新增攻击。已有恢复弱点不因这项结果改变。
