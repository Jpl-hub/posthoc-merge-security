# posthoc-merge-security

**Defense2进展：同一任务SVD构造已完成MNIST、DTD、SVHN验证。** 独立准确率保持99.69%／79.41%／97.46%；原版AGA后分别为9.80%／2.39%／6.45%，完整测试预测均不变。构造不使用图片、不重新训练专家。见[三任务MD／TeX](results/task_directed_pairing/other_tasks/summary.md)、[方法与代码说明](results/task_directed_pairing/method.md)。MNIST固定距离消融也完成：去掉任务方向项后，直接融合13.99%、原AGA11.27%／11.08%，加入后为10.98%／9.80%／9.80%；这是一个固定消融，不宣称所有排序都更好。原固定结构和失败对照保留，方法贡献与针对性恢复权限单独研究。

**已知方式恢复的限制已实际验证。** 只给发布权重和公开基座，知道全网换序方式，不给原专家、真实任务向量、秘密排列或图片，当前三个模型及匹配无保护控制均全部158张量精确还原。匹配后再原AGA仅目标，目标准确率恢复至99.30%／68.40%／91.56%；因状态精确相同，引用已核原分数，不称新的图片前向。当前方案能防住已测原版AGA，不能抵抗这项针对性恢复，两者并排保留。[MD／TeX恢复表](results/task_directed_pairing/recovery/summary.md)、[方法贡献和权限](results/task_directed_pairing/contribution.md)。不通过改排序或增大距离回避这项结果。

同一三份发布已补齐[全部六个保护方向](results/task_directed_pairing/cross_pairs/summary.md)，包含DTD/SVHN独立组合。新增原AGA仅目标后：MNIST/DTD为9.80%，DTD/SVHN为2.34%，SVHN/DTD为6.53%；伙伴成绩、双方AGA和各自无保护控制均完整保留。全部新分数按原128张批上下文重放通过，最大误差0。这里是三种无序任务组合，不是六个独立数据集或完整八任务评价。
[论文实验总表：TeX、Markdown、PDF及缺项](results/paper_tables/README.md)

2026-10-22：同一规则新增DTD和SVHN保护验证。DTD单独79.41%、TA2.23%、原AGA仅目标／双方2.18%／2.13%；SVHN单独97.46%、TA7.74%、原AGA9.74%／9.63%。完整测试预测均保持，独立核验通过。三个保护任务对应两个无序组合；不是三组独立伙伴，也不因此声称新算法原创性。[三任务MD／TeX／图](results/residual_coordinate_control/other_tasks/summary.md)。

同日补齐DTD、SVHN的完整默认PaRaMS基线：独立预测均保持，原AGA仅目标／双方后，DTD恢复至68.35%／68.78%，SVHN恢复至91.56%／91.77%；固定结构Defense2同设置仍为2.18%／2.13%和9.74%／9.63%。未缩小PaRaMS默认参数范围或按测试选参。[完整基线MD／TeX](results/params_reference/other_tasks/summary.md)，[统一主表](results/paper_tables/tables.md)。原始分数、逐类配对及原批上下文核验均已完成；任务方向新构造的三任务验证已完成，新算法贡献仍需与已有工作比较。

研究已微调 CLIP ViT-B/32 专家的发布前参数后处理保护：保持独立任务能力，评价未经授权融合及恢复攻击后的任务能力。

当前围绕两套防御：**投影核空间偏置扰动、任务子空间约束的跨模块补偿**，同时补齐攻击调研与对比实验。历史发布权重冻结；O/B、A/B均有任务预算反例，不能宣称严格可行。新方法须先完成创新性审查及源端协议，再独立确认。

当前只使用RTX3090，暂不访问4090。已核验TA/TIES、原AGA、PaRaMS对角恢复适配、NUFILT、三种子DARE及ProDistill两专家适配。融合算法与恢复攻击分别记录，不把结果行数当攻击种类数。

2026-10-20：**已验证全网特征坐标换序保持MNIST独立99.69%，直接融合降至7.39%，原版AGA后融合为10.09%。** [完整双任务结果](results/residual_coordinate_control/summary.md)与17行／272上下文重放已核验。它是已知结构的有效对照，尚非原创Defense2；知道置换类型后的公开基座匹配可精确恢复158张量，另列为针对性恢复，不混入原AGA结论。[最新联合优化点](results/recovery_feedback/diagnostic/summary.md)则无明显保护，失败保留。

[参数改动与攻击权限调研](notes/parameter_scale_and_threat_model.md)：原D≤0.5是本项目约束，不是PaRaMS、MergeLock等论文的统一要求。此后主比较保留完整论文配置，按独立能力、真实TA及原AGA效果评价；原D0.5对照及功能R0.5检查保持原样。不会仅因参数距离超过0.5否定有效结果，也不会把已知变换或针对性恢复改称原创或原版AGA。本次未新增模型训练。

- [攻击调研、论文与源码核对、权限及评价协议](notes/phase3_attack_review.md)
- [Defense 2机制、任务内外消融与新任务预算反例](notes/phase2_coupling.md)
- [4090已归档重复文件清理记录](results/phase3/4090_cleanup.json)
- [冻结模型、源码、数据与参数指纹](configs/phase3.json)
- [已独立复核的结果主表](results/phase3/tables.md)
- 补充固定批次：[PaRaMS恢复适配](results/phase3/params_adaptive/tables.md) · [DARE三种子](results/phase3/dare/tables.md) · [O/B与NUFILT](results/phase3/nufilt_OB/tables.md) · [ProDistill MNIST](results/phase3/prodistill/tables.md) · [DTD/SVHN扩展](results/phase3/prodistill_other_tasks/tables.md)
- 最新扩展：[八专家、三任务评价](results/phase3/eight_task/tables.md) · [去目标向量对照](results/phase3/controls/leave_out/tables.md) · [NUFILT的DTD/SVHN缺失单元](results/phase3/nufilt_other_tasks/tables.md) · [统一双任务恢复差距表与论文图](results/phase3/recovery_overview/tables.md)
- [组会要求与已完成／未完成对应](notes/teacher_feedback.md)：已有有效Defense2结构方案；任务SVD参与构造的新版本已完成三任务及固定距离消融，方法贡献仍需推进。
- [公开基座／单专家知识来源对照](results/phase3/controls/single/tables.md)：九条新跨任务评价，三条原任务分数核哈希复用；不把任务准确率恢复等同于独有知识提取。
- 偏置恢复：[原生FP32的MNIST结果](results/phase3/bias_canonicalization/tables.md) · [单列混合精度的DTD/SVHN结果](results/phase3/bias_tail_precision/tables.md)
- [逐类计数、配对统计及完整指标](results/phase3/results.json) · [CSV](results/phase3/results.csv) · [复核与绘图指纹](results/phase3/provenance.json)
- [研究状态](research_state.json) · [相关工作与创新性审查](notes/phase2_related_work_and_novelty.md)
- [科研汇报 PDF](presentation/main.pdf) · [PowerPoint放映版](presentation/research_report.pptx) · [LaTeX源稿](presentation/main.tex)。运行`python3 presentation/update_tables.py`从结果文件更新表格。

## 冻结对照

| 方法 | 证据及限制 |
|---|---|
| 无保护专家 | 各融合方法、任务组合和攻击范围分别提供匹配控制 |
| Phase 1 最终 LN 偏置 | 已有 MNIST、DTD、SVHN；不同预算参考。原版 AGA 未修改关键偏置，不能据此声称抗任意恢复 |
| Phase 2 O/B | 有真实融合干扰；已核验任务漂移0.789627超过0.5，只作效果参考 |
| Phase 2 A/B | 既往有限确认通过；本阶段完整模型已重放任务漂移39.920的越界见证，只作效果参考 |
| PaRaMS 完整官方配置 | 完整作者配置基线；函数保持，距离另报，不因超过项目D0.5排除出论文比较 |
| PaRaMS Attention 消融 | 官方组件的预算内消融，不是完整方法的同预算复现 |

[Phase 1 数学机制与历史结果](notes/projection_bias_noise.md)、[O/B 构造与历史结果](notes/phase2_boundary.md)、[预算越界更正](notes/phase2_finite_signed_condition.md)、[本阶段 O/B 与 A/B 完整模型反例](results/phase3/utility_witnesses.json)、[官方 PaRaMS 与 AGA 协议核对](notes/phase2_params_and_structure.md)。其他历史实验、权重和失败证据保留在原目录；本页不再重复逐轮日志。

## 评价范围

TA、TIES 属于融合算法；AGA 属于参数对齐恢复攻击。分别报告 **No attack / AGA (protected expert) / AGA (all experts)**，每项同时列被保护任务和伙伴任务。

固定组合为 MNIST→SVHN、MNIST→DTD、DTD→MNIST、SVHN→MNIST。前两组只更换 MNIST 的伙伴；后两组评价已有 Phase 1 专家，不能当作 A/B 的跨被保护任务验证。历史测试集已多次查看，本阶段不是首次盲测。

TA 系数0.5；TIES 使用固定官方纯函数，保留前20%、mass 符号、disjoint mean、系数1.0。AGA 使用固定作者逐头 float64 求解、1e-8正则和 MLP 匈牙利匹配。不同范围及不同预算不混同。

NUFILT 已完成固定官方内核与 native checkpoint 对应验证（MNIST/SVHN、两个顺序、三种范围）；它是融合算法。Phase 1 在该设置下与无保护模型完全相同，A/B 的 MNIST 也恢复至98.86%–99.33%。MergeLock、MergeGuard、MergeBarrier 中尚未获得可核验官方执行版本的内容只列论文分析，不填自产复现成绩。完成状态以结果与源码指纹为准。

AGA后NUFILT的O/B目标恢复至98.92%–99.24%。ProDistill使用每任务64张无类别标签训练图及任务归属、只优化攻击方融合系数；O/B与A/B的目标恢复至约99.42%。这是更高数据权限下的结果，不能与无数据AGA混为同一攻击设置。逐元素系数使块内8505万个坐标在代数上全部可达，累计1.70亿系数按块训练；这不是低维标量合并，也不意味着数值上任意目标可实现。该ProDistill实现不更新最终LN/P，Phase1仍低分不代表抗覆盖这些位置的恢复。

PaRaMS新增恢复使用官方MLP匹配内核及明确配对逆尺度的Attention适配，不是完整作者攻击代码复现。在MNIST/SVHN上，完整PaRaMS参考恢复至99.37%／92.50%；Phase1仍为9.80%／6.70%，但此攻击同样不改最终LN偏置，不能据此称通用抗恢复。O/B、A/B的任务预算反例仍成立，没有新防御模型。

## 复现

每个结果使用其固定源码与协议；旧源码哈希对应历史Git版本，不能直接用最新HEAD复跑旧协议。以下为原TA/TIES批次的命令形式：

```bash
python experiments/phase3_evaluate.py --pilot --case 0
python experiments/phase3_evaluate.py --verify outputs/phase3/pilot/00_MNIST_SVHN
python experiments/phase3_evaluate.py --case 0
python experiments/phase3_evaluate.py --verify outputs/phase3/full/00_MNIST_SVHN
python experiments/phase3_evaluate.py --audit
python experiments/phase3_report.py
```

其余组合使用 `--case 1/2/3`。输出目录不得覆盖。若需匹配原评价的128张批上下文重放，使用 `python experiments/phase3_replay.py <已完成组合目录>`；保留原16个抽样索引和0.0005分数误差上限，记录额外前向数量。原始16张重放失败记录保留。

NUFILT 的只读源码检查为 `python experiments/phase3_nufilt_check.py`；对应验证为 `python experiments/phase3_nufilt.py`，通过后 `--full` 使用同一批冻结合并权重，不重新优化。

新增4090恢复实验使用`configs/params_adaptive.json`及`experiments/params_adaptive.py`，按`--pilot`、`--verify`、`--full`、`--verify --full-output`依次执行；目录已完成时不能覆盖。`python experiments/params_adaptive_report.py`只从已验证结果生成新对照表，不修改旧表或保护参数。

## 文件组织

- `experiments/`：保护、评价、独立验证及绘图源码。
- `configs/`：固定版本、预算、数据与模型指纹。
- `notes/`：数学推导、论文核对和历史结论。
- `results/phase3/`：轻量指标、表格及 PDF/SVG 论文图。
- 服务器 `outputs/phase3/`：原始分数、重权重、门控、运行日志与原论文缓存，不上传 GitHub。

不以参数距离、代理改判或单一任务下降代替真实保护结论。不根据 Phase 3 的测试或攻击结果重新选择防御参数。

本轮显式任务内／外／联合构造保留原0.5与另列2.0幅度参考。任务外0.5虽通过新鲜512张确认，完整独立评价仍出现5.942429的任务预算反例，已关闭，无新合规防御或完整AGA融合结果。详见[六点消融与反例](results/postactivation_subspace/)。

偏置规范化恢复已在MNIST两伙伴的原生FP32评价，以及DTD/SVHN保护任务的单列FP32前缀／FP64最终读出协议下完成。后者不能撤销原生FP32数值控制失败；原分数、阈值、失败证据及两套精度结果分别保留。MNIST原生FP32清理前后分数差0.00069427也超过0.0005；恢复模型重放误差0是另一项检查。这是架构相关补充诊断，不是原版AGA或新的论文攻击算法。

历史汇报文件保留，其数据截止与预算说明不代表本次最新结论。最新研究进度见[老师要求落实与缺口](notes/teacher_feedback.md)。[固定Unmask全表](results/phase3/unmask/tables.md)显示无保护控制也退化，不将低分误记为防御成功。

2026-10-20清理：当前树移除5个已完成的一次性归档／汇总脚本，共384行；构造、攻击、评价与独立验证入口保留。历史归档脚本可从固定提交 `705af4d606d728dc49372ddd2f72aeff25ed6ec2` 取回，不新增归档目录或替代框架。删除可再生预览与缓存、合并逐字节相同副本；6.70GB唯一诊断证据迁到3090系统盘，原路径及SHA保持。详见[清理收据](results/cleanup_20261020.json)。

状态文件只保留当前任务与证据索引；此前累计状态保存在Git提交 `281be3fc01d6c63d6194c2ef4849bcdadcf1135a`。实验配置、结果和失败证据仍在原目录。


## 科研工程约定

沿用现有公共加载、TA／原版AGA评价及独立重放入口；新方法只增加必要的算法部分，不为每个方案复制一套评价流程，也不另造训练平台。

- 每项实验先写清要回答的问题，再固定配置、源码版本、检查点、数据划分和种子含义。输出已有时停止，不覆盖。
- 缓存复用同时核对模型、配置、预处理、索引和文件哈希；只有文件存在不构成复用依据。
- 原始分数及逐类、配对统计留在3090；轻量配置、独立核验、成本和失败结论进入Git。图表由同一份结果生成，均值和标准差注明重复对象。
- 历史方法按当时Git版本复现。已关闭且不被共享流程导入的独立检查入口从当前树移除，保留固定源码提交和恢复路径；模型及失败证据不随之删除。

2026-10-22移除17个历史独立检查入口，共1279行；共享依赖检查通过，GitHub原始源码及3090对应文件逐项核对。详见[清理记录](results/cleanup_20261022.json)。
