# posthoc-merge-security

研究已微调 CLIP ViT-B/32 专家的发布前参数后处理保护：保持独立任务能力，评价未经授权融合及恢复攻击后的任务能力。

当前围绕两套防御：**投影核空间偏置扰动、任务子空间约束的跨模块补偿**，同时补齐攻击调研与对比实验。历史发布权重冻结；O/B、A/B均有任务预算反例，不能宣称严格可行。新方法须先完成创新性审查及源端协议，再独立确认。

主力GPU为RTX3090，4090只作CPU存储。已核验TA/TIES、原AGA、PaRaMS对角恢复适配、NUFILT、三种子DARE及ProDistill两专家适配。融合算法与恢复攻击分别记录，不把结果行数当攻击种类数。

- [攻击调研、论文与源码核对、权限及评价协议](notes/phase3_attack_review.md)
- [Defense 2机制、任务内外消融与新任务预算反例](notes/phase2_coupling.md)
- [4090已归档重复文件清理记录](results/phase3/4090_cleanup.json)
- [冻结模型、源码、数据与参数指纹](configs/phase3.json)
- [已独立复核的结果主表](results/phase3/tables.md)
- 补充固定批次：[PaRaMS恢复适配](results/phase3/params_adaptive/tables.md) · [DARE三种子](results/phase3/dare/tables.md) · [O/B与NUFILT](results/phase3/nufilt_OB/tables.md) · [ProDistill MNIST](results/phase3/prodistill/tables.md) · [DTD/SVHN扩展](results/phase3/prodistill_other_tasks/tables.md)
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
| PaRaMS 完整官方配置 | 函数保持，但参数距离超出0.5；不同约束参考 |
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
