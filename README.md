# posthoc-merge-security

研究已微调 CLIP ViT-B/32 专家的发布前参数后处理保护：保持独立任务能力，评价未经授权融合及恢复攻击后的任务能力。

当前围绕两套防御：**投影核空间偏置扰动、任务子空间约束的跨模块补偿**，同时补齐攻击调研与对比实验。本批244行结果、3904次完整模型抽样重放已通过；O/B、A/B均发现任务预算反例，不能宣称严格可行。历史发布权重冻结，新方法须另立源端构造与独立确认协议。

2026-10-16组会后已获准改用RTX4090；当前等待登录信息，尚未在4090启动作业。3090无GPU，只保留CPU归档用途；既有结果仍明确记为3090。

- [攻击调研、论文与源码核对、权限及评价协议](notes/phase3_attack_review.md)
- [冻结模型、源码、数据与参数指纹](configs/phase3.json)
- [已独立复核的结果主表](results/phase3/tables.md)
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

## 复现

在已有3090环境、已登记 checkpoint 和数据目录中：

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

## 文件组织

- `experiments/`：保护、评价、独立验证及绘图源码。
- `configs/`：固定版本、预算、数据与模型指纹。
- `notes/`：数学推导、论文核对和历史结论。
- `results/phase3/`：轻量指标、表格及 PDF/SVG 论文图。
- 服务器 `outputs/phase3/`：原始分数、重权重、门控、运行日志与原论文缓存，不上传 GitHub。

不以参数距离、代理改判或单一任务下降代替真实保护结论。不根据 Phase 3 的测试或攻击结果重新选择防御参数。
