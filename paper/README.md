# NBP manuscript

`main.tex` 是 CVPR 2027 双栏审稿稿件，`sec/` 是正文。`cvpr.sty` 与参考文献样式沿用官方模板。投稿前与作者确认真实的 LLM Usage Statement。

在项目根目录运行 `python paper/make_tables.py` 可从已核验记录导出 Markdown 和 TeX 表格。`python paper/make_figures.py` 生成逐任务融合结果点图、幅度、读出下界和预测分布图。机制图在 `fig/mechanism.tex`。生成表格和图不运行模型。

主表对比原专家、PaRaMS 与 NBP，覆盖七个保护任务和两种专家规模。八专家包括 EuroSAT，准确率与平均值基于其余七个测试集。另有 B/16 三任务结果、公开末层偏置清理、三任务方向消融，以及三任务的末端 MLP 替换和 ProDistill 对照。偏置清理移除了保护，AGA 与固定层系数拟合只覆盖各自定义的参数。

实验记录在 `results/phase3/bias_weight_only/`，B/16 在 `results/backbone_b16/`。七个保护方向的 TIES 与 DARE 对照已核验并进入正文均值表，逐任务表在 `tables/fusion_tasks.tex` 和同名 Markdown 中。原始计数导出在 `results/phase3/bias_weight_only/fusion_methods/extension/comparison.csv`。等权八专家与原系数0.2的三任务对照也已入表。七任务的42个有向配对已完成，其中35个组合为本轮新增完整评价。完整计数见 `results/phase3/bias_weight_only/partner_analysis/comparison.csv`。EuroSAT 原专家测试划分尚未核实，不报告八任务平均。

PaRaMS 主文和附录、MergeGuard、Trap² 等原论文及作者实现的阅读记录见 `notes/phase3_attack_review.md`。不同模型、数据划分与训练权限的论文数值保留在原文中，稿件的定量对照使用本项目匹配条件。

正文使用三张单栏图：方法原理图、七任务融合结果点图和幅度曲线。只有包含全部任务及基线的主结果表跨双栏。读出下界图、预测频数图和逐任务融合表随源文件保留，便于后续补充材料使用。

同机 CPU 构造计时已完成：三个任务的 NBP 为0.82—0.86秒，作者默认 PaRaMS 为179.57—187.70秒；每条件一次调用，计时边界见 `tables/cost.md`。`experiment.construct --benchmark` 只写计时元数据，不保存新的模型发布。
