# NBP manuscript

`main.tex` 是 CVPR 2027 双栏审稿稿件，`sec/` 是正文。`cvpr.sty` 与参考文献样式沿用官方模板。投稿前与作者确认真实的 LLM Usage Statement。

在项目根目录运行 `python paper/make_tables.py` 可从已核验记录导出 Markdown 和 TeX 表格。`python paper/make_figures.py` 生成幅度、读出下界和预测分布图。机制图在 `fig/mechanism.tex`。生成表格和图不运行模型。

主表对比原专家、PaRaMS 与 NBP，覆盖七个保护任务和两种专家规模。八专家包括 EuroSAT，准确率与平均值基于其余七个测试集。另有 B/16 三任务结果、公开末层偏置清理以及三任务的末端 MLP 替换和 ProDistill 对照。偏置清理移除了保护，AGA 与固定层系数拟合只覆盖各自定义的参数。

实验记录在 `results/phase3/bias_weight_only/`，B/16 在 `results/backbone_b16/`。原有 SUN397 融合规则对照已写入正文。其余六个保护方向的 TIES 与 DARE 对照见 `fusion_methods/extension_protocol.json` 和 `fusion_methods/completion_protocol.json`，结果通过核验后再入表。当前伙伴为固定配对，融合系数和完整八任务覆盖仍需分别核对。

PaRaMS 主文和附录、MergeGuard、Trap² 等原论文及作者实现的阅读记录见 `notes/phase3_attack_review.md`。不同模型、数据划分与训练权限的论文数值保留在原文中，稿件的定量对照使用本项目匹配条件。
