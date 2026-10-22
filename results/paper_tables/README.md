# 实验表与完成情况

**没有全部做完。** 已完成的真实结果已经生成同一份TeX和Markdown；缺项明确保留。

- [Markdown总表](tables.md)：主实验、三个保护任务、不同恢复权限、融合方法、八专家三数据集、任务内外与补偿消融、缺项清单。
- [TeX表格](tables.tex)：可取对应table段落放入论文，需要booktabs、array；宽表放在两栏论文时改为table*。
- [完整TeX预览入口](main.tex)（当前源码已通过桌面编译）。图的[TeX片段](figure.tex)、[PDF](comparison.pdf)和[PNG](comparison.png)同数据生成。
- [数值与来源](table_data.json)、[输入哈希与原记录行号](table_sources.json)。新制表没有重跑GPU，也没有冒充新一轮原始分数复核。

已完成：三个保护任务的相同TA／原版AGA下无保护、Defense1、完整默认PaRaMS和有效Defense2结构方案对比；对角适配、位置已知偏置清理、MNIST固定数据蒸馏、TIES/DARE/NUFILT/ProDistill、八专家参与的三任务评价及Unmask公式适配。

已完成同一任务SVD构造的MNIST／DTD／SVHN验证；尚缺新方法贡献的充分验证；完整八个保护任务和八数据集成绩；MergeLock、完整MergeGuard/ARF/MergeBarrier防御基线；GradErase；更多独立防御重复。MergeLock官方库此前核对仅README，其他未运行方法不填论文成绩。不同训练阶段/架构/权限须另列。完整TSV-Merge、CAT、LOT也未运行，不把融合器数量当攻击数量。

历史完整分表保留，不用本次摘要替代：

- [TA/TIES完整双任务表](../phase3/tables.md)
- [PaRaMS对角恢复](../phase3/params_adaptive/tables.md)
- [DARE三个掩码](../phase3/dare/tables.md)
- [NUFILT O/B补充](../phase3/nufilt_OB/tables.md)
- [NUFILT DTD/SVHN](../phase3/nufilt_other_tasks/tables.md)
- [ProDistill MNIST](../phase3/prodistill/tables.md)及[DTD/SVHN](../phase3/prodistill_other_tasks/tables.md)
- [公开基座/单专家知识来源对照](../phase3/controls/single/tables.md)及[固定七专家去目标对照](../phase3/controls/leave_out/tables.md)
- [位置已知清理](../phase3/bias_canonicalization/tables.md)、[单列读出精度协议](../phase3/bias_tail_precision/tables.md)、[固定数据蒸馏](../phase3/distillation/tables.md)
- [八专家三数据集](../phase3/eight_task/tables.md)、[Unmask](../phase3/unmask/tables.md)
- [先前补偿失败对照](../mean_compensation/summary.md)、[已知全网置换对照](../residual_coordinate_control/summary.md)

重建表和图：`python -m experiment.report --output results/paper_tables`。本机系统Matplotlib与用户NumPy版本不同，当前使用`PYTHONNOUSERSITE=1`调用匹配的系统环境；没有修改3090科研环境。main.tex可由桌面LaTeX编辑器编译。

当前任务方向Defense2的[已知方式恢复](../task_directed_pairing/recovery/summary.md)已完成：不使用原专家、真实任务向量、秘密排列或图片参与攻击，三模型及匹配控制全部158张量精确还原。原版AGA低分仍保留；已知方式恢复是单独结果，不能声称当前方法抵抗它。

同一三份发布的[六个保护方向](../task_directed_pairing/cross_pairs/summary.md)已完成，包含独立DTD/SVHN组合；未新增模型或按伙伴成绩调参。

同一任务方向防御二的[八专家TA0.2结果](../task_directed_pairing/eight_expert/summary.md)已完成：三个数据集及原版AGA仅保护专家，保持原发布与已知方式恢复限制；不报完整八任务平均。

[修改幅度与任务方向对照](../task_directed_pairing/amplitude_ablation/summary.md)已完成：同一固定顺序的96／192／384对交换，两种目标分别报告，0与384模型复用；不作为独立种子或新防御。
