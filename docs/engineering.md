# 工程整理与历史复现

此次整理基于本项目现有实验，不引入 IQA 项目的内容。公共算法放 `_impl/`，实验顺序与命令行放 `experiment/`，科学选择放 `configs/`；不再为每个探索点复制一套加载、融合和核验脚本。

当前代码入口见根目录README。55548的工作目录固定为`/root/autodl-tmp/posthoc-merge-security`；历史资产保留在`merge_security_20261009`并链接使用。当前数值防御二源码为`experiment/defense2.py`：从已执行MNIST源码提取，补排版、注释及main入口，计算语句AST相同；没有改变算法或重跑构造。原源码、SVHN/DTD任务定义和源码SHA继续保留在原结果JSON。原始合成缓存是构造的必要输入，打开源码查看无需这些缓存。

## 保留什么

- 当前防御一、防御二、原版 AGA、单列公开匹配、完整 PaRaMS、TA/TIES/DARE 核和八专家顺序保留在活动代码中。
- 原始模型、唯一冻结发布、原始分数、数据索引、失败记录至少保留一份；历史重数据留3090，新计算与原分数留4090。已结束的诊断文件可无损压缩，解压后的字节和原SHA必须相同，归档路径另行记录。
- 现有 `results/`、`notes/` 和历史汇报文件保留。重构不修改历史准确率、阈值、源码 SHA 或失败状态。
- NUFILT、ProDistill、混合精度清理以及已关闭的补偿探索不另造新实现。当前数值发布的蒸馏恢复放在experiment/distill.py，复用Git1701f09的原训练函数，评价继续走experiment/main_table.py。它们的已验证结果继续保留，原工程与协议使用历史版本复现。

## 历史代码怎样取回

整理前完整本地提交：

```text
3e6be4c4fe0d2cba86f540b9a4f2ec5fcc126282
```

逐文件 SHA256、Git blob 和旧行数在 `results/engineering_refactor/snapshot.json`，移出当前树的具体路径在 `retired.json`。代码与配置原字节已经核对；不是把失败实验证据删掉。GitHub 对应的完整快照为 `c44c1ff184b126dbec7c776c79f12c3ab144e2f7`，目录树逐项一致；全部阶段映射见 `results/engineering_refactor/publication.json`。补传日期不会冒充实验执行前的公开登记。

例如查看某个旧入口：

```bash
git show c44c1ff184b126dbec7c776c79f12c3ab144e2f7:experiments/aga_coordinate_audit.py
```

需要完整恢复时，在另一个目录取出历史树，避免覆盖当前工程：

```bash
mkdir -p /tmp/posthoc-history
git archive c44c1ff184b126dbec7c776c79f12c3ab144e2f7 | tar -x -C /tmp/posthoc-history
```

**整理前最新代码也不等于每项旧实验的执行版本。** 真正复现时，先读对应结果里的协议和源码历史，切到它记录的执行提交，核对模型、源码、配置和索引 SHA，再使用原入口。原始数据、模型和输出根目录须在实验机器上指向原来保留的资产；不要向已存在的实验目录重写文件。

Oct25已结束的SVD删除、共同响应和分类响应构造亦按此方式复现：定义提交分别为`e3939e5906f4077a73083dcaad1c93fe610b423b`、`8d0554389c99d8a1a4ab39a68c8d7a3a8f96b4db`、`364d1a815cef949837cb21e01a89dd85469a4237`（本地执行前提交，公开补传不冒充提前登记）。两份失败构造和一份条件求解的重复运行源码已退役；完整原文及SHA分别在`no_task_result.json`、`shared_response_result.json`和`class_response_result.json`，4090原始记录与唯一模型不删。活动入口中没有增加这些失败方法。

任务方向方法的详细执行版本在 `results/task_directed_pairing/frozen_source_history.json`；更早补充实验以各自 `protocol.json`／`execution.json`／独立核验文件为准。历史记录中的 `experiments/` 路径属于这些固定版本，不伪造为当前 `_impl/` 路径。

## 当前命令与历史命令

| 原职责 | 当前入口 |
|---|---|
| 任务 SVD、任务方向／固定距离构造 | `experiment.construct`、`_impl.subspace` |
| 最终 LN 核空间偏置 | `experiment.construct --method bias`、`_impl.defenses` |
| 冻结发布的独立任务检查 | `experiment.native` |
| 两专家 TA／原版 AGA 作用范围评价 | `experiment.main_table` |
| 八专家固定顺序及三个数据集评价 | `experiment.eight_task` |
| AGA / 已知家族公开匹配 / MLP联合权重匹配 | `experiment.recover --method aga / patch / mlp-match` |
| 当前统一论文表 | `experiment.report` |
| 固定攻击泛化对比表 | `experiment.report --attack-directory results/attack_generalization_20261028` |
| 已结束探索及其他历史补充表 | 固定历史提交与原协议 |

`experiment.native` 的原始分数独立重计与主实验的原批上下文重放是两个检查，不能混称。主实验 `--verify` 保留128张原批上下文和原5e-4误差上限。单独功能检查继续记录原R0.5和改判，不改写历史失败。

## 维护约定

1. 只有一个共享模型加载器和一套主评价流程。新方法优先增加纯参数函数，不复制入口或引入注册框架。
2. 构造不能读取伙伴／测试来选参；发布先固定哈希，再检查功能、融合和恢复。专家不重新训练。
3. 配置写出科学选择与权限；程序拒绝覆盖已有输出，外部作者代码和检查点须核 SHA。新源码指纹属于重构后的新协议，不替换旧协议指纹。
4. 读出表格只写指定输出目录，不改原始数据或来源记录。论文语言与指标按已读论文表达，不另造名称。
5. 删除结束的一次性入口前记录 Git 版本和哈希；唯一证据至少留一份。缓存仅在确认可再生或重复后清理。

本次检查结果及确切删减数量在 `results/engineering_refactor/validation.json`。连接恢复后已在3090通过18项测试、Ruff检查和真实模型核验：三份发布逐张量相同，原版AGA恢复权重全部158张量相同，8个原128张批上下文的分数最大误差0。仅新增1024张工程核验前向及一次原版AGA，不称完整科研实验重跑；详见GPU_validation.json。已有科研实验结果仍来自此前冻结代码。

Oct23清理：删除已结束Oct18汇报专用的`presentation/export_pptx.py`和`presentation/update_tables.py`，共144行。当前实验／配置／测试没有依赖；源码可从Git 5b3c71951fdd51ab554eff118b68700af9157b62 取回，SHA见validation.json。旧TeX/PDF/PPT与build.json保留，当前论文只使用experiment.report，不新增汇报入口。

清理已结束分析入口：8个历史results目录的独立Python脚本及两个已关闭重构变体从活动代码移除，版本/SHA见validation.json；原结果、唯一模型、逐图分数和固定协议均保留。旧定义请从Git ad5800c复现，当前主构造、SVD/前缀消融和统一评价入口保留。


Oct23断连期间的源码与结果已补同步，37份历史快照均与本地Git树一致；映射见`results/engineering_refactor/publication.json`的`oct23_source_and_result_history`。这是恢复连接后的补发，不是更早的公开预登记。当前无数据响应计算没有新防御模型或分类成绩，带真实训练图的数值结果继续保留为历史参考。


Oct23书面报告收尾：将Oct18旧PPT/TeX/PDF及构建记录从当前目录移除，完整文件留在Git `cc0973910d44ee742e363921e2127518cd093a79`。同时移除已结束的 `experiment/nonlinear_ablation.py`、无数据失败先行点专用的三个矩计算/共同补偿函数及对应测试；详细文件和函数SHA见validation.json的report_cleanup_20261023。当前任务SVD/参数置换、防御一、原版AGA和主表入口保留，核心算法AST未变。历史数值试验须按原协议从该快照取回，不将现文件路径冒充历史版本。清理后17项核心回归及同源论文表格回归通过。唯一权重、原始分数和失败证据未删。最新书面报告在 `output/pdf/teacher_report_20261023/`，不再保留旧演示稿的重复产物。


Oct24磁盘清理：41.29GB同大小候选的SHA扫描没有找到新的完整重复文件，未声称释放重复模型。清理已结束的joint_response训练中间缓存；attention_cascade校准文件仅移除可重建的中间激活，全部参数、SVD、索引及原始视觉输出逐张量保留在retained_records.pt。旧缓存SHA与重建来源仍记录，历史完整缓存核验需按原源码重建。原模型、预测分数和失败日志未删除。四份ProDistill模型转存同一3090系统盘，原路径改符号链接，字节/SHA保持；见cleanup_20261023.json的resumed_cleanup_20261024。


Oct24完成的两份只读二阶计算源码仅保留在历史提交62f2a9f（公开对应9dfbe3789a2ae8169957d8d4470e2d9b032ee400）及3090原outputs中，从活动目录移除；原字节/SHA已固定。无新增活动算法入口或优化器。最近源码映射见publication.json的oct24_reconnection_history。

Oct26改回3090计算。清理15份已结束文件中的Attention键值、隐藏激活、Jacobian等中间缓存，其余参数、分数、特征、索引及协议存入同目录的`*.retained.pt`并核对保持相同；旧完整缓存删除，不能把精简文件当作原SHA的文件，历史重放需按归档源码重建所列缓存。另将2份字节相同的文件改为硬链接，保留路径和SHA。两盘可用空间合计实际增加4,713,619,456字节；原模型和唯一发布权重未删除。逐项路径、删除字段及前后SHA见[清理记录](../results/engineering_refactor/cleanup_20261026.json)。4090上的Oct25/26原始结果尚未迁回，不能称已备份或完成其待核验项目。


Oct24继续清理：已完成且使用真实训练图的逐层数值构造从活动入口移除，两个函数及原配置保留在Git `7a59e9dd6101eeed9158f7fa8a27dd0570ff1cbf` 和 `outputs/2026-10-24_research/before_cleanup_source.tar.gz`。历史数值结果及论文表格不改；防御一、通道置换、原版AGA和统一评价继续使用当前入口。当前配置的源码SHA随本次入口变化更新，历史协议不回写。

磁盘清理记录见 `results/engineering_refactor/cleanup_20261023.json` 的 `additional_cleanup_20261024`。旧诊断的 `.pt.gz` 保留原文件全部字节，已校验解压SHA。需要历史重放时，根据清理记录中的两个绝对路径，用 `gzip -dc <archive_path> > <original_path>` 恢复尚不存在的原 `.pt`，核对原SHA后再使用冻结源码；无需在当前工程增加加载分支。query_direction校准缓存只保留原始视觉输出、索引和基，其余中间激活按冻结source.zip重建。已关闭探索不能通过清理改为成功。


2026-10-25：已结束的临时实验脚本在核对精确归档后从当前目录移除；另移除10份未被运行代码和论文表引用的一次性调试记录。噪声验证脚本的有效构造已接入共享核心，删除的是重复入口。具体路径、源码行数、SHA和Git快照见results/engineering_refactor/validation.json的oct25_core_cleanup；唯一模型、原始分数和失败记录保留。核心方法的新实验只保留当前必要源码，不增加框架。

防御一当前只用权重构造，不再接收图片索引、不调用数据加载器、不保存图片校准缓存。`_impl/defenses.py`从213行缩为117行，原有图片幅度校准及捕获流程删除；`experiment.construct --method bias`直接输出发布参数。旧数据辅助算法可从4c15495复现。其函数体旧哈希属于历史定义，新实现改用功能测试，并逐张量确认与已完成评价的纯权重噪声点相同；17项回归及行为测试通过。当前主配置指纹已更新，历史冻结协议不回写。
