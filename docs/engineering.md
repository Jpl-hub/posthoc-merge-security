# 工程整理与历史复现

此次整理基于本项目现有实验，不引入 IQA 项目的内容。公共算法放 `_impl/`，实验顺序与命令行放 `experiment/`，科学选择放 `configs/`；不再为每个探索点复制一套加载、融合和核验脚本。

## 保留什么

- 当前防御一、防御二、原版 AGA、单列公开匹配、完整 PaRaMS、TA/TIES/DARE 核和八专家顺序保留在活动代码中。
- 原始模型、唯一冻结发布、原始分数、数据索引、失败日志不删除或换名；重数据留 3090。
- 现有 `results/`、`notes/` 和历史汇报文件保留。重构不修改历史准确率、阈值、源码 SHA 或失败状态。
- NUFILT、ProDistill、蒸馏、混合精度清理以及已关闭的补偿探索不另造新实现。它们的已验证结果继续保留，原工程与协议使用历史版本复现。

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

任务方向方法的详细执行版本在 `results/task_directed_pairing/frozen_source_history.json`；更早补充实验以各自 `protocol.json`／`execution.json`／独立核验文件为准。历史记录中的 `experiments/` 路径属于这些固定版本，不伪造为当前 `_impl/` 路径。

## 当前命令与历史命令

| 原职责 | 当前入口 |
|---|---|
| 任务 SVD、任务方向／固定距离构造 | `experiment.construct`、`_impl.subspace` |
| 最终 LN 核空间偏置 | `experiment.construct --method bias`、`_impl.defenses` |
| 冻结发布的独立任务检查 | `experiment.native` |
| 两专家 TA／原版 AGA 作用范围评价 | `experiment.main_table` |
| 八专家固定顺序及三个数据集评价 | `experiment.eight_task` |
| 原版 AGA / 已知家族公开匹配 | `experiment.recover --method aga / patch` |
| 当前统一论文表 | `experiment.report` |
| 已结束探索及其他历史补充表 | 固定历史提交与原协议 |

`experiment.native` 的原始分数独立重计与主实验的原批上下文重放是两个检查，不能混称。主实验 `--verify` 保留128张原批上下文和原5e-4误差上限。单独功能检查继续记录原R0.5和改判，不改写历史失败。

## 维护约定

1. 只有一个共享模型加载器和一套主评价流程。新方法优先增加纯参数函数，不复制入口或引入注册框架。
2. 构造不能读取伙伴／测试来选参；发布先固定哈希，再检查功能、融合和恢复。专家不重新训练。
3. 配置写出科学选择与权限；程序拒绝覆盖已有输出，外部作者代码和检查点须核 SHA。新源码指纹属于重构后的新协议，不替换旧协议指纹。
4. 读出表格只写指定输出目录，不改原始数据或来源记录。论文语言与指标按已读论文表达，不另造名称。
5. 删除结束的一次性入口前记录 Git 版本和哈希；唯一证据至少留一份。缓存仅在确认可再生或重复后清理。

本次检查结果及确切删减数量在 `results/engineering_refactor/validation.json`。连接恢复后已在3090通过18项测试、Ruff检查和真实模型核验：三份发布逐张量相同，原版AGA恢复权重全部158张量相同，8个原128张批上下文的分数最大误差0。仅新增1024张工程核验前向及一次原版AGA，不称完整科研实验重跑；详见GPU_validation.json。已有科研实验结果仍来自此前冻结代码。
