"""Tables for the completed rank and FFN probes, derived from raw counts."""


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(map(str, r)) + " |" for r in rows]
    )


def generate(root, rank, polynomial_runs):
    sections, detail, speech = [], [], []
    if rank:

        def acc(preset, group, dataset):
            return 100 * next(
                r["top1"]
                for r in rank
                if r["preset"] == preset and r["group"] == group and r["dataset"] == dataset
            )

        presets = list(dict.fromkeys(r["preset"] for r in rank))
        summary = table(
            [
                "最后四层 V 编辑",
                "独立 MNIST (%)",
                "损失 (pp)",
                "直接融合 MNIST / SVHN (%)",
                "pretrained 补全融合 MNIST / SVHN (%)",
            ],
            [
                [
                    p,
                    f"{acc(p, 'standalone', 'MNIST'):.2f}",
                    f"{acc('unprotected', 'standalone', 'MNIST') - acc(p, 'standalone', 'MNIST'):.2f}",
                    " / ".join(f"{acc(p, 'direct', d):.2f}" for d in ["MNIST", "SVHN"]),
                    " / ".join(f"{acc(p, 'completion', d):.2f}" for d in ["MNIST", "SVHN"]),
                ]
                for p in presets
            ],
        )
        conclusion = "四个预设配置的独立效用损失均在 0.5 pp 预算内，但直接融合已经有效。rank 16 删去 75% 的单 head 输出维度，独立损失 0.08 pp；公开 pretrained 补全后融合恢复 99.37% / 92.60%。局部秩下界没有转化为融合保护，停止继续扫描该配置族。不能仅凭此实验断言是哪一层保留了任务信息。"
        sections.append(
            "### 不可逆秩编辑\n\n"
            + summary
            + "\n\n"
            + conclusion
            + "\n\n![秩扰动筛选](figures/rank_screening.png)\n\n图：完整测试、单种子；准确率相对各自未保护对照的差值，单位 pp。补全无需数据或秘密投影。"
        )
        detail.append(
            summary
            + "\n\n"
            + table(
                ["V 编辑", "融合 / 攻击", "MNIST (%)", "SVHN (%)"],
                [
                    [p, g, *[f"{acc(p, g, d):.4f}" for d in ["MNIST", "SVHN"]]]
                    for p in presets
                    for g in ["direct", "AGA", "paired", "completion"]
                ],
            )
            + "\n\n"
            + conclusion
        )
        detail.append(
            "standalone 仅以 MNIST 作为保护效用；原 MNIST 模型在 SVHN 的迁移结果保存在 JSON，未当作 SVHN 专家的独立效用。activation 两项使用 32 张无标签合成图像；weight-rank 与随机置零不使用校准。全部配置评估两个完整测试集，系数 0.5。"
        )
        speech.append(
            "第一项候选是不可逆秩收缩：用 32 张合成笔画图像的激活确定投影，把最后四层 V 输出压到 rank 16，删去每个 head 四分之三的维度。独立 MNIST 从 99.69% 降到 99.61%，只损失 0.08 个百分点，但直接融合已经有效。攻击者从发布参数估计保留空间，用公开预训练模型补全其余方向，不需要数据或秘密投影，融合达到 99.37% 和 92.60%。"
        )
    labels = [
        ("unprotected_standalone", "原始 MNIST 专家独立"),
        ("protected_standalone", "重表示 MNIST 专家独立"),
        ("converted_other_standalone", "转换后的 SVHN 专家独立"),
        ("reconstructed_protected_standalone", "保护模型重构后独立"),
        ("reconstructed_control_standalone", "未保护模型重构后独立"),
        ("original_direct_merge", "原始未保护融合"),
        ("shared_format_merge", "共同格式直接融合"),
        ("shared_format_AGA_merge", "共同格式兼容层 AGA 后融合"),
        ("reconstructed_protected_merge", "保护模型原架构重构后融合"),
        ("reconstructed_control_merge", "未保护模型同攻击后融合"),
    ]
    for name, rows, meta in polynomial_runs:

        def acc(group, dataset):
            return 100 * next(
                r["top1"] for r in rows if r["group"] == group and r["dataset"] == dataset
            )

        layers = meta["aga_mlp_skipped_layers"]
        loss = acc("unprotected_standalone", "MNIST") - acc("protected_standalone", "MNIST")
        heading = f"### 完整 FFN 二次重表示：{len(layers)} 层\n\n"
        values = [[label, *[f"{acc(g, d):.4f}" for d in ["MNIST", "SVHN"]]] for g, label in labels]
        summary = table(
            ["设置", "MNIST (%)", "SVHN (%)"],
            [
                v
                for v, (g, _) in zip(values, labels, strict=True)
                if g
                in {
                    "protected_standalone",
                    "original_direct_merge",
                    "shared_format_merge",
                    "shared_format_AGA_merge",
                    "reconstructed_protected_merge",
                    "reconstructed_control_merge",
                }
            ],
        )
        conclusion = f"独立 MNIST 损失 {loss:.2f} pp，{'满足' if loss <= meta['protocol']['utility_budget_pp'] else '超出'}预设 0.5 pp 预算。每层原 FFN 的 4,722,432 个参数被 430,848 个系数及公开基替代，不保留该块 W1/W2。需要 32 张合成校准图像作闭式拟合，没有额外微调专家。攻击者用另一 seed 的 32 张合成图像、公开 pretrained 和发布模型拟合回原架构，融合为 {acc('reconstructed_protected_merge', 'MNIST'):.2f}% / {acc('reconstructed_protected_merge', 'SVHN'):.2f}%。"
        conclusion += (
            "共同格式融合也接近未保护融合，本次未形成保护。"
            if loss <= meta["protocol"]["utility_budget_pp"]
            else "该配置已经损伤独立任务，融合下降无法作为保护证据。"
        )
        scope = f"AGA 明确跳过 {len(layers)} 个替换 FFN；共同格式攻击使用公开 defender 基与校准协议，原架构攻击使用独立图像。独立行的非所属任务数值表示迁移性能。"
        if loss > meta["protocol"]["utility_budget_pp"]:
            scope += " 本配置因独立效用超预算淘汰；直接融合下降不能计作保护成功，不继续扩大覆盖或扫参数。"
        sections.append(
            heading
            + summary
            + "\n\n"
            + conclusion
            + "\n\n"
            + scope
            + f"\n\n![FFN 重表示及功能重构](figures/{name}.png)\n\n图：完整测试、单种子；系数 0.5，纵轴从零开始，无统计误差条。"
        )
        detail.append(
            heading
            + table(["设置", "MNIST (%)", "SVHN (%)"], values)
            + "\n\n"
            + conclusion
            + "\n\n"
            + scope
        )
        detail.append(
            f"整体运行 {meta['total_seconds']:.2f} s，包含模型加载、拟合与全部评价，不是单独攻击时间。发布块每层含 buffers 总计 1,833,344 bytes；参数规模不代表部署速度。"
        )
        if meta.get("reconstruction_seconds"):
            detail.append(
                f"保护模型重构耗时 {meta['reconstruction_seconds']['protected_reconstruction']:.3f} s，包括激活采集、闭式求解与模型拷贝；三模型转换共 {meta['conversion_seconds']:.3f} s。单次记录，非严格速度比较。"
            )
        if len(layers) == 1:
            speech.append(
                f"第二项候选是完整 FFN 重表示：将最后一个 FFN 改写为二次映射，删除该块原 W1、W2。用 32 张无标签合成笔画图像闭式拟合，没有梯度或优化器，但存在新参数拟合，不能称无需数据。独立 MNIST 保持 {acc('protected_standalone', 'MNIST'):.2f}%；共同格式融合仍然有效。攻击者用另一批 32 张合成图像拟合回原架构后，融合为 {acc('reconstructed_protected_merge', 'MNIST'):.2f}% 和 {acc('reconstructed_protected_merge', 'SVHN'):.2f}%。"
            )
        else:
            speech.append(
                f"在单层失败后，我做了固定四层的覆盖消融，保持同一 rank、阶数和正则。独立 MNIST 降到 {acc('protected_standalone', 'MNIST'):.2f}%，损失 {loss:.2f} 个百分点，远超预算；功能重构后融合又达到 {acc('reconstructed_protected_merge', 'MNIST'):.2f}% 和 {acc('reconstructed_protected_merge', 'SVHN'):.2f}%。这项配置因独立效用不达标而淘汰。"
            )
    if detail:
        direction = root / "candidate_direction.md"
        prefix = direction.read_text(encoding="utf-8").split("## 实测")[0]
        ending = "\n\n## 本轮结论与下一项可证伪问题\n\n本轮没有得到新防御。秩压缩、二次映射及 ridge 都有前作，不包装为创新；保留原始计数、校准来源、失败配置与攻击对照。四层覆盖是单层负结果后的范围消融，固定同一 rank/degree/ridge，没有在 test 上扫参，不能称全部独立预先计划。\n\n下一项优先回答：固定公开校准与攻击预算内，是否存在满足独立效用预算、却无法被共同表示和 anchor-centered 功能拟合恢复的任务特异信息编辑？先建立任务增量保留与删除信息的可识别性条件，再做小模型解析反例和 CLIP 单候选检验。若只能以明显降低独立效用为代价，应停止该路线。\n\n本轮只有一个保护任务、一个模型族和一个种子；没有完成张量分解、端到端蒸馏、其他融合方法或样本数扫描。不把绕过本原型当作对 MergeBarrier 原方法的攻击结论，也不将有限查新称作新颖性证明。\n"
        direction.write_text(
            prefix + "## 实测\n\n" + "\n\n".join(detail) + ending, encoding="utf-8"
        )
    return "\n\n".join(sections), "\n\n".join(speech)


def meeting_report(real, rank, polynomial_runs, *, real_count, mechanism_count):
    """Meeting narrative; full controls remain in the experiment records."""
    datasets = ["MNIST", "SVHN"]

    def accuracy(rows, group, dataset="MNIST", **filters):
        hits = [
            r
            for r in rows
            if r["group"] == group
            and r["dataset"] == dataset
            and all(r.get(k) == v for k, v in filters.items())
        ]
        assert len(hits) == 1, (group, dataset, filters)
        return 100 * hits[0]["top1"]

    def pair(rows, group, **filters):
        return " / ".join(f"{accuracy(rows, group, d, **filters):.2f}" for d in datasets)

    baseline = accuracy(real, "A_unprotected")
    results = [
        [
            "未保护对照",
            f"{baseline:.2f}",
            "0.00",
            pair(real, "D_unprotected_merge", coefficient=0.5),
            pair(real, "D_unprotected_merge", coefficient=0.5),
            "参照",
        ]
    ]
    if rank:
        utility = accuracy(rank, "standalone", preset="activation_rank16")
        results.append(
            [
                "V 秩收缩：rank 16",
                f"{utility:.2f}",
                f"{baseline - utility:.2f}",
                pair(rank, "direct", preset="activation_rank16"),
                pair(rank, "completion", preset="activation_rank16"),
                "可融合",
            ]
        )
    for _, rows, meta in polynomial_runs:
        blocks = len(meta["aga_mlp_skipped_layers"])
        utility = accuracy(rows, "protected_standalone")
        loss = baseline - utility
        verdict = "可被重构" if loss <= meta["protocol"]["utility_budget_pp"] else "效用超预算"
        results.append(
            [
                f"完整 FFN 重表示：{blocks} 层",
                f"{utility:.2f}",
                f"{loss:.2f}",
                pair(rows, "shared_format_merge"),
                pair(rows, "reconstructed_protected_merge"),
                verdict,
            ]
        )
    result_table = table(
        [
            "设置",
            "独立 MNIST (%)",
            "损失 (pp)",
            "直接融合 M / S (%)",
            "自适应恢复融合 M / S (%)",
            "判断",
        ],
        results,
    )
    return rf"""# 发布前免微调的模型融合保护

2026-10-09 组会

**本轮结论：已跑通真实评估，检验了不可逆秩编辑和完整 FFN 重表示；两个候选均未形成有效防御。攻击者能够获得另一种可融合表示，无需恢复原始权重。**

## 1. 研究目标

对已经微调好的专家，在发布前施加后处理保护，保持原有任务能力，同时抵抗未经授权的模型融合和自适应恢复。原专家不再微调；允许明确记录的公开无标签校准与极小性能损失。

攻击者可以读取发布模型、使用公开 pretrained 模型及自己的其他专家。评价关注恢复后的融合效用，参数距离或格式变化只用于解释机制。

## 2. 文献带来的判断

| 工作 | 与本课题有关的结论 | 对研究设计的影响 |
|---|---|---|
| PaRaMS、MergeLock | 置换、缩放及可逆混合已有前作 | 扩大变换幅度不能直接当作创新 |
| AGA、配对对齐、PLeaS | 参数对齐与功能拟合都是攻击路径 | 同时检查参数恢复和功能恢复 |
| MergeBarrier、TaylorMLP | FFN 多项式表示已有前作；相关表示已有提取攻击 | 改结构后仍需检查共享格式融合与功能重构 |
| TSV-C、SVD-LLM | 低秩压缩已有数学与实验基础，压缩甚至可能促进融合 | 不可逆信息损失不等于融合保护 |

MergeGuard 完整方法包含额外微调，不直接满足本课题限制；其免训练第二阶段也已有消融。上述审读用于排除重复思路，尚不足以证明新颖性。[原文位置与适用条件](literature_triage.md)

## 3. 本轮具体检验了什么

**候选 A：删去 Attention 的部分输出子空间。** 对最后四层各 head 的 V 仿射算子作谱投影，发布 $\tilde B=P_rB$。它降低算子秩，超出可逆变换；局部近似误差有谱截断依据，但不保证整体任务能力或安全。

攻击者从发布算子估计保留空间，用公开 pretrained 补全其余方向。该攻击不需要校准数据或秘密投影。

**候选 B：用完整 FFN 的二次映射替代原参数。** 用 32 张程序生成的无标签笔画图像，将完整 FFN 闭式拟合为低维二次函数，发布汇总系数和公开基，移除替换块的原 W1/W2。该过程没有梯度或优化器，但有新参数拟合，不能称 data-free。

攻击者可以把自己的模型转换到同一格式融合，也可以用另一批 32 张合成图像，固定 pretrained 的第一层、闭式拟合第二层，重构回原架构。数学依据和边界见 [候选论证](candidate_direction.md)。压缩、二次近似和最小二乘本身均不作为创新。

## 4. 真实实验结果

CLIP ViT-B/32；保护 MNIST 专家，与原始 SVHN 专家融合。完整测试集分别为 10,000 和 26,032 张图像。下表融合系数固定为 0.5，独立 MNIST 损失预算为 0.5 pp；一个种子，用于初筛。

{result_table}

M / S 分别表示融合模型在 MNIST / SVHN 上的准确率。FFN 的“直接融合”使用共同表示；结构不兼容不计作安全。自适应攻击分别为 pretrained 子空间补全和原架构闭式重构。未保护行的自适应列重复基线，仅作参照。

原 AGA、兼容层 AGA、配对对齐及同攻击未保护对照均已记录。AGA 在新格式下明确跳过替换的 FFN，不能当作完整覆盖新架构的攻击。所有对照和原始计数见 [实验详细记录](experiment_details.md)、[候选完整结果](candidate_direction.md) 与 [机器结果表](results.csv)。

![单层 FFN 重表示与恢复](figures/polynomial_probe.png)

图：单层改写保持独立准确率，但共同格式融合及原架构重构后融合都接近原始基线。完整测试、单种子，不画统计误差条。

## 5. 如何解释负结果

**秩收缩没有阻止融合。** rank 16 删去每个 head 四分之三的输出维度，独立性能仅损失 0.08 pp，但直接融合仍然有效，pretrained 补全又恢复到基线附近。局部线性代数下界不能直接转化为分类或安全下界。

**删除原参数没有阻止功能恢复。** 单层重表示保持独立准确率，却可被另一批合成图像的功能拟合绕过。四层覆盖后独立性能损失 10.91 pp，已经超预算；直接融合下降不能当作保护成功。它说明当前近似在扩大覆盖时损伤效用，尚未确定误差累积与校准分布各自的影响。

此前的基础 PaRaMS、正交保护、LayerNorm 空方向和低活跃扰动也未挡住相应恢复攻击。它们用于建立对照，详细过程留在实验记录中。上述结果仅针对本次配置，不证明所有后处理防御不可能，也不是对 MergeBarrier 原方法的攻击结论。

## 6. 下一步与复现状态

1. 停止继续扫描当前秩投影和二次表示配置族。下一项先分析：哪些任务特异信息可以在小效用损失下删除，同时难以被公共 anchor 或功能拟合补回；有可检验构造后再追加实验。
2. 固定攻击信息、校准样本和计算预算；候选先过独立效用门槛，再检查预算内攻击取得的融合效用。通过初筛后才扩多种子、任务及融合方法。
3. 八任务复现并行补资源。全部分类 head 已取得，DTD 第一划分及 GTSRB 完整测试入口已验证；六个官方专家 checkpoint 仍受下载配额限制。公开镜像逐参数不一致，已拒绝替代，当前没有八任务准确率。[资源状态](eight_task/status.md)

本轮保留 {real_count} 条真实完整测试评估与 {mechanism_count} 条机制验证记录；这些是配置数，不是独立重复数。重型资产及完整日志在服务器，仓库保存代码、轻量结果和报告。
"""
