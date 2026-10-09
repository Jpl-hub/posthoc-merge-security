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
            "我把最后四层 V 输出压到 rank 16，删去每个 head 四分之三的维度。独立 MNIST 只损失 0.08 个百分点，但直接融合已经有效，公开 pretrained 补全后达到 99.37% 和 92.60%。数学上的秩损失不能直接推成分类或安全下界，这条简单压缩路线已停止扫描。"
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
        speech.append(
            f"再把{len(layers)}个完整 FFN 改写为二次映射，删除原 W1、W2。32 张合成图像闭式拟合，无梯度或优化器，但存在新参数拟合。独立损失 {loss:.2f} 个百分点；用另外 32 张图像重构原架构后，融合仍为 {acc('reconstructed_protected_merge', 'MNIST'):.2f}% 和 {acc('reconstructed_protected_merge', 'SVHN'):.2f}%。隐藏原权重没有阻止攻击者得到另一种可融合表示。"
        )
    if detail:
        direction = root / "candidate_direction.md"
        prefix = direction.read_text(encoding="utf-8").split("## 实测")[0]
        ending = "\n\n## 本轮结论与下一项可证伪问题\n\n本轮没有得到新防御。秩压缩、二次映射及 ridge 都有前作，不包装为创新；保留原始计数、校准来源、失败配置与攻击对照。四层覆盖是单层负结果后的范围消融，固定同一 rank/degree/ridge，没有在 test 上扫参，不能称全部独立预先计划。\n\n下一项优先回答：固定公开校准与攻击预算内，是否存在满足独立效用预算、却无法被共同表示和 anchor-centered 功能拟合恢复的任务特异信息编辑？先建立任务增量保留与删除信息的可识别性条件，再做小模型解析反例和 CLIP 单候选检验。若只能以明显降低独立效用为代价，应停止该路线。\n\n本轮只有一个保护任务、一个模型族和一个种子；没有完成张量分解、端到端蒸馏、其他融合方法或样本数扫描。不把绕过本原型当作对 MergeBarrier 原方法的攻击结论，也不将有限查新称作新颖性证明。\n"
        direction.write_text(
            prefix + "## 实测\n\n" + "\n\n".join(detail) + ending, encoding="utf-8"
        )
    return "\n\n".join(sections), "\n\n".join(speech)
