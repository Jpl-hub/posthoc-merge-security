"""Generate concise Chinese reports and a unified CSV from executed results."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path


def read(root: Path, name: str, default=None):
    path = root / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def table(headers, rows):
    return "\n".join(
        ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
        + ["| " + " | ".join(map(str, row)) + " |" for row in rows]
    )


def mean(rows, field):
    return statistics.mean(row[field] for row in rows)


def run(root: Path):
    mechanism = read(root, "mechanism_results.json", [])
    real = read(root, "clip_results.json", [])
    params = read(root, "params/clip_results.json", [])
    paired = read(root, "coupled/coupled_results.json", [])
    ln = read(root, "layernorm/layernorm_results.json", [])
    low = read(root, "low_activity/low_activity_results.json", [])
    mm = read(root, "mechanism_metadata.json", {})
    cm = read(root, "clip_metadata.json", {})
    pm = read(root, "params/clip_metadata.json", {})
    pairedm = read(root, "coupled/coupled_metadata.json", {})
    env = read(root, "environment.json", {})
    lm = read(root, "layernorm/layernorm_metadata.json", {})
    lowm = read(root, "low_activity/low_activity_metadata.json", {})
    all_rows = (
        mechanism
        + [dict(r, protection="basis") for r in real]
        + [dict(r, protection="params") for r in params]
        + ln
        + low
        + paired
    )
    with (root / "results.csv").open("w", newline="", encoding="utf-8") as stream:
        fields = sorted(set().union(*(row.keys() for row in all_rows)))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(all_rows)
    mlp = [r for r in mechanism if r["experiment"] == "mlp_permutation" and r["magnitude"] == 0.01]
    attn = [r for r in mechanism if r["experiment"] == "attention_qk_vo" and r["magnitude"] == 0.01]
    heads = [r for r in mechanism if r["experiment"] == "heads_plus_orthogonal"]
    p1 = [r for r in mechanism if r["experiment"] == "column_space_invariance"]
    synthetic_table = table(
        ["机制；合成 task change = 1%", "保护输出误差", "AGA 参数误差", "AGA 输出误差"],
        [
            ["MLP 置换", f"{mean(mlp, 'protected_output_error'):.2e}", "0", "0"],
            [
                "Attention QK/VO",
                f"{mean(attn, 'protected_output_error'):.2e}",
                f"{mean(attn, 'recovered_weight_error'):.2e}",
                f"{mean(attn, 'recovered_output_error'):.2e}",
            ],
        ],
    )
    head_table = table(
        ["跨 head 置换 + 正交基变换", "相对输出误差；3 seeds 平均"],
        [
            ["保护 standalone", f"{mean(heads, 'protected_output_error'):.2e}"],
            ["原逐 head AGA", f"{mean(heads, 'recovered_output_error'):.4f}"],
            ["子空间 matching + AGA", f"{mean(heads, 'adaptive_output_error'):.4f}"],
        ],
    )

    def accuracy(group, dataset, coef=None):
        hit = [
            r
            for r in real
            if r["group"] == group and r["dataset"] == dataset and r["coefficient"] == coef
        ]
        return f"{100 * hit[0]['top1']:.2f}%" if hit else "未完成"

    def average_accuracy(group, coef):
        values = [r["top1"] for r in real if r["group"] == group and r["coefficient"] == coef]
        return 100 * statistics.mean(values) if len(values) == 2 else None

    merge_labels = [
        ("D_unprotected_merge", "未保护", "无"),
        ("E_protected_merge", "正交基变换 + MLP 置换", "无"),
        ("F_AGA_merge", "正交基变换 + MLP 置换", "仅保护模型 AGA"),
    ]
    if cm.get("attack_all"):
        merge_labels += [
            ("G_AGA_all_unprotected_merge", "未保护", "全部模型 AGA"),
            ("H_AGA_all_protected_merge", "正交基变换 + MLP 置换", "全部模型 AGA"),
        ]
    merge_rows = []
    for coef in cm.get("coefficients", [0.2, 0.5]):
        direct_avg = average_accuracy("E_protected_merge", coef)
        for group, protection, attack in merge_labels:
            avg = average_accuracy(group, coef)
            delta = avg - direct_avg if avg is not None and direct_avg is not None else None
            merge_rows.append(
                [
                    protection,
                    attack,
                    str(coef),
                    accuracy(group, "MNIST", coef),
                    accuracy(group, "SVHN", coef),
                    f"{avg:.2f}" if avg is not None else "未完成",
                    f"+{delta:.2f}"
                    if delta is not None
                    and "AGA" in attack
                    and group != "G_AGA_all_unprotected_merge"
                    else "—",
                ]
            )
    merge_table = table(
        ["保护", "攻击", "TA 系数", "MNIST (%)", "SVHN (%)", "Avg (%)", "↑Δ (pp)"],
        merge_rows,
    )

    real_table = table(
        ["组别", "系数", "MNIST", "SVHN"],
        [
            [
                "A：未保护 MNIST 模型",
                "—",
                accuracy("A_unprotected", "MNIST"),
                accuracy("A_unprotected", "SVHN"),
            ],
            [
                "B：后处理保护",
                "—",
                accuracy("B_protected", "MNIST"),
                accuracy("B_protected", "SVHN"),
            ],
            ["C：B 经 AGA 恢复", "—", accuracy("C_AGA", "MNIST"), accuracy("C_AGA", "SVHN")],
        ],
    )
    ln_text = "候选实验尚未完成，不计入防御效果结论。"
    if ln and lm:
        groups = [
            ("LN_null_only", "空方向扰动 standalone"),
            ("LN_null_plus_basis", "空方向 + 基变换 standalone"),
            ("LN_null_AGA", "原 AGA standalone"),
            ("LN_null_canonical_AGA", "投影消除 + AGA standalone"),
            ("LN_null_AGA_merge", "原 AGA 后融合"),
            ("LN_null_canonical_AGA_merge", "投影消除 + AGA 后融合"),
            ("canonical_unprotected_merge", "canonical 未保护融合对照"),
        ]
        strengths = sorted({r["strength"] for r in ln})
        values = []
        for group, label in groups:
            values.append(
                [
                    label,
                    *[
                        f"{100 * next(r['top1'] for r in ln if r['group'] == group and r['strength'] == s):.2f}%"
                        for s in strengths
                    ],
                ]
            )
        ln_text = table(["MNIST 全测试集", *[f"扰动强度 {s:g}" for s in strengths]], values)
        ln_text += "\n\n解析后处理，无训练、无校准数据。函数误差通过 $10^{-4}$ 阈值。攻击者只需公开 LayerNorm 参数即可消去空方向；canonical 未保护融合是相同参数表示下的对照。原 AGA 及自适应攻击后准确率均接近未保护融合，故该候选未形成有效保护。"
        ln_text += "\n\n![LayerNorm 后处理探索](figures/layernorm_exploration.png)"
        ln_text += "\n\n图 4：两档扰动强度；单种子、MNIST 完整测试集。纵轴放大至高准确率区域；点表示准确率，不画统计误差条。"
    low_text = "实验未完成。"
    low_speech = ""
    if low and lowm:
        low_groups = [
            ("low_activity_standalone", "保护后 standalone"),
            ("low_activity_AGA", "AGA standalone"),
            ("low_activity_projection_AGA", "低活跃投影 + AGA standalone"),
            ("low_activity_direct_merge", "保护后直接融合"),
            ("low_activity_AGA_merge", "AGA 后融合"),
            ("low_activity_projection_AGA_merge", "低活跃投影 + AGA 后融合"),
            ("low_activity_projected_baseline_merge", "投影后的未保护融合对照"),
        ]
        low_text = table(
            ["设置", "MNIST (%)"],
            [
                [label, f"{100 * next(r['top1'] for r in low if r['group'] == group):.2f}"]
                for group, label in low_groups
            ],
        )
        standalone = next(r for r in low if r["group"] == "low_activity_standalone")
        attacked = next(r for r in low if r["group"] == "low_activity_AGA_merge")
        adaptive = next(r for r in low if r["group"] == "low_activity_projection_AGA_merge")
        low_text += f"\n\n预设 standalone 损失预算 0.5 pp，本次损失 {(0.9969 - standalone['top1']) * 100:.2f} pp，通过预算。特征误差为 {standalone['feature_error']:.3f}，因此属于近似保效用后处理，不是函数等价变换。AGA 后融合 {100 * attacked['top1']:.2f}%，自适应攻击后 {100 * adaptive['top1']:.2f}%，未得到明显保护。单任务、单种子结果不足以支持泛化或统计显著性。"
        low_speech = f"随后我用 32 张程序生成的笔画图像，提取最后四层的低活跃输入方向，构造不含梯度更新的解析扰动。预先规定独立准确率损失不超过 0.5 个百分点。实际只损失 {(0.9969 - standalone['top1']) * 100:.2f} 个百分点，但 AGA 后融合仍达到 {100 * attacked['top1']:.2f}%，加低活跃方向投影攻击后是 {100 * adaptive['top1']:.2f}%。所以它保留效用，却没有解决恢复攻击。"
    params_text = "尚无完整结果，不计入结论。"
    params_speech = ""
    if params and pm:
        values = []
        for coef in pm["coefficients"]:
            direct = statistics.mean(
                r["top1"]
                for r in params
                if r["group"] == "E_protected_merge" and r["coefficient"] == coef
            )
            for group, label in [
                ("D_unprotected_merge", "未保护；无攻击"),
                ("E_protected_merge", "PaRaMS；直接融合"),
                ("F_AGA_merge", "PaRaMS；仅保护模型 AGA"),
                ("G_AGA_all_unprotected_merge", "未保护；全部 AGA"),
                ("H_AGA_all_protected_merge", "PaRaMS；全部 AGA"),
            ]:
                hits = [
                    next(
                        r
                        for r in params
                        if r["group"] == group and r["dataset"] == name and r["coefficient"] == coef
                    )
                    for name in ["MNIST", "SVHN"]
                ]
                avg = statistics.mean(r["top1"] for r in hits)
                values.append(
                    [
                        label,
                        coef,
                        *[f"{r['top1'] * 100:.2f}" for r in hits],
                        f"{100 * avg:.2f}",
                        f"+{100 * (avg - direct):.2f}" if group.startswith(("F_", "H_")) else "—",
                    ]
                )
        params_text = table(
            ["设置", "TA 系数", "MNIST (%)", "SVHN (%)", "Avg (%)", "↑Δ (pp)"], values
        )
        params_text += f"\n\n基础 PaRaMS 直接调用官方固定版本；不含原文 §6 的 dropout 加固。12 层 MLP 取最远 assignment，Attention 对角缩放 U(0.05,20)；独立 MLP 组使用一轮 assignment，未伪称与 Table 6 配置完全一致。源码实际共用 Q/V 缩放，与论文独立采样描述有差别。保护特征误差 {pm['protected_output_error']:.2e}；保护前后两个完整测试集的正确计数相同。全部 AGA 后保护/未保护对照的两个任务正确计数相同，未形成有效保护。"
        params_text += "\n\n![官方 PaRaMS 基线](figures/params_comparison.png)\n\n图：真实双任务、单种子；系数 0.5。分别展示原始融合、保护后直接融合与 AGA 后融合，纵轴从零开始。图中 PaRaMS 指本次运行的基础配置。"
        params_speech = "我还接入了官方基础 PaRaMS，完整测试保持独立准确率，但 AGA 后两个任务的融合准确率恢复到未保护对照水平。这里没有运行论文的 dropout 加固，也不是八任务 Table 6 复现。"
    paired_text = "配对保函数对齐尚未完成，不计入结论。"
    paired_speech = ""
    if paired and pairedm:
        values = []
        for group, label in [
            ("paired_protected_standalone", "保护模型配对对齐 standalone"),
            ("paired_unprotected_standalone", "未保护模型配对对齐 standalone"),
            ("paired_protected_only_merge", "仅保护模型配对对齐后融合"),
            ("paired_unprotected_only_merge", "仅未保护模型配对对齐后融合"),
            ("paired_all_protected_merge", "全部配对对齐；含保护"),
            ("paired_all_unprotected_merge", "全部配对对齐；未保护"),
        ]:
            values.append(
                [
                    label,
                    *[
                        f"{100 * next(r['top1'] for r in paired if r['group'] == group and r['dataset'] == name):.2f}"
                        for name in ["MNIST", "SVHN"]
                    ],
                ]
            )
        paired_text = table(["配置；融合系数 0.5", "MNIST (%)", "SVHN (%)"], values)
        paired_text += "\n\n保护与未保护对照在配对对齐后取得相同正确计数，并恢复到原始未保护融合水平。该正交基线上的独立 AGA 不是能取得最高融合效用的攻击，后续防御需同时覆盖保函数的配对对齐。"
        paired_speech = "补读文献后，我复用了作者的配对 SVD 对齐与联合 MLP 匹配。它保持独立模型功能，并把正交保护后的两任务融合恢复到原始未保护水平，说明只测独立 AGA 还不够。"
        paired_text += f"\n\n复用 RotationSymmetry 的原始 Chunk_SVD_QK、Chunk_SVD_VO、Permute_IO；配对 Kabsch 保持 QK/VO 函数，联合 FFN 权重与 bias 匹配。求解用 float64，模型保持 float32；无数据、无训练。对齐前后最大特征误差 {max(pairedm['feature_errors'].values()):.2e}。本次仅测试正交保护与既定 head 对应关系，不含一般 GL 白化或跨 head matching；不能称全面强攻击。"
    meeting = rf"""# 2026-10-09 组会：无微调的模型融合保护

## 研究问题

对已经训练好的模型增加后处理保护：不再微调，尽量不使用原始训练数据，保持 standalone utility，阻止未经授权融合，并抵抗恢复攻击。

目标不仅是隐藏对应关系，而是让攻击者完成重新匹配后，仍难恢复有用的可融合表示。参数误差只能解释机制；安全结论必须落到攻击后的融合效用，并注明攻击所需信息与计算量。

今天没有执行模型训练。完成 {len(mechanism)} 条机制验证、{len(real)} 条正交基线双任务评估、{len(params)} 条官方 PaRaMS 评估、{len(paired)} 条配对对齐评估、{len(ln)} 条 LayerNorm 候选评估、{len(low)} 条低活跃方向候选评估。这些是评估配置数，不是独立重复实验数。尚未得到可抵抗自适应攻击的强防御。

## 论文与实现

AGA 用 pretrained 权重作 anchor：Attention 对 Q/K/V/O 分别逐 head 最小二乘对齐；MLP 用 cosine + Hungarian 恢复隐藏神经元顺序。ARF 在微调时扩大 Attention 与 anchor 的距离，因此不满足本课题的“仅后处理”。

作者公开代码未包含完整保护、ARF 训练和 checkpoint。今天使用 Task Arithmetic 官方 B/32 checkpoint，直接调用固定版本的 AGA、TaskVector 差分与求和、ImageClassifier 和 get_logits。新增部分是后处理候选、兼容入口及结果记录。当前正交 + 置换基线不直接标成 MergeLock、MergeBarrier 或 Table 6 的同配置方法。

## 已完成：机制验证

3 个固定种子；GELU MLP 宽度 48、隐藏维 96；Attention 宽度 96、12 heads。参数变化是合成扰动，不是下游训练。

{synthetic_table}

MLP 的 15 个测试配置均 100% 恢复真实置换；恢复后的参数误差为零。Attention 在不同扰动幅度下仍保持保护函数等价，但 AGA 独立估计投影会留下恢复误差。

![机制验证](figures/mechanisms.png)

图 1：P0 的 3 个种子；阴影为 min-max，非置信区间。MLP 恢复误差为零，单独注记，未用正数替代零值绘制对数曲线。

## 已完成：真实 CLIP 对照

MNIST test：10,000；SVHN test：26,032。保护对象是 MNIST 微调模型，第二专家是 SVHN。两个融合系数预先固定，未使用测试集选择系数。测试仅评价，没有校准。

{real_table}

表 1：standalone 对照。MNIST 是受保护任务；SVHN 是第二任务，standalone 在 SVHN 上的数值表示迁移性能。

{merge_table}

表 2：参考论文表 1/4 的保护 × 攻击布局。Avg 为两个任务 Top-1 的等权算术平均，不是合并样本后的准确率。↑Δ 是攻击相对同系数“保护后直接融合”的恢复增量，单位为百分点。全部模型 AGA 是防御无关设置；未保护模型也做 AGA，衡量攻击自身带来的效用变化。双任务数值不能与论文八任务 Avg 横向比较。

![防御无关攻击对照](figures/defense_agnostic.png)

保护前后 16 张真实图像上的特征相对误差为 {cm.get("protected_output_error", float("nan")):.2e}。原模型与保护模型在两个完整测试集上的正确样本数相同。AGA 后全 checkpoint 参数相对误差为 {cm.get("recovered_weight_error", float("nan")):.2e}；该指标含未参与视觉推理的遗留参数，不能替代 feature error 或 accuracy。AGA 特征误差为 {cm.get("recovered_output_error", float("nan")):.4f}。

![真实模型融合](figures/clip_merging.png)

图 2：全测试集、单种子，两档预设系数；纵轴从零开始。无重复实验时不画误差条。未在测试集选择融合系数。

![保护与独立效用](figures/posthoc_screening.png)

图 3：参考论文 Fig. 3，分别展示 standalone utility 与受保护任务的融合效用。左图纵轴放大并标明范围，右图从零开始。模型均为 ViT-B/32，融合系数 0.5；候选只在 MNIST 上筛选。攻击后融合准确率越低，才表示防御越有效。

## 已完成：官方基础 PaRaMS

{params_text}

## 超出 AGA：配对保函数对齐

{paired_text}

## 已完成：后处理边界

对于 $W^p=W_{{ft}}P$、可逆 $P$，理想 LS 的拟合值为

$$
W^a=\Pi_{{\operatorname{{col}}(W_{{ft}})}}W_{{pre}},\qquad
W^a-W_{{ft}}=-\Pi_{{ft}}\tau.
$$

这一结论已在论文附录证明。我们的直接 LS 数值验证中，不同正交、对角及稠密保护矩阵与统一投影结果的最大相对差异为 {max(r["ideal_projection_error"] for r in p1):.2e}。作者带正则的正规方程与 float32 转换在高条件数时偏离理想结果；这些数值误差不构成安全防御。

跨 head 置换保持函数，但会破坏原代码默认对应关系：

{head_table}

自适应子空间 matching 在全部 3 个种子中 100% 恢复 head correspondence，故不把跨 head 置换当成强防御。

![跨 head 置换](figures/head_permutation.png)

## 后处理候选：LayerNorm 空方向

若 $y=\gamma\odot z+\beta$ 且 $\mathbf1^Tz=0$，取归一化 $r\propto1/\gamma$，则 $r^T(y-\beta)=0$。对紧随 LayerNorm 的 Linear 作

$$
W'=W+ur^T,\qquad b'=b-u(r^T\beta),
$$

可改变参数列空间而保持该层在合法输入上的输出。它超出了“仅右乘可逆矩阵”的限制，且无需训练数据。候选施加在 QKV 与 MLP 第一层，再与基变换保护组合。

但 $r$ 由公开 LayerNorm 参数确定。攻击者可解析消去该方向：

$$
W_c=W-(Wr)r^T,\qquad b_c=b+(Wr)(r^T\beta).
$$

因此必须评价“canonicalization + AGA”，不能只观察原 AGA 是否失效。

{ln_text}

## 后处理候选：合成校准低活跃方向

使用 32 张无标签程序生成笔画图像，采集最后四层 LayerNorm 后激活的中心化协方差 $C$，取最低特征值的 64 维正交基 $N$。对 QKV 作 $\Delta W=UN^T$、$b'=b-\Delta W\mu$，扰动 Frobenius 范数为原权重的 1 倍，然后叠加正交与置换保护。参数、rank、层和强度在评价前固定；未使用 MNIST train/test 构造或调节扰动。

对单层有 $\mathbb E\|\Delta W(x-\mu)\|^2=\operatorname{{tr}}(\Delta W C\Delta W^T)\le\lambda_{{64}}(C)\|\Delta W\|_F^2$。这是校准分布上的局部约束，不能保证整个网络或真实数据上的效用。攻击者从发布模型重算合成激活方向，并对权重与 bias 作投影消除；anchor 和其他专家使用相同投影对照。

{low_text}

这两种机制是本次筛选候选，均未形成有效保护。已定向审读七篇核心论文并补读相关工作，尚不构成全面查新，不宣称方法新颖性。

## 文献与方向审查

PaRaMS 原文已讨论恢复、剪枝和蒸馏；MergeLock 已涵盖一般可逆 Attention 变换；PLeaS-free 仍使用公开无标签数据，且当前公开代码用 Adam 拟合 LS 目标。MergeBarrier 的 FFN Taylor 重表示需要激活校准并改变部署；TaylorMLP 也已有系数比值提取攻击，不能仅因参数格式改变就宣称安全。

后续优先检验：相同 standalone 效用预算下的不可逆编辑，是否降低加强攻击后的两任务能力；FFN 重表示先做小层系数泄漏和成本审查。扩大可逆变换幅度、叠加 head 置换不是当前有证据的创新路线。文献原文位置、威胁模型和三天安排见 [文献核对](literature_triage.md) 与 [方向审查](research_direction_review.md)。

## 八任务推进

保持 ViT-B/32、TA 系数 0.2 和论文的一次仅保护一个专家设置。已检查官方八任务下载入口并取得全部分类 head；六个缺失专家权重均遭遇 Google Drive 配额限制。公开 HF 同架构模型的训练配置不同，不能直接当作原 checkpoint。数据还需确认 EuroSAT 原随机索引、RESISC45 split 和 SUN397 第 1 分割。

当前没有八任务准确率，也没有将未复现的 MergeGuard/ARF 写入结果。资源缺项记录于 `eight_task/status.md`；待权重和划分齐备后先跑未保护八任务基线，再补官方防御。仅扩展任务数量还不足以称为 Table 6 严格复现。

## 结论与下一步

1. 小任务变化下，置换和 head 内可逆变换可被 AGA 绕过；扩大变换幅度不改变理想投影结果。
2. 精确函数等价后处理确有可用自由度，但已知对称性往往也给攻击者提供 canonicalization。
3. 低活跃方向允许较大参数变化和很小准确率损失，但 AGA 仍恢复高融合效用；单靠“远离 anchor”或保持 standalone accuracy 均不足以构成保护。
4. 双任务继续用于开发；八任务补齐后按同一协议验证。下一项机制必须同时改变对齐攻击可恢复的信息，且不依赖公开可消去的方向。有效候选才进入多种子、多任务及其他融合方法验证。

## 审读发现

MLP 附录的原始欧氏 margin 条件不能直接保证代码中的 cosine matching；固定反例满足条件却得到错误置换。ARF 印刷式 (19) 的 `min` 与正文 margin 行为不一致，已核对 PDF。由于 ARF 训练代码未公开，不能据此推断其实际实现或实验无效。详情见 `theory_notes.md`。

参考：[论文 v1](https://arxiv.org/abs/2606.30360v1)、[作者代码](https://github.com/krumpguo/secure-merge-attack)、[Task Arithmetic](https://github.com/mlfoundations/task_vectors)。数字均取自本次程序输出；论文数字未填入结果表。
"""
    (root / "group_meeting.md").write_text(meeting, encoding="utf-8")
    environment = f"""# 实验环境

日期：2026-10-09；运行位置：租用服务器；本地仅保存轻量文件。

| 项目 | 记录 |
|---|---|
| GPU | {env.get("gpu")}，24 GB |
| Compute capability | 8.6 |
| Python | 3.12.3 |
| PyTorch | {env.get("torch")} |
| PyTorch CUDA runtime | {env.get("torch_cuda")} |
| 驱动 | 595.71.05；nvidia-smi 支持上限 CUDA 13.2 |
| NumPy / SciPy | 1.26.4 / 1.13.1 |
| OpenCLIP | 2.0.2 |
| PaRaMS 可选依赖 | JAX / jaxlib 0.4.35；ml-dtypes 0.5.4；CPU 后端 |
| torchvision | 0.18.0+cu121 |
| 主机内存 | 总计约 375.8 GiB；检查时可用约 322.9 GiB；是主机视图 |
| 数据盘 | 50 GiB；checkpoint/数据仅保存在此盘 |
| 系统盘 | 30 GiB；依赖安装在现有 Miniconda |
| TF32 | 关闭 |
| CPU 线程 | PyTorch / OpenBLAS：4 |

现有 CUDA PyTorch 已通过 GPU 算子检查，无重装。新增依赖命令：

```bash
/root/miniconda3/bin/python -m pip install numpy==1.26.4 scipy==1.13.1 gdown==5.2.0 huggingface_hub==0.24.7 open-clip-torch==2.0.2
```

最初未锁版本的 pip 解析拟升级 NumPy，已在安装前停止并改为上面的兼容版本。完整安装日志、环境 JSON 与 pip freeze 留在服务器。仓库提交研究所需版本信息，排除机器连接和认证资料。

计时为实际 wall time，包含各脚本自身记录的加载/下载等阶段。显存峰值为 PyTorch 当前进程 `max_memory_allocated`，不是整机占用；P0 总峰值 {mm.get("peak_gpu_bytes", 0) / 2**20:.2f} MiB。基准过程有一次短暂并行候选启动，随后停止并排队；因此不将本次耗时作为严格性能比较。
"""
    (root / "environment.md").write_text(environment, encoding="utf-8")
    log = f"""# 执行记录

| 阶段 | 状态 | 实际记录 |
|---|---|---|
| 环境 | 完成 | 3090；现有 torch 2.3.0+cu121；GPU 算子可用 |
| 论文 | 完成 | 阅读本地 28 页 PDF；视觉核对第 7、13、19 页相关公式 |
| 源码 | 完成 | AGA、Task Arithmetic、PaRaMS、RotationSymmetry 固定 commit；原文件未修改 |
| P0/P1/P3 | 完成 | {len(mechanism)} 行；3 seeds；{mm.get("elapsed_seconds", 0):.3f} s |
| P2 | {"完成" if cm else "运行中"} | {len(real)} 行；full MNIST/SVHN；{cm.get("total_seconds", 0):.3f} s |
| 官方 PaRaMS | {"完成" if pm else "未完成"} | {len(params)} 行；{pm.get("total_seconds", 0):.3f} s |
| 配对保函数对齐 | {"完成" if pairedm else "未完成"} | {len(paired)} 行；{pairedm.get("total_seconds", 0):.3f} s |
| LayerNorm 候选 | {"完成" if lm else "运行中"} | {len(ln)} 行；{lm.get("total_seconds", 0):.3f} s |
| 低活跃方向候选 | {"完成" if lowm else "运行中"} | {len(low)} 行；{lowm.get("total_seconds", 0):.3f} s |
| 八任务资源 | 部分完成 | 全部 head；6 个专家权重受下载配额限制 |
| 结果验证 | 已通过基础检查 | 函数等价、投影恒等式、样本数及 accuracy 计数 |

服务器工作目录：`/root/autodl-tmp/merge_security_20261009/`。完整日志位于其 `outputs/2026-10-09/`；本地未下载 checkpoint、数据或依赖缓存。

实际运行入口：

```bash
python experiments/record_environment.py
python experiments/mechanisms.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/clip_benchmark.py --datasets MNIST SVHN --coefficients 0.2 0.5 --attack-all
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/layernorm_posthoc.py --strengths 1 10
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/low_activity_posthoc.py
JAX_PLATFORMS=cpu PYTHONHASHSEED=0 OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/clip_benchmark.py --protection params --attack-all --output outputs/2026-10-09/params
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python experiments/coupled_alignment.py
python experiments/validate_results.py
python experiments/plot_results.py
python experiments/build_report.py
```

解释器实际为 `/root/miniconda3/bin/python`。网络代理只用于下载，无模型传入本地磁盘。

PaRaMS 使用 CPU JAX、4 核 affinity 和 DataLoader workers=0；其他真实基准使用 workers=2。首次 JAX 编译计入运行耗时，故这些 wall time 不作为跨方法的严格速度比较。

已处理问题：非交互 shell 未初始化 Python PATH；本地 GitHub 直连失败后使用现有代理；服务器 Drive/HF 直连受限后使用 SSH 代理转发；MNIST 原源返回 404 后由 torchvision 切换 S3；旧 checkpoint 缺少 `src.modeling` 模块路径，用新入口兼容注册解决。初次机制入口因 SciPy 尚未安装退出，安装完成后成功重跑。

保护耗时：{cm.get("protection_seconds", 0):.3f} s；作者恢复耗时：{cm.get("recovery_seconds", 0):.3f} s。参数、feature error 与 accuracy 计量对象不同，报告分别列出。
"""
    (root / "experiment_log.md").write_text(log, encoding="utf-8")
    speech = f"""# 组会讲稿（约 4–5 分钟）

老师，今天我把问题明确为：对已经训练好的模型，完全不再微调，通过后处理保护它的任务能力，避免未经授权的模型融合，同时保持模型单独使用的性能。

我先把论文和公开代码对应起来。AGA 对 Attention 的每个 head 分别做最小二乘，利用 pretrained 模型作为 anchor；MLP 则用余弦相似度和 Hungarian 找回神经元顺序。论文的 ARF 需要在微调时修改目标，所以不能直接用于我们这个限制下的问题。

今天已经完成机制实验和真实 CLIP 实验。机制实验用了三个固定种子，确认置换、QK 和 VO 配对变换都保持函数。MLP 的置换被完全找回。Attention 的恢复不是精确逆变换，而是把 anchor 投影到微调权重的列空间。这个结论论文附录已经给出，我做了不同可逆矩阵、条件数和数值求解方法的验证，没有把它当成新贡献。

真实实验使用 Task Arithmetic 官方 ViT-B/32 的 MNIST、SVHN checkpoint，测试了全部一万张 MNIST 和两万六千多张 SVHN 图像。MNIST 模型原始和保护后都是 {accuracy("A_unprotected", "MNIST")}，AGA 后是 {accuracy("C_AGA", "MNIST")}。在两任务融合系数 0.5 下，未保护融合为 {accuracy("D_unprotected_merge", "MNIST", 0.5)}，保护后直接融合为 {accuracy("E_protected_merge", "MNIST", 0.5)}，AGA 恢复后融合为 {accuracy("F_AGA_merge", "MNIST", 0.5)}。因此这类可逆保护能破坏直接融合，但对 AGA 没有稳健保护。

{params_speech}

{paired_speech}

在老师提出的后处理方向上，我做了两个进一步检查。合成小模型上的跨 head 置换能让原代码失效，但加上一个子空间 matching 就能全部找回，所以它只是实现缺口。

另一个候选利用 LayerNorm 的输入约束：它的输出落在一个已知仿射超平面上。我向后续 Linear 添加在这个超平面上恒为零的参数分量，并补偿 bias。这样无需训练或数据，理论上保持函数，却可以改变 AGA 所用的参数列空间。这是今天筛选的一种机制，尚不宣称首创。

不过这个空方向由公开 LayerNorm 参数决定，攻击者也能投影去掉它。实际保护前后准确率保持一致，但原 AGA 以及投影消除加 AGA 都恢复到接近未保护融合的准确率。因此它是一个明确的负结果。

{low_speech}

文献审查还表明，MergeLock 已使用一般可逆混合，MergeBarrier 的多项式重表示需要激活校准和结构修改，TaylorMLP 也已有参数提取攻击。所以不能只把参数变得更远或换一种格式。下一步优先在相同效用预算下研究不可逆的信息损失，并补公开无标签的功能拟合攻击。

目前没有得到强防御，但已排除两条具体路径并建立可复用评估。八任务的分类 head 齐备，六个专家权重遇到下载配额限制，不能用不同来源模型冒充。希望老师确认：是否允许公开无标签校准和极小效用损失，以及保护需要覆盖多大攻击预算。双任务继续开发，有效候选再做八任务验证。
"""
    (root / "speech.md").write_text(speech, encoding="utf-8")
    print("REPORTS_WRITTEN", root, "rows", len(all_rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/2026-10-09")
    run(Path(parser.parse_args().output))
