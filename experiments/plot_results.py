"""Publication-style static figures from actual results; no handwritten numbers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

COLORS = ["#2867A6", "#D48B26", "#478B67", "#AC536B"]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.dpi": 220,
        "pdf.fonttype": 42,
    }
)


def save(fig, out: Path, name: str):
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.svg", bbox_inches="tight")
    svg = out / f"{name}.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def run(root: Path):
    out = root / "figures"
    out.mkdir(exist_ok=True)
    rows = json.loads((root / "mechanism_results.json").read_text())
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), layout="constrained")
    magnitudes = [0.001, 0.01, 0.1, 1.0]
    for idx, metric in enumerate(["recovered_weight_error", "recovered_output_error"]):
        for j, experiment in enumerate(["mlp_permutation", "attention_qk_vo"]):
            values = np.array(
                [
                    [
                        r[metric]
                        for r in rows
                        if r["experiment"] == experiment and r["magnitude"] == m
                    ]
                    for m in magnitudes
                ]
            )
            means = values.mean(1)
            if j == 0:
                axes[idx].text(
                    0.02,
                    0.92,
                    "MLP: exact recovery (error = 0)",
                    transform=axes[idx].transAxes,
                    fontsize=8,
                    color=COLORS[j],
                )
                continue
            axes[idx].plot(
                magnitudes,
                means,
                marker="o",
                color=COLORS[j],
                label="MLP permutation (exact recovery)" if j == 0 else "Attention QK/VO",
            )
            if j == 1:
                axes[idx].fill_between(
                    magnitudes,
                    values.min(1),
                    values.max(1),
                    color=COLORS[j],
                    alpha=0.18,
                )
        axes[idx].set(
            xscale="log",
            yscale="log",
            xlabel="Synthetic task-change magnitude",
            ylabel="Relative Frobenius error" if idx == 0 else "Relative output error",
            title=f"({'ab'[idx]}) Author AGA recovery",
        )
        axes[idx].grid(alpha=0.2)
    axes[0].legend(loc="lower right")
    fig.suptitle("Mechanism verification: 3 seeds, toy modules; shading = min-max", fontsize=10)
    save(fig, out, "mechanisms")
    p3 = [r for r in rows if r["experiment"] == "heads_plus_orthogonal"]
    fig, ax = plt.subplots(figsize=(4.4, 2.7), layout="constrained")
    fields = ["protected_output_error", "recovered_output_error", "adaptive_output_error"]
    values = np.array([[r[k] for r in p3] for k in fields])
    ax.bar(range(3), values.mean(1), color=COLORS[:3])
    ax.errorbar(
        range(3),
        values.mean(1),
        yerr=[values.mean(1) - values.min(1), values.max(1) - values.mean(1)],
        fmt="none",
        color="black",
        capsize=3,
    )
    ax.set(
        yscale="log",
        ylabel="Relative output error",
        xticks=range(3),
        xticklabels=["Protected\nstandalone", "Author AGA", "Head matching\n+ AGA"],
        title="Head permutation + orthogonal bases (toy, 3 seeds)",
        xlabel="Bars: mean; error bars: min-max across seeds",
    )
    ax.grid(axis="y", alpha=0.2)
    save(fig, out, "head_permutation")
    if (root / "clip_results.json").exists():
        real = json.loads((root / "clip_results.json").read_text())
        clip_rows = real
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
        groups = ["D_unprotected_merge", "E_protected_merge", "F_AGA_merge"]
        for i, dataset in enumerate(["MNIST", "SVHN"]):
            for j, coef in enumerate([0.2, 0.5]):
                y = [
                    100
                    * next(
                        r["top1"]
                        for r in real
                        if r["group"] == g and r["dataset"] == dataset and r["coefficient"] == coef
                    )
                    for g in groups
                ]
                axes[i].bar(
                    np.arange(3) + (j - 0.5) * 0.35,
                    y,
                    width=0.35,
                    label=f"TA coefficient {coef}",
                    color=COLORS[j],
                )
            axes[i].set(
                xticks=range(3),
                xticklabels=["Unprotected", "Protected", "AGA"],
                ylim=(0, 105),
                ylabel="Test accuracy (%)",
                title=f"({'ab'[i]}) {dataset}",
            )
            axes[i].grid(axis="y", alpha=0.2)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside lower center", ncol=2, fontsize=8)
        fig.suptitle("Official CLIP ViT-B/32 checkpoints, MNIST + SVHN two-task merge", fontsize=10)
        save(fig, out, "clip_merging")
        if any(r["group"] == "H_AGA_all_protected_merge" for r in real):
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), layout="constrained")
            for i, dataset in enumerate(["MNIST", "SVHN"]):
                groups = [
                    "D_unprotected_merge",
                    "F_AGA_merge",
                    "G_AGA_all_unprotected_merge",
                    "H_AGA_all_protected_merge",
                ]
                y = [
                    100
                    * next(
                        r["top1"]
                        for r in real
                        if r["group"] == g and r["dataset"] == dataset and r["coefficient"] == 0.5
                    )
                    for g in groups
                ]
                axes[i].bar(range(4), y, color=COLORS)
                axes[i].set(
                    xticks=range(4),
                    xticklabels=[
                        "No defense",
                        "AGA\nprotected",
                        "All AGA\nunprotected",
                        "All AGA\nprotected",
                    ],
                    ylim=(0, 105),
                    ylabel="Test accuracy (%)",
                    title=f"({'ab'[i]}) {dataset}",
                )
                axes[i].grid(axis="y", alpha=0.2)
            for ax in axes:
                ax.tick_params(axis="x", labelsize=7.5)
            fig.suptitle("Defense-agnostic AGA control: two tasks, TA coefficient 0.5", fontsize=10)
            save(fig, out, "defense_agnostic")
    if (root / "layernorm/layernorm_metadata.json").exists():
        real = json.loads((root / "layernorm/layernorm_results.json").read_text())
        groups = [
            "LN_null_plus_basis",
            "LN_null_AGA_merge",
            "LN_null_canonical_AGA_merge",
            "canonical_unprotected_merge",
        ]
        labels = [
            "Protected\nstandalone",
            "AGA\nmerge",
            "Canonicalization\n+ AGA merge",
            "Canonical\nbaseline merge",
        ]
        fig, ax = plt.subplots(figsize=(6.3, 2.8), layout="constrained")
        strengths = sorted(set(r["strength"] for r in real))
        for i, strength in enumerate(strengths):
            y = [
                100 * next(r["top1"] for r in real if r["group"] == g and r["strength"] == strength)
                for g in groups
            ]
            ax.plot(
                np.arange(4) + (i - (len(strengths) - 1) / 2) * 0.35,
                y,
                "o",
                label=f"Null strength {strength:g}",
                color=COLORS[i],
            )
        ax.set(
            xticks=range(4),
            xticklabels=labels,
            ylim=(98.8, 100.0),
            ylabel="MNIST accuracy (%; zoomed axis)",
            title="Post-hoc LayerNorm-null exploration (real model, full MNIST test)",
        )
        ax.grid(axis="y", alpha=0.2)
        fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=8)
        save(fig, out, "layernorm_exploration")
        low_path = root / "low_activity/low_activity_results.json"
        if low_path.exists():
            low = json.loads(low_path.read_text())

            def get(rows, group, **filters):
                return 100 * next(
                    r["top1"]
                    for r in rows
                    if r["group"] == group and all(r.get(k) == v for k, v in filters.items())
                )

            names = [
                "Unprotected",
                "Basis +\npermutation",
                "LN null +\nbasis",
                "Low-activity\n+ basis",
            ]
            ln_rows = json.loads((root / "layernorm/layernorm_results.json").read_text())
            standalone = [
                get(clip_rows, "A_unprotected", dataset="MNIST"),
                get(clip_rows, "B_protected", dataset="MNIST"),
                get(ln_rows, "LN_null_plus_basis", strength=10),
                get(low, "low_activity_standalone"),
            ]
            author_attack = [
                get(clip_rows, "D_unprotected_merge", dataset="MNIST", coefficient=0.5),
                get(clip_rows, "F_AGA_merge", dataset="MNIST", coefficient=0.5),
                get(ln_rows, "LN_null_AGA_merge", strength=10),
                get(low, "low_activity_AGA_merge"),
            ]
            adaptive_attack = [
                author_attack[0],
                author_attack[1],
                get(ln_rows, "LN_null_canonical_AGA_merge", strength=10),
                get(low, "low_activity_projection_AGA_merge"),
            ]
            fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
            axes[0].plot(range(4), standalone, "o", color=COLORS[0])
            axes[0].axhline(99.19, color=COLORS[3], linestyle="--", linewidth=1)
            axes[0].text(
                0.03,
                0.52,
                "Low-activity budget: 99.19%",
                transform=axes[0].transAxes,
                fontsize=7,
                color=COLORS[3],
            )
            axes[0].set(
                ylim=(98.8, 100.0),
                ylabel="Accuracy (%; zoomed axis)",
                title="(a) Standalone utility (higher is better)",
            )
            for x, value in enumerate(standalone):
                axes[0].annotate(
                    f"{value:.2f}",
                    (x, value),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha="center",
                    fontsize=7,
                )
            x = np.arange(4)
            axes[1].bar(
                x - 0.17,
                author_attack,
                width=0.34,
                color=COLORS[1],
                label="Author AGA / no-defense reference",
            )
            axes[1].bar(
                x[2:] + 0.17,
                adaptive_attack[2:],
                width=0.34,
                color=COLORS[2],
                label="Adaptive projection + AGA",
            )
            axes[1].set(
                ylim=(0, 110),
                ylabel="Accuracy (%)",
                title="(b) Attacked merge (lower protects better)",
            )
            for ax in axes:
                ax.set(xticks=range(4), xticklabels=names)
                ax.grid(axis="y", alpha=0.2)
            fig.legend(
                *axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=7
            )
            fig.suptitle(
                "Post-hoc screening: full MNIST test, one seed; TA coefficient 0.5", fontsize=10
            )
            save(fig, out, "posthoc_screening")
    if (root / "params/clip_metadata.json").exists():
        params = json.loads((root / "params/clip_results.json").read_text())
        basis = json.loads((root / "clip_results.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
        for ax, dataset in zip(axes, ["MNIST", "SVHN"], strict=True):
            groups = [
                ("D_unprotected_merge", "E_protected_merge", "E_protected_merge"),
                (None, "F_AGA_merge", "F_AGA_merge"),
                (
                    "G_AGA_all_unprotected_merge",
                    "H_AGA_all_protected_merge",
                    "H_AGA_all_protected_merge",
                ),
            ]
            for j, (labels, name) in enumerate(
                zip(groups, ["No attack", "AGA protected model", "AGA all models"], strict=True)
            ):
                xs, ys = [], []
                for i, (group, data) in enumerate(zip(labels, [basis, basis, params], strict=True)):
                    if group is not None:
                        xs.append(i + (j - 1) * 0.24)
                        ys.append(
                            100
                            * next(
                                r["top1"]
                                for r in data
                                if r["group"] == group
                                and r["dataset"] == dataset
                                and r["coefficient"] == 0.5
                            )
                        )
                ax.bar(xs, ys, width=0.23, color=COLORS[j], label=name)
            ax.set(
                xticks=range(3),
                xticklabels=["No defense", "Basis + MLP\npermutation", "PaRaMS\n(basic)"],
                ylim=(0, 105),
                ylabel="Top-1 accuracy (%)",
                title=dataset,
            )
            ax.grid(axis="y", alpha=0.2)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=3)
        fig.suptitle(
            "Official PaRaMS baseline: full tests, one seed; TA coefficient 0.5", fontsize=10
        )
        save(fig, out, "params_comparison")
    if (root / "rank_probe/metadata.json").exists():
        rank = json.loads((root / "rank_probe/results.json").read_text())
        names = ["activation_rank32", "activation_rank16", "weight_rank32", "random_drop10"]

        def value(preset, group, dataset):
            return 100 * next(
                r["top1"]
                for r in rank
                if r["preset"] == preset and r["group"] == group and r["dataset"] == dataset
            )

        fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.8), layout="constrained")
        xs = np.arange(len(names))
        axes[0].bar(
            xs,
            [
                value(p, "standalone", "MNIST") - value("unprotected", "standalone", "MNIST")
                for p in names
            ],
            color=COLORS[0],
        )
        axes[0].set(title="(a) Standalone MNIST", ylabel="Accuracy change (pp)", ylim=(-0.55, 0.15))
        axes[0].axhline(-0.5, color=COLORS[3], linestyle="--", linewidth=1)
        axes[0].text(
            0.02, 0.13, "Utility budget: -0.50 pp", transform=axes[0].transAxes, fontsize=7
        )
        for ax, dataset in zip(axes[1:], ["MNIST", "SVHN"], strict=True):
            for j, group in enumerate(["direct", "AGA", "completion"]):
                changes = [
                    value(p, group, dataset) - value("unprotected", group, dataset) for p in names
                ]
                ax.bar(
                    xs + (j - 1) * 0.24,
                    changes,
                    width=0.23,
                    color=COLORS[j],
                    label={
                        "direct": "Direct TA",
                        "AGA": "Author AGA",
                        "completion": "Pretrained completion",
                    }[group],
                )
            ax.set(title=f"Merged {dataset}", ylabel="Accuracy change (pp)", ylim=(-0.65, 0.5))
        for ax in axes:
            ax.axhline(0, color="black", linewidth=0.7)
            ax.set(xticks=xs, xticklabels=["Act.\nr=32", "Act.\nr=16", "Weight\nr=32", "Drop\n10%"])
            ax.grid(axis="y", alpha=0.2)
        fig.legend(*axes[1].get_legend_handles_labels(), loc="outside lower center", ncol=3)
        fig.suptitle(
            "Rank-loss screening: differences from matched unprotected controls; full tests, one seed",
            fontsize=10,
        )
        save(fig, out, "rank_screening")
    for name in ["polynomial_probe", "polynomial_multilayer"]:
        if not (root / name / "metadata.json").exists():
            continue
        rows = json.loads((root / name / "results.json").read_text())
        meta = json.loads((root / name / "metadata.json").read_text())
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
        groups = [
            "original_direct_merge",
            "shared_format_merge",
            "shared_format_AGA_merge",
            "reconstructed_protected_merge",
            "reconstructed_control_merge",
        ]
        for ax, dataset in zip(axes, ["MNIST", "SVHN"], strict=True):
            values = [
                100 * next(r["top1"] for r in rows if r["group"] == g and r["dataset"] == dataset)
                for g in groups
            ]
            ax.bar(
                range(len(groups)),
                values,
                color=[COLORS[0], COLORS[1], COLORS[1], COLORS[2], COLORS[2]],
            )
            for x, y in enumerate(values):
                ax.text(x, y + 1, f"{y:.2f}", ha="center", fontsize=7)
            ax.set(
                xticks=range(len(groups)),
                xticklabels=[
                    "Original\nTA",
                    "Shared\nformat",
                    "Shared\n+ AGA",
                    "Protected\nrefit",
                    "Control\nrefit",
                ],
                ylim=(0, 112),
                ylabel="Top-1 accuracy (%)",
                title=dataset,
            )
            ax.grid(axis="y", alpha=0.2)
        fig.suptitle(
            f"Whole-FFN quadratic representation: {len(meta['aga_mlp_skipped_layers'])} block(s); full tests, one seed\n"
            + "Standalone MNIST: "
            + " -> ".join(
                f"{100 * next(r['top1'] for r in rows if r['group'] == g and r['dataset'] == 'MNIST'):.2f}%"
                for g in ["unprotected_standalone", "protected_standalone"]
            ),
            fontsize=10,
        )
        save(fig, out, name)
    print("FIGURES_WRITTEN", out)


def plot_subspace(root: Path):
    """Independent post-meeting probe; leave earlier report figures intact."""
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["completed"]
    subspaces = json.loads((root / "subspaces.json").read_text())
    presets = json.loads((root / "development_screen.json").read_text())
    selected = json.loads((root / "selection.json").read_text())["selected"]
    rows = json.loads((root / "results.json").read_text())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    values = [r for r in subspaces if r["site"].startswith("V:")]
    x = list(range(1, 13))
    for i, (task, label) in enumerate(
        [("MNIST", "MNIST"), ("SVHN_on_MNIST_expert", "SVHN on MNIST expert")]
    ):
        axes[0].plot(x, [r[task]["rank"] for r in values], "o-", color=COLORS[i], label=label)
    axes[0].set(
        xlabel="Transformer block (1-based)",
        ylabel="PCA dimensions for 99.99% energy",
        ylim=(0, 790),
        title="Attention input dimensions",
    )
    axes[0].legend(fontsize=7)
    for i, (key, label) in enumerate(
        [
            ("MNIST_expert_MNIST", "MNIST expert / MNIST"),
            ("MNIST_expert_SVHN", "MNIST expert / SVHN"),
            ("unprotected_merge_MNIST", "Merged model / MNIST"),
            ("unprotected_merge_SVHN", "Merged model / SVHN"),
        ]
    ):
        axes[1].semilogy(
            x,
            [max(r["input_complement"][key]["fraction"], 1e-12) for r in values],
            "o-",
            color=COLORS[i],
            label=label,
            markersize=3,
        )
    axes[1].set(
        xlabel="Transformer block (1-based)",
        ylabel="Input energy outside MNIST PCA span",
        title="Calibration distribution shift",
    )
    axes[1].legend(fontsize=6.8)
    fig.suptitle("256 training images per task; all tokens; one seed", fontsize=10)
    save(fig, out, "subspaces")

    fig, ax = plt.subplots(figsize=(5.4, 3.4), layout="constrained")
    for i, (layer, target) in enumerate([(0, "V"), (0, "FC"), (11, "V"), (11, "FC")]):
        group = [p for p in presets if p["site"] == f"{target}:{layer}"]
        ax.plot(
            [p["utility_loss_pp"] for p in group],
            [100 * p["mean_merged_accuracy"] for p in group],
            "o-",
            color=COLORS[i],
            label=f"{target}, block {layer + 1}",
            markersize=4,
        )
    if selected:
        ax.scatter(
            [selected["utility_loss_pp"]],
            [100 * selected["mean_merged_accuracy"]],
            marker="*",
            s=125,
            color="black",
            label="Selected using development data",
            zorder=4,
        )
    ax.axvline(0.5, ls="--", lw=0.8, color="gray", label="0.5 pp utility budget")
    ax.set_xscale("symlog", linthresh=0.5)
    ax.set_xticks([0, 0.5, 5, 50], labels=["0", "0.5", "5", "50"])
    ax.set(
        xlabel="Standalone MNIST development loss (pp; symlog)",
        ylabel="Mean merged development accuracy (%)",
        title="Fixed strength grid: 0.1, 1, 10, 100",
        ylim=(0, 105),
    )
    ax.legend(fontsize=7)
    ax.grid(alpha=0.15)
    save(fig, out, "development_screen")
    if selected:
        groups = ["direct", "AGA", "projection", "completion"]
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), layout="constrained")
        for ax, dataset in zip(axes, ["MNIST", "SVHN"], strict=True):
            for i, (preset, label) in enumerate(
                [("unprotected", "Unprotected, same attack"), (selected["name"], "Protected")]
            ):
                accuracy = [
                    100
                    * next(
                        r["top1"]
                        for r in rows
                        if r["split"] == "official_test"
                        and r["preset"] == preset
                        and r["group"] == group
                        and r["dataset"] == dataset
                    )
                    for group in groups
                ]
                ax.plot(range(4), accuracy, "o-", color=COLORS[i], label=label)
                for j, value in enumerate(accuracy):
                    ax.annotate(
                        f"{value:.2f}",
                        (j, value),
                        xytext=(0, 7 if i == 0 or value < 20 else -12),
                        textcoords="offset points",
                        ha="center",
                        fontsize=7,
                    )
            ax.set(
                xticks=range(4),
                xticklabels=["Direct", "AGA", "Projection", "Pretrained\ncompletion"],
                ylim=(0, 108),
                ylabel="Merged top-1 accuracy (%)",
                title=dataset,
            )
            ax.grid(axis="y", alpha=0.2)
        axes[0].legend(fontsize=7, loc="lower right")
        fig.suptitle("Full official tests; selected on development data; one seed", fontsize=10)
        save(fig, out, "attacks")
    print("SUBSPACE_FIGURES_WRITTEN", out)


def plot_subspace_diagnostics(root: Path):
    meta = json.loads((root / "metadata.json").read_text())
    assert meta["completed"]
    rows = json.loads((root / "results.json").read_text())
    out = root / "figures"
    out.mkdir(exist_ok=True)

    def accuracy(contrast, group, task):
        return 100 * next(
            r["top1"]
            for r in rows
            if r["contrast"] == contrast and r["group"] == group and r["dataset"] == task
        )

    components = ["V_full", "V_layernorm_component", "V_non_layernorm_component"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    for ax, task in zip(axes, ["MNIST", "SVHN"], strict=True):
        for i, group in enumerate(["direct", "AGA"]):
            values = [accuracy(c, group, task) for c in components]
            positions = np.arange(3) + (i - 0.5) * 0.34
            ax.bar(positions, values, width=0.32, color=COLORS[i], label=group)
            for x, y in zip(positions, values, strict=True):
                ax.text(x, y + (1.5 if i == 0 else 6), f"{y:.2f}", ha="center", fontsize=7)
        ax.axhline(accuracy("control", "direct", task), color="black", lw=0.8, ls=":")
        ax.set(
            xticks=range(3),
            xticklabels=["Full noise", "LN\ncomponent", "Other\ncomponent"],
            ylim=(0, 112),
            ylabel="Merged accuracy (%)",
            title=task,
        )
        ax.grid(axis="y", alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc="outside lower center", ncol=2)
    fig.suptitle("Additive decomposition; no rescaling; full-noise counts reused", fontsize=10)
    save(fig, out, "components")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8), layout="constrained")
    groups = ["direct", "AGA", "projection", "completion"]
    cleanup_root = root.parent / "anchor_cleanup"
    if (cleanup_root / "metadata.json").exists():
        cleanup_rows = json.loads((cleanup_root / "results.json").read_text())
        rows.extend({**r, "group": "norm_cleanup"} for r in cleanup_rows if r["group"] == "direct")
        groups.append("norm_cleanup")
    for ax, task in zip(axes, ["MNIST", "SVHN"], strict=True):
        for i, (contrast, label) in enumerate(
            [("control", "Unprotected, same attack"), ("FC_development_selected", "FC protected")]
        ):
            values = [accuracy(contrast, g, task) for g in groups]
            positions = np.arange(len(groups)) + (i - 0.5) * 0.35
            ax.bar(positions, values, width=0.33, color=COLORS[i], label=label)
            for x, y in zip(positions, values, strict=True):
                ax.text(x, y + (1.5 if i == 0 else 6), f"{y:.2f}", ha="center", fontsize=6.7)
        ax.set(
            xticks=range(len(groups)),
            xticklabels=["Direct", "AGA", "PCA\nproject", "PCA\nfill", "Norm\nreplace"][
                : len(groups)
            ],
            ylim=(0, 112),
            ylabel="Merged accuracy (%)",
            title=task,
        )
        ax.grid(axis="y", alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), fontsize=7, loc="outside lower center", ncol=2)
    fig.suptitle(
        "FC complement noise: selected from earlier development data; one seed", fontsize=10
    )
    save(fig, out, "fc_attacks")
    print("SUBSPACE_DIAGNOSTIC_FIGURES_WRITTEN", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/2026-10-09")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--subspace", action="store_true", help="Plot the standalone subspace probe.")
    mode.add_argument("--subspace-diagnostics", action="store_true", help="Plot causal contrasts.")
    args = parser.parse_args()
    runner = (
        plot_subspace_diagnostics
        if args.subspace_diagnostics
        else plot_subspace
        if args.subspace
        else run
    )
    runner(Path(args.output))
