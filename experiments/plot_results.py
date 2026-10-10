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


def plot_bounded_complement(root: Path):
    config = json.loads((root / "protocol.json").read_text())
    protected = config.get("protected_task", "MNIST")
    profile = json.loads((root / "replacement_profile.json").read_text())
    screen = json.loads((root / "development_screen.json").read_text())
    rows = json.loads((root / "results.json").read_text())
    selection = json.loads((root / "selection.json").read_text())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.2, 2.7), layout="constrained")
    for i, (kind, label) in enumerate(
        [("V", "V matrix"), ("FC", "FC matrix"), ("block", "Entire block")]
    ):
        values = {
            int(item["site"].split(":")[1]): item["standalone_loss_pp"]
            for item in profile
            if item["site"].startswith(kind + ":")
        }
        ax.plot(
            range(12),
            [values[layer] for layer in range(12)],
            marker=["o", "s", "^"][i],
            markersize=3,
            color=COLORS[i],
            label=label,
        )
    ax.axhline(config["utility_budget_pp"], ls=":", color="black", lw=0.8, label="Utility budget")
    ax.set(
        xticks=range(12),
        xlabel="Transformer block",
        ylabel="Standalone loss (pp)",
        title=f"Public pretrained replacement; protected {protected}; development data",
    )
    ax.grid(axis="y", alpha=0.2)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=7)
    save(fig, out, "replacement_profile")
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 5.3), layout="constrained", sharey=True)
    labels = [
        f"{100 * p['energy']:g}% / {p['site_group'].replace('V_FC_all', 'V+FC all').replace('FC_early', 'FC 0-3').replace('FC_all', 'FC all')} / {'remove' if p['scale'] == 1 else 'reflect'}"
        + (" *" if not p["utility_feasible"] else "")
        for p in screen
    ]
    lookup = {
        (r["preset"], r["group"], r["dataset"]): 100 * r["top1"]
        for r in rows
        if r["split"] == "development"
    }
    for ax, task in zip(axes, [protected, "mean"], strict=True):
        for i, group in enumerate(["direct", "angular"]):
            values = [
                100 * p[f"mean_{group}"] if task == "mean" else lookup[p["name"], group, task]
                for p in screen
            ]
            positions = np.arange(len(screen)) + (i - 0.5) * 0.34
            ax.barh(
                positions,
                values,
                height=0.32,
                color=COLORS[i],
                label=["Direct", "Angular cleanup"][i],
            )
        baseline = (
            sum(lookup["unprotected", "direct", t] for t in ["MNIST", "SVHN"]) / 2
            if task == "mean"
            else lookup["unprotected", "direct", task]
        )
        ax.axvline(baseline, ls=":", lw=0.8, color="black")
        ax.set(
            xlim=(0, 103),
            yticks=range(len(screen)),
            yticklabels=labels,
            xlabel="Merged accuracy (%)",
            title=f"Protected {protected}" if task != "mean" else "MNIST / SVHN mean",
        )
        ax.tick_params(axis="y", labelsize=7)
        ax.grid(axis="x", alpha=0.2)
    axes[0].invert_yaxis()
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2, fontsize=8)
    fig.suptitle(
        f"Development screening: {selection['recovery_gate_passed_count']}/12 passed; * utility over budget",
        fontsize=10,
    )
    save(fig, out, "bounded_screen")
    print("BOUNDED_FIGURES_WRITTEN", out)


def plot_distribution_shift(root: Path):
    rows = json.loads((root / "results.json").read_text())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9), layout="constrained")
    for ax, target in zip(axes, ["V", "FC"], strict=True):
        for i, task in enumerate(["MNIST", "SVHN"]):
            subset = {
                int(r["site"].split(":")[1]): r
                for r in rows
                if r["protected_task"] == task
                and r["retained_energy"] == 0.9999
                and r["site"].startswith(target + ":")
            }
            for group, style, label in [
                ("unprotected_merge", "-", "Merged"),
                ("expert", "--", "Expert"),
            ]:
                ax.plot(
                    range(12),
                    [100 * subset[layer][group]["fraction"] for layer in range(12)],
                    color=COLORS[i],
                    ls=style,
                    label=f"{label} / {task}",
                )
        ax.set(
            xticks=range(12),
            yscale="log",
            ylim=(0.002, 100),
            xlabel="Transformer block",
            ylabel="Energy outside own-task PCA (%)",
            title="Attention V input" if target == "V" else "MLP FC input",
        )
        ax.grid(axis="y", alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4, fontsize=7)
    fig.suptitle("Frozen PCA at 99.99%; training images; merge coefficient 0.5", fontsize=10)
    save(fig, out, "distribution_shift")
    print("SHIFT_FIGURE_WRITTEN", out)


def plot_task_support(root: Path):
    """Separate module costs used for selection from the frozen-support confirmation."""
    rows = json.loads((root / "results.json").read_bytes())
    selection = json.loads((root / "selection.json").read_bytes())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), layout="constrained")
    profiles = {tuple(map(tuple, r["support"])): r for r in rows if r["split"] == "selection"}
    for offset, component, label, color in [
        (-0.25, "attn", "Attention", COLORS[0]),
        (0, "mlp", "MLP", COLORS[1]),
        (0.25, "both", "Attention + MLP", COLORS[2]),
    ]:
        supports = [
            ((i, "attn"), (i, "mlp")) if component == "both" else ((i, component),)
            for i in range(12)
        ]
        losses = [profiles[key]["paired"]["loss_pp"] for key in supports]
        axes[0].bar(np.arange(12) + offset, losses, width=0.25, label=label, color=color)
    axes[0].axhline(0, linewidth=0.7, color="0.4")
    axes[0].set(
        title="(a) Single-layer public resets",
        xlabel="Block",
        ylabel="Standalone MNIST loss (pp)",
        xticks=range(12),
    )
    axes[0].legend(fontsize=7)
    confirm = [r for r in rows if r["split"] == "confirmation"]
    for offset, modified, label, color in [
        (-0.18, False, "Original", COLORS[0]),
        (0.18, True, "Selected support reset", COLORS[1]),
    ]:
        scores = [
            100
            * next(
                r["top1"] for r in confirm if r["group"] == group and bool(r["support"]) == modified
            )
            for group in ["standalone", "fusion"]
        ]
        bars = axes[1].bar(np.arange(2) + offset, scores, width=0.36, label=label, color=color)
        axes[1].bar_label(bars, fmt="%.3f", fontsize=8, padding=3)
    axes[1].set(
        title=f"(b) Frozen support ({selection['modules']} modules)",
        ylabel="MNIST accuracy (%)",
        xticks=[0, 1],
        xticklabels=["Standalone", "Fusion"],
        ylim=(90, 101.5),
    )
    axes[1].legend(loc="lower left", fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle(
        "MNIST-only support selection; not a defense result\n(a) inspected 2000 training images; (b) disjoint 4000 training images",
        fontsize=9,
    )
    save(fig, out, "task_support")
    print("TASK_SUPPORT_FIGURES_WRITTEN", out)


def plot_response_transfer(root: Path):
    """Plot diagnostic fits on a frozen release; these are not accuracy results."""
    rows = json.loads((root / "results.json").read_text())
    spectra = json.loads((root / "spectra.json").read_text())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    layers = [r["layer"] for r in spectra]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), layout="constrained")
    for ax, subset, title in zip(
        axes[:2],
        ["calibration", "confirmation_prefix"],
        ["(a) Calibration (256 images)", "(b) Inspected confirmation (256 images)"],
        strict=True,
    ):
        selected = {r["layer"]: r for r in rows if r["subset"] == subset}
        for key, label, color in [
            ("frozen", "Frozen ridge fit", COLORS[0]),
            ("least_squares", "Diagnostic truncated LS", COLORS[1]),
        ]:
            values = [100 * selected[layer][key]["relative_teacher_error"] for layer in layers]
            ax.plot(layers, values, marker="o", label=label, color=color)
        ax.set(title=title, xlabel="Block", ylabel="Relative block error (%)", xticks=layers)
        ax.set_yscale("log")
        ax.grid(alpha=0.2)
        ax.legend(fontsize=7)
    axes[2].plot(
        layers, [r["coefficient_norm_ratio"] for r in spectra], marker="o", color=COLORS[1]
    )
    axes[2].set(
        title="(c) Coefficient amplification",
        xlabel="Block",
        ylabel="LS / frozen coefficient norm",
        xticks=layers,
    )
    axes[2].set_yscale("log")
    axes[2].grid(alpha=0.2)
    fig.suptitle(
        "Frozen response-transfer diagnosis: no new release or accuracy claim\n"
        "All panels use logarithmic y scales; numerical truncation is not a rigorous error floor",
        fontsize=9,
    )
    save(fig, out, "response_transfer_probe")
    print("RESPONSE_TRANSFER_FIGURES_WRITTEN", out)


def plot_joint_diagnostics(root: Path):
    """Keep calibration fit, classification and confirmation error separate."""
    protocol = json.loads((root / "protocol.json").read_bytes())
    rows = json.loads((root / "results.json").read_bytes())
    errors = json.loads((root / "layer_errors.json").read_bytes())
    fits = json.loads((Path(protocol["parent"]) / "fitting.json").read_bytes())
    keyed = {(r["subset"], r["group"]): r["top1"] * 100 for r in rows}
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), layout="constrained")
    subsets = list(protocol["subsets"])
    for i, group in enumerate(["unprotected", "joint_transfer"]):
        values = [keyed[s, group] for s in subsets]
        positions = np.arange(2) + (i - 0.5) * 0.3
        axes[0].bar(
            positions,
            values,
            width=0.28,
            color=COLORS[i],
            label="Unprotected" if i == 0 else "Rejected release",
        )
        for x, value in zip(positions, values, strict=True):
            axes[0].text(x, value + 1, f"{value:.2f}", ha="center", fontsize=7)
    axes[0].set(
        xticks=[0, 1],
        xticklabels=["Calibration\n256 images", "Confirmation prefix\n256 inspected images"],
        ylim=(0, 108),
        ylabel="Accuracy (%)",
        title="Calibration utility is insufficient",
    )
    axes[0].legend(fontsize=7, loc="lower left")
    for i, subset in enumerate(subsets):
        values = [e["actual_block_relative_error"] * 100 for e in errors if e["subset"] == subset]
        axes[1].plot(
            range(12),
            values,
            "o-",
            color=COLORS[i],
            label="Calibration" if i == 0 else "Confirmation prefix",
        )
    axes[1].set(
        xlabel="Block",
        ylabel="Relative block output error (%)",
        title="Error outside the fitting images",
        xticks=[0, 3, 6, 9, 11],
    )
    axes[1].legend(fontsize=7)
    for i, (name, label) in enumerate(
        [("attention_input_projection", "QKV"), ("mlp_input_projection", "MLP input")]
    ):
        tail = [(1 - f[name]["centered_input_energy_retained"]) * 100 for f in fits]
        response = [f[name]["linear_response_energy_removed_fraction"] * 100 for f in fits]
        axes[2].scatter(tail, response, color=COLORS[i], label=label, s=22)
    axes[2].set(
        xscale="log",
        xlabel="Discarded input variance (%)",
        ylabel="Removed linear response energy (%)",
        title="Input variance is a limited proxy",
    )
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(alpha=0.2)
    fig.suptitle(
        "Fixed joint-transfer diagnosis: training subsets, no parameter refit or new candidate\n"
        "Confirmation prefix was already evaluated; representation error is not an accuracy bound",
        fontsize=9,
    )
    save(fig, out, "joint_transfer_diagnostics")
    print("JOINT_DIAGNOSTIC_FIGURES_WRITTEN", out)


def plot_eight_protection(root: Path):
    """Plot per-task author-AGA controls without an incomplete eight-task mean."""
    rows = json.loads((root / "results.json").read_bytes())
    protocol = json.loads((root / "protocol.json").read_bytes())
    assert len(rows) == 56 and not protocol["table6_reproduction"]
    tasks = protocol["evaluation_tasks"]
    keyed = {(r["dataset"], r["group"]): r["top1"] * 100 for r in rows}
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(
        1, 2, figsize=(10.5, 3.25), layout="constrained", width_ratios=[2.8, 1]
    )
    groups = ["unprotected_merge", "params_merge", "AGA_unprotected_merge", "AGA_params_merge"]
    labels = ["Unprotected", "PaRaMS", "AGA on unprotected", "AGA on PaRaMS"]
    positions = np.arange(len(tasks))
    for idx, (group, label) in enumerate(zip(groups, labels, strict=True)):
        values = [keyed[t, group] for t in tasks]
        axes[0].bar(
            positions + (idx - 1.5) * 0.2, values, width=0.18, color=COLORS[idx], label=label
        )
    axes[0].set(
        xticks=positions,
        xticklabels=tasks,
        ylim=(0, 105),
        ylabel="Task accuracy (%)",
        title="Eight experts fused; protect and attack one expert at a time",
    )
    axes[0].tick_params(axis="x", rotation=20)
    axes[0].grid(axis="y", alpha=0.2)
    utility = [keyed[t, "unprotected_standalone"] - keyed[t, "params_standalone"] for t in tasks]
    gap = [keyed[t, "AGA_unprotected_merge"] - keyed[t, "AGA_params_merge"] for t in tasks]
    axes[1].plot(positions, utility, "o-", color=COLORS[1], label="Standalone loss")
    axes[1].plot(positions, gap, "s--", color=COLORS[3], label="Matched AGA gap")
    bound = max(0.6, max(abs(v) for v in utility + gap) * 1.2)
    axes[1].set(
        xticks=positions,
        xticklabels=tasks,
        ylim=(-bound, bound),
        ylabel="Difference (percentage points)",
        title="Matched differences",
    )
    axes[1].tick_params(axis="x", rotation=60)
    axes[1].axhline(0, color="0.65", linewidth=0.7)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend(fontsize=7, loc="upper right")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    fig.suptitle(
        "Official PaRaMS and author AGA: seven full test splits, TA coefficient 0.2\n"
        "EuroSAT split unresolved; supplementary check, not Table 6 reproduction",
        fontsize=9,
    )
    save(fig, out, "params_seven_protection")
    print("EIGHT_PROTECTION_FIGURES_WRITTEN", out)


def plot_joint_transfer(root: Path):
    """Show utility, every cheap attack and the ridge transfer fit separately."""
    out = root / "figures"
    out.mkdir(exist_ok=True)
    rows = json.loads((root / "results.json").read_bytes())
    fits = json.loads((root / "fitting.json").read_bytes())
    lookup = {(r["preset"], r["group"], r["dataset"]): r for r in rows}
    groups = json.loads((root / "protocol.json").read_bytes())["groups"]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.4), layout="constrained")
    presets = ["unprotected", "joint_transfer"]
    values = [100 * lookup[p, "direct_standalone", "MNIST"]["top1"] for p in presets]
    axes[0].bar(range(2), values, color=COLORS[:2], width=0.6)
    for i, value in enumerate(values):
        axes[0].text(i, value + 0.5, f"{value:.3f}", ha="center", fontsize=8)
    axes[0].set(
        xticks=range(2),
        xticklabels=["Original", "Joint transfer"],
        ylim=(90, 101),
        ylabel="Accuracy (%)",
        title="(a) Standalone MNIST (axis starts at 90%)",
    )
    x = np.arange(len(groups))
    for i, preset in enumerate(presets):
        scores = [100 * lookup[preset, g, "MNIST"]["top1"] for g in groups]
        axes[1].plot(
            x,
            scores,
            marker="o",
            markersize=4,
            color=COLORS[i],
            label="Original" if i == 0 else "Joint transfer",
        )
    axes[1].set(
        xticks=x,
        xticklabels=["Direct", "FC", "MLP", "Attn", "Outputs", "A+M", "Sign\n+ MLP"],
        ylim=(0, 105),
        ylabel="Fusion MNIST (%)",
        title="(b) Attacker chooses the best result",
    )
    axes[1].legend(loc="lower right", fontsize=7)
    for name, label in [
        ("residual_target_relative_error", "Transferred response"),
        ("original_block_relative_error", "Whole block output"),
    ]:
        axes[2].plot(
            [f["layer"] for f in fits],
            [100 * f[name] for f in fits],
            marker="o",
            markersize=3,
            label=label,
        )
    axes[2].set(
        xlabel="Transformer block (0-based)",
        ylabel="Relative fit error (%)",
        title="(c) Unlabeled calibration, not accuracy",
    )
    axes[2].legend(fontsize=7)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "CLIP ViT-B/32: frozen construction rejected; MNIST training confirmation, n=4000",
        fontsize=10,
    )
    save(fig, out, "joint_transfer")


def plot_redundancy_erasure(root: Path):
    rows = json.loads((root / "results.json").read_bytes())
    decision = json.loads((root / "decision.json").read_bytes())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    lookup = {(r["preset"], r["group"], r["dataset"]): r for r in rows}
    presets = ["unprotected", "erasure_only", "frozen_release", "combined"]
    names = ["Original", "Attention\nerasure", "Frozen\nrelease", "Combined"]
    panels = [
        ("direct_standalone", "MNIST", "Standalone MNIST"),
        ("direct", "MNIST", "Direct merge"),
        ("pretrained_fc", "MNIST", "FC replacement"),
        ("pretrained_mlp_all", "MNIST", "All-MLP replacement"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.8, 3.1), layout="constrained")
    for ax, (group, task, title) in zip(axes, panels, strict=True):
        values = [100 * lookup[p, group, task]["top1"] for p in presets]
        ax.bar(range(len(presets)), values, color=COLORS[:4])
        for pos, value in enumerate(values):
            ax.text(pos, value + 0.9, f"{value:.3f}", ha="center", fontsize=7)
        ax.set(
            xticks=range(len(presets)),
            xticklabels=names,
            ylim=(0, 108),
            title=title,
            ylabel="Accuracy (%)",
        )
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    outcome = "passed necessary gate" if decision["passed_initial_gate"] else "rejected"
    fig.suptitle(
        f"Task-update erasure: {outcome}; inspected MNIST training subset n=4,000; TA 0.5\n"
        "Frozen protection reused; no new perturbation, fitting or strength selection",
        fontsize=9,
    )
    save(fig, out, "redundancy_erasure")
    print("REDUNDANCY_FIGURES_WRITTEN", out)


def plot_operator_diagnostics(root: Path):
    rows = json.loads((root / "results.json").read_bytes())
    config = json.loads((root / "protocol.json").read_bytes())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    lookup = {(r["preset"], r["split"], r["group"], r["dataset"]): r for r in rows}
    if not config.get("components"):
        task = config.get("protected_task", config.get("task"))
        official = "projection_sha256" in config
        split = "test" if official else "confirmation"
        panels = (
            [("direct", "MNIST"), ("direct", "SVHN")]
            if official
            else [
                ("standalone", task),
                ("direct", task),
            ]
        )
        presets = list(dict.fromkeys(r["preset"] for r in rows if r["split"] == split))
        names = {
            "unprotected": "Original",
            "joint_mlp": "All MLPs",
            "joint_attn": "All attention",
            "joint_block": "All blocks\n(incl. LN)",
            "joint_norms": "LayerNorm\nonly",
            "joint_attn_mlp": "Attention\n+ MLP",
            "public_backbone": "Public\nbackbone",
            "readout_unprotected": "Readout\ncontrol",
            "readout_protected": "Readout\nprotected",
        }
        fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
        for ax, (group, task) in zip(axes, panels, strict=True):
            values = [100 * lookup[p, split, group, task]["top1"] for p in presets]
            ax.bar(range(len(presets)), values, color=COLORS[: len(presets)])
            for i, value in enumerate(values):
                ax.text(i, value + 0.65, f"{value:.3f}", ha="center", fontsize=8)
            ax.set(
                xticks=range(len(presets)),
                xticklabels=[names[p] for p in presets],
                ylim=(0, 105),
                ylabel="Accuracy (%)",
                title=f"{'Standalone' if group == 'standalone' else 'Merged'} {task}"
                + (f" (n={lookup[presets[0], split, group, task]['n']:,})" if official else ""),
            )
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="y", alpha=0.2)
        count = lookup[presets[0], split, panels[0][0], panels[0][1]]["n"]
        fig.suptitle(
            "Official complete tests; frozen output-only recovery; TA 0.5"
            if official
            else f"Frozen training confirmation subset: n={count:,}; TA 0.5; no new defense",
            fontsize=9,
        )
        save(fig, out, "readout_test" if official else "operator_confirmation")
        print("OPERATOR_FIGURES_WRITTEN", out)
        return
    components, layers = config["components"], config["layers"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), layout="constrained")
    matrices = [
        np.array(
            [
                [
                    lookup[f"replace_L{layer}_{component}", "development", group, "SVHN"]["paired"][
                        "loss_pp"
                    ]
                    for layer in layers
                ]
                for component in components
            ]
        )
        for group in ["standalone", "direct"]
    ]
    bound = max(0.1, max(float(np.abs(m).max()) for m in matrices))
    for ax, values, title in zip(axes, matrices, ["Standalone SVHN", "Merged SVHN"], strict=True):
        picture = ax.imshow(values, cmap="RdBu_r", vmin=-bound, vmax=bound, aspect="auto")
        ax.set(
            xticks=range(len(layers)),
            xticklabels=layers,
            yticks=range(len(components)),
            yticklabels=components,
            xlabel="Transformer block",
            title=title,
        )
        for i in range(len(components)):
            for j in range(len(layers)):
                ax.text(
                    j,
                    i,
                    f"{values[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=8,
                    color="white" if abs(values[i, j]) > 0.6 * bound else "black",
                )
    fig.colorbar(picture, ax=axes, label="Accuracy loss vs control (pp)", shrink=0.85)
    fig.suptitle("Single-component pretrained replacement; development SVHN n=1,000", fontsize=10)
    save(fig, out, "component_profile")
    presets = config["confirmation_presets"]
    labels = ["Original", "Replace blocks", "Replace MLPs", "Reconstruct MLPs"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    for ax, group, title in zip(
        axes, ["standalone", "direct"], ["Standalone SVHN", "Merged SVHN"], strict=True
    ):
        values = [100 * lookup[preset, "confirmation", group, "SVHN"]["top1"] for preset in presets]
        ax.bar(range(len(presets)), values, color=COLORS)
        for i, value in enumerate(values):
            ax.text(i, value + 0.65, f"{value:.3f}", ha="center", fontsize=8)
        ax.set(
            xticks=range(len(presets)),
            xticklabels=labels,
            ylim=(0, 105),
            ylabel="Top-1 accuracy (%)",
            title=title,
        )
        ax.tick_params(axis="x", rotation=22, labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle(
        "Fixed confirmation subset: 4,000 disjoint training images; no new defense", fontsize=10
    )
    save(fig, out, "operator_confirmation")
    print("OPERATOR_FIGURES_WRITTEN", out)


def plot_nufilt_alignment(root: Path):
    result = json.loads((root / "results.json").read_bytes())
    rows, config = result["rows"], result["protocol"]
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), layout="constrained")
    for i, task in enumerate(config["tasks"]):
        for j, target in enumerate(config["targets"]):
            ax = axes[i, j]
            for k, rank in enumerate(config["task_vector_ranks"]):
                selected = {
                    int(row["site"].split(":")[1]): row
                    for row in rows
                    if row["task"] == task
                    and row["moment"] == "uncentered"
                    and row["task_vector_rank"] == rank
                    and row["site"].startswith(target + ":")
                }
                layers = list(range(config["layers"]))
                values = [
                    100 * selected[layer]["activation_energy_outside_proxy_fraction"]
                    for layer in layers
                ]
                ax.plot(
                    layers,
                    values,
                    marker="o",
                    markersize=3,
                    color=COLORS[k],
                    label=f"Task-vector rank {rank}",
                )
            ax.set(
                xlabel="Transformer block (0-based)",
                ylabel="Input energy outside proxy (%)",
                ylim=(0, 100),
                xticks=[0, 3, 6, 9, 11],
                title=f"{task}: {'attention V' if target == 'V' else 'MLP FC'} input",
            )
            ax.grid(alpha=0.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.suptitle("Uncentered moments; 256 frozen training images per task, all tokens", fontsize=10)
    save(fig, out, "task_vector_coverage")
    print("NUFILT_FIGURE_WRITTEN", out)


def plot_reconstruction_budget(root: Path):
    rows = json.loads((root / "results.json").read_bytes())
    fitting = json.loads((root / "fitting.json").read_bytes())
    config = json.loads((root / "protocol.json").read_bytes())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    presets = ["unprotected", *[f"fit_{n}" for n in config["budgets"]]]
    groups = [("standalone", "SVHN"), ("direct", "SVHN"), ("direct", "MNIST")]
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 2.9), layout="constrained")
    for ax, (group, task) in zip(axes, groups, strict=True):
        lookup = {r["preset"]: r for r in rows if r["group"] == group and r["dataset"] == task}
        values = [100 * lookup[preset]["top1"] for preset in presets]
        ax.bar(range(4), values, color=COLORS)
        for index, value in enumerate(values):
            ax.text(index, value + 0.7, f"{value:.3f}", ha="center", fontsize=8)
        ax.set(
            xticks=range(4),
            xticklabels=["Original", "32", "128", "256"],
            ylim=(0, 105),
            ylabel="Top-1 accuracy (%)",
            xlabel="Fitting images",
            title=f"{task} {'standalone' if group == 'standalone' else 'merged'} (n={lookup[presets[0]]['n']:,})",
        )
        ax.grid(axis="y", alpha=0.2)
    fig.suptitle("Unprotected reconstruction control; reused training subsets", fontsize=10)
    save(fig, out, "budget_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), layout="constrained")
    lookup = {(r["budget"], r["layer"]): r for r in fitting}
    for ax, metric, title in zip(
        axes,
        ["relative_full_output_error", "relative_task_update_error"],
        ["Full MLP output", "Task-increment output"],
        strict=True,
    ):
        for index, layer in enumerate(config["layers"]):
            values = [100 * lookup[n, layer]["check"][metric] for n in config["budgets"]]
            ax.plot(
                config["budgets"], values, marker="o", color=COLORS[index], label=f"Block {layer}"
            )
        ax.set(
            xscale="log", xlabel="Fitting images", ylabel="Relative check error (%)", title=title
        )
        ax.set_xticks(config["budgets"], labels=config["budgets"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=4)
    fig.suptitle("32 disjoint training check images; frozen released-expert inputs", fontsize=10)
    save(fig, out, "budget_local_error")
    print("BUDGET_FIGURES_WRITTEN", out)


def plot_reconstruction_context(root: Path):
    rows = json.loads((root / "results.json").read_bytes())
    config = json.loads((root / "protocol.json").read_bytes())
    local = json.loads((root / "local_errors.json").read_bytes())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    names = ["unprotected", *config["conditions"]]
    display = {
        "unprotected": "Original",
        "public_expert": "Public\nexpert",
        "public_merged": "Public\nmerged",
        "released_expert": "Released\nexpert",
        "released_merged": "Released\nmerged",
        "public_selfmerged": "Public\nself-merged",
    }
    labels = [display[name] for name in names]
    groups = [("standalone", "SVHN"), ("direct", "SVHN"), ("direct", "MNIST")]
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.0), layout="constrained")
    for ax, (group, task) in zip(axes, groups):
        selected = {r["preset"]: r for r in rows if r["group"] == group and r["dataset"] == task}
        values = [100 * selected[name]["top1"] for name in names]
        ax.bar(range(len(names)), values, color=["#777777", *COLORS[: len(names) - 1]], width=0.7)
        for x, value in enumerate(values):
            ax.text(x, value + 1, f"{value:.3f}", ha="center", fontsize=7)
        ax.set(
            xticks=range(len(names)),
            xticklabels=labels,
            ylim=(0, 105),
            ylabel="Accuracy (%)",
            title=f"{task}: {'standalone' if group == 'standalone' else 'merged'}",
        )
        ax.tick_params(axis="x", labelsize=7)
        ax.grid(axis="y", alpha=0.2)
    headline = "Unprotected control; previously inspected training subsets"
    if "public_selfmerged" in config["conditions"]:
        headline += "\nSecond pass: same 256 images, twice the total fitting-target token budget"
    fig.suptitle(headline, fontsize=9)
    save(fig, out, "context_accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1), layout="constrained")
    for ax, operator in zip(axes, ["expert_operator", "merged_operator"]):
        for i, preset in enumerate(config["conditions"]):
            selected = {
                r["layer"]: r
                for r in local
                if r["preset"] == preset
                and r["dataset"] == "SVHN"
                and r["input_context"] == "merged"
            }
            values = [
                100 * selected[layer][operator]["relative_full_output_error"]
                for layer in config["layers"]
            ]
            ax.plot(
                config["layers"],
                values,
                marker="o",
                markersize=3,
                color=COLORS[i],
                label=preset.replace("_", " / "),
            )
        ax.set(
            yscale="linear" if len(config["conditions"]) == 2 else "log",
            xlabel="Transformer block (0-based)",
            ylabel="Output error (%)",
            xticks=config["layers"],
            title=operator.replace("_", " ").capitalize(),
        )
        if len(config["conditions"]) == 2:
            ax.set_ylim(bottom=0)
        ax.grid(alpha=0.2)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.suptitle("32 disjoint SVHN check images; frozen original-merged inputs", fontsize=9)
    save(fig, out, "context_local_error")
    print("CONTEXT_FIGURES_WRITTEN", out)


def plot_nufilt_response(root: Path):
    result = json.loads((root / "results.json").read_bytes())
    rows, config = result["rows"], result["protocol"]
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), layout="constrained")
    for i, task in enumerate(config["tasks"]):
        for j, target in enumerate(config["targets"]):
            ax = axes[i, j]
            for k, rank in enumerate(config["task_vector_ranks"]):
                selected = {
                    int(r["site"].split(":")[1]): r
                    for r in rows
                    if r["task"] == task
                    and r["moment"] == "uncentered"
                    and r["task_vector_rank"] == rank
                    and r["site"].startswith(target + ":")
                }
                layers = list(range(config["layers"]))
                values = [
                    100 * selected[layer]["task_response_removed_fraction"] for layer in layers
                ]
                ax.plot(
                    layers,
                    values,
                    marker="o",
                    markersize=3,
                    color=COLORS[k],
                    label=f"Task-vector rank {rank}",
                )
            ax.set(
                xlabel="Transformer block (0-based)",
                ylabel="Response energy (%)",
                ylim=(0, 1.0),
                xticks=[0, 3, 6, 9, 11],
                title=f"{task}: {'attention V' if target == 'V' else 'MLP FC'} increment",
            )
            ax.grid(alpha=0.2)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    fig.suptitle(
        "Linear task-response energy outside proxy\nCached expert inputs; bias/downstream effects excluded",
        fontsize=9,
    )
    save(fig, out, "task_response_energy")
    print("RESPONSE_FIGURE_WRITTEN", out)


def plot_subspace_interference(root: Path):
    """Plot measured utility and merging with matched unprotected attack controls."""
    out = root / "figures"
    out.mkdir(exist_ok=True)
    config = json.loads((root / "protocol.json").read_bytes())
    rows = json.loads((root / "results.json").read_bytes())
    keyed = {(r["preset"], r["group"], r["dataset"]): 100 * r["top1"] for r in rows}
    attacked = any(r["group"] in {"angular", "aga_all"} for r in rows)
    fig, axes = plt.subplots(1, 3, figsize=(12 if attacked else 9, 3.25), layout="constrained")
    protected_task, partner_task = config["protected_task"], config["other_task"]
    metrics = [
        ("standalone", protected_task, f"Standalone {protected_task}"),
        ("direct", protected_task, f"Merged {protected_task}"),
        ("direct", partner_task, f"Merged {partner_task}"),
    ]
    if attacked:
        name = config["presets"][0]["name"]
        groups = [
            "direct",
            "angular",
            "pretrained_fc",
            "spectral",
            "calibrated",
            "aga",
            "reconstruction_1",
            "reconstruction_2",
        ]
        labels = [
            "Direct",
            "Anchor\nscan",
            "FC\nreset",
            "Rank\nremoval",
            "Input\nfilter",
            "AGA",
            "Fit 1",
            "Fit 2",
        ]
        if config.get("aga_all", False):
            groups.insert(6, "aga_all")
            labels.insert(6, "AGA\nboth")
        available = {row["group"] for row in rows}
        retained = [(g, label) for g, label in zip(groups, labels, strict=True) if g in available]
        groups, labels = zip(*retained, strict=True)
        for ax, (metric, task, title) in zip(axes, metrics, strict=True):
            for j, preset in enumerate(["unprotected", name]):
                values = [
                    keyed[
                        preset,
                        (
                            "standalone"
                            if (preset, "standalone", task) in keyed and g == "direct"
                            else g + "_standalone"
                        )
                        if metric == "standalone"
                        else g,
                        task,
                    ]
                    for g in groups
                ]
                x = np.arange(len(groups)) + (j - 0.5) * 0.36
                ax.bar(
                    x,
                    values,
                    width=0.34,
                    color=COLORS[j],
                    label="Unprotected" if j == 0 else "Modified",
                )
                for pos, value in zip(x, values, strict=True):
                    ax.text(
                        pos,
                        value + 1,
                        f"{value:.1f}",
                        ha="center",
                        va="bottom",
                        fontsize=5.5,
                        rotation=90,
                    )
            ax.set(
                title=title,
                xticks=np.arange(len(groups)),
                xticklabels=labels,
                ylim=(0, 112),
                ylabel="Accuracy (%)",
            )
            ax.tick_params(axis="x", labelsize=7)
            ax.grid(axis="y", alpha=0.2)
        fig.legend(*axes[0].get_legend_handles_labels(), loc="outside lower center", ncol=2)
    else:
        presets = ["unprotected", *[p["name"] for p in config["presets"]]]
        for ax, (group, task, title) in zip(axes, metrics, strict=True):
            values = [keyed[p, group, task] for p in presets]
            ax.bar(
                np.arange(len(presets)),
                values,
                color=[COLORS[0]] + [COLORS[1]] * (len(presets) - 1),
            )
            for pos, value in enumerate(values):
                ax.text(pos, value + 1, f"{value:.1f}", ha="center", va="bottom", fontsize=7)
            ax.set(
                title=title,
                xticks=np.arange(len(presets)),
                xticklabels=presets,
                ylim=(0, 110),
                ylabel="Accuracy (%)",
            )
            ax.tick_params(axis="x", labelsize=7, rotation=20)
            ax.grid(axis="y", alpha=0.2)
    scope = (
        "full official test sets, frozen candidate"
        if config.get("test_usage") == "fixed_full_test_evaluation"
        else "inspected training subsets, one seed"
    )
    fig.suptitle(
        f"Complement perturbation and recovery: {scope}\n"
        "TA coefficient 0.5; calibrated output compensation"
        if config.get("output_compensation", False)
        else "Complement mechanism check: inspected training subsets, one seed; TA coefficient 0.5",
        fontsize=9,
    )
    save(fig, out, "interference_accuracy")
    print("INTERFERENCE_FIGURES_WRITTEN", out)


def plot_fine_subspace(root: Path):
    """Show confirmed contrast separately from achievable replacement floors."""
    directions = json.loads((root / "directions.json").read_bytes())
    rows = json.loads((root / "results.json").read_bytes())
    config = json.loads((root / "protocol.json").read_bytes())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), layout="constrained")
    positions = np.arange(len(directions))
    for index, task in enumerate(["MNIST", "DTD"]):
        axes[0].bar(
            positions + (index - 0.5) * 0.36,
            [r["confirmation"]["rms_ratios"][task] for r in directions],
            width=0.36,
            color=COLORS[index],
            label=task,
        )
    axes[0].axhline(config["minimum_confirmed_rms_ratio"], color="0.5", ls="--", lw=0.8)
    axes[0].set(
        title="Fresh-image local response contrast",
        ylabel="RMS response / Cars RMS (log scale)",
        yscale="log",
        ylim=(1, 20),
        xticks=positions,
        xticklabels=[r["site"] + ("*" if not r["contrast_passed"] else "") for r in directions],
    )
    axes[0].tick_params(axis="x", rotation=55, labelsize=7)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.15)
    lookup = {(r["preset"], r["group"]): r for r in rows}
    presets = ["qkv", "fc", "down", "eligible_union"]
    for index, group in enumerate(["standalone", "fusion"]):
        losses = [lookup[preset, group]["paired"]["loss_pp"] for preset in presets]
        axes[1].bar(
            np.arange(4) + (index - 0.5) * 0.36,
            losses,
            width=0.36,
            color=COLORS[index],
            label="Standalone" if group == "standalone" else "Eight-expert fusion",
        )
    axes[1].axhline(config["minimum_recovery_gap_pp"], color="0.5", ls="--", lw=0.8)
    axes[1].set(
        title="Public parameter replacement cost",
        ylabel="Accuracy loss (pp)",
        xticks=np.arange(4),
        xticklabels=["QKV", "MLP up", "8 neurons/layer", "Eligible union"],
        ylim=(0, 6),
    )
    axes[1].tick_params(axis="x", rotation=20, labelsize=7)
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Cars preflight; * rejected stability/contrast; replacement on inspected training data\n"
        "No protected model constructed; no AGA or official-test defense result",
        fontsize=9,
    )
    save(fig, out, "fine_subspace_preflight")


def plot_subspace_recovery(root: Path):
    """Keep oracle information and public-covariance diagnostics distinct."""
    rows = json.loads((root / "results.json").read_bytes())
    geometry = json.loads((root / "geometry.json").read_bytes())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3), layout="constrained")
    lookup = {(r["preset"], r["group"]): r for r in rows}
    presets = ["direction_oracle", "public_covariance"]
    for index, group in enumerate(["standalone", "fusion"]):
        bars = axes[0].bar(
            np.arange(2) + (index - 0.5) * 0.36,
            [lookup[preset, group]["paired"]["loss_pp"] for preset in presets],
            width=0.36,
            color=COLORS[index],
            label="Standalone" if group == "standalone" else "Eight-expert fusion",
        )
        axes[0].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].set(
        title="Projection replacement cost",
        ylabel="Accuracy loss (pp)",
        ylim=(-0.05, 1),
        xticks=np.arange(2),
        xticklabels=["Exact direction (oracle)", "Public covariance tail"],
    )
    axes[0].tick_params(axis="x", labelsize=8, rotation=15)
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.15)
    axes[1].bar(
        np.arange(len(geometry)),
        [100 * r["public_noise_energy_removed_fraction"] for r in geometry],
        color=COLORS[2],
    )
    axes[1].set(
        title="Public tail coverage of fixed directions",
        ylabel="Input-direction energy removed (%)",
        xticks=np.arange(len(geometry)),
        xticklabels=[r["site"] for r in geometry],
        ylim=(0, 105),
    )
    axes[1].tick_params(axis="x", rotation=55, labelsize=7)
    axes[1].grid(axis="y", alpha=0.15)
    fig.suptitle(
        "Unprotected Cars expert; inspected training confirmation; TA coefficient 0.2\n"
        "Protected-release recovery remains untested",
        fontsize=9,
    )
    save(fig, out, "subspace_recovery_cost")


def plot_functional_contrast(root: Path):
    """Display both frozen points, including bypasses and cross-task effects."""
    rows = json.loads((root / "results.json").read_bytes())
    decision = json.loads((root / "decision.json").read_bytes())
    config = json.loads((root / "protocol.json").read_bytes())
    out = root / "figures"
    out.mkdir(exist_ok=True)
    lookup = {r["id"]: r for r in rows}
    task = config["task"]
    presets = ["original", "budget_0", "budget_1"]
    utility = [0] + [p["utility_loss_pp"] for p in decision["points"]]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3), layout="constrained")
    for index, attack in enumerate(["direct", "aga", "best"]):
        values = []
        for preset in presets:
            if attack == "best":
                accuracy = max(
                    r["top1"]
                    for r in rows
                    if r["preset"] == preset and r["group"] == "fusion" and r["dataset"] == task
                )
                baseline = max(
                    r["top1"]
                    for r in rows
                    if r["preset"] == "original" and r["group"] == "fusion" and r["dataset"] == task
                )
            else:
                accuracy = lookup[f"{preset}/{attack}/fusion/{task}"]["top1"]
                baseline = lookup[f"original/{attack}/fusion/{task}"]["top1"]
            values.append(100 * (baseline - accuracy))
        order = np.argsort(utility)
        axes[0].plot(
            np.asarray(utility)[order],
            np.asarray(values)[order],
            marker=["o", "s", "^"][index],
            color=COLORS[index],
            label={
                "direct": "Direct merging",
                "aga": "Original AGA",
                "best": "Best tested recovery",
            }[attack],
        )
    axes[0].axvline(config["utility_reference_pp"], color="0.5", ls="--", lw=0.8)
    axes[0].axhline(0, color="0.5", lw=0.8)
    axes[0].axhline(config["minimum_interesting_gap_pp"], color="0.5", ls=":", lw=0.8)
    axes[0].set(
        xlabel="Standalone accuracy loss (pp)",
        ylabel="Fusion loss vs matched control (pp)",
        title="Performance and recovery tradeoff",
        ylim=(-0.5, 6),
    )
    axes[0].grid(alpha=0.15)
    axes[0].legend()
    positions = np.arange(len(config["evaluation_tasks"]))
    for index, preset in enumerate(presets):
        values = [
            100 * lookup[f"{preset}/direct/fusion/{dataset}"]["top1"]
            for dataset in config["evaluation_tasks"]
        ]
        axes[1].bar(
            positions + (index - 1) * 0.25,
            values,
            width=0.25,
            color=COLORS[index],
            label={
                "original": "Unprotected",
                "budget_0": "RMS budget 0.01",
                "budget_1": "RMS budget 0.04",
            }[preset],
        )
    axes[1].set(
        xticks=positions,
        xticklabels=config["evaluation_tasks"],
        ylim=(0, 105),
        ylabel="Accuracy (%)",
        title="Direct eight-expert fusion",
    )
    axes[1].grid(axis="y", alpha=0.15)
    axes[1].legend()
    fig.suptitle(
        "Cars-only construction; three full official test sets; TA coefficient 0.2", fontsize=9
    )
    save(fig, out, "functional_contrast")


def plot_local_erasure(root: Path):
    """Matched controls on full tests; no seed error bars for one release point."""
    config = json.loads((root / "protocol.json").read_bytes())
    rows = {r["id"]: r for r in json.loads((root / "results.json").read_bytes())}
    assert json.loads((root / "validation.json").read_bytes())["status"] == "passed"
    out = root / "figures"
    out.mkdir(exist_ok=True)
    labels = ["Direct", "AGA", "Reset A", "Reset M", "Rank-1", "Target=1", "Readout"]
    x = np.arange(len(labels))
    fig, axes = plt.subplots(2, 3, figsize=(10, 5), layout="constrained", sharey=True)
    for i, target in enumerate(config["tasks"]):
        original = rows[f"{target}/original/direct/standalone/{target}"]["top1"] * 100
        erased = rows[f"{target}/erased/direct/standalone/{target}"]["top1"] * 100
        for j, dataset in enumerate(config["evaluation_tasks"]):
            ax = axes[i, j]
            for index, preset in enumerate(("original", "erased")):
                values = [
                    100 * rows[f"{target}/{preset}/{a}/fusion/{dataset}"]["top1"]
                    for a in config["attacks"]
                ]
                ax.plot(
                    x,
                    values,
                    marker="o" if index == 0 else "s",
                    ms=3.2,
                    lw=1.2,
                    color=COLORS[index],
                    label="Unprotected control" if index == 0 else "Local deletion",
                )
            ax.set(
                ylim=(0, 102),
                xticks=x,
                xticklabels=labels,
                title=f"Release: {target}; test: {dataset}",
            )
            ax.tick_params(axis="x", rotation=45, labelsize=7)
            ax.grid(axis="y", alpha=0.2)
            if j == 0:
                ax.set_ylabel("Merged accuracy (%)")
                ax.text(
                    0.02,
                    0.06,
                    f"Standalone {original:.2f}% → {erased:.2f}%",
                    transform=ax.transAxes,
                    fontsize=8,
                )
    axes[0, 0].legend(loc="lower left", bbox_to_anchor=(0, 0.12), fontsize=7)
    fig.suptitle("Eight experts; full official tests; fixed matched attacks", fontsize=11)
    fig.supxlabel(
        "Target=1: target weight 1, other weights 0.2; Readout: 256 teacher queries. Other attacks: TA 0.2.",
        fontsize=8,
    )
    save(fig, out, "local_erasure")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/2026-10-09")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--task-support", action="store_true")
    mode.add_argument("--functional-contrast", action="store_true")
    mode.add_argument("--fine-subspace", action="store_true")
    mode.add_argument("--local-erasure", action="store_true")
    mode.add_argument("--subspace-recovery", action="store_true")
    mode.add_argument(
        "--subspace-interference",
        action="store_true",
        help="Plot complement perturbation and recovery.",
    )
    mode.add_argument("--subspace", action="store_true", help="Plot the standalone subspace probe.")
    mode.add_argument("--subspace-diagnostics", action="store_true", help="Plot causal contrasts.")
    mode.add_argument("--redundancy-erasure", action="store_true")
    mode.add_argument("--joint-transfer", action="store_true")
    mode.add_argument("--joint-diagnostics", action="store_true")
    mode.add_argument("--response-transfer", action="store_true")
    mode.add_argument("--eight-protection", action="store_true")
    mode.add_argument("--bounded-complement", action="store_true", help="Plot bounded edits.")
    mode.add_argument("--distribution-shift", action="store_true", help="Plot input diagnostics.")
    mode.add_argument(
        "--reconstruction-context",
        action="store_true",
        help="Plot fitting feature/context controls.",
    )
    mode.add_argument(
        "--nufilt-alignment", action="store_true", help="Plot task-vector proxy coverage."
    )
    mode.add_argument(
        "--nufilt-response", action="store_true", help="Plot task-increment responses."
    )
    mode.add_argument(
        "--reconstruction-budget", action="store_true", help="Plot fixed reconstruction budgets."
    )
    mode.add_argument(
        "--operator-diagnostics",
        action="store_true",
        help="Plot component and reconstruction controls.",
    )
    args = parser.parse_args()
    runner = (
        plot_local_erasure
        if args.local_erasure
        else plot_fine_subspace
        if args.fine_subspace
        else plot_subspace_recovery
        if args.subspace_recovery
        else plot_functional_contrast
        if args.functional_contrast
        else plot_task_support
        if args.task_support
        else plot_response_transfer
        if args.response_transfer
        else plot_subspace_interference
        if args.subspace_interference
        else plot_reconstruction_context
        if args.reconstruction_context
        else plot_reconstruction_budget
        if args.reconstruction_budget
        else plot_nufilt_response
        if args.nufilt_response
        else plot_nufilt_alignment
        if args.nufilt_alignment
        else plot_operator_diagnostics
        if args.operator_diagnostics
        else plot_redundancy_erasure
        if args.redundancy_erasure
        else plot_joint_transfer
        if args.joint_transfer
        else plot_joint_diagnostics
        if args.joint_diagnostics
        else plot_eight_protection
        if args.eight_protection
        else plot_distribution_shift
        if args.distribution_shift
        else plot_bounded_complement
        if args.bounded_complement
        else plot_subspace_diagnostics
        if args.subspace_diagnostics
        else plot_subspace
        if args.subspace
        else run
    )
    runner(Path(args.output))
