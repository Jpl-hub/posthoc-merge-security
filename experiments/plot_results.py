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
    axes[0].legend(loc="upper left")
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
    )
    ax.grid(axis="y", alpha=0.2)
    save(fig, out, "head_permutation")
    if (root / "clip_results.json").exists():
        real = json.loads((root / "clip_results.json").read_text())
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
            ax.bar(
                np.arange(4) + (i - (len(strengths) - 1) / 2) * 0.35,
                y,
                width=0.35,
                label=f"Null strength {strength:g}",
                color=COLORS[i],
            )
        ax.set(
            xticks=range(4),
            xticklabels=labels,
            ylim=(0, 105),
            ylabel="MNIST test accuracy (%)",
            title="Post-hoc LayerNorm-null exploration (real model, full MNIST test)",
        )
        ax.grid(axis="y", alpha=0.2)
        ax.legend(loc="upper left", fontsize=7)
        save(fig, out, "layernorm_exploration")
    print("FIGURES_WRITTEN", out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/2026-10-09")
    run(Path(parser.parse_args().output))
