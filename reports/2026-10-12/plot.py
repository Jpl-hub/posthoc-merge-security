"""Plot the selected frozen MNIST experiment; no model execution."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    root = Path(__file__).resolve().parents[2]
    out = Path(__file__).resolve().parent / "figures"
    out.mkdir(exist_ok=True)
    audit_path = root / "results/mnist/evaluation/results.json"
    labels_path = root / "results/mnist/hard_label_recovery/results.json"
    audit = {r["key"]: r for r in json.loads(audit_path.read_text())}
    labels = {r["key"]: r for r in json.loads(labels_path.read_text())}
    settings = [
        ("Standalone", "direct", "standalone", audit),
        ("Direct\nmerge", "direct", "fusion", audit),
        ("Author\nAGA", "aga", "fusion", audit),
        ("MLP\nreplacement", "mlp_reset", "fusion", audit),
        ("Full local\nreplacement", "support_reset", "fusion", audit),
        ("Replacement\n+ 256 labels", "support_reset_prototype", "fusion", labels),
        ("Replacement\n+ 256 scores", "support_reset_query", "fusion", audit),
    ]
    rows = []
    for caption, attack, group, source in settings:
        control = source[f"original/{attack}/{group}/MNIST"]
        protected = source[f"protected/{attack}/{group}/MNIST"]
        assert control["n"] == protected["n"] == 10000
        rows.append({
            "label": caption,
            "control_key": control["key"],
            "protected_key": protected["key"],
            "control_correct": control["correct"],
            "protected_correct": protected["correct"],
            "n": control["n"],
            "control_percent": 100 * control["correct"] / control["n"],
            "protected_percent": 100 * protected["correct"] / protected["n"],
        })
    provenance = {
        "data": rows,
        "sources": {
            str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [audit_path, labels_path]
        },
        "fusion": "Eight experts, task arithmetic coefficient 0.2",
        "scope": "MNIST complete official test, one fixed candidate. No seed error bars.",
    }
    (out / "data.json").write_text(json.dumps(provenance, indent=2) + "\n")
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "svg.fonttype": "none", "axes.spines.top": False,
                         "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(10.6, 4.2), layout="constrained")
    x = np.arange(len(rows))
    for offset, field, color, caption in [
        (-0.19, "control_percent", "#65758B", "Original expert (matched control)"),
        (0.19, "protected_percent", "#A83F39", "Released expert"),
    ]:
        bars = ax.bar(x + offset, [r[field] for r in rows], width=0.36,
                      color=color, label=caption)
        ax.bar_label(bars, fmt="%.2f", padding=3, fontsize=8)
    ax.set_xticks(x, [r["label"] for r in rows])
    ax.set_ylabel("MNIST accuracy (%)")
    ax.set_ylim(0, 112)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.grid(axis="y", alpha=0.18)
    ax.set_axisbelow(True)
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), frameon=False, ncols=2)
    fig.suptitle("Local posthoc recoding: AGA degradation and measured recovery", fontsize=12)
    for suffix in ["png", "svg"]:
        fig.savefig(out / f"mnist_results.{suffix}", dpi=190, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
