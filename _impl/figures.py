"""Plot frozen NBP results with the same data used by the table exporter."""

import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def figures(output, table_sources):
    OUT = Path(output)
    OUT.mkdir(parents=True, exist_ok=True)
    record = ROOT / "results/phase3/bias_weight_only/readout_analysis.json"
    protocol = json.loads(record.with_name("readout_analysis_protocol.json").read_text())
    # The original source SHA is retained in the protocol; later CLI extensions do not invalidate frozen results.
    for path, sha in protocol["input_protocol_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
    data = json.loads(record.read_text())
    assert data["independent_numpy_formula_check"] and len(data["rows"]) == 42
    tasks = ["SUN397", "Cars", "RESISC45", "SVHN", "GTSRB", "MNIST", "DTD"]
    lookup = {(r["task"], r["scenario"], r["protected"]): r for r in data["rows"]}
    assert len(lookup) == 42
    for row in lookup.values():
        assert row["constant_prediction_certified_real_arithmetic"] == (row["minimum_margin_lower_bound"] > 0)

    plt.rcParams.update({"font.family": "serif", "font.size": 9,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    ablation_root = ROOT / "results/phase3/bias_weight_only/ablation"
    ablation = json.loads((ablation_root / "results.json").read_text())
    validation = json.loads((ablation_root / "independent_validation.json").read_text())
    assert validation["status"] == "passed" and validation["rows"] == len(ablation) == 40
    scores = {(r["defense"], r["merger"], r["attack_scope"], r["dataset"]): r for r in ablation}
    amplitudes = [.125, .25, .5, 1., 2.]
    names = ["Scale_0.125", "Scale_0.25", "Scale_0.5", "NBP", "Scale_2"]
    fig, ax = plt.subplots(figsize=(3.35, 2.25), layout="constrained")
    for merger, scope, label, color, marker, style in [
        ("Standalone", "None", "Standalone", "#333333", "s", "-"),
        ("TA", "None", "TA", "#25639c", "o", "-"),
        ("TA", "Protected expert", "AGA + TA", "#b55524", "^", "--"),
    ]:
        accuracy = [100 * scores[n, merger, scope, "MNIST"]["correct"]
                    / scores[n, merger, scope, "MNIST"]["n"] for n in names]
        ax.plot(amplitudes, accuracy, label=label, color=color, marker=marker,
                linestyle=style, linewidth=1, markersize=4)
    ax.axvline(.5, color="#999999", linewidth=.8, linestyle=":")
    ax.text(.5, .52, "Proxy bound", rotation=90, ha="right", va="center", fontsize=9,
            transform=ax.get_xaxis_transform())
    ax.set_xscale("log", base=2)
    ax.set(xticks=amplitudes, xticklabels=["0.125", "0.25", "0.5", "1", "2"],
           xlabel="Perturbation magnitude / NBP magnitude",
           ylabel="MNIST accuracy (%)", ylim=(0, 105), yticks=[0, 25, 50, 75, 100])
    ax.legend(frameon=False, fontsize=9, loc="center right")
    out = OUT / "amplitude.pdf"
    fig.savefig(out, bbox_inches="tight", metadata={"Creator": "Matplotlib", "Title": "Fixed NBP amplitude controls"})
    plt.close(fig)
    print("Wrote", out, "from verified fixed-amplitude controls.")

    # A single-column dot plot exposes taskwise differences hidden by macro-averages.
    manifest = json.loads((Path(table_sources)).read_text())
    for relative, expected in manifest["input_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    comparison = json.loads((ROOT / "results/comparison/sources.json").read_text())
    ta_rows = comparison["tables"]["paper_main"]["rows"] + comparison["tables"]["defense1_additional_pairs"]["rows"]
    ta = {(r[0].split("/")[0], r[1]): float(r[4].split(" / ")[0]) for r in ta_rows if r[1] in ["Unprotected", "Defense 1"]}
    assert len(ta) == 14
    fusion_root = ROOT / "results/phase3/bias_weight_only/fusion_methods"
    fusion = []
    for folder in [fusion_root] + [fusion_root / "extension" / task for task in tasks if task != "SUN397"]:
        fusion.extend(json.loads((folder / "results.json").read_text()))
    fused = {(r["target"], r["merger"], r["defense"]): 100*r["correct"]/r["n"]
             for r in fusion if r["dataset"] == r["target"] and r["attack_scope"] == "Protected expert"}
    fig, axes = plt.subplots(1, 3, figsize=(3.35, 2.8), sharey=True)
    fig.subplots_adjust(left=.22, right=.985, bottom=.23, top=.87, wspace=.12)
    y = np.arange(len(tasks))
    for ax, merger in zip(axes, ["TA", "TIES", "DARE+TA"]):
        own = [ta[t, "Defense 1"] if merger == "TA" else fused[t, merger, "Defense1_"+t] for t in tasks]
        clean = [ta[t, "Unprotected"] if merger == "TA" else fused[t, merger, "Unprotected"] for t in tasks]
        for i, (left, right) in enumerate(zip(own, clean)):
            ax.plot([left, right], [i]*2, color="#cccccc", linewidth=.8)
        ax.scatter(clean, y, marker="o", s=20, edgecolors="#596F83", facecolors="white", zorder=3, label="Unprotected")
        ax.scatter(own, y, marker="o", s=20, color="#B95335", zorder=3, label="NBP")
        ax.set(title=merger, yticks=y, yticklabels=tasks, xlim=(-7,107), xticks=[0,50,100], ylim=(len(tasks)-.5,-.5))
        ax.tick_params(axis="y", length=0)
        ax.grid(axis="x", color="#eeeeee", linewidth=.6, zorder=0)
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9,
               ncol=2, loc="lower center", bbox_to_anchor=(.55,-.015), columnspacing=1, handletextpad=.3)
    fig.supxlabel("Target accuracy after AGA + fusion (%)", x=.59, y=.105, fontsize=9)
    fig.savefig(OUT / "fusion_tasks.pdf", metadata={"Title": "Matched target accuracy after AGA and fusion"})
    plt.close(fig)
    print("Wrote paired fusion comparison from recorded complete-test results.")

    # Match all35 newly evaluated pairs to their fixed direct-TA readout certificate.
    diagnostic_path = ROOT / "results/phase3/bias_weight_only/partner_analysis/prediction_diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    for relative, expected in diagnostic["input_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    rows = [r for r in diagnostic["rows"] if r["attack_scope"] == "None"]
    assert len(rows) == 70
    fig, ax = plt.subplots(figsize=(3.35, 2.65), layout="constrained")
    bounds = [r["minimum_margin_lower_bound"] for r in rows]
    ax.axvspan(0, max(bounds)*1.08, color="#B95335", alpha=.055, zorder=0)
    for method, label, color, marker in [("None", "Unprotected", "#596F83", "o"),
            ("NBP", "NBP", "#B95335", "^")]:
        selected = [r for r in rows if r["method"] == method]
        ax.scatter([r["minimum_margin_lower_bound"] for r in selected],
                   [100*r["dominant_fraction"] for r in selected],
                   color=color, marker=marker, s=22, label=label, alpha=.8, linewidths=.5)
    ax.axvline(0, color="black", linewidth=.8, linestyle="--")
    ax.set(xlabel="Minimum margin lower bound", ylabel="Largest predicted class (%)",
           xlim=(min(bounds)*1.12,max(bounds)*1.08), ylim=(-2, 115), yticks=[0,25,50,75,100])
    ax.text(.02,.97,"Nonpositive:\ninconclusive",transform=ax.transAxes,ha="left",va="top",fontsize=8)
    ax.text(.98,.97,"Positive:\nconstant class",transform=ax.transAxes,ha="right",va="top",fontsize=8)
    ax.legend(frameon=False, fontsize=9, loc="lower right")
    fig.savefig(OUT / "readout_predictions.pdf", metadata={"Title": "Fixed-readout bound and observed class concentration"})
    plt.close(fig)
    print("Wrote observed prediction concentration for all35 additional pairs; no image forwards.")
