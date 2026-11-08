"""Plot frozen NBP results with the same data used by the table exporter."""

import hashlib
import csv
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

    control, protected, baseline = "#4D4D4D", "#B95335", "#0072B2"
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Nimbus Roman"],
                         "mathtext.fontset": "stix", "font.size": 8.5,
                         "axes.titlesize": 9, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
                         "legend.fontsize": 8.5,
                         "pdf.fonttype": 42, "ps.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False})
    ablation_root = ROOT / "results/phase3/bias_weight_only/ablation"
    ablation = json.loads((ablation_root / "results.json").read_text())
    validation = json.loads((ablation_root / "independent_validation.json").read_text())
    assert validation["status"] == "passed" and validation["rows"] == len(ablation) == 40
    scores = {(r["defense"], r["merger"], r["attack_scope"], r["dataset"]): r for r in ablation}
    amplitudes = [.125, .25, .5, 1., 2.]
    names = ["Scale_0.125", "Scale_0.25", "Scale_0.5", "NBP", "Scale_2"]
    fig, ax = plt.subplots(figsize=(3.25, 2.45))
    fig.subplots_adjust(left=.17, right=.98, bottom=.23, top=.75)
    for merger, scope, label, color, marker, style, size, face in [
        ("Standalone", "None", "Standalone", control, "s", "-", 3.5, control),
        ("TA", "None", "TA", protected, "o", "-", 6, "white"),
        ("TA", "Protected expert", "AGA + TA", protected, "^", "--", 4, protected),
    ]:
        accuracy = [100 * scores[n, merger, scope, "MNIST"]["correct"]
                    / scores[n, merger, scope, "MNIST"]["n"] for n in names]
        ax.plot(amplitudes, accuracy, label=label, color=color, marker=marker,
                linestyle=style, linewidth=1, markersize=size, markerfacecolor=face)
    ax.axvline(.5, color="#777777", linewidth=.8, linestyle=":", label="Proxy threshold", zorder=0)
    ax.set_xscale("log", base=2)
    ax.set(xticks=amplitudes, xticklabels=["0.125", "0.25", "0.5", "1", "2"],
           xlabel=r"NBP magnitude multiplier (log$_2$ scale)",
           ylabel="MNIST accuracy (%)", ylim=(0, 105), yticks=[0, 25, 50, 75, 100])
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=2,
               loc="upper center", bbox_to_anchor=(.55, 1), columnspacing=1.2)
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
    ta = {(r[0].split("/")[0], r[1]): float(r[4].split(" / ")[0])
          for r in ta_rows if r[1] in ["Unprotected", "PaRaMS", "Defense 1"]}
    assert len(ta) == 21
    fusion_root = ROOT / "results/phase3/bias_weight_only/fusion_methods"
    fusion = []
    for folder in [fusion_root] + [fusion_root / "extension" / task for task in tasks if task != "SUN397"]:
        fusion.extend(json.loads((folder / "results.json").read_text()))
    fused = {(r["target"], r["merger"], r["defense"]): 100*r["correct"]/r["n"]
             for r in fusion if r["dataset"] == r["target"] and r["attack_scope"] == "Protected expert"}
    fig, axes = plt.subplots(3, 1, figsize=(3.25, 3.4), sharex=True, sharey=True)
    fig.subplots_adjust(left=.16, right=.98, bottom=.18, top=.88, hspace=.18)
    x = np.arange(len(tasks))
    for ax, merger in zip(axes, ["TA", "TIES", "DARE+TA"]):
        for offset, method, label, color, marker, face in [
            (-.17, "Unprotected", "Unprotected", control, "o", "white"),
            (0, "PaRaMS", "PaRaMS", baseline, "s", "white"),
            (.17, "Defense 1", "NBP", protected, "^", protected),
        ]:
            values = [ta[t, method] if merger == "TA" else
                      fused[t, merger, "Defense1_"+t if method == "Defense 1" else
                            "PaRaMS_"+t if method == "PaRaMS" else method] for t in tasks]
            ax.scatter(x+offset, values, marker=marker, s=22, color=color,
                       facecolors=face, linewidths=.8, label=label, zorder=3)
            if method == "Defense 1" and merger in ["TIES", "DARE+TA"]:
                task = "SVHN" if merger == "TIES" else "MNIST"
                i = tasks.index(task)
                ax.annotate(f"{values[i]:.2f}", (x[i]+offset, values[i]),
                            xytext=(0, 7), textcoords="offset points", ha="center", color=protected)
        ax.set(ylim=(-5, 112), yticks=[0, 50, 100], xlim=(-.55, len(tasks)-.45))
        ax.text(.015, .45, merger, transform=ax.transAxes, fontsize=9)
        ax.grid(axis="y", color="#eeeeee", linewidth=.6, zorder=0)
        ax.tick_params(axis="x", length=0)
    axes[-1].set_xticks(x, tasks, rotation=45, ha="right")
    fig.legend(*axes[0].get_legend_handles_labels(), frameon=False, fontsize=9,
               ncol=3, loc="upper center", bbox_to_anchor=(.53,1), columnspacing=.8, handletextpad=.3)
    fig.supylabel("Target accuracy after AGA + fusion (%)", x=.005, fontsize=8.5)
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
    fig, ax = plt.subplots(figsize=(3.25, 2.65), layout="constrained")
    bounds = [r["minimum_margin_lower_bound"] for r in rows]
    ax.axvspan(0, max(bounds)*1.15, color="#777777", alpha=.06, zorder=0)
    for method, label, color, marker in [("None", "Unprotected", control, "o"),
            ("NBP", "NBP", protected, "^")]:
        selected = [r for r in rows if r["method"] == method]
        ax.scatter([r["minimum_margin_lower_bound"] for r in selected],
                   [100*r["dominant_fraction"] for r in selected],
                   edgecolors=color, facecolors="white" if method == "None" else color,
                   marker=marker, s=22, label=label, alpha=.85, linewidths=.7)
    ax.axvline(0, color="black", linewidth=.8, linestyle="--")
    ax.set_xscale("symlog", linthresh=50)
    ax.set(xlabel="Minimum margin lower bound (symlog)", ylabel="Dominant-class predictions (%)",
           xlim=(min(bounds)*1.25,max(bounds)*1.4), ylim=(-2, 112), yticks=[0,25,50,75,100],
           xticks=[-1000,-100,0,100,1000], xticklabels=["−1000","−100","0","100","1000"])
    own = [r for r in rows if r["method"] == "NBP"]
    certified = sum(r["minimum_margin_lower_bound"] > 0 for r in own)
    assert certified == 30 and len(own) == 35
    ax.text(.03,.83,"Bound ≤ 0:\ninconclusive",transform=ax.transAxes,ha="left",va="top",multialignment="left")
    ax.text(.97,.83,"Bound > 0:\n30 / 35 NBP pairs",transform=ax.transAxes,ha="right",va="top",multialignment="right")
    weakest = min(own, key=lambda r:r["dominant_fraction"])
    ax.annotate(f"{weakest['target']} / {weakest['partner']}\n{100*weakest['dominant_fraction']:.2f}%",
                (weakest["minimum_margin_lower_bound"], 100*weakest["dominant_fraction"]),
                xytext=(5,-27), textcoords="offset points", fontsize=8.5,
                arrowprops=dict(arrowstyle="-",color="#777777",lw=.6))
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(OUT / "readout_predictions.pdf", metadata={"Title": "Fixed-readout bound and observed class concentration"})
    plt.close(fig)
    print("Wrote observed prediction concentration for all35 additional pairs; no image forwards.")

    # All grid points are calibration measurements; never present them as test curves.
    selection_root = ROOT / "results/phase3/bias_weight_only/coefficient_selection"
    selection_check = json.loads((selection_root / "selection_validation.json").read_text())
    assert selection_check["status"] == "passed" and selection_check["selection_rows"] == 18
    with (selection_root / "calibration_counts.csv").open(newline="") as stream:
        calibration = list(csv.DictReader(stream))
    assert len(calibration) == 270
    curve = {(r["target"], r["defense"], r["dataset"], float(r["coefficient"])):
             100*int(r["correct"])/int(r["n"])
             for r in calibration if r["merger"] == "TA"}
    assert len(curve) == 180 and all(int(r["n"]) == 256 for r in calibration)
    choices = json.loads((selection_root / "selected.json").read_text())
    grid = json.loads((selection_root / "selection_rule.json").read_text())["coefficient_grid"]
    fig, axes = plt.subplots(3, 1, figsize=(3.25, 4.35), sharex=True, sharey=True)
    fig.subplots_adjust(left=.17, right=.98, bottom=.12, top=.80, hspace=.43)
    for ax, task in zip(axes, ["MNIST", "SVHN", "DTD"]):
        partner = next(r["partner"] for r in calibration if r["target"] == task)
        for method, label, color, marker in [("Unprotected", "Unprotected", control, "o"),
                ("PaRaMS_"+task, "PaRaMS", baseline, "s"),
                ("NBP_"+task, "NBP", protected, "^")]:
            for dataset, style in [(task, "-"), (partner, "--")]:
                values = [curve[task, method, dataset, a] for a in grid]
                ax.plot(grid, values, color=color, linestyle=style, linewidth=.85,
                        marker=marker if dataset == task else None, markersize=3,
                        markerfacecolor="white", markeredgewidth=.65,
                        label=label if dataset == task else None)
            selected = next(r for r in choices if r["task"] == task
                            and r["defense"] == method and r["objective"] == "mean")
            ax.scatter(selected["coefficient"], selected["target_accuracy"],
                       marker="*", s=50, color=color, edgecolors="white", linewidths=.35, zorder=5)
        ax.set(ylim=(-5, 105), yticks=[0, 50, 100], xlim=(.075, 1.025))
        ax.set_title(task, loc="left", fontsize=9, pad=3)
        ax.grid(axis="y", color="#eeeeee", linewidth=.6)
    axes[-1].set(xticks=[.1, .3, .5, .7, 1.], xlabel=r"Shared TA coefficient $\lambda$")
    handles, labels = axes[0].get_legend_handles_labels()
    from matplotlib.lines import Line2D
    fig.legend(handles, labels, ncol=3, frameon=False, loc="upper center",
               bbox_to_anchor=(.54, 1), columnspacing=.7, handlelength=1.6, handletextpad=.35)
    styles = [Line2D([], [], color=control, label="Target", linewidth=.9),
                Line2D([], [], color=control, linestyle="--", label="Companion", linewidth=.9),
                Line2D([], [], color=control, marker="*", linestyle="None", label="Selected", markersize=7)]
    fig.legend(styles, ["Target", "Companion", "Selected"], ncol=3, frameon=False,
               loc="upper center", bbox_to_anchor=(.54, .94), columnspacing=.7,
               handlelength=1.6, handletextpad=.35)
    fig.supylabel("Calibration accuracy (%)", x=.005, fontsize=8.5)
    fig.savefig(OUT / "coefficient_calibration.pdf", metadata={"Title": "Complete labelled calibration curves; 256 images per task"})
    plt.close(fig)
    print("Wrote complete coefficient calibration curves from270 frozen rows; no new model evaluation.")
