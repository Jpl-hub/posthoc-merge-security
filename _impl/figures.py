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

    # Per-task comparisons use the same complete-test results as the tables.
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
    def fusion_score(task, merger, method):
        if merger == "TA":
            return ta[task, method]
        name = "Defense1_"+task if method == "Defense 1" else "PaRaMS_"+task if method == "PaRaMS" else method
        return fused[task, merger, name]

    mergers = ["TA", "TIES", "DARE+TA"]
    # PaRaMS is retained in the tables; its recovered scores overlap the control.
    assert max(abs(fusion_score(t, m, "PaRaMS") - fusion_score(t, m, "Unprotected"))
               for t in tasks for m in mergers) < .06
    values = np.array([[fusion_score(t, m, method) for m in mergers
                        for method in ["Unprotected", "Defense 1"]] for t in tasks])
    fig, ax = plt.subplots(figsize=(3.4, 3.25))
    fig.subplots_adjust(left=.24, right=.99, bottom=.28, top=.81)
    cells = ax.pcolormesh(np.arange(7)-.5, np.arange(8)-.5, values,
                          cmap="Blues", vmin=0, vmax=100, shading="flat", rasterized=False)
    ax.set(xlim=(-.5, 5.5), ylim=(6.5, -.5))
    ax.set_xticks(np.arange(6), ["None", "NBP"]*3)
    for label in ax.get_xticklabels():
        if label.get_text() == "NBP":
            label.set_fontweight("bold")
    ax.set_yticks(np.arange(7), tasks)
    ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=4)
    for i, merger in enumerate(mergers):
        ax.text((2*i+1)/6, 1.18, merger, transform=ax.transAxes, ha="center", fontsize=9)
    for y in range(7):
        for x in range(6):
            ax.text(x, y, f"{values[y,x]:.1f}", ha="center", va="center",
                    color="white" if values[y,x] > 60 else "black", fontsize=8)
    ax.set_xticks(np.arange(-.5, 6, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 7, 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=.65)
    ax.tick_params(which="minor", length=0)
    for x in [1.5, 3.5]:
        ax.axvline(x, color="white", linewidth=3)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cax = fig.add_axes([.32, .12, .58, .035])
    bar = fig.colorbar(cells, cax=cax, orientation="horizontal", ticks=[0, 50, 100])
    bar.set_label("Target accuracy after AGA + fusion (%)", fontsize=8.5, labelpad=3)
    bar.outline.set_visible(False)
    bar.solids.set_rasterized(False)
    bar.ax.tick_params(length=2, labelsize=8)
    fig.savefig(OUT / "fusion_tasks.pdf", metadata={"Title": "Matched target accuracy after AGA and fusion"})
    plt.close(fig)
    print("Wrote annotated fusion comparison from recorded complete-test results.")

    # AGA edits preceding blocks only, leaving the certified readout unchanged.
    diagnostic_path = ROOT / "results/phase3/bias_weight_only/partner_analysis/prediction_diagnostics.json"
    diagnostic = json.loads(diagnostic_path.read_text())
    for relative, expected in diagnostic["input_sha256"].items():
        assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
    rows = [r for r in diagnostic["rows"] if r["attack_scope"] == "Protected expert"]
    assert len(rows) == 70
    own = [r for r in rows if r["method"] == "NBP"]
    groups = [
        [r for r in own if r["minimum_margin_lower_bound"] > 0],
        [r for r in own if r["minimum_margin_lower_bound"] <= 0],
        [r for r in rows if r["method"] == "None"],
    ]
    assert all(r["predicted_class_counts"][r["certificate_class"]] == r["n"] for r in groups[0])
    single = np.array([sum(r["dominant_fraction"] == 1 for r in group) for group in groups])
    multiple = np.array([len(group) for group in groups]) - single
    assert [len(g) for g in groups] == [30, 5, 35] and single.tolist() == [30, 3, 0]
    fig, ax = plt.subplots(figsize=(3.25, 2.15))
    fig.subplots_adjust(left=.39, right=.98, bottom=.24, top=.75)
    y = np.arange(3)
    ax.barh(y, single, height=.52, color=protected, label="One predicted class")
    ax.barh(y, multiple, left=single, height=.52, color="#D8D8D8", edgecolor="#777777",
            linewidth=.5, hatch="///", label="Multiple predicted classes")
    for i, (a, b) in enumerate(zip(single, multiple)):
        if a:
            ax.text(a/2, i, str(a), ha="center", va="center", color="white")
        if b:
            ax.text(a+b/2, i, str(b), ha="center", va="center", color="black")
    ax.set(yticks=y, yticklabels=["NBP: bound > 0", "NBP: bound ≤ 0", "Unprotected"],
           xlim=(0, 36), xticks=[0, 10, 20, 30], xlabel="Number of task pairs")
    ax.invert_yaxis()
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    fig.legend(*ax.get_legend_handles_labels(), frameon=False, ncol=1,
               loc="upper center", bbox_to_anchor=(.5, 1.02), handlelength=1.5)
    fig.savefig(OUT / "readout_predictions.pdf", metadata={"Title": "Readout condition and observed prediction counts"})
    plt.close(fig)
    print("Wrote observed prediction concentration for all35 additional pairs; AGA+TA; no image forwards.")

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
    fig, axes = plt.subplots(3, 1, figsize=(3.25, 4.15), sharex=True, sharey=True)
    fig.subplots_adjust(left=.17, right=.98, bottom=.12, top=.78, hspace=.43)
    for ax, task in zip(axes, ["MNIST", "SVHN", "DTD"]):
        partner = next(r["partner"] for r in calibration if r["target"] == task)
        for method, dataset, label, color, style, marker in [
                ("Unprotected", task, "Unprotected target", control, "-", "o"),
                ("NBP_"+task, task, "NBP target", protected, "-", "^"),
                ("NBP_"+task, partner, "NBP companion", baseline, "--", None)]:
            values = [curve[task, method, dataset, a] for a in grid]
            ax.plot(grid, values, color=color, linestyle=style, linewidth=1,
                    marker=marker, markersize=3, markerfacecolor="white",
                    markeredgewidth=.65, label=label)
        selected = next(r for r in choices if r["task"] == task
                        and r["defense"] == "NBP_"+task and r["objective"] == "mean")
        ax.scatter(selected["coefficient"], selected["target_accuracy"],
                   marker="*", s=65, color=protected, edgecolors="white", linewidths=.4, zorder=5)
        if task == "SVHN":
            ax.annotate(r"Selected $\lambda=0.7$", xy=(selected["coefficient"], selected["target_accuracy"]),
                        xytext=(.12, 38), fontsize=8, color=control,
                        arrowprops={"arrowstyle": "->", "color": "#777777", "linewidth": .6})
        ax.set(ylim=(-5, 105), yticks=[0, 50, 100], xlim=(.075, 1.025))
        ax.set_title(task, loc="left", fontsize=9, pad=3)
        ax.grid(axis="y", color="#eeeeee", linewidth=.6)
    axes[-1].set(xticks=[.1, .3, .5, .7, 1.], xlabel=r"Shared TA coefficient $\lambda$")
    fig.legend(*axes[0].get_legend_handles_labels(), ncol=1, frameon=False, loc="upper center",
               bbox_to_anchor=(.54, 1), handlelength=2, handletextpad=.6)
    fig.supylabel("Calibration accuracy (%)", x=.005, fontsize=8.5)
    fig.savefig(OUT / "coefficient_calibration.pdf", metadata={"Title": "Complete labelled calibration curves; 256 images per task"})
    plt.close(fig)
    print("Wrote complete coefficient calibration curves from270 frozen rows; no new model evaluation.")
