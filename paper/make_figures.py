"""Draw manuscript figures from the recorded NBP results."""

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
record = ROOT / "results/phase3/bias_weight_only/readout_analysis.json"
protocol = json.loads(record.with_name("readout_analysis_protocol.json").read_text())
assert hashlib.sha256((ROOT / "experiment/analyze_bias.py").read_bytes()).hexdigest() == protocol["source_sha256"]
for path, sha in protocol["input_protocol_sha256"].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == sha
data = json.loads(record.read_text())
assert data["independent_numpy_formula_check"] and len(data["rows"]) == 42
tasks = ["SUN397", "Cars", "RESISC45", "SVHN", "GTSRB", "MNIST", "DTD"]
lookup = {(r["task"], r["scenario"], r["protected"]): r for r in data["rows"]}
assert len(lookup) == 42
for row in lookup.values():
    assert row["constant_prediction_certified_real_arithmetic"] == (row["minimum_margin_lower_bound"] > 0)

plt.rcParams.update({"font.family": "serif", "font.size": 8,
                     "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(3.35, 2.4), layout="constrained")
x = np.arange(len(tasks))
for shift, scenario, label, color, marker in [
    (-.16, "Public-base proxy", "Public-base proxy", "#777777", "o"),
    (0, "Two experts", "Two experts", "#25639c", "s"),
    (.16, "Eight experts", "Eight experts", "#b55524", "^"),
]:
    y = [lookup[t, scenario, True]["minimum_margin_lower_bound"] for t in tasks]
    ax.scatter(x + shift, y, s=23, color=color, marker=marker, label=label, zorder=3)
ax.axhline(0, color="black", linewidth=.7)
ax.set(xticks=x, xticklabels=["SUN", "Cars", "R45", "SVHN", "GTSRB", "MNIST", "DTD"],
       ylabel="Minimum class-margin lower bound", ylim=(-900, 1200))
ax.tick_params(axis="x", labelsize=7)
ax.legend(frameon=False, fontsize=7, loc="upper right")
out = ROOT / "paper/fig/readout_bound.pdf"
fig.savefig(out, bbox_inches="tight", metadata={"Creator": "Matplotlib", "Title": "Fixed-readout margin bounds"})
plt.close(fig)
print("Wrote", out.relative_to(ROOT), "from 42 recorded readouts; no model execution.")

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
ax.set_xscale("log", base=2)
ax.set(xticks=amplitudes, xticklabels=["0.125", "0.25", "0.5", "1", "2"],
       xlabel="Perturbation magnitude / NBP magnitude",
       ylabel="MNIST accuracy (%)", ylim=(0, 105), yticks=[0, 25, 50, 75, 100])
ax.legend(frameon=False, fontsize=7, loc="center right")
out = ROOT / "paper/fig/amplitude.pdf"
fig.savefig(out, bbox_inches="tight", metadata={"Creator": "Matplotlib", "Title": "Fixed NBP amplitude controls"})
plt.close(fig)
print("Wrote", out.relative_to(ROOT), "from verified fixed-amplitude controls.")

# Plot actual class counts from the matched AGA merge, without embedding sampling.
fig, axes = plt.subplots(2, 1, figsize=(3.4, 2.25), sharex=True, layout="constrained")
classes = np.arange(10)
for ax, name, label, color in zip(axes, ["Unprotected", "NBP"],
        ["Unprotected", "NBP"], ["#25639c", "#b55524"]):
    row = scores[name, "TA", "Protected expert", "MNIST"]
    counts = np.asarray(row["predicted_class_counts"])
    assert counts.sum() == row["n"]
    ax.bar(classes, 100 * counts / row["n"], width=.65, color=color)
    ax.set(ylim=(0, 106), yticks=[0, 50, 100])
    ax.text(.98, .82, label, ha="right", transform=ax.transAxes, weight="bold")
    ax.grid(axis="y", alpha=.18)
axes[-1].set(xticks=classes, xlabel="Predicted digit")
fig.supylabel("Predictions (%)", fontsize=8)
fig.savefig(ROOT / "paper/fig/predictions.pdf", metadata={"Title": "MNIST predictions after AGA and TA"})
plt.close(fig)

# A single-column dot plot exposes taskwise differences hidden by macro-averages.
manifest = json.loads((ROOT / "paper/tables/sources.json").read_text())
for relative, expected in manifest["input_sha256"].items():
    assert hashlib.sha256((ROOT / relative).read_bytes()).hexdigest() == expected
comparison = json.loads((ROOT / "results/comparison/sources.json").read_text())
ta_rows = comparison["tables"]["paper_main"]["rows"] + comparison["tables"]["defense1_additional_pairs"]["rows"]
ta = {r[0].split("/")[0]: float(r[4].split(" / ")[0]) for r in ta_rows if r[1] == "Defense 1"}
fusion_root = ROOT / "results/phase3/bias_weight_only/fusion_methods"
fusion = []
for folder in [fusion_root] + [fusion_root / "extension" / task for task in tasks if task != "SUN397"]:
    fusion.extend(json.loads((folder / "results.json").read_text()))
fused = {(r["target"], r["merger"]): 100*r["correct"]/r["n"]
         for r in fusion if r["dataset"] == r["target"] and r["attack_scope"] == "Protected expert"
         and r["defense"].startswith("Defense1_")}
fig, ax = plt.subplots(figsize=(3.35, 2.8), layout="constrained")
y = np.arange(len(tasks))
for offset, merger, marker, color in [(-.18, "TA", "o", "#25639c"),
        (0, "TIES", "s", "#b55524"), (.18, "DARE+TA", "^", "#5b7b42")]:
    values = [ta[t] if merger == "TA" else fused[t, merger] for t in tasks]
    ax.scatter(values, y+offset, marker=marker, s=22, color=color, label=merger, zorder=3)
ax.set(yticks=y, yticklabels=tasks, xlabel="NBP target accuracy after AGA (%)",
       xlim=(-.6, 25), xticks=[0,5,10,15,20,25], ylim=(len(tasks)-.5,-1.05))
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color="#e6e6e6", linewidth=.6, zorder=0)
ax.legend(frameon=False, fontsize=7, ncol=3, loc="upper left", columnspacing=.8, handletextpad=.25)
fig.savefig(ROOT / "paper/fig/fusion_tasks.pdf", metadata={"Title": "Taskwise NBP accuracy after AGA and fusion"})
plt.close(fig)
print("Wrote single-column taskwise fusion comparison from recorded complete-test results.")
