"""Draw the fixed-readout bound from the recorded, image-free analysis."""

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
ax.legend(frameon=False, fontsize=7, loc="upper right", ncols=1)
out = ROOT / "paper/fig/readout_bound.pdf"
fig.savefig(out, bbox_inches="tight", metadata={"Creator": "Matplotlib", "Title": "Fixed-readout margin bounds"})
plt.close(fig)
print("Wrote", out.relative_to(ROOT), "from 42 recorded readouts; no model execution.")
