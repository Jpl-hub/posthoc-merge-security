"""Export manuscript tables from the existing, fingerprinted result collection."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "tables"
source = json.loads((ROOT / "results/comparison/sources.json").read_text())
for path, expected in source["input_sha256"].items():
    assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
tables = source["tables"]
tasks = ["SUN397", "Cars", "RESISC45", "SVHN", "GTSRB", "MNIST", "DTD"]
methods = ["Unprotected", "PaRaMS", "Defense 1"]
display = {"Unprotected": "None", "PaRaMS": "PaRaMS", "Defense 1": "NBP"}
rows = tables["paper_main"]["rows"] + tables["defense1_additional_pairs"]["rows"]
values = {(row[0].split("/")[0], row[1]): row[2:] for row in rows if row[1] in methods}


def write_table(name, caption, label, headers, groups, wide=False):
    env = "table*" if wide else "table"
    columns = len(headers)
    grouped = any(group for group, _ in groups)
    spec = "ll" + "r" * (columns - 2) if grouped else "l" + "r" * (columns - 1)
    lines = [rf"\begin{{{env}}}[!t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\begin{tabular}{" + spec + "}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    md = [caption, "", "| " + " | ".join(headers) + " |",
          "| " + " | ".join(["---"] * columns) + " |"]
    for group_id, (group, records) in enumerate(groups):
        if group_id:
            lines.append(r"\midrule")
        for i, record in enumerate(records):
            prefix = rf"\multirow{{{len(records)}}}{{*}}{{{group}}}" if i == 0 else ""
            lines.append(" & ".join(([prefix] if grouped else []) + record) + r" \\")
            md.append("| " + " | ".join(([group] if grouped else []) + record) + " |")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n")


groups = []
for setting, index in [("Standalone", 0), ("TA", 1), ("AGA + TA", 2)]:
    records = [[display[method]] + [values[task, method][index].split(" / ")[0]
               for task in tasks] for method in methods]
    groups.append((setting, records))
write_table("main", "Seven protected tasks in two-expert merging. Standalone accuracy should remain high; lower merged protected-task accuracy indicates stronger protection. TA uses coefficient 0.5. AGA processes the protected expert, including in each matching unprotected control. MNIST pairs with SVHN; other targets pair with MNIST. All values are complete-test accuracy (\\%).",
            "tab:main", ["Setting", "Defense"] + tasks, groups, wide=True)

records = [[task] + [values[task, method][2].split(" / ")[1] for method in methods]
           for task in tasks]
# A single-column table: the first column identifies the protected task.
write_table("partners", "Companion-task accuracy after AGA and TA (\\%). Companions are fixed as in Table~\\ref{tab:main}. NBP can impair other participating tasks as well as the protected task.",
            "tab:partners", ["Target", "None", "PaRaMS", "NBP"], [("", records)])

small_tasks = ["MNIST", "SVHN", "DTD"]
replacement = tables["defense1_B_replacement"]["rows"]
reset = {(r[0], r[2]): r[5] for r in replacement if r[0] == r[1]}
distill = {(r[0], r[1]): r[2] for r in tables["feature_distillation"]["rows"]}
groups = [
    ("B reset", [[display[m]] + [reset[t, m] for t in small_tasks]
                 for m in ["Unprotected", "Defense 1"]]),
    ("ProDistill", [[display[m]] + [distill[t, m] for t in small_tasks] for m in methods]),
]
write_table("recovery", "Protected-task accuracy after the specified recovery procedures (\\%). B reset replaces the final MLP output weight and bias after AGA, then applies TA. ProDistill uses its fixed layer-wise fitting protocol; final LayerNorm and projection are not optimized. The control receives the same processing.",
            "tab:recovery", ["Procedure", "Defense"] + small_tasks, groups)

fusion_root = ROOT / "results/phase3/bias_weight_only/fusion_methods"
fusion_rows = json.loads((fusion_root / "results.json").read_text())
audit = json.loads((fusion_root / "independent_validation.json").read_text())
execution = json.loads((fusion_root / "execution.json").read_text())
assert audit["status"] == "passed" and audit["rows"] == len(fusion_rows) == 27
assert audit["raw_sha256"] == execution["predictions_sha256"]
lookup = {(r["merger"], r["attack_scope"], r["defense"], r["dataset"]): r for r in fusion_rows}
groups = []
for merger in ["TIES", "DARE+TA"]:
    for scope in ["None", "Protected expert"]:
        records = []
        for method, label in [("Unprotected", "None"), ("PaRaMS_SUN397", "PaRaMS"),
                              ("Defense1_SUN397", "NBP")]:
            cells = []
            for task in ["SUN397", "MNIST"]:
                row = lookup[merger, scope, method, task]
                assert sum(c["correct"] for c in row["per_class"]) == row["correct"]
                cells.append(f"{100 * row['correct'] / row['n']:.2f}")
            records.append([label] + cells)
        groups.append((merger + (" + AGA" if scope != "None" else ""), records))
write_table("fusion", "Additional fusion rules on the fixed SUN397/MNIST pair (\\%). TIES retains 20\\% of entries with scale 1. DARE+TA uses drop rate 0.2, seed 42, and TA coefficient 0.5. AGA processes the protected expert. Both complete test sets and matching controls are reported.",
            "tab:fusion", ["Fusion / recovery", "Defense", "SUN397", "MNIST"], groups)
print("Generated four manuscript tables from verified existing and newly completed results.")
