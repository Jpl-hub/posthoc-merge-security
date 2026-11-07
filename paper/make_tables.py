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


eight = {}


def add_eight(target, method, scope, task, n, correct, parameter_sha):
    if task != target:
        return
    method = "Defense 1" if method.startswith("Defense1") else "PaRaMS" if method.startswith("PaRaMS") else method
    key = target, method, scope
    value = n, correct, parameter_sha
    assert key not in eight or eight[key] == value, key
    eight[key] = value


for path, field in [("results/phase3/bias_weight_only/eight_expert/summary.json", "paired_summary"),
                    ("results/params_reference/eight_expert/summary.json", "independent_statistics")]:
    summary = json.loads((ROOT / path).read_text())
    assert summary["status"] == "completed_independently_verified"
    for row in summary[field]:
        target, name, scope, task = row["key"].split("/")
        add_eight(target, name, scope, task, row["n"], row["correct"],
                  summary["parameters"]["/".join((target, name, scope))])

extension = ROOT / "results/phase3/bias_weight_only/eight_expert/extension"
execution = json.loads((extension / "execution.json").read_text())
assert execution["status"] == "completed_independently_verified"
for name in ["results.json", "validation.json", "independent_statistics.json"]:
    path = extension / name
    expected = execution["files_sha256"]["outputs/nbp_eight_extension_20261106/full/" + name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    source["input_sha256"][str(path.relative_to(ROOT))] = expected
path = extension / "execution.json"
source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
validation = json.loads((extension / "validation.json").read_text())
statistics = json.loads((extension / "independent_statistics.json").read_text())
extension_rows = json.loads((extension / "results.json").read_text())
assert validation["status"] == statistics["status"] == "passed"
assert validation["rows"] == statistics["rows"] == len(extension_rows) == 168
assert validation["raw_sha256"] == statistics["raw_sha256"] == execution["raw_sha256"]
recount = {r["key"]: r for r in statistics["independent_statistics"]}
for row in extension_rows:
    key = "/".join(row[k] for k in ["target", "defense", "attack_scope", "dataset"])
    assert (row["n"], row["correct"]) == (recount[key]["n"], recount[key]["correct"])
    add_eight(row["target"], row["defense"], row["attack_scope"], row["dataset"],
              row["n"], row["correct"], row["model_parameter_sha256"])
assert len(eight) == len(tasks) * len(methods) * 2


def write_table(name, caption, label, headers, groups, wide=False):
    env = "table*" if wide else "table"
    columns = len(headers)
    grouped = any(group for group, _ in groups)
    spec = "ll" + "r" * (columns - 2) if grouped else "l" + "r" * (columns - 1)
    lines = [rf"\begin{{{env}}}[!t]", r"\centering", r"\small",
             r"\setlength{\tabcolsep}{4pt}", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\begin{tabular}{" + spec + "}",
             r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    md_caption = caption.replace(r"\%", "%").replace(r"Table~\ref{tab:main}", "the main table")
    md = [md_caption, "", "| " + " | ".join(headers) + " |",
          "| " + " | ".join(["---"] * (2 if grouped else 1)
                              + ["---:"] * (columns - (2 if grouped else 1))) + " |"]
    for group_id, (group, records) in enumerate(groups):
        if group_id:
            lines.append(r"\midrule")
        best = None
        if name == "main" and group != "Standalone":
            best = [min(column, key=float) for column in zip(*(row[1:] for row in records))]
        for i, record in enumerate(records):
            prefix = rf"\multirow{{{len(records)}}}{{*}}{{{group}}}" if i == 0 else ""
            highlighted = [record[0] == "NBP"] + [best is not None and value == best[j]
                for j, value in enumerate(record[1:])]
            tex_record = [rf"\textbf{{{v}}}" if bold else v for v, bold in zip(record, highlighted)]
            lines.append(" & ".join(([prefix] if grouped else []) + tex_record) + r" \\")
            md.append("| " + " | ".join(([group] if grouped else []) + record) + " |")
    lines += [r"\bottomrule", r"\end{tabular}", rf"\end{{{env}}}"]
    (OUT / f"{name}.tex").write_text("\n".join(lines) + "\n")
    (OUT / f"{name}.md").write_text("\n".join(md) + "\n")


groups = []
for setting, index in [("Standalone", 0), ("2-expert TA", 1), ("2-expert AGA + TA", 2)]:
    records = [[display[method]] + [values[task, method][index].split(" / ")[0]
               for task in tasks] for method in methods]
    groups.append((setting, records))
for setting, scope in [("8-expert TA", "None"), ("8-expert AGA + TA", "Protected expert")]:
    records = []
    for method in methods:
        cells = [f"{100 * eight[task, method, scope][1] / eight[task, method, scope][0]:.2f}"
                 for task in tasks]
        records.append([display[method]] + cells)
    groups.append((setting, records))
# Report the task macro-average from the displayed accuracies.
for _, records in groups:
    for record in records:
        record.append(f"{sum(float(v) for v in record[1:]) / len(tasks):.2f}")
write_table("main", "Standalone and protected-task accuracy (\\%) with matched checkpoints and processing. Bold scores mark the lowest merged accuracy per setting. TA coefficients are 0.5 for two experts and 0.2 for eight experts. Avg. is the unweighted mean of the seven displayed task scores. Companions are SVHN for MNIST and MNIST otherwise. EuroSAT participates in eight-expert merging.",
            "tab:main", ["Setting", "Defense"] + tasks + ["Avg."], groups, wide=True)

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

ablation_root = ROOT / "results/phase3/bias_weight_only/ablation"
ablation_rows = json.loads((ablation_root / "results.json").read_text())
ablation_audit = json.loads((ablation_root / "independent_validation.json").read_text())
ablation_execution = json.loads((ablation_root / "execution.json").read_text())
ablation_protocol = json.loads((ablation_root / "protocol.json").read_text())
assert ablation_audit["status"] == "passed" and ablation_audit["rows"] == len(ablation_rows) == 40
assert ablation_audit["raw_sha256"] == ablation_execution["predictions_sha256"]
assert ablation_execution["evaluated_parameter_sha256"] == ablation_protocol["recheck"]["expected_parameter_sha256"]
for name in ["results.json", "independent_validation.json", "execution.json", "protocol.json"]:
    path = ablation_root / name
    source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
ablation_lookup = {(r["defense"], r["merger"], r["attack_scope"], r["dataset"]): r for r in ablation_rows}
records = []
for name, label in [("Unprotected", "None"), ("NBP", "NBP"),
                    ("Random_null", "Random nullspace"), ("Ambient", "Random direction")]:
    cells = []
    for merger, scope in [("Standalone", "None"), ("TA", "None"), ("TA", "Protected expert")]:
        row = ablation_lookup[name, merger, scope, "MNIST"]
        cells.append(f"{100 * row['correct'] / row['n']:.2f}")
    records.append([label] + cells)
write_table("ablation", "Direction controls on MNIST (\\%). The three perturbations have the same update norm. The random nullspace direction is projected into the original visual-projection kernel; the unconstrained random direction uses the same sampled vector before projection. All controls use the same MNIST/SVHN pair and TA coefficient 0.5.",
            "tab:ablation", ["Update", "Standalone", "TA", "AGA + TA"], [("", records)])
# Reuse completed B/16 and public-bias cleanup records; no new evaluation.
b16_path = ROOT / "results/backbone_b16/summary.json"
b16 = json.loads(b16_path.read_text())
assert b16["status"] == "completed_independently_verified"
by_key = {(r["target"], r["key"]): r for r in b16["full_rows"]}
groups = []
for setting, merger, scope in [("Standalone", "Standalone", "None"),
        ("TA", "TA", "None"), ("AGA + TA", "TA", "Protected expert")]:
    records = []
    for label in ["None", "NBP"]:
        cells = []
        for t in small_tasks:
            name = "Unprotected" if label == "None" else "Defense1_" + t
            r = by_key[t, "/".join((name, merger, scope, t))]
            cells.append(f"{100*r['correct']/r['n']:.2f}")
        records.append([label] + cells)
    groups.append((setting, records))
write_table("backbone", "CLIP ViT-B/16 protected-task accuracy (\\%). The same NBP construction is applied to three experts. Two-expert TA uses coefficient 0.5; AGA processes the protected expert before merging.",
    "tab:backbone", ["Setting", "Defense"] + small_tasks, groups)
cleanup_path = ROOT / "results/phase3/bias_weight_only/attacks/projection.json"
cleanup = json.loads(cleanup_path.read_text())
assert cleanup["status"] == "completed_independently_verified"
assert cleanup["matched_prediction_differences"] == 0
records = [[r[0]] + r[3:7] for r in cleanup["table_cells"]]
write_table("cleanup", "TA accuracy after public final-bias projection (\\%). Columns give target (T) and companion (C) accuracy. Both releases receive the same cleanup before TA at 0.5. Their full prediction vectors are identical in these experiments.",
    "tab:cleanup", ["Target", "None T", "None C", "NBP T", "NBP C"], [("",records)])
for path in [b16_path, cleanup_path]:
    source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
(OUT / "sources.json").write_text(json.dumps({"input_sha256": source["input_sha256"],
    "eight_expert_protected_task_cells": len(eight),
    "tables": ["main", "partners", "recovery", "fusion", "ablation", "backbone", "cleanup"]}, indent=2) + "\n")
print("Generated seven manuscript tables from recorded results.")
