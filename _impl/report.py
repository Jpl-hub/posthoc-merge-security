"""Rebuild NBP result tables from frozen evaluation records; no models required."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def tables(output):
    OUT = Path(output)
    if OUT.resolve() == (ROOT / "results/comparison").resolve():
        raise ValueError("Choose an output directory separate from the frozen input records")
    OUT.mkdir(parents=True, exist_ok=True)
    source = json.loads((ROOT / "results/comparison/sources.json").read_text())
    for path, expected in source["input_sha256"].items():
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == expected, path
    tables = source["tables"]
    tasks = ["SUN397", "Cars", "RESISC45", "SVHN", "GTSRB", "MNIST", "DTD"]
    methods = ["Unprotected", "PaRaMS", "Defense 1"]
    display = {"Unprotected": "None", "PaRaMS": "PaRaMS", "Defense 1": "NBP"}
    rows = tables["paper_main"]["rows"] + tables["defense1_additional_pairs"]["rows"]
    main_values = {(row[0].split("/")[0], row[1]): row[2:] for row in rows if row[1] in methods}


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


    def write_table(name, caption, label, headers, groups, wide=False, tabcolsep=4, section_headers=False, column_spec=None):
        env = "table*" if wide else "table"
        columns = len(headers)
        grouped = any(group for group, _ in groups) and not section_headers
        spec = column_spec or ("ll" + "r" * (columns - 2) if grouped else "l" + "r" * (columns - 1))
        lines = [rf"\begin{{{env}}}[!t]", r"\centering", r"\small",
                 rf"\setlength{{\tabcolsep}}{{{tabcolsep}pt}}", rf"\caption{{{caption}}}",
                 rf"\label{{{label}}}", r"\begin{tabular}{" + spec + "}",
                 r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
        md_caption = caption.replace(r"\%", "%").replace(r"Table~\ref{tab:main}", "the main table").replace(r"Table~\ref{tab:fusion}", "the fusion summary table")
        md_headers = []
        for h in headers:
            if r"\shortstack{" in h:
                h = h.replace(r"\shortstack{", "").replace("}", "")
            md_headers.append(h.replace(r"$\lambda$", "Coeff.").replace(r"\\", " ")
                              .replace("$_t$", " (target)").replace("$_c$", " (companion)"))
        md = [md_caption, "", "| " + " | ".join(md_headers) + " |",
              "| " + " | ".join(["---"] * columns if column_spec else
                                  ["---"] * (2 if grouped else 1)
                                  + ["---:"] * (columns - (2 if grouped else 1))) + " |"]
        for group_id, (group, records) in enumerate(groups):
            if group_id:
                lines.append(r"\midrule")
            if name == "training_baseline" and group_id % 3 == 0:
                protocol = "Author Trap$^2$ training" if group_id == 0 else "Paper-based MergeGuard reproduction"
                lines.append(r"\multicolumn{5}{l}{\textit{" + protocol + r"}} \\")
                md.append("| **" + protocol + "** | | | | |")
            if section_headers:
                lines.append(rf"\multicolumn{{{columns}}}{{l}}{{{group}}}" + r" \\")
                md.append("| **" + group + "** |" + " |" * (columns - 1))
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
        records = [[display[method]] + [main_values[task, method][index].split(" / ")[0]
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

    # Complete directed partner pairs: 35 new pairs plus the seven fixed pairs.
    pair_root = ROOT / "results/phase3/bias_weight_only/partner_analysis"
    pair_rows = []
    for folder in sorted((pair_root / "completed").iterdir()):
        if not folder.is_dir():
            continue
        measured = json.loads((folder / "results.json").read_text())
        audit = json.loads((folder / "independent_validation.json").read_text())
        execution = json.loads((folder / "execution.json").read_text())
        assert audit["status"] == "passed" and audit["rows"] == len(measured) == 10
        assert audit["raw_sha256"] == execution["predictions_sha256"]
        for name in ["results.json", "independent_validation.json", "execution.json"]:
            path = folder / name
            source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        pair_rows.extend(r for r in measured if r["merger"] == "TA")
    assert len(pair_rows) == 35 * 8
    fixed_path = ROOT / "results/phase3/bias_weight_only/eight_expert/remaining_pairs_result.json"
    fixed = json.loads(fixed_path.read_text())
    assert fixed["status"] == "completed_verified" and fixed["completed_tasks"] == 4
    for record in fixed["records"]:
        for row in record["rows"]:
            if row["merger"] == "TA" and not row["defense"].startswith("PaRaMS"):
                pair_rows.append(dict(row, target=record["target"], partner=record["partner"]))
    original_path = ROOT / "results/phase3/bias_weight_only/other_tasks/results.json"
    original = json.loads(original_path.read_text())
    for role, field in [("target", "combined_target_table"), ("companion", "combined_partner_table")]:
        for row in original[field]:
            dataset = row["target"] if role == "target" else row["partner"]
            for scope, index in [("None", 1 if role == "target" else 0),
                                 ("Protected expert", 2 if role == "target" else 1)]:
                correct, n = row["counts"][index]
                pair_rows.append(dict(target=row["target"], partner=row["partner"], dataset=dataset,
                    defense=row["method"], merger="TA", attack_scope=scope, n=n, correct=correct))
    for path in [fixed_path, original_path]:
        source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    pair_lookup = {}
    for row in pair_rows:
        method = "None" if row["defense"] == "Unprotected" else "NBP"
        key = row["target"], row["partner"], method, row["attack_scope"], row["dataset"]
        assert key not in pair_lookup, key
        pair_lookup[key] = row
    assert len(pair_lookup) == 42 * 8
    for role, title in [("target", "Protected-target"), ("companion", "Companion-task")]:
        groups = []
        for task in tasks:
            records = []
            for method in ["None", "NBP"]:
                cells, values = [], []
                for companion in tasks:
                    if companion == task:
                        cells.append("--")
                        continue
                    dataset = task if role == "target" else companion
                    row = pair_lookup[task, companion, method, "Protected expert", dataset]
                    value = 100*row["correct"]/row["n"]
                    cells.append(f"{value:.2f}")
                    values.append(value)
                records.append([method]+cells+[f"{sum(values)/len(values):.2f}", f"{max(values):.2f}"])
            groups.append((task, records))
        write_table("pair_"+role, title+" accuracy after AGA and TA0.5 for all 42 directed pairs (\\%). Each protected expert has separate unprotected (None) and NBP rows. Columns select the companion; the diagonal is not evaluated. Mean and Max summarize its six companions; a larger NBP score means more retained merged capability.",
            "tab:pair-"+role, ["Protected expert", "Defense"]+tasks+["Mean", "Max"], groups, wide=True, tabcolsep=3)
    records = []
    pair_summary = []
    for task in tasks:
        companions = [t for t in tasks if t != task]
        groups = []
        for role in ["target", "companion"]:
            for method in ["None", "NBP"]:
                scores = [100 * pair_lookup[task, c, method, "Protected expert", task if role == "target" else c]["correct"]
                    / pair_lookup[task, c, method, "Protected expert", task if role == "target" else c]["n"] for c in companions]
                groups.append(sum(scores) / len(scores))
                pair_summary.append(dict(target=task, role=role, defense=method, partners=len(scores),
                    mean_accuracy_percent=groups[-1], minimum_accuracy_percent=min(scores), maximum_accuracy_percent=max(scores)))
        records.append([task]+[f"{v:.2f}" for v in groups])
    write_table("partners", "Accuracy after AGA and TA across all 42 directed pairs (\\%). Each row averages the six other validated tasks as companions. Subscripts t and c denote the protected target and companion; U is the unprotected control. The same seven NBP releases are used throughout.",
        "tab:partners", ["Task", r"U$_t$", r"NBP$_t$", r"U$_c$", r"NBP$_c$"], [("", records)])
    # Export all counts, including the direct-TA condition, without rounding the source.
    import csv
    with (OUT / "partners_full.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["target", "partner", "defense", "attack_scope", "dataset", "n", "correct", "accuracy_percent"])
        for key, row in sorted(pair_lookup.items()):
            writer.writerow(list(key)+[row["n"], row["correct"], 100*row["correct"]/row["n"]])
    (OUT / "partners_summary.json").write_text(json.dumps(dict(directed_pairs=42, comparison_rows=len(pair_lookup),
        additional_complete_evaluation_rows=350, additional_score_replays=5600, rows=pair_summary), indent=2)+"\n")

    small_tasks = ["MNIST", "SVHN", "DTD"]
    replacement = tables["defense1_B_replacement"]["rows"]
    reset = {(r[0], r[2]): r[5] for r in replacement if r[0] == r[1]}
    distill = {(r[0], r[1]): r[2] for r in tables["feature_distillation"]["rows"]}
    groups = [
        ("MLP reset", [[display[m]] + [reset[t, m] for t in small_tasks]
                     for m in ["Unprotected", "Defense 1"]]),
        ("ProDistill", [[display[m]] + [distill[t, m] for t in small_tasks] for m in methods]),
    ]
    cleanup_path = ROOT / "results/phase3/bias_weight_only/attacks/projection.json"
    cleanup = json.loads(cleanup_path.read_text())
    assert cleanup["status"] == "completed_independently_verified" and cleanup["matched_prediction_differences"] == 0
    cleanup_scores = {r[0]: r for r in cleanup["table_cells"]}
    groups.append(("Bias projection", [[label] + [cleanup_scores[t][index] for t in small_tasks]
        for label, index in [("None", 3), ("NBP", 5)]]))
    source["input_sha256"][str(cleanup_path.relative_to(ROOT))] = hashlib.sha256(cleanup_path.read_bytes()).hexdigest()
    capabilities = {"MLP reset": "MLP reset: no images; final readout fixed",
                    "ProDistill": "ProDistill: 64 images/task; final readout fixed",
                    "Bias projection": "Bias projection: no images; final bias edited"}
    groups = [(capabilities[group], records) for group, records in groups]
    write_table("recovery", "Target accuracy after recovery (\\%). Group headings state image access and final-parameter edits. MLP replacement follows AGA; bias projection precedes TA. Controls receive the same operation. Bias projection gives identical NBP/control predictions.",
                "tab:recovery", ["Defense"] + small_tasks, groups, section_headers=True)
    write_table("recovery_access", "Implemented recovery settings. All procedures can read the published weights. AGA and MLP replacement use the public base; coefficient fitting and selection use the public base and expert checkpoints. Bias projection needs only the published bias and projection. The changed parameters describe recovery itself; subsequent fusion also combines final normalization and projection parameters.",
        "tab:recovery-access", ["Procedure", "Images/task", "Labels", "Changed parameters", "Budget"], [("", [
            ["AGA", "0", "No", "Attention and MLP blocks", "One author recovery pass"],
            ["MLP replacement", "0", "No", "Last MLP output matrix and bias", "One replacement after AGA"],
            ["ProDistill", "64", "No", "Element-wise block coefficients", "100 steps/block"],
            ["Coefficient selection", "256", "Yes", "Shared TA coefficient", "10 values, 0.1--1.0"],
            ["Bias projection", "0", "No", "Final LayerNorm bias", "One least-squares solve"],
        ])], wide=True, tabcolsep=3, column_spec="lllp{5.2cm}l")

    fusion_root = ROOT / "results/phase3/bias_weight_only/fusion_methods"
    fusion_rows = []
    for folder in [fusion_root] + [fusion_root / "extension" / task for task in tasks if task != "SUN397"]:
        rows = json.loads((folder / "results.json").read_text())
        audit = json.loads((folder / "independent_validation.json").read_text())
        execution = json.loads((folder / "execution.json").read_text())
        assert audit["status"] == "passed" and audit["rows"] == len(rows) == 27
        assert audit["raw_sha256"] == execution["predictions_sha256"]
        for name in ["results.json", "independent_validation.json", "execution.json"]:
            path = folder / name
            source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        fusion_rows.extend(rows)
    lookup = {(r["target"], r["merger"], r["attack_scope"], r["defense"], r["dataset"]): r for r in fusion_rows}
    assert len(lookup) == 189
    groups, details, companion_details = [], [], []
    for merger in ["TIES", "DARE+TA"]:
        for scope in ["None", "Protected expert"]:
            records, target_records, companion_records = [], [], []
            for method, label in [("Unprotected", "None"), ("PaRaMS_", "PaRaMS"), ("Defense1_", "NBP")]:
                target_scores, partner_scores = [], []
                for task in tasks:
                    name = method if method == "Unprotected" else method + task
                    for dataset, scores in [(task, target_scores), ("SVHN" if task == "MNIST" else "MNIST", partner_scores)]:
                        row = lookup[task, merger, scope, name, dataset]
                        assert sum(c["correct"] for c in row["per_class"]) == row["correct"]
                        scores.append(100 * row["correct"] / row["n"])
                records.append([label, f"{sum(target_scores)/len(tasks):.2f}", f"{sum(partner_scores)/len(tasks):.2f}"])
                target_records.append([label] + [f"{v:.2f}" for v in target_scores])
                companion_records.append([label] + [f"{v:.2f}" for v in partner_scores])
            setting = ("AGA + " if scope != "None" else "") + merger
            groups.append((setting, records))
            details.append((setting, target_records))
            companion_details.append((setting, companion_records))
    write_table("fusion", "Seven-task macro-average accuracy (\\%) for additional fusion rules. Target and companion accuracies are averaged across the same seven fixed pairs as Table~\\ref{tab:main}. TIES retains 20\\%; DARE drops 20\\% with seed 42, followed by TA at 0.5. AGA processes the protected expert before fusion.",
                "tab:fusion", ["Procedure", "Defense", "Target", "Companion"], groups)
    write_table("fusion_tasks", "Complete protected-task accuracy (\\%) for TIES and DARE+TA. Same fixed pairs as the main comparison. TIES keeps 20\\%; DARE drops 20\\% with seed 42 before TA0.5.",
                "tab:fusion-tasks", ["Procedure", "Defense"] + tasks, details, wide=True)
    write_table("fusion_companions", "Complete companion-task accuracy (\\%) for TIES and DARE+TA. Column names identify the protected expert. Its companion is SVHN for MNIST and MNIST otherwise. Same frozen models and settings as the protected-task table.",
        "tab:fusion-companions", ["Procedure", "Defense"]+tasks, companion_details, wide=True)


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
    groups = []
    ablation_targets = [("MNIST", ablation_rows)]
    for task in ["SVHN", "DTD"]:
        folder = ablation_root / "extension" / task
        if not (folder / "independent_validation.json").exists():
            continue
        extra = json.loads((folder / "results.json").read_text())
        checked = json.loads((folder / "independent_validation.json").read_text())
        execution = json.loads((folder / "execution.json").read_text())
        assert checked["status"] == "passed" and checked["rows"] == len(extra) == 20
        assert checked["raw_sha256"] == execution["predictions_sha256"]
        ablation_targets.append((task, extra))
        for name in ["results.json", "independent_validation.json", "execution.json", "freeze.json"]:
            path = folder / name
            source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    import statistics
    for task, measured in ablation_targets:
        lookup = {(r["defense"], r["merger"], r["attack_scope"], r["dataset"]): r for r in measured}
        folder = ablation_root / "direction_controls" / task
        extra = json.loads((folder / "results.json").read_text())
        audit = json.loads((folder / "independent_validation.json").read_text())
        execution = json.loads((folder / "execution.json").read_text())
        assert audit["status"] == "passed" and audit["rows"] == len(extra) == 30
        assert audit["raw_sha256"] == execution["predictions_sha256"]
        for name in ["results.json", "independent_validation.json", "execution.json", "freeze.json"]:
            path = folder / name
            source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        projection = {(r["merger"], r["attack_scope"]): r for r in extra
                      if r["defense"] == "Projected_contrast" and r["dataset"] == task}
        records = []
        for name, label in [("Unprotected", "None"), ("NBP", "NBP"),
                            ("Projected_contrast", "Class projection"),
                            ("Random_null", "Random null"), ("Ambient", "Random bias")]:
            cells = []
            for merger, scope in [("Standalone", "None"), ("TA", "Protected expert")]:
                if name == "Random_null":
                    draws = [r for r in measured+extra if r["defense"].startswith("Random_null")
                             and r["dataset"] == task and r["merger"] == merger and r["attack_scope"] == scope]
                    assert len(draws) == 5
                    values = [100*r["correct"]/r["n"] for r in draws]
                    if merger == "Standalone":
                        assert len(set(values)) == 1
                        cells.append(f"{values[0]:.2f}")
                    else:
                        cells.append(f"{statistics.mean(values):.2f} $\\pm$ {statistics.stdev(values):.2f}")
                else:
                    row = projection[merger, scope] if name == "Projected_contrast" else lookup[name, merger, scope, task]
                    cells.append(f"{100*row['correct']/row['n']:.2f}")
            records.append([label] + cells)
        groups.append((task, records))
    write_table("ablation", "Same-norm direction controls (\\%). Random null reports the mean and sample standard deviation over five fixed seeds after AGA; all five have the displayed standalone accuracy. Class projection uses the same preselected class as NBP, projected directly into the nullspace without the multi-class least-squares solve. All controls use the frozen NBP update norm; companions and TA0.5 match the main table.",
                "tab:ablation", ["Task", "Update", "Standalone", "AGA+TA"], groups)
    records = []
    for task in tasks:
        folder = ablation_root / "direction_controls" / (task if task in small_tasks else "extension/"+task)
        measured = json.loads((folder / "results.json").read_text())
        checked = json.loads((folder / "independent_validation.json").read_text())
        assert checked["status"] == "passed" and checked["rows"] == len(measured)
        for filename in ["results.json", "independent_validation.json", "execution.json", "freeze.json"]:
            path = folder / filename
            source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        selected = {(r["merger"],r["attack_scope"]):r for r in measured if r["defense"] == "Projected_contrast" and r["dataset"] == task}
        cells = [f"{100*selected[k]['correct']/selected[k]['n']:.2f}" for k in [("Standalone","None"),("TA","Protected expert")]]
        records.append([task, main_values[task,"Defense 1"][0].split(" / ")[0],cells[0],
                        main_values[task,"Defense 1"][2].split(" / ")[0],cells[1]])
    write_table("class_projection", "Fixed class projection and NBP on all seven tasks (\\%). The same preselected class and NBP update norm are used. Standalone columns evaluate each released expert; merged columns use AGA and TA0.5 with the main fixed companions. No class or direction is selected using images.",
        "tab:class-projection", ["Task",r"\shortstack{NBP\\native}",r"\shortstack{Proj.\\native}",r"\shortstack{NBP\\merged}",r"\shortstack{Proj.\\merged}"], [("",records)])
    bound_root = ablation_root / "direction_controls/bound_amplitude"
    bound_available = (bound_root / "summary.json").exists()
    if bound_available:
        summary = json.loads((bound_root / "summary.json").read_text())
        assert summary["status"] == "completed_verified" and summary["full_rows"] == 70
        records = []
        for task in tasks:
            folder = bound_root / task
            measured = json.loads((folder / "results.json").read_text())
            check = json.loads((folder / "independent_validation.json").read_text())
            freeze = json.loads((folder / "freeze.json").read_text())
            assert check["status"] == "passed" and check["rows"] == len(measured) == 10
            own = [r for r in measured if r["defense"] == "Projected_contrast" and r["dataset"] == task]
            native = next(r for r in own if r["merger"] == "Standalone")
            merged = next(r for r in own if r["attack_scope"] == "Protected expert")
            previous = json.loads((ablation_root / "direction_controls" / (task if task in small_tasks else "extension/"+task) / "freeze.json").read_text())
            original_norm = next(r["actual_delta_norm"] for r in previous["controls"] if r["name"] == "Projected_contrast")
            ratio = freeze["controls"][0]["actual_delta_norm"] / original_norm
            records.append([task,f"{ratio:.4f}",str(native["changed_predictions"]),f"{100*native['correct']/native['n']:.2f}",f"{100*merged['correct']/merged['n']:.2f}"])
            for filename in ["results.json", "independent_validation.json", "execution.json", "freeze.json", "evaluation.json"]:
                p = folder / filename
                source["input_sha256"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        write_table("projection_bound", "Class projection at its own analytic magnitude. Ratio is the update norm divided by the frozen NBP norm. Changed counts standalone prediction differences from the original expert. Target accuracy is after AGA+TA0.5 with the main fixed companion. Every direction uses the same original class and twice its own public-base sufficient bound; no image-based magnitude search is used.",
            "tab:projection-bound", ["Task","Ratio","Changed","Standalone","AGA+TA"], [("",records)])
    geometry_path = ablation_root / "direction_controls/weight_analysis/results.json"
    geometry = json.loads(geometry_path.read_text())
    assert len(geometry["rows"]) == 1496 and geometry["new_publications"] == 0
    source["input_sha256"][str(geometry_path.relative_to(ROOT))] = hashlib.sha256(geometry_path.read_bytes()).hexdigest()
    records = []
    precision = {(r["task"],r["storage_dtype"]):r for r in geometry["precision"]}
    for row in geometry["tasks"]:
        task = row["task"]
        rounding = [f"{precision[task,dtype]['rounding_only_embedding_residual']:.2g}" if precision[task,dtype]["finite_bias"] else "Overflow" for dtype in ["FP32","FP16","BF16"]]
        records.append([task, str(row["classes"]), str(row["class_map_rank"]), f"{row['frozen_update_norm']:.2g}"]+rounding)
    write_table("precision", "Weight geometry and rounding-only embedding residuals. C is the class count and rank is the retained rank of H. Norm is the released update norm. Each residual is computed after rounding both biases and the expert projection to the specified storage type, followed by FP64 multiplication. These are not low-precision inference accuracies. Cars has nonfinite released FP16 bias values.",
        "tab:precision", ["Task","C","Rank","Norm","FP32","FP16","BF16"], [("",records)], tabcolsep=3)
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
    equal_root = ROOT / "results/phase3/bias_weight_only/eight_expert/equal_weight"
    equal_rows = json.loads((equal_root / "results.json").read_text())
    equal_validation = json.loads((equal_root / "validation.json").read_text())
    equal_statistics = json.loads((equal_root / "independent_statistics.json").read_text())
    assert equal_validation["status"] == equal_statistics["status"] == "passed"
    assert equal_validation["rows"] == equal_statistics["rows"] == len(equal_rows) == 36
    assert equal_validation["raw_sha256"] == equal_statistics["raw_sha256"]
    equal_extension = equal_root / "extension"
    completed = json.loads((equal_extension / "execution.json").read_text())
    assert completed["status"] == "completed_independently_verified"
    for path in equal_extension.glob("*.json"):
        expected = completed["files_sha256"].get("outputs/nbp_eight_equal_extension_20261109/full/" + path.name)
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert expected is None or actual == expected
        source["input_sha256"][str(path.relative_to(ROOT))] = actual
    extra = json.loads((equal_extension / "results.json").read_text())
    verified = json.loads((equal_extension / "independent_statistics.json").read_text())
    validation = json.loads((equal_extension / "validation.json").read_text())
    assert verified["status"] == validation["status"] == "passed"
    assert len(extra) == verified["rows"] == validation["rows"] == 112
    assert verified["raw_sha256"] == validation["raw_sha256"] == completed["raw_sha256"]
    recount = {r["key"]: r for r in verified["independent_statistics"]}
    for row in extra:
        key = "/".join(row[k] for k in ["target", "defense", "attack_scope", "dataset"])
        assert (row["n"], row["correct"]) == (recount[key]["n"], recount[key]["correct"])
    equal = {(r["target"], "Unprotected" if r["defense"] == "Unprotected" else "Defense 1", r["attack_scope"]): r
             for r in equal_rows + extra if r["target"] == r["dataset"]}
    assert len(equal) == len(tasks) * 4
    records, values = [], []
    for task in tasks:
        cells = []
        for coefficient in [0.125, 0.2]:
            for method in ["Unprotected", "Defense 1"]:
                if coefficient == 0.125:
                    row = equal[task, method, "Protected expert"]
                    n, correct = row["n"], row["correct"]
                else:
                    n, correct, _ = eight[task, method, "Protected expert"]
                cells.append(100 * correct / n)
        records.append([task] + [f"{v:.2f}" for v in cells])
        values.append(cells)
    means = [sum(column) / len(tasks) for column in zip(*values)]
    write_table("equal_weight", "Protected-task accuracy after AGA and eight-expert TA (\\%). U denotes the unprotected control; parentheses give the shared coefficient. At 1/8, the merge is the arithmetic mean of eight checkpoints. The same seven NBP releases are used at both coefficients. Mean weights tasks equally.",
        "tab:equal-weight", ["Task", "U (1/8)", "NBP (1/8)", "U (0.2)", "NBP (0.2)"],
        [("", records), ("", [["Mean"] + [f"{v:.2f}" for v in means]])], tabcolsep=3)
    for name in ["results.json", "validation.json", "independent_statistics.json"]:
        path = equal_root / name
        source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Same-host construction timings are separate from model utility tests.
    cost_root = ablation_root / "extension" / "cost"
    cost_files = [cost_root / f"{task}_{method}.json" for task in small_tasks for method in ["bias", "params"]]
    exported_tables = ["main", "partners", "pair_target", "pair_companion", "recovery", "recovery_access", "fusion", "fusion_tasks", "fusion_companions", "ablation", "class_projection", "precision", "backbone", "cleanup", "equal_weight"]
    if bound_available:
        exported_tables.append("projection_bound")
    if all(p.exists() for p in cost_files):
        cost = {}
        for path in cost_files:
            row = json.loads(path.read_text())
            assert row["construction_device"] == "cpu" and row["repetitions"] == 1
            assert row["images"] == row["expert_updates"] == row["new_publication_files"] == 0
            cost[row["target"], row["method"]] = row
            source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
        records = [[task]+[f"{cost[task, method]['kernel_seconds']:.2f}" for method in ["bias", "params"]]
                   for task in small_tasks]
        write_table("cost", "Construction time (seconds), one call per method/task on the same CPU host. The timer excludes checkpoint loading, correctness checks and serialization.",
            "tab:cost", ["Task", "NBP", "PaRaMS"], [("", records)])
        exported_tables.append("cost")
    selection_root = ROOT / "results/phase3/bias_weight_only/coefficient_selection"
    selection_path = selection_root / "summary.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        checked = json.loads((selection_root / "selection_validation.json").read_text())
        assert selection["status"] == "completed_verified" and checked["status"] == "passed"
        assert checked["selection_rows"] == len(selection["choices"]) == 18
        test_protocol = json.loads((selection_root / "selected_test_protocol.json").read_text())
        test_rows = {}
        for i, case in enumerate(test_protocol["cases"]):
            folder = selection_root / "test/full" / f"{i:02d}_{case['target']}_{case['partner']}"
            measured = json.loads((folder / "results.json").read_text())
            validation = json.loads((folder / "independent_validation.json").read_text())
            assert validation["status"] == "passed" and validation["rows"] == len(measured)
            for r in measured:
                if r["merger"] == "TA":
                    test_rows[case["target"],case["TA_coefficient"],r["defense"],r["dataset"]] = r
            for filename in ["results.json", "independent_validation.json", "execution.json"]:
                p = folder / filename
                source["input_sha256"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        for r in selection["choices"]:
            companion = "SVHN" if r["task"] == "MNIST" else "MNIST"
            for dataset, field in [(r["task"],"test_target_accuracy"),(companion,"test_companion_accuracy")]:
                measured = test_rows[r["task"],r["coefficient"],r["defense"],dataset]
                assert abs(r[field]-100*measured["correct"]/measured["n"]) < 1e-10
        for filename in ["summary.json", "selection_validation.json", "selected.json", "selected_test_protocol.json", "selection_freeze.json", "calibration_counts.csv", "calibration_validation.json", "selection_rule.json"]:
            p = selection_root / filename
            source["input_sha256"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        for objective, suffix in [("mean", ""), ("target", "_target")]:
            groups = []
            for task in small_tasks:
                records = []
                for method, label in [("Unprotected", "None"), ("PaRaMS_"+task, "PaRaMS"), ("NBP_"+task,"NBP")]:
                    r = next(r for r in selection["choices"] if r["task"] == task and r["defense"] == method and r["objective"] == objective)
                    records.append([label, f"{r['coefficient']:.1f}", f"{r['test_target_accuracy']:.2f}", f"{r['test_companion_accuracy']:.2f}"])
                groups.append((task, records))
            name = "coefficient_selection"+suffix
            criterion = "the mean target/companion" if objective == "mean" else "target"
            write_table(name, "Complete-test accuracy after labeled coefficient selection (\\%). Each method separately maximizes "+criterion+" accuracy on the same 256 training examples per task, using TA coefficients 0.1--1.0 in steps of 0.1. Coefficients are frozen before test evaluation. AGA processes the protected expert first; both task updates use the selected coefficient.",
                "tab:"+name.replace("_","-"), ["Task","Defense",r"$\lambda$","Target","Companion"], groups)
            exported_tables.append(name)
        readout_path = selection_root / "readout_analysis.json"
        readout = json.loads(readout_path.read_text())
        assert readout["numpy_formula_check"]
        old = json.loads(original_path.read_text())["new_batches"]["SVHN"]["evaluation"]["full"]["rows"]
        fixed = next(r for r in old if r["defense"] == "Defense1" and r["merger"] == "TA"
                     and r["attack_scope"] == "Protected expert" and r["dataset"] == "SVHN")
        selected = next(r for r in selection["choices"] if r["task"] == "SVHN"
                        and r["defense"] == "NBP_SVHN" and r["objective"] == "mean")
        records = []
        for coefficient, measured in [(0.5, fixed), (selected["coefficient"],
                test_rows["SVHN", selected["coefficient"], "NBP_SVHN", "SVHN"])]:
            r = next(r for r in readout["rows"] if r["task"] == "SVHN" and r["protected"]
                     and r["coefficient"] == coefficient)
            counts = measured["predicted_class_counts"]
            assert sum(counts) == measured["n"] == 26032
            dominant = max(range(len(counts)), key=counts.__getitem__)
            records.append([f"{coefficient:.1f}", f"{r['embedding_shift_norm']:.2f}",
                f"{r['minimum_margin_lower_bound']:.2f}", str(dominant),
                f"{100*counts[dominant]/measured['n']:.2f}", f"{100*measured['correct']/measured['n']:.2f}"])
        write_table("coefficient_readout", "SVHN after AGA and TA, using the same protected checkpoint and all 26,032 test images. Coefficient 0.7 is selected on training examples. $L_{\\min}$ is the minimum pairwise margin bound; Class and Share give the most frequent prediction and its percentage. Acc. is test accuracy (\\%).",
            "tab:coefficient-readout", [r"$\lambda$", r"$\|\Delta t\|_2$", r"$L_{\min}$", "Class", "Share", "Acc."],
            [("", records)], tabcolsep=3)
        source["input_sha256"][str(readout_path.relative_to(ROOT))] = hashlib.sha256(readout_path.read_bytes()).hexdigest()
        exported_tables.append("coefficient_readout")
    training_groups = []
    for method, folder in [("Trap2", "trap2_baseline"), ("MergeGuard", "mergeguard_baseline")]:
        path = ROOT / "results/phase3" / folder / "summary.json"
        if not path.exists():
            continue
        training = json.loads(path.read_text())
        assert training["status"] == "completed_independently_verified"
        for task in ["MNIST", "SVHN", "DTD"]:
            records = []
            for defense in ["Unprotected", method]:
                title = "CE" if defense == "Unprotected" else (r"Trap$^2$" if defense == "Trap2" else "MergeGuard")
                values = []
                for merger, scope in [("Standalone", "None"), ("TA", "None"), ("TA", "Protected expert")]:
                    row, = [r for r in training["rows"] if r["target"] == task
                            and r["defense"] == defense and r["role"] == "target"
                            and r["merger"] == merger and r["attack_scope"] == scope]
                    values.append(f"{100 * row['correct'] / row['n']:.2f}")
                records.append([title] + values)
            training_groups.append((task, records))
        source["input_sha256"][str(path.relative_to(ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    if training_groups:
        write_table("training_baseline", "Training-based comparisons with a separate matched CE control for each method. Scores are full-test target accuracy (\\%); TA uses 0.5 and AGA aligns the target. Trap$^2$ uses author training code; MergeGuard is our paper-based B/32 reproduction. Training details and differences from Table~\\ref{tab:main} are stated in the text.",
                    "tab:training-baseline", ["Task", "Training", "Alone", "TA", "AGA+TA"],
                    training_groups, tabcolsep=3)
        exported_tables.append("training_baseline")
    (OUT / "sources.json").write_text(json.dumps({"input_sha256": source["input_sha256"],
        "eight_expert_protected_task_cells": len(eight),
        "tables": exported_tables}, indent=2) + "\n")
    print(f"Generated {len(exported_tables)} manuscript table files from recorded results.")
