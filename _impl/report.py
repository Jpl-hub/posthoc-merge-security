"""Paper MD/TeX tables and figures from independently verified stored records."""

import hashlib
import json
import math
from pathlib import Path

def defense_comparison(directory):
    """Join current, already evaluated defenses; no model loading or new experiments."""
    out = Path(directory)
    out.mkdir(parents=True, exist_ok=True)
    sources, tables = {}, {}

    def read(path):
        data = Path(path).read_bytes()
        sources[str(path)] = hashlib.sha256(data).hexdigest()
        return json.loads(data)

    def one(rows, **where):
        matches = [r for r in rows if all(r.get(k) == v for k, v in where.items())]
        assert len(matches) == 1, (where, len(matches))
        return matches[0]

    def accuracy(row):
        value = 100 * row["correct"] / row["n"]
        assert abs(value - row["accuracy_percent"]) < 1e-8
        return f"{value:.2f}"

    def write(name, title, headers, rows, note, colspec):
        tables[name] = dict(headers=headers, rows=rows, note=note)
        md = ["# " + title, "", note, "", "| " + " | ".join(headers) + " |",
              "| " + " | ".join(["---"] * len(headers)) + " |"]
        md += ["| " + " | ".join(row) + " |" for row in rows]
        tex = [r"\begin{table*}[t]", r"\centering\small", r"\setlength{\tabcolsep}{4pt}",
               r"\caption{" + title + r". Accuracy (\%).}", r"\label{tab:" + name + "}",
               r"\begin{tabular}{@{}" + colspec + "@{}}", r"\toprule"]
        if name in ["attacks", "partners"]:
            first = 4 if name == "attacks" else 3
            tex += [" & " * (first - 1) + r"\multicolumn{5}{c}{Merged, no real attack data} & Extra data \\",
                    rf"\cmidrule(lr){{{first}-{first + 4}}}\cmidrule(lr){{{first + 5}-{first + 5}}}"]
        tex += [" & ".join(headers) + r" \\", r"\midrule"]
        for index, row in enumerate(rows):
            if index and row[0] != rows[index - 1][0]:
                tex.append(r"\midrule")
            elif name == "eight_expert" and index % 4 == 0 and index:
                tex.append(r"\addlinespace")
            tex.append(" & ".join(row) + r" \\")
        tex += [r"\bottomrule", r"\end{tabular}", r"\par\smallskip",
                r"\begin{minipage}{\textwidth}\footnotesize " + note + r"\end{minipage}",
                r"\end{table*}"]
        (out / (name + ".md")).write_text("\n".join(md) + "\n")
        (out / (name + ".tex")).write_text("\n".join(tex) + "\n")

    tasks = ["MNIST", "SVHN", "DTD"]
    methods = ["Unprotected", "PaRaMS", "Defense 1", "Defense 2"]
    root = Path("results/params_reference/eight_expert")
    cfg = read(root / "protocol.json")
    inputs = [(str(root / "summary.json"), "independent_statistics"),
              ("results/phase3/bias_weight_only/eight_expert/summary.json", "paired_summary"),
              ("results/attack_generalization_20261028/eight_expert/full/results.json", None),
              ("results/attack_generalization_20261028/eight_expert_extra/full/results.json", None),
              ("results/attack_generalization_20261028/cars/summary.json", "independent_statistics"),
              ("results/attack_generalization_20261028/sun397/summary.json", "independent_statistics")]
    indexed, parameters = {}, {}
    for path, field in inputs:
        data = read(path)
        if field:
            assert data["status"] == "completed_independently_verified"
        records = data[field] if field else data
        for i, row in enumerate(records):
            if field:
                target, defense, scope, dataset = row["key"].split("/")
                row = dict(row, target=target, defense=defense, attack_scope=scope, dataset=dataset,
                           model_parameter_sha256=data["parameters"]["/".join([target, defense, scope])])
            else:
                assert row["participants"] == 8 and row["coefficient"] == .2
                assert sum(c["correct"] for c in row["per_class"]) == row["correct"]
            method = next((m for prefix, m in [("Defense1", "Defense 1"), ("Defense2", "Defense 2"),
                                               ("PaRaMS", "PaRaMS")] if row["defense"].startswith(prefix)), row["defense"])
            key = (row["target"], method, row["attack_scope"], row["dataset"])
            assert row["n"] == cfg["test_sizes"][row["dataset"]]
            value = (row["correct"], row["n"], row["model_parameter_sha256"])
            if key in indexed:
                assert indexed[key][0] == value, key
            else:
                indexed[key] = (value, path, i)
            assert parameters.setdefault(key[:3], value[2]) == value[2], key
    eight = []
    selected = []
    for target in tasks:
        for scope, label in [("None", "TA"), ("Protected expert", "AGA")]:
            for method in methods:
                row = [target, method, label]
                for dataset in cfg["evaluate"]:
                    value, path, i = indexed[(target, method, scope, dataset)]
                    row.append(f"{100 * value[0] / value[1]:.2f}")
                    selected.append(dict(target=target, method=method, scope=scope, dataset=dataset,
                                         correct=value[0], n=value[1], parameter_sha256=value[2], source=path, row=i))
                eight.append(row)
    write("eight_expert", "Eight-expert comparison", ["Target", "Defense", "Processing"] + cfg["evaluate"], eight,
          "Eight experts, three protected tasks, seven complete test datasets; TA coefficient 0.2. AGA processes only the protected expert. EuroSAT is excluded; no eight-task mean. Partner losses remain in the table.", "lllrrrrrrr")

    campaign = read("results/attack_generalization_20261028/results.json")["table_rows"]
    bias = read("results/phase3/bias_weight_only/other_tasks/results.json")
    calibrations = {m: read(p)["rows"] for m, p in [
        ("Defense 1", "results/phase3/bias_weight_only/attacks/calibration.json"),
        ("PaRaMS", "results/params_reference/other_tasks/calibration.json")]}
    matching = {m: read(p)["rows"] for m, p in [
        ("PaRaMS", "results/params_reference/other_tasks/matching.json"),
        ("Defense 1", "results/phase3/bias_weight_only/attacks/matching.json")]}
    students = read("results/params_reference/other_tasks/distillation.json")["rows"]
    original = read("results/phase3/results.json")
    baselines = {"MNIST": original}
    for index, task in enumerate(["DTD", "SVHN"]):
        baselines[task] = read(f"results/params_reference/other_tasks/evaluation/full/{index:02d}_{task}_MNIST/results.json")
    target_rows, partner_rows = [], []
    for task in tasks:
        partner = "SVHN" if task == "MNIST" else "MNIST"
        for method in methods:
            columns = [[], []]
            for role, dataset in enumerate([task, partner]):
                if method in ["Unprotected", "Defense 2"]:
                    def cell(pipeline, scope):
                        hits = [r for r in campaign if r[:3] == [task, partner, method] and r[3:5] == [pipeline, scope]]
                        assert len(hits) == 1, (task, method, pipeline, scope)
                        return hits[0]
                    ta = cell("TA / AGA", "None")
                    columns[role] = [ta[5] if role == 0 else "--", ta[6 + role],
                                     cell("TA / AGA", "Protected expert")[6 + role],
                                     cell("64-noise calibration", "None")[6 + role],
                                     cell("64-noise calibration", "Protected expert")[6 + role],
                                     cell("MLP matching", "None")[6 + role]]
                else:
                    if method == "Defense 1":
                        b = one(bias["combined_target_table" if role == 0 else "combined_partner_table"], target=task, method="Defense1")
                        values = [f"{100 * c / n:.2f}" for c, n in b["counts"]]
                        columns[role] = values[:3] if role == 0 else ["--"] + values[:2]
                    else:
                        records = [r for r in baselines[task] if r["target"] == task and r["partner"] == partner]
                        name = "PaRaMS" if task == "MNIST" else "PaRaMS_" + task
                        columns[role] = [accuracy(one(records, defense=name, merger="Standalone", attack_scope="None", dataset=task)) if role == 0 else "--"]
                        columns[role] += [accuracy(one(records, defense=name, merger="TA", attack_scope=s, dataset=dataset)) for s in ["None", "Protected expert"]]
                    name = "Defense1" if method == "Defense 1" else ("PaRaMS" if task == "MNIST" else "PaRaMS_" + task)
                    columns[role] += [accuracy(one(calibrations[method], key="/".join([task, name, "TA", s, dataset]))) for s in ["None", "Protected expert"]]
                    columns[role].append(accuracy(one(matching[method], target=task, defense="Defense1" if method == "Defense 1" else "PaRaMS", merger="TA", dataset=dataset)))
                teacher = "Defense 2" if method == "Defense 2" else "Unprotected / Defense 1 / PaRaMS"
                student = one(students, task=task, teacher=teacher)
                columns[role].append(f"{student['TA'] if role == 0 else student['partner']['None']:.2f}")
            target_rows.append([task, method] + columns[0])
            partner_rows.append([task + "/" + partner, method] + columns[1])
    headers = ["Task", "Defense", "Standalone", "TA", "AGA", "Calib.", "AGA+calib.", "MLP match", "Distill."]
    note = ("Two experts, TA coefficient 0.5. TA uses no recovery; AGA processes the protected expert. "
            "Calib. uses 64 fixed random images for output calibration. MLP matching uses public weights. "
            "Distill. denotes hard-label student training on extra unlabeled images; ProDistill is reported separately. "
            "Defense 1 and PaRaMS share the unprotected student because their teacher labels coincide.")
    write("attacks", "Recovery attacks on the protected task", headers, target_rows,
          note + " Higher standalone accuracy and lower merged accuracy favor protection.", "llrrrrrrr")
    write("partners", "Partner-task accuracy under the same attacks", ["Target/partner"] + headers[1:2] + headers[3:],
          [row[:2] + row[3:] for row in partner_rows], note + " These are partner accuracies, not protection scores.", "llrrrrrr")
    progressive = read("results/attack_generalization_20261028/prodistill/summary.json")
    assert progressive["status"] == "completed_independently_verified"
    distilled = []
    for task in tasks:
        partner = "SVHN" if task == "MNIST" else "MNIST"
        for method in methods:
            name = method.replace(" ", "")
            rows = [one(progressive["rows"], target=task, defense=name, dataset=t, stage="full")
                    for t in [task, partner]]
            distilled.append([task, method, accuracy(rows[0]), partner, accuracy(rows[1])])
    write("feature_distillation", "ProDistill recovery comparison",
          ["Target", "Defense", "Target accuracy", "Partner", "Partner accuracy"], distilled,
          "Two-expert adaptation of the fixed author coefficient kernel. The attacker uses 64 unlabeled training images per task, task identity, and 100 epochs per block. Initial coefficient 0.3; no AGA preprocessing. Final LayerNorm and projection are not optimized. Defense 1 reuses identical unprotected blocks. Lower target accuracy favors protection; partner accuracy is reported separately. Standalone accuracy is reported in the main attack table.",
          "llrlr")
    updated = read("results/data_free_compensation/finite_classification_result.json")
    assert updated["prodistill"]["status"] == "completed_independently_verified"
    versions = [[], []]
    for task in tasks:
        partner = "SVHN" if task == "MNIST" else "MNIST"
        record = updated["tasks"][task]
        evaluation = record["evaluation"]
        defense = "FiniteClassification_" + task
        for role, dataset in enumerate([task, partner]):
            old = next(
                row for row in (target_rows if role == 0 else partner_rows)
                if row[0] == (task if role == 0 else task + "/" + partner)
                and row[1] == "Defense 2")
            distilled_old = one(progressive["rows"], target=task, defense="Defense2",
                                dataset=dataset, stage="full")
            versions[role].append([task, "Earlier"] +
                                  ([old[2]] if role == 0 else []) +
                                  [old[i] for i in [3, 4, 6, 7]] +
                                  [accuracy(distilled_old)])
            rows = evaluation["raw_rows"]
            values = [accuracy(one(rows, defense=defense, merger="TA", dataset=dataset,
                                   attack_scope=scope)) for scope in ["None", "Protected expert"]]
            calibrated = one(evaluation["calibrated_rows"],
                             key="/".join([task, defense, "TA", "Protected expert", dataset]))
            match = one(record["recovery"]["matching"]["rows"], defense=defense + "_matched",
                        merger="TA", attack_scope="None", dataset=dataset)
            distilled_new = one(updated["prodistill"]["rows"], target=task, defense="Defense2",
                                dataset=dataset, stage="full")
            native = [accuracy(one(rows, defense=defense, merger="Standalone",
                                   attack_scope="None", dataset=task))] if role == 0 else []
            versions[role].append([task, "Updated"] + native + values +
                                  [accuracy(calibrated), f"{100 * match['correct'] / match['n']:.2f}",
                                   accuracy(distilled_new)])
    note = ("The earlier and updated Defense 2 releases use distinct frozen construction rules; "
            "each row uses one release throughout. TA coefficient 0.5; AGA processes the protected expert. "
            "Calibration uses 64 random images. ProDistill uses 64 unlabeled training images per task "
            "and coefficient initialization 0.3, with no AGA preprocessing. "
            "MNIST is paired with SVHN; SVHN and DTD are paired with MNIST. "
            "Known-module replacement is reported separately; neither version resists it.")
    write("defense2_versions", "Defense 2 versions on the protected task",
          ["Task", "Version", "Standalone", "TA", "AGA", "AGA+calib.", "MLP match", "ProDistill"],
          versions[0], note, "llrrrrrr")
    write("defense2_version_partners", "Partner accuracy for the same Defense 2 versions",
          ["Target", "Version", "TA", "AGA", "AGA+calib.", "MLP match", "ProDistill"],
          versions[1], note, "llrrrrr")
    finite_eight = read("results/data_free_compensation/finite_eight/summary.json")
    assert finite_eight["status"] == "completed_independently_verified"
    assert finite_eight["evaluated_datasets"] == cfg["evaluate"]
    eight_versions = []
    for task in tasks:
        for scope, label in [("None", "TA"), ("Protected expert", "AGA")]:
            for method, version in [("Unprotected", "Unprotected"), ("Defense 2", "Earlier")]:
                old = next(r for r in eight if r[:3] == [task, method, label])
                eight_versions.append([task, version, label] + old[3:])
            row = [task, "Updated", label]
            for dataset in cfg["evaluate"]:
                clean = one(finite_eight["independent_statistics"],
                            key="/".join([task, "Unprotected", scope, dataset]))
                previous = indexed[(task, "Unprotected", scope, dataset)][0]
                assert (clean["correct"], clean["n"], clean["model_parameter_sha256"]) == previous
                value = one(finite_eight["independent_statistics"],
                            key="/".join([task, "Defense2_" + task, scope, dataset]))
                assert value["model_parameter_sha256"] == finite_eight["parameters"][
                    "/".join([task, "Defense2_" + task, scope])]
                row.append(accuracy(value))
            eight_versions.append(row)
    write("defense2_eight_versions", "Defense 2 versions in eight-expert merging",
          ["Target", "Version", "Processing"] + cfg["evaluate"], eight_versions,
          "The same earlier and updated releases as the two-expert version tables. "
          "Eight experts, TA coefficient 0.2; AGA processes only the protected expert. "
          "Seven complete test datasets, including all evaluated partner tasks. "
          "Unprotected weights and scores match between versions. EuroSAT is excluded; no eight-task mean. "
          "Lower protected-task accuracy favors protection; losses on other tasks are reported without averaging.",
          "lllrrrrrrr")
    calibration = read("results/data_free_compensation/finite_eight/calibration.json")
    assert calibration["status"] == "completed_independently_verified"
    calibrated_rows, target_rows = [], []
    for task in tasks:
        for scope, label in [("None", "TA"), ("Protected expert", "AGA")]:
            target_row = [task, label]
            for name, method in [("Unprotected", "Unprotected"), ("Defense2_" + task, "Defense 2")]:
                row = [task, method, label]
                for dataset in cfg["evaluate"]:
                    key = "/".join([task, name, scope, dataset])
                    value = one(calibration["rows"], key=key)
                    row.append(accuracy(value))
                    if task == dataset:
                        before = one(finite_eight["independent_statistics"], key=key)
                        target_row.extend([accuracy(before), accuracy(value)])
                calibrated_rows.append(row)
            target_rows.append(target_row)
    note = ("Same frozen updated Defense 2 releases; eight experts, TA coefficient 0.2. "
            "AGA processes the protected expert. Each merged model is calibrated independently "
            "using 64 fixed random images and its public classifier, without real images, labels "
            "or changed-module knowledge. Full archived scores are reused. "
            "Seven complete datasets; EuroSAT is excluded, and no eight-task mean is reported.")
    write("defense2_eight_calibration", "Output calibration in eight-expert merging",
          ["Target", "Defense", "Processing"] + cfg["evaluate"], calibrated_rows, note, "lllrrrrrrr")
    write("defense2_eight_calibration_target", "Protected-task accuracy before and after output calibration",
          ["Task", "Processing", "Unprotected", "+calib.", "Defense 2", "+calib."],
          target_rows, note, "llrrrr")
    baseline_calibration = {
        method: read(f"results/comparison/eight_calibration/{name}.json")
        for method, name in [("Defense 1", "Defense1"), ("PaRaMS", "PaRaMS")]
    }
    for result in baseline_calibration.values():
        assert result["status"] == "completed_independently_verified"
        for row in result["rows"]:
            if row["key"].split("/")[1] == "Unprotected":
                clean = one(calibration["rows"], key=row["key"])
                assert (row["correct"], row["n"]) == (clean["correct"], clean["n"])
    comparison, all_tasks = [], []
    for task in tasks:
        for method in methods:
            if method == "Defense 2":
                native = one(updated["tasks"][task]["evaluation"]["raw_rows"],
                             defense="FiniteClassification_" + task, merger="Standalone",
                             attack_scope="None", dataset=task)
                standalone = accuracy(native)
            else:
                standalone = next(r[2] for r in tables["attacks"]["rows"]
                                  if r[:2] == [task, method])
            row = [task, method, standalone]
            for scope, label in [("None", "TA"), ("Protected expert", "AGA")]:
                result = baseline_calibration.get(method, calibration)
                name = {"Unprotected": "Unprotected", "Defense 2": "Defense2_" + task,
                        "Defense 1": "Defense1_" + task,
                        "PaRaMS": "PaRaMS" if task == "MNIST" else "PaRaMS_" + task}[method]
                corrected = [one(result["rows"], key="/".join([task, name, scope, dataset]))
                             for dataset in cfg["evaluate"]]
                if method == "Defense 2":
                    raw = one(finite_eight["independent_statistics"],
                              key="/".join([task, name, scope, task]))
                    raw_accuracy = accuracy(raw)
                else:
                    counts = indexed[(task, method, scope, task)][0]
                    raw_accuracy = f"{100 * counts[0] / counts[1]:.2f}"
                row += [raw_accuracy, accuracy(corrected[cfg["evaluate"].index(task)])]
                all_tasks.append([task, method, label] + [accuracy(r) for r in corrected])
            comparison.append(row)
    note = ("Eight experts, TA coefficient 0.2; AGA processes the protected expert. "
            "Each merged model uses the same 64 fixed random inputs for output calibration. "
            "No real attack images, clean target expert, or changed-module knowledge is used. "
            "Defense 2 denotes the updated release throughout; standalone accuracy is measured before merging. "
            "Seven complete test datasets; EuroSAT is excluded, and no eight-task mean is reported.")
    write("eight_calibration_target", "Matched recovery comparison in eight-expert merging",
          ["Task", "Defense", "Standalone", "TA", "+calib.", "AGA", "+calib."],
          comparison, note, "llrrrrr")
    write("eight_calibration", "All evaluated tasks after output calibration",
          ["Target", "Defense", "Processing"] + cfg["evaluate"], all_tasks, note, "lllrrrrrrr")
    mergers = read("results/data_free_compensation/finite_mergers_result.json")
    assert mergers["status"] == "completed_independently_verified"
    merger_rows = []
    for task in tasks:
        records = next(r["rows"] for r in mergers["records"]
                       if r["stage"] == "full" and r["rows"][0]["target"] == task)
        partner = "SVHN" if task == "MNIST" else "MNIST"
        for method, name in [("Unprotected", "Unprotected"), ("Defense 2", "Defense2_" + task)]:
            native = one(records, defense=name, merger="Standalone", dataset=task)
            row = [task, method, accuracy(native)]
            for merger in ["TIES", "DARE+TA"]:
                row += [accuracy(one(records, defense=name, merger=merger, dataset=t))
                        for t in [task, partner]]
            merger_rows.append(row)
    write("defense2_mergers", "Updated Defense 2 with TIES and DARE",
          ["Task", "Defense", "Standalone", "TIES target", "TIES partner", "DARE target", "DARE partner"],
          merger_rows,
          "Same updated releases as the TA and recovery tables. Two experts; MNIST is paired with SVHN, "
          "SVHN and DTD with MNIST. Fixed author TIES kernel: top 20 percent, coefficient 1.0. "
          "DARE+TA: drop rate 0.2, seed 42, coefficient 0.5. No AGA preprocessing or parameter search. "
          "Complete test sets; matching unprotected scores and standalone scores reused after parameter-hash checks. "
          "TIES and DARE are merging algorithms, not additional recovery attacks.", "llrrrrr")
    projection = read("results/data_free_compensation/finite_projection_ablation_result.json")
    assert projection["status"] == "completed_independently_verified"
    projection_rows = [[], []]
    for task in tasks:
        partner = "SVHN" if task == "MNIST" else "MNIST"
        for role, dataset in enumerate([task, partner]):
            for label, field in [("Retained", "main"), ("Removed", "without")]:
                row = [task, label]
                settings = ([("raw", "Standalone", "None")] if role == 0 else [])
                settings += [("raw", "TA", scope) for scope in ["None", "Protected expert", "All experts"]]
                settings += [("calibrated", "TA", scope) for scope in ["None", "Protected expert", "All experts"]]
                for stage, merger, scope in settings:
                    value = one(projection["main_comparisons"], target=task, dataset=dataset,
                                stage=stage, merger=merger, attack_scope=scope)
                    correct, n = value[field + "_correct"], value["n"]
                    assert abs(100 * correct / n - value[field + "_accuracy_percent"]) < 1e-8
                    row.append(f"{100 * correct / n:.2f}")
                projection_rows[role].append(row)
    note = ("Fixed component comparison of the current Defense 2. Only the two task-SVD constraints on A "
            "are removed; synthetic inputs, bias compensation, functional bound, optimizer and stopping rules "
            "are unchanged. Task-shaped synthetic probes are retained. Two experts, TA coefficient 0.5; "
            "AGA-t processes the protected expert, AGA-b both experts. Calibration uses the same 64 random inputs. "
            "All three controls are frozen before real-image evaluation. MNIST and SVHN controls stop with "
            "nonzero synthetic classification violations; DTD reaches zero. All controls are reported, "
            "including these failures. No task-wise replacement of the main method or proof of improved performance on other datasets.")
    headers = ["Task", "Projection", "Standalone", "TA", "AGA-t", "AGA-b",
               "Calib.", "AGA-t+calib.", "AGA-b+calib."]
    write("defense2_projection", "Effect of task-projection constraints", headers,
          projection_rows[0], note, "llrrrrrrr")
    write("defense2_projection_partners", "Partner accuracy in the task-projection comparison",
          headers[:2] + headers[3:], projection_rows[1],
          note + " MNIST is paired with SVHN; SVHN and DTD with MNIST.", "llrrrrrr")
    recovery = read("results/data_free_compensation/projection_recovery_result.json")
    assert recovery["status"] == "completed_independently_verified"
    recovery_tables = [[], []]
    for task in tasks:
        partner = "SVHN" if task == "MNIST" else "MNIST"
        native = one(projection["main_comparisons"], target=task, merger="Standalone")
        for role, dataset in enumerate([task, partner]):
            for label, field in [("Unprotected", None), ("Retained", "before"), ("Removed", "after")]:
                row = [task, label]
                if role == 0:
                    clean = projection["tasks"][task]["native"]["rows"][0]
                    value = 100 * clean["before_correct"] / clean["n"] if field is None else native[
                        "main_accuracy_percent" if field == "before" else "without_accuracy_percent"]
                    row.append(f"{value:.2f}")
                for method in ["matching", "replacement", "prodistill"]:
                    if field is None:
                        group = one(recovery["records"], target=task, method=method, stage="full")
                        value = one(group["rows"], defense="Unprotected", dataset=dataset,
                                    merger="ProDistill" if method == "prodistill" else "TA")
                        row.append(accuracy(value))
                    else:
                        value = one(recovery["main_comparisons"], target=task, dataset=dataset,
                                    method=method, variant="merged" if method == "prodistill" else "TA")
                        row.append(f"{value['before_accuracy_percent' if field == 'before' else 'accuracy_percent']:.2f}")
                recovery_tables[role].append(row)
    recovery_note = ("The same fixed projection controls as the preceding ablation. MLP matching uses public "
                     "weights without images. B reset replaces the last MLP output matrix and bias with the "
                     "public base, requiring knowledge of the modified location; its unprotected control receives "
                     "the same replacement. Both use TA coefficient 0.5. ProDistill uses 64 additional unlabeled "
                     "training images per task, 100 epochs per block and coefficient initialization 0.3; no AGA "
                     "preprocessing. Data, indices and training steps match the retained-projection comparison. "
                     "These recovery conditions are separate, not one common attacker. Standalone columns precede recovery. "
                     "MNIST and SVHN synthetic source-condition failures remain reported; no new defense updates.")
    write("defense2_projection_recovery", "Recovery of the fixed task-projection controls",
          ["Task", "Projection", "Standalone", "MLP match", "B reset", "ProDistill"],
          recovery_tables[0], recovery_note, "llrrrr")
    write("defense2_projection_recovery_partners", "Partner accuracy after recovery of projection controls",
          ["Target", "Projection", "MLP match", "B reset", "ProDistill"],
          recovery_tables[1], recovery_note + " MNIST is paired with SVHN; SVHN and DTD with MNIST.", "llrrr")
    removed = read("results/data_free_compensation/projection_eight/calibration.json")
    paired = read("results/data_free_compensation/projection_eight/paired.json")
    assert removed["independent_validation"]["comparisons"] == 126
    assert len(paired["paired_comparisons"]) == 84
    summary_rows, dataset_rows = [], []
    for task, label, standalone, *_ in recovery_tables[0]:
        scores = removed if label == "Removed" else calibration
        model = "Unprotected" if label == "Unprotected" else "Defense2_" + task
        summary = [task, label, standalone]
        for scope, processing in [("None", "TA"), ("Protected expert", "AGA")]:
            raw, corrected = [], []
            for dataset in cfg["evaluate"]:
                key = "/".join([task, model, scope, dataset])
                row = one(scores["rows"], key=key)
                before = row["correct"] + row["paired"]["lost"] - row["paired"]["gained"]
                raw.append(f"{100 * before / row['n']:.2f}")
                corrected.append(accuracy(row))
                if dataset == task:
                    summary.extend([raw[-1], corrected[-1]])
            if label == "Removed":
                dataset_rows.append([task, processing] + raw)
                dataset_rows.append([task, processing + " + calibration"] + corrected)
        summary_rows.append(summary)
    eight_note = ("Eight experts participate in TA with coefficient 0.2. AGA processes the protected "
                  "expert. Output calibration uses 64 random images. The removed condition drops the "
                  "two projections used to construct A and keeps the task-specific synthetic inputs. "
                  "Seven complete test sets are available. The original EuroSAT split is unverified.")
    write("defense2_projection_eight", "Task projection in eight-expert merging",
          ["Task", "Projection", "Standalone", "TA", "+calib.", "AGA", "+calib."],
          summary_rows, eight_note, "llrrrrr")
    write("defense2_projection_eight_datasets", "Seven test sets with the task projection removed",
          ["Protected task", "Processing"] + cfg["evaluate"], dataset_rows, eight_note, "llrrrrrrr")
    hard_labels = read("results/data_free_compensation/hard_label/result.json")
    assert hard_labels["status"] == "completed_independently_verified"
    student_rows = []
    for task in tasks:
        partner = "SVHN" if task == "MNIST" else "MNIST"
        for model in ["Original expert", "Student from unprotected",
                      "Student from earlier Defense 2", "Student from updated Defense 2"]:
            values = [accuracy(one(hard_labels["rows"], task=task, model=model,
                                   merger=merger, dataset=dataset))
                      for merger, dataset in [("Standalone", task), ("TA", task), ("TA", partner)]]
            student_rows.append([task, model] + values)
    write("defense2_hard_label", "Hard-label student recovery",
          ["Task", "Model", "Standalone", "Fused target", "Fused partner"], student_rows,
          "TA coefficient 0.5. Additional unlabeled training images: MNIST and SVHN 2048 each, "
          "DTD train1 1880. The attacker trains a public-base student for 50 epochs using teacher "
          "class predictions, without ground-truth labels. The fixed images, seed, initialization, "
          "optimizer and training budget match the historical comparison. Historical students used "
          "an RTX 3090; new students used an RTX 4090 D. Small differences are not isolated causal "
          "effects of the teacher labels. MNIST reuses the same student because all 2048 teacher "
          "labels coincide. Standalone reports the evaluated student or expert. Defender weights "
          "are unchanged; additional real images are used only by the attacker.", "llrrr")
    components = updated["prodistill"].get("components_full")
    if components:
        assert components["status"] == "completed_independently_verified"
        component_rows = []
        for task in tasks:
            partner = "SVHN" if task == "MNIST" else "MNIST"
            for dataset in [task, partner]:
                component_rows.append([task, dataset] + [
                    accuracy(one(components["rows"], target=task, dataset=dataset, variant=variant))
                    for variant in ["00", "10", "01", "11"]])
        write("prodistill_components", "Final MLP components after ProDistill",
              ["Target", "Dataset", "Initial A/B", "Trained A", "Trained B", "Both trained"],
              component_rows,
              "Complete test accuracy. Other parameters remain fixed at the trained ProDistill model. "
              "A and B include their respective biases. Initial values use the same 0.3 initialization; "
              "trained values come from one completed joint coefficient fit. "
              "These are fixed component substitutions, not separately trained A-only or B-only attacks. "
              "Both-trained scores are reused from the same frozen model. No new optimization.", "llrrrr")
    components = read("results/data_free_compensation/finite_components_result.json")
    assert components["status"] == "completed_independently_verified"
    component_rows, component_partners = [], []
    for task in tasks:
        partner = "SVHN" if task == "MNIST" else "MNIST"
        for label, name in [("Unprotected", "Unprotected"), ("A only", "A_only_" + task),
                            ("B only", "B+b_only_" + task), ("Gaussian", "Random_noise_" + task),
                            ("A+B", "Defense2_" + task)]:
            def component_score(method, scope, dataset):
                return accuracy(one(components["records"], target=task,
                                    key="/".join([name, method, scope, dataset])))
            native = component_score("Standalone", "None", task)
            scopes = ["None", "Protected expert", "All experts"]
            component_rows.append([task, label, native] + [
                component_score("TA", scope, task) for scope in scopes])
            component_partners.append([task, partner, label] + [
                component_score("TA", scope, partner) for scope in scopes])
    note = ("Same current Defense 2 releases. A and B include their respective biases. "
            "A-only and B-only retain the selected changes without refitting. Gaussian perturbations "
            "match each of the four tensor-change norms, with one fixed seed. TA coefficient 0.5; "
            "AGA-T processes the protected expert and AGA-B both experts. "
            "Low merged accuracy is useful only alongside preserved standalone accuracy. "
            "No output calibration or distillation is added to these component comparisons.")
    write("defense2_components", "Components of the current Defense 2",
          ["Task", "Variant", "Standalone", "TA", "AGA-T", "AGA-B"],
          component_rows, note, "llrrrr")
    write("defense2_component_partners", "Partner accuracy in the component comparison",
          ["Target", "Partner", "Variant", "TA", "AGA-T", "AGA-B"],
          component_partners, note, "lllrrr")
    views = read("results/data_free_compensation/native_view_response_result.json")
    view_rows = []
    for task in views["evaluated"]:
        for dataset in [task, "MNIST"]:
            for label, field in [("Previous", "before_accuracy_percent"), ("Two views", "accuracy_percent")]:
                row = [task, dataset, label]
                for method, scope, stage in [("Standalone", "None", "raw"), ("TA", "None", "raw"),
                                              ("TA", "Protected expert", "raw"), ("TA", "All experts", "raw"),
                                              ("TA", "Protected expert", "calibrated")]:
                    if method == "Standalone" and task != dataset:
                        row.append("--")
                        continue
                    prefix = task + "/" if stage == "calibrated" else ""
                    key = prefix + "/".join(["NativeResponse_" + task, method, scope, dataset])
                    row.append(f"{one(views['vs_main'], key=key, stage=stage)[field]:.2f}")
                view_rows.append(row)
    write("defense2_native_views", "Matching native responses to complementary synthetic views",
          ["Target", "Dataset", "Variant", "Standalone", "TA", "AGA-T", "AGA-B", "AGA-T+cal."],
          view_rows, "Two-expert TA at 0.5. AGA-T processes the target expert and AGA-B processes both. "
          "Calibration uses 64 random images. The MNIST construction failed one source decision among "
          "the 18 required conditions and stopped before natural evaluation. All completed tasks are shown.",
          "lllrrrrr")
    (out / "sources.json").write_text(json.dumps(dict(input_sha256=sources, eight_expert_rows=selected,
        tables=tables, new_evaluations=0, report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2) + "\n")


def attack_comparison(directory):
    """Collect verified attack rows; MD and TeX use the same table cells."""
    root = Path(directory)
    result = json.loads((root / "results.json").read_text())
    protocol = json.loads((root / "protocol.json").read_text())
    rows = list(result["reused_rows"])
    batches = []
    campaign = Path(protocol["output"]).parent
    evaluations = [(protocol, "Public replacement")]
    optional = [campaign / "baselines/evaluation_protocol.json", root / "mergers_protocol.json"]
    optional += [
        campaign / "matching" / (task + "_protocol.json")
        for task in ["MNIST", "SVHN", "DTD"]
    ]
    for path in optional:
        if path.exists():
            operation = "TA / AGA"
            if path.name == "mergers_protocol.json":
                operation = "Fusion strategies"
            elif "matching" in path.parts:
                operation = "MLP matching"
            evaluations.append((json.loads(path.read_text()), operation))
    for config, operation in evaluations:
        for index, case in enumerate(config["cases"]):
            folder = Path(config["output"]) / "full" / f"{index:02d}_{case['target']}_{case['partner']}"
            check = folder / "independent_validation.json"
            if not check.exists():
                continue
            validation = json.loads(check.read_text())
            execution = json.loads((folder / "execution.json").read_text())
            assert validation["status"] == "passed"
            assert validation["raw_sha256"] == execution["predictions_sha256"]
            pipeline = operation
            if "public_replacement" in case:
                pipeline += " " + case["public_replacement"]["side"]
            new = json.loads((folder / "results.json").read_text())
            for row in new:
                assert sum(c["correct"] for c in row["per_class"]) == row["correct"]
                rows.append(dict(row, pipeline=row["merger"] if operation == "Fusion strategies" else pipeline))
            batches.append(dict(case=index, directory=str(folder), execution=execution, validation=validation))
    calibration = campaign / "DTD_calibration/results.json"
    if calibration.exists():
        calibrated = json.loads(calibration.read_text())
        assert calibrated["status"] == "completed" and calibrated["maximum_numpy_prior_error"] < 1e-10
        for row in calibrated["rows"]:
            target, name, merger, scope, task = row["key"].split("/")
            rows.append(dict(row, target=target, partner="MNIST", defense=name, merger=merger,
                             attack_scope=scope, dataset=task, pipeline="64-noise calibration"))
    grouped, native = {}, {}
    for row in rows:
        assert abs(row["accuracy_percent"] - 100 * row["correct"] / row["n"]) < 1e-8
        name = row["defense"]
        if name.startswith(("NativeConstraintClass_", "Defense2_")):
            name = "Defense 2"
        elif name.startswith("NoSVD_") or name == "Without_both_SVD_components":
            name = "Defense 2 w/o SVD"
        elif name.startswith("Random_noise"):
            name = "Random noise"
        elif name.startswith("A_only"):
            name = "A only"
        elif name.startswith(("B_only", "B+b_only")):
            name = "B+b only"
        elif name == "No_target":
            name = "No target"
        elif name == "Defense1":
            name = "Defense 1"
        elif name.startswith("TaskDirected_"):
            name = "Task-directed permutation"
        pipeline = "TA / AGA" if row["pipeline"] == "Component deletion" else row["pipeline"]
        prefix = (row["target"], row["partner"], name, pipeline)
        if row["merger"] == "Standalone":
            native[prefix] = row["accuracy_percent"]
            continue
        key = (*prefix, row["attack_scope"])
        values = grouped.setdefault(key, {})
        if row["dataset"] in values:
            assert abs(values[row["dataset"]] - row["accuracy_percent"]) < 1e-8, key
        values[row["dataset"]] = row["accuracy_percent"]
    cells = []
    for (target, partner, defense, pipeline, scope), values in grouped.items():
        prefix = (target, partner, defense, pipeline)
        standalone = native.get(prefix)
        if standalone is None and pipeline in ["TIES", "DARE+TA", "64-noise calibration"]:
            standalone = native.get((target, partner, defense, "TA / AGA"))
        cells.append([target, partner, defense, pipeline, scope,
                      f"{standalone:.2f}" if standalone is not None else "--",
                      f"{values[target]:.2f}" if target in values else "--",
                      f"{values[partner]:.2f}" if partner in values else "--"])
    headers = ["Target", "Partner", "Defense", "Processing", "AGA scope", "Standalone", "Target accuracy", "Partner accuracy"]
    md = ["# Attack comparison", "", "All accuracies are percentages. Only completed, verified evaluations appear; missing entries are --.", "",
          "| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    md += ["| " + " | ".join(row) + " |" for row in cells]
    md += ["", "Standalone in public-replacement rows is measured after replacement. Calibration rows reuse fused scores and do not imply a new standalone evaluation. No_target uses the public base in place of the entire target expert at the unchanged coefficient. A-only/B-only are frozen deletions, not separately optimized methods. Current coverage is three datasets, not eight."]
    def tex(text):
        return text.replace("_", r"\_").replace("%", r"\%").replace("&", r"\&")
    lines = []
    short = {"TA / AGA": "TA", "TA / AGA (reference)": "TA [reference]", "64-noise calibration": "Output calibration",
             "Public replacement A": "Base A", "Public replacement B+b": "Base B+b",
             "Public replacement MLP": "Base MLP", "MLP matching": "MLP matching",
             "Protected expert": "Target", "All experts": "Both"}
    for target in ["MNIST", "SVHN", "DTD"]:
        group = [row for row in cells if row[0] == target]
        for offset in range(0, len(group), 26):
            selected = group[offset:offset + 26]
            partner = selected[0][1]
            lines += [r"\begin{table*}[t]", r"\centering\small",
                      r"\caption{" + target + " protection, " + partner + r" partner. Accuracy (\%).}",
                      r"\begin{tabular}{lllrrr}", r"\toprule",
                      "Defense & Processing & AGA & Standalone & Target & Partner" + r" \\", r"\midrule"]
            lines += [" & ".join(tex(short.get(v, v)) for v in row[2:]) + r" \\" for row in selected]
            lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (root / "attack_comparison.md").write_text("\n".join(md) + "\n")
    (root / "attack_comparison.tex").write_text("\n".join(lines).rstrip() + "\n")
    result["new_batches"] = batches
    result["table_rows"] = cells
    result["new_full_cases_completed"] = len(batches)
    for extension, folders, title, processing, note in [
        ("eight_expert_extension", ["eight_expert", "eight_expert_extra"],
         "Eight-expert fusion", "AGA",
         "AGA processes only the protected expert. Additional datasets use identical merged weights."),
        ("unmask_extension", ["unmask"], "Unmask", "Attack",
         "Subtract 0.2 times the whole published visual weights from the fused model, including in the unprotected control. "
         "No AGA. The protected and matched unprotected models make identical predictions on all five datasets; "
         "low accuracy here does not establish an additional defense success."),
    ]:
        paths = [root / f for f in folders if (root / f / "independent_validation.json").exists()]
        if not paths:
            continue
        full, sources, validations, grouped_eight = [], {}, {}, {}
        parameters = None
        for folder in paths:
            audit = json.loads((folder / "independent_validation.json").read_text())
            assert audit["status"] == "passed_independent_numpy_metrics_and_frozen_artifact_hashes"
            batch = json.loads((folder / "full/results.json").read_text())
            validation = json.loads((folder / "full/validation.json").read_text())
            audited = next(s for s in audit["stages"] if s["stage"] == "full")
            assert validation["raw_sha256"] == audited["raw_sha256"]
            expected = {r["key"]: r for r in audited["rows"]}
            assert len(expected) == len(batch)
            current = json.loads((folder / "full/pre_evaluation.json").read_text())
            assert parameters is None or current == parameters
            parameters = current
            for row in batch:
                key = "/".join(row[k] for k in ["target", "defense", "attack_scope", "dataset"])
                assert (row["n"], row["correct"]) == (expected[key]["n"], expected[key]["correct"])
                name = "Defense 2" if row["defense"].startswith("Defense2_") else "Unprotected"
                group = (row["target"], name, short.get(row["attack_scope"], row["attack_scope"]))
                values = grouped_eight.setdefault(group, {})
                assert row["dataset"] not in values
                values[row["dataset"]] = 100 * row["correct"] / row["n"]
            full.extend(batch)
            validations[str(folder)] = validation
            for name in ["full/results.json", "full/pre_evaluation.json", "independent_validation.json"]:
                p = folder / name
                sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        datasets = [t for t in ["MNIST", "SVHN", "DTD", "GTSRB", "RESISC45"]
                    if any(t in v for v in grouped_eight.values())]
        assert all(set(v) == set(datasets) for v in grouped_eight.values())
        eight_cells = [list(key) + [f"{values[t]:.2f}" for t in datasets]
                       for key, values in grouped_eight.items()]
        headers = ["Protected task", "Defense", processing] + datasets
        note = (f"Eight experts, TA coefficient 0.2. {len(datasets)} complete datasets evaluated; "
                "no eight-task average. Same frozen releases; no reconstruction or tuning. " + note)
        lines = ["# " + title, "", note, "", "| " + " | ".join(headers) + " |",
                 "| " + " | ".join(["---"] * len(headers)) + " |"]
        lines += ["| " + " | ".join(row) + " |" for row in eight_cells]
        (paths[0] / "comparison.md").write_text("\n".join(lines) + "\n")
        lines = [r"\begin{table*}[t]", r"\centering\small",
                 r"\caption{" + title + r". Accuracy (\%).}",
                 r"\begin{tabular}{lll" + "r" * len(datasets) + "}", r"\toprule",
                 " & ".join(headers) + r" \\", r"\midrule"]
        lines += [" & ".join(tex(v) for v in row) + r" \\" for row in eight_cells]
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", "% " + note]
        (paths[0] / "comparison.tex").write_text("\n".join(lines) + "\n")
        result[extension] = dict(
            full_records=len(full), table_rows=eight_cells, validations=validations,
            datasets=datasets, source_sha256=sources,
        )
    kd = root / "distillation"
    if (kd / "paired_validation.json").exists():
        cells_kd, validations, sources = [], {}, {}
        records = 0
        for task, partner in [("MNIST", "SVHN"), ("SVHN", "MNIST"), ("DTD", "MNIST")]:
            path = kd if task == "MNIST" else kd / task
            if not (path / "paired_validation.json").exists():
                continue
            audit = json.loads((path / "paired_validation.json").read_text())
            assert audit["status"] in {"passed_independent_numpy_paired",
                "passed_independent_numpy_paired_and_exact_student_equivalence"}
            audited = next(s for s in audit["stages"] if s["stage"] == "full")
            folder = path / "evaluation/full" / f"00_{task}_{partner}"
            validation = json.loads((folder / "independent_validation.json").read_text())
            assert validation["status"] == "passed"
            assert audited["raw_sha256"] == validation["raw_sha256"]
            full = json.loads((folder / "results.json").read_text())
            expected = {r["key"]: r for r in audited["original_expert_comparisons"]}
            values = {}
            for row in full:
                key = "/".join(row[k] for k in ["defense", "merger", "attack_scope", "dataset"])
                assert (row["n"], row["correct"]) == (expected[key]["n"], expected[key]["correct"])
                values[key] = f"{100 * row['correct'] / row['n']:.2f}"
            for name, label in [("Unprotected", "Original expert"), ("KD_Unprotected", "Student from unprotected"),
                                ("KD_Defense2", "Student from Defense 2")]:
                for scope in ["None", "Protected expert", "All experts"]:
                    cells_kd.append([task + "/" + partner, label, short.get(scope, scope),
                        values[f"{name}/Standalone/None/{task}"], values[f"{name}/TA/{scope}/{task}"],
                        values[f"{name}/TA/{scope}/{partner}"]])
            records += len(full)
            validations[task] = validation
            for p in [folder / "results.json", path / "paired_validation.json"]:
                sources[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        headers = ["Task pair", "Model", "AGA", "Standalone", "Fused target", "Fused partner"]
        note = ("TA0.5. Extra unlabeled official training images: MNIST/SVHN 2048, DTD train1 1880; "
                "teacher hard labels; public-base student, 50 epochs, one fixed training seed. "
                "MNIST query labels are identical, so the registered rule reuses one historical final student; "
                "SVHN and DTD each train two students with the same images, initialization and ordering. "
                "Standalone measures the evaluated model, not the original protected teacher. "
                "No defense weights are updated. This extra-data attack is separate from weight-only recovery.")
        md = ["# Data-assisted recovery", "", note, "", "| " + " | ".join(headers) + " |",
              "| " + " | ".join(["---"] * len(headers)) + " |"]
        md += ["| " + " | ".join(row) + " |" for row in cells_kd]
        (kd / "comparison.md").write_text("\n".join(md) + "\n")
        lines = [r"\begin{table*}[t]", r"\centering\small",
                 r"\caption{Data-assisted recovery. Accuracy (\%).}",
                 r"\begin{tabular}{lllrrr}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
        lines += [" & ".join(tex(v) for v in row) + r" \\" for row in cells_kd]
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", "% " + note]
        (kd / "comparison.tex").write_text("\n".join(lines) + "\n")
        result["distillation_extension"] = dict(full_records=records, table_rows=cells_kd,
            validations=validations, source_sha256=sources)
    (root / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print("ATTACK_TABLE", len(cells), "verified new cases", len(batches))


def paper_tables(_source, output):
    """Collect frozen comparisons into matching TeX/Markdown paper tables."""
    import matplotlib
    import numpy as np

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    sources = {}
    used = []
    tables = []
    loaded = {}
    seen = set()

    def read(path):
        if path in loaded:
            return loaded[path]
        p = Path(path)
        sources[path] = hashlib.sha256(p.read_bytes()).hexdigest()
        loaded[path] = json.loads(p.read_text())
        return loaded[path]

    def pick(path, **where):
        data = read(path)
        hits = [(i, r) for i, r in enumerate(data) if all(r.get(k) == v for k, v in where.items())]
        assert len(hits) == 1, (path, where, len(hits))
        i, r = hits[0]
        if "correct" in r:
            value = 100 * r["correct"] / r["n"]
            reported = r.get("accuracy_percent", 100 * r.get("top1", value / 100))
            assert abs(value - reported) < 1e-9, (path, i)
            if "per_class" in r:
                assert (
                    sum(x["correct"] for x in r["per_class"]) == r["correct"]
                    and sum(x["n"] for x in r["per_class"]) == r["n"]
                )
        if (path, i) not in seen:
            used.append(dict(path=path, row=i, selector=where))
            seen.add((path, i))
        return r

    def acc(r):
        return r.get("accuracy_percent", 100 * r.get("top1", 0))

    def pair(path, tasks=("MNIST", "SVHN"), **where):
        return " / ".join(f"{acc(pick(path, dataset=t, **where)):.2f}" for t in tasks)

    def table(key, title, headers, rows, note, colspec=None):
        assert all(len(r) == len(headers) for r in rows), key
        tables.append(
            dict(
                key=key,
                title=title,
                headers=headers,
                rows=rows,
                note=note,
                colspec=colspec or "l" + "r" * (len(headers) - 1),
            )
        )

    numerical = []
    for target, partner, folder, name in [
        ("MNIST", "SVHN", "nonlinear_prefix", "Prefix_compensation"),
        ("DTD", "MNIST", "nonlinear_dtd", "Numerical"),
        ("SVHN", "MNIST", "nonlinear_svhn", "Numerical"),
    ]:
        path = Path("results") / folder / "evaluation/full/results.json"
        if path.with_name("independent_validation.json").exists():
            check = read(str(path.with_name("independent_validation.json")))
            execution = read(str(path.with_name("execution.json")))
            assert check["status"] == "passed"
            assert check["raw_sha256"] == execution["predictions_sha256"]
            numerical.append((target, partner, str(path), name))

    main = "results/phase3/results.json"
    scopes = ["None", "Protected expert", "All experts"]
    # These are previously completed independent audits, not new raw-score replays.
    for path in [
        "results/phase3/provenance.json",
        "results/phase3/params_adaptive/independent_validation.json",
        "results/phase3/bias_canonicalization/validation.json",
        "results/phase3/bias_tail_precision/validation.json",
        "results/phase3/distillation/independent_validation.json",
        "results/phase3/eight_task/independent_validation.json",
        "results/phase3/unmask/independent_validation.json",
        "results/phase3/recovery_overview/provenance.json",
        "results/phase3/nufilt_other_tasks/independent_validation.json",
        "results/phase3/prodistill_other_tasks/independent_validation.json",
        "results/mean_compensation/evaluation/independent_validation.json",
        "results/residual_coordinate_control/evaluation/independent_validation.json",
        "results/mean_compensation/components_independent_validation.json",
    ]:
        read(path)
    pairing_root = Path("results/task_directed_pairing")
    pairing_path = pairing_root / "evaluation/full/00_MNIST_SVHN/results.json"
    pairing_complete = pairing_path.with_name("independent_validation.json").exists()
    pairing_other_complete = all(
        (pairing_root / "other_tasks/evaluation/full" / c / "independent_validation.json").exists()
        for c in ["00_DTD_MNIST", "01_SVHN_MNIST"]
    )
    distance_root = pairing_root / "distance_ablation"
    distance_path = distance_root / "evaluation/full/00_MNIST_SVHN/results.json"
    distance_complete = distance_path.with_name("independent_validation.json").exists()
    if pairing_complete:
        check = read(str(pairing_path.with_name("independent_validation.json")))
        execution = read(str(pairing_path.with_name("execution.json")))
        assert (
            check["status"] == "passed" and check["raw_sha256"] == execution["predictions_sha256"]
        )
        assert read(str(pairing_root / "native/MNIST/independent_validation.json"))["passed"]
    rows = []
    names = [
        ("Unprotected", "Unprotected"),
        ("PaRaMS", "PaRaMS (full)"),
        ("Phase1_MNIST", "Defense 1"),
        ("O/B", "O/B [H]"),
        ("A/B", "A/B [H]"),
    ]
    for name, label in names:
        where = dict(target="MNIST", partner="SVHN", defense=name)
        native = acc(pick(main, merger="Standalone", attack_scope="None", dataset="MNIST", **where))
        rows.append(
            [label, f"{native:.2f}"]
            + [pair(main, merger="TA", attack_scope=s, **where) for s in scopes]
        )
    for folder, label in [
        ("residual_coordinate_control", "Defense 2 (structure) [K]"),
        ("mean_compensation", "Defense 2 candidate (failed) [F]"),
    ]:
        path = f"results/{folder}/evaluation/results.json"
        # Matching untreated controls must agree before adding these rows.
        for s, stage in zip(scopes, ["direct", "AGA_target_only", "AGA_both"]):
            for task in ["MNIST", "SVHN"]:
                a = pick(path, preset="original", stage=stage, dataset=task)
                b = pick(
                    main,
                    target="MNIST",
                    partner="SVHN",
                    defense="Unprotected",
                    merger="TA",
                    attack_scope=s,
                    dataset=task,
                )
                assert (a["correct"], a["n"]) == (b["correct"], b["n"])
        native = acc(pick(path, preset="released", stage="standalone", dataset="MNIST"))
        rows.append(
            [label, f"{native:.2f}"]
            + [
                pair(path, preset="released", stage=s)
                for s in ["direct", "AGA_target_only", "AGA_both"]
            ]
        )
    if pairing_complete:
        path = str(pairing_path)
        where = dict(defense="TaskDirected_MNIST")
        for scope in scopes:
            for task in ["MNIST", "SVHN"]:
                a = pick(path, defense="Unprotected", merger="TA", attack_scope=scope, dataset=task)
                b = pick(
                    main,
                    target="MNIST",
                    partner="SVHN",
                    defense="Unprotected",
                    merger="TA",
                    attack_scope=scope,
                    dataset=task,
                )
                assert (a["correct"], a["n"]) == (b["correct"], b["n"])
        native = acc(pick(path, merger="Standalone", attack_scope="None", dataset="MNIST", **where))
        rows.append(
            ["Defense 2 (task-directed)", f"{native:.2f}"]
            + [pair(path, merger="TA", attack_scope=s, **where) for s in scopes]
        )
    if numerical:
        _, _, path, name = numerical[0]
        rows.append(
            [
                "Defense 2 (numerical)",
                f"{acc(pick(path, defense=name, merger='Standalone', attack_scope='None', dataset='MNIST')):.2f}",
            ]
            + [pair(path, defense=name, merger="TA", attack_scope=s) for s in scopes]
        )
    public_backbone = Path("results/nonlinear_public_backbone")
    public_path = public_backbone / "evaluation/full/00_MNIST_SVHN/results.json"
    if public_path.exists():
        audit = read(str(public_path.with_name("numpy_statistics.json")))
        assert audit["status"].startswith("passed_independent_numpy")
        assert audit["results_sha256"] == hashlib.sha256(public_path.read_bytes()).hexdigest()
        rows.append(
            [
                "Defense 2 (public backbone)",
                f"{acc(pick(str(public_path), defense='Public_backbone', merger='Standalone', attack_scope='None', dataset='MNIST')):.2f}",
            ]
            + [
                pair(str(public_path), defense="Public_backbone", merger="TA", attack_scope=s)
                for s in scopes
            ]
        )
    table(
        "main",
        "Main comparison: two experts, TA coefficient 0.5",
        ["Protection", "Standalone M", "No recovery M / S", "AGA target M / S", "AGA both M / S"],
        rows,
        "M/S: protected MNIST / partner SVHN. Original AGA processes the target or both experts. Numerical reconstruction uses 1,024 target-training images and 12 linear solves, with experts frozen. Standalone loss is 0.19 pp for the original construction and 0.22 pp after returning the other parameters to the public base. Both strict relative-gap checks failed; the original construction also failed its small confirmation gate. The second fixed rule was registered separately and does not replace the original release. [H] Historical task-bound failures; [K] known symmetry; [F] previous failed candidate. PaRaMS retains official defaults. Additional recovery knowledge is reported separately.",
    )
    rows = []
    for target, partner, case in [
        ("MNIST", "SVHN", None),
        ("DTD", "MNIST", "00_DTD_MNIST"),
        ("SVHN", "MNIST", "01_SVHN_MNIST"),
    ]:
        task_names = [
            ("Unprotected", "Unprotected"),
            ("Phase1_" + target, "Defense 1"),
            ("PaRaMS", "PaRaMS (full)"),
            ("Residual", "Defense 2 (structure)"),
        ]
        if pairing_complete and (case is None or pairing_other_complete):
            task_names.append(("TaskDirected", "Defense 2 (task-directed)"))
        for name, label in task_names:
            if case and name in ["PaRaMS", "Residual", "TaskDirected"]:
                root = (
                    "results/"
                    + {
                        "PaRaMS": "params_reference",
                        "Residual": "residual_coordinate_control",
                        "TaskDirected": "task_directed_pairing",
                    }[name]
                    + "/other_tasks"
                )
                check = read(root + "/" + target + "/independent_validation.json")
                assert check["passed"] or (
                    name == "PaRaMS" and check["complete_native_observation"]
                )
                check = read(root + "/evaluation/full/" + case + "/independent_validation.json")
                assert check["status"] == "passed"
                path = root + "/evaluation/full/" + case + "/results.json"
                where = dict(defense=name + "_" + target)
                native = acc(
                    pick(path, merger="Standalone", attack_scope="None", dataset=target, **where)
                )
                values = [
                    pair(path, (target, partner), merger="TA", attack_scope=s, **where)
                    for s in scopes
                ]
                for s in scopes:
                    for t in [target, partner]:
                        a = pick(
                            path, defense="Unprotected", merger="TA", attack_scope=s, dataset=t
                        )
                        b = pick(
                            main,
                            target=target,
                            partner=partner,
                            defense="Unprotected",
                            merger="TA",
                            attack_scope=s,
                            dataset=t,
                        )
                        assert (a["correct"], a["n"]) == (b["correct"], b["n"])
            elif name == "TaskDirected":
                path = str(pairing_path)
                where = dict(defense="TaskDirected_MNIST")
                native = acc(
                    pick(path, merger="Standalone", attack_scope="None", dataset=target, **where)
                )
                values = [
                    pair(path, (target, partner), merger="TA", attack_scope=s, **where)
                    for s in scopes
                ]
            elif name == "Residual":
                path = "results/residual_coordinate_control/evaluation/results.json"
                native = acc(pick(path, preset="released", stage="standalone", dataset=target))
                values = [
                    pair(path, (target, partner), preset="released", stage=s)
                    for s in ["direct", "AGA_target_only", "AGA_both"]
                ]
            else:
                where = dict(target=target, partner=partner, defense=name)
                native = acc(
                    pick(main, merger="Standalone", attack_scope="None", dataset=target, **where)
                )
                values = [
                    pair(main, (target, partner), merger="TA", attack_scope=s, **where)
                    for s in scopes
                ]
            rows.append([target + " / " + partner, label, f"{native:.2f}"] + values)
    for target, partner, path, name in numerical:
        rows.append(
            [
                target + " / " + partner,
                "Defense 2 (numerical)",
                f"{acc(pick(path, defense=name, merger='Standalone', attack_scope='None', dataset=target)):.2f}",
            ]
            + [
                pair(path, (target, partner), defense=name, merger="TA", attack_scope=s)
                for s in scopes
            ]
        )
    table(
        "three_tasks",
        "Matched defenses across three protected tasks",
        [
            "Target / partner",
            "Protection",
            "Standalone",
            "No recovery T / P",
            "AGA target T / P",
            "AGA both T / P",
        ],
        rows,
        "TA0.5 and unchanged original AGA; every merged cell is protected / partner accuracy. Full PaRaMS uses official defaults without clipping. Structure uses a fixed global rule; task-directed uses the frozen finite objective with each target task SVD. Three protected tasks correspond to two unordered task pairs, not a complete eight-task table. Novelty and known-family/location-aware recovery are separate. Complete original-task predictions, parameter scales and numerical protocols are reported separately.",
    )
    cross_root = pairing_root / "cross_pairs"
    cross_cases = ["00_MNIST_DTD", "01_DTD_SVHN", "02_SVHN_DTD"]
    cross_complete = all(
        (cross_root / "full" / c / "independent_validation.json").exists() for c in cross_cases
    )
    if cross_complete:
        paths = {
            ("MNIST", "SVHN"): pairing_path,
            ("DTD", "MNIST"): pairing_root
            / "other_tasks/evaluation/full/00_DTD_MNIST/results.json",
            ("SVHN", "MNIST"): pairing_root
            / "other_tasks/evaluation/full/01_SVHN_MNIST/results.json",
        }
        for case in cross_cases:
            folder = cross_root / "full" / case
            check = read(str(folder / "independent_validation.json"))
            meta = read(str(folder / "execution.json"))
            assert (
                check["status"] == "passed"
                and check["original_batch_context_replays"]
                and check["raw_sha256"] == meta["predictions_sha256"]
            )
            data = read(str(folder / "results.json"))
            paths[data[0]["target"], data[0]["partner"]] = folder / "results.json"
        rows = []
        for (target, peer), path in paths.items():
            for defense, label in [
                ("Unprotected", "Unprotected"),
                ("TaskDirected_" + target, "Defense 2 (task-directed)"),
            ]:
                rows.append(
                    [target + " / " + peer, label]
                    + [
                        pair(
                            str(path), (target, peer), defense=defense, merger="TA", attack_scope=s
                        )
                        for s in scopes
                    ]
                )
        table(
            "cross_pairs",
            "Frozen task-directed releases across all six task-pair orientations",
            [
                "Target / partner",
                "Protection",
                "No recovery T / P",
                "AGA target T / P",
                "AGA both T / P",
            ],
            rows,
            "The same three releases and cached unchanged author-AGA edits; TA0.5. Six orientations correspond to three unordered task pairs,not six independent datasets. The three additional evaluations were preregistered after earlier results; no release,rank,objective or parameter was selected or changed. Matching untreated controls,per-class/pair counts and original128-context score replays are retained. Historical tests are not blind. Known-family patch recovery restores these models and is reported separately; this table does not show adaptive recovery resistance.",
        )
    recovery_root = pairing_root / "recovery"
    if (recovery_root / "validation.json").exists():
        check = read(str(recovery_root / "validation.json"))
        assert check["status"] == "passed_independent_public_matching_and_all_tensors"
        recovered = read(str(recovery_root / "results.json"))
        rows = []
        for case in recovered["cases"]:
            assert case["status"] == "exact_original_state_recovered" and not any(
                case["tensor_differences"]
            )
            path = case["verified_evaluation"]["results"]["path"]
            target, peer = case["target"], case["partner"]
            for defense, label in [
                ("TaskDirected_" + target, "Without family matching"),
                ("Unprotected", "Known-family matching"),
            ]:
                rows.append(
                    [target + " / " + peer, label]
                    + [
                        pair(path, (target, peer), defense=defense, merger="TA", attack_scope=s)
                        for s in scopes
                    ]
                )
        table(
            "task_directed_recovery",
            "Task-directed protection and known-family public recovery",
            ["Target / partner", "Recovery rule", "TA T / P", "AGA target T / P", "AGA both T / P"],
            rows,
            "All percentages. Same published/public-base weight access; targeted matching knows the global residual-permutation family and native CLIP layout. No clean expert, true task vector, secret order or images enter the attack. Each target and matched untreated control equals all158 original tensors exactly; original independently audited scores are therefore reused,not rerun. Original AGA remains unchanged and is reported separately. This does not establish recovery of arbitrary defenses.",
            colspec="llrrr",
        )
    rows = []
    for target, partner in [
        ("MNIST", "SVHN"),
        ("MNIST", "DTD"),
        ("DTD", "MNIST"),
        ("SVHN", "MNIST"),
    ]:
        for name, label in [("Unprotected", "Unprotected"), ("Phase1_" + target, "Defense 1")]:
            where = dict(target=target, partner=partner, defense=name)
            native = acc(
                pick(main, merger="Standalone", attack_scope="None", dataset=target, **where)
            )
            rows.append(
                [target + " / " + partner, label, f"{native:.2f}"]
                + [
                    pair(main, (target, partner), merger="TA", attack_scope=s, **where)
                    for s in scopes
                ]
            )
    table(
        "defense1",
        "Defense 1 across three protected tasks",
        [
            "Target / partner",
            "Protection",
            "Standalone",
            "No recovery T / P",
            "AGA target T / P",
            "AGA both T / P",
        ],
        rows,
        "TA0.5. Every merged cell is target / partner accuracy. The task role is fixed by the first column, not by a global dataset order. The original AGA does not modify the final LayerNorm bias; these results do not imply resistance to location-aware or data-assisted recovery.",
    )
    rows = []
    for s, label, access in [
        ("None", "No recovery", "Published expert and partner"),
        ("Protected expert", "AGA target", "Public base; defense-agnostic"),
        ("All experts", "AGA both", "Public base; defense-agnostic"),
    ]:
        rows.append(
            [label, access]
            + [
                pair(main, target="MNIST", partner="SVHN", defense=n, merger="TA", attack_scope=s)
                for n in ["Unprotected", "Phase1_MNIST"]
            ]
        )
    path = "results/phase3/params_adaptive/results.json"
    rows.append(
        ["Diagonal adaptation", "Specified diagonal family; no images"]
        + [
            pair(path, defense=n, merger="TA", attack_scope="Protected expert")
            for n in ["Unprotected", "Phase1_MNIST"]
        ]
    )
    path = "results/phase3/bias_canonicalization/results.json"
    rows.append(
        ["LN-bias cleanup", "Known location; published projection"]
        + [
            pair(
                path,
                target="MNIST",
                partner="SVHN",
                defense=n,
                merger="TA",
                attack_scope="Protected expert",
            )
            for n in ["Unprotected", "Phase1_MNIST"]
        ]
    )
    path = "results/phase3/distillation/results.json"
    for mode, label in [
        ("TA", "Student + TA"),
        ("AGA-target-only", "Student + AGA target"),
        ("AGA-both", "Student + AGA both"),
    ]:
        value = pair(path, mode=mode)
        rows.append([label, "2048 target-domain queries; student training", value, value])
    table(
        "recovery",
        "Recovery permissions: protected MNIST and partner SVHN",
        ["Recovery pipeline", "Additional access", "Matched control M / S", "Defense 1 M / S"],
        rows,
        "All merges use TA0.5. Diagonal adaptation is the official MLP kernel plus an explicitly specified native Attention adaptation, not a complete official attack release. Cleanup is a known-location diagnostic. Native FP32 cleanup changes some scores beyond the historical numerical-control tolerance; no exact FP32 function identity is claimed. Student rows share one training problem because all2048 queried hard labels coincide; they are not independent seed replications or the complete PaRaMS KD implementation.",
        r"p{39mm}p{49mm}rr",
    )
    overview = "results/phase3/recovery_overview/results.json"
    rows = []
    for method in ["TA", "TIES", "DARE+TA", "NUFILT M→S", "NUFILT S→M", "ProDistill"]:
        cells = [
            " / ".join(
                f"{pick(overview, merger=method, defense=n, dataset=t)['accuracy_percent']:.2f}"
                for t in ["MNIST", "SVHN"]
            )
            for n in ["Unprotected", "Phase1_MNIST", "O/B", "A/B"]
        ]
        rows.append([method] + cells)
    table(
        "mergers",
        "Supplementary mergers after original AGA on the protected expert",
        ["Merger", "Unprotected M / S", "Defense 1 M / S", "O/B [H] M / S", "A/B [H] M / S"],
        rows,
        "MNIST target / SVHN partner. Each merger has its own matching control. DARE+TA is the mean of fixed mask seeds42/43/44; other entries are fixed runs. NUFILT reports both task orders and uses native adapters of official kernels. ProDistill uses64+64 unlabeled training calibration images and attacker-side elementwise coefficient training; it is not a data-free attack. Mergers are not counted as separate recovery attacks. [H] Task-bound failure retained.",
    )
    rows = []
    for name, label in names[:1] + names[2:]:
        cells = []
        for t in ["MNIST", "SVHN"]:
            values = pick(overview, merger="DARE+TA", defense=name, dataset=t)["accuracy_values"]
            assert len(values) == 3
            cells.append(f"{np.mean(values):.2f} +/- {np.std(values, ddof=1):.2f}")
        rows.append([label] + cells)
    table(
        "dare",
        "DARE mask variability under AGA target",
        ["Protection", "MNIST mean +/- SD", "SVHN mean +/- SD"],
        rows,
        "Sample standard deviation across the same three fixed masks42/43/44. This is mask variability, not three independent expert or defense trainings. All three mask values are retained in the cited source rows.",
    )
    rows = []
    for target in ["DTD", "SVHN"]:
        path = "results/phase3/nufilt_other_tasks/results.json"
        for order in [[target, "MNIST"], ["MNIST", target]]:
            for scope in scopes:
                if target == "SVHN" and scope != "Protected expert":
                    value = pair(
                        main,
                        ("SVHN", "MNIST"),
                        target="MNIST",
                        partner="SVHN",
                        defense="Unprotected",
                        merger="NUFILT",
                        order=order,
                        attack_scope=scope,
                    )
                    cells = [value, value]
                else:
                    cells = [
                        pair(
                            path,
                            (target, "MNIST"),
                            target=target,
                            partner="MNIST",
                            defense=name,
                            order=order,
                            attack_scope=scope,
                        )
                        for name in ["Unprotected", "Phase1_" + target]
                    ]
                rows.append(
                    [
                        target + " / MNIST",
                        "NUFILT " + " / ".join(order),
                        {"None": "None", "Protected expert": "Target", "All experts": "Both"}[
                            scope
                        ],
                    ]
                    + cells
                )
        path = "results/phase3/prodistill_other_tasks/results.json"
        for scope in scopes:
            rows.append(
                [
                    target + " / MNIST",
                    "ProDistill",
                    {"None": "None", "Protected expert": "Target", "All experts": "Both"}[scope],
                ]
                + [
                    pair(
                        path,
                        (target, "MNIST"),
                        target=target,
                        partner="MNIST",
                        defense=name,
                        attack_scope=scope,
                    )
                    for name in ["Unprotected", "Phase1_" + target]
                ]
            )
    table(
        "other_tasks",
        "Supplementary mergers: DTD and SVHN protected experts",
        ["Target / partner", "Merger / order", "AGA scope", "Control T / P", "Defense 1 T / P"],
        rows,
        "NUFILT uses both task orders. SVHN no-attack/both-AGA cells reuse the identical earlier MNIST/SVHN merger models with dataset roles exchanged; no new fits. The selected-parameter identity for Defense1 is retained. ProDistill uses64+64 unlabeled calibration images and official coefficient kernels. These are merger evaluations, not additional recovery attacks.",
    )
    for folder, key, title, scope in [
        (
            "eight_task",
            "eight",
            "Eight participating experts: three evaluated tasks",
            "Protected expert",
        ),
        (
            "unmask",
            "unmask",
            "Paper-formula Unmask: a separate eight-expert recovery setting",
            "Unmask",
        ),
    ]:
        path = f"results/phase3/{folder}/results.json"
        rows = []
        current_eight = "results/task_directed_pairing/eight_expert"
        current_eight_complete = Path(current_eight + "/independent_validation.json").exists()
        if folder == "eight_task" and current_eight_complete:
            assert (
                read(current_eight + "/independent_validation.json")["status"]
                == "passed_independent_numpy_metrics_and_frozen_artifact_hashes"
            )
            current_protocol = read(current_eight + "/protocol.json")
            old_protocol = read("results/phase3/eight_task/protocol.json")
            for field in ["coefficient", "task_order", "evaluate", "author_TaskVector"]:
                assert current_protocol[field] == old_protocol[field]
        if folder == "unmask":
            for name in ["rows", "validation"]:
                ref = read("results/phase3/unmask/protocol.json")["no_attack_reference"][name]
                read(ref["path"])
                assert sources[ref["path"]] == ref["sha256"]
        for target in ["MNIST", "DTD", "SVHN"]:
            methods = ["Unprotected", "Phase1_" + target] + (
                ["O/B", "A/B", "PaRaMS", "PaRaMS_Attention"] if target == "MNIST" else []
            )
            for name in methods:
                label = (
                    "Defense 1"
                    if name.startswith("Phase1")
                    else {
                        "O/B": "O/B [H]",
                        "A/B": "A/B [H]",
                        "PaRaMS": "PaRaMS (full)",
                        "PaRaMS_Attention": "PaRaMS attention ablation",
                    }.get(name, name)
                )
                rows.append(
                    [target, label]
                    + [
                        pair(
                            "results/phase3/eight_task/results.json"
                            if folder == "unmask" and s == "None"
                            else path,
                            ("MNIST", "DTD", "SVHN"),
                            target=target,
                            defense=name,
                            attack_scope=s,
                        )
                        for s in ["None", scope]
                    ]
                )
            if folder == "eight_task" and current_eight_complete:
                for s in ["None", scope]:
                    for task in ["MNIST", "DTD", "SVHN"]:
                        a = pick(
                            current_eight + "/results.json",
                            target=target,
                            defense="Unprotected",
                            attack_scope=s,
                            dataset=task,
                        )
                        b = pick(
                            path, target=target, defense="Unprotected", attack_scope=s, dataset=task
                        )
                        assert (a["correct"], a["n"], a["model_parameter_sha256"]) == (
                            b["correct"],
                            b["n"],
                            b["model_parameter_sha256"],
                        )
                rows.append(
                    [target, "Defense 2 (task-directed)"]
                    + [
                        pair(
                            current_eight + "/results.json",
                            ("MNIST", "DTD", "SVHN"),
                            target=target,
                            defense="TaskDirected_" + target,
                            attack_scope=s,
                        )
                        for s in ["None", scope]
                    ]
                )
        note = "TA0.2 with eight experts; cells always MNIST / DTD / SVHN, including partner tasks. Other five task datasets were not evaluated. No eight-task average. [H] Original task-bound failures retained. Complete PaRaMS and its Attention ablation remain distinct."
        if folder == "unmask":
            note += " Fixed merged parameters minus0.3 times the whole published expert. Both protected and unprotected controls receive the operation. Paper-formula native adaptation, not official code or a complete paper table. No AGA composition. The unprotected model also collapses; low absolute scores do not demonstrate successful expert recovery."
        if folder == "eight_task" and current_eight_complete:
            note += " Task-directed rows use the same three frozen releases as TA0.5. No all-expert AGA or new release was run. Known-family recovery remains applicable."
        table(
            key,
            title,
            [
                "Protected task",
                "Protection",
                "No recovery M / D / S",
                ("AGA target" if folder == "eight_task" else "Unmask0.3") + " M / D / S",
            ],
            rows,
            note,
        )
    rows = []
    for target, partner, path, mode in [
        ("MNIST", "SVHN", "results/phase3/bias_canonicalization/results.json", "Native FP32"),
        ("MNIST", "DTD", "results/phase3/bias_canonicalization/results.json", "Native FP32"),
        (
            "DTD",
            "MNIST",
            "results/phase3/bias_tail_precision/results.json",
            "FP32 prefix; FP64 readout",
        ),
        (
            "SVHN",
            "MNIST",
            "results/phase3/bias_tail_precision/results.json",
            "FP32 prefix; FP64 readout",
        ),
    ]:
        for scope, label in [("Protected expert", "Target"), ("All experts", "Both")]:
            rows.append(
                [target + " / " + partner, mode, label]
                + [
                    pair(
                        path,
                        (target, partner),
                        target=target,
                        partner=partner,
                        merger="TA",
                        defense=name,
                        attack_scope=scope,
                    )
                    for name in ["Unprotected", "Phase1_" + target]
                ]
            )
    table(
        "cleanup",
        "Targeted bias cleanup: numerical protocols kept separate",
        ["Target / partner", "Numerics", "Scope", "Control T / P", "Defense 1 T / P"],
        rows,
        "TA0.5; no images, original expert or secret perturbation enters recovery construction. The FP64-readout rows are a separately frozen protocol and must not be pooled with native FP32. Original native DTD control failure0.00082493 and MNIST before/after score difference0.00069427 remain recorded. Equal recovered predictions do not establish exact FP32 function identity.",
    )
    geometry = read("results/mean_compensation/allocation_geometry.json")
    svd = read("results/task_subspace_comparison/results.json")
    old = next(
        g for g in svd["groups"] if g["method"] == "global_permutation" and g["site"] == "all"
    )
    geometry_rows = []
    for folder, label, di, do in [
        (
            "residual_coordinate_control",
            "Global permutation [K]",
            old["distances"]["right_inside_distance"],
            old["distances"]["right_outside_distance"],
        ),
        (
            "mean_compensation",
            "Previous compensation [F]",
            geometry["task_inside_distance"],
            geometry["task_outside_distance"],
        ),
    ]:
        path = f"results/{folder}/evaluation/results.json"
        geometry_rows.append(
            [label, f"{di:.5f}", f"{do:.5f}"]
            + [
                f"{acc(pick(path, preset='released', stage=s, dataset='MNIST')):.2f}"
                for s in ["standalone", "direct", "AGA_both"]
            ]
        )
    if pairing_complete:
        measured = read(str(pairing_root / "construction/matrices.json"))
        for variant, label in [
            ("distance_only", "Distance-only pairing"),
            ("task_aware", "Task-directed pairing"),
        ]:
            rows = [r for r in measured if r["variant"] == variant]
            assert len(rows) == 74
            di, do = [
                math.sqrt(sum(r[k] for r in rows)) for k in ["inside_squared", "outside_squared"]
            ]
            values = (
                [
                    f"{acc(pick(str(pairing_path), defense='TaskDirected_MNIST', merger=m, attack_scope=s, dataset='MNIST')):.2f}"
                    for m, s in [("Standalone", "None"), ("TA", "None"), ("TA", "All experts")]
                ]
                if variant == "task_aware"
                else ["--"] * 3
            )
            if variant == "distance_only" and distance_complete:
                values = [
                    f"{acc(pick(str(distance_path), defense='DistanceOnly_MNIST', merger=m, attack_scope=s, dataset='MNIST')):.2f}"
                    for m, s in [("Standalone", "None"), ("TA", "None"), ("TA", "All experts")]
                ]
            geometry_rows.append([label, f"{di:.5f}", f"{do:.5f}"] + values)
    table(
        "geometry",
        "Task-direction changes and actual classification effects",
        [
            "Fixed model",
            "Task-inside distance",
            "Task-outside distance",
            "Standalone M",
            "TA M",
            "AGA both M",
        ],
        geometry_rows,
        "Same 74 matrices and original expert-minus-base rank 32 right singular directions. Squared per-matrix relative norms are summed. These matrix-only distances exclude biases/LN/embeddings and are not the full-model distance. Comparable task-outside distances do not imply comparable protection. [K] Known symmetry; [F] failed efficacy. Distance-only pairing is a fixed control,not a replacement selected by accuracy; -- means no classification run. The task-directed release is selected before images; no novelty claim for standard symmetry/SVD/greedy alone.",
    )
    distance_other = distance_root / "other_tasks"
    if distance_complete:
        rows = []
        for target, partner, case in [
            ("MNIST", "SVHN", "00_MNIST_SVHN"),
            ("DTD", "MNIST", "00_DTD_MNIST"),
            ("SVHN", "MNIST", "01_SVHN_MNIST"),
        ]:
            left = (
                pairing_path
                if target == "MNIST"
                else pairing_root / "other_tasks/evaluation/full" / case / "results.json"
            )
            right = (
                distance_path
                if target == "MNIST"
                else distance_other / "evaluation/full" / case / "results.json"
            )
            if not right.with_name("independent_validation.json").exists():
                continue
            assert read(str(right.with_name("independent_validation.json")))["status"] == "passed"
            native = (
                distance_root / "native/independent_validation.json"
                if target == "MNIST"
                else distance_other / "native" / target / "results.json"
            )
            assert read(str(native))["passed"]
            for path, name, label in [
                (str(left), "TaskDirected_" + target, "Task-directed"),
                (str(right), "DistanceOnly_" + target, "Distance-only"),
            ]:
                for scope in scopes:
                    for task in [target, partner]:
                        a = pick(
                            path,
                            defense="Unprotected",
                            merger="TA",
                            attack_scope=scope,
                            dataset=task,
                        )
                        b = pick(
                            str(left),
                            defense="Unprotected",
                            merger="TA",
                            attack_scope=scope,
                            dataset=task,
                        )
                        assert (a["correct"], a["n"]) == (b["correct"], b["n"])
                rows.append(
                    [
                        target + " / " + partner,
                        label,
                        f"{acc(pick(path, defense=name, merger='Standalone', attack_scope='None', dataset=target)):.2f}",
                    ]
                    + [
                        pair(
                            path, tasks=(target, partner), defense=name, merger="TA", attack_scope=s
                        )
                        for s in scopes
                    ]
                )
        table(
            "distance_ablation",
            "Fixed ablation of the task-direction objective",
            ["Protected / partner", "Construction", "Standalone", "TA", "AGA target", "AGA both"],
            rows,
            "Same target, base, rank32,74 matrices,greedy rule,TA0.5 and originalAGA. All distance-only orders were fixed before main images; classification supplements were registered later. The SVD term does not improve every dataset. Full results are retained; no replacement selection or independent expert-seed claim.",
        )
    amplitude_root = pairing_root / "amplitude_ablation"
    amplitude_path = amplitude_root / "evaluation/full/00_MNIST_SVHN/results.json"
    amplitude_complete = amplitude_path.with_name("independent_validation.json").exists()
    if amplitude_complete:
        assert (
            read(str(amplitude_path.with_name("independent_validation.json")))["status"] == "passed"
        )
        rows = []
        points = [("Unprotected", 0, str(pairing_path), "Unprotected", None)]
        for count in [96, 192, 384]:
            for variant, label in [
                ("task_aware", "Task-directed"),
                ("distance_only", "Distance-only"),
            ]:
                if count == 384:
                    path = str(pairing_path if variant == "task_aware" else distance_path)
                    name = "TaskDirected_MNIST" if variant == "task_aware" else "DistanceOnly_MNIST"
                    meta = (
                        pairing_root / "construction/before_images.json"
                        if variant == "task_aware"
                        else distance_root / "construction/before_images.json"
                    )
                else:
                    path, name = str(amplitude_path), f"Amplitude_{variant}_{count}"
                    folder = str(count) if variant == "task_aware" else f"distance_only_{count}"
                    meta = amplitude_root / "construction" / folder / "before_images.json"
                    assert read(str(amplitude_root / "native" / folder / "results.json"))["passed"]
                points.append((label, count, path, name, meta))
        for label, count, path, name, meta in points:
            distance = read(str(meta))["logical_parameter_distance"] if meta is not None else 0
            rows.append(
                [
                    label,
                    str(count),
                    f"{distance:.3f}",
                    f"{acc(pick(path, defense=name, merger='Standalone', attack_scope='None', dataset='MNIST')):.2f}",
                ]
                + [pair(path, defense=name, merger="TA", attack_scope=s) for s in scopes]
            )
        table(
            "amplitude",
            "Modification extent with the same fixed construction orders",
            [
                "Construction",
                "Pairs",
                "Distance",
                "Standalone M",
                "TA M / S",
                "AGA target M / S",
                "AGA both M / S",
            ],
            rows,
            "MNIST protected,SVHN partner. Use the first96/192/384 disjoint swaps of each frozen22October trace;0/384 scores are reused. Four new controls,no new ordering,objective tuning or expert training. Distances use separate logical matrices and biases. These are extent controls in the prospective structural range,notD0.5 comparisons. Known-family recovery remains applicable.",
        )
    numerical_root = Path("results/nonlinear_prefix")
    if numerical:
        cases = [("Complete method", numerical_root, "Prefix_compensation", "source.json")]
        for name, label in [("no_task", "Without SVD"), ("no_prefix", "Without prefix correction")]:
            root = numerical_root / "ablations" / name
            if (root / "evaluation/full/00_MNIST_SVHN/independent_validation.json").exists():
                cases.append((label, root, name, "before_images.json"))
        rows = []
        for label, root, name, source in cases:
            result = root / "evaluation/full"
            if name != "Prefix_compensation":
                result /= "00_MNIST_SVHN"
            assert read(str(result / "independent_validation.json"))["status"] == "passed"
            details = read(str(root / source))["details"]
            response = max(
                max(r["task_A_response_relative"], r["task_B_response_relative"])
                for r in details["per_layer"]
            )
            path = str(result / "results.json")
            rows.append(
                [
                    label,
                    f"{details['logical_distance']:.3f}",
                    f"{response:.2g}",
                    f"{acc(pick(path, defense=name, merger='Standalone', attack_scope='None', dataset='MNIST')):.2f}",
                ]
                + [pair(path, defense=name, merger="TA", attack_scope=s) for s in scopes]
            )
        table(
            "numerical_ablation",
            "Numerical Defense 2: matched component removals",
            [
                "Construction",
                "Distance",
                "Task probe",
                "SA M",
                "TA M / S",
                "AGA target M / S",
                "AGA both M / S",
            ],
            rows,
            "Same 1,024 construction images, 12 layers, rank 32, damping 0.01 and parameter ranges. The task probe measures the largest relative parameter change along teacher task directions; it does not guarantee task retention. Removing SVD still protects MNIST, so a classification benefit from SVD is not established. Removing prefix correction fits original teacher MLP outputs from actual student inputs. All controls are reported without replacing the main release.",
        )
    components = numerical_root / "source_components/results.json"
    if components.exists():
        rows = [
            [r["component"], f"{r['correct']}/{r['n']}", str(r["changed"])]
            for r in read(str(components))["rows"]
        ]
        table(
            "numerical_components",
            "Numerical changes: source-only component checks",
            ["Component", "Correct", "Prediction changes"],
            rows,
            "Same 1,024 target-training images and frozen release. A-only and B-only use subsets of its edits. No-reconstruction retains A changes and the pre-fit B transforms. No weights are refitted or selected. These are construction-set observations, not independent test results.",
        )
    reconstruction = []
    for label, folder, name in [
        ("Prefix reconstruction", None, None),
        ("With attention correction", "nonlinear_pre_ln", "Pre_LN"),
        ("Same-input teacher function", "nonlinear_local_function", "Local_function"),
    ]:
        for target, partner, path, original_name in numerical:
            source = (
                Path("results")
                / {"MNIST": "nonlinear_prefix", "DTD": "nonlinear_dtd", "SVHN": "nonlinear_svhn"}[
                    target
                ]
            )
            if folder is not None:
                source = Path("results") / folder / target
                path = str(source / "evaluation/full/results.json")
                if not Path(path).with_name("numpy_statistics.json").exists():
                    continue
                audit = read(str(Path(path).with_name("numpy_statistics.json")))
                assert audit["status"].startswith("passed_independent_numpy")
            native = acc(
                pick(
                    path,
                    defense=name or original_name,
                    merger="Standalone",
                    attack_scope="None",
                    dataset=target,
                )
            )
            reconstruction.append(
                [label, target, f"{native:.2f}"]
                + [
                    pair(
                        path,
                        tasks=(target, partner),
                        defense=name or original_name,
                        merger="TA",
                        attack_scope=s,
                    )
                    for s in scopes
                ]
            )
    if reconstruction:
        for target, partner, path in [
            ("MNIST", "SVHN", public_path),
            (
                "SVHN",
                "MNIST",
                public_backbone / "other_tasks/SVHN/evaluation/full/00_SVHN_MNIST/results.json",
            ),
            (
                "DTD",
                "MNIST",
                public_backbone / "other_tasks/DTD/evaluation/full/00_DTD_MNIST/results.json",
            ),
        ]:
            if not path.exists():
                continue
            audit = read(str(path.with_name("numpy_statistics.json")))
            assert audit["status"].startswith("passed_independent_numpy")
            reconstruction.append(
                [
                    "Public-backbone reconstruction",
                    target,
                    f"{acc(pick(str(path), defense='Public_backbone', merger='Standalone', attack_scope='None', dataset=target)):.2f}",
                ]
                + [
                    pair(
                        str(path),
                        tasks=(target, partner),
                        defense="Public_backbone",
                        merger="TA",
                        attack_scope=s,
                    )
                    for s in scopes
                ]
            )
        table(
            "numerical_reconstruction",
            "Numerical reconstruction: all fixed target-task measurements",
            [
                "Construction",
                "Target",
                "Standalone",
                "TA T / P",
                "AGA target T / P",
                "AGA both T / P",
            ],
            reconstruction,
            "TA coefficient 0.5. MNIST uses SVHN as partner; DTD/SVHN use MNIST. Construction sets contain 1,024 MNIST, 1,024 SVHN and 512 DTD images. Attention correction adds an O solve before each A/B step. Same-input reconstruction changes only the teacher target. Every fixed point is reported; low merged accuracy is not protection success when standalone accuracy collapses. No control replaces the main release.",
        )
    coefficient_rows = []
    for coefficient, path in [
        (0.2, numerical_root / "coefficients/0p2/full/00_MNIST_SVHN/results.json"),
        (0.5, numerical_root / "evaluation/full/results.json"),
        (0.8, numerical_root / "coefficients/0p8/full/00_MNIST_SVHN/results.json"),
    ]:
        if not path.exists():
            continue
        for name, label in [
            ("Unprotected", "Unprotected"),
            ("Prefix_compensation", "Numerical357"),
        ]:
            coefficient_rows.append(
                [label, f"{coefficient:.1f}"]
                + [pair(str(path), defense=name, merger="TA", attack_scope=s) for s in scopes]
            )
    if coefficient_rows:
        table(
            "numerical_coefficient",
            "Fixed numerical release at three TA coefficients",
            ["Protection", "Coefficient", "TA M / S", "AGA target M / S", "AGA both M / S"],
            coefficient_rows,
            "One frozen release: standalone MNIST is 99.50%, versus 99.69% untreated. Coefficients 0.2 and 0.8 were fixed after observing 0.5 and were not used to tune or select defense weights. Both original AGA scopes and matched untreated controls are reported.",
        )
    reset = numerical_root / "public_reset/evaluation/full/00_MNIST_SVHN/results.json"
    if reset.exists():
        rows = []
        reset_models = [
            (numerical_root / "evaluation/full/results.json", "Unprotected", "Unprotected"),
            (
                numerical_root / "evaluation/full/results.json",
                "Prefix_compensation",
                "Numerical357",
            ),
            (reset, "Public_MLP_reset", "Reset on untreated / original numerical model"),
        ]
        if public_path.exists():
            reset_models += [
                (public_path, "Public_backbone", "Public-backbone construction"),
                (public_path, "Reset_protected", "Reset on public-backbone construction"),
                (
                    public_backbone / "no_target/evaluation/full/00_MNIST_SVHN/results.json",
                    "No_target",
                    "No target expert",
                ),
            ]
        for path, name, label in reset_models:
            native = acc(
                pick(
                    str(path),
                    defense=name,
                    merger="Standalone",
                    attack_scope="None",
                    dataset="MNIST",
                )
            )
            rows.append(
                ["SVHN", label, f"{native:.2f}"]
                + [pair(str(path), defense=name, merger="TA", attack_scope=s) for s in scopes]
            )
        cross = public_backbone / "cross_partner/full/00_MNIST_DTD/results.json"
        if cross.exists():
            audit = read(str(cross.with_name("numpy_statistics.json")))
            assert audit["status"].startswith("passed_independent_numpy")
            for name, label in [
                ("Unprotected", "Unprotected"),
                ("Public_backbone", "Public-backbone construction"),
                ("Reset_untreated", "Reset on untreated model"),
                ("Reset_protected", "Reset on public-backbone construction"),
                ("No_target", "No target expert"),
            ]:
                native = acc(
                    pick(
                        str(cross),
                        defense=name,
                        merger="Standalone",
                        attack_scope="None",
                        dataset="MNIST",
                    )
                )
                rows.append(
                    ["DTD", label, f"{native:.2f}"]
                    + [
                        pair(
                            str(cross),
                            tasks=("MNIST", "DTD"),
                            defense=name,
                            merger="TA",
                            attack_scope=s,
                        )
                        for s in scopes
                    ]
                )
        reset_note = "The attacker knows the 12 modified MLP modules and replaces their 48 tensors with public-base values. No original expert, true task vector, secret transform or images enter this operator. Resetting the untreated and original numerical models gives identical states. Original AGA and known-module reset are separate conditions."
        if public_path.exists():
            comparison = read(str(public_backbone / "matched_no_target.json"))
            gain = next(
                r
                for r in comparison["rows"]
                if r["model"] == "Reset_protected"
                and r["scope"] == "Protected expert"
                and r["dataset"] == "MNIST"
            )
            lo, hi = gain["gain95ci_pp"]
            reset_note += f" Resetting the public-backbone construction leaves only three published final-readout tensors beyond the base. No target means base plus 0.5 times the partner task vector; its target-only column repeats direct merging, while its both column attacks only the partner. With SVHN the new reset adds {gain['gain_pp']:.2f} pp over this baseline, paired 95% interval [{lo:.2f},{hi:.2f}] pp. This is an accuracy comparison, not a measure of knowledge leakage or proof of complete recovery resistance."
        table(
            "numerical_reset",
            "Known-module recovery using the public base",
            ["Partner", "Model", "SA M", "TA M / P", "AGA target M / P", "AGA both M / P"],
            rows,
            reset_note,
            r"l>{\raggedright\arraybackslash}p{42mm}rrrr",
        )
    eight = numerical_root / "eight_expert/full"
    if (eight / "numpy_statistics.json").exists():
        audit = read(str(eight / "numpy_statistics.json"))
        assert audit["status"].startswith("passed_independent_numpy")
        rows = []
        for name, label in [
            ("Unprotected", "Unprotected"),
            ("Prefix_compensation", "Numerical357"),
        ]:
            for scope, attack in [("None", "None"), ("Protected expert", "AGA target")]:
                rows.append(
                    [label, attack]
                    + [
                        f"{acc(pick(str(eight / 'results.json'), defense=name, attack_scope=scope, dataset=task)):.2f}"
                        for task in ["MNIST", "DTD", "SVHN"]
                    ]
                )
        newer = public_backbone / "eight_expert/full/results.json"
        if newer.exists():
            audit = read(str(newer.with_name("numpy_statistics.json")))
            assert audit["status"].startswith("passed_independent_numpy")
            for name, label in [
                ("Public_backbone", "Public-backbone construction"),
                ("Reset_protected", "Known-module reset"),
                ("No_target", "No target expert"),
            ]:
                for scope, attack in [("None", "None"), ("Protected expert", "AGA target")]:
                    rows.append(
                        [label, attack]
                        + [
                            f"{acc(pick(str(newer), defense=name, attack_scope=scope, dataset=task)):.2f}"
                            for task in ["MNIST", "DTD", "SVHN"]
                        ]
                    )
        table(
            "numerical_eight",
            "Same numerical release: eight experts,three evaluated datasets",
            ["Protection", "Recovery", "MNIST", "DTD", "SVHN"],
            rows,
            "Eight experts in author order, TA coefficient 0.2, only MNIST protected. Original AGA processes only the protected expert. The release and AGA weights are reused. DTD/SVHN are evaluated partners. Original numerical357: 12 records and 192 scores; public-backbone construction and matched reset/no-target controls: 24 records and 384 scores. Original batch-context replays have zero error, with independent paired/per-class counts. Reset versus no-target must be compared at this coefficient, not at the two-expert coefficient. No complete eight-task table or average is claimed.",
        )
    comp = read("results/mean_compensation/components.json")
    rows = []
    labels = {
        "task_inside": "Task-inside matrices only",
        "task_outside": "Task-outside matrices only",
        "all_matrices": "All changed matrices",
        "other_parameters": "Other parameters only",
        "complete": "Complete joint change",
    }
    for r in comp["rows"]:
        rows.append(
            [
                labels[r["name"]],
                f"{r['correct']}/{r['n']}",
                str(r["metric"]["changed"]),
                f"{r['metric']['worst_task_drift']:.6g}",
            ]
        )
    table(
        "ablation",
        "Previous Defense 2 candidate: source-only ablations",
        ["Counterfactual", "Correct", "Prediction changes", "Max. relative gap change"],
        rows,
        "The fixed512 target-training source images, not a test table. Matrix inside/outside components use the same original rank32 task basis; non-matrix changes are separate. Each counterfactual is quantized separately. Complete-release replay and independent NumPy recount both have zero error. No candidate was repaired or selected from these counterfactual outcomes.",
    )
    joint_path = "results/data_free_compensation/pilot_evaluation.json"
    joint_check = Path("results/data_free_compensation/pilot_evaluation_validation.json")
    if joint_check.exists():
        check = read(str(joint_check))
        execution = read("results/data_free_compensation/pilot_evaluation_execution.json")
        assert check["status"] == "passed"
        assert check["raw_sha256"] == execution["predictions_sha256"]
        rows = []
        for name, label in [
            ("Unprotected", "Unprotected"),
            ("Joint_MNIST", "Joint numerical pilot"),
        ]:
            rows.append(
                [
                    label,
                    "--" if name == "Unprotected" else "0",
                    f"{acc(pick(joint_path, defense=name, merger='Standalone', dataset='MNIST')):.2f}",
                ]
                + [pair(joint_path, defense=name, merger="TA", attack_scope=s) for s in scopes]
            )
        table(
            "data_free_joint",
            "Data-free joint numerical compensation: fixed pilot",
            [
                "Model",
                "Construction images",
                "SA M",
                "TA M / S",
                "AGA target M / S",
                "AGA both M / S",
            ],
            rows,
            "One first-layer full-width MLP, 8192 generated LN vectors per domain, frozen teacher and 20 fixed joint iterations. No real construction images. A/B relative changes 1.0000/0.4599; combined logical distance 1.1007, separately declared from the historical 0.5 comparison. Synthetic 1-percent RMS and strict relative-margin criteria failed. Complete MNIST standalone loss is 0.08 percentage points. TA coefficient 0.5 and original AGA; target/partner accuracies are both reported. All 14 records and 224 original batch-context score replays passed with zero error. Fusion protection failed; this pilot is closed, not a completed Defense 2 method.",
        )
    coverage = [
        [
            "TA + original AGA",
            "Completed: two-expert pairs; target/both recovery scopes; protected and partner accuracy.",
        ],
        [
            "Defense 1",
            "MNIST, DTD, SVHN completed; targeted cleanup and fixed-budget hard-label recovery completed with separate permissions/numerics.",
        ],
        [
            "Defense 2",
            "Effective fixed data-free structural scheme verified on MNIST/DTD/SVHN. Task-SVD-guided construction and new method contribution remain under study. Later failed candidates and historical O/B/A/B task-bound failures remain distinct.",
        ],
        [
            "PaRaMS",
            "Full-default protection completed for MNIST/DTD/SVHN, including matched TA and original AGA. Attention-only ablation completed for MNIST; diagonal recovery is a native adaptation.",
        ],
        [
            "MergeLock",
            "Not run: official repository currently contains only README/Coming Soon. No invented implementation or paper scores.",
        ],
        [
            "MergeGuard / ARF",
            "Full defenses not run; training-stage conditions differ. Unmask formula was tested separately; GradErase not reproduced.",
        ],
        [
            "MergeBarrier",
            "Full defense not run; architecture adaptation and verified implementation missing.",
        ],
        [
            "Eight-task benchmark",
            "Eight experts participated; protected/evaluated targets cover MNIST/DTD/SVHN. Complete protected cases and evaluation for SUN397/Cars/RESISC45/EuroSAT/GTSRB are missing.",
        ],
        [
            "Other mergers",
            "TIES, three DARE masks, NUFILT and ProDistill completed in documented scopes. Full TSV-Merge, CAT and LOT not run.",
        ],
        [
            "Reliability",
            "Per-class/pair counts and source audits retained. Only DARE has the stated three-mask comparison; do not present deterministic rows as multi-seed results.",
        ],
    ]
    if pairing_complete:
        coverage[2][1] = (
            "Fixed structural scheme verified on three protected tasks. A task-SVD-directed construction now has a complete MNIST native/TA/originalAGA check and a parameter-only distance ablation. New construction cross-task evidence,novelty and known-family recovery remain separate."
        )
    if pairing_other_complete:
        coverage[2][1] = (
            "Fixed structure and the frozen task-SVD-directed construction verified on MNIST/DTD/SVHN. Parameter-only distance ablation retained. Method contribution and known-family recovery remain separate; not a complete eight-task benchmark."
        )
    if (recovery_root / "validation.json").exists():
        coverage[2][1] += (
            " Known-family public patch matching restores all158 tensors for each target and matched control; no true task vector or secret order is supplied."
        )
    if cross_complete:
        coverage[2][1] += (
            " Same three releases evaluated in all six task-pair orientations; the independent DTD/SVHN pair is now included."
        )
    if distance_complete:
        coverage[2][1] = coverage[2][1].replace(
            "Parameter-only distance ablation retained.",
            "Fixed distance-objective classification ablation completed.",
        )
    if (distance_other / "evaluation/full/01_SVHN_MNIST/independent_validation.json").exists():
        coverage[2][1] += (
            " The fixed objective classification ablation covers all three protected tasks."
        )
    if amplitude_complete:
        coverage[2][1] += (
            " Matching96/192/384-pair extent controls completed onMNIST;not independent seeds."
        )
    if current_eight_complete:
        coverage[2][1] += (
            " The same three task-directed releases also completed author-order eight-expert TA0.2 and originalAGA target-only,with three evaluated datasets."
        )
    if numerical:
        coverage[2][1] += (
            " Numerical12MLP reconstruction and fixed SVD/prefix removals completed. MNIST retains99.50 accuracy;SVHN loses1.21pp andDTD retention fails. Public-module reset recovers94.47MNIST merged accuracy;coefficient and eight-expert results are weaker thanTA0.5. Original small-sample/strict-gap failures retained. SVD classification contribution and robust task retention remain unresolved."
        )
    if public_path.exists():
        coverage[2][1] += (
            " A separately registered public-backbone construction retains99.47MNIST with originalAGA11.57;known-module reset84.63 is compared with84.06 without the target. Same-rule SVHN95.94 andDTD41.81 fail task retention. With a DTD partner, MNIST originalAGA is18.54;eight-expert originalAGA is44.19. No release is selected or retuned from these observations."
        )
    table(
        "coverage",
        "Completed work and remaining comparisons",
        ["Item", "Status"],
        coverage,
        "This is a completion inventory, not an accuracy table. Missing methods are not assigned zero accuracy. More merger names do not increase the number of recovery attacks.",
        r">{\raggedright\arraybackslash}p{37mm}>{\raggedright\arraybackslash}p{132mm}",
    )
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "pdf.fonttype": 42,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    shown = [
        r[0]
        .replace("PaRaMS (full)", "PaRaMS")
        .replace("Defense 2 candidate (failed) [F]", "Previous candidate")
        .replace(" [K]", "")
        .replace(" [H]", " (historical)")
        for r in tables[0]["rows"]
    ]
    fig, axes = plt.subplots(
        1, 2, figsize=(9.5, max(4.7, 0.5 * len(shown))), sharey=True, layout="constrained"
    )
    for task, ax, title in zip(range(2), axes, ["Protected task: MNIST", "Partner task: SVHN"]):
        for j, (label, color) in enumerate(
            zip(["Direct TA", "AGA target", "AGA both"], ["#bfd3e6", "#6baed6", "#2171b5"])
        ):
            values = [float(r[2 + j].split(" / ")[task]) for r in tables[0]["rows"]]
            ax.barh(
                np.arange(len(shown)) + (j - 1) * 0.23,
                values,
                height=0.21,
                label=label,
                color=color,
                edgecolor="white",
                linewidth=0.3,
            )
        ax.set_yticks(range(len(shown)), shown)
        ax.set_xlim(0, 100)
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("Accuracy (%)")
        ax.set_title(title, loc="left", fontsize=10)
        ax.grid(axis="x", color=".9", linewidth=0.5)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.07), ncol=3, frameon=False
    )
    fig.savefig(
        out / "comparison.pdf",
        metadata={"CreationDate": None, "ModDate": None},
        bbox_inches="tight",
    )
    fig.savefig(out / "comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    figure = r"\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{comparison.pdf}\caption{Two-expert TA0.5 before and after original AGA. Both protected and partner tasks are shown. Every bar is one fixed configuration; historical O/B and A/B task-bound failures remain labelled. Exact numbers appear in Table~\ref{tab:main}.}\label{fig:comparison}\end{figure}"
    if amplitude_complete:
        data = next(t["rows"] for t in tables if t["key"] == "amplitude")
        fig, ax = plt.subplots(figsize=(5.4, 3.6), layout="constrained")
        for label, color in [("Task-directed", "#2171b5"), ("Distance-only", "#d95f02")]:
            selected = [r for r in data if r[0] in ["Unprotected", label]]
            ax.plot(
                [int(r[1]) for r in selected],
                [float(r[5].split(" / ")[0]) for r in selected],
                marker="o",
                color=color,
                label=label,
            )
        ax.set(
            xlabel="Swapped coordinate pairs",
            ylabel="Protected-task accuracy (%)",
            xticks=[0, 96, 192, 384],
            ylim=(0, 103),
        )
        ax.grid(axis="y", color=".9", linewidth=0.5)
        ax.legend(frameon=False)
        fig.savefig(
            out / "amplitude.pdf",
            metadata={"CreationDate": None, "ModDate": None},
            bbox_inches="tight",
        )
        fig.savefig(out / "amplitude.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        figure += r"\begin{figure}[!htbp]\centering\includegraphics[width=0.75\linewidth]{amplitude.pdf}\caption{Modification extent on MNIST with SVHN as partner. Original AGA processes the protected expert before TA0.5. Each curve uses one fixed construction order;0/384 scores are reused. Standalone accuracy is99.69\% at every level. Full protected/partner results appear in Table~\ref{tab:amplitude}.}\label{fig:amplitude}\end{figure}"
    (out / "figure.tex").write_text(figure + "\n")
    md = [
        "# 论文实验表汇总",
        "",
        "主实验采用TA融合与原版AGA。以下均为本项目实际结果，不是原论文的数值。**对比尚未全部完成，缺项见最后一表。**",
        "",
        "表格形式参考AGA原文v1第9页Table1：明确保护方法与攻击状态、数据集列、独立未保护控制；使用booktabs横线与简短表注。此处内容、评价范围与原论文不同。",
        "",
        "[AGA原文](https://arxiv.org/pdf/2606.30360v1)。MergeLock仓库于2026-10-23核对仍仅README；未把其论文结果填入我们的表。",
        "",
        "TeX正文为英文，Markdown解释使用中文。SA为单独使用准确率；M/D/S分别为MNIST/DTD/SVHN；T/P为该行目标/伙伴。所有准确率单位为百分比。",
        "",
    ]
    tex = [
        "% Generated from frozen, previously audited project records. Requires booktabs,array,graphicx.",
        "% Accuracy in percent. No new model evaluation or raw-score audit.",
    ]
    chinese = {
        "data_free_joint": (
            "无真实图片的联合数值补偿：固定先行实验",
            "只修改第一层完整MLP的A/B；构造输入为生成的LN向量，未用真实图、伙伴或分类分数选参。独立MNIST从99.69降至99.61%，但TA及原AGA后仍约99%，没有保护效果。完整相对间隔和合成RMS原标准失败均保留。不把它与使用真实训练图的旧12层数值结果混在一起，也不因距离增大称为有效防御。",
        ),
        "numerical_reconstruction": (
            "数值补偿的完整跨任务结果",
            "同一任务使用同一构造集，全部固定结果保留。M/S/D分别为MNIST/SVHN/DTD；目标与伙伴分列。原方法、增加Attention补偿和改为同输入教师函数重构分别比较，不根据测试成绩替换主模型。DTD独立准确率明显下降的行是任务保持失败，不能因为融合准确率低就称成功。",
        ),
        "numerical_coefficient": (
            "同一数值发布在不同TA系数下的结果",
            "唯一主发布357不变。0.2和0.8是在0.5结果之后固定的两项补充观察，没有按成绩选择新防御。MNIST独立准确率均99.50%，无保护99.69%。0.5上的低准确率不能代表其他系数同样有效。",
        ),
        "numerical_reset": (
            "知道修改模块后的公开基座恢复",
            "仅使用发布模型、公开基座及12个MLP修改位置，替换其48个权重和偏置；没有原专家、真实任务向量、秘密变换或图片。同一操作处理无保护控制后得到完全相同的158个张量，因此只需评价一份相同模型。该恢复牺牲2.04个百分点独立准确率，使AGA后融合恢复到94.47%；与原版AGA单独列出。",
        ),
        "numerical_eight": (
            "同一数值发布：八专家融合、三个数据集评价",
            "只有MNIST受保护，DTD和SVHN是被评价的伙伴。作者顺序、TA系数0.2和原版AGA仅保护专家均固定。没有重构新防御或重复AGA，不报完整八任务表或平均。",
        ),
        "numerical_ablation": (
            "数值版防御二：SVD与逐层补偿消融",
            "相同1024张训练构造图、12层、rank32和0.01阻尼。Task probe为原任务右奇异方向上的相对参数改动，不是分类功能不变的证明。去掉SVD仍有效，不能把保护全归功于SVD；逐层补偿的结果与去除控制直接对照。完整方法主点冻结，不按消融成绩替换。独立准确率损失与原严格间隔/小样本失败分别报告。",
        ),
        "numerical_components": (
            "数值版防御二：同一源上的分量检查",
            "仅1024张构造训练图。A单改和B单改直接取主模型对应矩阵；去掉重构保留A及拟合前B传输；联合分数复用原构造缓存。不据这些控制再拟合或选择发布，不把源结果冒充独立测试。",
        ),
        "cross_pairs": (
            "同一防御二：三个任务的六个保护方向",
            "每格为目标 / 伙伴。六方向对应三种无序任务组合，不是六个独立数据集。三份发布和原作者AGA权重均冻结；新增评价是在旧结果之后登记，没有改排序、rank、目标或参数。TA0.5、原AGA两范围、各自无保护控制、逐类配对及原128批上下文重放均保留。测试集历史暴露，不称盲测；已知方式恢复会还原这些模型，另表报告。",
        ),
        "task_directed_recovery": (
            "任务方向防御二：已知修改方式的针对性恢复",
            "每格为目标 / 伙伴。两种设置都只给发布权重与公开基座；针对性匹配另知道全网残差坐标换序和CLIP结构，不给原专家、真实任务向量、秘密排列或图片。三个恢复模型及无保护控制均与原专家全部158张量精确相同，因此复用已核原分数，不称新图片前向。原版AGA与该针对性恢复分别报告，不外推所有防御。",
        ),
        "three_tasks": (
            "主对比：三个保护任务、相同TA与原版AGA",
            "每格为该行受保护任务 / 伙伴准确率。PaRaMS保留官方完整默认设置；固定结构与任务方向构造分列，后者使用同一有限目标和各自原任务SVD。三保护任务对应两个无序任务组合，不冒充完整八任务表。方法贡献、已知结构和已知位置等针对性恢复单独报告。参数改动尺度和数值协议分列，不通过裁剪基线追求同参数距离。",
        ),
        "main": (
            "主实验：双专家TA融合与原版AGA",
            "每格为受保护MNIST / 伙伴SVHN。H表示旧O/B、A/B不符合原任务间隔限制；K为已知全网置换对照；F为先前失败候选。完整PaRaMS保留。新增数值版使用1024目标训练图校准、12次线性求解、不微调原专家；独立准确率下降0.19个百分点，原严格间隔检查未通过，不能称函数完全相同。",
        ),
        "defense1": (
            "Defense 1：三个保护任务",
            "每格为该行目标 / 伙伴，均为TA0.5。原版AGA不修改最终LayerNorm偏置；这些低分不等于能抵抗所有恢复。",
        ),
        "recovery": (
            "恢复权限对比",
            "不同恢复拥有的信息与数据不同，不能混称原版AGA。蒸馏使用2048张同分布训练图片和学生训练；两个教师的查询标签相同，因此只训练了一个问题。偏置清理知道修改位置；原生FP32前后分数差超限的旧记录保留。",
        ),
        "mergers": (
            "补充：更换融合方法",
            "均先执行原版AGA仅处理保护专家，每格MNIST / SVHN，各融合器各用自己的无保护控制。DARE为三个掩码种子的均值；NUFILT两种顺序分开；ProDistill另需每任务64张无标签训练校准图片和攻击方融合系数训练。融合方法不算新增恢复攻击。",
        ),
        "dare": (
            "DARE：三个固定掩码的波动",
            "均值及样本标准差来自42、43、44三个掩码，不能称三个独立专家训练或三次防御训练。",
        ),
        "other_tasks": (
            "补充：DTD、SVHN保护专家的融合比较",
            "每格为该行目标 / MNIST伙伴。NUFILT保留两种顺序；SVHN无恢复、双方AGA的单元复用此前相同融合模型并交换任务列，不重复训练。ProDistill需要每任务64张无标签校准图和融合系数训练，不能算新增无数据恢复攻击。",
        ),
        "eight": (
            "八专家参与、三个数据集评价",
            "TA0.2，每格固定MNIST / DTD / SVHN顺序。任务方向防御二使用双专家实验的同一三份发布，未改参数；原版AGA只处理保护专家。其余五个数据集未评价，不能写完整八任务表或八任务均值；此处同时改变参与数与系数，不当作孤立系数消融。已知方式恢复的限制仍保留。",
        ),
        "unmask": (
            "Unmask：独立的八专家恢复设置",
            "按论文公式，从融合参数减去0.3倍整份发布专家；未保护控制也执行同样操作，不与AGA组合。这里是公式适配而非官方完整复现。未保护模型也会崩溃，不能只凭绝对低分宣布防御成功或知识恢复。",
        ),
        "cleanup": (
            "针对偏置的恢复：数值协议分列",
            "每格目标 / 伙伴。MNIST采用原生FP32；DTD、SVHN采用预先固定的FP32前缀、FP64最终读出。两类不可混表。原生DTD数值控制失败与MNIST分数差超限都保留，不声称完整FP32函数严格相同。",
        ),
        "geometry": (
            "任务方向内外变化与真实融合效果",
            "相同74个矩阵、原专家减基座的32个任务右奇异方向；矩阵距离不含偏置、LayerNorm与embedding。任务外距离相近，但融合效果差别很大，不能把距离本身当作保护效果。Distance-only pairing为固定消融，不按分类成绩择优；未实测项留空。",
        ),
        "distance_ablation": (
            "固定消融：去除任务方向项",
            "三任务使用相同的目标、基座、74矩阵、rank32及贪心规则，仅去除任务方向项。对照顺序在主方法读图前已固定，分类补测随后登记。SVD并非每个任务都更好；完整目标和伙伴成绩保留，不按结果换发布。",
        ),
        "amplitude": (
            "修改幅度：同一固定顺序的不同交换数量",
            "保护MNIST，伙伴SVHN；融合列为目标 / 伙伴准确率。沿用各固定顺序的前96、192、384对交换，0和384结果直接复用，只新增四个对照。参数距离按逻辑矩阵及各偏置计算；保持独立能力，不按成绩选顺序或幅度。已知方式恢复的限制仍保留。",
        ),
        "ablation": (
            "旧候选：任务内外与联合补偿消融",
            "固定512张构造训练图，不是测试结果。仅矩阵任务内、仅任务外、全部矩阵、其他参数与完整联合共五项。完整联合保持原能力，但真实AGA后融合仍接近99%；没有根据消融改参数。",
        ),
        "coverage": (
            "已完成与尚缺的实验",
            "这是一份完成情况清单，不能说所有对比已做完；缺项不填零分，也不拿别人的论文成绩冒充我们的实测。",
        ),
    }
    translations = {
        "Unprotected": "无保护",
        "Defense 1": "防御一",
        "Global permutation [K]": "全网置换 [K]",
        "Defense 2 candidate (failed) [F]": "先前失败候选 [F]",
        "Defense 2 (structure) [K]": "防御二（固定结构）[K]",
        "Defense 2 (task-directed)": "防御二（任务方向构造）",
        "Without family matching": "不作针对性匹配",
        "Known-family matching": "已知方式匹配",
    }
    recovery_section = None
    for i, t in enumerate(tables):
        title, note = chinese[t["key"]]
        md += [
            "## " + str(i + 1) + ". " + title,
            "",
            "| " + " | ".join(t["headers"]) + " |",
            "|" + "---|" * len(t["headers"]),
        ]
        md += ["| " + " | ".join(translations.get(x, x) for x in row) + " |" for row in t["rows"]]
        md += ["", note, ""]

        def escape(s):
            return (
                s.replace("→", r"$\to$")
                .replace("&", r"\&")
                .replace("%", r"\%")
                .replace("_", r"\_")
                .replace("#", r"\#")
                .replace(" +/- ", r" $\pm$ ")
            )

        tex += [
            r"\begin{table}[!htbp]",
            r"\centering",
            r"\small",
            r"\caption{" + escape(t["title"]) + "}",
            r"\label{tab:" + t["key"] + "}",
            r"\begin{tabular}{" + t["colspec"] + "}",
            r"\toprule",
            " & ".join(escape(x) for x in t["headers"]) + r" \\",
            r"\midrule",
        ]
        tex += [" & ".join(escape(x) for x in row) + r" \\" for row in t["rows"]]
        tex += [
            r"\bottomrule",
            r"\end{tabular}",
            r"\par\vspace{3pt}\begin{minipage}{0.98\linewidth}\footnotesize "
            + escape(t["note"])
            + r"\end{minipage}",
            r"\end{table}",
            "",
        ]
        if t["key"] == "task_directed_recovery":
            recovery_section = True
    md += [
        "## 图1. 同一组模型的直接融合与AGA后融合",
        "",
        "![主实验准确率](comparison.png)",
        "",
        "目标任务和伙伴任务分别显示；每个柱来自固定配置，未添加不存在的随机重复误差条。详细数值见表1。",
        "",
    ]
    if amplitude_complete:
        md += [
            "## 图2. 修改幅度与原版AGA后的准确率",
            "",
            "![修改幅度](amplitude.png)",
            "",
            "MNIST保护、SVHN伙伴，TA0.5；原版AGA只处理保护专家。沿用同一固定构造顺序，独立准确率均99.69%。完整目标与伙伴成绩见修改幅度表。",
            "",
        ]
    (out / "tables.md").write_text("\n".join(md) + "\n")
    (out / "tables.tex").write_text("\n".join(tex) + "\n")
    (out / "table_data.json").write_text(json.dumps(tables, ensure_ascii=False, indent=2) + "\n")
    (out / "README.md").write_text("""# 实验表与完成情况

**没有全部做完。** 已完成的真实结果已经生成同一份TeX和Markdown；缺项明确保留。

- [Markdown总表](tables.md)：主实验、三个保护任务、不同恢复权限、融合方法、八专家三数据集、任务内外与补偿消融、缺项清单。
- [TeX表格](tables.tex)：可取对应table段落放入论文，需要booktabs、array；宽表放在两栏论文时改为table*。
- [完整TeX预览入口](main.tex)（当前源码已通过桌面编译）。图的[TeX片段](figure.tex)、[PDF](comparison.pdf)和[PNG](comparison.png)同数据生成。
- [数值与来源](table_data.json)、[输入哈希与原记录行号](table_sources.json)。新制表没有重跑GPU，也没有冒充新一轮原始分数复核。

已完成：三个保护任务的相同TA／原版AGA下无保护、Defense1、完整默认PaRaMS和有效Defense2结构方案对比；对角适配、位置已知偏置清理、MNIST固定数据蒸馏、TIES/DARE/NUFILT/ProDistill、八专家参与的三任务评价及Unmask公式适配。

尚缺：Defense2任务方向参与构造及新方法贡献的充分验证；独立DTD/SVHN组合；完整八个保护任务和八数据集成绩；MergeLock、完整MergeGuard/ARF/MergeBarrier防御基线；GradErase；更多独立防御重复。MergeLock官方库此前核对仅README，其他未运行方法不填论文成绩。不同训练阶段/架构/权限须另列。完整TSV-Merge、CAT、LOT也未运行，不把融合器数量当攻击数量。

历史完整分表保留，不用本次摘要替代：

- [TA/TIES完整双任务表](../phase3/tables.md)
- [PaRaMS对角恢复](../phase3/params_adaptive/tables.md)
- [DARE三个掩码](../phase3/dare/tables.md)
- [NUFILT O/B补充](../phase3/nufilt_OB/tables.md)
- [NUFILT DTD/SVHN](../phase3/nufilt_other_tasks/tables.md)
- [ProDistill MNIST](../phase3/prodistill/tables.md)及[DTD/SVHN](../phase3/prodistill_other_tasks/tables.md)
- [公开基座/单专家知识来源对照](../phase3/controls/single/tables.md)及[固定七专家去目标对照](../phase3/controls/leave_out/tables.md)
- [位置已知清理](../phase3/bias_canonicalization/tables.md)、[单列读出精度协议](../phase3/bias_tail_precision/tables.md)、[固定数据蒸馏](../phase3/distillation/tables.md)
- [八专家三数据集](../phase3/eight_task/tables.md)、[Unmask](../phase3/unmask/tables.md)
- [先前补偿失败对照](../mean_compensation/summary.md)、[已知全网置换对照](../residual_coordinate_control/summary.md)

重建表和图：`python -m experiment.report --output results/paper_tables`。本机系统Matplotlib与用户NumPy版本不同，当前使用`PYTHONNOUSERSITE=1`调用匹配的系统环境；没有修改3090科研环境。main.tex可由桌面LaTeX编辑器编译。
""")
    if pairing_complete:
        p = out / "README.md"
        p.write_text(
            p.read_text().replace(
                "尚缺：Defense2任务方向参与构造及新方法贡献的充分验证；",
                "已完成任务SVD参与构造的MNIST闭环；尚缺新构造的跨任务及新方法贡献验证；",
            )
        )
    if pairing_other_complete:
        p = out / "README.md"
        p.write_text(
            p.read_text().replace(
                "已完成任务SVD参与构造的MNIST闭环；尚缺新构造的跨任务及新方法贡献验证；",
                "已完成同一任务SVD构造的MNIST／DTD／SVHN验证；尚缺新方法贡献的充分验证；",
            )
        )
    if recovery_section:
        p = out / "README.md"
        p.write_text(
            p.read_text()
            + "\n当前任务方向Defense2的[已知方式恢复](../task_directed_pairing/recovery/summary.md)已完成：不使用原专家、真实任务向量、秘密排列或图片参与攻击，三模型及匹配控制全部158张量精确还原。原版AGA低分仍保留；已知方式恢复是单独结果，不能声称当前方法抵抗它。\n"
        )
    if cross_complete:
        p = out / "README.md"
        p.write_text(
            p.read_text().replace("独立DTD/SVHN组合；", "")
            + "\n同一三份发布的[六个保护方向](../task_directed_pairing/cross_pairs/summary.md)已完成，包含独立DTD/SVHN组合；未新增模型或按伙伴成绩调参。\n"
        )
    if current_eight_complete:
        p = out / "README.md"
        p.write_text(
            p.read_text()
            + "\n同一任务方向防御二的[八专家TA0.2结果](../task_directed_pairing/eight_expert/summary.md)已完成：三个数据集及原版AGA仅保护专家，保持原发布与已知方式恢复限制；不报完整八任务平均。\n"
        )
    if amplitude_complete:
        p = out / "README.md"
        p.write_text(
            p.read_text()
            + "\n[修改幅度与任务方向对照](../task_directed_pairing/amplitude_ablation/summary.md)已完成：同一固定顺序的96／192／384对交换，两种目标分别报告，0与384模型复用；不作为独立种子或新防御。\n"
        )
    if public_path.exists():
        p = out / "README.md"
        p.write_text(
            p.read_text()
            + "\n[公开基座数值补偿](../nonlinear_public_backbone/summary.md)已完成MNIST固定点：独立99.47%、TA15.65%、原AGA11.57%/11.52%。已知模块重置84.63%与无目标基线84.06%同时报告，不把伙伴已有能力算成恢复目标知识。SVHN/DTD同规则独立95.94%/41.81%，任务保持失败；DTD伙伴原AGA18.54%，八专家44.19%。原357不变。\n"
        )
    if numerical:
        p = out / "README.md"
        p.write_text(
            p.read_text()
            + "\n[数值版防御二](../nonlinear_prefix/summary.md)已完成MNIST及同规则SVHN/DTD、SVD/前缀消融、源分量、公开模块恢复、两项TA系数和八专家三数据集评价。MNIST独立99.50%、TA0.5原AGA后13.02%是实测有效结果；DTD独立43.46%是明确失败。去掉SVD仍有效，尚未证明SVD分类贡献。已知模块后的公开基座替换恢复融合94.47%，其他系数和八专家下保护较弱，均与主表分列。固定重构目标对照完整列出，不按成绩替换主发布。\n"
        )
    manifest = dict(
        status="completed_table_collection",
        report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=sources,
        selected_rows=used,
        tables=len(tables),
        new_GPU_forwards=0,
        new_raw_score_revalidation=False,
        scope="Checks imported integer/class totals and joins; reuses prior independent raw-score audits. Not a new attack/defense experiment.",
        style_reference=dict(
            paper="https://arxiv.org/pdf/2606.30360v1", page=9, table=1, visually_checked=True
        ),
        output_sha256={
            p: hashlib.sha256((out / p).read_bytes()).hexdigest()
            for p in [
                "README.md",
                "tables.md",
                "tables.tex",
                "table_data.json",
                "figure.tex",
                "comparison.pdf",
                "comparison.png",
            ]
            + (["amplitude.pdf", "amplitude.png"] if amplitude_complete else [])
        },
    )
    (out / "table_sources.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    print("PAPER_TABLES_WRITTEN", len(tables), "source selections", len(used))
