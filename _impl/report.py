"""Paper MD/TeX tables and figures from independently verified stored records."""

import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def paper_tables(_source, output):
    """Collect frozen comparisons into matching TeX/Markdown paper tables."""
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
    table(
        "main",
        "Main comparison: two experts, TA coefficient 0.5",
        ["Protection", "Standalone M", "No recovery M / S", "AGA target M / S", "AGA both M / S"],
        rows,
        "M: protected MNIST; S: partner SVHN. AGA target/both denotes recovery of only the protected expert/both experts. [H] Historical O/B and A/B fail the original complete-task bound; effect references only. [K] Known global symmetry control, not an original method; family-aware public recovery is reported separately. [F] Previous candidate preserves the complete task check but does not provide useful protection. Full PaRaMS is the full-default baseline; parameter scales differ and are not a universal exclusion rule.",
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
            "Same1024construction images,12layers,rank32,damping0.01 and parameter ranges. Task probe is max ||DeltaW V32||/||DeltaW|| on original teacher bases,not a functional guarantee. Without SVD removes it from both construction and output correction;without prefix correction fits original teacher MLP output using actual student inputs. Report every control,no replacement of main357. Main full accuracy loss0.19pp;fresh512 source-independent gate failed and strictR0.5 failed,retained separately. All are calibration-based numerical changes,not exact permutation identities.",
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
            "Same1024target-training images and frozen main357 point. A-only and B-only use literal subsets of its release. No-reconstruction retains all A changes and their pre-fit B transports;no weights refitted or selected. Joint scores reuse the frozen calibration cache. Source observations,not independent test results.",
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
            " Numerical12MLP reconstruction now has measured native/TA/originalAGA outcomes and fixed SVD/prefix removals. Original small-sample/strict-gap failures are retained;standalone accuracy and calibration access are reported explicitly. Cross-task,coefficient and public-module-reset evidence is separate."
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
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 4.7), sharey=True, layout="constrained")
    shown = [
        "Unprotected",
        "PaRaMS",
        "Defense 1",
        "O/B (historical)",
        "A/B (historical)",
        "Defense 2 (structure)",
        "Previous candidate",
    ]
    if pairing_complete:
        shown.append("Defense 2 (task-directed)")
    if numerical:
        shown.append("Defense 2 (numerical)")
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
        "[AGA原文](https://arxiv.org/pdf/2606.30360v1)。MergeLock仓库于2026-10-21核对仍仅README；未把其论文结果填入我们的表。",
        "",
        "TeX正文为英文，Markdown解释使用中文。SA为单独使用准确率；M/D/S分别为MNIST/DTD/SVHN；T/P为该行目标/伙伴。所有准确率单位为百分比。",
        "",
    ]
    tex = [
        "% Generated from frozen, previously audited project records. Requires booktabs,array,graphicx.",
        "% Accuracy in percent. No new model evaluation or raw-score audit.",
    ]
    chinese = {
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
