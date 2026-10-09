"""Locate task dependencies and measure public-anchor functional reconstruction.

No protection is proposed here. Reuse the author evaluation/merger and the
existing closed-form MLP reconstruction on an unprotected released control.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import time
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate, merge
from polynomial_probe import capture, reconstruct
from scipy.stats import binomtest
from subspace_probe import make_loader, save_json
from torchvision.datasets import MNIST, SVHN


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@torch.no_grad()
def replace_components(original, anchor, layers, component):
    result = copy.deepcopy(original).cpu().eval()
    state, base = result.state_dict(), anchor.state_dict()
    keys = []
    for layer in layers:
        prefix = f"model.visual.transformer.resblocks.{layer}."
        for key in state:
            if not key.startswith(prefix):
                continue
            suffix = key.removeprefix(prefix)
            matches = (
                component == "block"
                or (component == "norms" and suffix.startswith(("ln_1.", "ln_2.")))
                or suffix.startswith(component + ".")
                or (component == "attn.in_proj" and suffix.startswith("attn.in_proj_"))
            )
            if matches:
                keys.append(key)
                state[key] = base[key].clone()
    assert keys and len(keys) == len(set(keys))
    result.load_state_dict(state, strict=True)
    return result, keys


@torch.no_grad()
def check_local_fit(released, restored, anchor, images, layer):
    x = capture(released, images, layer)
    modules = []
    outputs = []
    for model in [released, restored, anchor]:
        module = copy.deepcopy(model.model.visual.transformer.resblocks[layer].mlp)
        module = module.double().cuda().eval()
        modules.append(module)
        outputs.append(module(x))
    target, fitted, base = outputs
    error, task_update = (fitted - target).norm(), (target - base).norm()
    result = dict(
        check_tokens=len(x),
        output_error_norm=float(error),
        target_output_norm=float(target.norm()),
        task_update_output_norm=float(task_update),
        relative_full_output_error=float(error / target.norm().clamp_min(1e-12)),
        relative_task_update_error=float(error / task_update.clamp_min(1e-12)),
        check_source="independent defender calibration images; frozen released inputs",
    )
    del modules, outputs
    return result


def adjust_comparisons(rows):
    """Holm adjustment within each split/group/dataset family, excluding control."""
    families = {}
    for row in rows:
        if row["preset"] != "unprotected":
            families.setdefault((row["split"], row["group"], row["dataset"]), []).append(row)
    for family in families.values():
        running = 0.0
        for rank, row in enumerate(sorted(family, key=lambda r: r["paired"]["exact_mcnemar_p"])):
            running = max(running, (len(family) - rank) * row["paired"]["exact_mcnemar_p"])
            row["paired"]["holm_p"] = min(running, 1.0)
            row["paired"]["family_size"] = len(family)


def paired_counts(reference, correct):
    lost = int((reference & ~correct).sum())
    gained = int((~reference & correct).sum())
    return dict(
        both_correct=int((reference & correct).sum()),
        lost=lost,
        gained=gained,
        both_wrong=int((~reference & ~correct).sum()),
        loss_pp=100 * (lost - gained) / len(correct),
        exact_mcnemar_p=float(binomtest(lost, lost + gained).pvalue) if lost + gained else 1.0,
    )


def signed_margin(logits, labels):
    target = logits.gather(1, labels[:, None]).squeeze(1)
    competing = logits.clone()
    competing.scatter_(1, labels[:, None], -torch.inf)
    return target - competing.max(1).values


@torch.no_grad()
def run_budget(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    assert config["task"] == "SVHN" and config["budgets"] == [32, 128, 256]
    assert config["test_usage"] == "none"
    (out / "protocol.json").write_bytes(config_bytes)
    parent = Path(config["parent"])
    previous = json.loads((parent / "training_indices.json").read_bytes())
    fit_parent = Path(config["fitting_source"])
    frozen = json.loads((fit_parent / "training_indices.json").read_bytes())
    old_meta = json.loads((parent / "metadata.json").read_bytes())
    old_predictions_path = parent / "artifacts/predictions.pt"
    assert digest(old_predictions_path) == old_meta["predictions_sha256"]
    old_predictions = torch.load(old_predictions_path, map_location="cpu", weights_only=True)
    indices = dict(
        development={"MNIST": previous["development"]["MNIST"]},
        confirmation=previous["confirmation"],
        fitting={str(n): frozen["SVHN"]["attack"][:n] for n in config["budgets"]},
        local_check={"SVHN": frozen["SVHN"]["calibration"][: config["local_check_samples"]]},
    )
    fit_set = set(indices["fitting"]["256"])
    assert len(fit_set) == 256
    assert not fit_set.intersection(indices["confirmation"]["SVHN"])
    assert not fit_set.intersection(indices["local_check"]["SVHN"])
    assert indices["fitting"]["32"] == previous["fitting"]["SVHN"]
    save_json(out / "training_indices.json", indices)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "SVHN/finetuned.pt"))
    second = load(str(root / "MNIST/finetuned.pt"))
    heads = {task: load(str(root / f"head_{task}.pt")) for task in ["MNIST", "SVHN"]}
    training = dict(
        MNIST=MNIST(args.data, train=True, transform=original.val_preprocess),
        SVHN=SVHN(args.data, split="train", transform=original.val_preprocess),
    )
    loaders = {
        (split, task): make_loader(training[task], subset)
        for split in ["development", "confirmation"]
        for task, subset in indices[split].items()
    }
    fit_images = torch.stack([training["SVHN"][i][0] for i in indices["fitting"]["256"]])
    check_images = torch.stack([training["SVHN"][i][0] for i in indices["local_check"]["SVHN"]])
    rows, controls, predictions, fitting = [], {}, {}, []

    def record(preset, group, model, split, task):
        predicted, labels, scores = [], [], []

        def capture_predictions(prediction, label):
            predicted.append(prediction.clone())
            labels.append(label.clone())

        metric = evaluate(
            model, heads[task], loaders[split, task], "cuda", capture_predictions, scores.append
        )
        prediction, label, logits = torch.cat(predicted), torch.cat(labels), torch.cat(scores)
        correct = prediction == label
        assert int(correct.sum()) == metric["correct"]
        assert torch.equal(prediction, logits.argmax(1))
        key = (split, group, task)
        if preset == "unprotected":
            controls[key] = dict(correct=correct, labels=label, logits=logits)
        control = controls[key]
        assert torch.equal(label, control["labels"])
        margin = signed_margin(logits, label)
        original_margin = signed_margin(control["logits"], label)
        lost = control["correct"] & ~correct
        margins = dict(
            median=float(margin.median()),
            p10=float(torch.quantile(margin.double(), 0.1)),
            relative_score_error=float(
                (logits - control["logits"]).norm() / control["logits"].norm()
            ),
            lost_images_original_margin_median=float(original_margin[lost].median())
            if lost.any()
            else None,
            score_scope="raw fixed-head logits; signed true-class margin, no label use during fitting",
        )
        prior_match = None
        if preset in {"unprotected", "fit_32"}:
            old_name = "unprotected" if preset == "unprotected" else "reconstruct_joint"
            old = old_predictions[(old_name, *key)]
            prior_match = torch.equal(label, old["labels"]) and torch.equal(
                prediction, old["prediction"]
            )
            assert prior_match
        row = dict(
            preset=preset,
            group=group,
            split=split,
            dataset=task,
            coefficient=None if group == "standalone" else config["coefficient"],
            paired=paired_counts(control["correct"], correct),
            margins=margins,
            previous_prediction_exact_match=prior_match,
            **metric,
        )
        rows.append(row)
        predictions[(preset, *key)] = dict(prediction=prediction, labels=label, logits=logits)
        save_json(out / "results.json", rows)
        print("BUDGET_RESULT", json.dumps(row), flush=True)

    def evaluate_variant(preset, model):
        record(preset, "standalone", model, "confirmation", "SVHN")
        combined = merge(anchor, model, second, config["coefficient"])
        record(preset, "direct", combined, "confirmation", "SVHN")
        record(preset, "direct", combined, "development", "MNIST")

    evaluate_variant("unprotected", original)
    for budget in config["budgets"]:
        joint = copy.deepcopy(original).cpu().eval()
        for layer in config["layers"]:
            fit_start = time.perf_counter()
            fitted, details = reconstruct(
                original, anchor, fit_images[:budget], layer, config["ridge_ratio"], solver="auto"
            )
            seconds = time.perf_counter() - fit_start
            check = check_local_fit(original, fitted, anchor, check_images, layer)
            joint.model.visual.transformer.resblocks[layer].mlp.load_state_dict(
                fitted.model.visual.transformer.resblocks[layer].mlp.state_dict(),
                strict=True,
            )
            fitting.append(
                dict(budget=budget, layer=layer, fit=details, check=check, seconds=seconds)
            )
            save_json(out / "fitting.json", fitting)
            del fitted
        torch.save(
            {
                layer: joint.model.visual.transformer.resblocks[layer].mlp.state_dict()
                for layer in config["layers"]
            },
            artifacts / f"fitted_mlp_{budget}.pt",
        )
        evaluate_variant(f"fit_{budget}", joint)
        del joint
    adjust_comparisons(rows)
    save_json(out / "results.json", rows)
    torch.save(predictions, artifacts / "predictions.pt")
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            new_defense=False,
            test_usage="none",
            seed_count=1,
            new_evaluations=len(rows),
            reused_measurements=0,
            evaluation_scope="previously inspected disjoint training subsets, not a fresh holdout",
            optimizer_steps=0,
            fitting_labels_used=False,
            seconds=time.perf_counter() - start,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            indices_sha256=digest(out / "training_indices.json"),
            parent_indices_sha256=digest(parent / "training_indices.json"),
            fitting_parent_indices_sha256=digest(fit_parent / "training_indices.json"),
            parent_predictions_sha256=digest(old_predictions_path),
            source_sha256=digest(__file__),
            helper_sha256={
                name: digest(Path(__file__).with_name(name))
                for name in ["clip_benchmark.py", "polynomial_probe.py", "subspace_probe.py"]
            },
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p)
                for p in [
                    root / "zeroshot.pt",
                    root / "SVHN/finetuned.pt",
                    root / "MNIST/finetuned.pt",
                    root / "head_SVHN.pt",
                    root / "head_MNIST.pt",
                ]
            },
            artifact_sha256={p.name: digest(p) for p in artifacts.glob("*.pt")},
        ),
    )
    print("BUDGET_COMPLETE", flush=True)


@torch.no_grad()
def run_context(args):
    """Separate fixed hidden features from fitting context on a released control."""
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    assert config["conditions"] == [
        "public_expert",
        "public_merged",
        "released_expert",
        "released_merged",
    ]
    assert config["samples"] == 256 and config["test_usage"] == "none"
    (out / "protocol.json").write_bytes(config_bytes)
    parent, fit_parent = Path(config["parent"]), Path(config["fitting_source"])
    previous = json.loads((parent / "training_indices.json").read_bytes())
    frozen = json.loads((fit_parent / "training_indices.json").read_bytes())
    parent_meta = json.loads((parent / "metadata.json").read_bytes())
    parent_artifacts = parent / "artifacts"
    for name in ["predictions.pt", "fitted_mlp_256.pt"]:
        assert digest(parent_artifacts / name) == parent_meta["artifact_sha256"][name]
    old_predictions = torch.load(
        parent_artifacts / "predictions.pt", map_location="cpu", weights_only=True
    )
    public_state = torch.load(
        parent_artifacts / "fitted_mlp_256.pt", map_location="cpu", weights_only=True
    )
    old_fitting = json.loads((parent / "fitting.json").read_bytes())
    indices = dict(
        development=previous["development"],
        confirmation=previous["confirmation"],
        fitting={"SVHN": previous["fitting"]["256"]},
        local_check={task: frozen[task]["calibration"][:32] for task in ["MNIST", "SVHN"]},
    )
    for task in ["MNIST", "SVHN"]:
        checks = set(indices["local_check"][task])
        assert not checks.intersection(indices["development"].get(task, []))
        assert not checks.intersection(indices["confirmation"].get(task, []))
        assert not checks.intersection(indices["fitting"].get(task, []))
    save_json(out / "training_indices.json", indices)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "SVHN/finetuned.pt"))
    second = load(str(root / "MNIST/finetuned.pt"))
    original_merged = merge(anchor, original, second, config["coefficient"])
    heads = {task: load(str(root / f"head_{task}.pt")) for task in ["MNIST", "SVHN"]}
    training = dict(
        MNIST=MNIST(args.data, train=True, transform=original.val_preprocess),
        SVHN=SVHN(args.data, split="train", transform=original.val_preprocess),
    )
    loaders = {
        (split, task): make_loader(training[task], subset)
        for split in ["development", "confirmation"]
        for task, subset in indices[split].items()
    }
    fit_images = torch.stack([training["SVHN"][i][0] for i in indices["fitting"]["SVHN"]])
    check_inputs, shifts = {}, []
    for task, subset in indices["local_check"].items():
        images = torch.stack([training[task][i][0] for i in subset])
        for layer in config["layers"]:
            for context, model in [("expert", original), ("merged", original_merged)]:
                check_inputs[(task, layer, context)] = capture(model, images, layer).cpu()
            expert_x = check_inputs[(task, layer, "expert")]
            merged_x = check_inputs[(task, layer, "merged")]
            shifts.append(
                dict(
                    dataset=task,
                    layer=layer,
                    tokens=len(expert_x),
                    relative_input_shift=float((merged_x - expert_x).norm() / expert_x.norm()),
                )
            )
    torch.save(check_inputs, artifacts / "check_inputs.pt")
    save_json(out / "input_shift.json", shifts)
    rows, predictions, controls, fitting, local = [], {}, {}, [], []

    def record(preset, group, model, split, task):
        predicted, labels, scores = [], [], []

        def sink(prediction, label):
            predicted.append(prediction.clone())
            labels.append(label.clone())

        metric = evaluate(model, heads[task], loaders[split, task], "cuda", sink, scores.append)
        prediction, label, logits = torch.cat(predicted), torch.cat(labels), torch.cat(scores)
        correct = prediction == label
        assert int(correct.sum()) == metric["correct"]
        assert torch.equal(prediction, logits.argmax(1))
        key = (split, group, task)
        if preset == "unprotected":
            controls[key] = dict(correct=correct, labels=label, logits=logits)
        control = controls[key]
        assert torch.equal(label, control["labels"])
        prior_match = None
        if preset in {"unprotected", "public_expert"}:
            previous_name = "unprotected" if preset == "unprotected" else "fit_256"
            prior = old_predictions[(previous_name, *key)]
            prior_match = torch.equal(label, prior["labels"]) and torch.equal(
                prediction, prior["prediction"]
            )
            assert prior_match
        row = dict(
            preset=preset,
            group=group,
            split=split,
            dataset=task,
            coefficient=None if group == "standalone" else config["coefficient"],
            paired=paired_counts(control["correct"], correct),
            relative_score_error=float(
                (logits - control["logits"]).norm() / control["logits"].norm()
            ),
            previous_prediction_exact_match=prior_match,
            **metric,
        )
        rows.append(row)
        predictions[(preset, *key)] = dict(prediction=prediction, labels=label, logits=logits)
        save_json(out / "results.json", rows)
        print("CONTEXT_RESULT", json.dumps(row), flush=True)

    def evaluate_variant(preset, model, combined):
        record(preset, "standalone", model, "confirmation", "SVHN")
        record(preset, "direct", combined, "confirmation", "SVHN")
        record(preset, "direct", combined, "development", "MNIST")

    def local_error(candidate, target, base):
        error = (candidate - target).norm()
        return dict(
            output_error_norm=float(error),
            target_output_norm=float(target.norm()),
            task_update_output_norm=float((target - base).norm()),
            relative_full_output_error=float(error / target.norm().clamp_min(1e-12)),
            relative_task_update_error=float(error / (target - base).norm().clamp_min(1e-12)),
        )

    evaluate_variant("unprotected", original, original_merged)
    for preset in config["conditions"]:
        feature_source, input_source = preset.split("_")
        joint = copy.deepcopy(original).cpu().eval()
        for layer in config["layers"]:
            if preset == "public_expert":
                joint.model.visual.transformer.resblocks[layer].mlp.load_state_dict(
                    public_state[layer]
                )
                previous_fit = next(
                    f for f in old_fitting if f["budget"] == 256 and f["layer"] == layer
                )
                fitting.append(
                    dict(
                        preset=preset,
                        layer=layer,
                        new_fit=False,
                        feature_source=feature_source,
                        input_source=input_source,
                        fit=previous_fit["fit"],
                        seconds=previous_fit["seconds"],
                    )
                )
            else:
                fit_start = time.perf_counter()
                fitted, details = reconstruct(
                    original,
                    anchor,
                    fit_images,
                    layer,
                    config["ridge_ratio"],
                    solver="auto",
                    feature_model=original if feature_source == "released" else None,
                    input_model=original_merged if input_source == "merged" else None,
                )
                seconds = time.perf_counter() - fit_start
                joint.model.visual.transformer.resblocks[layer].mlp.load_state_dict(
                    fitted.model.visual.transformer.resblocks[layer].mlp.state_dict()
                )
                fitting.append(
                    dict(
                        preset=preset,
                        layer=layer,
                        new_fit=True,
                        feature_source=feature_source,
                        input_source=input_source,
                        fit=details,
                        seconds=seconds,
                    )
                )
                del fitted
            save_json(out / "fitting.json", fitting)
        if preset != "public_expert":
            torch.save(
                {
                    layer: joint.model.visual.transformer.resblocks[layer].mlp.state_dict()
                    for layer in config["layers"]
                },
                artifacts / f"fitted_mlp_{preset}.pt",
            )
        combined = merge(anchor, joint, second, config["coefficient"])
        for layer in config["layers"]:
            modules = [
                copy.deepcopy(model.model.visual.transformer.resblocks[layer].mlp).double().cuda()
                for model in [anchor, original, joint, original_merged, combined]
            ]
            for task in ["MNIST", "SVHN"]:
                for context in ["expert", "merged"]:
                    x = check_inputs[(task, layer, context)].cuda()
                    base, target, restored, merged_target, merged_restored = [m(x) for m in modules]
                    local.append(
                        dict(
                            preset=preset,
                            layer=layer,
                            dataset=task,
                            input_context=context,
                            tokens=len(x),
                            expert_operator=local_error(restored, target, base),
                            merged_operator=local_error(merged_restored, merged_target, base),
                        )
                    )
            del modules
        save_json(out / "local_errors.json", local)
        evaluate_variant(preset, joint, combined)
        del joint, combined
    adjust_comparisons(rows)
    save_json(out / "results.json", rows)
    torch.save(predictions, artifacts / "predictions.pt")
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            new_defense=False,
            test_usage="none",
            seed_count=1,
            new_evaluations=len(rows),
            reused_measurements=0,
            new_fits=12,
            reused_fits=4,
            fitting_labels_used=False,
            optimizer_steps=0,
            seconds=time.perf_counter() - start,
            evaluation_scope=config["evaluation"],
            control_scope=config["scope"],
            config_sha256=digest(out / "protocol.json"),
            indices_sha256=digest(out / "training_indices.json"),
            parent_indices_sha256=digest(parent / "training_indices.json"),
            fitting_parent_indices_sha256=digest(fit_parent / "training_indices.json"),
            parent_results_sha256=digest(parent / "results.json"),
            parent_artifact_sha256={
                name: digest(parent_artifacts / name)
                for name in ["predictions.pt", "fitted_mlp_256.pt"]
            },
            source_sha256=digest(__file__),
            helper_sha256={
                name: digest(Path(__file__).with_name(name))
                for name in ["clip_benchmark.py", "polynomial_probe.py", "subspace_probe.py"]
            },
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p)
                for p in [
                    root / "zeroshot.pt",
                    root / "SVHN/finetuned.pt",
                    root / "MNIST/finetuned.pt",
                    root / "head_SVHN.pt",
                    root / "head_MNIST.pt",
                ]
            },
            artifact_sha256={p.name: digest(p) for p in artifacts.glob("*.pt")},
        ),
    )
    print("CONTEXT_COMPLETE", flush=True)


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    assert config["protected_task"] == "SVHN" and config["test_usage"] == "none"
    (out / "protocol.json").write_bytes(config_bytes)
    parent = Path(config["parent"])
    index_bytes = (parent / "training_indices.json").read_bytes()
    frozen = json.loads(index_bytes)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    checkpoints = Path(args.checkpoints)
    anchor = load(str(checkpoints / "zeroshot.pt"))
    original = load(str(checkpoints / "SVHN/finetuned.pt"))
    second = load(str(checkpoints / "MNIST/finetuned.pt"))
    heads = {name: load(str(checkpoints / f"head_{name}.pt")) for name in ["MNIST", "SVHN"]}
    training = {
        "MNIST": MNIST(args.data, train=True, transform=original.val_preprocess),
        "SVHN": SVHN(args.data, split="train", transform=original.val_preprocess),
    }
    excluded = set(itertools.chain.from_iterable(frozen["SVHN"].values()))
    order = torch.randperm(
        len(training["SVHN"]), generator=torch.Generator().manual_seed(config["seed"])
    ).tolist()
    confirmation = [i for i in order if i not in excluded][: config["confirmation_samples"]]
    indices = dict(
        development={task: splits["development"] for task, splits in frozen.items()},
        confirmation={"SVHN": confirmation},
        fitting={"SVHN": frozen["SVHN"]["attack"][: config["attacker_samples"]]},
        local_check={"SVHN": frozen["SVHN"]["calibration"][: config["local_check_samples"]]},
    )
    assert len(confirmation) == config["confirmation_samples"]
    save_json(out / "training_indices.json", indices)
    loaders = {
        split: {task: make_loader(training[task], subset) for task, subset in by_task.items()}
        for split, by_task in indices.items()
        if split in {"development", "confirmation"}
    }
    fit_images = torch.stack([training["SVHN"][i][0] for i in indices["fitting"]["SVHN"]])
    check_images = torch.stack([training["SVHN"][i][0] for i in indices["local_check"]["SVHN"]])
    rows, predictions, controls = [], {}, {}
    profile, fitting = [], []

    def record(preset, group, model, split="development"):
        for task, loader in loaders[split].items():
            if group == "standalone" and task != "SVHN":
                continue
            predicted, labels = [], []

            def capture_predictions(prediction, label):
                predicted.append(prediction.clone())
                labels.append(label.clone())

            metric = evaluate(model, heads[task], loader, "cuda", capture_predictions)
            predicted, labels = torch.cat(predicted), torch.cat(labels)
            correct = predicted == labels
            assert int(correct.sum()) == metric["correct"] and len(correct) == metric["n"]
            key = (split, group, task)
            if preset == "unprotected":
                controls[key] = (correct, labels)
            reference, reference_labels = controls[key]
            assert torch.equal(labels, reference_labels)
            lost = int((reference & ~correct).sum())
            gained = int((~reference & correct).sum())
            paired = dict(
                both_correct=int((reference & correct).sum()),
                lost=lost,
                gained=gained,
                both_wrong=int((~reference & ~correct).sum()),
                loss_pp=100 * (lost - gained) / len(correct),
                exact_mcnemar_p=float(binomtest(lost, lost + gained).pvalue)
                if lost + gained
                else 1.0,
            )
            row = dict(
                preset=preset,
                group=group,
                dataset=task,
                split=split,
                coefficient=None if group == "standalone" else config["coefficient"],
                paired=paired,
                **metric,
            )
            rows.append(row)
            predictions[(preset, *key)] = dict(prediction=predicted, labels=labels)
            save_json(out / "results.json", rows)
            print("OPERATOR_RESULT", json.dumps(row), flush=True)

    def evaluate_variant(name, model):
        record(name, "standalone", model)
        combined = merge(anchor, model, second, config["coefficient"])
        record(name, "direct", combined)
        if name in config["confirmation_presets"]:
            record(name, "standalone", model, "confirmation")
            record(name, "direct", combined, "confirmation")
        del combined

    evaluate_variant("unprotected", original)
    for layer, component in itertools.product(config["layers"], config["components"]):
        name = f"replace_L{layer}_{component}"
        model, keys = replace_components(original, anchor, [layer], component)
        profile.append(dict(preset=name, layers=[layer], component=component, replaced_keys=keys))
        evaluate_variant(name, model)
        del model
    for component in config["joint_components"]:
        name = f"joint_{component}"
        model, keys = replace_components(original, anchor, config["layers"], component)
        profile.append(
            dict(preset=name, layers=config["layers"], component=component, replaced_keys=keys)
        )
        evaluate_variant(name, model)
        del model
    joint = copy.deepcopy(original).cpu().eval()
    for layer in config["layers"]:
        fitting_start = time.perf_counter()
        model, details = reconstruct(original, anchor, fit_images, layer, config["ridge_ratio"])
        fitting_seconds = time.perf_counter() - fitting_start
        check = check_local_fit(original, model, anchor, check_images, layer)
        joint.model.visual.transformer.resblocks[layer].mlp.load_state_dict(
            model.model.visual.transformer.resblocks[layer].mlp.state_dict(), strict=True
        )
        fitting.append(
            dict(layer=layer, fit=details, check=check, reconstruction_seconds=fitting_seconds)
        )
        evaluate_variant(f"reconstruct_L{layer}", model)
        del model
        save_json(out / "fitting.json", fitting)
    evaluate_variant("reconstruct_joint", joint)
    del joint
    adjust_comparisons(rows)
    save_json(out / "results.json", rows)
    save_json(out / "replacement_profile.json", profile)
    torch.save(predictions, artifacts / "predictions.pt")
    helpers = ["clip_benchmark.py", "polynomial_probe.py", "subspace_probe.py"]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            total_seconds=time.perf_counter() - start,
            training="no_gradient_or_expert_fine_tuning",
            parameter_fitting="attack-only closed-form ridge; unprotected control",
            new_defense=False,
            test_usage="none",
            new_evaluations=len(rows),
            reused_measurements=0,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            parent_indices_sha256=hashlib.sha256(index_bytes).hexdigest(),
            data_indices_sha256=digest(out / "training_indices.json"),
            experiment_source_sha256=digest(__file__),
            helper_sha256={name: digest(Path(__file__).with_name(name)) for name in helpers},
            predictions_sha256=digest(artifacts / "predictions.pt"),
            seed_count=1,
            evaluation_implementation="author ImageClassifier/get_logits through shared evaluate",
            merge_implementation="author TaskVector through shared merge",
        ),
    )
    print("OPERATOR_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/operator_diagnostics.json")
    parser.add_argument("--output", default="outputs/2026-10-09/post_meeting/operator_diagnostics")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reconstruction-budget", action="store_true")
    mode.add_argument("--reconstruction-context", action="store_true")
    args = parser.parse_args()
    (
        run_context
        if args.reconstruction_context
        else run_budget
        if args.reconstruction_budget
        else run
    )(args)
