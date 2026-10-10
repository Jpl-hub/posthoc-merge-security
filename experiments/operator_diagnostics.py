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
                or (component == "attn_mlp" and suffix.startswith(("attn.", "mlp.")))
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
def query_readout(model, head, images, capture_features=False):
    """Query predictions only; optional features come from the public student."""
    from modeling import ImageClassifier
    from utils import get_logits

    features, predictions = [], []
    handle = None
    if capture_features:
        handle = model.model.visual.ln_post.register_forward_hook(
            lambda _m, _a, value: features.append(value.detach().cpu())
        )
    classifier = ImageClassifier(model, head).cuda().eval()
    try:
        for batch in images.split(32):
            predictions.append(get_logits(batch.cuda(), classifier).argmax(-1).cpu())
    finally:
        if handle is not None:
            handle.remove()
        classifier.cpu()
    prediction = torch.cat(predictions)
    design = torch.cat(features).double().cuda() if capture_features else None
    if design is not None:
        assert design.shape == (len(images), 768)
    return prediction, design


@torch.no_grad()
def readout_sensitivity(args):
    """Enumerate all single MNIST target changes in the fixed linear readout.

    This oracle diagnostic uses confirmation labels to summarize the worst
    case. It is not a deployable protection or a held-out selected candidate.
    """
    parent = Path(args.readout_sensitivity)
    prior = json.loads((parent / "metadata.json").read_bytes())
    config = json.loads((parent / "protocol.json").read_bytes())
    raw = parent / "artifacts/certificate.pt"
    assert prior["completed"] and digest(raw) == prior["certificate_sha256"]
    teacher_parent = Path(config["parent"])
    teacher_meta = json.loads((teacher_parent / "metadata.json").read_bytes())
    target_file = teacher_parent / "artifacts/teacher_predictions.pt"
    assert digest(target_file) == teacher_meta["teacher_predictions_sha256"]
    saved = torch.load(raw, map_location="cpu", weights_only=True)
    classes = torch.load(target_file, map_location="cpu", weights_only=True)["readout_protected"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    save_json(
        out / "protocol.json",
        dict(
            parent=str(parent),
            changed_queries=1,
            trials=256 * 9,
            scope="exact linear readout; every one of 256 MNIST targets changed to every other class",
            selection="oracle worst case on already inspected confirmation labels; not a held-out defense result",
            new_defense=False,
            test_usage="none",
        ),
    )
    torch.set_num_threads(4)
    start = time.perf_counter()
    influence = saved["influence"].cuda()
    targets = torch.nn.functional.one_hot(classes.cuda(), 10).double()
    scores = influence @ targets
    baseline = scores.argmax(-1)
    assert torch.equal(baseline.cpu(), saved["prediction"])
    labels = saved["labels"].cuda()
    identity = torch.eye(10, dtype=torch.float64, device="cuda")
    cases = []
    for j in range(256):
        change = identity - identity[int(classes[j])]
        altered = scores.unsqueeze(0) + influence[:, j][None, :, None] * change[:, None, :]
        predicted = altered.argmax(-1)
        counts = predicted.eq(labels[None, :]).sum(dim=1).cpu()
        disagreements = predicted.ne(baseline[None, :]).sum(dim=1).cpu()
        for new_class in range(10):
            if new_class != int(classes[j]):
                cases.append([j, new_class, int(counts[new_class]), int(disagreements[new_class])])
    cases = torch.tensor(cases)
    assert cases.shape == (2304, 4)
    worst = cases[cases[:, 2].argmin()].tolist()
    best = cases[cases[:, 2].argmax()].tolist()
    rows = dict(
        n=len(labels),
        trials=len(cases),
        baseline_correct=int(baseline.eq(labels).sum()),
        worst_correct=worst[2],
        best_correct=best[2],
        worst_changed_query_index=worst[0],
        worst_new_class=worst[1],
        worst_prediction_disagreements=worst[3],
        max_prediction_disagreements=int(cases[:, 3].max()),
        median_correct=float(cases[:, 2].double().median()),
        worst_accuracy=worst[2] / len(labels),
        worst_loss_pp=100 * (int(baseline.eq(labels).sum()) - worst[2]) / len(labels),
    )
    save_json(out / "results.json", rows)
    torch.save(cases, out / "artifacts/cases.pt")
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            new_defense=False,
            new_models=0,
            new_teacher_queries=0,
            new_fits=0,
            ground_truth_used="oracle diagnostic summarization, not fitting or a valid selected defense",
            test_usage="none",
            source_sha256=digest(__file__),
            parent_certificate_sha256=digest(raw),
            parent_metadata_sha256=digest(parent / "metadata.json"),
            cases_sha256=digest(out / "artifacts/cases.pt"),
            results_sha256=digest(out / "results.json"),
            seconds=time.perf_counter() - start,
        ),
    )
    print("READOUT_SENSITIVITY_COMPLETE", json.dumps(rows), flush=True)


@torch.no_grad()
def readout_stability(args):
    """Certify a fixed ridge readout against any k changed MNIST query labels.

    The certificate is conservative. Failure to certify does not exhibit a
    perturbation, a successful defense, or an accuracy bound for a whole model.
    """
    parent = Path(args.readout_stability)
    config = json.loads((parent / "protocol.json").read_bytes())
    prior = json.loads((parent / "metadata.json").read_bytes())
    indices = json.loads((parent / "training_indices.json").read_bytes())
    fits = json.loads((parent / "fitting.json").read_bytes())
    assert prior["completed"] and prior["target_disagreement"] == 0
    assert prior["test_usage"] == "none"
    assert digest(parent / "artifacts/predictions.pt") == prior["predictions_sha256"]
    assert (
        digest(parent / "artifacts/teacher_predictions.pt") == prior["teacher_predictions_sha256"]
    )
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    budgets = [0, 1, 2, 4, 8, 16]
    save_json(
        out / "protocol.json",
        dict(
            parent=str(parent),
            task="MNIST",
            coefficient=config["coefficient"],
            budgets=budgets,
            changed_targets="any k of the first 256 MNIST hard predictions; SVHN fixed",
            condition="top1 margin greater than twice the sum of k largest absolute influence coefficients",
            evaluation="same frozen 4000-image training confirmation set; no test use",
            new_defense=False,
            interpretation="an uncertified prediction is not a successful perturbation",
            certificate_scope="fixed linear ridge readout; not whole-network or float32 compilation robustness",
        ),
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    root = Path(args.checkpoints)
    for name in ["zeroshot.pt", "SVHN/finetuned.pt", "head_MNIST.pt", "head_SVHN.pt"]:
        assert digest(root / name) == prior["checkpoint_sha256"][name]
    anchor = load(str(root / "zeroshot.pt"))
    partner = load(str(root / "SVHN/finetuned.pt"))
    heads = {t: load(str(root / f"head_{t}.pt")) for t in ["MNIST", "SVHN"]}
    context = merge(anchor, anchor, partner, config["coefficient"])
    training = dict(
        MNIST=MNIST(args.data, train=True, transform=anchor.val_preprocess),
        SVHN=SVHN(args.data, split="train", transform=anchor.val_preprocess),
    )
    designs = []
    for task in ["MNIST", "SVHN"]:
        images = torch.stack([training[task][i][0] for i in indices["fitting"][task]])
        _, x = query_readout(context, heads[task], images, capture_features=True)
        designs.append(x)
    design = torch.cat(designs)
    gram = design.T @ design
    ridge = config["ridge_ratio"] * gram.trace() / len(design)
    assert abs(float(ridge) - fits[0]["ridge"]) < 1e-10
    system = gram + ridge * torch.eye(gram.shape[0], device="cuda", dtype=torch.float64)
    solver = torch.linalg.solve(system, design.T)
    classes = torch.load(
        parent / "artifacts/teacher_predictions.pt", map_location="cpu", weights_only=True
    )["readout_protected"]
    targets = torch.nn.functional.one_hot(classes.cuda(), 10).double()
    confirmation = indices["confirmation"]["MNIST"]
    images = torch.stack([training["MNIST"][i][0] for i in confirmation])
    _, check = query_readout(context, heads["MNIST"], images, capture_features=True)
    influence = check @ solver
    scores = influence @ targets
    pred = scores.argmax(-1).cpu()
    prior_predictions = torch.load(
        parent / "artifacts/predictions.pt", map_location="cpu", weights_only=False
    )["readout_protected", "confirmation", "direct", "MNIST"]
    assert torch.equal(pred, prior_predictions["prediction"])
    labels = prior_predictions["labels"]
    top = scores.topk(2, dim=1).values
    margins = top[:, 0] - top[:, 1]
    largest = influence[:, :256].abs().topk(max(budgets), dim=1).values.cumsum(dim=1)
    rows = []
    for k in budgets:
        bound = torch.zeros_like(margins) if k == 0 else 2 * largest[:, k - 1]
        certified = margins > bound + 1e-10
        rows.append(
            dict(
                changed_queries=k,
                n=len(pred),
                certified_unchanged=int(certified.sum()),
                certified_fraction=float(certified.double().mean()),
                certified_correct=int((certified.cpu() & pred.eq(labels)).sum()),
                lower_bound_accuracy=float((certified.cpu() & pred.eq(labels)).double().mean()),
            )
        )
    assert all(a["certified_unchanged"] >= b["certified_unchanged"] for a, b in zip(rows, rows[1:]))
    assert rows[0]["certified_unchanged"] == len(pred)
    save_json(out / "results.json", rows)
    torch.save(
        dict(influence=influence.cpu(), margins=margins.cpu(), prediction=pred, labels=labels),
        artifacts / "certificate.pt",
    )
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            new_defense=False,
            new_models=0,
            new_teacher_queries=0,
            new_fits=0,
            test_usage="none",
            fitting_labels_used=False,
            fixed_ridge=float(ridge),
            confirmation_predictions_exactly_match_parent=True,
            baseline_correct=int(pred.eq(labels).sum()),
            baseline_n=len(pred),
            source_sha256=digest(__file__),
            parent_metadata_sha256=digest(parent / "metadata.json"),
            config_sha256=digest(out / "protocol.json"),
            certificate_sha256=digest(artifacts / "certificate.pt"),
            seconds=time.perf_counter() - start,
        ),
    )
    print("READOUT_STABILITY_COMPLETE", json.dumps(rows), flush=True)


@torch.no_grad()
def evaluate_readout(args):
    """Evaluate frozen projection recovery on official tests without new fitting."""
    parent = Path(args.evaluate_readout)
    config = json.loads((parent / "protocol.json").read_bytes())
    prior = json.loads((parent / "metadata.json").read_bytes())
    assert prior["completed"] and prior["test_usage"] == "none"
    assert digest(parent / "artifacts/predictions.pt") == prior["predictions_sha256"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    presets = ["unprotected", "public_backbone", "readout_unprotected", "readout_protected"]
    frozen = {
        preset: digest(parent / f"artifacts/{preset}_projection.pt") for preset in presets[2:]
    }
    save_json(
        out / "protocol.json",
        dict(
            parent=str(parent),
            coefficient=config["coefficient"],
            presets=presets,
            projection_sha256=frozen,
            test_usage="official complete MNIST and SVHN; frozen recovery, no fitting",
            new_defense=False,
        ),
    )
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    root = Path(args.checkpoints)
    for name, value in prior["checkpoint_sha256"].items():
        assert digest(root / name) == value
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "MNIST/finetuned.pt"))
    partner = load(str(root / "SVHN/finetuned.pt"))
    heads = {task: load(str(root / f"head_{task}.pt")) for task in ["MNIST", "SVHN"]}
    datasets = dict(
        MNIST=MNIST(args.data, train=False, transform=original.val_preprocess),
        SVHN=SVHN(args.data, split="test", transform=original.val_preprocess),
    )
    assert len(datasets["MNIST"]) == 10000 and len(datasets["SVHN"]) == 26032
    loaders = {task: make_loader(data, list(range(len(data)))) for task, data in datasets.items()}
    rows, predictions, controls = [], {}, {}
    for preset in presets:
        model = original if preset == "unprotected" else copy.deepcopy(anchor).cpu().eval()
        if preset in frozen:
            projection = torch.load(
                parent / f"artifacts/{preset}_projection.pt", map_location="cpu", weights_only=True
            )
            model.model.visual.proj.copy_(projection)
        combined = merge(anchor, model, partner, config["coefficient"])
        for group, evaluated in [("standalone", model), ("direct", combined)]:
            for task, loader in loaders.items():
                if group == "standalone" and task != "MNIST":
                    continue
                p, y = [], []

                def sink(pred, labels):
                    p.append(pred)
                    y.append(labels)

                metric = evaluate(evaluated, heads[task], loader, "cuda", sink)
                pred, labels = torch.cat(p), torch.cat(y)
                correct = pred == labels
                key = ("test", group, task)
                if preset == "unprotected":
                    controls[key] = dict(correct=correct, labels=labels)
                assert torch.equal(controls[key]["labels"], labels)
                row = dict(
                    preset=preset,
                    split="test",
                    group=group,
                    dataset=task,
                    coefficient=None if group == "standalone" else config["coefficient"],
                    paired=paired_counts(controls[key]["correct"], correct),
                    **metric,
                )
                if preset == "readout_protected":
                    matched = predictions["readout_unprotected", *key]
                    row["matched_recovery"] = paired_counts(
                        matched["prediction"] == labels, correct
                    )
                    if prior["target_disagreement"] == 0:
                        assert torch.equal(pred, matched["prediction"])
                rows.append(row)
                predictions[preset, *key] = dict(prediction=pred, labels=labels)
                save_json(out / "results.json", rows)
                print("READOUT_TEST_RESULT", preset, group, task, metric, flush=True)
        del combined, model
    for preset, value in frozen.items():
        assert digest(parent / f"artifacts/{preset}_projection.pt") == value
    torch.save(predictions, artifacts / "predictions.pt")
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            new_defense=False,
            test_usage="official complete MNIST and SVHN; frozen recovery, no fitting",
            new_evaluations=len(rows),
            new_fits=0,
            optimizer_steps=0,
            fitting_labels_used=False,
            target_disagreement=prior["target_disagreement"],
            seconds=time.perf_counter() - start,
            source_sha256=digest(__file__),
            config_sha256=digest(out / "protocol.json"),
            parent_metadata_sha256=digest(parent / "metadata.json"),
            checkpoint_sha256=prior["checkpoint_sha256"],
            projection_sha256=frozen,
            predictions_sha256=digest(artifacts / "predictions.pt"),
        ),
    )
    print("READOUT_TEST_COMPLETE", flush=True)


@torch.no_grad()
def run_readout(args):
    """Compile hard predictions into one public backbone, then merge at fixed lambda.

    This is a calibration-dependent functional recovery probe, not a new defense
    or a complete implementation of a prior model merging method.
    """
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    protocol = Path(args.config).read_bytes()
    config = json.loads(protocol)
    assert config["task"] == "MNIST" and config["partner"] == "SVHN"
    assert config["test_usage"] == "none" and config["samples_per_task"] == 256
    (out / "protocol.json").write_bytes(protocol)
    parent = Path(config["index_source"])
    frozen = json.loads((parent / "training_indices.json").read_bytes())
    confirmation = json.loads(
        (Path(config["confirmation_source"]) / "training_indices.json").read_bytes()
    )
    indices = dict(
        development={t: frozen[t]["development"] for t in ["MNIST", "SVHN"]},
        confirmation=confirmation["confirmation"],
        fitting={t: frozen[t]["attack"][:256] for t in ["MNIST", "SVHN"]},
    )
    for task in ["MNIST", "SVHN"]:
        assert len(indices["fitting"][task]) == len(set(indices["fitting"][task])) == 256
        assert not set(indices["fitting"][task]).intersection(indices["development"][task])
        assert not set(indices["fitting"][task]).intersection(frozen[task]["calibration"])
        assert not set(indices["fitting"][task]).intersection(indices["confirmation"].get(task, []))
    save_json(out / "training_indices.json", indices)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "MNIST/finetuned.pt"))
    partner = load(str(root / "SVHN/finetuned.pt"))
    heads = {t: load(str(root / f"head_{t}.pt")) for t in ["MNIST", "SVHN"]}
    assert torch.equal(heads["MNIST"].weight, heads["SVHN"].weight)
    assert torch.equal(heads["MNIST"].bias, heads["SVHN"].bias)
    assert torch.count_nonzero(heads["MNIST"].bias) == 0
    release_file = Path(config["release"])
    assert digest(release_file) == config["release_sha256"]
    protected = copy.deepcopy(original).cpu().eval()
    released = torch.load(release_file, map_location="cpu", weights_only=False)
    for layer, state in released.items():
        protected.model.visual.transformer.resblocks[layer].mlp.load_state_dict(state, strict=True)
    training = dict(
        MNIST=MNIST(args.data, train=True, transform=original.val_preprocess),
        SVHN=SVHN(args.data, split="train", transform=original.val_preprocess),
    )
    loaders = {
        (split, task): make_loader(training[task], subset)
        for split in ["development", "confirmation"]
        for task, subset in indices[split].items()
    }
    images = {t: torch.stack([training[t][i][0] for i in indices["fitting"][t]]) for t in training}
    coefficient = config["coefficient"]
    context = merge(anchor, anchor, partner, coefficient)
    designs, targets, fit_rows = [], {}, []
    for task in ["MNIST", "SVHN"]:
        _, design = query_readout(context, heads[task], images[task], capture_features=True)
        designs.append(design)
    design = torch.cat(designs)
    gram = design.T @ design
    ridge = config["ridge_ratio"] * gram.trace() / len(design)
    system = gram + ridge * torch.eye(gram.shape[0], device="cuda", dtype=torch.float64)
    head = heads["MNIST"].weight.double().cuda()
    right_inverse = torch.linalg.solve(head @ head.T, head)
    assert torch.allclose(
        head @ right_inverse.T,
        torch.eye(10, device="cuda", dtype=torch.float64),
        atol=1e-8,
        rtol=1e-8,
    )
    public_projection = context.model.visual.proj.double().cuda()
    partner_predictions, _ = query_readout(partner, heads["SVHN"], images["SVHN"])
    rows, predictions, controls = [], {}, {}

    def measure(preset, model, combined):
        for split, task in loaders:
            for group, evaluated in [("standalone", model), ("direct", combined)]:
                if group == "standalone" and task != "MNIST":
                    continue
                p, y = [], []

                def sink(pred, labels):
                    p.append(pred)
                    y.append(labels)

                metric = evaluate(evaluated, heads[task], loaders[split, task], "cuda", sink)
                pred, labels = torch.cat(p), torch.cat(y)
                correct = pred == labels
                key = split, group, task
                if preset == "unprotected":
                    controls[key] = correct
                row = dict(
                    preset=preset,
                    split=split,
                    group=group,
                    dataset=task,
                    coefficient=None if group == "standalone" else coefficient,
                    paired=paired_counts(controls[key], correct),
                    **metric,
                )
                if preset == "readout_protected":
                    matched = predictions["readout_unprotected", *key]
                    assert torch.equal(matched["labels"], labels)
                    row["matched_recovery"] = paired_counts(
                        matched["prediction"] == labels, correct
                    )
                rows.append(row)
                predictions[preset, *key] = dict(prediction=pred, labels=labels)
                save_json(out / "results.json", rows)
                print(
                    "READOUT_RESULT",
                    preset,
                    split,
                    group,
                    task,
                    metric["correct"],
                    metric["n"],
                    flush=True,
                )

    measure("unprotected", original, merge(anchor, original, partner, coefficient))
    measure("public_backbone", anchor, context)
    for preset, teacher in [("readout_unprotected", original), ("readout_protected", protected)]:
        own_predictions, _ = query_readout(teacher, heads["MNIST"], images["MNIST"])
        classes = torch.cat([own_predictions, partner_predictions])
        targets[preset] = classes
        target = torch.nn.functional.one_hot(classes.cuda(), 10).double()
        effective = torch.linalg.solve(system, design.T @ target)
        residual = system @ effective - design.T @ target
        normal_error = float(residual.norm() / (design.T @ target).norm().clamp_min(1e-30))
        assert normal_error < 1e-8
        projection = public_projection + (effective - public_projection @ head.T) @ right_inverse
        mapping_error = float((projection @ head.T - effective).norm() / effective.norm())
        assert mapping_error < 1e-8
        restored = copy.deepcopy(anchor).cpu().eval()
        a0, ap = anchor.model.visual.proj.double().cuda(), partner.model.visual.proj.double().cuda()
        restored.model.visual.proj.copy_(
            (a0 + (projection - a0 - coefficient * (ap - a0)) / coefficient).float().cpu()
        )
        combined = merge(anchor, restored, partner, coefficient)
        projection_error = float(
            (combined.model.visual.proj.double().cuda() - projection).norm() / projection.norm()
        )
        assert projection_error < 1e-6
        fit_rows.append(
            dict(
                preset=preset,
                images=len(design),
                features=design.shape[1],
                ridge=float(ridge),
                normal_equation_relative_residual=normal_error,
                head_mapping_relative_error=mapping_error,
                merged_projection_relative_error=projection_error,
                fit_correct=int((design @ effective).argmax(-1).cpu().eq(classes).sum()),
            )
        )
        torch.save(restored.model.visual.proj.detach().cpu(), artifacts / f"{preset}_projection.pt")
        measure(preset, restored, combined)
    adjust_comparisons(rows)
    save_json(out / "results.json", rows)
    save_json(out / "fitting.json", fit_rows)
    torch.save(predictions, artifacts / "predictions.pt")
    torch.save(targets, artifacts / "teacher_predictions.pt")
    disagreement = int((targets["readout_unprotected"] != targets["readout_protected"]).sum())
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            new_defense=False,
            test_usage="none",
            seed_count=1,
            total_seconds=time.perf_counter() - start,
            new_evaluations=len(rows),
            fitting_labels_used=False,
            teacher_signals="hard predictions only; no released hidden activations or original protected-task parameters",
            target_disagreement=disagreement,
            target_difference_frobenius=(2 * disagreement) ** 0.5,
            public_features_source="anchor plus coefficient times the known SVHN task vector; frozen before teacher queries",
            samples_per_task=256,
            task_head_weights_exactly_equal=True,
            task_head_biases_zero=True,
            optimizer_steps=0,
            parameter_fitting="ridge effective task readout, mapped into original visual projection; original head unchanged",
            source_sha256=digest(__file__),
            config_sha256=digest(out / "protocol.json"),
            indices_sha256=digest(out / "training_indices.json"),
            release_sha256=digest(release_file),
            predictions_sha256=digest(artifacts / "predictions.pt"),
            teacher_predictions_sha256=digest(artifacts / "teacher_predictions.pt"),
            checkpoint_sha256={
                str(p.relative_to(root)): digest(p)
                for p in [
                    root / "zeroshot.pt",
                    root / "MNIST/finetuned.pt",
                    root / "SVHN/finetuned.pt",
                    root / "head_MNIST.pt",
                    root / "head_SVHN.pt",
                ]
            },
            helper_sha256={
                p: digest(Path(__file__).with_name(p))
                for p in ["clip_benchmark.py", "subspace_probe.py"]
            },
            threat_scope="One original-architecture merged ViT, not an ensemble or test-time routing; partner-specific, calibrated additional attack",
            novelty="Existing linear readout distillation and least squares; mechanism stress test, no algorithmic novelty claim",
        ),
    )
    print("READOUT_COMPLETE", flush=True)


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
    assert config["conditions"] in [
        ["public_expert", "public_merged", "released_expert", "released_merged"],
        ["public_expert", "public_selfmerged"],
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
    self_merged = None
    if "public_selfmerged" in config["conditions"]:
        first_pass = copy.deepcopy(original).cpu().eval()
        for layer in config["layers"]:
            first_pass.model.visual.transformer.resblocks[layer].mlp.load_state_dict(
                public_state[layer]
            )
        self_merged = merge(anchor, first_pass, second, config["coefficient"])
        del first_pass
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
                    input_model={
                        "expert": original,
                        "merged": original_merged,
                        "selfmerged": self_merged,
                    }[input_source],
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
            new_fits=(len(config["conditions"]) - 1) * len(config["layers"]),
            reused_fits=len(config["layers"]),
            query_budget=config.get(
                "query_budget", "one 12800-token MLP target per condition/layer"
            ),
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
    protected_task = config["protected_task"]
    if args.joint_components:
        assert not config.get("reconstruction", True) and not config["components"]
        assert len(args.joint_components) == len(set(args.joint_components))
        config["joint_components"] = args.joint_components
        config["confirmation_presets"] = [
            "unprotected",
            *[f"joint_{component}" for component in args.joint_components],
        ]
        config["protocol_amendment"] = (
            "Joint support explicitly changed before evaluation; no new protection or test use"
        )
        config_bytes = json.dumps(config, indent=2).encode()
    partner_task = config.get("other_task", "MNIST")
    reconstruction = config.get("reconstruction", True)
    assert {protected_task, partner_task} == {"MNIST", "SVHN"}
    assert config["test_usage"] == "none"
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
    original = load(str(checkpoints / f"{protected_task}/finetuned.pt"))
    second = load(str(checkpoints / f"{partner_task}/finetuned.pt"))
    heads = {name: load(str(checkpoints / f"head_{name}.pt")) for name in ["MNIST", "SVHN"]}
    training = {
        "MNIST": MNIST(args.data, train=True, transform=original.val_preprocess),
        "SVHN": SVHN(args.data, split="train", transform=original.val_preprocess),
    }
    excluded = set(itertools.chain.from_iterable(frozen[protected_task].values()))
    order = torch.randperm(
        len(training[protected_task]), generator=torch.Generator().manual_seed(config["seed"])
    ).tolist()
    confirmation = [i for i in order if i not in excluded][: config["confirmation_samples"]]
    indices = dict(
        development={task: splits["development"] for task, splits in frozen.items()},
        confirmation={protected_task: confirmation},
        fitting={protected_task: frozen[protected_task]["attack"][: config["attacker_samples"]]},
        local_check={
            protected_task: frozen[protected_task]["calibration"][: config["local_check_samples"]]
        },
    )
    assert len(confirmation) == config["confirmation_samples"]
    save_json(out / "training_indices.json", indices)
    loaders = {
        split: {task: make_loader(training[task], subset) for task, subset in by_task.items()}
        for split, by_task in indices.items()
        if split in {"development", "confirmation"}
    }
    fit_images, check_images = None, None
    if reconstruction:
        fit_images = torch.stack(
            [training[protected_task][i][0] for i in indices["fitting"][protected_task]]
        )
        check_images = torch.stack(
            [training[protected_task][i][0] for i in indices["local_check"][protected_task]]
        )
    else:
        assert config["attacker_samples"] == config["local_check_samples"] == 0
    rows, predictions, controls = [], {}, {}
    profile, fitting = [], []

    def record(preset, group, model, split="development"):
        for task, loader in loaders[split].items():
            if group == "standalone" and task != protected_task:
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
    if reconstruction:
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
    save_json(out / "fitting.json", fitting)
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
            parameter_fitting="attack-only closed-form ridge; unprotected control"
            if reconstruction
            else "none; public pretrained component replacement only",
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
            checkpoint_sha256={
                str(p.relative_to(checkpoints)): digest(p)
                for p in [
                    checkpoints / "zeroshot.pt",
                    checkpoints / f"{protected_task}/finetuned.pt",
                    checkpoints / f"{partner_task}/finetuned.pt",
                    checkpoints / f"head_{protected_task}.pt",
                    checkpoints / f"head_{partner_task}.pt",
                ]
            },
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
    parser.add_argument(
        "--joint-components",
        nargs="+",
        choices=["block", "mlp", "attn", "norms", "attn_mlp"],
        help="Declare a replacement-only joint-support follow-up without changing the saved parent protocol.",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--reconstruction-budget", action="store_true")
    mode.add_argument("--reconstruction-context", action="store_true")
    mode.add_argument("--readout-recovery", action="store_true")
    mode.add_argument("--evaluate-readout", metavar="FROZEN_RUN")
    mode.add_argument("--readout-stability", metavar="FROZEN_RUN")
    mode.add_argument("--readout-sensitivity", metavar="CERTIFICATE_RUN")
    args = parser.parse_args()
    (
        readout_sensitivity
        if args.readout_sensitivity
        else readout_stability
        if args.readout_stability
        else evaluate_readout
        if args.evaluate_readout
        else run_readout
        if args.readout_recovery
        else run_context
        if args.reconstruction_context
        else run_budget
        if args.reconstruction_budget
        else run
    )(args)
