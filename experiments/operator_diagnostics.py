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
    run(parser.parse_args())
