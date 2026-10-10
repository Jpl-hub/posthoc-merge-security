"""Probe task-class response relocation with a single closed-form fit, no gradients."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate, merge
from eight_task_benchmark import author_recover
from joint_posthoc import digest, restoration
from operator_diagnostics import paired_counts, replace_components
from polynomial_probe import ridge_update
from subspace_probe import make_loader, save_json
from torchvision.datasets import MNIST, SVHN


@torch.no_grad()
def capture_cls(model, images, hidden=False):
    block = model.model.visual.transformer.resblocks[-1]
    values, features = [], []
    handles = [block.register_forward_hook(lambda _m, _a, v: values.append(v[0].cpu()))]
    if hidden:
        handles.append(
            block.mlp.gelu.register_forward_hook(lambda _m, _a, v: features.append(v[0].cpu()))
        )
    model.cuda().eval()
    try:
        for batch in images.split(32):
            model(batch.cuda())
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return torch.cat(values), torch.cat(features) if hidden else None


def class_geometry(teacher, base, model, head):
    """Match normalized class differences while retaining base mean and variance."""
    norm = model.model.visual.ln_post
    assert torch.count_nonzero(head.bias) == 0
    e = model.model.visual.proj.double() @ head.weight.double().T
    differences = e[:, 1:] - e[:, :1]
    c = norm.weight.double()[:, None] * differences
    c = c - c.mean(0)
    inverse = torch.linalg.pinv(c)
    center = base - base.mean(1, keepdim=True)
    scale = (center.square().mean(1, keepdim=True) + norm.eps).sqrt()
    unit = center / scale
    tcenter = teacher - teacher.mean(1, keepdim=True)
    tunit = tcenter / (tcenter.square().mean(1, keepdim=True) + norm.eps).sqrt()
    task = (tunit @ c) @ inverse
    null = unit - (unit @ c) @ inverse
    remaining = unit.square().sum(1, keepdim=True) - task.square().sum(1, keepdim=True)
    assert float(remaining.min()) > 0 and float(null.square().sum(1).min()) > 0
    target = base.mean(1, keepdim=True) + scale * (
        task + (remaining / null.square().sum(1, keepdim=True)).sqrt() * null
    )
    z = target - target.mean(1, keepdim=True)
    zunit = z / (z.square().mean(1, keepdim=True) + norm.eps).sqrt()
    response_error = float((zunit @ c - tunit @ c).norm() / (tunit @ c).norm())
    assert response_error < 1e-10
    assert torch.allclose(z.square().mean(1), center.square().mean(1), rtol=1e-10, atol=1e-10)
    assert torch.allclose(target.mean(1), base.mean(1), rtol=1e-10, atol=1e-10)
    return (
        target,
        dict(
            class_difference_rank=int(torch.linalg.matrix_rank(c)),
            exact_target_response_relative_error=response_error,
            mean_and_variance_preserved=True,
            classifier_bias_zero=True,
        ),
        dict(class_directions=c, epsilon=norm.eps),
    )


@torch.no_grad()
def run(args):
    config = json.loads(Path(args.config).read_bytes())
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    save_json(out / "protocol.json", config)
    parent = json.loads((Path(config["index_parent"]) / "training_indices.json").read_bytes())
    excluded = set()
    for selected in parent["MNIST"].values():
        excluded.update(selected)
    for previous in config["exclude_confirmations"]:
        excluded.update(
            json.loads((Path(previous) / "training_indices.json").read_bytes())["confirmation"][
                "MNIST"
            ]
        )
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(config["seed"])).tolist()
    available = [i for i in order if i not in excluded]
    n = config["calibration_samples"]
    calibration = available[:n]
    confirmation = available[n : n + config["confirmation_samples"]]
    assert len(confirmation) == config["confirmation_samples"]
    assert not set(calibration) & set(confirmation) and not set(available) & excluded
    indices = dict(
        calibration=calibration,
        confirmation={"MNIST": confirmation, "SVHN": parent["SVHN"]["development"]},
    )
    save_json(out / "training_indices.json", indices)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    root = Path(args.checkpoints)
    original = load(str(root / "MNIST/finetuned.pt"))
    anchor = load(str(root / "zeroshot.pt"))
    partner = load(str(root / "SVHN/finetuned.pt"))
    heads = {t: load(str(root / f"head_{t}.pt")) for t in ["MNIST", "SVHN"]}
    datasets = {
        "MNIST": MNIST(args.data, train=True, transform=original.val_preprocess),
        "SVHN": SVHN(args.data, split="train", transform=original.val_preprocess),
    }
    images = torch.stack([datasets["MNIST"][i][0] for i in calibration])
    teacher, _ = capture_cls(original, images)
    candidate, erased = replace_components(original, anchor, config["layers"], "attn_mlp")
    base, hidden = capture_cls(candidate, images, hidden=True)
    target, geometry, directions = class_geometry(
        teacher.double(), base.double(), candidate, heads["MNIST"]
    )
    design = torch.cat([hidden.double(), torch.ones(n, 1, dtype=torch.float64)], dim=1).cuda()
    delta, fitting = ridge_update(design, (target - base).cuda(), config["ridge_ratio"])
    output = candidate.model.visual.transformer.resblocks[-1].mlp.c_proj
    before = torch.cat([output.weight.T.double(), output.bias[None].double()], dim=0)
    output.weight.add_(delta[:-1].T.float().cpu())
    output.bias.add_(delta[-1].float().cpu())
    raw_fit = out / "artifacts/fitting.pt"
    torch.save(
        dict(
            design=design.cpu(),
            target_delta=(target - base),
            update=delta.cpu(),
            before=before,
            teacher=teacher.double(),
            base=base.double(),
            target=target,
            **directions,
        ),
        raw_fit,
    )
    save_json(out / "fitting.json", dict(**geometry, **fitting, calibration_images=n))
    states = {
        layer: {
            k: v.clone() for k, v in block.state_dict().items() if k.startswith(("attn.", "mlp."))
        }
        for layer, block in enumerate(candidate.model.visual.transformer.resblocks)
    }
    torch.save(states, out / "artifacts/release.pt")
    rows, predictions = [], {}
    loaders = {
        t: make_loader(datasets[t], selected) for t, selected in indices["confirmation"].items()
    }

    def record(preset, group, model, task):
        pred, labels = [], []

        def sink(p, y):
            pred.append(p.clone())
            labels.append(y.clone())

        score = evaluate(model, heads[task], loaders[task], "cuda", prediction_sink=sink)
        p, y = torch.cat(pred), torch.cat(labels)
        key = (preset, group, task)
        predictions[key] = dict(prediction=p, labels=y)
        ref = predictions["unprotected", group, task]
        assert torch.equal(y, ref["labels"])
        pair = paired_counts(ref["prediction"].eq(y), p.eq(y))
        rows.append(dict(preset=preset, group=group, dataset=task, paired=pair, **score))
        save_json(out / "results.json", rows)
        print(
            "CLASS_RECOMPILE_RESULT", preset, group, task, score["correct"], score["n"], flush=True
        )

    record("unprotected", "standalone", original, "MNIST")
    record("recompiled", "standalone", candidate, "MNIST")
    loss = 100 * (rows[0]["top1"] - rows[1]["top1"])
    decision = dict(utility_loss_pp=loss, utility_passed=loss <= config["utility_budget_pp"])
    audits = []
    if decision["utility_passed"]:
        for group in config["groups"]:
            reset = {}
            for preset, source in [("unprotected", original), ("recompiled", candidate)]:
                if group == "aga_selected":
                    restored, evidence = author_recover(source, anchor, Path(args.author_repo))
                elif group == "output_delta_twice":
                    restored = copy.deepcopy(source).cpu()
                    local = restored.model.visual.transformer.resblocks[-1].mlp.c_proj
                    public = anchor.model.visual.transformer.resblocks[-1].mlp.c_proj
                    local.weight.copy_(public.weight + 2 * (local.weight - public.weight))
                    local.bias.copy_(public.bias + 2 * (local.bias - public.bias))
                    evidence = dict(public_output_delta_scale=2)
                else:
                    restored, evidence = restoration(source, anchor, group, config["layers"])
                audits.append(dict(preset=preset, group=group, evidence=evidence))
                if group == "pretrained_attn_mlp_all":
                    reset[preset] = restored
                for task in datasets:
                    record(
                        preset, group, merge(anchor, restored, partner, config["coefficient"]), task
                    )
            if reset:
                assert all(
                    torch.equal(v, reset["recompiled"].state_dict()[k])
                    for k, v in reset["unprotected"].state_dict().items()
                )
        best = {
            p: max(
                r["top1"]
                for r in rows
                if r["preset"] == p and r["group"] != "standalone" and r["dataset"] == "MNIST"
            )
            for p in ["unprotected", "recompiled"]
        }
        gap = 100 * (best["unprotected"] - best["recompiled"])
        decision.update(
            best_measured_attack_gap_pp=gap,
            attack_gate_passed=gap >= config["minimum_attack_gap_pp"],
        )
    decision.update(new_defense_validated=False, functional_recovery_tested=False)
    save_json(out / "decision.json", decision)
    save_json(out / "attack_audits.json", audits)
    torch.save(predictions, out / "artifacts/predictions.pt")
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            elapsed_seconds=time.perf_counter() - start,
            checkpoint_sha256={
                p: digest(root / p)
                for p in [
                    "zeroshot.pt",
                    "MNIST/finetuned.pt",
                    "SVHN/finetuned.pt",
                    "head_MNIST.pt",
                    "head_SVHN.pt",
                ]
            },
            helpers_sha256={
                p: digest(Path(__file__).with_name(p))
                for p in [
                    "joint_posthoc.py",
                    "operator_diagnostics.py",
                    "polynomial_probe.py",
                    "clip_benchmark.py",
                    "subspace_probe.py",
                    "eight_task_benchmark.py",
                ]
            },
            artifacts_sha256={p.name: digest(p) for p in (out / "artifacts").iterdir()},
            edited_support=erased,
            calibration_labels_used=False,
            optimizer_steps=0,
            closed_form_fits=1,
            test_usage="none",
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/class_response_recompile.json")
    parser.add_argument("--output", default="outputs/2026-10-10/class_response_recompile")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
