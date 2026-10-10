"""Test a frozen task-span compression and Attention-to-MLP response transfer.

Reuse the author merger/evaluator and existing ridge solver. This is an
explicitly calibrated mechanism probe, without a novelty or security claim.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate, merge
from operator_diagnostics import paired_counts, replace_components
from polynomial_probe import ridge_update
from subspace_probe import make_loader, save_json
from torchvision.datasets import MNIST, SVHN


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def flatten_tokens(value):
    """Keep image/token order identical across full-model forward captures."""
    return value.detach().permute(1, 0, 2).reshape(-1, value.shape[-1]).cpu()


@torch.no_grad()
def capture_blocks(model, images, layers, details=False):
    captures = {layer: {"output": []} for layer in layers}
    handles = []
    for layer in layers:
        block = model.model.visual.transformer.resblocks[layer]
        local = captures[layer]
        handles.append(
            block.register_forward_hook(
                lambda _m, _a, value, local=local: local["output"].append(flatten_tokens(value))
            )
        )
        if details:
            local.update(attention_input=[], post_attention=[], mlp_input=[])
            handles.extend(
                [
                    block.ln_1.register_forward_hook(
                        lambda _m, _a, value, local=local: local["attention_input"].append(
                            flatten_tokens(value)
                        )
                    ),
                    block.ln_2.register_forward_pre_hook(
                        lambda _m, args, local=local: local["post_attention"].append(
                            flatten_tokens(args[0])
                        )
                    ),
                    block.ln_2.register_forward_hook(
                        lambda _m, _a, value, local=local: local["mlp_input"].append(
                            flatten_tokens(value)
                        )
                    ),
                ]
            )
    model.cuda().eval()
    try:
        for batch in images.split(8):
            model(batch.cuda())
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return {
        layer: {key: torch.cat(chunks) for key, chunks in local.items()}
        for layer, local in captures.items()
    }


@torch.no_grad()
def project_input(weight, bias, inputs, rank):
    """Fit a centered task span and delete linear dependence on its complement."""
    x = inputs.double().cuda()
    mean = x.mean(0)
    centered = x - mean
    covariance = centered.T @ centered / len(x)
    values, vectors = torch.linalg.eigh(covariance)
    basis = vectors[:, -rank:]
    original = weight.double().cuda()
    projected = (original @ basis) @ basis.T
    discarded = original - projected
    new_bias = bias.double().cuda() + discarded @ mean
    response = centered @ original.T
    removed = centered @ discarded.T
    evidence = dict(
        rank=rank,
        samples=len(x),
        centered_input_energy_retained=float(values[-rank:].sum() / values.sum()),
        linear_response_energy_removed_fraction=float(
            removed.square().sum() / response.square().sum()
        ),
        parameter_span_residual=float(
            (projected - (projected @ basis) @ basis.T).norm() / projected.norm()
        ),
    )
    weight.copy_(projected.float().cpu())
    bias.copy_(new_bias.float().cpu())
    return evidence


@torch.no_grad()
def build_release(original, images, config, out):
    candidate = copy.deepcopy(original).cpu().eval()
    targets = capture_blocks(original, images, config["layers"])
    fitting = []
    for layer in config["layers"]:
        current = capture_blocks(candidate, images, [layer], details=True)[layer]
        block = candidate.model.visual.transformer.resblocks[layer]
        attention = project_input(
            block.attn.in_proj_weight,
            block.attn.in_proj_bias,
            current["attention_input"],
            config["representation_rank"],
        )
        block.attn.out_proj.weight.mul_(config["attention_output_gain"])
        block.attn.out_proj.bias.mul_(config["attention_output_gain"])
        changed = capture_blocks(candidate, images, [layer], details=True)[layer]
        mlp_projection = project_input(
            block.mlp.c_fc.weight,
            block.mlp.c_fc.bias,
            changed["mlp_input"],
            config["representation_rank"],
        )
        mlp = copy.deepcopy(block.mlp).double().cuda().eval()
        x = changed["mlp_input"].double().cuda()
        hidden = mlp.gelu(mlp.c_fc(x))
        design = torch.cat([hidden, torch.ones_like(hidden[:, :1])], dim=1)
        target = (
            targets[layer]["output"].double().cuda() - changed["post_attention"].double().cuda()
        )
        initial = torch.cat([mlp.c_proj.weight.T, mlp.c_proj.bias[None, :]], dim=0)
        update, evidence = ridge_update(design, target - design @ initial, config["ridge_ratio"])
        fitted = initial + update
        block.mlp.c_proj.weight.copy_(fitted[:-1].T.float().cpu())
        block.mlp.c_proj.bias.copy_(fitted[-1].float().cpu())
        error = design @ fitted - target
        fitting.append(
            dict(
                layer=layer,
                attention_input_projection=attention,
                mlp_input_projection=mlp_projection,
                residual_target_relative_error=float(error.norm() / target.norm()),
                original_block_relative_error=float(
                    error.norm() / targets[layer]["output"].double().cuda().norm()
                ),
                attention_output_gain=config["attention_output_gain"],
                **evidence,
            )
        )
        save_json(out / "fitting.json", fitting)
        print("JOINT_FIT", layer, fitting[-1]["residual_target_relative_error"], flush=True)
        del current, changed, mlp, x, hidden, design, target, initial, update, fitted, error
    states = {
        layer: {
            key: value.clone()
            for key, value in block.state_dict().items()
            if key.startswith(("attn.", "mlp."))
        }
        for layer, block in enumerate(candidate.model.visual.transformer.resblocks)
    }
    torch.save(states, out / "artifacts/release.pt")
    final = capture_blocks(candidate, images, [11])[11]["output"].double()
    reference = targets[11]["output"].double()
    save_json(
        out / "calibration_check.json",
        dict(final_block_relative_error=float((final - reference).norm() / reference.norm())),
    )
    return candidate


@torch.no_grad()
def restoration(model, anchor, group, layers):
    if group == "direct":
        return model, []
    components = {
        "pretrained_fc_all": ["mlp.c_fc"],
        "pretrained_mlp_all": ["mlp"],
        "pretrained_attn_all": ["attn"],
        "pretrained_attn_mlp_all": ["attn_mlp"],
        "pretrained_outputs_all": ["attn.out_proj", "mlp.c_proj"],
        "canonical_gain_mlp_reset": ["mlp"],
    }
    result = model
    edits = []
    if group == "canonical_gain_mlp_reset":
        result = copy.deepcopy(model).cpu().eval()
        for layer in layers:
            output = result.model.visual.transformer.resblocks[layer].attn.out_proj
            reference = anchor.model.visual.transformer.resblocks[layer].attn.out_proj
            gain = float(
                (output.weight.double() * reference.weight.double()).sum()
                / reference.weight.double().square().sum()
            )
            if gain < 0:
                output.weight.neg_()
                output.bias.neg_()
            edits.append(
                dict(layer=layer, public_output_gain_estimate=gain, sign_reversed=gain < 0)
            )
    for component in components[group]:
        result, keys = replace_components(result, anchor, layers, component)
        edits.append(dict(component=component, keys=keys))
    return result, edits


@torch.no_grad()
def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    assert config["test_usage"] == "none" and config["layers"] == list(range(12))
    (out / "protocol.json").write_bytes(config_bytes)
    parent_path = Path(config["indices_parent"]) / "training_indices.json"
    parent = json.loads(parent_path.read_bytes())
    previous_path = Path(config["exclude_confirmation"])
    previous = json.loads(previous_path.read_bytes())
    excluded = set(previous["confirmation"]["MNIST"])
    for subset in parent["MNIST"].values():
        excluded.update(subset)
    order = torch.randperm(60000, generator=torch.Generator().manual_seed(config["seed"])).tolist()
    confirmation = [i for i in order if i not in excluded][: config["confirmation_samples"]]
    assert len(confirmation) == config["confirmation_samples"] and not set(confirmation) & excluded
    indices = dict(
        calibration=parent["MNIST"]["calibration"],
        confirmation={"MNIST": confirmation, "SVHN": parent["SVHN"]["development"]},
    )
    assert len(indices["calibration"]) == config["calibration_samples"]
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
    datasets = {
        "MNIST": MNIST(args.data, train=True, transform=original.val_preprocess),
        "SVHN": SVHN(args.data, split="train", transform=original.val_preprocess),
    }
    images = torch.stack([datasets["MNIST"][i][0] for i in indices["calibration"]])
    heads = {task: load(str(root / f"head_{task}.pt")) for task in datasets}
    loaders = {
        task: make_loader(datasets[task], subset)
        for task, subset in indices["confirmation"].items()
    }
    candidate = build_release(original, images, config, out)
    state, base_state = candidate.state_dict(), original.state_dict()
    changed_keys = [key for key in state if not torch.equal(state[key], base_state[key])]
    assert 72 <= len(changed_keys) <= 96 and all(
        ".attn." in k or ".mlp." in k for k in changed_keys
    )
    save_json(out / "release_support.json", dict(keys=changed_keys))
    rows, predictions, controls, edit_records, reset_models = [], {}, {}, [], {}

    def record(preset, group, model):
        for task, loader in loaders.items():
            if group.endswith("standalone") and task != "MNIST":
                continue
            chunks, labels = [], []

            def sink(prediction, target):
                chunks.append(prediction.clone())
                labels.append(target.clone())

            score = evaluate(model, heads[task], loader, "cuda", prediction_sink=sink)
            pred, target = torch.cat(chunks), torch.cat(labels)
            current = pred == target
            if preset == "unprotected":
                controls[group, task] = current, target
            reference, reference_labels = controls[group, task]
            assert torch.equal(target, reference_labels)
            rows.append(
                dict(
                    preset=preset,
                    group=group,
                    dataset=task,
                    split="training_confirmation" if task == "MNIST" else "inspected_training",
                    paired=paired_counts(reference, current),
                    **score,
                )
            )
            predictions[preset, group, task] = dict(prediction=pred, labels=target)
            save_json(out / "results.json", rows)
            print("JOINT_RESULT", preset, group, task, score["correct"], score["n"], flush=True)

    for preset, model in [("unprotected", original), ("joint_transfer", candidate)]:
        for group in config["groups"]:
            restored, edits = restoration(model, anchor, group, config["layers"])
            edit_records.append(dict(preset=preset, group=group, edits=edits))
            if group == "pretrained_attn_mlp_all":
                reset_models[preset] = restored
            record(preset, group + "_standalone", restored)
            merged = merge(anchor, restored, partner, config["coefficient"])
            record(preset, group, merged)
            del merged
    left, right = (reset_models[p].state_dict() for p in ["unprotected", "joint_transfer"])
    assert all(torch.equal(left[k], right[k]) for k in left)
    reset_group = "pretrained_attn_mlp_all"
    for group, task in [
        (reset_group, "MNIST"),
        (reset_group, "SVHN"),
        (reset_group + "_standalone", "MNIST"),
    ]:
        assert torch.equal(
            predictions["unprotected", group, task]["prediction"],
            predictions["joint_transfer", group, task]["prediction"],
        )
    lookup = {(r["preset"], r["group"], r["dataset"]): r for r in rows}
    loss = 100 * (
        lookup["unprotected", "direct_standalone", "MNIST"]["top1"]
        - lookup["joint_transfer", "direct_standalone", "MNIST"]["top1"]
    )
    gaps = {
        group: 100
        * (
            lookup["unprotected", group, "MNIST"]["top1"]
            - lookup["joint_transfer", group, "MNIST"]["top1"]
        )
        for group in config["groups"]
    }
    # Full support reset has exactly zero matched gap for every such release.
    # Judge overall security using the best measured attack on either release.
    reset_cost = 100 * (
        lookup["unprotected", "direct", "MNIST"]["top1"]
        - lookup["joint_transfer", reset_group, "MNIST"]["top1"]
    )
    best = {
        preset: max((lookup[preset, g, "MNIST"] for g in config["groups"]), key=lambda r: r["top1"])
        for preset in ["unprotected", "joint_transfer"]
    }
    envelope_gap = 100 * (best["unprotected"]["top1"] - best["joint_transfer"]["top1"])
    passed = loss <= config["utility_budget_pp"] and envelope_gap >= config["minimum_attack_gap_pp"]
    save_json(
        out / "decision.json",
        dict(
            utility_loss_pp=loss,
            matched_attack_gap_pp=gaps,
            complete_support_reset_cost_pp=reset_cost,
            best_measured_attack_gap_pp=envelope_gap,
            best_measured_attack={
                p: dict(group=r["group"], correct=r["correct"], n=r["n"], top1=r["top1"])
                for p, r in best.items()
            },
            passed_initial_gate=passed,
            further_aga_or_reconstruction_run=False,
            reason="Proceed to original AGA and reconstruction"
            if passed
            else "Reject this frozen construction; no rank, gain or ridge retuning",
        ),
    )
    save_json(out / "edits.json", edit_records)
    torch.save(predictions, out / "artifacts/predictions.pt")
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            new_defense_validated=False,
            optimizer_steps=0,
            closed_form_defender_fits=12,
            original_protected_checkpoint_available_to_attack=False,
            source_sha256=digest(__file__),
            helpers_sha256={
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "clip_benchmark.py",
                    "operator_diagnostics.py",
                    "polynomial_probe.py",
                    "subspace_probe.py",
                ]
            },
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            parent_indices_sha256=digest(parent_path),
            excluded_indices_sha256=digest(previous_path),
            indices_sha256=digest(out / "training_indices.json"),
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
            predictions_sha256=digest(out / "artifacts/predictions.pt"),
            release_sha256=digest(out / "artifacts/release.pt"),
            evaluations=len(rows),
            elapsed_seconds=time.perf_counter() - start,
            gpu=torch.cuda.get_device_name(),
            test_usage="none",
            full_support_reset_parameters_and_predictions_exactly_equal=True,
        ),
    )
    print("JOINT_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/joint_transfer_mnist.json")
    parser.add_argument("--output", default="outputs/2026-10-10/joint_transfer_mnist")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
