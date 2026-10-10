"""Cost of deleting target-carrying rank-one updates across local modules.

This is a bounded necessary-condition test, not a protected-model result.
Directions use pinned NUFILT SVD; public replacements and exact-direction
oracle diagnostics are reported separately. No optimizer or response fitting.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate
from eight_task_benchmark import digest, merge_all
from fine_subspace import fresh_indices
from joint_posthoc import flatten_tokens
from nufilt_alignment import author_svd
from operator_diagnostics import paired_counts
from subspace_probe import make_loader, save_json


def parameters(state, site):
    kind, layer = site.split(":")
    prefix = f"model.visual.transformer.resblocks.{layer}."
    suffix = {
        "v": "attn.in_proj_",
        "out": "attn.out_proj.",
        "up": "mlp.c_fc.",
        "down": "mlp.c_proj.",
    }[kind]
    keys = (prefix + suffix + "weight", prefix + suffix + "bias")
    weight, bias = (state[key] for key in keys)
    if kind == "v":
        width = weight.shape[1]
        weight, bias = weight[2 * width :], bias[2 * width :]
    return keys, weight, bias


@torch.no_grad()
def capture(model, images, layers):
    """Read real linear inputs; recover MHA contexts with identity output clones."""
    raw, handles, errors, clones = {}, [], [], []
    model.cuda().eval()
    for layer in layers:
        block = model.model.visual.transformer.resblocks[layer]
        for kind, module in [("v", block.ln_1), ("up", block.ln_2)]:
            site = f"{kind}:{layer}"
            raw[site] = []
            handles.append(
                module.register_forward_hook(
                    lambda _m, _a, value, k=site: raw[k].append(flatten_tokens(value))
                )
            )
        site = f"down:{layer}"
        raw[site] = []
        handles.append(
            block.mlp.c_proj.register_forward_pre_hook(
                lambda _m, args, k=site: raw[k].append(flatten_tokens(args[0]))
            )
        )
        clone = copy.deepcopy(block.attn).eval()
        clone.out_proj.weight.copy_(torch.eye(clone.embed_dim, device="cuda"))
        clone.out_proj.bias.zero_()
        clones.append(clone)
        site = f"out:{layer}"
        raw[site] = []

        def attention_hook(module, args, kwargs, result, c=clone, k=site):
            context = c(*args, **kwargs)[0]
            reconstructed = torch.nn.functional.linear(
                context, module.out_proj.weight, module.out_proj.bias
            )
            error = float((reconstructed - result[0]).abs().max())
            errors.append(dict(site=k, max_absolute_error=error))
            assert error < 2e-5, (k, error)
            raw[k].append(flatten_tokens(context))

        handles.append(block.attn.register_forward_hook(attention_hook, with_kwargs=True))
    try:
        for batch in images.split(16):
            model(batch.cuda())
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return {k: torch.cat(v) for k, v in raw.items()}, errors


@torch.no_grad()
def directions(teacher, anchor, inputs, layers, svd):
    records, evidence = [], {}
    own, base = teacher.state_dict(), anchor.state_dict()
    for layer in layers:
        for kind in ("v", "out", "up", "down"):
            site = f"{kind}:{layer}"
            _, weight, _ = parameters(own, site)
            _, pretrained, _ = parameters(base, site)
            delta = (weight.double() - pretrained.double()).cuda()
            _, singular, vectors = svd(delta, full_matrices=False)
            v = vectors[:, 0].clone()
            null_leakage, adjustment = None, 0.0
            if kind in ("v", "up"):
                block = teacher.model.visual.transformer.resblocks[layer]
                norm = block.ln_1 if kind == "v" else block.ln_2
                null = norm.weight.double().cuda().reciprocal()
                null /= null.norm()
                v -= (v @ null) * null
                adjustment = float(v.norm())
                assert adjustment > 1e-8, site
                v /= v.norm()
                null_leakage = float(abs(v @ null))
                assert null_leakage < 1e-10
            v *= 1 if v[v.abs().argmax()] >= 0 else -1
            outgoing = delta @ v
            mean = inputs[site].double().mean(0)
            power = float(((inputs[site].double() - mean) @ v.cpu()).square().mean())
            records.append(
                dict(
                    site=site,
                    leading_singular_value=float(singular[0]),
                    leading_task_weight_energy_fraction=float(singular[0].square())
                    / float(singular.square().sum()),
                    removed_task_weight_energy_fraction=float(outgoing.square().sum())
                    / float(delta.square().sum()),
                    centered_input_power=power,
                    rank_one_task_response_rms=float(outgoing.norm()) * power**0.5,
                    affine_null_leakage=null_leakage,
                    norm_after_affine_null_removal=adjustment if kind in ("v", "up") else None,
                )
            )
            evidence[site] = dict(mean=mean, incoming=v.cpu(), outgoing=outgoing.cpu())
            print("DIRECTION", records[-1], flush=True)
    return records, evidence


@torch.no_grad()
def reset(teacher, anchor, evidence, mode):
    model = copy.deepcopy(teacher).cpu().eval()
    state, own, base = model.state_dict(), teacher.state_dict(), anchor.state_dict()
    edited = set()
    for site, direction in evidence.items():
        kind = site.split(":")[0]
        if mode == "oracle_attention" and kind not in ("v", "out"):
            continue
        if mode == "oracle_mlp" and kind not in ("up", "down"):
            continue
        keys, weight, bias = parameters(state, site)
        _, w0, b0 = parameters(base, site)
        _, we, be = parameters(own, site)
        if mode == "public_support":
            weight.copy_(w0)
            bias.copy_(b0)
        else:
            delta = we.double() - w0.double()
            v, mean = direction["incoming"], direction["mean"]
            removed = (delta @ v[:, None]) @ v[None, :]
            weight.copy_((we.double() - removed).float())
            bias.copy_((be.double() + removed @ mean).float())
        edited.update(keys)
    assert all(torch.equal(t, own[k]) for k, t in state.items() if k not in edited)
    for layer in {int(site.split(":")[1]) for site in evidence}:
        key = f"model.visual.transformer.resblocks.{layer}.attn.in_proj_"
        width = state[key + "weight"].shape[1]
        for suffix in ("weight", "bias"):
            assert torch.equal(state[key + suffix][: 2 * width], own[key + suffix][: 2 * width])
    return model


def cached_loader(data, indices=None):
    from torch.utils.data import DataLoader, TensorDataset

    batches = list(make_loader(data, indices))
    return DataLoader(
        TensorDataset(torch.cat([b[0] for b in batches]), torch.cat([b[1] for b in batches])),
        batch_size=128,
        pin_memory=True,
    )


@torch.no_grad()
def run(args):
    config = json.loads(Path(args.config).read_bytes())
    out, root = Path(args.output), Path(args.checkpoints)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", config)
    started = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    parent = Path(config["profile_parent"])
    selection = json.loads((parent / "selection.json").read_bytes())[config["task"]]
    assert config["layers"] == selection["layers"] and not selection["partner_used"]
    fusion = json.loads(Path(config["fusion_config"]).read_bytes())
    assert config["coefficient"] == fusion["coefficient"] == 0.2
    load = checkpoint_loader(Path(args.author_repo))
    teacher = load(str(root / f"{config['task']}/finetuned.pt")).cpu().eval()
    anchor = load(str(root / "zeroshot.pt")).cpu().eval()
    package = types.ModuleType("dataset")
    package.__path__ = [str(Path(args.author_repo).resolve() / "datasets")]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    data = get_dataset(config["task"], teacher.val_preprocess, args.data)
    indices = fresh_indices(config, config["task"], len(data.train_dataset))
    save_json(out / "training_indices.json", {config["task"]: indices})
    images = {
        name: torch.stack([data.train_dataset[i][0] for i in indices[name]])
        for name in ("calibration", "confirmation")
    }
    captures, checks = {}, {}
    for split, inputs in images.items():
        captures[split], checks[split] = capture(teacher, inputs, config["layers"])
    svd_config = json.loads(Path(config["nufilt_config"]).read_bytes())
    svd = author_svd(args.nufilt_repo, svd_config)
    records, evidence = directions(teacher, anchor, captures["calibration"], config["layers"], svd)
    for row in records:
        site = row["site"]
        direction = evidence[site]
        z = captures["confirmation"][site].double() - direction["mean"]
        row["confirmation_input_power"] = float((z @ direction["incoming"]).square().mean())
        row["confirmation_over_calibration_power"] = row["confirmation_input_power"] / max(
            row["centered_input_power"], 1e-30
        )
    torch.save(captures, artifacts / "activations.pt")
    torch.save(evidence, artifacts / "directions.pt")
    save_json(out / "directions.json", records)
    save_json(out / "attention_context_checks.json", checks)
    save_json(
        out / "direction_freeze.json",
        dict(
            sha256=digest(artifacts / "directions.pt"),
            future_partner_checkpoints_loaded=False,
            calibration_labels_used=False,
            optimizer_steps=0,
            response_fits=0,
        ),
    )
    del captures
    models = {task: load(str(root / task / "finetuned.pt")) for task in fusion["tasks"]}
    models.update({config["task"]: teacher, "anchor": anchor})
    head = load(str(root / f"head_{config['task']}.pt"))
    confirmation = json.loads((parent / "training_indices.json").read_bytes())[config["task"]][
        "confirmation"
    ]
    save_json(out / "cost_indices.json", confirmation)
    loader = cached_loader(data.train_dataset, confirmation)
    rows, raw = [], {}
    for preset in config["presets"]:
        assert time.perf_counter() - started < config["time_limit_seconds"]
        own = teacher if preset == "original" else reset(teacher, anchor, evidence, preset)
        combined = merge_all(dict(models, **{config["task"]: own}), root, config["coefficient"])
        for group, model in [("standalone", own), ("fusion", combined)]:
            p, y = [], []
            score = evaluate(
                model, head, loader, "cuda", prediction_sink=lambda a, b: (p.append(a), y.append(b))
            )
            key = f"{preset}/{group}"
            raw[key] = dict(prediction=torch.cat(p), labels=torch.cat(y))
            reference = raw[f"original/{group}"]
            assert torch.equal(raw[key]["labels"], reference["labels"])
            rows.append(
                dict(
                    id=key,
                    preset=preset,
                    group=group,
                    dataset=config["task"],
                    paired=paired_counts(
                        reference["prediction"].eq(reference["labels"]),
                        raw[key]["prediction"].eq(raw[key]["labels"]),
                    ),
                    **score,
                )
            )
            torch.save(raw, artifacts / "predictions.pt")
            save_json(out / "results.json", rows)
            print("COST", key, score["correct"], score["n"], flush=True)
        del combined
    costs = {r["preset"]: r["paired"]["loss_pp"] for r in rows if r["group"] == "fusion"}
    proceed = all(
        costs[p] >= config["minimum_fusion_erasure_cost_pp"]
        for p in ("public_support", "oracle_union")
    )
    save_json(
        out / "decision.json",
        dict(
            recovery_cost_gate_passed=proceed,
            fusion_cost_pp=costs,
            candidate_constructed=False,
            protected_model_evaluated=False,
            aga_run=False,
            next_stage="joint functional-cancellation probe"
            if proceed
            else "stop this direction family",
            scope="Unprotected rank-one deletion cost on previously inspected training confirmation",
        ),
    )
    paths = [
        "zeroshot.pt",
        *[f"{t}/finetuned.pt" for t in fusion["tasks"]],
        f"head_{config['task']}.pt",
    ]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            checkpoint_sha256={p: digest(root / p) for p in paths},
            helper_sha256={
                name: digest(Path(__file__).with_name(name))
                for name in [
                    "clip_benchmark.py",
                    "eight_task_benchmark.py",
                    "fine_subspace.py",
                    "functional_contrast.py",
                    "joint_posthoc.py",
                    "mechanisms.py",
                    "nufilt_alignment.py",
                    "operator_diagnostics.py",
                    "subspace_probe.py",
                ]
            },
            parent_selection_sha256=digest(parent / "selection.json"),
            exclusions_sha256={
                p: digest(p) for p in [*config["exclusions"], config["mnist_exclusions"]]
            },
            upstream_commit=svd_config["upstream_commit"],
            upstream_svd_sha256=svd_config["upstream_svd_sha256"],
            elapsed_seconds=time.perf_counter() - started,
            optimizer_steps=0,
            calibration_labels_used=False,
            constructor_future_partners_used=False,
            official_test_used=False,
            rows=len(rows),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/critical_subspace_mnist.json")
    parser.add_argument("--output", default="outputs/2026-10-11/critical_subspace_mnist")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--nufilt-repo", default="external/NUFILT")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
