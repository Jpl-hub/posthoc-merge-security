"""Target-only singular MLP projection and its beta-path mechanism screen.

Existing author neuron folding is reused with supplied cluster assignments.
No gradient, iterative finetuning, response reconstruction or partner access.
Construction results are not standalone accuracy or a successful defense.
"""

from __future__ import annotations

import argparse
import ast
import copy
import math
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from critical_subspace import capture
from eight_task_benchmark import digest
from local_erasure import rescale
from path_subspace import context, read, response
from subspace_probe import save_json


def upstream_fold(cfg):
    """Extract the unchanged author method; do not import training dependencies."""
    path = Path(cfg["upstream_source"])
    assert digest(path) == cfg["upstream_sha256"]
    tree = ast.parse(path.read_text())
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "CLIPViT_ModelFolding"
    )
    function = next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "compress_function"
    )
    module = ast.Module(body=[function], type_ignores=[])
    namespace = dict(torch=torch)
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["compress_function"], namespace


def activation(z):
    return z * torch.sigmoid(1.702 * z)


@torch.no_grad()
def pair_error(original, anchor, inputs, beta, pair):
    """Exact frozen-input error, including simultaneous up/down cross terms."""
    i, j = pair
    x = inputs.double().cuda()
    ew, eb = original.c_fc.weight.double().cuda(), original.c_fc.bias.double().cuda()
    av, ev = anchor.c_proj.weight.double().cuda(), original.c_proj.weight.double().cuda()
    aw, ab = anchor.c_fc.weight.double().cuda(), anchor.c_fc.bias.double().cuda()
    z = x @ ((1 - beta) * aw[[i, j]] + beta * ew[[i, j]]).T
    z += (1 - beta) * ab[[i, j]] + beta * eb[[i, j]]
    difference = x @ (ew[j] - ew[i]) + eb[j] - eb[i]
    updated = z + beta * difference[:, None] * torch.tensor([0.5, -0.5], device="cuda")
    before, after = activation(z), activation(updated)
    average = after.mean(1, keepdim=True)
    coefficients = torch.cat([(1 - beta) * (after - before), beta * (average - before)], 1)
    columns = torch.cat([av[:, [i, j]], ev[:, [i, j]]], 1)
    energy = float(((coefficients @ (columns.T @ columns)) * coefficients).sum())
    return coefficients, columns, max(energy, 0.0)


@torch.no_grad()
def shortlist(mlp, inputs, count):
    x = inputs.double().cuda()
    z = x @ mlp.c_fc.weight.double().cuda().T + mlp.c_fc.bias.double().cuda()
    gram = z.T @ z / len(z)
    distance = (gram.diag()[:, None] + gram.diag()[None, :] - 2 * gram).clamp_min(0)
    norms = mlp.c_proj.weight.double().cuda().norm(dim=0)
    bounds = distance * (norms[:, None] + norms[None, :]).square() * ((1 + 1 / math.e) / 2) ** 2
    lower = torch.tril(torch.ones_like(bounds, dtype=torch.bool))
    bounds[lower] = torch.inf
    flat = torch.argsort(bounds.flatten(), stable=True)[:count]
    pairs = [(int(i // len(z[0])), int(i % len(z[0]))) for i in flat.cpu()]
    native = activation(z) @ mlp.c_proj.weight.double().cuda().T + mlp.c_proj.bias.double().cuda()
    return pairs, float(native.square().sum()), float(native.square().sum() / len(native))


@torch.no_grad()
def compile_layer(mlp, pairs, fold, namespace):
    size = mlp.c_fc.out_features
    groups, used = list(pairs), {index for pair in pairs for index in pair}
    groups.extend((i,) for i in range(size) if i not in used)
    labels = torch.empty(size, dtype=torch.long, device="cuda")
    for index, group in enumerate(groups):
        labels[list(group)] = index

    class FixedClustering:
        def __init__(self, **_kwargs):
            pass

        def __call__(self, _weights):
            return labels.clone()

    namespace["WeightClustering"] = FixedClustering
    params = {
        "up": mlp.c_fc.weight.double().cuda(),
        "up.bias": mlp.c_fc.bias.double().cuda(),
        "down": mlp.c_proj.weight.double().cuda(),
        "down.bias": mlp.c_proj.bias.double().cuda(),
    }
    compiled, _ = fold(
        SimpleNamespace(keep_ratio=(len(groups) + 0.25) / size, min_channels=1),
        [("up", None), ("down", None)],
        params,
    )
    result = copy.deepcopy(mlp).cpu().eval()
    for index, pair in enumerate(pairs):
        for neuron in pair:
            result.c_fc.weight[neuron].copy_(compiled["up.weight"][index].float())
            result.c_fc.bias[neuron].copy_(compiled["up.bias"][index].float())
            result.c_proj.weight[:, neuron].copy_((compiled["down.weight"][:, index] / 2).float())
    return result


@torch.no_grad()
def construct(original, anchor, captured, cfg):
    fold, namespace = upstream_fold(cfg)
    release = copy.deepcopy(original).cpu().eval()
    selected, scores, checks = {}, {}, []
    for layer in cfg["layers"]:
        native = original.model.visual.transformer.resblocks[layer].mlp
        base = anchor.model.visual.transformer.resblocks[layer].mlp
        native.cuda().double().eval()
        base.cuda().double().eval()
        inputs = {name: value[f"up:{layer}"].double().cuda() for name, value in captured.items()}
        pairs, energy, per_token = shortlist(native, inputs["expert"], cfg["shortlist_pairs"])
        records = []
        for pair in pairs:
            _, _, own = pair_error(native, base, inputs["expert"], 1.0, pair)
            paths = {
                str(beta): pair_error(native, base, inputs[str(beta)], beta, pair)[2]
                for beta in cfg["betas"]
            }
            denominator = own + energy * cfg["ratio_relative_rms_floor"] ** 2
            records.append(
                dict(
                    pair=list(pair),
                    native_energy=own,
                    path_energy=paths,
                    ratio=sum(paths.values()) / len(paths) / denominator,
                )
            )
        records.sort(key=lambda item: (-item["ratio"], item["pair"]))
        accepted, used = [], set()
        error = torch.zeros(
            len(inputs["expert"]), native.c_proj.out_features, dtype=torch.float64, device="cuda"
        )
        for item in records:
            pair = item["pair"]
            if used.intersection(pair):
                continue
            coefficients, columns, _ = pair_error(native, base, inputs["expert"], 1.0, pair)
            trial = error + coefficients @ columns.T
            relative = float(trial.norm() / energy**0.5)
            if relative <= cfg["local_relative_rms_budget"]:
                accepted.append(pair)
                used.update(pair)
                error = trial
        result = compile_layer(native, accepted, fold, namespace)
        release.model.visual.transformer.resblocks[layer].mlp = result
        selected[str(layer)] = accepted
        scores[str(layer)] = records
        native.cuda().double().eval()
        result.cuda().double().eval()
        x = inputs["expert"].double().cuda()
        actual = result(x) - native(x)
        native.cpu().float()
        base.cpu().float()
        result.cpu().float()
        relative = float(actual.norm() / energy**0.5)
        prediction_error = float((actual - error).norm() / actual.norm().clamp_min(1e-20))
        assert relative <= cfg["local_relative_rms_budget"] + 2e-6
        assert prediction_error < 2e-4 or float(actual.norm()) < 1e-6
        checks.append(
            dict(
                layer=layer,
                pairs=len(accepted),
                discarded_hidden_dimensions=len(accepted),
                native_relative_rms=relative,
                pair_sum_prediction_error=prediction_error,
                native_output_per_token_energy=per_token,
            )
        )
        print("PROJECTION", checks[-1], flush=True)
    changed = {
        key: value.clone()
        for key, value in release.state_dict().items()
        if not torch.equal(value, original.state_dict()[key])
    }
    assert all(".mlp.c_fc." in key or key.endswith(".mlp.c_proj.weight") for key in changed)
    assert all(torch.isfinite(value).all() for value in changed.values())
    return release, changed, selected, scores, checks


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    save_json(out / "protocol.json", cfg)
    root, load, _fusion, models = context(cfg)
    task = cfg["task"]
    anchor = models["anchor"]
    original = load(str(root / task / "finetuned.pt")).cpu().eval()
    pins = {
        item["path"].split("/", 1)[1]: item["sha256"]
        for item in read("configs/checkpoints.json")["files"]
    }
    for relative in ["zeroshot.pt", f"{task}/finetuned.pt"]:
        assert digest(root / relative) == pins[relative]
    from dataset.registry import get_dataset

    data = get_dataset(task, original.val_preprocess, "data", batch_size=128, num_workers=2)
    indices = read(Path(cfg["indices_parent"]) / "training_indices.json")[task]
    assert len(indices["calibration"]) == 64 and len(indices["confirmation"]) == 24
    assert not set(indices["calibration"]).intersection(indices["confirmation"])
    assert cfg["layers"] == read(Path(cfg["profile_parent"]) / "selection.json")[task]["layers"]
    images = torch.stack([data.train_dataset[i][0] for i in indices["calibration"]])
    confirmation = torch.stack([data.train_dataset[i][0] for i in indices["confirmation"]])
    captured = {}
    for name, beta in [("expert", 1.0)] + [(str(beta), beta) for beta in cfg["betas"]]:
        model = original if beta == 1 else rescale(original, anchor, beta)
        value, _ = capture(model, images, cfg["layers"])
        captured[name] = {key: tensor for key, tensor in value.items() if key.startswith("up:")}
    release, changed, selected, scores, checks = construct(original, anchor, captured, cfg)
    torch.save(changed, artifacts / "release_edits.pt")
    torch.save(captured, artifacts / "inputs.pt")
    save_json(out / "selected.json", selected)
    save_json(out / "pair_scores.json", scores)
    save_json(out / "construction_checks.json", checks)
    save_json(out / "training_indices.json", indices)
    save_json(
        out / "release_manifest.json",
        dict(
            sha256=digest(artifacts / "release_edits.pt"),
            edited_keys=list(changed),
            future_partner_loaded=False,
            labels_used=False,
            optimizer_steps=0,
            response_reconstruction_fits=0,
            calibration_dependent_selection=True,
        ),
    )
    print("RELEASE_FROZEN", digest(artifacts / "release_edits.pt"), flush=True)
    functions = {}
    for name, beta in [("expert", 1.0)] + [(str(beta), beta) for beta in cfg["betas"]]:
        before = original if beta == 1 else rescale(original, anchor, beta)
        after = release if beta == 1 else rescale(release, anchor, beta)
        functions[name] = {
            split: response(before, after, batch)
            for split, batch in [("calibration", images), ("confirmation", confirmation)]
        }
    save_json(out / "functions.json", functions)
    helpers = [
        "critical_subspace.py",
        "eight_task_benchmark.py",
        "local_erasure.py",
        "path_subspace.py",
        "clip_benchmark.py",
        "functional_contrast.py",
        "joint_posthoc.py",
        "subspace_probe.py",
        "operator_diagnostics.py",
        "fine_subspace.py",
        "nufilt_alignment.py",
        "mechanisms.py",
        "score_recovery.py",
        "polynomial_probe.py",
        "low_activity_posthoc.py",
    ]
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            elapsed_seconds=time.perf_counter() - start,
            stage="construction_screen_only",
            defense_success_claimed=False,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            helper_sha256={name: digest(Path("experiments") / name) for name in helpers},
            checkpoint_sha256={
                str(path.relative_to(root)): digest(path)
                for path in [root / "zeroshot.pt", root / task / "finetuned.pt"]
            },
            input_sha256={
                name: digest(name)
                for name in [
                    cfg["fusion_config"],
                    "configs/checkpoints.json",
                    str(Path(cfg["indices_parent"]) / "training_indices.json"),
                    str(Path(cfg["profile_parent"]) / "selection.json"),
                    cfg["upstream_source"],
                ]
            },
            raw_sha256={path.name: digest(path) for path in artifacts.iterdir() if path.is_file()},
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/neuron_projection.json")
    parser.add_argument("--output", default="outputs/2026-10-11/neuron_projection")
    run(parser.parse_args())
