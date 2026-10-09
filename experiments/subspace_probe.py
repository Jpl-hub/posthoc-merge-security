"""Measure task activation subspaces and test complement-only post-hoc noise.

Reuses the pinned Task Arithmetic merger, classifier and original AGA attack.
Calibrates on training images, selects on disjoint training development subsets,
then tests one selected configuration. No expert training or basis obfuscation.
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
from clip_benchmark import checkpoint_loader, evaluate, merge, recover
from mechanisms import load_author
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import MNIST, SVHN


def save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def make_loader(dataset, indices=None, batch_size=128):
    if indices is not None:
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)


def split_training(dataset, sizes, seed):
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed)).tolist()
    result = {}
    offset = 0
    for name, count in sizes.items():
        result[name] = order[offset : offset + count]
        offset += count
    assert offset <= len(dataset)
    assert len(set(itertools.chain.from_iterable(result.values()))) == offset
    return result


def site_parameters(model, site):
    target, layer_text = site.split(":")
    block = model.model.visual.transformer.resblocks[int(layer_text)]
    if target == "V":
        return block.attn.in_proj_weight[1536:], block.attn.in_proj_bias[1536:]
    return block.mlp.c_fc.weight, block.mlp.c_fc.bias


@torch.no_grad()
def collect_moments(model, loader, sites):
    """Streaming centered moments of actual linear inputs, including all tokens."""
    sums, grams, counts = {}, {}, {}
    handles = []

    def capture(site, value):
        x = value.detach().reshape(-1, value.shape[-1]).double()
        if site not in sums:
            sums[site] = torch.zeros(x.shape[1], device=x.device, dtype=x.dtype)
            grams[site] = torch.zeros(x.shape[1], x.shape[1], device=x.device, dtype=x.dtype)
            counts[site] = 0
        sums[site].add_(x.sum(0))
        grams[site].add_(x.T @ x)
        counts[site] += len(x)

    for site in sites:
        target, layer_text = site.split(":")
        block = model.model.visual.transformer.resblocks[int(layer_text)]
        normalization = block.ln_1 if target == "V" else block.ln_2
        handles.append(
            normalization.register_forward_hook(
                lambda _module, _args, value, name=site: capture(name, value)
            )
        )
    try:
        model.cuda().eval()
        for images, _ in loader:
            model(images.cuda(non_blocking=True))
    finally:
        model.cpu()
        for handle in handles:
            handle.remove()
    result = {}
    for site in sites:
        mean = sums[site] / counts[site]
        covariance = grams[site] / counts[site] - mean[:, None] @ mean[None, :]
        result[site] = dict(
            mean=mean.cpu(),
            covariance=((covariance + covariance.T) / 2).cpu(),
            tokens=counts[site],
        )
    return result


@torch.no_grad()
def fit_basis(moment, energy):
    values, vectors = torch.linalg.eigh(moment["covariance"].cuda())
    values, vectors = values.flip(0).clamp_min(0), vectors.flip(1)
    fractions = values.cumsum(0) / values.sum()
    rank = int(torch.searchsorted(fractions, energy).item()) + 1
    assert 0 < rank < len(values), (rank, energy)
    curve = {str(r): float(fractions[r - 1]) for r in [16, 32, 64, 128, 256, 512, 768]}
    thresholds = {
        str(e): int(torch.searchsorted(fractions, e).item()) + 1 for e in [0.9, 0.99, 0.999, 0.9999]
    }
    basis = dict(
        mean=moment["mean"],
        u=vectors[:, :rank].cpu(),
        rank=rank,
        complement_rank=len(values) - rank,
    )
    stats = dict(
        rank=rank,
        complement_rank=len(values) - rank,
        ranks_at_energy=thresholds,
        explained_energy=curve,
        numerical_rank=int((values > values[0] * 1e-8).sum()),
        retained_fraction=float(fractions[rank - 1]),
        tokens=moment["tokens"],
    )
    return basis, stats


@torch.no_grad()
def complement_power(moment, basis):
    u = basis["u"]
    center_shift = moment["mean"] - basis["mean"]
    covariance = moment["covariance"] + center_shift[:, None] @ center_shift[None, :]
    total = covariance.trace()
    outside = total - (u.T @ covariance @ u).trace()
    return dict(total=float(total), outside=float(outside), fraction=float(outside / total))


@torch.no_grad()
def perturb(model, site, basis, strength, seed):
    """Delta W = scaled R(I-UU^T), Delta b = -Delta W mu.

    Strength is ||Delta W||_F / ||W||_F, not a claimed function-distance bound.
    """
    result = copy.deepcopy(model).cpu().eval()
    weight, bias = site_parameters(result, site)
    generator = torch.Generator().manual_seed(seed)
    random_matrix = torch.randn(weight.shape, dtype=torch.float64, generator=generator).cuda()
    u = basis["u"].cuda()
    delta = random_matrix - (random_matrix @ u) @ u.T
    delta *= strength * weight.double().norm().cuda() / delta.norm()
    orthogonality_error = float((delta @ u).norm() / delta.norm())
    assert orthogonality_error < 1e-10, orthogonality_error
    shift = delta @ basis["mean"].cuda()
    assert torch.isfinite(delta).all() and torch.isfinite(shift).all()
    weight.add_(delta.float().cpu())
    bias.sub_(shift.float().cpu())
    # Deployment uses float32; also check its actual main-space leakage.
    old_weight, _ = site_parameters(model, site)
    actual = (weight.double() - old_weight.double()).cuda()
    actual_leakage = float((actual @ u).norm() / actual.norm())
    assert actual_leakage < 1e-5, actual_leakage
    return result, dict(
        relative_parameter_change=float(actual.norm() / old_weight.double().norm().cuda()),
        main_space_relative_leakage_float64=orthogonality_error,
        main_space_relative_leakage_float32=actual_leakage,
        bias_change_norm=float(shift.norm()),
    )


@torch.no_grad()
def clean_complement(model, anchor, site, basis, mode):
    """Independent-data white-box attacks; never receive defender PCA or seed."""
    result = copy.deepcopy(model).cpu().eval()
    weight, bias = site_parameters(result, site)
    base, _ = site_parameters(anchor, site)
    difference = weight.double() if mode == "projection" else weight.double() - base.double()
    u = basis["u"]
    removed = difference - (difference @ u) @ u.T
    weight.sub_(removed.float())
    bias.add_((removed @ basis["mean"]).float())
    return result


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    artifact_dir = output / "artifacts"
    artifact_dir.mkdir()
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    (output / "protocol.json").write_bytes(config_bytes)
    source_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(config["seed"])
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    author, author_sha = load_author(Path(args.author_repo) / "utils_attack_B32.py")
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "MNIST/finetuned.pt"))
    second = load(str(root / "SVHN/finetuned.pt"))
    heads = {name: load(str(root / f"head_{name}.pt")) for name in ["MNIST", "SVHN"]}
    training = {
        "MNIST": MNIST(args.data, train=True, transform=original.val_preprocess),
        "SVHN": SVHN(args.data, split="train", transform=original.val_preprocess),
    }
    indices = {
        name: split_training(
            data,
            dict(
                calibration=config["calibration_samples"],
                attack=config["attack_samples"],
                development=config["development_samples"][name],
            ),
            config["seed"] + i,
        )
        for i, (name, data) in enumerate(training.items())
    }
    save_json(output / "training_indices.json", indices)
    calibration_loaders = {
        name: make_loader(data, indices[name]["calibration"], batch_size=32)
        for name, data in training.items()
    }
    development_loaders = {
        name: make_loader(data, indices[name]["development"]) for name, data in training.items()
    }
    attack_loader = make_loader(training["MNIST"], indices["MNIST"]["attack"], batch_size=32)
    all_sites = [f"{target}:{i}" for i in range(12) for target in ["V", "FC"]]
    moments = {
        name: collect_moments(original, loader, all_sites)
        for name, loader in calibration_loaders.items()
    }
    baseline_merge = merge(anchor, original, second, config["coefficient"])
    merged_moments = {
        name: collect_moments(baseline_merge, loader, all_sites)
        for name, loader in calibration_loaders.items()
    }
    bases, diagnostics = {}, []
    for site in all_sites:
        basis, stats = fit_basis(moments["MNIST"][site], config["retained_energy"])
        svhn_basis, svhn_stats = fit_basis(moments["SVHN"][site], config["retained_energy"])
        bases[site] = basis
        cross = torch.linalg.svdvals(basis["u"].T.cuda() @ svhn_basis["u"].cuda())
        diagnostics.append(
            dict(
                site=site,
                MNIST=stats,
                SVHN_on_MNIST_expert=svhn_stats,
                principal_overlap_mean_cosine_squared=float((cross * cross).mean()),
                input_complement={
                    "MNIST_expert_MNIST": complement_power(moments["MNIST"][site], basis),
                    "MNIST_expert_SVHN": complement_power(moments["SVHN"][site], basis),
                    **{
                        f"unprotected_merge_{name}": complement_power(by_site[site], basis)
                        for name, by_site in merged_moments.items()
                    },
                },
            )
        )
    save_json(output / "subspaces.json", diagnostics)
    torch.save(bases, artifact_dir / "defender_bases.pt")
    print("SUBSPACES_READY", len(diagnostics), flush=True)
    rows = []

    def record(split, group, model, preset, loaders, only_mnist=False):
        measurements = {}
        for name, loader in loaders.items():
            if only_mnist and name != "MNIST":
                continue
            row = dict(
                experiment="input_complement_noise",
                split=split,
                group=group,
                preset=preset["name"],
                site=preset.get("site"),
                strength=preset.get("strength", 0),
                dataset=name,
                coefficient=None if group.endswith("standalone") else config["coefficient"],
                seed=config["seed"],
                **evaluate(model, heads[name], loader, "cuda"),
            )
            rows.append(row)
            measurements[name] = row["top1"]
            save_json(output / "results.json", rows)
            print("SUBSPACE_RESULT", json.dumps(row), flush=True)
        return measurements

    control = dict(name="unprotected")
    standalone_dev = record(
        "development", "standalone", original, control, development_loaders, only_mnist=True
    )
    baseline_dev = record("development", "direct", baseline_merge, control, development_loaders)
    del baseline_merge
    presets = []
    for layer, target, strength in itertools.product(
        config["layers"], config["targets"], config["strengths"]
    ):
        site = f"{target}:{layer}"
        preset = dict(name=f"{target}_L{layer}_a{strength:g}", site=site, strength=strength)
        candidate, check = perturb(original, site, bases[site], strength, config["seed"] + layer)
        standalone = record(
            "development", "standalone", candidate, preset, development_loaders, only_mnist=True
        )
        combined = merge(anchor, candidate, second, config["coefficient"])
        accuracy = record("development", "direct", combined, preset, development_loaders)
        loss_pp = 100 * (standalone_dev["MNIST"] - standalone["MNIST"])
        preset.update(
            utility_loss_pp=loss_pp,
            utility_feasible=loss_pp <= config["utility_budget_pp"] + 1e-9,
            mean_merged_accuracy=sum(accuracy.values()) / len(accuracy),
            merge_drop_pp=100 * (sum(baseline_dev.values()) - sum(accuracy.values())) / 2,
            check=check,
        )
        presets.append(preset)
        save_json(output / "development_screen.json", presets)
        del candidate, combined
    feasible = [p for p in presets if p["utility_feasible"]]
    selection = dict(
        selection_data="disjoint_training_development_subsets",
        test_usage="none_before_selection",
        feasible_count=len(feasible),
        selected=min(feasible, key=lambda p: (p["mean_merged_accuracy"], p["strength"]))
        if feasible
        else None,
    )
    save_json(output / "selection.json", selection)
    print("SUBSPACE_SELECTION", json.dumps(selection), flush=True)
    if selection["selected"] is not None:
        preset = selection["selected"]
        candidate, _ = perturb(
            original,
            preset["site"],
            bases[preset["site"]],
            preset["strength"],
            config["seed"] + int(preset["site"].split(":")[1]),
        )
        torch.save(candidate.state_dict(), artifact_dir / "protected_state.pt")
        test_loaders = {
            "MNIST": make_loader(MNIST(args.data, train=False, transform=original.val_preprocess)),
            "SVHN": make_loader(SVHN(args.data, split="test", transform=original.val_preprocess)),
        }
        for model, item in [(original, control), (candidate, preset)]:
            record("official_test", "standalone", model, item, test_loaders, only_mnist=True)
            combined = merge(anchor, model, second, config["coefficient"])
            record("official_test", "direct", combined, item, test_loaders)
            del combined
            # The attack receives only released weights, a public anchor and
            # independent calibration. The original expert is a control only.
            moment = collect_moments(model, attack_loader, [preset["site"]])[preset["site"]]
            attack_basis, attack_stats = fit_basis(moment, config["retained_energy"])
            attack_dir = output / item["name"]
            attack_dir.mkdir()
            save_json(attack_dir / "attacker_subspace.json", attack_stats)
            torch.save(attack_basis, artifact_dir / f"{item['name']}_attacker_basis.pt")
            for attack in ["AGA", "projection", "completion"]:
                attacked = (
                    recover(model, anchor, author, attack_dir)
                    if attack == "AGA"
                    else clean_complement(model, anchor, preset["site"], attack_basis, attack)
                )
                record(
                    "official_test",
                    f"{attack}_standalone",
                    attacked,
                    item,
                    test_loaders,
                    only_mnist=True,
                )
                combined = merge(anchor, attacked, second, config["coefficient"])
                record("official_test", attack, combined, item, test_loaders)
                del attacked, combined
    metadata = dict(
        completed=True,
        total_seconds=time.perf_counter() - start,
        config_sha256=hashlib.sha256(config_bytes).hexdigest(),
        author_sha256=author_sha,
        experiment_source_sha256=source_sha,
        training="none",
        source="official_TaskArithmetic_ViT-B-32_MNIST_and_SVHN",
        calibration=config["calibration"],
        development=config["development"],
        attack_visibility=config["attack"],
        test_usage="evaluation_only_after_development_selection",
        utility_budget_pp=config["utility_budget_pp"],
        perturbation_other_protections="none",
        evaluation_implementation="author_ImageClassifier_get_logits",
        merge_implementation="author_TaskVector_difference_and_sum",
        seed_count=1,
        data_indices_sha256=hashlib.sha256(
            (output / "training_indices.json").read_bytes()
        ).hexdigest(),
    )
    save_json(output / "metadata.json", metadata)
    print("SUBSPACE_COMPLETE", json.dumps(metadata), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/subspace_probe.json")
    parser.add_argument("--output", default="outputs/2026-10-09/post_meeting/subspace_probe")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
