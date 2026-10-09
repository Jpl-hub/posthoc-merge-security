"""Causal decomposition of complement noise and an FC recovery contrast.

Uses the parent experiment's frozen training indices and calibration bases.
Reused parent measurements are labelled; new contrasts receive full tests.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import time
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate, merge, recover
from mechanisms import load_author
from subspace_probe import (
    clean_complement,
    collect_moments,
    fit_basis,
    make_loader,
    perturb,
    save_json,
    site_parameters,
)
from torchvision.datasets import MNIST, SVHN


@torch.no_grad()
def decompose(original, protected, site):
    target, layer_text = site.split(":")
    assert target == "V"
    layer = int(layer_text)
    norm = original.model.visual.transformer.resblocks[layer].ln_1
    assert float(norm.weight.abs().min()) > 1e-8
    normal = norm.weight.double().reciprocal()
    normal /= normal.norm()
    weight, bias = site_parameters(original, site)
    released_weight, released_bias = site_parameters(protected, site)
    delta = released_weight.double() - weight.double()
    bias_delta = released_bias.double() - bias.double()
    component = delta @ normal
    null_delta = component[:, None] @ normal[None, :]
    null_bias = -component * (normal @ norm.bias.double())
    models = {}
    for name, dw, db in [
        ("V_layernorm_component", null_delta, null_bias),
        ("V_non_layernorm_component", delta - null_delta, bias_delta - null_bias),
    ]:
        model = copy.deepcopy(original).cpu().eval()
        w, b = site_parameters(model, site)
        w.add_(dw.float())
        b.add_(db.float())
        models[name] = model
    reconstruction = (
        site_parameters(models["V_layernorm_component"], site)[0].double()
        + site_parameters(models["V_non_layernorm_component"], site)[0].double()
        - weight.double()
    )
    stats = dict(
        site=site,
        full_relative_edit=float(delta.norm() / weight.double().norm()),
        null_edit_energy_fraction=float(null_delta.square().sum() / delta.square().sum()),
        null_relative_edit=float(null_delta.norm() / weight.double().norm()),
        other_relative_edit=float((delta - null_delta).norm() / weight.double().norm()),
        reconstruction_relative_error=float(
            (reconstruction - released_weight.double()).norm() / released_weight.double().norm()
        ),
        component_scaling="none; additive decomposition of the actual published float32 edit",
    )
    assert stats["reconstruction_relative_error"] < 1e-6
    return models, stats


@torch.no_grad()
def parameter_diagnostic(model, reference, original, anchor, site, moments):
    weight, bias = site_parameters(model, site)
    rw, rb = site_parameters(reference, site)
    ow, _ = site_parameters(original, site)
    aw, _ = site_parameters(anchor, site)
    w, b, dw, db = (
        weight.double(),
        bias.double(),
        weight.double() - rw.double(),
        bias.double() - rb.double(),
    )
    local = {}
    for name, moment in moments.items():
        center, covariance = moment["mean"], moment["covariance"]
        changed_mean = dw @ center + db
        original_mean = rw.double() @ center + rb.double()
        effect = ((dw @ covariance) * dw).sum() + changed_mean.square().sum()
        base = ((rw.double() @ covariance) * rw.double()).sum() + original_mean.square().sum()
        assert float(effect) > -1e-6 * max(float(base), 1), (name, effect)
        local[name] = dict(
            relative_linear_output_rms=float((effect.clamp_min(0) / base).sqrt()),
            mean_shift_norm=float(changed_mean.norm()),
        )
    return dict(
        site=site,
        weight_norm=float(w.norm()),
        norm_over_original=float(w.norm() / ow.double().norm()),
        norm_over_pretrained=float(w.norm() / aw.double().norm()),
        bias_norm=float(b.norm()),
        relative_weight_difference_to_reference=float(dw.norm() / rw.double().norm()),
        same_input_linear_outputs=local,
        output_scope="local V or FC preactivation on fixed reference inputs; not full network error",
    )


def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    (out / "protocol.json").write_bytes(config_bytes)
    parent = Path(config["parent"])
    parent_meta = json.loads((parent / "metadata.json").read_text())
    assert parent_meta["completed"]
    parent_config = json.loads((parent / "protocol.json").read_bytes())
    assert config["coefficient"] == parent_config["coefficient"]
    torch.set_num_threads(4)
    torch.manual_seed(parent_config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    load = checkpoint_loader(Path(args.author_repo))
    author, author_sha = load_author(Path(args.author_repo) / "utils_attack_B32.py")
    root = Path(args.checkpoints)
    anchor = load(str(root / "zeroshot.pt"))
    original = load(str(root / "MNIST/finetuned.pt"))
    second = load(str(root / "SVHN/finetuned.pt"))
    protected = copy.deepcopy(original).cpu().eval()
    protected.load_state_dict(torch.load(parent / "artifacts/protected_state.pt"), strict=True)
    selected = json.loads((parent / "selection.json").read_text())["selected"]
    bases = torch.load(parent / "artifacts/defender_bases.pt")
    contrast_models, decomposition = decompose(original, protected, selected["site"])
    save_json(out / "decomposition.json", decomposition)
    presets = json.loads((parent / "development_screen.json").read_text())
    feasible_fc = [p for p in presets if p["site"].startswith("FC:") and p["utility_feasible"]]
    fc = min(feasible_fc, key=lambda p: (p["mean_merged_accuracy"], p["strength"]))
    save_json(out / "fc_selection.json", dict(source="parent_development_only", selected=fc))
    fc_model, _ = perturb(
        original,
        fc["site"],
        bases[fc["site"]],
        fc["strength"],
        parent_config["seed"] + int(fc["site"].split(":")[1]),
    )
    contrast_models["FC_development_selected"] = fc_model
    torch.save(fc_model.state_dict(), artifacts / "fc_protected_state.pt")
    heads = {task: load(str(root / f"head_{task}.pt")) for task in ["MNIST", "SVHN"]}
    training = {
        "MNIST": MNIST(args.data, train=True, transform=original.val_preprocess),
        "SVHN": SVHN(args.data, split="train", transform=original.val_preprocess),
    }
    indices = json.loads((parent / "training_indices.json").read_text())
    test_loaders = {
        "MNIST": make_loader(MNIST(args.data, train=False, transform=original.val_preprocess)),
        "SVHN": make_loader(SVHN(args.data, split="test", transform=original.val_preprocess)),
    }
    rows = []
    reused_names = {"unprotected": "control", selected["name"]: "V_full"}
    parent_rows_bytes = (parent / "results.json").read_bytes()
    for row in json.loads(parent_rows_bytes):
        if row["split"] == "official_test" and row["group"] in {
            "standalone",
            "direct",
            "AGA_standalone",
            "AGA",
        }:
            rows.append(
                dict(**row, contrast=reused_names[row["preset"]], origin="parent_measurement")
            )
    save_json(out / "results.json", rows)

    def record(name, group, model, site):
        for task, loader in test_loaders.items():
            if group.endswith("standalone") and task != "MNIST":
                continue
            row = dict(
                contrast=name,
                group=group,
                dataset=task,
                site=site,
                origin="new_evaluation",
                coefficient=None if group.endswith("standalone") else config["coefficient"],
                seed=parent_config["seed"],
                **evaluate(model, heads[task], loader, "cuda"),
            )
            rows.append(row)
            save_json(out / "results.json", rows)
            print("DIAGNOSTIC_RESULT", json.dumps(row), flush=True)

    sites = [selected["site"], fc["site"]]
    calibration_loaders = {
        task: make_loader(data, indices[task]["calibration"], batch_size=32)
        for task, data in training.items()
    }
    baseline_merge = merge(anchor, original, second, config["coefficient"])
    fixed_moments = {}
    for model_name, model in [("expert", original), ("unprotected_merge", baseline_merge)]:
        for task, loader in calibration_loaders.items():
            fixed_moments[f"{model_name}_{task}"] = collect_moments(model, loader, sites)
    del baseline_merge
    control_dir = out / "control_AGA"
    control_dir.mkdir()
    aga_control = recover(original, anchor, author, control_dir)
    diagnostics = {}
    for name, model, site in [
        ("V_full", protected, selected["site"]),
        *[
            (name, model, selected["site"])
            for name, model in contrast_models.items()
            if name.startswith("V_")
        ],
        ("FC_development_selected", fc_model, fc["site"]),
    ]:
        by_view = {view: moments[site] for view, moments in fixed_moments.items()}
        raw_diagnostic = parameter_diagnostic(model, original, original, anchor, site, by_view)
        attack_dir = out / f"{name}_AGA"
        attack_dir.mkdir()
        attacked = recover(model, anchor, author, attack_dir)
        recovered_diagnostic = parameter_diagnostic(
            attacked, aga_control, original, anchor, site, by_view
        )
        diagnostics[name] = dict(before_AGA=raw_diagnostic, after_AGA=recovered_diagnostic)
        save_json(out / "parameter_diagnostics.json", diagnostics)
        if name != "V_full":
            record(name, "standalone", model, site)
            combined = merge(anchor, model, second, config["coefficient"])
            record(name, "direct", combined, site)
            del combined
            record(name, "AGA_standalone", attacked, site)
            combined = merge(anchor, attacked, second, config["coefficient"])
            record(name, "AGA", combined, site)
            del combined
        del attacked
    attack_loader = make_loader(training["MNIST"], indices["MNIST"]["attack"], batch_size=32)
    for name, model in [("control", original), ("FC_development_selected", fc_model)]:
        moment = collect_moments(model, attack_loader, [fc["site"]])[fc["site"]]
        attack_basis, stats = fit_basis(moment, parent_config["retained_energy"])
        save_json(out / f"{name}_attacker_subspace.json", stats)
        for mode in ["projection", "completion"]:
            attacked = clean_complement(model, anchor, fc["site"], attack_basis, mode)
            record(name, f"{mode}_standalone", attacked, fc["site"])
            combined = merge(anchor, attacked, second, config["coefficient"])
            record(name, mode, combined, fc["site"])
            del attacked, combined
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            total_seconds=time.perf_counter() - start,
            config_sha256=hashlib.sha256(config_bytes).hexdigest(),
            experiment_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            author_sha256=author_sha,
            parent_config_sha256=parent_meta["config_sha256"],
            parent_results_sha256=hashlib.sha256(parent_rows_bytes).hexdigest(),
            reused_measurements=sum(r["origin"] == "parent_measurement" for r in rows),
            new_evaluations=sum(r["origin"] == "new_evaluation" for r in rows),
            training="none",
            calibration="parent_frozen_real_training_indices",
            test_selection="none; FC selected from parent development screen",
            diagnostic_scope="local linear outputs on fixed inputs; causal contrasts use full tests",
        ),
    )
    print("DIAGNOSTICS_COMPLETE", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/subspace_diagnostics.json")
    parser.add_argument("--output", default="outputs/2026-10-09/post_meeting/subspace_diagnostics")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
