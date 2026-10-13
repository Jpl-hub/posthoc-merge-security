"""One frozen final-MLP input-complement noise candidate; no parameter search."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from clip_benchmark import author_recover, checkpoint_loader, digest, evaluate, merge, paired_counts
from mechanisms import load_author
from subspace_probe import make_loader, save_json, split_training
from torchvision.datasets import MNIST, SVHN


@torch.no_grad()
def capture(model, loader):
    """Actual post-QuickGELU CLS input and final residual, not preactivation PCA."""
    hidden, residual = [], []
    block = model.model.visual.transformer.resblocks[11]
    handles = [
        block.mlp.c_proj.register_forward_pre_hook(
            lambda _m, args: hidden.append(args[0][0].detach().cpu())
        ),
        model.model.visual.ln_post.register_forward_pre_hook(
            lambda _m, args: residual.append(args[0].detach().cpu())
        ),
    ]
    try:
        model.cuda().eval()
        for images, _ in loader:
            model(images.cuda(non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return torch.cat(hidden).double(), torch.cat(residual).double()


@torch.no_grad()
def construct(original, anchor, head, training, indices, cfg):
    fit = make_loader(training, indices["fit"], 64)
    scale = make_loader(training, indices["scale"], 64)
    z, h = capture(original, fit)
    z_scale, h_scale = capture(original, scale)
    # This proxy uses only the target and the public anchor, never a partner.
    proxy = merge(anchor, original, anchor, cfg["proxy_beta"])
    z_proxy, _ = capture(proxy, fit)
    del proxy
    mean = z.mean(0)
    centered = z - mean
    _, singular, vh = torch.linalg.svd(centered.cuda(), full_matrices=False)
    fractions = singular.square().cumsum(0) / singular.square().sum()
    rank = int(torch.searchsorted(fractions, cfg["retained_energy"]).item()) + 1
    basis = vh[:rank].T.cpu()
    shift = z_proxy.mean(0) - mean
    direction = shift - basis @ (basis.T @ shift)
    assert direction.norm() > 1e-8, "No proxy mean shift outside the target space"
    direction /= direction.norm()
    span_error = float((basis.T @ direction).norm())
    assert span_error < 1e-8

    visual = original.model.visual
    assert head.normalize and torch.count_nonzero(head.bias) == 0
    contrast = head.weight.double() - head.weight.double().mean(0)
    readout = visual.ln_post.weight.double()[:, None] * (visual.proj.double() @ contrast.T)
    readout -= readout.mean(0)
    generator = torch.Generator().manual_seed(cfg["seed"])
    random = torch.randn(len(contrast), generator=generator, dtype=torch.float64)
    output = readout @ random
    output /= output.norm()
    scores = {name: (values - mean) @ direction for name, values in
              [("fit", z), ("scale", z_scale), ("proxy_fit", z_proxy)]}
    native_rms = max(float(scores[k].square().mean().sqrt()) for k in ["fit", "scale"])
    assert native_rms > 1e-12
    residual_rms = min(float(values.square().sum(-1).mean().sqrt())
                       for values in [h, h_scale])
    weight = visual.transformer.resblocks[11].mlp.c_proj.weight.double()
    amplitude = min(cfg["native_residual_fraction"] * residual_rms / native_rms,
                    cfg["weight_relative_cap"] * float(weight.norm()))
    delta = amplitude * output[:, None] @ direction[None, :]
    bias_delta = -(delta @ mean)
    released = copy.deepcopy(original).cpu().eval()
    layer = released.model.visual.transformer.resblocks[11].mlp.c_proj
    layer.weight.copy_((weight + delta).float())
    layer.bias.copy_((layer.bias.double() + bias_delta).float())
    actual = layer.weight.double() - weight
    actual_span_error = float((actual @ basis).norm() / actual.norm())
    assert actual_span_error < 1e-5
    changed = {k for k, value in released.state_dict().items()
               if not torch.equal(value, original.state_dict()[k])}
    assert changed == {"model.visual.transformer.resblocks.11.mlp.c_proj." + p
                       for p in ["weight", "bias"]}
    artifact = dict(mean=mean, basis=basis, direction=direction, output=output,
                    amplitude=amplitude, delta=delta, bias_delta=bias_delta,
                    released_weight=layer.weight.clone(), released_bias=layer.bias.clone(),
                    fit_hidden=z, scale_hidden=z_scale, proxy_hidden=z_proxy,
                    fit_residual=h, scale_residual=h_scale)
    geometry = dict(rank=rank, input_dimension=3072, output_dimension=768,
                    explained_energy=float(fractions[rank - 1]),
                    amplitude=amplitude, span_error=span_error,
                    actual_fp32_span_error=actual_span_error,
                    relative_weight_change=float(actual.norm() / weight.norm()),
                    predicted_native_residual_fraction=amplitude * native_rms / residual_rms,
                    projected_mean_shift=float(shift @ direction),
                    response_rms={k:float(v.square().mean().sqrt()) for k,v in scores.items()},
                    partner_used=False, labels_used=False, optimizer_steps=0)
    return released, artifact, geometry


@torch.no_grad()
def construct_class_noise(original, anchor, head, training, indices, cfg):
    """Minimum-norm class confusion, restricted to the frozen target complement."""
    source = Path(cfg["basis_source"])
    parent = json.loads((source / "execution.json").read_text())
    assert json.loads((source / "independent_validation.json").read_text())["status"] == "passed"
    assert digest(source / "artifacts/release.pt") == parent["release_sha256"] == cfg["basis_sha256"]
    assert indices == json.loads((source / "training_indices.json").read_text())
    saved = torch.load(source / "artifacts/release.pt", map_location="cpu")
    basis, mean = saved["basis"], saved["mean"]
    proxy = merge(anchor, original, anchor, cfg["proxy_beta"])
    z_proxy, h_proxy = capture(proxy, make_loader(training, indices["fit"], 64))
    assert torch.allclose(z_proxy, saved["proxy_hidden"], atol=1e-6, rtol=1e-5)
    visual = proxy.model.visual
    assert head.normalize and torch.count_nonzero(head.bias) == 0
    contrast = head.weight.double() - head.weight.double().mean(0)
    readout = visual.ln_post.weight.double()[:, None] * (visual.proj.double() @ contrast.T)
    readout -= readout.mean(0)
    values = torch.linalg.svdvals(readout)
    assert int((values > values[0] * 1e-10).sum()) == len(contrast) - 1
    weight_proxy = visual.transformer.resblocks[11].mlp.c_proj.weight.double()
    component = weight_proxy - (weight_proxy @ basis) @ basis.T
    mapping = torch.roll(torch.eye(len(contrast), dtype=torch.float64), shifts=1, dims=0)
    desired = (mapping - torch.eye(len(contrast), dtype=torch.float64)) @ (readout.T @ component)
    operator = torch.linalg.pinv(readout.T, atol=0.0, rtol=1e-10) @ desired
    assert float((readout.T @ operator - desired).norm() / desired.norm()) < 1e-9
    assert float((operator @ basis).norm() / operator.norm()) < 1e-8
    responses = {name: (saved[key] - mean) @ operator.T for name, key in
                 [("fit", "fit_hidden"), ("scale", "scale_hidden"), ("proxy_fit", "proxy_hidden")]}
    rms = {name: float(value.square().sum(-1).mean().sqrt()) for name, value in responses.items()}
    native_rms = max(rms["fit"], rms["scale"])
    assert native_rms > 1e-12
    residual_rms = min(float(saved[key].square().sum(-1).mean().sqrt())
                       for key in ["fit_residual", "scale_residual"])
    weight = original.model.visual.transformer.resblocks[11].mlp.c_proj.weight.double()
    amplitude = min(cfg["native_residual_fraction"] * residual_rms / native_rms,
                    cfg["weight_relative_cap"] * float(weight.norm()) / float(operator.norm()))
    delta = amplitude * operator
    bias_delta = -(delta @ mean)
    released = copy.deepcopy(original).cpu().eval()
    layer = released.model.visual.transformer.resblocks[11].mlp.c_proj
    layer.weight.copy_((weight + delta).float())
    layer.bias.copy_((layer.bias.double() + bias_delta).float())
    actual = layer.weight.double() - weight
    actual_span_error = float((actual @ basis).norm() / actual.norm())
    assert actual_span_error < 1e-5
    changed = {key for key, value in released.state_dict().items()
               if not torch.equal(value, original.state_dict()[key])}
    assert changed == {"model.visual.transformer.resblocks.11.mlp.c_proj." + part
                       for part in ["weight", "bias"]}
    artifact = dict(structure="class_confusion", basis_source=str(source), basis_sha256=cfg["basis_sha256"],
                    mean=mean, amplitude=amplitude, operator=operator, delta=delta,
                    bias_delta=bias_delta, released_weight=layer.weight.clone(),
                    released_bias=layer.bias.clone(), proxy_raw=h_proxy,
                    readout=readout, proxy_weight=weight_proxy, class_mapping=mapping)
    geometry = dict(structure="class_confusion", task_input_rank=basis.shape[1], noise_rank_upper_bound=9,
                    amplitude=amplitude, relative_weight_change=float(actual.norm() / weight.norm()),
                    actual_fp32_span_error=actual_span_error,
                    class_equation_error=float((readout.T @ operator - desired).norm() / desired.norm()),
                    native_residual_fraction=amplitude * native_rms / residual_rms,
                    proxy_residual_rms=cfg["proxy_beta"] * amplitude * rms["proxy_fit"],
                    operator_response_rms=rms, response_units="CLS vector Euclidean RMS",
                    same_input_space=True, same_response_budget=True, same_weight_cap=True,
                    partner_used=False, labels_used_in_construction=False,
                    optimizer_steps=0, response_fits=0)
    # Diagnose actual finite normalization after the recipe and amplitude are fixed.
    def scores(model, raw):
        v = model.model.visual
        centered = raw - raw.mean(-1, keepdim=True)
        scale = (centered.square().mean(-1, keepdim=True) + v.ln_post.eps).sqrt()
        normalized = centered / scale * v.ln_post.weight.double() + v.ln_post.bias.double()
        feature = normalized @ v.proj.double()
        feature /= feature.norm(dim=-1, keepdim=True)
        return feature @ head.weight.double().T + head.bias.double()
    diagnostics = []
    for name, model, raw, response, sample_ids, factor in [
        ("native_fit", original, saved["fit_residual"], responses["fit"], indices["fit"], 1),
        ("native_scale", original, saved["scale_residual"], responses["scale"], indices["scale"], 1),
        ("proxy_fit", proxy, h_proxy, responses["proxy_fit"], indices["fit"], cfg["proxy_beta"]),
    ]:
        labels = training.targets[sample_ids]
        before, after = scores(model, raw), scores(model, raw + factor * amplitude * response)
        diagnostics.append(dict(context=name, n=len(labels),
            original_correct=int(before.argmax(-1).eq(labels).sum()),
            changed_correct=int(after.argmax(-1).eq(labels).sum()),
            selected_from_accuracy=False))
    artifact["proxy_diagnostics"] = diagnostics
    geometry["proxy_diagnostics"] = diagnostics
    return released, artifact, geometry


@torch.no_grad()
def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    cfg = json.loads(Path(args.protocol).read_text())
    save_json(out / "protocol.json", cfg)
    for name, pin in cfg["source_sha256"].items():
        assert digest(name) == pin, name
    assert digest(__file__) == cfg["runner_sha256"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path(args.author_repo))
    root = Path(args.checkpoints)
    files = json.loads(Path("configs/checkpoints.json").read_text())["files"]
    hashes = {}

    def checked(name):
        hashes[name] = digest(root / name)
        assert hashes[name] == next(p["sha256"] for p in files
                                    if p["path"] == "ViT-B-32/" + name)
        return load(str(root / name)).cpu().eval()

    anchor = checked("zeroshot.pt")
    original = checked("MNIST/finetuned.pt")
    head = checked("head_MNIST.pt")
    training = MNIST(args.data, train=True, transform=original.val_preprocess)
    indices = split_training(training, cfg["calibration_sizes"], cfg["seed"])
    save_json(out / "training_indices.json", indices)
    builder = construct_class_noise if cfg.get("structure") == "class_confusion" else construct
    released, artifact, geometry = builder(original, anchor, head, training, indices, cfg)
    torch.save(artifact, out / "artifacts/release.pt")
    save_json(out / "geometry.json", geometry)
    save_json(out / "release_freeze.json", dict(
        release_sha256=digest(out / "artifacts/release.pt"), checkpoint_sha256=hashes.copy(),
        source_sha256=cfg["source_sha256"], runner_sha256=cfg["runner_sha256"],
        protocol_sha256=digest(args.protocol), partner_loaded=False,
        artifact_type="Frozen geometry and two deployed parameter tensors; not a full checkpoint"))
    print("RELEASE_FROZEN", json.dumps(geometry), flush=True)
    rows, raw = [], {}

    def record(model, classifier_head, loader, preset, stage, task):
        batches = []
        metrics = evaluate(model, classifier_head, loader, "cuda",
                           prediction_sink=lambda p,y:batches.append((p,y)))
        prediction, target = [torch.cat([b[i] for b in batches]) for i in [0,1]]
        key = "/".join([preset,stage,task])
        raw[key] = dict(prediction=prediction, target=target)
        row = dict(preset=preset,stage=stage,dataset=task,**metrics)
        if preset == "released":
            base = raw["/".join(["original",stage,task])]
            assert torch.equal(base["target"], target)
            row["paired"] = paired_counts(base["prediction"].eq(target),prediction.eq(target))
            row["changed_predictions"] = int(prediction.ne(base["prediction"]).sum())
        rows.append(row)
        torch.save(raw, out / "artifacts/predictions.pt")
        save_json(out / "results.json", rows)
        print("FEATURE_NOISE_RESULT", json.dumps(row), flush=True)
        return metrics

    confirm = make_loader(training, indices["confirmation"])
    native = make_loader(MNIST(args.data, train=False, transform=original.val_preprocess))
    for name, model in [("original",original),("released",released)]:
        record(model,head,confirm,name,"confirmation","MNIST")
    native_metrics = {name:record(model,head,native,name,"standalone","MNIST")
                      for name,model in [("original",original),("released",released)]}
    loss = 100 * (native_metrics["original"]["top1"] - native_metrics["released"]["top1"])
    if loss <= cfg["standalone_loss_gate_pp"]:
        partner = checked("SVHN/finetuned.pt")
        head_s = checked("head_SVHN.pt")
        test_s = SVHN(args.data, split="test", transform=original.val_preprocess)
        assert len(test_s) == 26032
        loader_s = make_loader(test_s)
        record(partner,head_s,loader_s,"original","partner_standalone","SVHN")
        for name, model in [("original",original),("released",released)]:
            fused = merge(anchor,model,partner,cfg["coefficient"])
            record(fused,head,native,name,"direct","MNIST")
            record(fused,head_s,loader_s,name,"direct","SVHN")
            del fused
        author, author_sha = load_author(Path(args.author_repo) / "utils_attack_B32.py",True)
        assert author_sha == json.loads(Path("configs/sources.json").read_text())["AGA"]["sha256"]
        reference = Path(cfg["AGA_reference"]) if cfg.get("AGA_reference") else None
        reference_pins = None
        if reference:
            assert json.loads((reference / "independent_validation.json").read_text())["status"] == "passed"
            old_hashes = json.loads((reference / "execution.json").read_text())["checkpoint_sha256"]
            assert old_hashes == hashes
            reference_pins = json.loads((reference / "AGA_states.json").read_text())

        def recovered(model, name):
            if reference and name != "released":
                path = reference / f"artifacts/{name}_AGA.pt"
                assert digest(path) == reference_pins[name]
                result = copy.deepcopy(model).cpu().eval()
                result.load_state_dict(torch.load(path, map_location="cpu"), strict=True)
                return result
            return author_recover(model, anchor, author)

        attacked_partner = recovered(partner, "partner")
        for name, model in [("original",original),("released",released)]:
            attacked = recovered(model, name)
            record(attacked,head,native,name,"AGA_standalone","MNIST")
            if reference and name == "original":
                (out / f"artifacts/{name}_AGA.pt").hardlink_to(reference / f"artifacts/{name}_AGA.pt")
            else:
                torch.save(attacked.state_dict(),out / f"artifacts/{name}_AGA.pt")
            fused = merge(anchor,attacked,attacked_partner,cfg["coefficient"])
            record(fused,head,native,name,"AGA","MNIST")
            record(fused,head_s,loader_s,name,"AGA","SVHN")
            del attacked, fused
        if reference:
            (out / "artifacts/partner_AGA.pt").hardlink_to(reference / "artifacts/partner_AGA.pt")
        else:
            torch.save(attacked_partner.state_dict(),out / "artifacts/partner_AGA.pt")
    save_json(out / "execution.json", dict(
        completed=True, rows=len(rows), standalone_loss_pp=loss,
        fusion_run=loss <= cfg["standalone_loss_gate_pp"], new_candidates=1,
        predictions_sha256=digest(out / "artifacts/predictions.pt"),
        release_sha256=digest(out / "artifacts/release.pt"), checkpoint_sha256=hashes,
        protocol_sha256=digest(args.protocol), runner_sha256=digest(__file__),
        optimizer_steps=0, parameter_search=False, public_cleaning_run=False,
        gpu=torch.cuda.get_device_name(), source_commit=cfg["source_commit"],
        AGA_reference=str(reference) if loss <= cfg["standalone_loss_gate_pp"] and reference else None,
        reused_AGA_models=["original", "partner"] if loss <= cfg["standalone_loss_gate_pp"] and reference else []))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",default="configs/feature_noise.json")
    parser.add_argument("--output",required=True)
    parser.add_argument("--author-repo",default="external/secure-merge-attack")
    parser.add_argument("--checkpoints",default="checkpoints/ViT-B-32")
    parser.add_argument("--data",default="data")
    run(parser.parse_args())
