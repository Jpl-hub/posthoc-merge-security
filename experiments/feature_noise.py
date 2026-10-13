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
    released, artifact, geometry = construct(original, anchor, head, training, indices, cfg)
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
        attacked_partner = author_recover(partner,anchor,author)
        for name, model in [("original",original),("released",released)]:
            attacked = author_recover(model,anchor,author)
            record(attacked,head,native,name,"AGA_standalone","MNIST")
            torch.save(attacked.state_dict(),out / f"artifacts/{name}_AGA.pt")
            fused = merge(anchor,attacked,attacked_partner,cfg["coefficient"])
            record(fused,head,native,name,"AGA","MNIST")
            record(fused,head_s,loader_s,name,"AGA","SVHN")
            del attacked, fused
        torch.save(attacked_partner.state_dict(),out / "artifacts/partner_AGA.pt")
    save_json(out / "execution.json", dict(
        completed=True, rows=len(rows), standalone_loss_pp=loss,
        fusion_run=loss <= cfg["standalone_loss_gate_pp"], new_candidates=1,
        predictions_sha256=digest(out / "artifacts/predictions.pt"),
        release_sha256=digest(out / "artifacts/release.pt"), checkpoint_sha256=hashes,
        protocol_sha256=digest(args.protocol), runner_sha256=digest(__file__),
        optimizer_steps=0, parameter_search=False, public_cleaning_run=False,
        gpu=torch.cuda.get_device_name(), source_commit=cfg["source_commit"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",default="configs/feature_noise.json")
    parser.add_argument("--output",required=True)
    parser.add_argument("--author-repo",default="external/secure-merge-attack")
    parser.add_argument("--checkpoints",default="checkpoints/ViT-B-32")
    parser.add_argument("--data",default="data")
    run(parser.parse_args())
