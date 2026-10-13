"""One target-neutral complement-noise rule, evaluated on two protected tasks."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from torchvision.datasets import DTD, MNIST, SVHN

from clip_benchmark import author_recover, checkpoint_loader, digest, evaluate, merge, paired_counts
from feature_noise import capture
from mechanisms import load_author
from subspace_probe import make_loader, save_json, split_training


def dataset(task, root, transform, training=False):
    if task == "MNIST":
        return MNIST(root, train=training, transform=transform)
    if task == "DTD":
        return DTD(root, split="train" if training else "test", partition=1, transform=transform)
    if task == "SVHN":
        return SVHN(root, split="train" if training else "test", transform=transform)
    raise ValueError(task)


@torch.no_grad()
def construct(original, anchor, head, training, indices, cfg, target):
    reference = cfg["calibration_references"].get(target)
    if reference:
        source = Path(reference["path"])
        assert json.loads((source.parent.parent / "independent_validation.json").read_text())["status"] == "passed"
        assert digest(source) == reference["sha256"]
        assert indices == json.loads((source.parent.parent / "training_indices.json").read_text())
        previous = json.loads((source.parent.parent / "protocol.json").read_text())
        assert all(previous[k] == cfg[k] for k in ["retained_energy", "proxy_beta", "seed"])
        saved = torch.load(source, map_location="cpu")
        z, h = saved["fit_hidden"], saved["fit_residual"]
        z_scale, h_scale = saved["scale_hidden"], saved["scale_residual"]
        basis, mean = saved["basis"], saved["mean"]
        calibration = dict(reference=reference)
    else:
        z, h = capture(original, make_loader(training, indices["fit"], 64))
        z_scale, h_scale = capture(original, make_loader(training, indices["scale"], 64))
        mean = z.mean(0)
        _, singular, vh = torch.linalg.svd((z - mean).cuda(), full_matrices=False)
        fraction = singular.square().cumsum(0) / singular.square().sum()
        rank = int(torch.searchsorted(fraction, cfg["retained_energy"]).item()) + 1
        basis = vh[:rank].T.cpu()
        calibration = dict(fit_hidden=z, fit_residual=h, scale_hidden=z_scale, scale_residual=h_scale)

    proxy = merge(anchor, original, anchor, cfg["proxy_beta"])
    zp, hp = capture(proxy, make_loader(training, indices["fit"], 64))
    if reference:
        assert torch.allclose(zp, saved["proxy_hidden"], atol=1e-6, rtol=1e-5)
    visual = proxy.model.visual
    assert head.normalize and torch.count_nonzero(head.bias) == 0
    contrast = head.weight.double() - head.weight.double().mean(0)
    readout = visual.ln_post.weight.double()[:, None] * (visual.proj.double() @ contrast.T)
    readout -= readout.mean(0)
    q, singular, _ = torch.linalg.svd(readout, full_matrices=False)
    q = q[:, singular > singular[0] * cfg["numerical_relative_tolerance"]]
    coefficients = (hp - hp.mean(0)) @ q
    s = coefficients.T @ coefficients / len(hp)
    values, vectors = torch.linalg.eigh(s)
    keep = values > values[-1] * cfg["numerical_relative_tolerance"]
    assert keep.any()
    output_basis = q @ vectors[:, keep]
    white = ((hp - hp.mean(0)) @ output_basis / values[keep].sqrt()) @ output_basis.T
    incoming = zp - zp.mean(0)
    incoming -= (incoming @ basis) @ basis.T
    cross = white.T @ incoming / len(zp)
    assert cross.norm() > 1e-12
    operator = -cross / cross.norm()
    assert float((operator @ basis).norm()) < 1e-8
    alignment = float((white * ((zp - mean) @ operator.T)).sum() / len(zp))
    assert abs(alignment + float(cross.norm())) < 1e-8 * float(cross.norm())

    responses = {name: (values - mean) @ operator.T for name, values in
                 [("fit", z), ("scale", z_scale), ("proxy_fit", zp)]}
    rms = {name: float(value.square().sum(-1).mean().sqrt()) for name, value in responses.items()}
    h_rms = min(float(values.square().sum(-1).mean().sqrt()) for values in [h, h_scale])
    layer = original.model.visual.transformer.resblocks[11].mlp.c_proj
    weight = layer.weight.double()
    alpha = min(cfg["native_residual_fraction"] * h_rms / max(rms["fit"], rms["scale"]),
                cfg["weight_relative_cap"] * float(weight.norm()))
    delta, shift = alpha * operator, -(alpha * operator @ mean)
    released = copy.deepcopy(original).cpu().eval()
    changed_layer = released.model.visual.transformer.resblocks[11].mlp.c_proj
    changed_layer.weight.copy_((weight + delta).float())
    changed_layer.bias.copy_((layer.bias.double() + shift).float())
    actual = changed_layer.weight.double() - weight
    span_error = float((actual @ basis).norm() / actual.norm())
    assert span_error < 1e-5
    changed = {k for k, value in released.state_dict().items()
               if not torch.equal(value, original.state_dict()[k])}
    expected = {"model.visual.transformer.resblocks.11.mlp.c_proj." + part
                for part in ["weight", "bias"]}
    assert changed == expected
    edits = {k: released.state_dict()[k].clone() for k in changed}
    calibration.update(proxy_hidden=zp, proxy_residual=hp, readout=readout,
                       output_basis=output_basis, output_variances=values[keep],
                       basis=basis, mean=mean, operator=operator, amplitude=alpha)

    def scores(model, residual):
        v = model.model.visual
        center = residual - residual.mean(-1, keepdim=True)
        scale = (center.square().mean(-1, keepdim=True) + v.ln_post.eps).sqrt()
        feature = (center / scale * v.ln_post.weight.double() + v.ln_post.bias.double()) @ v.proj.double()
        feature /= feature.norm(dim=-1, keepdim=True)
        return feature @ head.weight.double().T + head.bias.double()

    diagnostics = []
    for name, model, residual, response, ids, factor in [
        ("native_fit", original, h, responses["fit"], indices["fit"], 1),
        ("native_scale", original, h_scale, responses["scale"], indices["scale"], 1),
        ("proxy_fit", proxy, hp, responses["proxy_fit"], indices["fit"], cfg["proxy_beta"]),
    ]:
        training_labels = getattr(training, "targets", getattr(training, "_labels", getattr(training, "labels", None)))
        assert training_labels is not None
        labels = torch.tensor([int(training_labels[i]) for i in ids])
        before, after = scores(model, residual), scores(model, residual + factor * alpha * response)
        diagnostics.append(dict(context=name, n=len(ids), original_correct=int(before.argmax(1).eq(labels).sum()),
                                released_correct=int(after.argmax(1).eq(labels).sum()), selected_from_accuracy=False))
    geometry = dict(task_input_rank=basis.shape[1], classification_span_rank=q.shape[1],
                    white_output_rank=output_basis.shape[1], noise_rank_upper_bound=output_basis.shape[1],
                    cross_covariance_norm=float(cross.norm()), unit_proxy_alignment=alignment,
                    amplitude=alpha, actual_fp32_span_error=span_error,
                    relative_weight_change=float(actual.norm() / weight.norm()),
                    native_residual_fraction=alpha * max(rms["fit"], rms["scale"]) / h_rms,
                    proxy_residual_rms=cfg["proxy_beta"] * alpha * rms["proxy_fit"],
                    operator_response_rms=rms, response_units="CLS vector Euclidean RMS",
                    labels_used_in_construction=False, partner_used=False, optimizer_steps=0,
                    response_fits=0, diagnostics=diagnostics)
    return released, edits, calibration, geometry


@torch.no_grad()
def run(args):
    cfg = json.loads(Path(args.protocol).read_text())
    assert dict(target=args.target, partner=args.partner) in cfg["cases"]
    for path, pin in cfg["source_sha256"].items():
        assert digest(path) == pin, path
    assert digest(__file__) == cfg["runner_sha256"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    save_json(out / "protocol.json", cfg)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path("external/secure-merge-attack"))
    root = Path("checkpoints/ViT-B-32")
    pins = {p["path"].removeprefix("ViT-B-32/"): p["sha256"]
            for p in json.loads(Path("configs/checkpoints.json").read_text())["files"]}
    hashes = {}

    def checked(name):
        hashes[name] = digest(root / name)
        assert hashes[name] == pins[name], name
        return load(str(root / name)).cpu().eval()

    anchor = checked("zeroshot.pt")
    original = checked(f"{args.target}/finetuned.pt")
    head = checked(f"head_{args.target}.pt")
    training = dataset(args.target, "data", original.val_preprocess, True)
    n = min(cfg["maximum_calibration_group"], len(training) // 3)
    sizes = {name: n for name in ["fit", "scale", "confirmation"]}
    indices = split_training(training, sizes, cfg["seed"])
    save_json(out / "training_indices.json", indices)
    released, edits, calibration, geometry = construct(original, anchor, head, training, indices, cfg, args.target)
    torch.save(edits, out / "artifacts/release.pt")
    torch.save(calibration, out / "artifacts/calibration.pt")
    save_json(out / "geometry.json", geometry)
    save_json(out / "case.json", dict(target=args.target, partner=args.partner, calibration_sizes=sizes,
                                      target_classes=head.weight.shape[0], target_train_n=len(training),
                                      uniform_chance_percent=100 / head.weight.shape[0]))
    save_json(out / "release_freeze.json", dict(release_sha256=digest(out / "artifacts/release.pt"),
        calibration_sha256=digest(out / "artifacts/calibration.pt"), checkpoint_sha256=hashes.copy(),
        source_sha256=cfg["source_sha256"], runner_sha256=cfg["runner_sha256"],
        protocol_sha256=digest(args.protocol), partner_loaded=False, test_loaded=False))
    print("RELEASE_FROZEN", args.target, json.dumps(geometry), flush=True)
    rows, raw, recovered_edits_sha = [], {}, None

    def record(model, classifier, loader, preset, stage, task):
        sink = []
        metrics = evaluate(model, classifier, loader, "cuda", prediction_sink=lambda p, y: sink.append((p, y)))
        prediction, target = [torch.cat([batch[i] for batch in sink]) for i in [0, 1]]
        key = "/".join([preset, stage, task])
        raw[key] = dict(prediction=prediction, target=target)
        counts = [dict(class_id=k, correct=int((prediction.eq(target) & target.eq(k)).sum()),
                       n=int(target.eq(k).sum())) for k in range(classifier.weight.shape[0])]
        row = dict(preset=preset, stage=stage, dataset=task, per_class=counts, **metrics)
        if preset == "released":
            ref = raw["/".join(["original", stage, task])]
            assert torch.equal(ref["target"], target)
            row["paired"] = paired_counts(ref["prediction"].eq(target), prediction.eq(target))
            row["changed_predictions"] = int(prediction.ne(ref["prediction"]).sum())
        rows.append(row)
        torch.save(raw, out / "artifacts/predictions.pt")
        save_json(out / "results.json", rows)
        print("COVARIANCE_RESULT", args.target, json.dumps(row), flush=True)
        return metrics

    confirmation = make_loader(training, indices["confirmation"])
    native = dataset(args.target, "data", original.val_preprocess)
    assert len(native) == cfg["test_counts"][args.target]
    loader = make_loader(native)
    for name, model in [("original", original), ("released", released)]:
        record(model, head, confirmation, name, "confirmation", args.target)
    native_score = {name: record(model, head, loader, name, "standalone", args.target)
                    for name, model in [("original", original), ("released", released)]}
    loss = 100 * (native_score["original"]["top1"] - native_score["released"]["top1"])
    if loss <= cfg["standalone_loss_gate_pp"]:
        partner = checked(f"{args.partner}/finetuned.pt")
        partner_head = checked(f"head_{args.partner}.pt")
        partner_test = dataset(args.partner, "data", original.val_preprocess)
        assert len(partner_test) == cfg["test_counts"][args.partner]
        partner_loader = make_loader(partner_test)
        record(partner, partner_head, partner_loader, "original", "partner_standalone", args.partner)
        for name, model in [("original", original), ("released", released)]:
            fused = merge(anchor, model, partner, cfg["coefficient"])
            record(fused, head, loader, name, "direct", args.target)
            record(fused, partner_head, partner_loader, name, "direct", args.partner)
            del fused

        def reference(model, task):
            ref = cfg["AGA_references"][task]
            assert json.loads(Path(ref["validation"]).read_text())["status"] == "passed"
            assert digest(ref["path"]) == ref["sha256"]
            restored = copy.deepcopy(model).cpu().eval()
            restored.load_state_dict(torch.load(ref["path"], map_location="cpu"), strict=True)
            return restored

        attacked_original, attacked_partner = reference(original, args.target), reference(partner, args.partner)
        author, pin = load_author(Path("external/secure-merge-attack/utils_attack_B32.py"), True)
        assert pin == json.loads(Path("configs/sources.json").read_text())["AGA"]["sha256"]
        attacked_release = author_recover(released, anchor, author)
        base = attacked_original.state_dict()
        attack_edits = {k: v.clone() for k, v in attacked_release.state_dict().items()
                        if not torch.equal(v, base[k])}
        torch.save(attack_edits, out / "artifacts/released_AGA_edits.pt")
        recovered_edits_sha = digest(out / "artifacts/released_AGA_edits.pt")
        save_json(out / "AGA_edits.json", dict(reference=cfg["AGA_references"][args.target],
                                               changed_keys=list(attack_edits), sha256=recovered_edits_sha,
                                               method="Unchanged author AGA; save exact differences from verified original"))
        for name, attacked in [("original", attacked_original), ("released", attacked_release)]:
            record(attacked, head, loader, name, "AGA_standalone", args.target)
            fused = merge(anchor, attacked, attacked_partner, cfg["coefficient"])
            record(fused, head, loader, name, "AGA", args.target)
            record(fused, partner_head, partner_loader, name, "AGA", args.partner)
            del fused
    save_json(out / "execution.json", dict(completed=True, rows=len(rows), standalone_loss_pp=loss,
        fusion_run=loss <= cfg["standalone_loss_gate_pp"], new_candidates=1,
        predictions_sha256=digest(out / "artifacts/predictions.pt"), release_sha256=digest(out / "artifacts/release.pt"),
        calibration_sha256=digest(out / "artifacts/calibration.pt"), checkpoint_sha256=hashes,
        protocol_sha256=digest(args.protocol), runner_sha256=digest(__file__),
        AGA_edits_sha256=recovered_edits_sha, optimizer_steps=0, parameter_search=False,
        public_cleaning_run=False, gpu=torch.cuda.get_device_name(), source_commit=cfg["source_commit"]))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/covariance_noise.json")
    parser.add_argument("--target", required=True)
    parser.add_argument("--partner", required=True)
    parser.add_argument("--output", required=True)
    run(parser.parse_args())
