"""Fixed-release chain audit: Attention change -> MLP/LN -> class margin.

Only historical target training inputs. No new release, fitting, full test,
AGA variant, gradient step, direction/amplitude search or partner input data.
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from eight_task_benchmark import digest, merge_all
from joint_posthoc import capture_blocks
from local_erasure import rescale
from operator_attention_ablation import load_inputs
from path_subspace import context, read
from query_partner_cost import runtime
from subspace_probe import save_json
from value_conditioned_routing import responses, weighted_sum


def ln_derivative(x, d, norm):
    c = x - x.mean(-1, keepdim=True)
    s = (c.square().mean(-1, keepdim=True) + norm.eps).sqrt()
    return norm.weight.double().cuda() * (
        (d - d.mean(-1, keepdim=True)) / s - c * (c * d).mean(-1, keepdim=True) / s.pow(3)
    )


def arrays(model):
    b = model.model.visual.transformer.resblocks[-1]
    return dict(
        w=b.mlp.c_fc.weight.detach().cpu().double().numpy(),
        b=b.mlp.c_fc.bias.detach().cpu().double().numpy(),
        v=b.mlp.c_proj.weight.detach().cpu().double().numpy(),
        d=b.mlp.c_proj.bias.detach().cpu().double().numpy(),
        gamma=b.ln_2.weight.detach().cpu().double().numpy(),
        beta=b.ln_2.bias.detach().cpu().double().numpy(),
        eps=b.ln_2.eps,
        post_gamma=model.model.visual.ln_post.weight.detach().cpu().double().numpy(),
        post_beta=model.model.visual.ln_post.bias.detach().cpu().double().numpy(),
        post_eps=model.model.visual.ln_post.eps,
        projection=model.model.visual.proj.detach().cpu().double().numpy(),
    )


def numpy_chain(x, p):
    def ln(x, g, b, eps):
        c = x - x.mean(-1, keepdims=True)
        return c / np.sqrt((c * c).mean(-1, keepdims=True) + eps) * g + b

    z = ln(x, p["gamma"], p["beta"], p["eps"]) @ p["w"].T + p["b"]
    y = x + (z / (1 + np.exp(np.clip(-1.702 * z, -700, 700)))) @ p["v"].T + p["d"]
    f = ln(y, p["post_gamma"], p["post_beta"], p["post_eps"]) @ p["projection"]
    return f / np.linalg.norm(f, axis=-1, keepdims=True)


@torch.no_grad()
def audit(name, model, changed, images, head, scale, geometry):
    before = capture_blocks(model, images, [11], details=True)[11]
    after = capture_blocks(changed, images, [11], details=True)[11]
    n = len(images)
    x = before["attention_input"].reshape(n, 50, 768).double().cuda()
    assert torch.equal(before["attention_input"], after["attention_input"])
    r = before["post_attention"].reshape(n, 50, 768)[:, 0].double().cuda()
    delta = (
        (after["post_attention"] - before["post_attention"])
        .reshape(n, 50, 768)[:, 0]
        .double()
        .cuda()
    )
    block = model.model.visual.transformer.resblocks[-1]
    q, k, val, maps = responses(block, x)
    new_maps = responses(changed.model.visual.transformer.resblocks[-1], x)[3]
    h = geometry["head"]
    o = block.attn.out_proj.weight[:, h * 64 : (h + 1) * 64].double().cuda()
    expected = weighted_sum(new_maps[:, h, 0] - maps[:, h, 0], val[:, h]) @ o.T
    attention_error = float((expected - delta).abs().max())
    assert attention_error < 1e-4
    z = torch.nn.functional.layer_norm(
        r,
        (768,),
        block.ln_2.weight.double().cuda(),
        block.ln_2.bias.double().cuda(),
        block.ln_2.eps,
    )
    a = torch.nn.functional.linear(
        z, block.mlp.c_fc.weight.double().cuda(), block.mlp.c_fc.bias.double().cuda()
    )
    dz = ln_derivative(r, delta, block.ln_2)
    da = dz @ block.mlp.c_fc.weight.double().cuda().T
    s = (1.702 * a).sigmoid()
    prime = s + 1.702 * a * s * (1 - s)
    dmlp = (prime * da) @ block.mlp.c_proj.weight.double().cuda().T
    dy = delta + dmlp
    y = before["output"].reshape(n, 50, 768)[:, 0].double().cuda()
    new_y = after["output"].reshape(n, 50, 768)[:, 0].double().cuda()
    post = model.model.visual.ln_post

    def final(value):
        value = torch.nn.functional.layer_norm(
            value, (768,), post.weight.double().cuda(), post.bias.double().cuda(), post.eps
        )
        return value @ model.model.visual.proj.double().cuda()

    raw = final(y)
    new_raw = final(new_y)
    draw = ln_derivative(y, dy, post) @ model.model.visual.proj.double().cuda()
    length = raw.norm(dim=-1, keepdim=True)
    df = draw / length - raw * (raw * draw).sum(-1, keepdim=True) / length.pow(3)
    f = raw / length
    new_f = new_raw / new_raw.norm(dim=-1, keepdim=True)
    assert head.normalize and torch.count_nonzero(head.bias) == 0
    c = head.weight.double().cuda()
    scores = f @ c.T
    new_scores = new_f @ c.T
    winners = scores.argmax(-1)
    rivals = scores.clone()
    rivals.scatter_(1, winners[:, None], -torch.inf)
    rival = rivals.argmax(-1)
    contrast = c[winners] - c[rival]
    margin = (f * contrast).sum(-1)
    finite = ((new_f - f) * contrast).sum(-1)
    linear = (df * contrast).sum(-1)
    p = arrays(model)
    rn = r[:4].cpu().numpy()
    dn = delta[:4].cpu().numpy()
    eps = 1e-3
    fd = (numpy_chain(rn + eps * dn, p) - numpy_chain(rn - eps * dn, p)) / (2 * eps)
    fd_error = float(np.linalg.norm(fd - df[:4].cpu().numpy()) / np.linalg.norm(fd))
    assert fd_error < 1e-4
    relative_margin = finite / margin
    row = dict(
        context=name,
        n=n,
        release_scale=scale,
        numpy_chain_derivative_relative_error=fd_error,
        finite_attention_delta_max_kernel_error=attention_error,
        map_relative_rms=float((new_maps[:, h, 0] - maps[:, h, 0]).norm() / maps[:, h, 0].norm()),
        local_head_output_relative_rms=float(
            delta.norm() / (weighted_sum(maps[:, h, 0], val[:, h]) @ o.T).norm()
        ),
        mlp_delta_to_attention_norm_ratio=float(dmlp.norm() / delta.norm()),
        mlp_attention_delta_cosine=float((dmlp * delta).sum() / (dmlp.norm() * delta.norm())),
        normalized_feature_delta_rms=float((new_f - f).square().sum(-1).mean().sqrt()),
        feature_first_order_relative_error=float((df - (new_f - f)).norm() / (new_f - f).norm()),
        mean_original_margin=float(margin.mean()),
        mean_fixed_rival_margin_change=float(finite.mean()),
        median_absolute_relative_margin_change=float(relative_margin.abs().median()),
        fixed_rival_crossings=int((margin + finite <= 0).sum()),
        teacher_prediction_changes=int(winners.ne(new_scores.argmax(-1)).sum()),
    )
    raw = dict(
        r=r.cpu(),
        delta=delta.cpu(),
        df=df.cpu(),
        f=f.cpu(),
        new_f=new_f.cpu(),
        margin=margin.cpu(),
        finite=finite.cpu(),
        linear=linear.cpu(),
        numpy_fd=torch.from_numpy(fd),
        numpy_arrays={
            k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in p.items()
        },
    )
    return row, raw


@torch.no_grad()
def run(args):
    runtime()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    cfg = read(args.config)
    parent = Path(cfg["parent"])
    assert read(parent / "validation.json")["status"] == "passed"
    meta = read(parent / "metadata.json")
    for f, pin in meta["raw_sha256"].items():
        assert digest(parent / "artifacts" / f) == pin
    construction = read(parent / "protocol.json")
    _pcfg, bundle, _old, _edits = load_inputs(construction)
    original, anchor = bundle[3]["original"], bundle[3]["anchor"]
    head = bundle[4]
    release = copy.deepcopy(original).cpu().eval()
    edits = torch.load(parent / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    for k, v in edits.items():
        release.state_dict()[k].copy_(v)
    assert digest(parent / "artifacts/release_edits.pt") == cfg["release_sha256"]
    root, _load, fusion, models = context(read(cfg["evaluation_config"]), all_tasks=True)
    contexts = dict(
        native=(original, release, 1),
        beta=(rescale(original, anchor, 0.2), rescale(release, anchor, 0.2), 0.2),
        fusion=(
            merge_all(models, root, 0.2),
            merge_all(dict(models, Cars=release), root, 0.2),
            0.2,
        ),
    )
    data = bundle[5]
    indices = dict(
        calibration=bundle[6]["calibration"][:64], confirmation=bundle[6]["confirmation"][:64]
    )
    assert not set(indices["calibration"]) & set(indices["confirmation"])
    geometry = torch.load(parent / "artifacts/geometry.pt", map_location="cpu", weights_only=True)
    rows, raw = [], {}
    for split, ids in indices.items():
        images = torch.stack([data.train_dataset[i][0] for i in ids])
        for name, (before, after, scale) in contexts.items():
            key = name + "/" + split
            row, raw[key] = audit(key, before, after, images, head, scale, geometry)
            rows.append(row)
            print(
                "ROUTING_AUDIT",
                key,
                row["mean_fixed_rival_margin_change"],
                row["teacher_prediction_changes"],
                flush=True,
            )
    torch.save(raw, out / "artifacts/chain.pt")
    save_json(out / "results.json", rows)
    save_json(out / "protocol.json", cfg)
    save_json(out / "training_indices.json", indices)
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            independent_numpy_chain_checks=6,
            same_upstream_inputs=True,
            actual_attention_compile_checks=6,
            new_releases=0,
            new_full_tests=0,
            gradient_updates=0,
            parameter_fits=0,
            historical_training_inputs=True,
            defense_success_claim=False,
        ),
    )
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_manifest=read(args.manifest)["source_manifest"],
            release_sha256=cfg["release_sha256"],
            raw_sha256={"chain.pt": digest(out / "artifacts/chain.pt")},
            configuration_sha256=digest(args.config),
        ),
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/routing_response_audit.json")
    p.add_argument("--output", default="outputs/2026-10-12/routing_response_audit")
    p.add_argument("--manifest", required=True)
    run(p.parse_args())
