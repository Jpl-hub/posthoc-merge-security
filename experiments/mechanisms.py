"""P0/P1/P3 mechanisms. No training; synthetic task changes are explicitly labelled.

Run on rented GPU. Author functions are parsed from a pinned pristine source file,
so unrelated JAX imports and hardcoded paths do not modify their implementation.
"""

import argparse
import ast
import contextlib
import csv
import hashlib
import io
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def load_author(path):
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        "solve_permutation",
        "solve_linear_transformation",
        "attack_attention_headwise_precise",
    }
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    from typing import Dict

    env = dict(torch=torch, np=np, scipy=scipy, Dict=Dict)
    exec(compile(tree, str(path), "exec"), env)
    return env, hashlib.sha256(source.encode()).hexdigest()


def relative(a, b):
    return float((a.double() - b.double()).norm() / b.double().norm().clamp_min(1e-30))


def state_error(a, b):
    num = sum((a[k].double() - b[k].double()).square().sum() for k in b)
    den = sum(b[k].double().square().sum() for k in b)
    return float((num / den.clamp_min(1e-30)).sqrt())


def clone(s):
    return {k: v.clone() for k, v in s.items()}


def perturbed(s, magnitude):
    out = {}
    for k, v in s.items():
        noise = torch.randn_like(v)
        out[k] = v + magnitude * v.norm() * noise / noise.norm().clamp_min(1e-30)
    return out


def transform(d, cond=1.0, dtype=torch.float32, device="cuda"):
    u = torch.linalg.qr(torch.randn(d, d, device=device, dtype=dtype)).Q
    v = torch.linalg.qr(torch.randn(d, d, device=device, dtype=dtype)).Q
    sig = torch.logspace(0, math.log10(cond), d, dtype=dtype, device=device)
    return u @ torch.diag(sig) @ v.T


PREFIX = "model.visual.transformer.resblocks.0.attn."


def attention(s, x, heads):
    d = x.shape[-1]
    q, k, v = F.linear(x, s[PREFIX + "in_proj_weight"], s[PREFIX + "in_proj_bias"]).chunk(3, -1)

    def split(t):
        return t.reshape(t.shape[0], t.shape[1], heads, d // heads).transpose(1, 2)

    q, k, v = map(split, (q, k, v))
    scores = q @ k.transpose(-1, -2) / math.sqrt(d // heads)
    y = (scores.softmax(-1) @ v).transpose(1, 2).reshape_as(x)
    return F.linear(y, s[PREFIX + "out_proj.weight"], s[PREFIX + "out_proj.bias"]), scores


def protect_attention(s, heads, cond=1.0):
    out = clone(s)
    d = s[PREFIX + "in_proj_weight"].shape[1]
    dh = d // heads
    for h in range(heads):
        a = transform(dh, cond, dtype=s[PREFIX + "in_proj_weight"].dtype)
        b = transform(dh, cond, dtype=s[PREFIX + "in_proj_weight"].dtype)
        mats = (a.T, torch.linalg.inv(a), b.T)
        for block, m in enumerate(mats):
            idx = slice(block * d + h * dh, block * d + (h + 1) * dh)
            out[PREFIX + "in_proj_weight"][idx] = m @ s[PREFIX + "in_proj_weight"][idx]
            out[PREFIX + "in_proj_bias"][idx] = m @ s[PREFIX + "in_proj_bias"][idx]
        idx = slice(h * dh, (h + 1) * dh)
        out[PREFIX + "out_proj.weight"][:, idx] = (
            s[PREFIX + "out_proj.weight"][:, idx] @ torch.linalg.inv(b).T
        )
    return out


def permute_heads(s, p):
    out = clone(s)
    heads = len(p)
    d = s[PREFIX + "in_proj_weight"].shape[1]
    dh = d // heads
    idx = (p[:, None] * dh + torch.arange(dh, device=p.device)).flatten()
    all_idx = torch.cat([idx + i * d for i in range(3)])
    out[PREFIX + "in_proj_weight"] = s[PREFIX + "in_proj_weight"][all_idx]
    out[PREFIX + "in_proj_bias"] = s[PREFIX + "in_proj_bias"][all_idx]
    out[PREFIX + "out_proj.weight"] = s[PREFIX + "out_proj.weight"][:, idx]
    return out


def head_match(s, anchor, heads):
    # Each block's column space is invariant to within-head invertible bases.
    d = s[PREFIX + "in_proj_weight"].shape[1]
    dh = d // heads

    def bases(state, h):
        w = state[PREFIX + "in_proj_weight"]
        idx = slice(h * dh, (h + 1) * dh)
        blocks = [w[i * d : (i + 1) * d][idx].T for i in range(3)]
        blocks.append(state[PREFIX + "out_proj.weight"][:, idx])
        return [torch.linalg.qr(x.double()).Q for x in blocks]

    a = [bases(s, h) for h in range(heads)]
    b = [bases(anchor, h) for h in range(heads)]
    cost = np.array(
        [[-sum(float((u.T @ v).square().sum()) / dh for u, v in zip(x, y)) for y in b] for x in a]
    )
    rows, cols = linear_sum_assignment(cost)
    restore = np.zeros(heads, dtype=np.int64)
    restore[cols] = rows
    return torch.tensor(restore, device="cuda"), cost


def run(args):
    outdir = Path(args.output)
    outdir.mkdir(parents=True, exist_ok=True)
    author, source_hash = load_author(args.author)
    rows = []
    start = time.perf_counter()

    def add(stage, experiment, seed, **kw):
        rows.append(
            dict(
                stage=stage,
                experiment=experiment,
                seed=seed,
                scope="synthetic_mechanism",
                **kw,
            )
        )
        print(json.dumps(rows[-1]), flush=True)

    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.cuda.reset_peak_memory_stats()
    for seed in args.seeds:
        torch.manual_seed(seed)
        np.random.seed(seed)
        # P0 A: frozen random MLP + synthetic changes; actual GELU nonlinear block.
        d, hidden = 48, 96
        w1 = torch.randn(hidden, d, device="cuda") / math.sqrt(d)
        w2 = torch.randn(d, hidden, device="cuda") / math.sqrt(hidden)
        b1 = torch.randn(hidden, device="cuda") * 0.1
        b2 = torch.randn(d, device="cuda") * 0.1
        x = torch.randn(4, 11, d, device="cuda")
        p = torch.randperm(hidden, device="cuda")
        for mag in [0.0, 0.001, 0.01, 0.1, 1.0]:
            s = perturbed(dict(w1=w1, w2=w2, b1=b1, b2=b2), mag)
            sp = dict(w1=s["w1"][p], w2=s["w2"][:, p], b1=s["b1"][p], b2=s["b2"])

            def mlp(t):
                return F.linear(F.gelu(F.linear(x, t["w1"], t["b1"])), t["w2"], t["b2"])

            restore = author["solve_permutation"](sp["w1"], w1)
            sr = dict(
                w1=sp["w1"][restore],
                w2=sp["w2"][:, restore],
                b1=sp["b1"][restore],
                b2=sp["b2"],
            )
            add(
                "P0",
                "mlp_permutation",
                seed,
                magnitude=mag,
                permutation_accuracy=float(
                    (p[restore] == torch.arange(hidden, device="cuda")).float().mean()
                ),
                protected_weight_error=state_error(sp, s),
                recovered_weight_error=state_error(sr, s),
                protected_output_error=relative(mlp(sp), mlp(s)),
                recovered_output_error=relative(mlp(sr), mlp(s)),
            )
        # P0 B: attention pairing with nonzero biases, 12 heads (author B32 convention).
        d, heads = 96, 12
        base = {
            PREFIX + "in_proj_weight": torch.randn(3 * d, d, device="cuda") / math.sqrt(d),
            PREFIX + "in_proj_bias": torch.randn(3 * d, device="cuda") * 0.1,
            PREFIX + "out_proj.weight": torch.randn(d, d, device="cuda") / math.sqrt(d),
            PREFIX + "out_proj.bias": torch.randn(d, device="cuda") * 0.1,
        }
        x = torch.randn(4, 11, d, device="cuda")
        for mag in [0.0, 0.001, 0.01, 0.1, 1.0]:
            ft = perturbed(base, mag)
            protected = protect_attention(ft, heads)
            with contextlib.redirect_stdout(io.StringIO()):
                recovered = author["attack_attention_headwise_precise"](
                    protected, base, 0, heads, "cuda"
                )
            y, sc = attention(ft, x, heads)
            yp, sp = attention(protected, x, heads)
            yr, _ = attention(recovered, x, heads)
            add(
                "P0",
                "attention_qk_vo",
                seed,
                magnitude=mag,
                protected_output_error=relative(yp, y),
                protected_score_error=relative(sp, sc),
                protected_weight_error=state_error(protected, ft),
                recovered_weight_error=state_error(recovered, ft),
                recovered_output_error=relative(yr, y),
            )
        # P3: head permutation alone and combined orthogonal bases; adaptive attack.
        ft = perturbed(base, 0.01)
        hp = torch.randperm(heads, device="cuda")
        hp = (
            torch.roll(torch.arange(heads, device="cuda"), 1)
            if (hp == torch.arange(heads, device="cuda")).any()
            else hp
        )
        for variant in ["heads_only", "heads_plus_orthogonal"]:
            protected = permute_heads(
                ft if variant == "heads_only" else protect_attention(ft, heads), hp
            )
            restore, cost = head_match(protected, base, heads)
            aligned = permute_heads(protected, restore)
            with contextlib.redirect_stdout(io.StringIO()):
                recovered = author["attack_attention_headwise_precise"](
                    protected, base, 0, heads, "cuda"
                )
                adaptive = author["attack_attention_headwise_precise"](
                    aligned, base, 0, heads, "cuda"
                )
            y, _ = attention(ft, x, heads)
            yp, _ = attention(protected, x, heads)
            yr, _ = attention(recovered, x, heads)
            ya, _ = attention(adaptive, x, heads)
            add(
                "P3",
                variant,
                seed,
                magnitude=0.01,
                permutation_accuracy=float(
                    (hp[restore] == torch.arange(heads, device="cuda")).float().mean()
                ),
                protected_output_error=relative(yp, y),
                recovered_weight_error=state_error(recovered, ft),
                recovered_output_error=relative(yr, y),
                adaptive_weight_error=state_error(adaptive, ft),
                adaptive_output_error=relative(ya, y),
            )
    # P1: ideal direct LS versus unmodified author's normal-equation ridge + float32 cast.
    torch.manual_seed(args.seeds[0])
    n, d = 128, 16
    anchor = torch.randn(n, d, dtype=torch.float64)
    delta = torch.randn(n, d, dtype=torch.float64)
    delta = 0.01 * anchor.norm() * delta / delta.norm()
    ft = anchor + delta
    proj = ft @ torch.linalg.lstsq(ft, anchor, driver="gelsd").solution
    u = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64)).Q
    v = torch.linalg.qr(torch.randn(d, d, dtype=torch.float64)).Q
    for family, cond in [
        ("orthogonal", 1.0),
        ("diagonal", 100.0),
        ("dense", 100.0),
        ("dense", 1e4),
        ("dense", 1e6),
    ]:
        sig = torch.logspace(0, math.log10(cond), d, dtype=torch.float64)
        p = (
            u
            if family == "orthogonal"
            else torch.diag(sig)
            if family == "diagonal"
            else u @ torch.diag(sig) @ v.T
        )
        for scale in [1e-4, 1.0, 1e4]:
            wp = ft @ (scale * p)
            ideal = wp @ torch.linalg.lstsq(wp, anchor, driver="gelsd").solution
            t = author["solve_linear_transformation"](wp.cuda(), anchor.cuda())
            recovered = wp.cuda() @ t.double()
            # Exact ridge reference solved without float32 conversion (same lambda).
            ridge = wp @ torch.linalg.solve(
                wp.T @ wp + 1e-8 * torch.eye(d, dtype=torch.float64), wp.T @ anchor
            )
            add(
                "P1",
                "column_space_invariance",
                args.seeds[0],
                family=family,
                condition=cond,
                scale=scale,
                ideal_projection_error=relative(ideal, proj),
                author_projection_error=relative(recovered.cpu(), proj),
                ridge_projection_error=relative(ridge, proj),
                ideal_weight_error=relative(ideal, ft),
                author_weight_error=relative(recovered.cpu(), ft),
                error_bound_ratio=float((ideal - ft).norm() / delta.norm()),
            )
    # Rank-deficient extension: projector equality still holds; explicit inverse doesn't.
    rank_ft = ft.clone()
    rank_ft[:, -1] = rank_ft[:, 0] + rank_ft[:, 1]
    p = u @ torch.diag(torch.linspace(1, 10, d, dtype=torch.float64)) @ v.T
    wa = rank_ft @ p @ torch.linalg.lstsq(rank_ft @ p, anchor, driver="gelsd").solution
    reference = rank_ft @ torch.linalg.pinv(rank_ft) @ anchor
    add(
        "P1",
        "rank_deficient",
        args.seeds[0],
        rank=int(torch.linalg.matrix_rank(rank_ft)),
        ideal_projection_error=relative(wa, reference),
        error_bound_ratio=float((wa - rank_ft).norm() / (rank_ft - anchor).norm()),
    )
    # Full column space square case removes every task change in recovered matrix.
    sq = torch.randn(d, d, dtype=torch.float64)
    sq_anchor = torch.randn_like(sq)
    sq_a = sq @ u @ torch.linalg.lstsq(sq @ u, sq_anchor, driver="gelsd").solution
    add(
        "P1",
        "square_invertible",
        args.seeds[0],
        ideal_projection_error=relative(sq_a, sq_anchor),
    )
    # Check cosine vs unnormalized Euclidean theorem: rows almost parallel, unequal norms.
    a = torch.tensor([[1.0, 0.0], [10.0, 0.01]], dtype=torch.float64)
    f = a.clone()
    f[0, 1] = 0.002
    f[1, 1] = -0.01
    idx = author["solve_permutation"](f, a)
    distance = (f[:, None, :] - a[None, :, :]).square().sum(-1).numpy()
    ri, ci = linear_sum_assignment(distance)
    euclidean = np.zeros(2, dtype=np.int64)
    euclidean[ci] = ri
    margin = float((a.flip(0) - a).norm())
    tau = float((f - a).norm())
    add(
        "audit",
        "cosine_margin_counterexample",
        args.seeds[0],
        tau_norm=tau,
        euclidean_margin=margin,
        theorem_condition=bool(2 * tau < margin),
        cosine_recovery_accuracy=float((idx == torch.arange(2)).float().mean()),
        euclidean_recovery_accuracy=float(
            (torch.tensor(euclidean) == torch.arange(2)).float().mean()
        ),
    )
    # Eq.19 as printed: min(0, margin-distance) is zero inside margin, unbounded below outside.
    for dist in [0.02, 0.1]:
        z = torch.tensor(dist, requires_grad=True)
        term = torch.minimum(torch.tensor(0.0), torch.tensor(0.05) - z)
        term.backward()
        add(
            "audit",
            "arf_printed_equation",
            args.seeds[0],
            distance=dist,
            margin=0.05,
            penalty=float(term.detach()),
            distance_gradient=float(z.grad),
        )
    torch.cuda.synchronize()
    meta = dict(
        seeds=args.seeds,
        device=torch.cuda.get_device_name(),
        torch=torch.__version__,
        cuda=torch.version.cuda,
        scipy=scipy.__version__,
        numpy=np.__version__,
        author_sha256=source_hash,
        elapsed_seconds=time.perf_counter() - start,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        rows=len(rows),
        tf32=False,
        note="No training or real benchmark accuracy; synthetic parameter perturbations.",
    )
    fields = sorted(set().union(*(r.keys() for r in rows)))
    with (outdir / "mechanism_results.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    (outdir / "mechanism_results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    (outdir / "mechanism_metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("COMPLETED", json.dumps(meta), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output", default="outputs/2026-10-09")
    p.add_argument("--author", default="external/secure-merge-attack/utils_attack_B32.py")
    p.add_argument("--seeds", nargs="+", type=int, default=[20261009, 20261010, 20261011])
    run(p.parse_args())
