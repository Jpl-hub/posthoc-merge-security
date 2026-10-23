"""Task SVD probes, finite permutation costs and nonlinear compensation."""

import json
import time
from pathlib import Path

import numpy as np
import torch

from .defenses import residual_permutation
from .io import digest, save_json
from .metrics import logical_parameter_distance
from .models import checked_models


def mlp_ball_moments(gram, bias, dimension, radius, radial_order=64, angular_order=128):
    """QuickGELU feature moments on a specified uniform ball, including a constant.

    gram contains inner products of the input rows in that ball's coordinates.
    This is a numerical integral for an explicit probe distribution, not the
    model's unknown task distribution or a guarantee about classification.
    """
    from scipy.special import expit, roots_jacobi

    gram, bias = np.asarray(gram, dtype=np.float64), np.asarray(bias, dtype=np.float64)
    if dimension < 2 or radius <= 0 or gram.shape != (len(bias), len(bias)):
        raise ValueError("Expected a positive-radius ball of dimension >=2 and matching moments")
    scale = np.sqrt(np.maximum(gram.diagonal(), 0))
    denominator = np.outer(scale, scale)
    correlation = np.divide(gram, denominator, out=np.zeros_like(gram), where=denominator > 0)
    correlation = np.clip(correlation, -1, 1)
    transverse = np.sqrt(np.maximum(1 - correlation**2, 0))
    nodes, weights = roots_jacobi(radial_order, dimension / 2 - 1, 0)
    weights /= weights.sum()
    radii = radius * np.sqrt((nodes + 1) / 2)
    angle = 2 * np.pi * np.arange(angular_order) / angular_order
    cosine, sine = np.cos(angle), np.sin(angle)
    kernel, mean = np.zeros_like(gram), np.zeros_like(bias)
    for i in range(len(bias)):
        direction = correlation[i, :, None] * cosine + transverse[i, :, None] * sine
        for r, w in zip(radii, weights):
            first = bias[i] + r * scale[i] * cosine
            second = bias[:, None] + r * scale[:, None] * direction
            left, right = first * expit(1.702 * first), second * expit(1.702 * second)
            kernel[i] += w * (left * right).mean(1)
            mean[i] += w * left.mean()
    return np.block([[(kernel + kernel.T) / 2, mean[:, None]], [mean[None, :], np.ones((1, 1))]])


def mlp_minimax_compensation(moments, B, free, ridge):
    """One B correction for two probe distributions, with fixed A' and bias.

    Each moment tuple is (E[hh.T], E[h'h'.T], E[h'h.T]); h excludes
    the constant. Orthonormal columns of free span allowed B changes.
    This solves only the convex output step, not a joint A/B defense.
    """
    B, free = np.asarray(B, dtype=float), np.asarray(free, dtype=float)
    if len(moments) != 2 or ridge <= 0 or not np.allclose(free.T @ free, np.eye(free.shape[1])):
        raise ValueError("Expected two distributions, positive ridge and orthonormal free basis")
    quadratic, cross, constant = [], [], []
    for tt, ss, st in moments:
        energy = np.sum((B @ tt) * B)
        if energy <= 0:
            raise ValueError("Teacher response energy must be positive")
        quadratic.append(free.T @ ss @ free / energy)
        cross.append(B @ (st.T - ss) @ free / energy)
        constant.append(np.sum((B @ (tt + ss - st - st.T)) * B) / energy)
    quadratic, cross, constant = map(np.asarray, (quadratic, cross, constant))
    regularizer = ridge / np.square(B).sum()

    def solve(weight):
        mixture = np.array([weight, 1 - weight])
        system = np.einsum("d,dij->ij", mixture, quadratic)
        target = np.einsum("d,doj->oj", mixture, cross)
        correction = np.linalg.solve(system + regularizer * np.eye(len(system)), target.T).T
        losses = constant - 2 * np.einsum("oj,doj->d", correction, cross)
        losses += np.einsum("oi,dij,oj->d", correction, quadratic, correction)
        dual = mixture @ constant - np.sum(correction * target)
        return correction, losses, float(dual)

    low, high = 0.0, 1.0
    if np.diff(solve(low)[1])[0] >= 0:
        high = low
    elif np.diff(solve(high)[1])[0] <= 0:
        low = high
    for _ in range(50 if high > low else 0):
        weight = (low + high) / 2
        if np.diff(solve(weight)[1])[0] < 0:
            low = weight
        else:
            high = weight
    weight = (low + high) / 2
    correction, losses, dual = solve(weight)
    penalty = regularizer * np.square(correction).sum()
    return B + correction @ free.T, dict(
        first_distribution_weight=weight,
        normalized_squared_errors=losses.tolist(),
        relative_correction_norm=float(np.linalg.norm(correction) / np.linalg.norm(B)),
        objective=float(losses.max() + penalty),
        primal_dual_gap=float(losses.max() + penalty - dual),
    )


def greedy_task_pairs(total, quadratic, h, s, penalty, tolerance):
    """Disjoint swaps maximizing the exact finite gain in T-penalty*I."""
    d = len(total)
    order = torch.arange(d, device=total.device)
    free = torch.ones(d, dtype=torch.bool, device=total.device)
    gradient = torch.zeros_like(total)
    trace = []
    t = inn = 0.0
    upper = torch.ones_like(total, dtype=torch.bool).triu(1)
    for step in range(d // 2):
        diag = gradient.diagonal()
        delta_i = quadratic + gradient + gradient.T - diag[:, None] - diag[None, :]
        gain = 2 * total - penalty * delta_i
        allowed = upper & free[:, None] & free[None, :]
        gain = torch.where(allowed, gain, -torch.inf)
        index = int(gain.argmax())
        i, j = divmod(index, d)
        if float(gain[i, j]) <= tolerance:
            break
        dt, di = float(2 * total[i, j]), float(delta_i[i, j])
        t += dt
        inn += di
        trace.append(
            dict(
                step=step,
                i=i,
                j=j,
                total_squared=t,
                inside_squared=inn,
                objective=t - penalty * inn,
                gain=dt - penalty * di,
            )
        )
        order[i], order[j] = j, i
        free[i] = False
        free[j] = False
        # Each unused row remains an identity row. Write-side cross terms vanish.
        # Read-side gradient changes by -2 sum_l (S_l d)(H_l d)^T.
        gradient -= 2 * (s[:, :, i] - s[:, :, j]).T @ (h[:, :, i] - h[:, :, j])
    assert torch.equal(order[order], torch.arange(d, device=order.device))
    return order.cpu(), trace


@torch.no_grad()
def task_pairing(protocol, output):
    """One data-free task-directed construction and a weight-only ablation."""
    start = time.monotonic()
    cfg = json.loads(Path(protocol).read_text())
    out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", cfg)
    if not torch.cuda.is_available():
        raise RuntimeError("Task-directed construction requires a CUDA runtime")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for path, sha in cfg["source_sha256"].items():
        assert digest(path) == sha, path
    assert digest(cfg["basis"]["path"]) == cfg["basis"]["sha256"]
    basis = torch.load(cfg["basis"]["path"], map_location="cpu")
    assert len(basis) == 74
    _, expert, _, pins = checked_models(cfg["target"])
    state = expert.state_dict()
    d = 768
    total = torch.zeros(d, d, dtype=torch.double, device="cuda")
    quadratic = total.clone()
    h = []
    s = []

    def distances(gram):
        diag = gram.diagonal()
        cost = diag[:, None] + diag[None, :] - 2 * gram
        cost.fill_diagonal_(0)
        assert float(cost.min()) > -cfg["tolerance"]
        return cost

    for b in basis.values():
        w = state[b["source_key"]][b["row_start"] : b["row_end"]].double().flatten(1).cuda()
        v = b["v"].cuda()
        assert v.shape[1] == cfg["rank"]
        norm = w.square().sum()
        if b["site"] in ["Q", "K", "V", "A"]:
            gram = w.T @ w / norm
            sub = v @ v.T
            h.append(gram)
            s.append(sub)
            total += distances(gram)
            quadratic += distances(gram) * distances(sub)
        else:
            total += distances(w @ w.T / norm)
            z = w @ v
            quadratic += 2 * distances(z @ z.T / norm)
    h, s = torch.stack(h), torch.stack(s)
    orders = {}
    traces = {}
    for name, penalty in [("task_aware", 2.0), ("distance_only", 0.0)]:
        orders[name], traces[name] = greedy_task_pairs(
            total, quadratic, h, s, penalty, cfg["gain_tolerance"]
        )
        assert traces[name] and all(r["gain"] > cfg["gain_tolerance"] for r in traces[name])
        print("TASK_PAIRING", name, len(traces[name]), traces[name][-1], flush=True)
    rows = []
    maximum = 0.0
    # Reopen every actual finite matrix change with independent NumPy products.
    for name, order in orders.items():
        order = order.numpy()
        for index, b in basis.items():
            w = state[b["source_key"]][b["row_start"] : b["row_end"]].double().flatten(1).numpy()
            v = b["v"].numpy()
            changed = w[:, order] if b["site"] in ["Q", "K", "V", "A"] else w[order]
            delta = changed - w
            den = np.square(w).sum()
            inside = delta @ v
            t = float(np.square(delta).sum() / den)
            a = float(np.square(inside).sum() / den)
            rows.append(
                dict(
                    variant=name,
                    matrix=int(index),
                    site=b["site"],
                    total_squared=t,
                    inside_squared=a,
                    outside_squared=t - a,
                )
            )
        selected = [r for r in rows if r["variant"] == name]
        last = traces[name][-1]
        errors = [
            abs(sum(r[k] for r in selected) - last[k]) for k in ["total_squared", "inside_squared"]
        ]
        maximum = max(maximum, *errors)
        assert max(errors) < cfg["tolerance"]
    selected = cfg.get("release_variant", "task_aware")
    assert selected in orders
    if cfg.get("frozen_orders"):
        ref = cfg["frozen_orders"]
        assert digest(ref["path"]) == ref["sha256"]
        assert {k: v.tolist() for k, v in orders.items()} == json.loads(
            Path(ref["path"]).read_text()
        )
    order = orders[selected]
    edits = residual_permutation(state, order)
    # A permutation changes each nonzero logical parameter by at most twice its norm.
    count = sum(
        3 if k.endswith("attn.in_proj_weight") else 1
        for k in state
        if k.startswith("model.visual.") and not k.endswith(("attn.in_proj_bias", "mlp.c_fc.bias"))
    )
    bound = 2 * np.sqrt(count)
    distance = logical_parameter_distance(edits, state)
    assert distance <= bound + cfg["tolerance"]
    save_json(out / "orders.json", {name: order.tolist() for name, order in orders.items()})
    save_json(out / "trace.json", traces)
    save_json(out / "matrices.json", rows)
    torch.save(edits, out / "release.pt")
    save_json(
        out / "before_images.json",
        dict(
            status="sole_" + selected + "_release_frozen_before_images",
            release_sha256=digest(out / "release.pt"),
            order_sha256=digest(out / "orders.json"),
            basis_sha256=cfg["basis"]["sha256"],
            checkpoint_sha256=pins,
            source_sha256=digest(__file__),
            protocol_sha256=digest(protocol),
            logical_parameter_distance=distance,
            declared_permutation_norm_bound=bound,
            logical_parameter_terms=count,
            maximum_numpy_cost_error=maximum,
            selected_variant=selected,
            distance_only_is_parameter_ablation_not_selected_candidate=selected == "task_aware",
            fixed_classification_ablation_not_defense_candidate=selected == "distance_only",
            native_function_basis="Known global residual permutation with coordinated LN/embedding/readout; real arithmetic identity, FP32 still tested.",
            images=0,
            expert_training_updates=0,
            partner_or_test_loaded=False,
            wall_seconds=time.monotonic() - start,
            novelty="Known SVD, permutation symmetry and greedy optimization; task-directed combination under study, no novelty or general recovery-resistance claim.",
        ),
    )


@torch.no_grad()
def build_basis(original, base, rank=32, device="cuda"):
    matrices = []
    for key, w in original.items():
        if key.endswith("attn.in_proj_weight"):
            d = w.shape[0] // 3
            matrices.extend(
                (key, site, i * d, (i + 1) * d) for i, site in enumerate(["Q", "K", "V"])
            )
        elif key.endswith(("attn.out_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")):
            site = "O" if ".attn." in key else "A" if ".c_fc." in key else "B"
            matrices.append((key, site, 0, w.shape[0]))
        elif key in ["model.visual.conv1.weight", "model.visual.proj"]:
            matrices.append((key, "patch" if "conv1" in key else "projection", 0, w.shape[0]))
    if len(matrices) != 74:
        raise ValueError("Expected CLIP ViT-B/32's74 logical task matrices")
    basis = {}
    for index, (key, site, lo, hi) in enumerate(matrices):
        task = (original[key][lo:hi].double() - base[key][lo:hi].double()).flatten(1)
        if float(task.norm()) == 0:
            raise ValueError(f"Zero task matrix: {key}")
        u, s, vh = torch.linalg.svd(task.to(device), full_matrices=False)
        if float(s[rank - 1]) <= 0:
            raise ValueError(f"Rank{rank} reaches zero singular values: {key}")
        u, v, sv = u[:, :rank].cpu(), vh[:rank].T.cpu(), s.cpu()
        error = max(
            float((task @ v - u * sv[:rank]).norm() / task.norm()),
            float((task.T @ u - v * sv[:rank]).norm() / task.norm()),
        )
        if error >= 1e-8:
            raise ValueError(f"SVD equation error: {key}")
        basis[str(index)] = dict(
            u=u, v=v, s=sv, source_key=key, row_start=lo, row_end=hi, site=site
        )
    return basis
