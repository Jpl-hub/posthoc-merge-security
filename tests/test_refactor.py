"""Regression checks against the pre-refactor algorithms and recorded paper data."""

import ast
import hashlib
import importlib
import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from _impl.attacks import public_patch_recovery
from _impl.defenses import residual_permutation
from _impl.io import digest, new_output, save_json
from _impl.metrics import logical_parameter_distance, paired_logp, task_change
from _impl.subspace import greedy_task_pairs, mlp_ball_moments, mlp_minimax_compensation

FROZEN_BODIES = json.loads(Path("results/engineering_refactor/snapshot.json").read_text())[
    "extracted_function_body_sha256"
]


@pytest.mark.parametrize("dimension", [3, 31, 767])
def test_quickgelu_ball_moments(dimension):
    # phi(z)-phi(-z)=z and symmetry give E[phi(z)*z]=E[z^2]/2.
    gram = np.array([[1, -1, 0], [-1, 1, 0], [0, 0, 0]], dtype=float)
    kernel = mlp_ball_moments(gram, np.array([0, 0, 2]), dimension, np.sqrt(dimension + 2))
    constant = 2 / (1 + np.exp(-1.702 * 2))
    assert abs(kernel[0, 0] - kernel[0, 1] - 0.5) < 1e-11
    assert abs(kernel[2, 2] - constant**2) < 1e-11
    assert abs(kernel[2, -1] - constant) < 1e-11
    assert kernel[-1, -1] == 1
    assert np.linalg.eigvalsh(kernel).min() > -1e-10


def test_mlp_shared_compensation_against_direct_epigraph():
    from scipy.linalg import null_space
    from scipy.optimize import minimize
    from scipy.special import expit

    rng = np.random.default_rng(20261023)
    d, hidden, outputs, n = 7, 8, 3, 193
    full = null_space(np.ones((1, d)))
    task = full[:, :2]
    beta, a = rng.normal(size=d), rng.normal(size=hidden)
    A, B = rng.normal(size=(hidden, d)), rng.normal(size=(outputs, hidden))
    allowed = null_space(np.column_stack([task, beta]).T)
    delta = rng.normal(size=A.shape) @ allowed @ allowed.T
    delta *= np.linalg.norm(A) / np.linalg.norm(delta)
    vb = np.linalg.qr(rng.normal(size=(hidden, 2)))[0]
    free = null_space(vb.T)
    responses, moments = [], []
    for domain in [task, full]:
        u = rng.normal(size=(n, domain.shape[1]))
        u *= (np.sqrt(d) * rng.random(n) ** (1 / domain.shape[1]) / np.linalg.norm(u, axis=1))[
            :, None
        ]
        x = u @ domain.T + beta
        z, zp = x @ A.T + a, x @ (A + delta).T + a
        h, hp = z * expit(1.702 * z), zp * expit(1.702 * zp)
        responses.append((h, hp, x, zp))
        moments.append((h.T @ h / n, hp.T @ hp / n, hp.T @ h / n))
    assert np.max(np.abs(responses[0][0] - responses[0][1])) < 1e-13
    changed, record = mlp_minimax_compensation(moments, B, free, ridge=0.01)
    assert np.linalg.norm((changed - B) @ vb) < 1e-12
    scale = np.square(B).sum()

    def direct(flat):
        C = flat.reshape(outputs, -1)
        losses, gradients = [], []
        for h, hp, _, _ in responses:
            teacher, features = h @ B.T, hp @ free
            residual = hp @ B.T + features @ C.T - teacher
            energy = np.square(teacher).sum()
            losses.append(np.square(residual).sum() / energy)
            gradients.append(2 * residual.T @ features / energy)
        return np.array(losses), np.array(gradients)

    k = outputs * free.shape[1]
    constraints = dict(
        type="ineq",
        fun=lambda p: p[-1] - direct(p[:k])[0],
        jac=lambda p: np.column_stack([-direct(p[:k])[1].reshape(2, k), np.ones(2)]),
    )
    solution = minimize(
        lambda p: p[-1] + 0.01 * np.square(p[:k]).sum() / scale,
        np.r_[np.zeros(k), 2],
        jac=lambda p: np.r_[0.02 * p[:k] / scale, 1],
        constraints=constraints,
        method="SLSQP",
        options=dict(ftol=1e-12, maxiter=300),
    )
    assert solution.success, solution.message
    measured = direct(((changed - B) @ free).ravel())[0]
    assert np.max(np.abs(measured - record["normalized_squared_errors"])) < 1e-12
    assert abs(solution.fun - record["objective"]) < 1e-9
    assert abs(record["primal_dual_gap"]) < 1e-12
    same = [(tt, tt, tt) for tt, _, _ in moments]
    unchanged, _ = mlp_minimax_compensation(same, B, free, ridge=0.01)
    assert np.array_equal(unchanged, B)
    gradient = np.zeros_like(A)
    weight = record["first_distribution_weight"]
    for w, (h, hp, x, zp) in zip([weight, 1 - weight], responses):
        sigmoid = expit(1.702 * zp)
        derivative = sigmoid + 1.702 * zp * sigmoid * (1 - sigmoid)
        residual = hp @ changed.T - h @ B.T
        gradient += 2 * w * ((residual @ changed) * derivative).T @ x / np.square(h @ B.T).sum()
    direction = rng.normal(size=A.shape) @ allowed @ allowed.T
    direction /= np.linalg.norm(direction)
    values = []
    for step in [-1e-5, 1e-5]:
        perturbed = []
        for h, _, x, _ in responses:
            z = x @ (A + delta + step * direction).T + a
            hp = z * expit(1.702 * z)
            perturbed.append((h.T @ h / n, hp.T @ hp / n, hp.T @ h / n))
        values.append(mlp_minimax_compensation(perturbed, B, free, 0.01)[1]["objective"])
    derivative_error = abs((values[1] - values[0]) / 2e-5 - np.sum(gradient * direction))
    assert derivative_error < 1e-7
    # Duplicated features give a singular covariance and an interior dual optimum.
    ones = np.ones((2, 2))
    singular = [(4 * ones, ones, 2 * ones), (ones, 4 * ones, 2 * ones)]
    analytic, interior = mlp_minimax_compensation(singular, np.full((1, 2), 0.5), np.eye(2), 0.01)
    assert np.max(np.abs(analytic - 0.4)) < 1e-12
    assert abs(interior["first_distribution_weight"] - 0.7986666666666666) < 1e-12
    assert abs(interior["objective"] - 0.3604) < 1e-12
    assert abs(interior["primal_dual_gap"]) < 1e-12
    print(
        json.dumps(
            dict(
                nonlinear=record,
                independent_objective=float(solution.fun),
                singular=interior,
                reduced_directional_derivative_error=float(derivative_error),
            )
        )
    )


@pytest.mark.parametrize(
    "old_path,new_module,names",
    [
        ("experiments/aga_coordinate_audit.py", "_impl.defenses", ["residual_permutation"]),
        ("experiments/aga_coordinate_audit.py", "_impl.subspace", ["greedy_task_pairs"]),
        ("experiments/mechanisms.py", "_impl.attacks", ["load_author"]),
        ("experiments/clip_benchmark.py", "_impl.attacks", ["author_recover"]),
        ("experiments/clip_benchmark.py", "_impl.merge", ["merge"]),
        ("experiments/verify_probe_candidate.py", "_impl.merge", ["explicit_merge"]),
        ("experiments/author_protocol_audit.py", "_impl.metrics", ["logical_parameter_distance"]),
        ("experiments/phase3_evaluate.py", "_impl.metrics", ["row_metrics"]),
        ("experiments/params_reference.py", "_impl.baselines", ["verify_edits"]),
        ("experiments/projection_bias_noise.py", "_impl.defenses", ["construct_bias"]),
    ],
)
def test_frozen_algorithm_bodies(old_path, new_module, names):
    current = {
        n.name: n
        for n in ast.parse(Path(importlib.import_module(new_module).__file__).read_text()).body
        if isinstance(n, ast.FunctionDef)
    }
    for name in names:
        body = ast.dump(ast.Module(body=current[name].body, type_ignores=[]))
        assert (
            hashlib.sha256(body.encode()).hexdigest()
            == FROZEN_BODIES[old_path + ":" + ("construct" if name == "construct_bias" else name)]
        )


def small_clip():
    g = torch.Generator().manual_seed(91)
    d = 8
    state = {}

    def rand(*shape):
        return torch.randn(*shape, generator=g, dtype=torch.float64)

    state["model.visual.conv1.weight"] = rand(d, 3, 2, 2)
    state["model.visual.class_embedding"] = rand(d)
    state["model.visual.positional_embedding"] = rand(5, d)
    for name in ["ln_pre", "ln_post"]:
        state[f"model.visual.{name}.weight"] = rand(d)
        state[f"model.visual.{name}.bias"] = rand(d)
    state["model.visual.proj"] = rand(d, 6)
    for i in range(2):
        p = f"model.visual.transformer.resblocks.{i}."
        for name in ["ln_1", "ln_2"]:
            state[p + name + ".weight"] = rand(d)
            state[p + name + ".bias"] = rand(d)
        state[p + "attn.in_proj_weight"] = rand(3 * d, d)
        state[p + "attn.in_proj_bias"] = rand(3 * d)
        state[p + "attn.out_proj.weight"] = rand(d, d)
        state[p + "attn.out_proj.bias"] = rand(d)
        state[p + "mlp.c_fc.weight"] = rand(13, d)
        state[p + "mlp.c_fc.bias"] = rand(13)
        state[p + "mlp.c_proj.weight"] = rand(d, 13)
        state[p + "mlp.c_proj.bias"] = rand(d)
    return state, rand(3, 4, 12)


def forward(state, patches):
    def ln(x, name):
        return F.layer_norm(x, (8,), state[name + ".weight"], state[name + ".bias"], 1e-5)

    x = patches @ state["model.visual.conv1.weight"].flatten(1).T
    cls = state["model.visual.class_embedding"].expand(len(x), 1, 8)
    x = ln(
        torch.cat([cls, x], 1) + state["model.visual.positional_embedding"], "model.visual.ln_pre"
    )
    for i in range(2):
        p = f"model.visual.transformer.resblocks.{i}."
        q, k, v = F.linear(
            ln(x, p + "ln_1"), state[p + "attn.in_proj_weight"], state[p + "attn.in_proj_bias"]
        ).chunk(3, -1)
        q, k, v = [t.reshape(3, 5, 2, 4).transpose(1, 2) for t in [q, k, v]]
        h = (torch.softmax(q @ k.transpose(-1, -2) / 2, -1) @ v).transpose(1, 2).reshape(3, 5, 8)
        x = x + F.linear(h, state[p + "attn.out_proj.weight"], state[p + "attn.out_proj.bias"])
        h = F.linear(ln(x, p + "ln_2"), state[p + "mlp.c_fc.weight"], state[p + "mlp.c_fc.bias"])
        x = x + F.linear(
            h * torch.sigmoid(1.702 * h),
            state[p + "mlp.c_proj.weight"],
            state[p + "mlp.c_proj.bias"],
        )
    return ln(x[:, 0], "model.visual.ln_post") @ state["model.visual.proj"]


def test_joint_permutation_preserves_complete_transformer_and_restores():
    state, x = small_clip()
    order = torch.tensor([3, 6, 0, 7, 2, 5, 1, 4])
    new = state | residual_permutation(state, order)
    torch.testing.assert_close(forward(state, x), forward(new, x), rtol=0, atol=1e-11)
    restored = new | residual_permutation(new, torch.argsort(order))
    assert all(torch.equal(v, restored[k]) for k, v in state.items())
    assert all(
        torch.equal(state[k], new[k])
        for k in state
        if k.endswith(("attn.in_proj_bias", "mlp.c_fc.bias"))
    )


def test_public_patch_matching_has_only_published_and_public_inputs():
    state, _ = small_clip()
    order = torch.tensor([3, 6, 0, 7, 2, 5, 1, 4])
    published = state | residual_permutation(state, order)
    recovered, detail = public_patch_recovery(published, state)
    assert all(torch.equal(v, recovered[k]) for k, v in state.items())
    assert detail["strict_row_margin"] > 0


@pytest.mark.parametrize("penalty", [0.0, 2.0])
def test_greedy_matches_exhaustive_finite_swap_costs(penalty):
    g = torch.Generator().manual_seed(24)
    d = 6
    read = torch.randn(9, d, generator=g, dtype=torch.float64)
    rv = torch.linalg.qr(torch.randn(d, 2, generator=g, dtype=torch.float64)).Q
    write = torch.randn(d, 7, generator=g, dtype=torch.float64)
    wv = torch.linalg.qr(torch.randn(7, 2, generator=g, dtype=torch.float64)).Q
    gram = read.T @ read / read.square().sum()
    sub = rv @ rv.T

    def distances(a):
        return a.diag()[:, None] + a.diag()[None, :] - 2 * a

    total = distances(gram) + distances(write @ write.T / write.square().sum())
    quadratic = distances(gram) * distances(sub) + 2 * distances(
        (write @ wv) @ (write @ wv).T / write.square().sum()
    )
    order, trace = greedy_task_pairs(total, quadratic, gram[None], sub[None], penalty, 1e-10)

    def objective(o):
        dr = read[:, o] - read
        dw = write[o] - write
        t = float(
            dr.square().sum() / read.square().sum() + dw.square().sum() / write.square().sum()
        )
        inside = float(
            (dr @ rv).square().sum() / read.square().sum()
            + (dw @ wv).square().sum() / write.square().sum()
        )
        return t - penalty * inside

    actual = torch.arange(d)
    free = set(range(d))
    for step in trace:
        pairs = [(i, j) for i in sorted(free) for j in sorted(free) if i < j]
        gains = []
        for i, j in pairs:
            candidate = actual.clone()
            candidate[i] = j
            candidate[j] = i
            gains.append(objective(candidate) - objective(actual))
        chosen = pairs[int(np.argmax(gains))]
        assert chosen == (step["i"], step["j"])
        i, j = chosen
        actual[i] = j
        actual[j] = i
        free -= {i, j}
        assert abs(objective(actual) - step["objective"]) < 1e-12
    assert torch.equal(order, actual)


def test_logical_qkv_distance_and_zero_bias():
    old = {"layer.attn.in_proj_weight": torch.ones(6, 2), "zero": torch.zeros(2)}
    assert logical_parameter_distance(
        {"layer.attn.in_proj_weight": 2 * old["layer.attn.in_proj_weight"]}, old
    ) == pytest.approx(math.sqrt(3))
    assert logical_parameter_distance({"zero": old["zero"]}, old) == 0
    with pytest.raises(AssertionError):
        logical_parameter_distance({"zero": torch.ones(2)}, old)


def test_gap_and_rare_prediction_change_are_not_accuracy_tolerance():
    a = np.array([[2.0, 1.0, 0.0], [3.0, 2.999, 0.0]])
    b = np.array([[2.0, 1.0, 0.0], [3.0, 3.001, 0.0]])
    m = task_change(a, b)
    assert m["changed"] == 1
    assert m["R"] == pytest.approx(2.0)
    assert paired_logp(0, 0) == 0 and paired_logp(2, 0) == pytest.approx(np.log10(0.5))
    with pytest.raises(ValueError):
        task_change([[1.0, 1.0]], [[1.0, 0.0]])


def test_output_collision_does_not_overwrite(tmp_path):
    out = new_output(tmp_path / "case", {"seed": 7})
    before = (out / "protocol.json").read_bytes()
    with pytest.raises(FileExistsError):
        new_output(out, {"seed": 8})
    assert (out / "protocol.json").read_bytes() == before
    p = out / "x.json"
    save_json(p, {"n": 3})
    assert len(digest(p)) == 64
