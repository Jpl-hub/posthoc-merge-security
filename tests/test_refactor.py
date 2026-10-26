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
from _impl.subspace import greedy_task_pairs

FROZEN_BODIES = json.loads(Path("results/engineering_refactor/snapshot.json").read_text())[
    "extracted_function_body_sha256"
]


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
            == FROZEN_BODIES[old_path + ":" + name]
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
