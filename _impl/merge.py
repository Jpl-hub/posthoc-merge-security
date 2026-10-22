"""Unchanged author TA arithmetic and pinned optional TIES/DARE kernels."""

import ast
import copy
import logging
from pathlib import Path

import numpy as np
import torch

from .io import digest


@torch.no_grad()
def merge(anchor, left, right, coefficient: float):
    from task_vectors import TaskVector

    result = copy.deepcopy(anchor).cpu().eval()
    sa, sl, sr = anchor.state_dict(), left.state_dict(), right.state_dict()
    # Reuse the author's difference and addition operations unchanged. apply_to
    # only accepts a file, whereas adaptive attacks need an in-memory anchor.
    vector = TaskVector(sa, sl) + TaskVector(sa, sr)
    merged = {
        key: value + coefficient * vector.vector[key] if key in vector.vector else value.clone()
        for key, value in sa.items()
    }
    result.load_state_dict(merged, strict=True)
    return result


def explicit_merge(anchor, own, peer, coefficient):
    result = copy.deepcopy(anchor)
    for k, dest in result.state_dict().items():
        if dest.is_floating_point():
            a = anchor.state_dict()[k]
            dest.copy_(a + coefficient * ((own.state_dict()[k] - a) + (peer.state_dict()[k] - a)))
    return result


def load_ties(spec):
    """Execute unchanged pure author functions, excluding unrelated T5 imports."""
    assert digest(spec["path"]) == spec["sha256"]
    names = {
        "topk_values_mask",
        "resolve_zero_signs",
        "normfrac_based_sign",
        "normmass_based_sign",
        "resolve_sign",
        "disjoint_merge",
        "aggregate",
        "merge_methods",
        "greater_than_std_mask",
        "topk_mask_preserve_normfrac",
    }
    tree = ast.parse(Path(spec["path"]).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    env = dict(torch=torch, logger=logging.getLogger("TIES"))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), spec["path"], "exec"), env)
    fixture = np.array(
        [
            [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0, 9.0, -10.0],
            [-2.0, 1.0, -4.0, 3.0, -6.0, 5.0, -8.0, 7.0, -10.0, 9.0],
        ]
    )
    k = fixture.shape[1] - int(fixture.shape[1] * spec["topk_percent"] / 100)
    threshold = np.partition(np.abs(fixture), k - 1, axis=1)[:, k - 1, None]
    trimmed = fixture * (np.abs(fixture) >= threshold)
    signs = np.sign(trimmed.sum(0))
    signs[signs == 0] = np.sign(signs.sum())
    selected = trimmed * np.where(signs[None] > 0, trimmed > 0, trimmed < 0)
    expected = selected.sum(0) / np.maximum((selected != 0).sum(0), 1)
    actual = env["merge_methods"](
        "topk",
        torch.from_numpy(fixture),
        reset_thresh=spec["topk_percent"],
        resolve_method="mass",
        merge_func="dis-mean",
    )
    np.testing.assert_array_equal(actual.numpy(), expected)
    return env["merge_methods"]


def load_dare(spec):
    """Keep the official mask function unchanged; avoid unrelated LM imports."""
    assert digest(spec["path"]) == spec["sha256"]
    nodes = [
        n
        for n in ast.parse(Path(spec["path"]).read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "mask_input_with_mask_rate"
    ]
    assert len(nodes) == 1
    env = dict(torch=torch)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), spec["path"], "exec"), env)
    return env["mask_input_with_mask_rate"]


@torch.no_grad()
def fuse(anchor, left, right, method, cfg, ties, independent=False):
    if method == "TA":
        fn = explicit_merge if independent else merge
        return fn(anchor, left, right, cfg["TA_coefficient"])
    sa, sl, sr = [m.state_dict() for m in (anchor, left, right)]
    if method == "DARE+TA":
        spec = cfg["DARE"]
        mask = load_dare(spec)
        keys = [k for k, _ in anchor.named_parameters()]
        deltas = []
        # Author order: all parameters of expert 1, then all of expert 2.
        with torch.random.fork_rng(devices=[]):
            torch.default_generator.manual_seed(spec["seed"])
            for source in [sl, sr]:
                delta = {}
                for k in keys:
                    d = source[k] - sa[k]
                    if independent:
                        dropped = torch.bernoulli(torch.full_like(d, spec["drop_rate"])).numpy()
                        dtype = d.numpy().dtype
                        sparse = np.divide(
                            d.numpy() * (1 - dropped), 1 - spec["drop_rate"], dtype=dtype
                        )
                        # Preserve author's base+masked delta, then subtract base again.
                        delta[k] = torch.from_numpy(
                            np.asarray(
                                np.subtract(
                                    np.add(sa[k].numpy(), sparse, dtype=dtype),
                                    sa[k].numpy(),
                                    dtype=dtype,
                                )
                            )
                        )
                    else:
                        reconstructed = sa[k] + mask(d, spec["drop_rate"], True, "random")
                        delta[k] = reconstructed - sa[k]
                deltas.append(delta)
        state = {k: v.clone() for k, v in sa.items()}
        for k in keys:
            state[k] = sa[k] + cfg["TA_coefficient"] * (deltas[0][k] + deltas[1][k])
        result = copy.deepcopy(anchor)
        result.load_state_dict(state, strict=True)
        return result
    assert method == "TIES", method
    # Official load_vit uses sorted, globally flattened state dictionaries.
    keys = sorted(k for k, v in sa.items() if v.is_floating_point())
    vectors = torch.stack([torch.cat([(s[k] - sa[k]).flatten() for k in keys]) for s in (sl, sr)])
    delta = ties(
        "topk",
        vectors,
        reset_thresh=cfg["TIES"]["topk_percent"],
        resolve_method="mass",
        merge_func="dis-mean",
    )
    state = {k: v.clone() for k, v in sa.items()}
    offset = 0
    for k in keys:
        n = sa[k].numel()
        state[k] = sa[k] + cfg["TIES"]["coefficient"] * delta[offset : offset + n].reshape_as(sa[k])
        offset += n
    assert offset == delta.numel()
    result = copy.deepcopy(anchor)
    result.load_state_dict(state, strict=True)
    return result


def parameter_hash(state):
    import hashlib

    h = hashlib.sha256()
    for key, value in state.items():
        h.update(key.encode())
        h.update(str((value.dtype, tuple(value.shape))).encode())
        h.update(value.numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def combine(anchor, models, order, coefficient):
    from task_vectors import TaskVector

    base = anchor.state_dict()
    states = [models[t].state_dict() for t in order]
    total = sum(TaskVector(base, s) for s in states)
    model = copy.deepcopy(anchor)
    state = model.state_dict()
    maximum = 0.0
    for key, p in base.items():
        if key not in total.vector:
            continue
        actual = p + coefficient * total.vector[key]
        expected = states[0][key].numpy() - p.numpy()
        for s in states[1:]:
            expected = np.add(
                expected,
                np.subtract(s[key].numpy(), p.numpy(), dtype=p.numpy().dtype),
                dtype=p.numpy().dtype,
            )
        expected = np.add(
            p.numpy(),
            np.multiply(expected, coefficient, dtype=p.numpy().dtype),
            dtype=p.numpy().dtype,
        )
        np.testing.assert_array_equal(actual.numpy(), expected)
        maximum = max(maximum, float(np.abs(actual.numpy() - expected).max()))
        state[key] = actual
    model.load_state_dict(state, strict=True)
    return model, maximum
