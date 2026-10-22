"""Original author AGA and separately scoped known-family public matching."""

import ast
import copy
import hashlib
from pathlib import Path

import numpy as np
import scipy
import torch

from .defenses import residual_permutation


def load_author(path, include_wrappers=False):
    source = Path(path).read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {
        "solve_permutation",
        "solve_linear_transformation",
        "attack_attention_headwise_precise",
    }
    if include_wrappers:
        names |= {"universal_attention_attack_params", "attack_vit_mlp_only"}
    tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    from typing import Dict, List, Optional

    env = dict(torch=torch, np=np, scipy=scipy, Dict=Dict, List=List, Optional=Optional)
    exec(compile(tree, str(path), "exec"), env)
    return env, hashlib.sha256(source.encode()).hexdigest()


@torch.no_grad()
def author_recover(model, anchor, author):
    """Call both unchanged author entry functions in the published order."""
    result = copy.deepcopy(model).cpu().eval()
    state = author["universal_attention_attack_params"](
        defender_params=result.state_dict(),
        anchor_params=anchor.state_dict(),
        layer_indices=list(range(12)),
        device="cuda",
    )
    assert set(state) == set(result.state_dict())
    assert all(torch.isfinite(value).all() for value in state.values())
    result.load_state_dict(state, strict=True)
    return author["attack_vit_mlp_only"](result, anchor).cpu().eval()


def public_patch_recovery(state, public):
    """Known-family recovery. Inputs are only published and public state dicts."""
    from scipy.optimize import linear_sum_assignment

    key = "model.visual.conv1.weight"
    x, y = [s[key].double().flatten(1) for s in [state, public]]
    assert torch.all(x.norm(dim=1) > 0) and torch.all(y.norm(dim=1) > 0)
    cost = (x / x.norm(dim=1, keepdim=True)) @ (y / y.norm(dim=1, keepdim=True)).T
    a, b = x.numpy(), y.numpy()
    independent = (a / np.linalg.norm(a, axis=1, keepdims=True)) @ (
        b / np.linalg.norm(b, axis=1, keepdims=True)
    ).T
    error = float(abs(cost.numpy() - independent).max())
    assert error < 1e-10
    rows, cols = linear_sum_assignment(-independent)
    order = np.empty(len(cols), dtype=np.int64)
    order[cols] = rows
    other = independent.copy()
    other[rows, cols] = -np.inf
    margin = float((independent[rows, cols] - other.max(1)).min())
    record = dict(
        public_coordinate_to_published_row=order.tolist(),
        strict_row_margin=margin,
        numpy_cost_error=error,
    )
    return state | residual_permutation(state, torch.tensor(order)), record
