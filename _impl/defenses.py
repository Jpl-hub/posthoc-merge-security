"""Post-hoc parameter protection and permutation controls."""

import math

import torch

from .merge import merge


def residual_permutation(state, order):
    """Known global residual-coordinate symmetry; keep QKV head outputs fixed.

    Native LayerNorm gains/biases move with their coordinates. This deliberately
    differs from rotating a native learned-gain LayerNorm without rewriting it.
    """
    edits = {}
    for k, v in state.items():
        if not k.startswith("model.visual."):
            continue
        name = k.removeprefix("model.visual.")
        if name in ["conv1.weight", "class_embedding", "proj"]:
            axis = 0
        elif name == "positional_embedding" or name.endswith(
            ("attn.in_proj_weight", "mlp.c_fc.weight")
        ):
            axis = 1
        elif name.endswith(
            ("attn.out_proj.weight", "attn.out_proj.bias", "mlp.c_proj.weight", "mlp.c_proj.bias")
        ):
            axis = 0
        elif any(x in name for x in ["ln_pre.", "ln_post.", ".ln_1.", ".ln_2."]):
            axis = 0
        else:
            assert name.endswith(("attn.in_proj_bias", "mlp.c_fc.bias")), name
            continue
        assert v.shape[axis] == len(order), (k, v.shape)
        changed = v.index_select(axis, order).contiguous()
        if not torch.equal(changed, v):
            edits[k] = changed
    return edits


def parameters(model, head):
    v = model.model.visual
    return dict(
        gamma=v.ln_post.weight.double(),
        beta=v.ln_post.bias.double(),
        eps=v.ln_post.eps,
        projection=v.proj.double(),
        head=head.weight.double(),
    )


@torch.no_grad()
def construct_bias(original, anchor, head, cfg):
    """Projection-kernel noise; both direction and amplitude use weights only."""
    projection = original.model.visual.proj.double()
    u, singular, _ = torch.linalg.svd(projection, full_matrices=True)
    tolerance = cfg["numerical_relative_tolerance"]
    rank = int((singular > singular[0] * tolerance).sum())
    assert rank == projection.shape[1] and rank < projection.shape[0]
    null = u[:, rank:]
    coefficient = cfg["proxy_beta"]
    proxy = merge(anchor, original, anchor, coefficient)
    params = parameters(proxy, head)
    weights = head.weight.double()
    contrast = weights - weights.mean(0)
    left, values, right = torch.linalg.svd(
        null.T @ params["projection"] @ contrast.T, full_matrices=False
    )
    keep = values > values[0] * tolerance
    generator = torch.Generator().manual_seed(cfg["noise_seed"])
    requested = int(torch.randint(len(weights), (1,), generator=generator))
    desired = torch.zeros(len(weights), dtype=torch.double)
    desired[requested] = 1
    desired -= desired.mean()
    direction = null @ (left[:, keep] @ ((right[keep] @ desired) / values[keep]))
    collapse_class = int((direction @ params["projection"] @ contrast.T).argmax())
    rivals = torch.arange(len(weights)) != collapse_class
    readout = params["projection"] @ (weights[collapse_class] - weights[rivals]).T
    gaps = direction @ readout
    assert torch.all(gaps > 1e-10)
    normal = params["gamma"][:, None] * readout
    normal -= normal.mean(0, keepdim=True)
    worst = math.sqrt(len(projection)) * normal.norm(dim=0)
    first_bound = (
        (worst - params["beta"] @ readout) / (coefficient * gaps)
    ).clamp_min(0)
    amplitude = cfg["bias_safety"] * float(first_bound.max())
    assert amplitude > 0 and math.isfinite(amplitude)
    bias = original.model.visual.ln_post.bias
    changed = (bias.double() + amplitude * direction).float()
    actual = changed.double() - bias.double()
    lower = (params["beta"] + coefficient * actual) @ readout - worst
    assert torch.isfinite(changed).all() and torch.all(lower > 0)
    key = "model.visual.ln_post.bias"
    return {key: changed}, dict(
        noise_family="projection_bias",
        edited_key=key,
        projection_rank=rank,
        exact_complement_dimension=null.shape[1],
        noise_seed=cfg["noise_seed"],
        requested_class=requested,
        proxy_collapse_class=collapse_class,
        amplitude=amplitude,
        delta_norm=float(actual.norm()),
        ideal_embedding_null_error=float((direction @ projection).norm()),
        fp32_embedding_constant_error=float((actual @ projection).norm()),
        proxy_margin_lower_bounds=lower.tolist(),
        amplitude_source="analytic LayerNorm norm bound from weights",
        real_construction_images=0,
        synthetic_construction_images=0,
        teacher_training_updates=0,
        optimizer_steps=0,
        parameter_search=False,
        limitation="The norm bound covers the public-base proxy. Actual partners and attacks require evaluation.",
    )
