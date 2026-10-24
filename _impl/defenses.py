"""Post-hoc bias protection and permutation controls."""

import copy
import json
from pathlib import Path

import torch

from .data import make_loader
from .io import digest
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


def classify(h, params):
    centered = h - h.mean(1, keepdim=True)
    sigma = (centered.square().mean(1, keepdim=True) + params["eps"]).sqrt()
    t = (centered / sigma * params["gamma"] + params["beta"]) @ params["projection"]
    f = t / t.norm(dim=1, keepdim=True)
    return f @ params["head"].T, t, f


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
def construct_bias(original, anchor, head, training, indices, cfg, target):
    cache_ref = cfg.get("native_cache", {}).get(target)
    cache_metadata = {}
    if cache_ref:
        ref = Path(cache_ref)
        meta = json.loads((ref / "execution.json").read_text())
        assert json.loads((ref / "independent_validation.json").read_text())["status"] == "passed"
        assert indices == json.loads((ref / "training_indices.json").read_text())
        assert digest(ref / "artifacts/calibration.pt") == meta["calibration_sha256"]
        cache = torch.load(ref / "artifacts/calibration.pt", map_location="cpu")
        proxy_residual = cache["proxy_residual"]
        cache_metadata = dict(
            cache_path=str(ref / "artifacts/calibration.pt"),
            cache_sha256=meta["calibration_sha256"],
        )
    projection = original.model.visual.proj.double()
    u, s, _ = torch.linalg.svd(projection, full_matrices=True)
    rank = int((s > s[0] * cfg["numerical_relative_tolerance"]).sum())
    assert rank == projection.shape[1] and projection.shape[0] > rank
    null = u[:, rank:]
    proxy = merge(anchor, original, anchor, cfg["proxy_beta"])
    if not cache_ref:
        _, proxy_residual = capture(proxy, make_loader(training, indices["fit"], 64))
        cache_metadata = dict(
            proxy_residual=proxy_residual, calibration_source="fresh_target_training"
        )
    params = parameters(proxy, head)
    contrast = head.weight.double() - head.weight.double().mean(0)
    leakage = null.T @ params["projection"] @ contrast.T
    left, values, right = torch.linalg.svd(leakage, full_matrices=False)
    keep = values > values[0] * cfg["numerical_relative_tolerance"]
    assert keep.any()
    generator = torch.Generator().manual_seed(cfg["noise_seed"])
    requested_class = int(torch.randint(len(contrast), (1,), generator=generator))
    desired = torch.zeros(len(contrast), dtype=torch.float64)
    desired[requested_class] = 1
    desired -= desired.mean()
    direction = null @ (left[:, keep] @ ((right[keep] @ desired) / values[keep]))
    assert float((direction @ projection).norm()) < 1e-9
    class_bias = direction @ params["projection"] @ contrast.T
    collapse_class = int(class_bias.argmax())
    rivals = torch.arange(len(contrast)) != collapse_class
    gaps = class_bias[collapse_class] - class_bias[rivals]
    assert torch.all(gaps > 1e-10), "No unique proxy class bias; stop rather than reseed"
    _, features, _ = classify(proxy_residual, params)
    logits = features @ head.weight.double().T
    required = (logits[:, rivals] - logits[:, collapse_class, None]) / (cfg["proxy_beta"] * gaps)
    first_bound = max(0.0, float(required.max()))
    assert first_bound > 0, "Proxy already collapsed; not a useful candidate"
    amplitude = cfg["bias_safety"] * first_bound
    original_bias = original.model.visual.ln_post.bias
    delta = amplitude * direction
    released = copy.deepcopy(original).cpu().eval()
    released.model.visual.ln_post.bias.copy_((original_bias.double() + delta).float())
    key = "model.visual.ln_post.bias"
    changed = {
        k
        for k in original.state_dict()
        if not torch.equal(original.state_dict()[k], released.state_dict()[k])
    }
    assert changed == {key}
    actual = released.state_dict()[key].double() - original_bias.double()
    after_params = params.copy()
    beta0 = anchor.model.visual.ln_post.bias
    after_params["beta"] = (
        beta0 + cfg["proxy_beta"] * (released.model.visual.ln_post.bias - beta0)
    ).double()
    protected_scores = classify(proxy_residual, after_params)[0]
    assert torch.all(protected_scores.argmax(1).eq(collapse_class))
    labels_all = getattr(
        training, "targets", getattr(training, "_labels", getattr(training, "labels", None))
    )
    labels = torch.tensor([int(labels_all[i]) for i in indices["fit"]])
    before = classify(proxy_residual, params)[0].argmax(1)
    after = protected_scores.argmax(1)
    per_class = [
        dict(
            class_id=k,
            n=int(labels.eq(k).sum()),
            original_correct=int((before.eq(labels) & labels.eq(k)).sum()),
            protected_correct=int((after.eq(labels) & labels.eq(k)).sum()),
        )
        for k in range(len(contrast))
    ]
    cal = dict(
        **cache_metadata,
        null_basis=null,
        projection_rank=rank,
        direction=direction,
        desired=desired,
        requested_class=requested_class,
        collapse_class=collapse_class,
        class_bias=class_bias,
        first_bound=first_bound,
        amplitude=amplitude,
        delta=delta,
    )
    geometry = dict(
        noise_family="projection_bias",
        edited_key=key,
        projection_rank=rank,
        exact_complement_dimension=null.shape[1],
        noise_seed=cfg["noise_seed"],
        requested_class=requested_class,
        proxy_collapse_class=collapse_class,
        amplitude=amplitude,
        delta_norm=float(actual.norm()),
        ideal_embedding_null_error=float((direction @ projection).norm()),
        fp32_embedding_constant_error=float((actual @ projection).norm()),
        partner_used=False,
        labels_used_in_direction=False,
        labels_used_in_amplitude=False,
        optimizer_steps=0,
        parameter_search=False,
        proxy_fit=dict(
            n=len(labels),
            original_correct=int(before.eq(labels).sum()),
            protected_correct=int(after.eq(labels).sum()),
            per_class=per_class,
        ),
        limitation="Author AGA does not edit final LN beta or visual projection. No claim of resistance to public nullspace cleanup or parameter replacement.",
    )
    return released, {key: released.state_dict()[key].clone()}, cal, geometry
