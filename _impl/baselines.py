"""Pinned author protection implementations and native CLIP adapters."""

import importlib.util
import os
import sys

import torch

from .io import digest


@torch.no_grad()
def mergelock(original, settings):
    """MergeLock Sec.3.4, with the unspecified R/D sampler stated in settings.

    For row features use Q'=QA, K'=KA^{-T}, V'=VB, O'=B^{-1}O.
    PyTorch stores linear weights transposed; biases follow their own branch.
    This is a paper implementation, not code released by the authors.
    """
    state = original.state_dict() if hasattr(original, "state_dict") else original
    generator = torch.Generator().manual_seed(settings["seed"])
    heads = settings["heads"]
    low, high = settings["diagonal_range"]
    if settings["random_matrix"] != "standard_normal" or not 0 < low < high:
        raise ValueError("Unsupported MergeLock sampler")
    conditions, residuals, edits = [], [], {}

    def sample(d):
        r = torch.randn(d, d, generator=generator, dtype=torch.float64)
        p = torch.randperm(d, generator=generator)
        scale = low + (high - low) * torch.rand(d, generator=generator, dtype=torch.float64)
        matrix = r[:, p] * scale
        inverse = torch.linalg.inv(matrix)
        conditions.append(float(torch.linalg.cond(matrix)))
        residuals.append(float((matrix @ inverse - torch.eye(d)).abs().max()))
        return matrix, inverse

    prefixes = sorted(k.removesuffix("in_proj_weight") for k in state
                      if k.endswith("attn.in_proj_weight"))
    if not prefixes:
        raise ValueError("No fused QKV attention weights found")
    for prefix in prefixes:
        w_key, b_key, o_key = (prefix + s for s in
                              ["in_proj_weight", "in_proj_bias", "out_proj.weight"])
        weight, bias, output = (state[k].double().clone() for k in [w_key, b_key, o_key])
        width = output.shape[0]
        if width % heads or weight.shape != (3 * width, width):
            raise ValueError("Unexpected attention dimensions")
        d = width // heads
        wq, wk, wv = weight.chunk(3)
        bq, bk, bv = bias.chunk(3)
        for h in range(heads):
            sl = slice(h * d, (h + 1) * d)
            a, ai = sample(d)
            b, bi = sample(d)
            wq[sl], bq[sl] = a.T @ wq[sl], a.T @ bq[sl]
            wk[sl], bk[sl] = ai @ wk[sl], ai @ bk[sl]
            wv[sl], bv[sl] = b.T @ wv[sl], b.T @ bv[sl]
            output[:, sl] = output[:, sl] @ bi.T
        for key, value in [(w_key, weight), (b_key, bias), (o_key, output)]:
            edits[key] = value.to(state[key].dtype)
    if not all(torch.isfinite(v).all() for v in edits.values()):
        raise ValueError("Non-finite MergeLock weights")
    return edits, dict(implementation="paper adaptation", settings=settings,
                       attention_layers=len(prefixes), matrices=len(conditions),
                       maximum_condition_number=max(conditions),
                       maximum_float64_inverse_residual=max(residuals),
                       construction_images=0, training_steps=0)


def trap2_training(source_files):
    """Load the unchanged author training functions without unused HF/LoRA imports."""
    import ast
    from pathlib import Path
    from types import SimpleNamespace

    import numpy as np
    from torch.cuda.amp import GradScaler, autocast
    from torch.nn.utils.stateless import functional_call
    from tqdm import tqdm

    def save_model(model, path):
        # Native visual state only; no LoRA adapter directory is required.
        state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Non-finite trained baseline")
        torch.save(state, path)

    env = dict(torch=torch, np=np, Path=Path, tqdm=tqdm,
               GradScaler=GradScaler, autocast=autocast,
               CrossEntropyLoss=torch.nn.CrossEntropyLoss,
               functional_call=functional_call, save_model=save_model,
               wandb=SimpleNamespace(run=None))
    names = {
        "utils.py": {"assign_learning_rate", "_warmup_lr", "cosine_lr",
                     "get_device", "evaluate_cliphead"},
        "8vision_training_full_trap2.py": {
            "EarlyStopper", "_save_best_performance", "_to_device",
            "train_cliphead_full_trap2"},
    }
    for path, expected in source_files.items():
        if digest(path) != expected:
            raise ValueError(f"Trap2 source SHA256 mismatch: {path}")
        tree = ast.parse(Path(path).read_text())
        wanted = names[Path(path).name]
        tree.body = [n for n in tree.body
                     if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in wanted]
        assert {n.name for n in tree.body} == wanted
        exec(compile(tree, path, "exec"), env)
    return env


def params_default(original, anchor, source, source_sha256):
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    if digest(source) != source_sha256:
        raise ValueError("PaRaMS source SHA256 mismatch")
    spec = importlib.util.spec_from_file_location("official_params_reference", source)
    author = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = author
    spec.loader.exec_module(author)
    with torch.no_grad():
        state = author.params(
            original.state_dict(),
            anchor.state_dict(),
            num_layers=12,
            enable_permutation=True,
            enable_scaling=True,
            perm_config=None,
            scaling_config=None,
            rng_seed=0,
        )
    if not all(torch.isfinite(v).all() for v in state.values()):
        raise ValueError("Non-finite PaRaMS state")
    return {k: v for k, v in state.items() if not torch.equal(v, original.state_dict()[k])}


def verify_edits(original, edits, seed=0):
    """Independent exact MLP row lookup and NumPy diagonal-scaling identities."""
    import numpy as np

    state = original.state_dict()
    changed = state | edits
    generator = torch.Generator().manual_seed(seed)
    allowed = set()
    for i in range(12):
        p = f"model.visual.transformer.resblocks.{i}."
        a = state[p + "mlp.c_fc.weight"].numpy()
        lookup = {row.tobytes(): j for j, row in enumerate(a)}
        assert len(lookup) == len(a)
        perm = np.array([lookup[row.tobytes()] for row in changed[p + "mlp.c_fc.weight"].numpy()])
        assert len(np.unique(perm)) == len(a)
        for key, expected in [
            ("mlp.c_fc.weight", a[perm]),
            ("mlp.c_fc.bias", state[p + "mlp.c_fc.bias"].numpy()[perm]),
            ("mlp.c_proj.weight", state[p + "mlp.c_proj.weight"].numpy()[:, perm]),
        ]:
            np.testing.assert_array_equal(changed[p + key], expected)
            allowed.add(p + key)
        alpha = (0.05 + 19.95 * torch.rand(768, generator=generator)).numpy()
        inverse = np.float32(1) / alpha
        for key in ["attn.in_proj_weight", "attn.in_proj_bias"]:
            w = state[p + key].numpy()
            scale = np.concatenate([alpha, inverse, alpha])
            scale = scale[:, None] if w.ndim == 2 else scale
            np.testing.assert_array_equal(changed[p + key], w * scale)
            allowed.add(p + key)
        key = p + "attn.out_proj.weight"
        np.testing.assert_array_equal(changed[key], state[key].numpy() * inverse[None])
        allowed.add(key)
    assert set(edits) <= allowed
    for key in set(state) - allowed:
        np.testing.assert_array_equal(changed[key], state[key])
    return dict(
        exact_MLP_row_permutation_and_numpy_scaling=True,
        layers=12,
        all_state_tensors_checked=len(state),
    )


def arf_distance_penalty(state, anchor, rho):
    """ARF prose interpretation: sum relu(margin-distance), not printed Eq.19 min.

    Each layer contributes four full matrices; fused QKV is split before taking
    flattened Euclidean (Frobenius) norms. Biases and other parameters are excluded.
    Pass named_parameters(), not a detached state_dict(), when gradients are needed.
    """
    if rho <= 0:
        raise ValueError("ARF rho must be positive")
    prefixes = sorted(k.removesuffix("in_proj_weight") for k in state
                      if k.endswith("attn.in_proj_weight"))
    if not prefixes:
        raise ValueError("ARF requires fused QKV attention weights")
    distances, margins = {}, {}
    for prefix in prefixes:
        qkv, output = (prefix + suffix for suffix in ["in_proj_weight", "out_proj.weight"])
        width = state[output].shape[0]
        if state[qkv].shape != (3 * width, width) or state[output].shape != (width, width):
            raise ValueError("Unexpected ARF attention dimensions")
        if any(state[key].shape != anchor[key].shape for key in [qkv, output]):
            raise ValueError("ARF anchor dimensions differ")
        weights = [*state[qkv].chunk(3), state[output]]
        anchors = [*anchor[qkv].detach().chunk(3), anchor[output].detach()]
        for branch, weight, reference in zip("QKVO", weights, anchors):
            name = prefix + branch
            distances[name] = torch.linalg.vector_norm(weight - reference)
            margins[name] = rho * torch.linalg.vector_norm(reference)
    loss = sum((margins[name] - distance).relu()
               for name, distance in distances.items())
    return loss, distances, margins


def mergeguard_training(visual, base, train_loader, val_loader, vectors, settings, cfg, out, defense):
    """Shared fixed training loop for MergeGuard/ARF and matched CE controls."""
    import math
    import time
    import torch.nn.functional as F
    from .io import save_json

    visual = visual.cuda().float().train()
    vectors = vectors.cuda().float()
    params = list(visual.parameters())
    is_arf = cfg.get("method") == "ARF"
    if is_arf:
        if cfg["arf"]["formulation"] != "prose_hinge_max":
            raise ValueError("Only the explicitly declared ARF prose interpretation is implemented")
        anchor = {k: v.detach().cuda().float() for k, v in base.items()
                  if k.endswith(("attn.in_proj_weight", "attn.out_proj.weight"))}

        @torch.no_grad()
        def distance_summary():
            penalty, distances, margins = arf_distance_penalty(
                dict(visual.named_parameters()), anchor, cfg["arf"]["rho"])
            if len(distances) != 4 * cfg["arf"]["attention_layers"]:
                raise ValueError("ARF attention layer count differs from protocol")
            return dict(penalty=float(penalty), matrices={name: dict(
                distance=float(distance), margin=float(margins[name]),
                distance_to_anchor_norm=float(distance / (margins[name] / cfg["arf"]["rho"])),
                margin_satisfied=bool(distance >= margins[name]))
                for name, distance in distances.items()})

        initial_distances = distance_summary()
    optimizer = torch.optim.AdamW(params, lr=settings["lr"], weight_decay=0.0)
    accumulation = cfg.get("gradient_accumulation", 1)
    steps_per_epoch = math.ceil(len(train_loader) / accumulation)
    total = settings["epochs"] * steps_per_epoch
    started = time.monotonic()
    for epoch in range(settings["epochs"]):
        sums = [0., 0.]
        for batch_index, (images, labels) in enumerate(train_loader):
            step = epoch * steps_per_epoch + batch_index // accumulation
            warm = settings["warm_up"]
            fraction = ((step + 1) / warm if step < warm else
                        .5 * (1 + math.cos(math.pi * (step - warm) / (total - warm))))
            for group in optimizer.param_groups:
                group["lr"] = settings["lr"] * fraction
            if batch_index % accumulation == 0:
                optimizer.zero_grad(set_to_none=True)
                group_samples = min(accumulation * train_loader.batch_size,
                                    len(train_loader.dataset) - batch_index * train_loader.batch_size)
            logits = 100 * F.normalize(visual(images.cuda()), dim=-1) @ vectors.T
            ce = F.cross_entropy(logits, labels.cuda())
            if is_arf:
                regularizer = (arf_distance_penalty(dict(visual.named_parameters()), anchor,
                                                   cfg["arf"]["rho"])[0]
                               if settings["lambda_reg"] else ce.new_zeros(()))
            else:
                # MergeGuard Eq.(2): squared weights, not a mean or anchor distance.
                regularizer = (sum(p.square().sum() for p in params)
                               if settings["lambda_reg"] else ce.new_zeros(()))
            loss = ce + settings["lambda_reg"] * regularizer
            if not torch.isfinite(loss):
                raise ValueError("Non-finite protection baseline training loss")
            if accumulation != 1:
                loss = loss * len(labels) / group_samples
            loss.backward()
            if (batch_index + 1) % accumulation == 0 or batch_index + 1 == len(train_loader):
                torch.nn.utils.clip_grad_norm_(params, settings["max_norm"])
                optimizer.step()
                if accumulation > 1 and (step == 0 or (step + 1) % 100 == 0):
                    print(dict(defense=defense, step=step + 1, total_steps=total,
                               ce=ce.item(), seconds=time.monotonic()-started), flush=True)
            sums[0] += ce.item()
            sums[1] += regularizer.item()
        print(dict(defense=defense, epoch=epoch + 1, epochs=settings["epochs"],
                   mean_ce=sums[0]/len(train_loader), mean_regularizer=sums[1]/len(train_loader),
                   seconds=time.monotonic()-started), flush=True)
    del optimizer
    visual.requires_grad_(False).eval()

    if is_arf:
        save_json(out / "arf_distances.json", dict(
            formulation=cfg["arf"]["formulation"], lambda_dist=settings["lambda_reg"],
            rho=cfg["arf"]["rho"], initial=initial_distances, final=distance_summary()))
        state = {k: v.detach().cpu() for k, v in visual.state_dict().items()}
        if defense == "ARF":
            # Retain the trained model before the complete ARF protection stage.
            torch.save(state, out / "stage1.pt")
            edits, details = mergelock(state, cfg["mergelock"])
            state.update(edits)
            save_json(out / "attention_protection.json", details | {
                "stage1_sha256": digest(out / "stage1.pt"),
                "role": "ARF post-training invertible attention protection; explicit sampler choice"})
        if not all(torch.isfinite(v).all() for v in state.values()):
            raise ValueError("Non-finite ARF or matched CE checkpoint")
        torch.save(state, out / "best.pt")
        return

    if defense == "MergeGuard":
        torch.save({k: v.detach().cpu() for k, v in visual.state_dict().items()}, out / "stage1.pt")
        mergeguard_negation(visual, base, val_loader, vectors, cfg, out)
    torch.save({k: v.cpu() for k, v in visual.state_dict().items()}, out / "best.pt")


@torch.no_grad()
def mergeguard_negation(visual, base, val_loader, vectors, cfg, out):
    """MergeGuard stage two, also usable with an existing stage-one checkpoint."""
    import math
    import torch.nn.functional as F
    from .io import save_json

    visual = visual.cuda().float().eval()
    vectors = vectors.cuda().float()

    def accuracy():
        correct = 0
        for images, labels in val_loader:
            logits = F.normalize(visual(images.cuda()), dim=-1) @ vectors.T
            correct += (logits.argmax(-1).cpu() == labels).sum().item()
        return correct / len(val_loader.dataset)

    before = accuracy()
    stage1_norm = sum(p.square().sum().item() for p in visual.parameters())
    state = visual.state_dict()
    groups = {}
    for name in state:
        parts = name.split('.')
        group = '.'.join(parts[:3]) if name.startswith('transformer.resblocks.') else parts[0]
        groups.setdefault(group, []).append(name)
    scope = cfg.get("critical_candidates", "visual_groups")
    if scope not in {"visual_groups", "transformer_blocks"}:
        raise ValueError("Unknown MergeGuard critical-layer grouping")
    candidates = [g for g in groups if scope == "visual_groups" or g.startswith("transformer.resblocks.")]
    if not candidates:
        raise ValueError("No candidate layers for MergeGuard")
    importance = {}
    for group in candidates:
        names = groups[group]
        saved = {n: state[n].clone() for n in names}
        for name in names:
            state[name].zero_()
        importance[group] = before - accuracy()
        for name in names:
            state[name].copy_(saved[name])
        print(dict(masked_group=group, accuracy_drop=importance[group]), flush=True)
    critical = sorted(candidates, key=lambda k: (-importance[k], k))[
        :math.ceil(cfg["critical_fraction"] * len(candidates))]
    # ViT perturbation includes parameters outside Transformer blocks as well.
    names = [n for group in groups if group not in critical for n in groups[group]]
    weights = torch.cat([state[n].abs().flatten() for n in names])
    count = math.ceil(cfg["perturbed_fraction"] * weights.numel())
    mask = torch.zeros_like(weights, dtype=torch.bool)
    mask[torch.topk(weights, count, sorted=False).indices] = True
    offset = 0
    edited = 0
    for name in names:
        n = state[name].numel()
        select = mask[offset:offset+n].reshape(state[name].shape)
        original = state[name].clone()
        state[name][select] -= cfg["beta"] * (state[name] - base[name].cuda())[select]
        edited += (state[name] != original).sum().item()
        offset += n
    save_json(out / "mask.json", dict(
        validation_before=before, validation_after=accuracy(),
        groups=groups, critical_candidates=scope, accuracy_drops=importance, critical_groups=critical,
        selected_coordinates=count, changed_coordinates=edited,
        stage1_squared_norm=stage1_norm,
        selection="Global top absolute stage1 weights outside critical groups"))
