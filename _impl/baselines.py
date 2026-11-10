"""Pinned author protection implementations and native CLIP adapters."""

import importlib.util
import os
import sys

import torch

from .io import digest


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


def mergeguard_training(visual, base, train_loader, val_loader, vectors, settings, cfg, out, defense):
    """Paper-defined CE+L2 and masked task-vector negation; choices are in config."""
    import math
    import time
    import torch.nn.functional as F
    from .io import save_json

    visual = visual.cuda().float().train()
    vectors = vectors.cuda().float()
    params = list(visual.parameters())
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
            # Eq.(2): sum of squared weights, not a mean or distance to the base.
            l2 = sum(p.square().sum() for p in params) if settings["lambda_reg"] else ce.new_zeros(())
            loss = ce + settings["lambda_reg"] * l2
            if not torch.isfinite(loss):
                raise ValueError("Non-finite MergeGuard training loss")
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
            sums[1] += l2.item()
        print(dict(defense=defense, epoch=epoch + 1, epochs=settings["epochs"],
                   mean_ce=sums[0]/len(train_loader), mean_l2=sums[1]/len(train_loader),
                   seconds=time.monotonic()-started), flush=True)
    del optimizer
    visual.requires_grad_(False).eval()

    @torch.no_grad()
    def accuracy():
        correct = 0
        for images, labels in val_loader:
            logits = F.normalize(visual(images.cuda()), dim=-1) @ vectors.T
            correct += (logits.argmax(-1).cpu() == labels).sum().item()
        return correct / len(val_loader.dataset)

    if defense == "MergeGuard":
        before = accuracy()
        stage1_norm = sum(p.square().sum().item() for p in params)
        state = visual.state_dict()
        groups = {}
        for name in state:
            parts = name.split('.')
            group = '.'.join(parts[:3]) if name.startswith('transformer.resblocks.') else parts[0]
            groups.setdefault(group, []).append(name)
        importance = {}
        for group, names in groups.items():
            saved = {n: state[n].clone() for n in names}
            for name in names:
                state[name].zero_()
            importance[group] = before - accuracy()
            for name in names:
                state[name].copy_(saved[name])
        critical = sorted(groups, key=lambda k: (-importance[k], k))[
            :math.ceil(cfg["critical_fraction"] * len(groups))]
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
            groups=groups, accuracy_drops=importance, critical_groups=critical,
            selected_coordinates=count, changed_coordinates=edited,
            stage1_squared_norm=stage1_norm,
            selection="Global top absolute stage1 weights outside critical groups"))
    torch.save({k: v.cpu() for k, v in visual.state_dict().items()}, out / "best.pt")
