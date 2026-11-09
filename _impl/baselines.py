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
