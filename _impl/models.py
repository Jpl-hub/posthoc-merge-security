"""Frozen CLIP checkpoint loading and changed-tensor publications."""

import copy
import sys
import types
from pathlib import Path

import torch

from cfg import AUTHOR, CHECKPOINTS, ROOT

from .io import digest, read_json


def checkpoint_loader(author_repo: Path):
    """Register the upstream pickle's legacy src.modeling name without editing it."""
    sys.path.insert(0, str(author_repo.resolve()))
    import modeling
    from utils import load_model_checkpoint

    package = types.ModuleType("src")
    package.__path__ = []
    sys.modules.setdefault("src", package)
    sys.modules.setdefault("src.modeling", modeling)
    return load_model_checkpoint


def initialize():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def load_checkpoint(name):
    pins = {
        x["path"].removeprefix("ViT-B-32/"): x["sha256"]
        for x in read_json(ROOT / "configs" / "checkpoints.json")["files"]
    }
    path = CHECKPOINTS / name
    if digest(path) != pins[name]:
        raise ValueError(f"Checkpoint SHA256 mismatch: {name}")
    model = checkpoint_loader(AUTHOR)(str(path)).cpu().eval()
    model.requires_grad_(False)
    return model


def checked_models(target):
    names = ["zeroshot.pt", f"{target}/finetuned.pt", f"head_{target}.pt"]
    values = [load_checkpoint(n) for n in names]
    return (*values, {n: digest(CHECKPOINTS / n) for n in names})


def apply(original, ref, full=False):
    if digest(ref["path"]) != ref["sha256"]:
        raise ValueError(f"Publication SHA256 mismatch: {ref['path']}")
    state = torch.load(ref["path"], map_location="cpu", weights_only=False)
    result = copy.deepcopy(original)
    result.load_state_dict(state if full else original.state_dict() | state, strict=True)
    result.requires_grad_(False)
    return result
