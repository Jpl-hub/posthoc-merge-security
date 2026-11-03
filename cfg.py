"""Shared locations and configuration checks; scientific choices stay in JSON."""

import os
from pathlib import Path

from _impl.io import check_files, read_json

ROOT = Path(__file__).resolve().parent
BACKBONE = os.environ.get("POSTHOC_BACKBONE", "ViT-B-32")
if BACKBONE not in {"ViT-B-32", "ViT-B-16"}:
    raise ValueError(f"Unsupported backbone: {BACKBONE}")
CHECKPOINTS = ROOT / "checkpoints" / BACKBONE
CHECKPOINT_MANIFEST = ROOT / "configs" / (
    "checkpoints.json" if BACKBONE == "ViT-B-32" else "checkpoints_vit_b16.json"
)
AUTHOR = ROOT / "external" / "secure-merge-attack"
DATA = ROOT / "data"


def load_config(path):
    cfg = read_json(path)
    if cfg.get("backbone", "ViT-B-32") != BACKBONE:
        raise ValueError("Protocol and selected backbone differ")
    if cfg.get("source_sha256"):
        check_files(cfg["source_sha256"])
    return cfg


def require_root():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run this command from {ROOT}")
