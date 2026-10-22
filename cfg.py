"""Shared locations and configuration checks; scientific choices stay in JSON."""

from pathlib import Path

from _impl.io import check_files, read_json

ROOT = Path(__file__).resolve().parent
CHECKPOINTS = ROOT / "checkpoints" / "ViT-B-32"
AUTHOR = ROOT / "external" / "secure-merge-attack"
DATA = ROOT / "data"


def load_config(path):
    cfg = read_json(path)
    if cfg.get("source_sha256"):
        check_files(cfg["source_sha256"])
    return cfg


def require_root():
    if Path.cwd().resolve() != ROOT:
        raise RuntimeError(f"Run this command from {ROOT}")
