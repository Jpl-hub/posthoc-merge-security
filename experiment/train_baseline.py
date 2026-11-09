"""Train Trap2 and a matched CE control with the pinned author training loop."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from _impl.baselines import trap2_training
from _impl.data import dataset, official_labels
from _impl.io import digest, new_output, read_json, save_json
from _impl.models import initialize, load_checkpoint
from cfg import CHECKPOINTS, DATA, require_root


def run(config, task, defense):
    require_root()
    initialize()
    cfg = read_json(config)
    settings = cfg["training"] | {"device": "cuda"}
    settings["lambda_reg"] = cfg["lambda_reg"] if defense == "Trap2" else 0.0
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    author = trap2_training(cfg["author_sources"])
    encoder = load_checkpoint("zeroshot.pt")
    head = load_checkpoint(f"head_{task}.pt")
    train_data = dataset(task, str(DATA), encoder.val_preprocess, training=True)
    all_indices = np.arange(len(train_data))
    if task == "DTD":
        from torchvision.datasets import DTD

        train_indices = all_indices
        val_data = DTD(str(DATA), split="val", partition=1, transform=encoder.val_preprocess)
        val_indices = np.arange(len(val_data))
    else:
        labels = official_labels(task, training=True)
        assert len(labels) == len(train_data)
        generator = np.random.default_rng(seed)
        selected = []
        for label in np.unique(labels):
            indices = generator.permutation(np.flatnonzero(labels == label))
            selected.extend(indices[:int(len(indices) * cfg["validation_fraction"])])
        val_indices = np.sort(selected)
        train_indices = np.setdiff1d(all_indices, val_indices)
        val_data = train_data
        assert not np.intersect1d(train_indices, val_indices).size
    out = new_output(Path(cfg["output"]) / task / defense, cfg | {
        "task": task, "defense": defense, "training": settings,
        "train_indices": train_indices.tolist(), "validation_indices": val_indices.tolist(),
        "base_sha256": digest(CHECKPOINTS / "zeroshot.pt"),
        "head_sha256": digest(CHECKPOINTS / f"head_{task}.pt"),
        "driver_sha256": digest(__file__), "adapter_sha256": digest("_impl/baselines.py"),
    })
    train_loader = DataLoader(Subset(train_data, train_indices.tolist()),
                              batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(val_data, val_indices.tolist()),
                            batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True)
    visual = encoder.model.visual
    visual.requires_grad_(True)
    visual.proj.requires_grad_(False)  # Matches the author's HFCLIPVisionModel.
    vectors = head.weight.detach().float() / 100.0
    assert torch.allclose(vectors.norm(dim=1), torch.ones(len(vectors)), atol=1e-5)
    del head, encoder
    started = time.monotonic()
    # Author loop logs a "Test" column; feed validation twice, never test images.
    author["train_cliphead_full_trap2"](
        visual, train_loader, val_loader, val_loader, vectors, settings, str(out / "best.pt"))
    assert (out / "best.pt").is_file()
    save_json(out / "completed.json", dict(
        task=task, defense=defense, training_seconds=time.monotonic() - started,
        checkpoint_sha256=digest(out / "best.pt"),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        training_examples=len(train_indices), validation_examples=len(val_indices),
        test_examples_used=0,
        evaluation="Pending: best validation checkpoint must be evaluated in FP32.",
    ))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/trap2_baseline.json")
    parser.add_argument("--task", choices=["DTD", "MNIST", "SVHN"], required=True)
    parser.add_argument("--defense", choices=["Trap2", "CE"], required=True)
    args = parser.parse_args()
    run(args.config, args.task, args.defense)
