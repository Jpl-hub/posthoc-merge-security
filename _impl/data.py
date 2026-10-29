"""Official data splits and independently parsed labels."""

import csv
import struct
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader, Subset


def dataset(task, root, transform, training=False):
    from torchvision.datasets import DTD, GTSRB, MNIST, SVHN, ImageFolder

    if task == "MNIST":
        return MNIST(root, train=training, transform=transform)
    if task == "DTD":
        return DTD(root, split="train" if training else "test", partition=1, transform=transform)
    if task == "SVHN":
        return SVHN(root, split="train" if training else "test", transform=transform)
    if task == "GTSRB":
        return GTSRB(root, split="train" if training else "test", transform=transform)
    if task == "RESISC45":
        split = "train" if training else "test"
        folder = Path(root) / "RESISC45" / split
        names = set((folder / f"RESISC45-{split}.txt").read_text().splitlines())
        return ImageFolder(
            folder, transform=transform, is_valid_file=lambda p: Path(p).name in names
        )
    raise ValueError(task)


def make_loader(dataset, indices=None, batch_size=128):
    if indices is not None:
        dataset = Subset(dataset, indices)
    return DataLoader(dataset, batch_size=batch_size, num_workers=2, pin_memory=True)


def official_labels(task, training=False):
    if task == "MNIST":
        name = "train" if training else "t10k"
        n = 60000 if training else 10000
        raw = Path(f"data/MNIST/raw/{name}-labels-idx1-ubyte").read_bytes()
        assert struct.unpack(">II", raw[:8]) == (2049, n)
        return np.frombuffer(raw[8:], dtype=np.uint8).astype(np.int64)
    if task == "DTD":
        root = Path("data/dtd/dtd")
        classes = sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
        names = (
            (root / "labels" / ("train1.txt" if training else "test1.txt")).read_text().splitlines()
        )
        return np.array([classes.index(name.split("/")[0]) for name in names])
    if task == "SVHN":
        labels = loadmat("data/train_32x32.mat" if training else "data/test_32x32.mat")[
            "y"
        ].reshape(-1)
        return np.where(labels == 10, 0, labels).astype(np.int64)
    if task == "GTSRB":
        if training:
            paths = sorted(Path("data/gtsrb/GTSRB/Training").glob("*/*.ppm"))
            return np.array([int(p.parent.name) for p in paths])
        with Path("data/gtsrb/GT-final_test.csv").open() as source:
            return np.array([int(r["ClassId"]) for r in csv.DictReader(source, delimiter=";")])
    if task == "RESISC45":
        split = "train" if training else "test"
        root = Path("data/RESISC45") / split
        classes = sorted(p.name for p in root.iterdir() if p.is_dir())
        names = (root / f"RESISC45-{split}.txt").read_text().splitlines()
        return np.array([classes.index(name.rsplit("_", 1)[0]) for name in sorted(names)])
    raise ValueError(task)
