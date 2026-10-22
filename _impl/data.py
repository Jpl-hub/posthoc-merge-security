"""Official data splits and independently parsed labels."""

import struct
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from torch.utils.data import DataLoader, Subset


def dataset(task, root, transform, training=False):
    from torchvision.datasets import DTD, MNIST, SVHN

    if task == "MNIST":
        return MNIST(root, train=training, transform=transform)
    if task == "DTD":
        return DTD(root, split="train" if training else "test", partition=1, transform=transform)
    if task == "SVHN":
        return SVHN(root, split="train" if training else "test", transform=transform)
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
    labels = loadmat("data/train_32x32.mat" if training else "data/test_32x32.mat")["y"].reshape(-1)
    return np.where(labels == 10, 0, labels).astype(np.int64)
