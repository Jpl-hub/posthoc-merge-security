"""Prepare verified DTD split 1 and original GTSRB wrappers on the server.

Does not invent EuroSAT/RESISC45 indices or alter any checkpoint. DTD images are
linked, not copied. Preparation manifests preserve failures and sample counts.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
from pathlib import Path

import torch
from torchvision.datasets import DTD


def prepare_dtd(root):
    sets = {
        s: DTD(str(root), split=s, partition=1, download=True) for s in ["train", "val", "test"]
    }
    for split, dataset in sets.items():
        assert len(dataset) == 1880
        destination = root / "DTD" / split
        for source in dataset._image_files:
            source = Path(source).resolve()
            link = destination / source.parent.name / source.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists():
                assert link.resolve() == source, link
            else:
                link.symlink_to(source)
    test = sets["test"]
    paths = [Path(p).relative_to(root).as_posix() for p in test._image_files]
    return dict(
        task="DTD",
        status="ready",
        test_count=len(test),
        partition=1,
        test_index_sha256=hashlib.sha256("\n".join(paths).encode()).hexdigest(),
        source="torchvision_DTD_official_test1",
        layout="author_ImageFolder_with_symlinks",
        classes=test.classes,
        calibration_usage="none",
    )


def prepare_gtsrb(root, author_repo):
    source = author_repo / "datasets" / "gtsrb.py"
    spec = importlib.util.spec_from_file_location("author_gtsrb_data", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    datasets = {
        s: module.PyTorchGTSRB(str(root), split=s, download=True) for s in ["train", "test"]
    }
    assert len(datasets["test"]) == 12630
    pairs = [(Path(p).relative_to(root).as_posix(), int(y)) for p, y in datasets["test"]._samples]
    return dict(
        task="GTSRB",
        status="ready",
        test_count=len(pairs),
        train_count=len(datasets["train"]),
        test_index_sha256=hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
        source="original_AGA_PyTorchGTSRB_official_images_and_labels",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        calibration_usage="none",
    )


def main(args):
    torch.set_num_threads(4)
    root, output = Path(args.data).resolve(), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    def prepare(task):
        try:
            row = (
                prepare_dtd(root) if task == "DTD" else prepare_gtsrb(root, Path(args.author_repo))
            )
        except Exception as exc:
            row = dict(task=task, status="failed", error=str(exc))
        print("DATA_PREPARATION", json.dumps(row), flush=True)
        return row

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        rows = list(pool.map(prepare, args.tasks))
    (output / "data_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if any(r["status"] != "ready" for r in rows):
        raise SystemExit("Some data remain unavailable; manifest preserves failures")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=["DTD", "GTSRB"], default=["DTD", "GTSRB"])
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs/2026-10-09/eight_task/data_1810")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    main(parser.parse_args())
