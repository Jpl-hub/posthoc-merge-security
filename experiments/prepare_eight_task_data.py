"""Prepare verified data and upstream split resources on the server.

Does not invent EuroSAT/RESISC45 indices or alter any checkpoint. DTD images are
linked, not copied. Preparation manifests preserve failures and sample counts.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import hashlib
import importlib.util
import json
import shutil
import stat
import zipfile
from pathlib import Path
from urllib.error import HTTPError

import torch
from torchvision.datasets import DTD
from torchvision.datasets.utils import check_integrity, download_url


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


def prepare_resisc_indices(root, author_repo):
    source = author_repo / "datasets" / "resisc45.py"
    tree = ast.parse(source.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RESISC45Dataset")
    constants = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in cls.body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in {"split_urls", "split_md5s", "classes"}
    }
    records = []
    index_sets = []
    for split, url in constants["split_urls"].items():
        folder = root / "RESISC45" / split
        name = f"RESISC45-{split}.txt"
        used_url = None
        original_error = None
        cached = check_integrity(str(folder / name), constants["split_md5s"][split])
        if not cached:
            used_url = url
            try:
                download_url(url, str(folder), filename=name, md5=constants["split_md5s"][split])
            except HTTPError as exc:
                original_error = str(exc)
                # Public mirror referenced by TorchGeo; retain the original MD5.
                used_url = f"https://huggingface.co/datasets/isaaccorley/resisc45/resolve/883edc0eee77b2c84225472f10f126e3ed83fa6e/resisc45-{split}.txt"
                download_url(
                    used_url, str(folder), filename=name, md5=constants["split_md5s"][split]
                )
        payload = (folder / name).read_bytes()
        names = payload.decode().splitlines()
        assert len(names) == len(set(names))
        assert all(Path(n).name == n and n.endswith(".jpg") for n in names)
        classes = {n.rsplit("_", 1)[0] for n in names}
        assert classes == set(constants["classes"])
        if split == "test":
            assert len(names) == 6300
        records.append(
            dict(
                split=split,
                count=len(names),
                url=url,
                download_url=used_url,
                original_url_error=original_error,
                acquisition="verified_cache" if cached else "download",
                md5=hashlib.md5(payload).hexdigest(),
                sha256=hashlib.sha256(payload).hexdigest(),
                upstream_md5_verified=True,
            )
        )
        index_sets.append(set(names))
    assert all(not a.intersection(b) for i, a in enumerate(index_sets) for b in index_sets[i + 1 :])
    assert sum(map(len, index_sets)) == 31500
    return dict(
        task="RESISC45",
        status="indices_ready_images_missing",
        dataset_ready=False,
        source="fixed_AGA_dataset_split_urls",
        source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        splits=records,
        image_download="not_performed",
        calibration_usage="none",
    )


def prepare_resisc_images(root, indices):
    """Verify TorchGeo's pinned public archive, then link original author splits."""
    url = "https://huggingface.co/datasets/isaaccorley/resisc45/resolve/883edc0eee77b2c84225472f10f126e3ed83fa6e/NWPU-RESISC45.zip"
    expected_sha = "beeecd0b63656290ae6d65cf7763185b0c1c4c54a753ef8088d6fba3faaf1f53"
    resources = root / "RESISC45" / "source"
    resources.mkdir(parents=True, exist_ok=True)
    archive_path = resources / "NWPU-RESISC45.zip"
    download_url(url, str(resources), filename=archive_path.name)
    digest = hashlib.sha256()
    with archive_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    assert digest.hexdigest() == expected_sha, "RESISC45 archive SHA256 mismatch"
    images_root = (resources / "images").resolve()
    images_root.mkdir(exist_ok=True)
    marker = images_root / ".archive_sha256"
    if not marker.exists():
        with zipfile.ZipFile(archive_path) as archive:
            members = archive.infolist()
            assert sum(m.file_size for m in members) < shutil.disk_usage(images_root).free
            for member in members:
                destination = (images_root / member.filename).resolve()
                assert destination.is_relative_to(images_root), member.filename
                assert not stat.S_ISLNK(member.external_attr >> 16), member.filename
            archive.extractall(images_root)
        marker.write_text(expected_sha)
    assert marker.read_text() == expected_sha
    files = list(images_root.rglob("*.jpg"))
    by_name = {p.name: p for p in files}
    assert len(files) == len(by_name) == 31500
    all_names = set()
    for record in indices["splits"]:
        split = record["split"]
        folder = root / "RESISC45" / split
        names = (folder / f"RESISC45-{split}.txt").read_text().splitlines()
        assert len(names) == record["count"]
        for name in names:
            source = by_name[name]
            label = name.rsplit("_", 1)[0]
            assert source.parent.name == label
            link = folder / label / name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists():
                assert link.resolve() == source, link
            else:
                link.symlink_to(source)
        all_names.update(names)
    assert all_names == set(by_name)
    return dict(
        **{
            k: v
            for k, v in indices.items()
            if k not in {"status", "dataset_ready", "image_download"}
        },
        status="ready",
        dataset_ready=True,
        image_download="verified_public_mirror_archive",
        archive_url=url,
        archive_sha256=expected_sha,
        archive_bytes=archive_path.stat().st_size,
        archive_digest_source="https://github.com/torchgeo/torchgeo/blob/main/torchgeo/datasets/resisc45.py",
        image_count=len(files),
        test_count=6300,
        layout="original_author_split_class_folders_with_symlinks",
        test_index_sha256=next(r["sha256"] for r in indices["splits"] if r["split"] == "test"),
    )


def main(args):
    torch.set_num_threads(4)
    root, output = Path(args.data).resolve(), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)

    def prepare(task):
        try:
            if task == "DTD":
                row = prepare_dtd(root)
            elif task == "GTSRB":
                row = prepare_gtsrb(root, Path(args.author_repo))
            else:
                row = prepare_resisc_indices(root, Path(args.author_repo))
                if args.resisc_images:
                    row = prepare_resisc_images(root, row)
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
    parser.add_argument(
        "--tasks", nargs="+", choices=["DTD", "GTSRB", "RESISC45"], default=["DTD", "GTSRB"]
    )
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs/2026-10-09/eight_task/data_1810")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--resisc-images", action="store_true", help="Download and verify images.")
    main(parser.parse_args())
