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
import tarfile
import time
import zipfile
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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


def prepare_cars(root, author_repo):
    """Use a pinned image mirror, verify Git blobs and the original devkit."""
    repo = "jhpohovey/StanfordCars-Dataset"
    revision = "c7421951b5a2292c8951ef69394231be7fc32474"
    resources = root / "stanford_cars_source"
    resources.mkdir(parents=True, exist_ok=True)
    manifest = resources / "git_tree.json"
    if not manifest.exists():
        url = f"https://api.github.com/repos/{repo}/git/trees/{revision}?recursive=1"
        with urlopen(url, timeout=30) as response:
            manifest.write_bytes(response.read())
    tree = json.loads(manifest.read_bytes())
    assert not tree["truncated"]
    entries = [
        e for e in tree["tree"] if e["type"] == "blob" and e["path"].startswith("stanford_cars/")
    ]
    assert len(entries) > 16185
    archive = resources / "cars_mirror.zip"
    extracted = resources / "extracted"
    if not extracted.exists():
        assert shutil.disk_usage(resources).free > 6 * 1024**3
        download_url(
            f"https://codeload.github.com/{repo}/zip/{revision}",
            str(resources),
            filename=archive.name,
        )
        with zipfile.ZipFile(archive) as zipped:
            for member in zipped.infolist():
                assert (extracted / member.filename).resolve().is_relative_to(extracted.resolve())
                assert not stat.S_ISLNK(member.external_attr >> 16)
            extracted.mkdir()
            zipped.extractall(extracted)
    source = extracted / f"StanfordCars-Dataset-{revision}"
    for entry in entries:
        payload = (source / entry["path"]).read_bytes()
        assert len(payload) == entry["size"]
        git_blob = hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()
        assert git_blob == entry["sha"], entry["path"]
    devkit = resources / "car_devkit.tgz"
    download_url(
        "https://github.com/pytorch/vision/files/11644847/car_devkit.tgz",
        str(resources),
        filename=devkit.name,
        md5="c3b158d763b6e2245038c8ad08e45376",
    )
    with tarfile.open(devkit) as tar:
        for name in ["cars_meta.mat", "cars_train_annos.mat", "cars_test_annos.mat"]:
            assert (
                tar.extractfile(f"devkit/{name}").read()
                == (source / "stanford_cars/devkit" / name).read_bytes()
            )
    destination = root / "stanford_cars"
    if not destination.exists():
        destination.symlink_to((source / "stanford_cars").resolve(), target_is_directory=True)
    assert destination.resolve() == (source / "stanford_cars").resolve()
    labelled_test = destination / "devkit/cars_test_annos_withlabels.mat"
    if not labelled_test.exists():
        labelled_test.symlink_to((destination / "cars_test_annos_withlabels.mat").resolve())
    source_file = author_repo / "datasets/cars.py"
    spec = importlib.util.spec_from_file_location("author_cars_data", source_file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    datasets = {s: module.PytorchStanfordCars(str(root), split=s) for s in ["train", "test"]}
    assert len(datasets["train"]) == 8144 and len(datasets["test"]) == 8041
    assert datasets["train"].classes == datasets["test"].classes
    assert len(datasets["test"].classes) == 196
    pairs = [(Path(p).name, int(y)) for p, y in datasets["test"]._samples]
    assert len({p for p, _ in pairs}) == 8041 and {y for _, y in pairs} == set(range(196))
    assert all(Path(p).is_file() for d in datasets.values() for p, _ in d._samples)
    return dict(
        task="Cars",
        status="ready",
        train_count=8144,
        test_count=8041,
        source="pinned_public_image_mirror_original_annotations",
        mirror_repository=repo,
        mirror_revision=revision,
        verified_git_blobs=len(entries),
        devkit_original_md5_verified=True,
        author_source_sha256=hashlib.sha256(source_file.read_bytes()).hexdigest(),
        tree_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        test_annotation_sha256=hashlib.sha256(labelled_test.read_bytes()).hexdigest(),
        test_index_sha256=hashlib.sha256(json.dumps(pairs).encode()).hexdigest(),
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


def prepare_sun397_indices(root, author_repo):
    """Verify a pinned copy of the original named partitions, without images."""
    revision = "99bbf6ad18ca4f01e47f8f96ca1a8b7e53cb235a"
    url = f"https://huggingface.co/zhengli97/SUN397_Dataset/resolve/{revision}/Partitions.zip"
    expected_sha = "e3311ea5f812d4cc6faef194688343d1111f6cd6f70f2c62f393c2b8d6ba1ec9"
    folder = root / "SUN397_sources"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "Partitions.zip"
    download_url(url, str(folder), filename=path.name)
    assert path.stat().st_size == 18293705
    assert hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha
    records, split_sets = [], []
    with zipfile.ZipFile(path) as archive:
        classes = archive.read("ClassName.txt").decode().splitlines()
        assert len(classes) == len(set(classes)) == 397
        for split, name in [("train", "Training_01.txt"), ("test", "Testing_01.txt")]:
            payload = archive.read(name)
            names = payload.decode().splitlines()
            assert len(names) == len(set(names)) == 19850
            assert all(n.startswith("/") and ".." not in Path(n).parts for n in names)
            counts = Counter(str(Path(n).parent) for n in names)
            assert set(counts) == set(classes) and set(counts.values()) == {50}
            destination = folder / name
            if destination.exists():
                assert destination.read_bytes() == payload
            else:
                destination.write_bytes(payload)
            records.append(
                dict(split=split, count=len(names), sha256=hashlib.sha256(payload).hexdigest())
            )
            split_sets.append(set(names))
    assert not split_sets[0].intersection(split_sets[1])
    return dict(
        task="SUN397",
        status="indices_ready_images_missing",
        dataset_ready=False,
        partition=1,
        source="pinned_mirror_of_named_original_partitions",
        mirror_revision=revision,
        archive_url=url,
        archive_sha256=expected_sha,
        digest_provenance="mirror_LFS_SHA256; not a separately published original checksum",
        classes_sha256=hashlib.sha256("\n".join(classes).encode()).hexdigest(),
        author_source_sha256=hashlib.sha256(
            (author_repo / "datasets/sun397.py").read_bytes()
        ).hexdigest(),
        splits=records,
        image_download="not_performed",
        calibration_usage="none",
    )


class VerifiedRangeReader:
    """Bounded HTTP ranges; hash the complete stream without caching the archive."""

    def __init__(self, url, length, chunk_bytes=64 * 1024 * 1024):
        self.url, self.length, self.chunk_bytes = url, length, chunk_bytes
        self.offset, self.buffer = 0, b""
        self.buffer_position = 0
        self.sha256, self.md5 = hashlib.sha256(), hashlib.md5()

    def read(self, size):
        assert size >= 0
        if not size or self.offset == self.length:
            return b""
        if self.buffer_position == len(self.buffer):
            end = min(self.offset + self.chunk_bytes, self.length) - 1
            for attempt in range(6):
                try:
                    request = Request(self.url, headers={"Range": f"bytes={self.offset}-{end}"})
                    with urlopen(request, timeout=60) as response:
                        assert response.status == 206, "Source must support exact byte ranges"
                        assert response.headers["Content-Range"] == (
                            f"bytes {self.offset}-{end}/{self.length}"
                        ), "Unexpected archive range"
                        data = response.read(end - self.offset + 1)
                    if len(data) != end - self.offset + 1:
                        raise OSError("Incomplete archive range")
                    self.buffer = data
                    self.buffer_position = 0
                    break
                except (OSError, URLError):
                    if attempt == 5:
                        raise
                    time.sleep(min(2**attempt, 30))
        result = self.buffer[self.buffer_position : self.buffer_position + size]
        self.buffer_position += len(result)
        self.offset += len(result)
        self.sha256.update(result)
        self.md5.update(result)
        return result


def prepare_sun397_images(root, indices, progress_path, use_mirror=False):
    """Stream original archive, retain only partition 1, verify before linking."""
    source = root / "SUN397_sources"
    revision = indices["mirror_revision"]
    url = f"https://huggingface.co/zhengli97/SUN397_Dataset/resolve/{revision}/SUN397.tar.gz"
    if use_mirror:
        url = url.replace("https://huggingface.co/", "https://hf-mirror.com/")
    expected_sha = "f404130965a7ad77bed5ececc71e720ab50f3cc1e1bb257257610d38f3b928ec"
    # torchvision 0.18.0 SUN397's original dataset checksum.
    expected_md5 = "8ca2778205c41d23104230ba66911c7a"
    length = 39077296924
    destination = source / "images_f4041309"
    marker = destination / ".verified_archive_sha256"
    assert not marker.exists(), "Archive already prepared; do not repeat the full transfer"
    destination.mkdir(parents=True, exist_ok=True)
    splits = {
        split: (source / filename).read_text().splitlines()
        for split, filename in [("train", "Training_01.txt"), ("test", "Testing_01.txt")]
    }
    wanted = set(splits["train"]) | set(splits["test"])
    classes = sorted({str(Path(name).parent) for name in wanted})
    mapping = {name: name[3:].replace("/", "_") for name in classes}
    assert len(set(mapping.values())) == 397, "Flattened class name collision"
    # The author's ImageFolder sorts directory names; original ClassName order
    # differs for jail/indoor and jail_cell. Never use its line number as label.
    class_order = sorted(mapping.values())
    reader = VerifiedRangeReader(url, length)
    seen, extracted_bytes = set(), 0

    def progress(ready=False):
        progress_path.write_text(
            json.dumps(
                dict(
                    status="verified" if ready else "streaming_unverified",
                    archive_bytes_read=reader.offset,
                    archive_bytes_expected=length,
                    selected_images=len(seen),
                    expected_selected_images=len(wanted),
                    selected_bytes=extracted_bytes,
                    dataset_ready=ready,
                ),
                indent=2,
            )
        )

    progress()
    with tarfile.open(fileobj=reader, mode="r|gz") as archive:
        for member in archive:
            parts = Path(member.name).parts
            if not parts or parts[0] != "SUN397":
                continue
            name = "/" + "/".join(parts[1:])
            if name not in wanted:
                continue
            assert name not in seen and member.isfile(), member.name
            assert 0 < member.size < 100 * 1024 * 1024
            assert extracted_bytes + member.size < 22 * 1024**3, "Selection storage limit"
            assert shutil.disk_usage(source).free > member.size + 2 * 1024**3
            path = (destination / name.lstrip("/")).resolve()
            assert path.is_relative_to(destination.resolve())
            payload = archive.extractfile(member).read()
            assert len(payload) == member.size
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                assert path.read_bytes() == payload, "Existing partial image differs"
            else:
                path.write_bytes(payload)
            seen.add(name)
            extracted_bytes += member.size
            if len(seen) % 100 == 0:
                progress()
    # tar iteration can stop before the compressed stream ends. Hash every byte.
    while reader.read(4 * 1024 * 1024):
        pass
    assert reader.offset == length and reader.sha256.hexdigest() == expected_sha
    assert reader.md5.hexdigest() == expected_md5
    assert seen == wanted
    for split, names in splits.items():
        for name in names:
            image = destination / name.lstrip("/")
            link = root / "sun397" / split / mapping[str(Path(name).parent)] / image.name
            link.parent.mkdir(parents=True, exist_ok=True)
            if link.exists():
                assert link.resolve() == image.resolve()
            else:
                link.symlink_to(image.resolve())
    marker.write_text(expected_sha)
    progress(ready=True)
    return dict(
        **{
            k: v
            for k, v in indices.items()
            if k not in {"status", "dataset_ready", "image_download"}
        },
        status="ready",
        dataset_ready=True,
        image_download="complete_archive_stream_verified_selected_partition1_only",
        image_archive_sha256=expected_sha,
        image_archive_original_md5=expected_md5,
        image_archive_bytes=length,
        image_archive_download_url=url,
        image_count=len(seen),
        selected_bytes=extracted_bytes,
        archive_stored=False,
        class_order="unchanged_author_ImageFolder_sorted_flattened_directory_names",
        class_order_sha256=hashlib.sha256("\n".join(class_order).encode()).hexdigest(),
        train_count=19850,
        test_count=19850,
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
            elif task == "Cars":
                row = prepare_cars(root, Path(args.author_repo))
            elif task == "SUN397":
                row = prepare_sun397_indices(root, Path(args.author_repo))
                if args.sun397_images:
                    row = prepare_sun397_images(
                        root, row, output / "stream_progress.json", use_mirror=args.sun397_mirror
                    )
            else:
                row = prepare_resisc_indices(root, Path(args.author_repo))
                if args.resisc_images:
                    row = prepare_resisc_images(root, row)
        except Exception as exc:
            row = dict(task=task, status="failed", error=str(exc))
            progress = output / "stream_progress.json"
            if task == "SUN397" and args.sun397_images and progress.exists():
                state = json.loads(progress.read_bytes())
                state.update(status="failed", error=str(exc), dataset_ready=False)
                progress.write_text(json.dumps(state, indent=2))
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
        "--tasks",
        nargs="+",
        choices=["DTD", "GTSRB", "RESISC45", "Cars", "SUN397"],
        default=["DTD", "GTSRB"],
    )
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs/2026-10-09/eight_task/data_1810")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--resisc-images", action="store_true", help="Download and verify images.")
    parser.add_argument(
        "--sun397-images", action="store_true", help="Stream original archive, retain partition 1."
    )
    parser.add_argument(
        "--sun397-mirror", action="store_true", help="Use mirror transport with the same checksums."
    )
    main(parser.parse_args())
