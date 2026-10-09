"""Download only selected official Task Arithmetic checkpoints onto server."""

import argparse
import concurrent.futures
import hashlib
import json
import time
from pathlib import Path

import gdown


def download(e, root):
    dest = root / e["path"]
    dest.parent.mkdir(parents=True, exist_ok=True)
    t = time.perf_counter()
    record = dict(source="official_task_vectors_google_drive", drive_id=e["id"], path=str(dest))
    try:
        if not dest.exists():
            result = gdown.download(id=e["id"], output=str(dest), quiet=True)
            if result is None:
                raise RuntimeError("gdown returned no file")
        h = hashlib.sha256()
        with dest.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
        actual_hash = h.hexdigest()
        if actual_hash != e["sha256"] or dest.stat().st_size != e["bytes"]:
            raise RuntimeError("Checkpoint checksum/size differs from verified scientific manifest")
        record.update(bytes=dest.stat().st_size, sha256=actual_hash, status="downloaded")
    except Exception as exc:
        record.update(status="failed", error=str(exc))
    record["seconds"] = time.perf_counter() - t
    print(json.dumps(record), flush=True)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/checkpoints.json")
    parser.add_argument("--root", default="checkpoints")
    parser.add_argument("--output", default="outputs/2026-10-09")
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    entries = json.loads(Path(args.config).read_text(encoding="utf-8"))["files"]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(lambda entry: download(entry, Path(args.root)), entries))
    (out / "checkpoint_manifest.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    if any(result["status"] != "downloaded" for result in results):
        raise SystemExit("One or more checkpoint downloads failed; see manifest")
    print("CHECKPOINT_DOWNLOAD_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
