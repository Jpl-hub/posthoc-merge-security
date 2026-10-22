"""Result IO and content fingerprints."""

import hashlib
import json
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def check_files(pins):
    for path, expected in pins.items():
        if digest(path) != expected:
            raise ValueError(f"SHA256 mismatch: {path}")


def new_output(path, protocol):
    out = Path(path)
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / "protocol.json", protocol)
    return out
