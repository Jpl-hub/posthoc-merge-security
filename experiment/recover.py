"""Apply original AGA, or separately scoped public-patch family matching."""

import argparse
import time

import torch

from _impl.attacks import author_recover, load_author, public_patch_recovery
from _impl.io import digest, new_output, save_json
from _impl.models import apply, initialize, load_checkpoint
from cfg import AUTHOR, load_config, require_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=["MNIST", "DTD", "SVHN"], required=True)
    p.add_argument("--method", choices=["aga", "patch"], default="aga")
    p.add_argument("--publication", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--full-state", action="store_true")
    p.add_argument("--output", required=True)
    a = p.parse_args()
    require_root()
    initialize()
    protocol = vars(a)
    out = new_output(a.output, protocol)
    base = load_checkpoint("zeroshot.pt")
    expert = load_checkpoint(f"{a.target}/finetuned.pt")
    published = apply(expert, dict(path=a.publication, sha256=a.sha256), a.full_state)
    start = time.monotonic()
    if a.method == "aga":
        cfg = load_config("configs/evaluate.json")
        source = AUTHOR / cfg["AGA"]["file"]
        if digest(source) != cfg["AGA"]["raw_sha256"]:
            raise ValueError("AGA source mismatch")
        author, sha = load_author(source, True)
        if sha != cfg["AGA"]["sha256"]:
            raise ValueError("AGA text mismatch")
        result = author_recover(published, base, author).state_dict()
        detail = dict(known_family=False, author_text_sha256=sha)
    else:
        # Recovery gets only the full published state and public base. The clean
        # expert above assembles an edits-only publication; it is not passed here.
        result, detail = public_patch_recovery(published.state_dict(), base.state_dict())
        detail["known_family"] = True
    edits = {k: v for k, v in result.items() if not torch.equal(v, expert.state_dict()[k])}
    torch.save(edits, out / "edits.pt")
    save_json(
        out / "execution.json",
        dict(
            method=a.method,
            seconds=time.monotonic() - start,
            output_sha256=digest(out / "edits.pt"),
            details=detail,
            new_training_updates=0,
        ),
    )


if __name__ == "__main__":
    main()
