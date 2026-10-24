"""Apply original AGA, or separately scoped public-patch family matching."""

import argparse
import time

import torch

from _impl.attacks import author_recover, load_author, public_mlp_reset, public_patch_recovery
from _impl.io import digest, new_output, save_json
from _impl.models import apply, initialize, load_checkpoint
from cfg import AUTHOR, load_config, require_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=["MNIST", "DTD", "SVHN"], required=True)
    p.add_argument("--method", choices=["aga", "patch", "reset"], default="aga")
    p.add_argument("--publication", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--full-state", action="store_true")
    p.add_argument(
        "--aligned-reference",
        action="store_true",
        help="Store AGA differences from the already frozen AGA of the clean expert",
    )
    p.add_argument(
        "--base-reference",
        action="store_true",
        help="Apply and store edits relative to the public base",
    )
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.aligned_reference and (a.method != "aga" or a.base_reference):
        p.error("aligned-reference requires aga without base-reference")
    require_root()
    initialize()
    protocol = vars(a)
    out = new_output(a.output, protocol)
    base = load_checkpoint("zeroshot.pt")
    expert = load_checkpoint(f"{a.target}/finetuned.pt")
    reference = base if a.base_reference else expert
    published = apply(reference, dict(path=a.publication, sha256=a.sha256), a.full_state)
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
        if a.aligned_reference:
            ref = cfg["AGA_references"][a.target]
            reference = apply(expert, ref, full=True)
            detail["aligned_reference"] = ref
    elif a.method == "patch":
        # Recovery gets only the full published state and public base. The clean
        # expert above assembles an edits-only publication; it is not passed here.
        result, detail = public_patch_recovery(published.state_dict(), base.state_dict())
        detail["known_family"] = True
    else:
        result = public_mlp_reset(published.state_dict(), base.state_dict())
        detail = dict(
            known_modified_modules=True,
            reset_tensors=sum(".mlp." in k for k in result),
            images=0,
            original_expert_used_by_attack=False,
        )
        reference = base
    edits = {k: v for k, v in result.items() if not torch.equal(v, reference.state_dict()[k])}
    assert all(torch.equal(v, (reference.state_dict() | edits)[k]) for k, v in result.items())
    torch.save(edits, out / "edits.pt")
    save_json(
        out / "execution.json",
        dict(
            method=a.method,
            seconds=time.monotonic() - start,
            output_sha256=digest(out / "edits.pt"),
            details=detail,
            reference="original_AGA"
            if a.aligned_reference
            else ("public_base" if a.base_reference or a.method == "reset" else "expert"),
            new_training_updates=0,
        ),
    )


if __name__ == "__main__":
    main()
