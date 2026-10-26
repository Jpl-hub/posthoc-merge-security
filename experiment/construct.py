"""Construct a defense from the target/base; no partner or test-based selection."""

import argparse
import time
from pathlib import Path

import torch

from _impl.baselines import params_default, verify_edits
from _impl.defenses import construct_bias
from _impl.io import digest, new_output, save_json
from _impl.models import checked_models, initialize
from _impl.subspace import build_basis, task_pairing
from cfg import load_config, require_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method",
        choices=["task-directed", "bias", "params", "basis"],
        default="task-directed",
    )
    p.add_argument("--target", choices=["MNIST", "DTD", "SVHN"], required=True)
    p.add_argument("--protocol", default="configs/construct.json")
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=["task_aware", "distance_only"], default="task_aware")
    a = p.parse_args()
    require_root()
    initialize()
    cfg = load_config(a.protocol)
    case = cfg["targets"][a.target]
    if a.method == "task-directed":
        c = dict(
            target=a.target,
            rank=cfg["rank"],
            tolerance=cfg["tolerance"],
            gain_tolerance=cfg["gain_tolerance"],
            basis=case["basis"],
            source_sha256=cfg["source_sha256"],
            release_variant=a.variant,
        )
        out = Path(a.output)
        protocol = out.parent / (out.name + ".protocol.json")
        if out.exists() or protocol.exists():
            raise FileExistsError(a.output)
        protocol.parent.mkdir(parents=True, exist_ok=True)
        save_json(protocol, c)
        task_pairing(str(protocol), a.output)
        return
    out = new_output(a.output, dict(method=a.method, target=a.target, configuration=cfg))
    anchor, expert, head, pins = checked_models(a.target)
    start = time.monotonic()
    publication_reference = "expert"
    if a.method == "basis":
        basis = build_basis(expert.state_dict(), anchor.state_dict(), cfg["rank"])
        torch.save(basis, out / "basis.pt")
        save_json(
            out / "execution.json",
            dict(
                basis_sha256=digest(out / "basis.pt"),
                checkpoint_sha256=pins,
                matrices=len(basis),
                rank=cfg["rank"],
                images=0,
            ),
        )
        return
    if a.method == "bias":
        edits, details = construct_bias(expert, anchor, head, cfg["bias"])
    else:
        c = cfg["params"]
        edits = params_default(expert, anchor, c["path"], c["sha256"])
        details = verify_edits(expert, edits)
    torch.save(edits, out / "release.pt")
    save_json(
        out / "before_images.json",
        dict(
            release_sha256=digest(out / "release.pt"),
            publication_reference=publication_reference,
            checkpoint_sha256=pins,
            method=a.method,
            target=a.target,
            seconds=time.monotonic() - start,
            details=details,
            expert_training_updates=0,
        ),
    )


if __name__ == "__main__":
    main()
