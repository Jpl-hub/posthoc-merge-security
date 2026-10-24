"""Construct a defense from the target/base; no partner or test-based selection."""

import argparse
import json
import time
from pathlib import Path

import torch

from _impl.baselines import params_default, verify_edits
from _impl.data import dataset
from _impl.defenses import construct_bias, nonlinear_mlp_compensation
from _impl.io import check_files, digest, new_output, save_json
from _impl.models import checked_models, initialize
from _impl.subspace import build_basis, task_pairing
from cfg import load_config, require_root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--method",
        choices=["task-directed", "bias", "params", "basis", "nonlinear"],
        default="task-directed",
    )
    p.add_argument("--target", choices=["MNIST", "DTD", "SVHN"], required=True)
    p.add_argument("--protocol", default="configs/construct.json")
    p.add_argument("--output", required=True)
    p.add_argument("--variant", choices=["task_aware", "distance_only"], default="task_aware")
    p.add_argument("--indices", help="Frozen target-training index JSON for Defense1")
    p.add_argument("--indices-sha256", help="Previously frozen index-file SHA256; no new sampling")
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
    if a.method == "nonlinear":
        c = cfg["nonlinear"] | case.get("nonlinear", {}) | {"target": a.target}
        ref = c["indices"]
        check_files({ref["path"]: ref["sha256"], case["basis"]["path"]: case["basis"]["sha256"]})
        indices = json.loads(Path(ref["path"]).read_text())["indices"][: c["source_n"]]
        basis = torch.load(case["basis"]["path"], map_location="cpu", weights_only=False)
        train = dataset(a.target, "data", expert.val_preprocess, True)
        edits, calibration, details = nonlinear_mlp_compensation(
            expert, head, train, indices, basis, c, anchor=anchor
        )
        torch.save(calibration, out / "calibration.pt")
        if c.get("publication_reference") == "public_base":
            state = expert.state_dict() | edits
            edits = {k: v for k, v in state.items() if not torch.equal(v, anchor.state_dict()[k])}
            assert all(torch.equal(v, (anchor.state_dict() | edits)[k]) for k, v in state.items())
            publication_reference = "public_base"
    elif a.method == "bias":
        if not a.indices or not a.indices_sha256:
            p.error("bias requires --indices and --indices-sha256")
        c = cfg["bias"]
        check_files({a.indices: a.indices_sha256})
        indices = json.loads(Path(a.indices).read_text())
        train = dataset(a.target, "data", expert.val_preprocess, True)
        _, edits, calibration, details = construct_bias(
            expert, anchor, head, train, indices, c, a.target
        )
        torch.save(calibration, out / "calibration.pt")
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
