"""Pinned official PaRaMS reference, without modifying its solver or defaults."""

import argparse
import copy
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
if hasattr(os, "sched_getaffinity"):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])

import torch

from clip_benchmark import checkpoint_loader, digest
from subspace_probe import save_json


@torch.no_grad()
def construct(protocol, output):
    cfg = json.loads(Path(protocol).read_text()); out = Path(output)
    out.mkdir(parents=True, exist_ok=False); (out / "artifacts").mkdir()
    save_json(out / "protocol.json", cfg)
    torch.set_num_threads(4)
    assert "3090" in torch.cuda.get_device_name()
    for p, sha in cfg["source_sha256"].items(): assert digest(p) == sha, p
    source = Path(cfg["official_source"]); assert digest(source) == cfg["official_source_sha256"]
    load = checkpoint_loader(Path("external/secure-merge-attack"))
    root = Path("checkpoints/ViT-B-32")
    pins = {x["path"].removeprefix("ViT-B-32/"): x["sha256"]
            for x in json.loads(Path("configs/checkpoints.json").read_text())["files"]}
    hashes = {}
    models = []
    for name in ["zeroshot.pt", "MNIST/finetuned.pt"]:
        hashes[name] = digest(root / name); assert hashes[name] == pins[name]
        m = load(str(root / name)).cpu().eval(); m.requires_grad_(False); models.append(m)
    anchor, own = models
    spec = importlib.util.spec_from_file_location("official_params_reference", source)
    author = importlib.util.module_from_spec(spec); sys.modules[spec.name] = author; spec.loader.exec_module(author)
    start = time.monotonic()
    # None passes through to the published defaults: max_iter200, tolerance1,
    # mismatching, diagonal uniform[.05,20], both protection modules, seed0.
    state = author.params(own.state_dict(), anchor.state_dict(), num_layers=12,
                          enable_permutation=True, enable_scaling=True,
                          perm_config=None, scaling_config=None, rng_seed=0)
    wall = time.monotonic() - start
    assert all(torch.isfinite(v).all() for v in state.values())
    rows = []; squares = []; weight_squares = []; perm_squares = []
    raw_num = raw_den = 0.
    for k, v in own.state_dict().items():
        if not v.is_floating_point(): continue
        delta = state[k].double() - v.double(); norm = float(v.double().norm())
        dnorm = float(delta.norm()); raw_num += dnorm**2; raw_den += norm**2
        if not torch.equal(state[k], v):
            assert norm > 0, k
            relative = dnorm/norm; squares.append(relative**2)
            if v.ndim == 2: weight_squares.append(relative**2)
            if ".mlp." in k and v.ndim == 2: perm_squares.append(relative**2)
            rows.append(dict(key=k, shape=list(v.shape), original_norm=norm,
                             delta_norm=dnorm, relative=relative))
    distance = sum(squares)**.5; matrices = sum(weight_squares)**.5
    perm_distance = sum(perm_squares)**.5
    torch.save(state, out / "artifacts/PaRaMS.pt")
    freeze = dict(status="frozen_before_any_data_or_peer", parameter_distance_all_changed_tensors=distance,
                  parameter_distance_matrices=matrices, MLP_matrices_distance_lower_bound=perm_distance,
                  global_model_relative_distance=(raw_num/raw_den)**.5, parameter_budget=.5,
                  within_parameter_budget=distance<=.5, is_different_budget_reference=distance>.5,
                  original_expert_updated=False, gradients=0, labels_used=False, images_used=False,
                  future_peer_loaded=False, test_loaded=False, source_sha256=cfg["source_sha256"],
                  original_checkpoint_sha256=hashes, official_source_sha256=digest(source),
                  release_sha256=digest(out / "artifacts/PaRaMS.pt"),
                  construction_wall_seconds=wall, actual_default_call=True, edited_tensors=rows,
                  raw_shape="official state_dict; no clipping, interpolation or scale adjustment")
    save_json(out / "release_freeze.json", freeze)
    for name, sha in hashes.items(): assert digest(root/name) == sha
    print("PARAMS_DEFAULT_FROZEN", json.dumps({k:v for k,v in freeze.items() if k!="edited_tensors"}), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", default="configs/params_reference.json")
    p.add_argument("--output", required=True); args = p.parse_args()
    construct(args.protocol, args.output)
