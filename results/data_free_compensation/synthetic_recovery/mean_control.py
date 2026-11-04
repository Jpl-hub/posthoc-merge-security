"""Fixed weight-only control of the task-irrelevant mean in feature supervision."""
import argparse
import copy
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from _impl.data import dataset, make_loader, official_labels
from _impl.evaluate import evaluate
from _impl.io import check_files, digest, new_output, read_json, save_json
from _impl.models import apply, checked_models, initialize


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    cfg = read_json(args.protocol)
    check_files(cfg["source_sha256"])
    initialize()
    out = new_output(cfg["output"], cfg)
    base, expert, head, _ = checked_models("MNIST")
    parent = apply(expert, cfg["parent"])
    key = "model.visual.transformer.resblocks.11.mlp.c_proj.weight"
    original = expert.state_dict()[key].double()
    public = base.state_dict()[key].double()
    before = parent.state_dict()[key].double()
    # Right task directions are probes; full native invariance follows from LN.
    task_basis = torch.linalg.svd(original - public, full_matrices=False).Vh[:32].T
    residual = original - (original @ task_basis) @ task_basis.T
    direction = torch.linalg.svd(residual, full_matrices=False).Vh[0]
    direction *= direction[direction.abs().argmax()].sign()
    mean = torch.ones(original.shape[0], dtype=torch.double) / math.sqrt(original.shape[0])
    update = mean[:, None] * direction[None, :]
    previous = before - original
    a = (previous * update).sum()
    radius = cfg["B_relative_limit"] * original.norm()
    scale = -a + (a.square() + radius.square() - previous.square().sum()).sqrt()
    changed = (before + scale * update).float()
    relative = float((changed.double() - original).norm() / original.norm())
    assert relative <= 4 and torch.isfinite(changed).all()
    edits = torch.load(cfg["parent"]["path"], map_location="cpu", weights_only=True)
    edits[key] = changed
    path = out / "point.pt"
    torch.save(edits, path)
    release = dict(path=str(path), sha256=digest(path))
    centered = lambda w: w.double() - w.double().mean(0, keepdim=True)
    save_json(out / "freeze.json", dict(release=release, B_relative=relative,
        task_right_residual=float((direction @ task_basis).norm()),
        centered_B_error=float((centered(changed)-centered(before)).abs().max()),
        added_norm=float((changed.double()-before).norm()), real_construction_images=0,
        known_cleanup="Column-centering B removes this mean addition in exact arithmetic.",
        source_sha256=digest(__file__)))
    model = apply(expert, release)
    data = dataset("MNIST", "data", expert.val_preprocess)
    scores = []
    for current in (parent, model):
        score = []
        evaluate(current, head, make_loader(data), "cuda", logit_sink=score.append)
        scores.append(torch.cat(score))
    y = official_labels("MNIST")
    for score in scores:
        assert len(score) == len(y) and torch.isfinite(score).all()
    same = int((scores[0].argmax(1) == scores[1].argmax(1)).sum())
    raw = out / "native_scores.pt"
    torch.save(dict(before=scores[0], after=scores[1], labels=y), raw)
    row = dict(n=len(y), same_predictions=same,
               before_correct=int(np.sum(scores[0].numpy().argmax(1)==y)),
               after_correct=int(np.sum(scores[1].numpy().argmax(1)==y)),
               max_score_error=float((scores[0]-scores[1]).abs().max()),
               raw_sha256=digest(raw))
    save_json(out / "native.json", row)
    print("NATIVE", row, flush=True)
    if same != len(y) or row["max_score_error"] > cfg["score_tolerance"]:
        return
    attack = copy.deepcopy(cfg["attack"])
    attack.update(output=str(out/"prodistill"), source_sha256=cfg["source_sha256"])
    attack["cases"][0]["releases"] = {"MeanControl": release}
    run_cfg = out / "prodistill.json"
    save_json(run_cfg, attack)
    subprocess.run([sys.executable, "-u", "-m", "experiment.distill", "--prodistill",
                    "--protocol", str(run_cfg)], check=True)


if __name__ == "__main__":
    main()
