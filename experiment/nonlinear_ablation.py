"""Reproduce fixed MNIST numerical component ablations without fitting new weights.

The source release,1024 indices and task basis are the frozen paper artifacts.
Run from the repository root; original results are never overwritten.
"""

import copy
import json
from pathlib import Path

import numpy as np
import torch

from _impl.data import dataset, make_loader, official_labels
from _impl.evaluate import evaluate
from _impl.io import digest
from _impl.metrics import task_change
from _impl.models import checked_models, initialize


def main():
    initialize()
    root = Path("outputs/nonlinear_prefix_20261023")
    out = root / "source_components"
    out.mkdir(exist_ok=False)
    _, expert, head, _ = checked_models("MNIST")
    edits = torch.load(root / "release.pt", map_location="cpu", weights_only=False)
    cal = torch.load(root / "calibration.pt", map_location="cpu", weights_only=False)
    ids = cal["indices"]
    loader = make_loader(dataset("MNIST", "data", expert.val_preprocess, True), ids)
    labels = official_labels("MNIST", True)[ids]
    before = cal["before"]
    rows = []
    raw = {}
    for name, selected in [
        ("A_only", {k: v for k, v in edits.items() if ".c_fc." in k}),
        ("B_only", {k: v for k, v in edits.items() if ".c_proj." in k}),
    ]:
        model = copy.deepcopy(expert)
        model.load_state_dict(expert.state_dict() | selected, strict=True)
        sink = []
        evaluate(model, head, loader, "cuda", logit_sink=sink.append)
        after = torch.cat(sink).double().numpy()
        raw[name] = after
        metric = task_change(before, after)
        rows.append(dict(component=name, correct=int((after.argmax(1) == labels).sum()), **metric))
        print(rows[-1], flush=True)
    # Matched transport-only control: remove every fitted nonlinear correction;
    # reconstruct exactly the pre-fitA and B matrices from the frozen original basis.
    basis = torch.load(
        "outputs/phase2_task_subspace_comparison_20261020/basis.pt",
        map_location="cpu",
        weights_only=False,
    )
    entries = {r["source_key"]: r for r in basis.values()}
    state = expert.state_dict()
    transport = {}
    for i in range(12):
        prefix = f"model.visual.transformer.resblocks.{i}.mlp."
        ka = prefix + "c_fc.weight"
        kb = prefix + "c_proj.weight"
        A = state[ka].double().cuda()
        B = state[kb].double().cuda()
        a = state[prefix + "c_fc.bias"].double().cuda()
        ba = entries[ka]
        bb = entries[kb]
        ua, va, vb = ba["u"].cuda(), ba["v"].cuda(), bb["v"].cuda()
        columns = torch.cat([ua, A @ va, vb, a[:, None]], dim=1)
        columns = columns[:, columns.norm(dim=0) > 0]
        columns /= columns.norm(dim=0)
        C, s, _ = torch.linalg.svd(columns, full_matrices=False)
        C = C[:, s > s[0] * 1e-10]
        Ap = (2 * C @ (C.T @ A) - A).float().cpu()
        assert torch.equal(Ap, edits[ka])
        transport[ka] = Ap
        transport[kb] = (2 * (B @ C) @ C.T - B).float().cpu()
    model = copy.deepcopy(expert)
    model.load_state_dict(state | transport, strict=True)
    sink = []
    evaluate(model, head, loader, "cuda", logit_sink=sink.append)
    raw["No_reconstruction"] = torch.cat(sink).double().numpy()
    raw["Joint"] = cal["after"]
    for name in ["No_reconstruction", "Joint"]:
        after = raw[name]
        rows.append(
            dict(
                component=name,
                correct=int((after.argmax(1) == labels).sum()),
                **task_change(before, after),
            )
        )
    for r in rows:
        after = raw[r["component"]]
        assert r["correct"] == int(np.sum(after.argmax(1) == labels))
        assert r["changed"] == int(np.sum(after.argmax(1) != before.argmax(1)))
    torch.save(dict(before=before, after=raw, labels=labels, indices=ids), out / "scores.pt")
    result = dict(
        rows=rows,
        point_sha256=digest(root / "release.pt"),
        raw_sha256=digest(out / "scores.pt"),
        new_optimizer_steps=0,
        source_only=True,
        new_defense_weights=0,
        main_publication_unchanged=True,
        predictions_selected_from_source=False,
    )
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
