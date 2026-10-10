"""White-box score recovery with two public partner heads preserved.

An extra attack on frozen releases, not a protection method or full SLOA.
It uses a single fixed ridge fit and no calibration ground-truth labels.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
import types
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate
from eight_task_benchmark import digest, merge_all
from local_erasure import cache
from operator_diagnostics import paired_counts, query_readout
from polynomial_probe import ridge_update
from subspace_probe import save_json


@torch.no_grad()
def head_inverse(target, partners):
    other = torch.cat([h.weight.double().cuda() for h in partners])
    c = target.weight.double().cuda()
    _, s, vh = torch.linalg.svd(other, full_matrices=False)
    assert float(s[-1] / s[0]) > 1e-9, "Partner rows must be independent in this fixed construction"
    null = torch.eye(c.shape[1], dtype=torch.float64, device="cuda") - vh.T @ vh
    projected = c @ null
    spectrum = torch.linalg.svdvals(projected)
    assert float(spectrum[-1] / spectrum[0]) > 1e-9
    inverse = torch.linalg.solve(projected @ projected.T, projected)
    error = float((c @ inverse.T - torch.eye(len(c), device="cuda")).norm())
    cross = float((other @ inverse.T).norm())
    assert error < 1e-8 and cross < 1e-8
    return inverse, dict(
        preserved_partner_classes=len(other),
        null_dimension=len(c[0]) - len(other),
        target_classes=len(c),
        target_projected_condition=float(spectrum[0] / spectrum[-1]),
        target_inverse_error=error,
        partner_null_error=cross,
        all_seven_partner_heads_preserved=False,
    )


@torch.no_grad()
def recover(release, student, head, inverse, images, ratio):
    c = head.weight.double().cuda()
    release.cuda().eval()
    scores, ordinary = [], []
    head.cuda().eval()
    for batch in images.split(32):
        features = release(batch.cuda())
        scores.append((features.double() @ c.T).cpu())
        ordinary.append(head(features).argmax(-1).cpu())
    release.cpu()
    head.cpu()
    scores, ordinary = torch.cat(scores).cuda(), torch.cat(ordinary)
    _, h = query_readout(student, head, images, capture_features=True)
    p = student.model.visual.proj.double().cuda()
    residual = scores - h @ p @ c.T
    update, fit = ridge_update(h, residual, ratio)
    delta = update @ inverse
    restored = copy.deepcopy(release).cpu().eval()
    restored.model.visual.proj.copy_(
        (release.model.visual.proj.double().cuda() + delta / 0.2).float().cpu()
    )
    fit.update(
        native_to_ordinary_argmax_agreement=int(scores.argmax(-1).cpu().eq(ordinary).sum()),
        calibration_prediction_agreement=int(
            (h @ (p + delta) @ c.T).argmax(-1).cpu().eq(ordinary).sum()
        ),
        calibration_samples=len(images),
        calibration_ground_truth_used=False,
        teacher_intermediate_features_used=False,
        teacher_query="white-box unnormalized class scores",
    )
    evidence = dict(
        teacher_scores=scores.cpu(),
        teacher_predictions=ordinary,
        design=h.cpu(),
        residual=residual.cpu(),
        update=update.cpu(),
        projection_delta=delta.cpu(),
        restored_projection=restored.model.visual.proj.detach().clone(),
        inverse=inverse.cpu(),
    )
    return restored, fit, evidence


@torch.no_grad()
def run(args):
    cfg = json.loads(Path(args.config).read_bytes())
    parent, out, root = Path(cfg["parent"]), Path(args.output), Path("checkpoints/ViT-B-32")
    prior = json.loads((parent / "metadata.json").read_bytes())
    assert prior["completed"]
    assert json.loads((parent / "validation.json").read_bytes())["status"] == "passed"
    for name, expected in prior["checkpoint_sha256"].items():
        assert digest(root / name) == expected
    for name, expected in prior["raw_sha256"].items():
        assert digest(parent / "artifacts" / name) == expected
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    save_json(out / "protocol.json", cfg)
    start = time.perf_counter()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path("external/secure-merge-attack"))
    fusion = json.loads(Path("configs/eight_tasks.json").read_bytes())
    models = {t: load(str(root / t / "finetuned.pt")) for t in fusion["tasks"]}
    models["anchor"] = load(str(root / "zeroshot.pt"))
    heads = {t: load(str(root / f"head_{t}.pt")) for t in cfg["evaluation_tasks"]}
    inverses, geometry = {}, {}
    for target in cfg["tasks"]:
        assert all(torch.count_nonzero(h.bias) == 0 and h.normalize for h in heads.values())
        inverses[target], geometry[target] = head_inverse(
            heads[target], [h for t, h in heads.items() if t != target]
        )
    save_json(out / "geometry.json", geometry)
    package = types.ModuleType("dataset")
    package.__path__ = [str(Path("external/secure-merge-attack/datasets").resolve())]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    data = {
        t: get_dataset(t, models["anchor"].val_preprocess, "data", batch_size=128, num_workers=2)
        for t in cfg["evaluation_tasks"]
    }
    loaders = {t: cache(d) for t, d in data.items()}
    indices = json.loads((parent / "attack_indices.json").read_bytes())
    control = torch.load(parent / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    raw, rows, fits, mapping = {}, [], {}, []
    for target in cfg["tasks"]:
        images = torch.stack([data[target].train_dataset[i][0] for i in indices[target]])
        for preset in ("original", "erased"):
            assert time.perf_counter() - start < cfg["time_limit_seconds"]
            release = copy.deepcopy(models[target]).cpu().eval()
            if preset == "erased":
                edits = torch.load(
                    parent / "artifacts" / f"{target}_edits.pt",
                    map_location="cpu",
                    weights_only=True,
                )
                state = release.state_dict()
                for key, value in edits.items():
                    state[key].copy_(value)
            student = merge_all(dict(models, **{target: release}), root, 0.2)
            before = student.model.visual.proj.detach().double().clone()
            restored, fit, evidence = recover(
                release, student, heads[target], inverses[target], images, cfg["ridge_ratio"]
            )
            combined = merge_all(dict(models, **{target: restored}), root, 0.2)
            actual = combined.model.visual.proj.double() - before
            assert (
                float(
                    (actual - evidence["projection_delta"]).norm()
                    / evidence["projection_delta"].norm()
                )
                < 1e-6
            )
            other = torch.cat([h.weight.double() for t, h in heads.items() if t != target])
            mapping.append(
                dict(
                    target=target,
                    preset=preset,
                    compiled_partner_score_error=float((actual @ other.T).norm() / actual.norm()),
                    ideal_partner_score_error=float(
                        (evidence["projection_delta"] @ other.T).norm()
                    ),
                )
            )
            case = f"{target}/{preset}"
            fits[case] = fit
            torch.save(evidence, artifacts / f"{target}_{preset}.pt")
            save_json(out / "fitting.json", fits)
            for task in cfg["evaluation_tasks"]:
                key = f"{case}/{task}"
                predictions, labels = [], []
                score = evaluate(
                    combined,
                    heads[task],
                    loaders[task],
                    "cuda",
                    prediction_sink=lambda p, y: (predictions.append(p), labels.append(y)),
                )
                raw[key] = dict(prediction=torch.cat(predictions), labels=torch.cat(labels))
                baseline = control[f"{target}/{preset}/direct/fusion/{task}"]
                assert torch.equal(raw[key]["labels"], baseline["labels"])
                rows.append(
                    dict(
                        id=key,
                        target=target,
                        preset=preset,
                        dataset=task,
                        **score,
                        changed_from_parent_direct=int(
                            raw[key]["prediction"].ne(baseline["prediction"]).sum()
                        ),
                        paired_to_parent_direct=paired_counts(
                            baseline["prediction"].eq(baseline["labels"]),
                            raw[key]["prediction"].eq(raw[key]["labels"]),
                        ),
                    )
                )
                torch.save(raw, artifacts / "predictions.pt")
                save_json(out / "results.json", rows)
                print("RESULT", key, score["correct"], score["n"], flush=True)
            del combined, student, release, restored
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            parent_files_sha256={
                n: digest(parent / n)
                for n in [
                    "metadata.json",
                    "validation.json",
                    "protocol.json",
                    "attack_indices.json",
                    "release_freeze.json",
                ]
            },
            parent_raw_sha256=prior["raw_sha256"],
            checkpoint_sha256=prior["checkpoint_sha256"],
            helper_sha256={
                n: digest(Path(__file__).with_name(n))
                for n in [
                    "clip_benchmark.py",
                    "eight_task_benchmark.py",
                    "local_erasure.py",
                    "operator_diagnostics.py",
                    "polynomial_probe.py",
                    "subspace_probe.py",
                ]
            },
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
            projection_mapping=mapping,
            elapsed_seconds=time.perf_counter() - start,
            rows=len(rows),
            fits=len(fits),
            protection_changed=False,
            all_seven_partner_heads_preserved=False,
            full_sloa=False,
            calibration_truth_used=False,
            attack_selected_after_parent_results=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/score_recovery.json")
    parser.add_argument("--output", default="outputs/2026-10-11/score_recovery")
    run(parser.parse_args())
