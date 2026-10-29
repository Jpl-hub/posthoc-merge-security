"""One frozen-model two-task evaluator and original-context independent replay."""

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import dataset, make_loader, official_labels
from .io import check_files, digest, save_json
from .merge import fuse, load_ties, parameter_hash
from .metrics import row_metrics
from .models import apply, initialize, load_checkpoint


def case_models(cfg, case, anchor, originals, aligned, name):
    """Assemble fixed releases and optional public replacement controls."""
    import copy

    from .attacks import public_mlp_replace, published_bias_projection

    target = case["target"]
    spec = cfg["defenses"].get(name)
    own = anchor if name == "No_target" else originals[target]
    if spec is not None:
        own = apply(own, spec["release"], spec.get("full_state", False))
    scopes = cfg.get("attack_scopes", ["None", "Protected expert", "All experts"])
    attacked = None
    if any(s != "None" for s in scopes):
        attacked = aligned[target] if spec is None else apply(aligned[target], spec["AGA_edits"])
    if "public_replacement" in case:
        assert scopes == ["None"], "Replacement and AGA are separate declared pipelines"
        own = copy.deepcopy(own)
        own.load_state_dict(
            public_mlp_replace(own.state_dict(), anchor.state_dict(), **case["public_replacement"]),
            strict=True,
        )
    if case.get("published_bias_projection"):
        assert scopes == ["None"], "Bias projection is evaluated separately from AGA"
        own = copy.deepcopy(own)
        own.load_state_dict(published_bias_projection(own.state_dict()), strict=True)
    return own, attacked


@torch.no_grad()
def evaluate(model, head, loader, device: str, prediction_sink=None, logit_sink=None):
    from modeling import ImageClassifier
    from utils import get_logits

    classifier = ImageClassifier(model, head).to(device).eval()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start = time.perf_counter()
    count, correct = 0, 0
    for images, labels in loader:
        logits = get_logits(images.to(device, non_blocking=True), classifier)
        if logit_sink is not None:
            logit_sink(logits.detach().cpu())
        prediction = logits.argmax(-1).cpu()
        if prediction_sink is not None:
            prediction_sink(prediction, labels)
        correct += int((prediction == labels).sum())
        count += labels.numel()
    torch.cuda.synchronize()
    result = dict(
        correct=correct,
        n=count,
        top1=correct / count,
        evaluation_seconds=time.perf_counter() - start,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
    )
    classifier.cpu()
    return result


def setup(cfg):
    initialize()
    if not torch.cuda.is_available():
        raise RuntimeError("CLIP evaluation requires a CUDA runtime")
    author_path = Path("external/secure-merge-attack/utils_attack_B32.py")
    assert digest(author_path) == cfg["AGA"]["raw_sha256"]
    assert hashlib.sha256(author_path.read_text().encode()).hexdigest() == cfg["AGA"]["sha256"]
    check_files(cfg["source_sha256"])
    anchor = load_checkpoint("zeroshot.pt")
    originals = {t: load_checkpoint(f"{t}/finetuned.pt") for t in cfg["tasks"]}
    heads = {t: load_checkpoint(f"head_{t}.pt") for t in cfg["tasks"]}
    aligned = {t: apply(originals[t], cfg["AGA_references"][t], full=True) for t in cfg["tasks"]}
    ties = load_ties(cfg["TIES"]) if "TIES" in cfg["mergers"] else None
    return anchor, originals, heads, aligned, ties


@torch.no_grad()
def run(protocol, case_index, pilot):
    cfg = json.loads(Path(protocol).read_text())
    case = cfg["cases"][case_index]
    if "DARE" in cfg:
        cfg["DARE"]["seed"] = case["seed"]
    out = (
        Path(cfg["output"])
        / ("pilot" if pilot else "full")
        / f"{case_index:02d}_{case['target']}_{case['partner']}"
    )
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    save_json(out / "protocol.json", cfg)
    start = time.monotonic()
    anchor, originals, heads, aligned, ties = setup(cfg)
    target, partner = case["target"], case["partner"]
    data = {t: dataset(t, "data", originals[t].val_preprocess) for t in (target, partner)}
    assert all(len(data[t]) == cfg["tasks"][t]["test_n"] for t in data)
    indices = {
        t: np.linspace(0, len(d) - 1, cfg["pilot_n"], dtype=int).tolist()
        if pilot
        else list(range(len(d)))
        for t, d in data.items()
    }
    rows, raw, baseline = [], {}, {}
    fingerprints = {}
    loaded_legacy = {}

    def record(model, defense, method, scope, task, legacy=None):
        key = "/".join((defense, method, scope, task))
        model_key = "/".join((defense, method, scope))
        if model_key not in fingerprints:
            fingerprints[model_key] = parameter_hash(model.state_dict())
        imported = None
        if not pilot and legacy:
            ref, old_key = legacy
            if ref["path"] not in loaded_legacy:
                assert digest(ref["path"]) == ref["sha256"]
                loaded_legacy[ref["path"]] = torch.load(
                    ref["path"], map_location="cpu", weights_only=False
                )
            cached = loaded_legacy[ref["path"]]
            if ref.get("native_score_field") in ["before", "after"]:
                imported = dict(
                    logits=cached[ref["native_score_field"]],
                    target=torch.as_tensor(cached["labels"], dtype=torch.long),
                    indices=cached["indices"],
                )
            else:
                imported = cached.get(old_key)
        if imported is not None and "logits" not in imported:
            imported = None  # Old prediction-only records cannot supply real scores.
        if imported is not None:
            scores, y = imported["logits"], imported["target"]
            metric = dict(
                evaluation_seconds=0.0, imported_from=legacy[0]["path"], imported_key=legacy[1]
            )
        else:
            sink, logits = [], []
            metric = evaluate(
                model,
                heads[task],
                make_loader(data[task], indices[task]),
                "cuda",
                prediction_sink=lambda p, y: sink.append((p, y)),
                logit_sink=logits.append,
            )
            scores, y = torch.cat(logits), torch.cat([r[1] for r in sink])
        expected = official_labels(task)[indices[task]]
        np.testing.assert_array_equal(y.numpy(), expected)
        assert (
            scores.shape == (len(expected), cfg["tasks"][task]["classes"])
            and torch.isfinite(scores).all()
        )
        ref_key = "/".join((method, scope, task))
        if defense == "Unprotected":
            baseline[ref_key] = scores
        row = dict(
            defense=defense,
            target=target,
            partner=partner,
            merger=method,
            attack_scope=scope,
            dataset=task,
            role="target" if task == target else "partner",
            **row_metrics(scores, y, baseline[ref_key], cfg["tasks"][task]["classes"]),
            evaluation_seconds=metric["evaluation_seconds"],
            provenance={k: v for k, v in metric.items() if k.startswith("imported")},
            predicted_class_counts=torch.bincount(
                scores.argmax(1), minlength=cfg["tasks"][task]["classes"]
            ).tolist(),
        )
        raw[key] = dict(logits=scores, target=y, indices=indices[task])
        rows.append(row)
        save_json(out / "results.json", rows)
        torch.save(raw, out / "artifacts/predictions.pt")
        print("PHASE3_ROW", defense, method, scope, task, row["correct"], row["n"], flush=True)

    for name in ["Unprotected"] + case["defenses"]:
        own, attacked = case_models(cfg, case, anchor, originals, aligned, name)
        legacy = cfg["legacy"].get(f"{target}_{partner}/{name}", {})

        def old(stage, task):
            ref = legacy.get(stage)
            if not ref:
                return None
            key = ref.get("key_template", "{preset}/{stage}/{task}").format(
                preset=ref.get("preset", "released"), stage=ref.get("stage", stage), task=task
            )
            return ref, key

        record(own, name, "Standalone", "None", target, old("standalone", target))
        for method in cfg["mergers"]:
            for scope, left, right in [
                ("None", own, originals[partner]),
                ("Protected expert", attacked, originals[partner]),
                ("All experts", attacked, aligned[partner]),
            ]:
                if scope not in cfg.get("attack_scopes", ["None", "Protected expert", "All experts"]):
                    continue
                fused = fuse(anchor, left, right, method, cfg, ties)
                stage = {
                    "None": "direct",
                    "Protected expert": "AGA_target_only",
                    "All experts": "AGA_both",
                }[scope]
                for t in (target, partner):
                    record(fused, name, method, scope, t, old(stage, t) if method == "TA" else None)
                del fused
        del own, attacked
    save_json(
        out / "execution.json",
        dict(
            completed=True,
            pilot=pilot,
            case_index=case_index,
            rows=len(rows),
            wall_seconds=time.monotonic() - start,
            GPU=torch.cuda.get_device_name(),
            new_defense_weights=0,
            optimizer_steps=0,
            new_AGA_calls=0,
            new_image_forwards=sum(r["n"] for r in rows if not r["provenance"]),
            evaluated_parameter_sha256=fingerprints,
            public_replacement=case.get("public_replacement"),
            published_bias_projection=bool(case.get("published_bias_projection")),
            cached_author_AGA_weights=True,
            defense_feedback=False,
            predictions_sha256=digest(out / "artifacts/predictions.pt"),
            source_sha256=digest(__file__),
            protocol_sha256=digest(out / "protocol.json"),
        ),
    )


@torch.no_grad()
def verify(output):
    out = Path(output)
    cfg = json.loads((out / "protocol.json").read_text())
    meta = json.loads((out / "execution.json").read_text())
    assert (
        meta["completed"] and digest(out / "artifacts/predictions.pt") == meta["predictions_sha256"]
    )
    anchor, experts, heads, aligned, ties = setup(cfg)
    case = cfg["cases"][meta["case_index"]]
    target, partner = case["target"], case["partner"]
    data = {t: dataset(t, "data", experts[t].val_preprocess) for t in (target, partner)}
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=False)
    rows = json.loads((out / "results.json").read_text())
    batch = 128
    maximum, replays, context_images = 0.0, 0, 0
    for name in ["Unprotected"] + case["defenses"]:
        own, attacked = case_models(cfg, case, anchor, experts, aligned, name)
        groups = [("Standalone", "None", own, [target])]
        for method in cfg["mergers"]:
            for scope, left, right in [
                ("None", own, experts[partner]),
                ("Protected expert", attacked, experts[partner]),
                ("All experts", attacked, aligned[partner]),
            ]:
                if scope not in cfg.get("attack_scopes", ["None", "Protected expert", "All experts"]):
                    continue
                groups.append(
                    (
                        method,
                        scope,
                        fuse(anchor, left, right, method, cfg, ties, True),
                        [target, partner],
                    )
                )
        for method, scope, model, tasks in groups:
            for task in tasks:
                key = "/".join((name, method, scope, task))
                score = raw[key]
                logits = score["logits"].numpy()
                y = official_labels(task)[score["indices"]]
                pred = logits.argmax(1)
                np.testing.assert_array_equal(y, score["target"].numpy())
                assert np.isfinite(logits).all()
                row = next(
                    r
                    for r in rows
                    if (r["defense"], r["merger"], r["attack_scope"], r["dataset"])
                    == (name, method, scope, task)
                )
                control = (
                    raw["/".join(("Unprotected", method, scope, task))]["logits"].numpy().argmax(1)
                )
                assert int((pred == y).sum()) == row["correct"]
                if "predicted_class_counts" in row:
                    assert np.bincount(pred, minlength=logits.shape[1]).tolist() == row["predicted_class_counts"]
                assert int((pred != control).sum()) == row["changed_predictions"]
                assert int(((control == y) & (pred != y)).sum()) == row["paired"]["lost"]
                assert int(((control != y) & (pred == y)).sum()) == row["paired"]["gained"]
                for c in row["per_class"]:
                    mask = y == c["class_id"]
                    assert (
                        int(mask.sum()) == c["n"]
                        and int(((pred == y) & mask).sum()) == c["correct"]
                    )
                ids = np.linspace(0, len(y) - 1, cfg["replay_n"], dtype=int)
                model.cuda()
                heads[task].cuda()
                for block in sorted(set(ids // batch)):
                    start, end = block * batch, min((block + 1) * batch, len(y))
                    images = torch.stack(
                        [data[task][score["indices"][i]][0] for i in range(start, end)]
                    ).cuda()
                    actual = heads[task](model(images)).cpu()
                    wanted = ids[(ids >= start) & (ids < end)]
                    subset = actual[wanted - start]
                    np.testing.assert_array_equal(subset.argmax(1).numpy(), pred[wanted])
                    error = float((subset.double() - score["logits"][wanted].double()).abs().max())
                    assert error <= cfg["score_tolerance"], (key, error)
                    maximum = max(maximum, error)
                    replays += len(wanted)
                    context_images += len(images)
                model.cpu()
                heads[task].cpu()
        del groups, own, attacked
    save_json(
        out / "independent_validation.json",
        dict(
            status="passed",
            rows=len(rows),
            official_label_numpy_recount=True,
            full_image_prediction_replays=replays,
            context_forward_images=context_images,
            original_evaluation_batch_size=batch,
            maximum_score_error=maximum,
            score_tolerance=cfg["score_tolerance"],
            explicit_TA_reconstruction="TA" in cfg["mergers"],
            original_TIES_functions="TIES" in cfg["mergers"],
            independent_TIES_small_numpy_fixture="TIES" in cfg["mergers"],
            independent_full_TIES_reimplementation=False,
            raw_sha256=meta["predictions_sha256"],
            DARE_NumPy_mask_arithmetic="DARE+TA" in cfg["mergers"],
            DARE_independent_RNG=False,
            source_sha256=digest(__file__),
            original_evaluation_source_sha256=meta["source_sha256"],
            reason="Preserve original batch/GEMM context after observed 16-vs-128 numerical discrepancy; no scores, weights or tolerance changed.",
        ),
    )
    print("PHASE3_CONTEXT_VERIFIED", out, replays, maximum, flush=True)
