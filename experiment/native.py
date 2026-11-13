"""Standalone retention for one frozen publication, followed by read-only recount."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from _impl.data import dataset, make_loader, official_labels
from _impl.evaluate import evaluate
from _impl.io import check_files, digest, new_output, save_json
from _impl.metrics import logical_parameter_distance, task_change
from _impl.models import apply, checked_models, initialize, load_checkpoint
from cfg import require_root


@torch.no_grad()
def precision_run(protocol):
    """Evaluate fixed weights under AMP, with an explicit FP32-readout control."""
    import copy
    from modeling import ImageClassifier

    cfg = json.loads(Path(protocol).read_text())
    check_files(cfg["source_sha256"])
    initialize()
    out = new_output(cfg["output"], cfg)
    rows = []
    cases = [(t, t, ref) for t, ref in cfg["publications"].items()]
    if cfg.get("merge"):
        cases += [(t, cfg["merge"]["partners"][t], ref)
                  for t, ref in cfg["publications"].items()]
    for target, task, ref in cases:
        anchor, expert, head, pins = checked_models(target)
        if cfg.get("merge"):
            from _impl.merge import merge
            spec = cfg["merge"]
            peer_name = spec["partners"][target]
            peer = load_checkpoint(f"{peer_name}/finetuned.pt")
            aligned = apply(expert, spec["AGA_references"][target], full=True)
            protected = merge(anchor, apply(aligned, ref), peer, spec["coefficient"])
            expert = merge(anchor, aligned, peer, spec["coefficient"])
            head = load_checkpoint(f"head_{task}.pt")
            del aligned, peer
        else:
            protected = apply(expert, ref)
        del anchor
        data = dataset(task, "data", expert.val_preprocess)
        labels = official_labels(task)
        assert len(data) == len(labels) == cfg["test_sizes"][task]
        baseline = {}
        for mode in cfg["modes"]:
            for name, source in [("None", expert), ("NBP", protected)]:
                classifier = ImageClassifier(copy.deepcopy(source), copy.deepcopy(head)).cuda().eval()
                visual = classifier.image_encoder.model.visual
                dtype = torch.bfloat16 if "bf16" in mode else torch.float16
                tail_fp32 = mode.endswith("fp32_readout")
                captured, handles = [], []
                if tail_fp32:
                    handles = [
                        visual.ln_post.register_forward_pre_hook(lambda m, a: (a[0].float(),)),
                        visual.ln_post.register_forward_hook(lambda m, a, y: captured.append(y)),
                    ]
                scores, observed_dtypes, offset = [], {}, 0
                torch.cuda.reset_peak_memory_stats()
                torch.cuda.synchronize()
                start = time.perf_counter()
                for images, batch_labels in make_loader(data, batch_size=cfg["batch_size"]):
                    np.testing.assert_array_equal(batch_labels.numpy(), labels[offset:offset+len(batch_labels)])
                    offset += len(batch_labels)
                    captured.clear()
                    with torch.autocast("cuda", dtype=dtype, enabled=mode != "fp32"):
                        features = classifier.image_encoder(images.cuda(non_blocking=True))
                        if not tail_fp32:
                            logits = classifier.classification_head(features)
                    if tail_fp32:
                        # Recompute only the projection/head outside AMP; the encoder is unchanged.
                        assert len(captured) == 1 and captured[0].dtype == torch.float32
                        features = captured[0] @ visual.proj.float()
                        logits = classifier.classification_head(features)
                    observed_dtypes = dict(features=str(features.dtype), logits=str(logits.dtype),
                        projection=str(visual.proj.dtype), final_bias=str(visual.ln_post.bias.dtype))
                    scores.append(logits.float().cpu())
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                for handle in handles:
                    handle.remove()
                values = torch.cat(scores).numpy()
                finite = np.isfinite(values).all(1)
                pred = values.argmax(1)
                valid = bool(finite.all())
                if mode == "fp32":
                    baseline[name] = pred.copy()
                row = dict(task=task, protected_task=target,
                    scenario="AGA+TA" if cfg.get("merge") else "Standalone",
                    method=name, mode=mode, n=len(labels),
                    finite_rows=int(finite.sum()), nonfinite_rows=int((~finite).sum()),
                    accuracy_percent=float(100*np.mean(pred == labels)) if valid else None,
                    correct=int((pred == labels).sum()) if valid else None,
                    changed_vs_own_fp32=int((pred != baseline[name]).sum()) if valid else None,
                    observed_dtypes=observed_dtypes, seconds=elapsed,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(), checkpoint_sha256=pins)
                prefix = f"{target}_{task}" if cfg.get("merge") else task
                path = out / f"{prefix}_{name}_{mode}.npz"
                np.savez_compressed(path, scores=values, labels=labels)
                row["raw_file"], row["raw_sha256"] = path.name, digest(path)
                rows.append(row)
                save_json(out / "results.json", dict(rows=rows, completed=False,
                    torch_version=torch.__version__, device=torch.cuda.get_device_name(0),
                    parameter_storage="FP32 for all modes", TF32=False))
                print("PRECISION", json.dumps(row), flush=True)
                del classifier, scores, values, features, logits
                captured.clear()
                torch.cuda.empty_cache()
    # Recount saved scores independently with NumPy; invalid outputs are not accuracies.
    for row in rows:
        raw = np.load(out / row["raw_file"])
        np.testing.assert_array_equal(raw["labels"], official_labels(row["task"]))
        finite = np.isfinite(raw["scores"]).all(1)
        assert int((~finite).sum()) == row["nonfinite_rows"]
        if finite.all():
            assert int((raw["scores"].argmax(1) == raw["labels"]).sum()) == row["correct"]
    meta = json.loads((out / "results.json").read_text())
    meta.update(completed=True, numpy_recount_passed=True)
    save_json(out / "results.json", meta)


@torch.no_grad()
def run(args):
    initialize()
    ref = dict(
        path=args.publication,
        sha256=args.sha256,
        reference="public_base" if args.base_reference else "expert",
    )
    ids = []
    if not args.test_only:
        check_files({args.indices: args.indices_sha256})
        ids = json.loads(Path(args.indices).read_text())
        ids = ids["indices"] if isinstance(ids, dict) else ids
        if len(ids) != len(set(ids)):
            raise ValueError("Training indices contain duplicates")
    _, expert, head, pins = checked_models(args.target)
    model = apply(expert, ref, args.full_state)
    out = new_output(args.output, vars(args))
    rows = []
    start = time.monotonic()
    edits = {
        k: v for k, v in model.state_dict().items() if not torch.equal(v, expert.state_dict()[k])
    }
    distance = logical_parameter_distance(edits, expert.state_dict())
    save_json(
        out / "before_images.json",
        dict(
            publication=ref,
            checkpoint_sha256=pins,
            parameter_distance=distance,
            construction_updates=0,
        ),
    )
    stages = [("full", False)] if args.test_only else [("training", True), ("full", False)]
    for stage, training in stages:
        data = dataset(args.target, "data", expert.val_preprocess, training)
        indices = ids if training else list(range(len(data)))
        if any(i < 0 or i >= len(data) for i in indices):
            raise ValueError("Index outside official split")
        scores = []
        for current in [expert, model]:
            sink = []
            evaluate(current, head, make_loader(data, indices), "cuda", logit_sink=sink.append)
            scores.append(torch.cat(sink).double())
        labels = official_labels(args.target, training)[indices]
        row = task_change(*(x.numpy() for x in scores))
        row.update(
            stage=stage,
            before_correct=int(np.sum(scores[0].numpy().argmax(1) == labels)),
            after_correct=int(np.sum(scores[1].numpy().argmax(1) == labels)),
        )
        row["strict_task_passed"] = row["changed"] == 0 and row["R"] <= args.task_budget
        row["accuracy_loss_pp"] = 100 * (row["before_correct"] - row["after_correct"]) / row["n"]
        row["passed"] = (
            row["strict_task_passed"]
            if args.max_accuracy_loss_pp is None
            else row["accuracy_loss_pp"] <= args.max_accuracy_loss_pp
        )
        torch.save(
            dict(indices=indices, labels=labels, before=scores[0], after=scores[1]),
            out / (stage + ".pt"),
        )
        row["raw_sha256"] = digest(out / (stage + ".pt"))
        rows.append(row)
        save_json(
            out / "results.json",
            dict(
                rows=rows,
                passed=all(r["passed"] for r in rows),
                image_forwards=sum(2 * r["n"] for r in rows),
                seconds=time.monotonic() - start,
            ),
        )
        print(json.dumps(row), flush=True)
        if not row["passed"] and args.max_accuracy_loss_pp is None:
            break


def verify(folder):
    out = Path(folder)
    cfg = json.loads((out / "protocol.json").read_text())
    meta = json.loads((out / "results.json").read_text())
    checks = []
    for row in meta["rows"]:
        path = out / (row["stage"] + ".pt")
        check_files({str(path): row["raw_sha256"]})
        raw = torch.load(path, map_location="cpu", weights_only=False)
        a, b = [raw[k].numpy() for k in ["before", "after"]]
        y = official_labels(cfg["target"], row["stage"] == "training")[raw["indices"]]
        np.testing.assert_array_equal(y, raw["labels"])
        yy = a.argmax(1)
        mask = np.arange(a.shape[1])[None, :] != yy[:, None]
        r = np.arange(len(y))
        gap = (a[r, yy, None] - a)[mask]
        delta = b - a
        expected = float(np.max(np.abs((delta[r, yy, None] - delta)[mask] / gap)))
        assert abs(expected - row["R"]) < 1e-10
        assert int(np.sum(yy != b.argmax(1))) == row["changed"]
        assert (
            int(np.sum(yy == y)) == row["before_correct"]
            and int(np.sum(b.argmax(1) == y)) == row["after_correct"]
        )
        strict = row["changed"] == 0 and expected <= cfg["task_budget"]
        loss = 100 * (row["before_correct"] - row["after_correct"]) / len(y)
        passed = (
            strict
            if cfg.get("max_accuracy_loss_pp") is None
            else loss <= cfg["max_accuracy_loss_pp"]
        )
        assert passed == row["passed"]
        checks.append(dict(stage=row["stage"], n=len(y), R=expected))
    save_json(
        out / "independent_validation.json",
        dict(
            status="passed_raw_score_and_official_label_recount",
            rows=checks,
            results_sha256=digest(out / "results.json"),
            new_image_forwards=0,
        ),
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision-protocol", help="Fixed full-test AMP/readout precision comparison")
    p.add_argument("--verify", metavar="RESULT_DIRECTORY")
    p.add_argument("--target", choices=["MNIST", "DTD", "SVHN"])
    p.add_argument("--publication")
    p.add_argument("--sha256")
    p.add_argument("--full-state", action="store_true")
    p.add_argument("--test-only", action="store_true", help="Evaluate only the complete test split")
    p.add_argument(
        "--base-reference", action="store_true", help="Publication is relative to public base"
    )
    p.add_argument("--indices")
    p.add_argument("--indices-sha256")
    p.add_argument("--output")
    p.add_argument("--task-budget", type=float, default=0.5)
    p.add_argument(
        "--max-accuracy-loss-pp",
        type=float,
        help="Separately declared accuracy protocol; still report the historical strict task check",
    )
    a = p.parse_args()
    require_root()
    if a.precision_protocol:
        initialize()
        # Load the upstream model definitions before constructing a classifier.
        from _impl.models import checkpoint_loader
        from cfg import AUTHOR
        checkpoint_loader(AUTHOR)
        precision_run(a.precision_protocol)
    elif a.verify:
        verify(a.verify)
    else:
        if not all([a.target, a.publication, a.sha256, a.output]):
            p.error("target,publication,sha256,output required")
        if not a.test_only and not all([a.indices, a.indices_sha256]):
            p.error("indices and indices-sha256 required unless --test-only")
        run(a)


if __name__ == "__main__":
    main()
