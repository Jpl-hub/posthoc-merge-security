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
from _impl.models import apply, checked_models, initialize
from cfg import require_root


@torch.no_grad()
def run(args):
    initialize()
    ref = dict(path=args.publication, sha256=args.sha256)
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
    for stage, training in [("training", True), ("full", False)]:
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
        if not row["passed"]:
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
    p.add_argument("--verify", metavar="RESULT_DIRECTORY")
    p.add_argument("--target", choices=["MNIST", "DTD", "SVHN"])
    p.add_argument("--publication")
    p.add_argument("--sha256")
    p.add_argument("--full-state", action="store_true")
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
    if a.verify:
        verify(a.verify)
    else:
        if not all([a.target, a.publication, a.sha256, a.indices, a.indices_sha256, a.output]):
            p.error("target,publication,sha256,indices,indices-sha256,output required")
        run(a)


if __name__ == "__main__":
    main()
