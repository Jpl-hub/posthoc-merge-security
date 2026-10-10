"""Eight-task baseline using upstream TaskVector and dataset registry unchanged.

Resource checks are executable without downloading or evaluating data. Missing
experts cause a hard stop; the task list is never reduced silently.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import types
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate, protect, recover
from mechanisms import load_author, relative


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def verify_checkpoints(args, root, config, output):
    """Validate official file pins, compatible expert tensors and task heads."""
    torch.set_num_threads(4)
    files = json.loads(Path("configs/checkpoints.json").read_bytes())["files"]
    load = checkpoint_loader(Path(args.author_repo))
    anchor = load(str(root / "zeroshot.pt")).cpu()
    base = anchor.state_dict()
    rows = []
    for entry in files:
        relative_path = entry["path"].split("/", 1)[1]
        path = root / relative_path
        sha = digest(path)
        assert sha == entry["sha256"] and path.stat().st_size == entry["bytes"], relative_path
        row = dict(
            path=relative_path,
            sha256=sha,
            bytes=path.stat().st_size,
            official_drive_id=entry["id"],
            checksum_matches_recorded_pin=True,
        )
        if relative_path.endswith("/finetuned.pt"):
            model = load(str(path)).cpu()
            state = model.state_dict()
            assert state.keys() == base.keys()
            assert all(
                t.shape == base[k].shape and torch.isfinite(t).all() for k, t in state.items()
            )
            row.update(
                tensor_fields=len(state),
                tensor_shapes_match_pretrained=True,
                finite=True,
                changed_fields=sum(not torch.equal(t, base[k]) for k, t in state.items()),
            )
            del model, state
        elif relative_path.startswith("head_"):
            task = relative_path[len("head_") : -len(".pt")]
            head = load(str(path))
            classes = {
                "Cars": 196,
                "DTD": 47,
                "EuroSAT": 10,
                "GTSRB": 43,
                "MNIST": 10,
                "RESISC45": 45,
                "SUN397": 397,
                "SVHN": 10,
            }
            assert head.weight.shape == (classes[task], 512)
            assert torch.isfinite(head.weight).all()
            row.update(task=task, head_shape=list(head.weight.shape))
        rows.append(row)
    report = dict(
        status="passed",
        official_experts=8,
        files=rows,
        new_pins="First successful official Drive acquisition hashes; not separately published author checksums",
        source_sha256=digest(__file__),
        config_sha256=digest("configs/checkpoints.json"),
        eight_task_accuracy_reported=False,
    )
    (output / "checkpoint_validation.json").write_text(json.dumps(report, indent=2))
    print(
        "CHECKPOINT_VERIFICATION",
        json.dumps(dict(status="passed", files=len(rows), official_experts=8)),
        flush=True,
    )


def expert_checks(args, root, config, output):
    """Measure original experts on ready official splits, without a merge claim."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    author_repo = Path(args.author_repo)
    load = checkpoint_loader(author_repo)
    package = types.ModuleType("dataset")
    package.__path__ = [str((author_repo / "datasets").resolve())]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    rows, predictions = [], {}
    for task in args.expert_checks:
        model = load(str(root / task / "finetuned.pt"))
        dataset = get_dataset(
            task, model.val_preprocess, args.data, batch_size=args.batch_size, num_workers=2
        )
        assert len(dataset.test_dataset) == config["expected_test_counts"][task]
        head = load(str(root / f"head_{task}.pt"))
        pred, labels = [], []

        def sink(p, y):
            pred.append(p)
            labels.append(y)

        metric = evaluate(model, head, dataset.test_loader, "cuda", prediction_sink=sink)
        predictions[task] = dict(prediction=torch.cat(pred), labels=torch.cat(labels))
        row = dict(
            stage="resource_validation",
            group="official_expert_check",
            dataset=task,
            eight_task_result=False,
            **metric,
        )
        rows.append(row)
        (output / "expert_checks.json").write_text(json.dumps(rows, indent=2))
        print("EXPERT_DATA_CHECK", json.dumps(row), flush=True)
        del model, head, dataset
    raw = output / "expert_predictions.pt"
    torch.save(predictions, raw)
    (output / "expert_check_metadata.json").write_text(
        json.dumps(
            dict(
                completed=True,
                source_sha256=digest(__file__),
                predictions_sha256=digest(raw),
                tasks=args.expert_checks,
                data_provider="unchanged_author_dataset_registry",
                eight_task_result=False,
            ),
            indent=2,
        )
    )


def pretrained_checks(args, root, config, output):
    """Validate ready data and heads; never present this subset as eight tasks."""
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    author_repo = Path(args.author_repo)
    load = checkpoint_loader(author_repo)
    package = types.ModuleType("dataset")
    package.__path__ = [str((author_repo / "datasets").resolve())]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    anchor = load(str(root / "zeroshot.pt"))
    rows = []
    for task in args.pretrained_checks:
        dataset = get_dataset(
            task, anchor.val_preprocess, args.data, batch_size=args.batch_size, num_workers=2
        )
        assert len(dataset.test_dataset) == config["expected_test_counts"][task], task
        head = load(str(root / f"head_{task}.pt"))
        row = dict(
            stage="resource_validation",
            group="pretrained_data_check",
            dataset=task,
            task_count=len(args.pretrained_checks),
            eight_task_result=False,
            data_provider="author_dataset_registry",
            **evaluate(anchor, head, dataset.test_loader, "cuda"),
        )
        rows.append(row)
        (output / "pretrained_checks.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print("PRETRAINED_DATA_CHECK", json.dumps(row), flush=True)


def run(args):
    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tasks = config["tasks"]
    assert len(tasks) == len(set(tasks)) == 8
    root, output = Path(args.checkpoints), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    required = [root / "zeroshot.pt"] + [
        path for task in tasks for path in [root / task / "finetuned.pt", root / f"head_{task}.pt"]
    ]
    missing = [str(path) for path in required if not path.is_file()]
    preflight = dict(
        tasks=tasks,
        coefficient=config["coefficient"],
        missing_checkpoints=missing,
        checkpoints_ready=not missing,
        data_splits_verified=False,
    )
    (output / "preflight.json").write_text(json.dumps(preflight, indent=2), encoding="utf-8")
    print("EIGHT_TASK_PREFLIGHT", json.dumps(preflight), flush=True)
    if args.verify_checkpoints:
        if missing:
            raise SystemExit("Checkpoint verification requires all official files")
        verify_checkpoints(args, root, config, output)
    if args.check_only:
        return
    if args.pretrained_checks:
        pretrained_checks(args, root, config, output)
        return
    if missing:
        raise SystemExit("Eight-task evaluation requires every official expert; see preflight.json")
    if args.expert_checks:
        expert_checks(args, root, config, output)
        return
    if args.verify_checkpoints:
        return
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    author_repo = Path(args.author_repo)
    load = checkpoint_loader(author_repo)
    from task_vectors import TaskVector

    # The author uses 'dataset' imports but the repository directory is 'datasets'.
    package = types.ModuleType("dataset")
    package.__path__ = [str((author_repo / "datasets").resolve())]
    sys.modules.setdefault("dataset", package)
    from dataset.registry import get_dataset

    anchor = load(str(root / "zeroshot.pt"))
    models = {task: load(str(root / task / "finetuned.pt")) for task in tasks}
    heads = {task: load(str(root / f"head_{task}.pt")) for task in tasks}
    datasets = {
        task: get_dataset(
            task, anchor.val_preprocess, args.data, batch_size=args.batch_size, num_workers=2
        )
        for task in tasks
    }
    for task, dataset in datasets.items():
        assert len(dataset.test_dataset) == config["expected_test_counts"][task], task
    overall = time.perf_counter()
    rows = []

    def record(group, model, eval_tasks, protected_task=None):
        for task in eval_tasks:
            metric = evaluate(model, heads[task], datasets[task].test_loader, "cuda")
            row = dict(
                stage="eight_task",
                group=group,
                dataset=task,
                protected_task=protected_task,
                tasks=tasks,
                coefficient=config["coefficient"],
                seed=args.seed,
                **metric,
            )
            rows.append(row)
            (output / "results.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
            print("EIGHT_TASK_RESULT", json.dumps(row), flush=True)

    def merged(candidates):
        vectors = [TaskVector(anchor.state_dict(), candidates[task].state_dict()) for task in tasks]
        return sum(vectors).apply_to(str(root / "zeroshot.pt"), scaling_coef=config["coefficient"])

    baseline = merged(models)
    record("unprotected_merge", baseline, tasks)
    del baseline
    if args.protected_task:
        task = args.protected_task
        assert task in tasks
        protected = protect(models[task], args.seed)
        x, _ = next(iter(datasets[task].test_loader))
        with torch.no_grad():
            original_features = models[task].cuda().eval()(x[:16].cuda()).cpu()
            protected_features = protected.cuda().eval()(x[:16].cuda()).cpu()
        models[task].cpu()
        protected.cpu()
        assert relative(protected_features, original_features) < 1e-4
        record("basis_protected_standalone", protected, [task], task)
        attack_dir = output / "protected_attack"
        attack_dir.mkdir(exist_ok=True)
        author, _ = load_author(author_repo / "utils_attack_B32.py")
        recovered = recover(protected, anchor, author, attack_dir)
        for group, candidate in [
            ("basis_protected_merge", protected),
            ("basis_AGA_merge", recovered),
        ]:
            candidates = dict(models, **{task: candidate})
            model = merged(candidates)
            record(group, model, tasks, task)
            del model
    metadata = dict(
        completed=True,
        total_seconds=time.perf_counter() - overall,
        tasks=tasks,
        coefficient=config["coefficient"],
        seed=args.seed,
        training="none",
        data_provider="author_dataset_registry",
        merge_implementation="author_TaskVector_apply_to",
        table6_reproduction=False,
        protection="orthogonal_plus_permutation" if args.protected_task else "none",
    )
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eight_tasks.json")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="outputs/2026-10-09/eight_task")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--verify-checkpoints", action="store_true")
    parser.add_argument(
        "--expert-checks",
        nargs="+",
        choices=["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"],
    )
    parser.add_argument(
        "--pretrained-checks",
        nargs="+",
        choices=["DTD", "GTSRB", "RESISC45"],
        help="Only check prepared data/heads; not an eight-task benchmark.",
    )
    parser.add_argument(
        "--protected-task",
        choices=["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261009)
    run(parser.parse_args())
