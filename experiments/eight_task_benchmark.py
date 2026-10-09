"""Eight-task baseline using upstream TaskVector and dataset registry unchanged.

Resource checks are executable without downloading or evaluating data. Missing
experts cause a hard stop; the task list is never reduced silently.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import types
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate, protect, recover
from mechanisms import load_author, relative


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
    if args.check_only:
        return
    if missing:
        raise SystemExit("Eight-task evaluation requires every official expert; see preflight.json")
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
    parser.add_argument(
        "--protected-task",
        choices=["SUN397", "Cars", "RESISC45", "EuroSAT", "SVHN", "GTSRB", "MNIST", "DTD"],
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20261009)
    run(parser.parse_args())
