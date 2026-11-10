"""Train a configured protection baseline and its matched CE control."""

import argparse
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from _impl.baselines import trap2_training, mergeguard_training
from _impl.data import dataset, official_labels
from _impl.io import digest, new_output, read_json, save_json
from _impl.models import initialize, load_checkpoint
from cfg import BACKBONE, CHECKPOINTS, DATA, require_root


def run(config, task, defense):
    require_root()
    initialize()
    cfg = read_json(config)
    if cfg.get("backbone", "ViT-B-32") != BACKBONE:
        raise ValueError("Training protocol and selected backbone differ")
    settings = cfg["training"] | {"device": "cuda"}
    settings["lambda_reg"] = cfg["lambda_reg"] if defense != "CE" else 0.0
    method = cfg.get("method", "Trap2")
    seed = cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    author = trap2_training(cfg["author_sources"]) if method == "Trap2" else None
    initial = "zeroshot.pt" if method == "Trap2" else f"{task}/finetuned.pt"
    encoder = load_checkpoint(initial)
    head = load_checkpoint(f"head_{task}.pt")
    train_data = dataset(task, str(DATA), encoder.val_preprocess, training=True)
    all_indices = np.arange(len(train_data))
    if task == "DTD":
        from torchvision.datasets import DTD

        train_indices = all_indices
        val_data = DTD(str(DATA), split="val", partition=1, transform=encoder.val_preprocess)
        val_indices = np.arange(len(val_data))
    else:
        labels = official_labels(task, training=True)
        assert len(labels) == len(train_data)
        generator = np.random.default_rng(seed)
        selected = []
        for label in np.unique(labels):
            indices = generator.permutation(np.flatnonzero(labels == label))
            selected.extend(indices[:int(len(indices) * cfg["validation_fraction"])])
        val_indices = np.sort(selected)
        train_indices = np.setdiff1d(all_indices, val_indices)
        val_data = train_data
        assert not np.intersect1d(train_indices, val_indices).size
    out = new_output(Path(cfg["output"]) / task / defense, cfg | {
        "task": task, "defense": defense, "training": settings,
        "train_indices": train_indices.tolist(), "validation_indices": val_indices.tolist(),
        "base_sha256": digest(CHECKPOINTS / "zeroshot.pt"),
        "initial_checkpoint_sha256": digest(CHECKPOINTS / initial),
        "head_sha256": digest(CHECKPOINTS / f"head_{task}.pt"),
        "driver_sha256": digest(__file__), "adapter_sha256": digest("_impl/baselines.py"),
    })
    train_loader = DataLoader(Subset(train_data, train_indices.tolist()),
                              batch_size=cfg["batch_size"], shuffle=True,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(Subset(val_data, val_indices.tolist()),
                            batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=2, pin_memory=True)
    visual = encoder.model.visual
    visual.requires_grad_(True)
    visual.proj.requires_grad_(method != "Trap2")
    vectors = head.weight.detach().float() / 100.0
    assert torch.allclose(vectors.norm(dim=1), torch.ones(len(vectors)), atol=1e-5)
    del head, encoder
    started = time.monotonic()
    if method == "Trap2":
        # Author loop logs a "Test" column; feed validation twice, never test images.
        author["train_cliphead_full_trap2"](
            visual, train_loader, val_loader, val_loader, vectors, settings, str(out / "best.pt"))
    else:
        base = load_checkpoint("zeroshot.pt").model.visual.state_dict()
        mergeguard_training(visual, base, train_loader, val_loader, vectors,
                            settings | {"epochs": cfg["epochs"][task]}, cfg, out, defense)
    assert (out / "best.pt").is_file()
    save_json(out / "completed.json", dict(
        task=task, defense=defense, training_seconds=time.monotonic() - started,
        checkpoint_sha256=digest(out / "best.pt"),
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        training_examples=len(train_indices), validation_examples=len(val_indices),
        test_examples_used=0,
        evaluation="Pending FP32 full-test evaluation; checkpoint selection is defined by the training protocol.",
    ))


@torch.no_grad()
def evaluate_pair(config, task):
    """Evaluate the two completed training runs through the shared TA/AGA code."""
    import copy

    from _impl.attacks import author_recover, load_author
    from _impl.evaluate import run as evaluate, verify

    initialize()
    cfg = read_json(config)
    if cfg.get("backbone", "ViT-B-32") != BACKBONE:
        raise ValueError("Evaluation protocol and selected backbone differ")
    method = cfg.get("method", "Trap2")
    folder = Path(cfg["output"]) / task
    evaluation = read_json("configs/evaluate.json")
    partner = "SVHN" if task == "MNIST" else "MNIST"
    evaluation["tasks"] = {k: evaluation["tasks"][k] for k in [task, partner]}
    evaluation["cases"] = [dict(target=task, partner=partner, defenses=[method])]
    evaluation["defenses"] = {}
    evaluation["backbone"] = BACKBONE
    evaluation["attack_scopes"] = cfg.get("attack_scopes", ["None", "Protected expert", "All experts"])
    use_aga = any(s != "None" for s in evaluation["attack_scopes"])
    if use_aga and BACKBONE == "ViT-L-14":
        raise ValueError("This AGA adapter covers 12-layer B models, not ViT-L/14")
    evaluation["output"] = str(folder / "evaluation")
    evaluation["experiment"] = f"{method} and matched CE control; original fixed companion"
    evaluation["source_sha256"]["experiment/train_baseline.py"] = digest(__file__)
    evaluation["source_sha256"]["_impl/baselines.py"] = digest("_impl/baselines.py")
    anchor = load_checkpoint("zeroshot.pt")
    path = Path("external/secure-merge-attack") / evaluation["AGA"]["file"]
    assert digest(path) == evaluation["AGA"]["raw_sha256"]
    if use_aga:
        author, sha = load_author(path, True)
        assert sha == evaluation["AGA"]["sha256"]
    for defense, name in [("CE", "Unprotected"), (method, method)]:
        run_dir = folder / defense
        completed = read_json(run_dir / "completed.json")
        assert digest(run_dir / "best.pt") == completed["checkpoint_sha256"]
        model = copy.deepcopy(anchor)
        state = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
        model.model.visual.load_state_dict(state, strict=True)
        release = run_dir / "release.pt"
        torch.save({"model.visual." + k: v for k, v in state.items()}, release)
        evaluation["defenses"][name] = dict(
            release=dict(path=str(release), sha256=digest(release), reference="public_base"),
        )
        if use_aga:
            recovered = author_recover(model, anchor, author)
            aligned = run_dir / "AGA.pt"
            torch.save({k: v for k, v in recovered.state_dict().items()
                        if not torch.equal(v, anchor.state_dict()[k])}, aligned)
            evaluation["defenses"][name]["AGA_edits"] = dict(
                path=str(aligned), sha256=digest(aligned), reference="public_base")
            del recovered
        del model
    protocol = folder / "evaluation.json"
    save_json(protocol, evaluation)
    evaluate(protocol, 0, False)
    verify(folder / "evaluation" / "full" / f"00_{task}_{partner}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/trap2_baseline.json")
    parser.add_argument("--task", choices=["DTD", "MNIST", "SVHN"], required=True)
    parser.add_argument("--defense", choices=["Trap2", "MergeGuard", "CE"])
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    if not args.evaluate and args.defense is None:
        parser.error("Training requires --defense")
    evaluate_pair(args.config, args.task) if args.evaluate else run(args.config, args.task, args.defense)
