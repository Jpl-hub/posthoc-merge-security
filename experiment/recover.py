"""Evaluate recovery of frozen publications with explicit data and edit scopes."""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from _impl.attacks import author_recover, load_author, public_mlp_matching, public_mlp_reset, public_patch_recovery
from _impl.io import digest, new_output, save_json
from _impl.models import apply, initialize, load_checkpoint
from cfg import AUTHOR, load_config, require_root


@torch.no_grad()
def readout_run(protocol):
    """Matched selective-projection and pre-normalization centering checks."""
    from _impl.data import dataset, make_loader, official_labels
    from _impl.io import check_files
    from _impl.merge import merge, parameter_hash

    cfg = json.loads(Path(protocol).read_text())
    check_files(cfg["source_sha256"])
    initialize()
    out = new_output(cfg["output"], cfg)
    base = load_checkpoint("zeroshot.pt")
    expert = load_checkpoint(f"{cfg['target']}/finetuned.pt")
    peer = load_checkpoint(f"{cfg['partner']}/finetuned.pt")
    protected = apply(expert, cfg["publication"])
    tasks = [cfg["target"], cfg["partner"]]
    heads = {t: load_checkpoint(f"head_{t}.pt").cuda() for t in tasks}
    train = {t: dataset(t, "data", expert.val_preprocess, training=True) for t in tasks}
    test = {t: dataset(t, "data", expert.val_preprocess) for t in tasks}
    rng = np.random.default_rng(cfg["seed"])
    ids = {t: sorted(rng.choice(len(train[t]), cfg["images_per_task"], replace=False).tolist())
           for t in tasks}
    save_json(out / "calibration_indices.json", dict(split="official training", labels_used=False, indices=ids))
    rows, fingerprints, means = [], {}, {}
    for name, own in [("None", expert), ("NBP", protected)]:
        merged = merge(base, own, peer, cfg["coefficient"])
        for operation in ["TA", "Expert projection"]:
            model = copy.deepcopy(merged)
            if operation == "Expert projection":
                model.model.visual.proj.copy_(own.model.visual.proj)
                assert torch.equal(model.model.visual.proj, own.model.visual.proj)
            fingerprints[f"{name}/{operation}"] = parameter_hash(model.state_dict())
            model.cuda().eval()
            mean = None
            if operation == "TA":
                features = []
                for task in tasks:
                    for images, _ in make_loader(train[task], ids[task], cfg["batch_size"]):
                        features.append(model(images.cuda()).double().cpu())
                mean = torch.cat(features).mean(0).float().cuda()
                means[name] = mean.cpu().numpy()
            for task in tasks:
                labels = official_labels(task)
                assert len(test[task]) == len(labels) == cfg["test_sizes"][task]
                scores, centered = [], []
                start = time.monotonic()
                for images, _ in make_loader(test[task], batch_size=cfg["batch_size"]):
                    features = model(images.cuda())
                    scores.append(heads[task](features).cpu())
                    if mean is not None:
                        centered.append(heads[task](features - mean).cpu())
                candidates = [(operation, torch.cat(scores))]
                if centered:
                    candidates.append(("Feature centering", torch.cat(centered)))
                for treatment, logits in candidates:
                    values = logits.numpy()
                    assert np.isfinite(values).all()
                    pred = values.argmax(1)
                    classes = values.shape[1]
                    counts = np.bincount(pred, minlength=classes)
                    frequency = counts[counts > 0] / len(pred)
                    path = out / f"{name}_{treatment.replace(' ', '_')}_{task}.npz"
                    np.savez_compressed(path, scores=values, labels=labels)
                    row = dict(defense=name, recovery=treatment, task=task,
                        role="target" if task == cfg["target"] else "partner",
                        n=len(labels), correct=int((pred == labels).sum()),
                        accuracy_percent=float(100*np.mean(pred == labels)),
                        predicted_classes=int((counts > 0).sum()),
                        dominant_fraction=float(counts.max()/len(pred)),
                        prediction_entropy_nats=float(-(frequency*np.log(frequency)).sum()),
                        balanced_accuracy_percent=float(100*np.mean([
                            np.mean(pred[labels == c] == c) for c in range(classes)])),
                        raw_file=path.name, raw_sha256=digest(path),
                        inference_seconds=time.monotonic()-start)
                    rows.append(row)
                    save_json(out / "results.json", dict(rows=rows, completed=False))
                    print("READOUT_RECOVERY", name, treatment, task, row["correct"], row["n"], flush=True)
                # Replay one complete original-size batch through the same model/head.
                images = torch.stack([test[task][i][0] for i in range(min(cfg["batch_size"],len(labels)))]).cuda()
                replay_features = model(images)
                for treatment, logits in candidates:
                    actual = heads[task](replay_features - mean if treatment == "Feature centering" else replay_features).cpu()
                    torch.testing.assert_close(actual, logits[:len(images)], rtol=0, atol=5e-4)
            model.cpu()
            del model
        del merged
    np.savez_compressed(out / "calibration_means.npz", **means)
    for row in rows:
        data = np.load(out / row["raw_file"])
        np.testing.assert_array_equal(data["labels"], official_labels(row["task"]))
        assert int((data["scores"].argmax(1) == data["labels"]).sum()) == row["correct"]
    agreements = []
    for treatment in ["TA", "Expert projection", "Feature centering"]:
        for task in tasks:
            a, b = [np.load(out / f"{name}_{treatment.replace(' ', '_')}_{task}.npz")["scores"] for name in ["None", "NBP"]]
            agreements.append(dict(recovery=treatment, task=task,
                changed_predictions=int((a.argmax(1) != b.argmax(1)).sum()),
                maximum_logit_difference=float(np.max(np.abs(a-b)))))
    save_json(out / "results.json", dict(rows=rows, completed=True,
        matched_prediction_comparison=agreements, model_parameter_sha256=fingerprints,
        calibration_mean_shift_norm=float(np.linalg.norm(means["NBP"]-means["None"])),
        numpy_recount_passed=True, first_batch_replay_passed=True,
        dtype="FP32", TF32=False, GPU=torch.cuda.get_device_name(0),
        calibration_images=2*cfg["images_per_task"], calibration_labels_used=False,
        optimizer_steps=0, new_defense_weights=0))


def main():
    if "--readout-evaluation" in sys.argv:
        p = argparse.ArgumentParser(description=readout_run.__doc__)
        p.add_argument("--readout-evaluation", required=True)
        a = p.parse_args()
        require_root()
        readout_run(a.readout_evaluation)
        return
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--target", choices=["SUN397", "Cars", "RESISC45", "SVHN", "GTSRB", "MNIST", "DTD"], required=True)
    p.add_argument("--method", choices=["aga", "patch", "reset", "mlp-match"], default="aga")
    p.add_argument("--protocol", default="configs/evaluate.json")
    p.add_argument("--publication", required=True)
    p.add_argument("--sha256", required=True)
    p.add_argument("--full-state", action="store_true")
    p.add_argument(
        "--aligned-reference",
        action="store_true",
        help="Store AGA differences from the already frozen AGA of the clean expert",
    )
    p.add_argument(
        "--base-reference",
        action="store_true",
        help="Apply and store edits relative to the public base",
    )
    p.add_argument("--output", required=True)
    a = p.parse_args()
    if a.aligned_reference and (a.method != "aga" or a.base_reference):
        p.error("aligned-reference requires aga without base-reference")
    require_root()
    initialize()
    protocol = vars(a)
    out = new_output(a.output, protocol)
    base = load_checkpoint("zeroshot.pt")
    expert = load_checkpoint(f"{a.target}/finetuned.pt")
    reference = base if a.base_reference else expert
    published = apply(reference, dict(path=a.publication, sha256=a.sha256), a.full_state)
    start = time.monotonic()
    if a.method == "aga":
        cfg = load_config(a.protocol)
        source = AUTHOR / cfg["AGA"]["file"]
        if digest(source) != cfg["AGA"]["raw_sha256"]:
            raise ValueError("AGA source mismatch")
        author, sha = load_author(source, True)
        if sha != cfg["AGA"]["sha256"]:
            raise ValueError("AGA text mismatch")
        result = author_recover(published, base, author).state_dict()
        detail = dict(known_family=False, author_text_sha256=sha)
        if a.aligned_reference:
            ref = cfg["AGA_references"][a.target]
            reference = apply(expert, ref, full=True)
            detail["aligned_reference"] = ref
    elif a.method == "patch":
        # Recovery gets only the full published state and public base. The clean
        # expert above assembles an edits-only publication; it is not passed here.
        result, detail = public_patch_recovery(published.state_dict(), base.state_dict())
        detail["known_family"] = True
    elif a.method == "mlp-match":
        result, layers = public_mlp_matching(published.state_dict(), base.state_dict())
        detail = dict(layers=layers, images=0, original_expert_used_by_attack=False,
                      source="samuela/git-re-basin@ef40098257ab97243930eba737d6dcb8edd5863e",
                      scope="joint incoming/bias/outgoing MLP matching; attention and residual coordinates fixed; native adaptation")
    else:
        result = public_mlp_reset(published.state_dict(), base.state_dict())
        detail = dict(
            known_modified_modules=True,
            reset_tensors=sum(".mlp." in k for k in result),
            images=0,
            original_expert_used_by_attack=False,
        )
        reference = base
    edits = {k: v for k, v in result.items() if not torch.equal(v, reference.state_dict()[k])}
    assert all(torch.equal(v, (reference.state_dict() | edits)[k]) for k, v in result.items())
    torch.save(edits, out / "edits.pt")
    save_json(
        out / "execution.json",
        dict(
            method=a.method,
            seconds=time.monotonic() - start,
            output_sha256=digest(out / "edits.pt"),
            details=detail,
            reference="original_AGA"
            if a.aligned_reference
            else ("public_base" if a.base_reference or a.method == "reset" else "expert"),
            new_training_updates=0,
        ),
    )


if __name__ == "__main__":
    main()
