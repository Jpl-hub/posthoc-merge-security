"""Replay a frozen complement experiment after moving to another GPU.

No parameter search, new candidate or new defense claim. Uses the recorded
release, calibration basis and protocol; checks both controls under original AGA.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, evaluate, merge, paired_counts, recover
from mechanisms import load_author
from subspace_probe import make_loader, perturb, save_json
from torchvision.datasets import MNIST


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


@torch.no_grad()
def run(args):
    reference, output = Path(args.reference), Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = json.loads((reference / "protocol.json").read_bytes())
    selected = json.loads((reference / "selection.json").read_bytes())["selected"]
    assert selected["site"] == "V:0" and selected["strength"] == 100
    assert protocol["coefficient"] == 0.5 and protocol["seed"] == 20261009
    source_pin = json.loads(Path("configs/sources.json").read_bytes())["AGA"]
    author, author_sha = load_author(Path(args.author_repo) / "utils_attack_B32.py")
    assert author_sha == source_pin["sha256"]
    root = Path(args.checkpoints)
    load = checkpoint_loader(Path(args.author_repo))
    anchor = load(str(root / "zeroshot.pt")).cpu().eval()
    original = load(str(root / "MNIST/finetuned.pt")).cpu().eval()
    partner = load(str(root / "SVHN/finetuned.pt")).cpu().eval()
    head = load(str(root / "head_MNIST.pt")).cpu().eval()
    pins = json.loads(Path("configs/checkpoints.json").read_bytes())["files"]
    checkpoint_hashes = {}
    for name in ["zeroshot.pt", "MNIST/finetuned.pt", "SVHN/finetuned.pt", "head_MNIST.pt"]:
        sha = digest(root / name)
        expected = next(p["sha256"] for p in pins if p["path"] == "ViT-B-32/" + name)
        assert sha == expected, name
        checkpoint_hashes[name] = sha
    state_path = reference / "artifacts/protected_state.pt"
    state = torch.load(state_path, map_location="cpu")
    bases = torch.load(reference / "artifacts/defender_bases.pt", map_location="cpu")
    rebuilt, leakage = perturb(
        original, selected["site"], bases[selected["site"]], selected["strength"], protocol["seed"]
    )
    differences = {
        key: float((tensor - state[key]).abs().max())
        for key, tensor in rebuilt.state_dict().items()
        if not torch.equal(tensor, state[key])
    }
    save_json(
        output / "reconstruction.json",
        dict(exact=not differences, differences=differences, **leakage),
    )
    assert not differences, "Frozen release reconstruction differs; stop before evaluating."
    del rebuilt, bases
    released = copy.deepcopy(original).cpu().eval()
    released.load_state_dict(state, strict=True)
    del state
    dataset = MNIST(args.data, train=False, transform=original.val_preprocess)
    assert len(dataset) == 10000
    loader = make_loader(dataset)
    expected_rows = json.loads((reference / "results.json").read_bytes())
    rows, predictions = [], {}
    for name, model in [("unprotected", original), (selected["name"], released)]:
        attack_dir = output / name
        attack_dir.mkdir()
        for stage in ["standalone", "direct", "AGA"]:
            if stage == "standalone":
                tested = model
            elif stage == "direct":
                tested = merge(anchor, model, partner, protocol["coefficient"])
            else:
                attacked = recover(model, anchor, author, attack_dir)
                tested = merge(anchor, attacked, partner, protocol["coefficient"])
                del attacked
            saved = []
            metric = evaluate(tested, head, loader, "cuda", lambda p, y: saved.append((p, y)))
            pred, target = [torch.cat([pair[i] for pair in saved]) for i in range(2)]
            predictions[name + "/" + stage] = dict(prediction=pred, target=target)
            prior = next(
                r
                for r in expected_rows
                if r["split"] == "official_test"
                and r["dataset"] == "MNIST"
                and r["preset"] == name
                and r["group"] == stage
            )
            row = dict(preset=name, stage=stage, **metric, expected_correct=prior["correct"])
            row["count_matches_reference"] = metric["correct"] == prior["correct"]
            if name != "unprotected":
                control = predictions["unprotected/" + stage]
                assert torch.equal(target, control["target"])
                row["paired"] = paired_counts(control["prediction"].eq(target), pred.eq(target))
            rows.append(row)
            save_json(output / "results.json", rows)
            print("REPLAY", json.dumps(row), flush=True)
            if stage != "standalone":
                del tested
    torch.save(predictions, output / "predictions.pt")
    save_json(
        output / "validation.json",
        dict(
            completed=True,
            exact_release_reconstruction=True,
            all_six_counts_match=all(r["count_matches_reference"] for r in rows),
            scope="Historical two-task migration check; no new candidate or parameter search.",
            selection_used_known_partner=True,
            gpu=torch.cuda.get_device_name(),
            author_source_normalized_sha256=author_sha,
            checkpoint_sha256=checkpoint_hashes,
            release_sha256=digest(state_path),
            protocol_sha256=digest(reference / "protocol.json"),
            source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            predictions_sha256=digest(output / "predictions.pt"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
