"""Frozen-weight, target-only author AGA audit; no defense construction or tuning."""

import argparse
import contextlib
import copy
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import author_recover, checkpoint_loader, digest, evaluate, merge, paired_counts
from covariance_noise import dataset
from mechanisms import load_author
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels
from verify_probe_candidate import explicit_merge


def logical_parameter_distance(edits,original):
    total=0.
    for k,v in edits.items():
        a,b=v.double(),original[k].double()
        pieces=zip(a.chunk(3,0),b.chunk(3,0)) if k.endswith(('attn.in_proj_weight','attn.in_proj_bias')) else [(a,b)]
        for changed,before in pieces:
            if float(before.norm())==0:assert torch.equal(changed,before);continue
            total+=float((changed-before).norm().square()/before.norm().square())
    return total**.5


def setup(cfg):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert "3090" in torch.cuda.get_device_name()
    for p, sha in cfg["source_sha256"].items():
        assert digest(p) == sha, p
    root = Path("checkpoints/ViT-B-32")
    pins = {x["path"].removeprefix("ViT-B-32/"): x["sha256"]
            for x in json.loads(Path("configs/checkpoints.json").read_text())["files"]}
    load = checkpoint_loader(Path("external/secure-merge-attack"))

    def checked(name):
        assert digest(root / name) == pins[name], name
        model = load(str(root / name)).cpu().eval()
        model.requires_grad_(False)
        return model

    anchor, own, peer = [checked(p) for p in ["zeroshot.pt", "MNIST/finetuned.pt", "SVHN/finetuned.pt"]]
    heads = {t: checked(f"head_{t}.pt") for t in ["MNIST", "SVHN"]}
    ref = cfg["release"]
    assert digest(ref["path"]) == ref["sha256"]
    released = copy.deepcopy(own)
    edits = torch.load(ref["path"], map_location="cpu")
    assert sorted(edits) == cfg["allowed_keys"]
    for k, v in edits.items():
        released.state_dict()[k].copy_(v)
    distance = sum(float(((v.double() - own.state_dict()[k].double()).norm()
                          / own.state_dict()[k].double().norm()) ** 2) for k, v in edits.items()) ** .5
    if cfg.get('logical_qkv_budget'):distance=logical_parameter_distance(edits,own.state_dict())
    assert distance <= cfg.get('parameter_budget',.5)
    confirmation = json.loads(Path(cfg["prior_confirmation"]).read_text())
    assert confirmation["status"] == "passed"
    return anchor, own, peer, released, heads, distance


@torch.no_grad()
def run(protocol, output):
    cfg = json.loads(Path(protocol).read_text())
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    save_json(out / "protocol.json", cfg)
    start = time.monotonic()
    anchor, own, peer, released, heads, distance = setup(cfg)
    # All defense parameters were frozen long before loading test data or peer.
    save_json(out / "pre_evaluation.json", dict(release=cfg["release"], distance=distance,
              source_sha256=cfg["source_sha256"], protocol_sha256=digest(protocol),
              new_candidates=0, optimization_steps=0, selection_from_results=False))
    data = {t: dataset(t, "data", own.val_preprocess) for t in heads}
    assert {t: len(d) for t, d in data.items()} == cfg["test_counts"]
    rows, raw = [], {}

    def record(model, task, preset, stage):
        sink, logits = [], []
        metric = evaluate(model, heads[task], make_loader(data[task]), "cuda",
                          prediction_sink=lambda p, y: sink.append((p, y)),
                          logit_sink=logits.append)
        pred, y = [torch.cat([b[i] for b in sink]) for i in [0, 1]]
        key = f"{preset}/{stage}/{task}"
        raw[key] = dict(prediction=pred, target=y, logits=torch.cat(logits))
        row = dict(preset=preset, stage=stage, dataset=task, **metric,
                   per_class=[dict(class_id=c, n=int(y.eq(c).sum()),
                                   correct=int((pred.eq(y) & y.eq(c)).sum())) for c in range(10)])
        if preset == "released":
            ref = raw[f"original/{stage}/{task}"]
            assert torch.equal(y, ref["target"])
            row.update(paired=paired_counts(ref["prediction"].eq(y), pred.eq(y)),
                       changed_predictions=int(pred.ne(ref["prediction"]).sum()))
        rows.append(row)
        save_json(out / "results.json", rows)
        torch.save(raw, out / "artifacts/predictions.pt")
        print("PROTOCOL_RESULT", json.dumps(row), flush=True)
        assert time.monotonic() - start < cfg["wall_limit_seconds"]

    for preset, model in [("original", own), ("released", released)]:
        record(model, "MNIST", preset, "standalone")
    record(peer, "SVHN", "original", "partner_standalone")
    for preset, model in [("original", own), ("released", released)]:
        fused = merge(anchor, model, peer, .5)
        for t in heads: record(fused, t, preset, "direct")
        del fused
    author, pin = load_author(Path("external/secure-merge-attack/utils_attack_B32.py"), True)
    assert pin == cfg["AGA_text_sha256"]
    attacked = {}
    for preset, model in [("original", own), ("released", released)]:
        log = io.StringIO()
        with contextlib.redirect_stdout(log): attacked[preset] = author_recover(model, anchor, author)
        text = log.getvalue(); (out / f"AGA_{preset}.log").write_text(text)
        assert not any(x in text for x in ["Skipping", "Warning:", "Error", "[Error]"])
        assert all(f"Attacking Layer {i}..." in text for i in range(12))
        assert len(model.model.visual.transformer.resblocks) == 12
        # The unchanged author loop visits every block but logs only i % 4 == 0.
        assert all(f"Block {i}: MLP alignment applied." in text for i in [0, 4, 8])
        assert ">>> MLP-only attack completed." in text
        record(attacked[preset], "MNIST", preset, "AGA_standalone")
        # Matches the public entrypoint: the peer is vanilla, never attacked here.
        fused = merge(anchor, attacked[preset], peer, .5)
        for t in heads: record(fused, t, preset, "AGA_target_only")
        del fused
    ref = cfg["AGA_original_reference"]
    assert digest(ref["path"]) == ref["sha256"]
    previous = torch.load(ref["path"], map_location="cpu")
    assert all(torch.equal(v, previous[k]) for k, v in attacked["original"].state_dict().items())
    changes = {k: v.clone() for k, v in attacked["released"].state_dict().items()
               if not torch.equal(v, previous[k])}
    torch.save(changes, out / "artifacts/AGA_release_edits.pt")
    save_json(out / "execution.json", dict(completed=True, new_candidates=0, optimizer_steps=0,
              parameter_distance=distance, original_expert_updated=False, peer_attack=False,
              author_attention_layers=12, author_mlp_layers=12, AGA_reference_exact=True,
              predictions_sha256=digest(out / "artifacts/predictions.pt"),
              AGA_edits_sha256=digest(out / "artifacts/AGA_release_edits.pt"),
              release_sha256=cfg["release"]["sha256"], wall_seconds=time.monotonic()-start,
              gpu=torch.cuda.get_device_name(), first_phase_changed=False))


@torch.no_grad()
def verify(output):
    out = Path(output); cfg = json.loads((out / "protocol.json").read_text())
    meta = json.loads((out / "execution.json").read_text()); assert meta["completed"]
    for name, field in [("predictions.pt", "predictions_sha256"), ("AGA_release_edits.pt", "AGA_edits_sha256")]:
        assert digest(out / "artifacts" / name) == meta[field]
    anchor, own, peer, released, heads, distance = setup(cfg)
    ref = cfg["AGA_original_reference"]; assert digest(ref["path"]) == ref["sha256"]
    own_a = copy.deepcopy(own); own_a.load_state_dict(torch.load(ref["path"], map_location="cpu"), strict=True)
    release_a = copy.deepcopy(own_a)
    for k, v in torch.load(out / "artifacts/AGA_release_edits.pt", map_location="cpu").items():
        release_a.state_dict()[k].copy_(v)
    data = {t: dataset(t, "data", own.val_preprocess) for t in heads}
    rows = json.loads((out / "results.json").read_text())
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu"); replays = 0
    for row in rows:
        p, s, t = [row[k] for k in ["preset", "stage", "dataset"]]
        r = raw[f"{p}/{s}/{t}"]; y = official_labels(t); pred = r["prediction"].numpy()
        np.testing.assert_array_equal(r["target"].numpy(), y)
        np.testing.assert_array_equal(r["logits"].numpy().argmax(1), pred)
        assert np.isfinite(r["logits"].numpy()).all()
        assert len(y) == row["n"] and int((pred == y).sum()) == row["correct"]
        for c in row["per_class"]:
            mask = y == c["class_id"]
            assert int(mask.sum()) == c["n"] and int(((pred == y) & mask).sum()) == c["correct"]
        if p == "released":
            before = raw[f"original/{s}/{t}"]["prediction"].numpy()
            lost, gained = int(((before == y) & (pred != y)).sum()), int(((before != y) & (pred == y)).sum())
            assert (lost, gained) == (row["paired"]["lost"], row["paired"]["gained"])
            assert int((before != pred).sum()) == row["changed_predictions"]
        if s == "partner_standalone": model = peer
        elif s == "standalone": model = own if p == "original" else released
        elif s == "AGA_standalone": model = own_a if p == "original" else release_a
        else:
            assert s in ["direct", "AGA_target_only"]
            left = (own if p == "original" else released) if s == "direct" else (own_a if p == "original" else release_a)
            model = explicit_merge(anchor, left, peer, .5)
        chosen = np.linspace(0, len(y)-1, 16, dtype=int)
        images = torch.stack([data[t][int(i)][0] for i in chosen]).cuda()
        model.cuda(); heads[t].cuda()
        actual = heads[t](model(images)).argmax(1).cpu().numpy()
        model.cpu(); heads[t].cpu(); np.testing.assert_array_equal(actual, pred[chosen]); replays += 16
    save_json(out / "independent_validation.json", dict(status="passed", source_sha256=digest(__file__),
              rows=len(rows), official_label_counts=True, numpy_per_class_and_pair_counts=True,
              explicit_TA_replays=replays, predictions_sha256=meta["predictions_sha256"],
              distance=distance, new_candidates=0, first_phase_changed=False))
    print("INDEPENDENT_PROTOCOL_AUDIT_PASSED", len(rows), replays, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/author_protocol_audit.json")
    parser.add_argument("--output", required=True); parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify(args.output) if args.verify else run(args.protocol, args.output)
