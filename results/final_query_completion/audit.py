"""One final-CLS Query public-completion diagnostic; no defense or training."""

from __future__ import annotations

import argparse
import copy
import hashlib
import inspect
import json
import sys
from pathlib import Path

import torch
from scipy.stats import binomtest
from torchvision.datasets import DTD, MNIST


def fingerprint(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        for value in [name, str(tensor.dtype), json.dumps(list(tensor.shape))]:
            h.update(value.encode() + b"\0")
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


@torch.no_grad()
def run(args):
    root = Path(args.root).resolve()
    out = root / "outputs/final_query_completion"
    out.mkdir(exist_ok=False)
    (out / "artifacts").mkdir()
    sys.path.insert(0, str(root / "experiments"))
    from clip_benchmark import checkpoint_loader, digest, evaluate, merge, paired_counts
    from subspace_probe import collect_moments, fit_basis, make_loader, save_json, split_training

    protocol = json.loads((root / "results/final_query_completion/protocol.json").read_text())
    assert digest(__file__) == protocol["runner_sha256"]
    for path, pin in protocol["source_sha256"].items():
        assert digest(root / path) == pin, path
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    save_json(out / "protocol.json", protocol)
    cp = root / "checkpoints/ViT-B-32"
    pins = json.loads((root / "configs/checkpoints.json").read_text())["files"]
    hashes = {}

    def verify_checkpoint(name):
        hashes[name] = digest(cp / name)
        assert hashes[name] == next(e["sha256"] for e in pins if e["path"] == "ViT-B-32/" + name)

    for name in ["zeroshot.pt", "MNIST/finetuned.pt", "head_MNIST.pt"]:
        verify_checkpoint(name)
    load = checkpoint_loader(root / "external/secure-merge-attack")
    anchor = load(cp / "zeroshot.pt").cpu().eval()
    original = load(cp / "MNIST/finetuned.pt").cpu().eval()
    model_source = Path(inspect.getfile(type(original.model.visual)))
    assert digest(model_source) == protocol["open_clip_model_sha256"]
    assert len(original.model.visual.transformer.resblocks) == 12
    train = MNIST(root / "data", train=True, transform=original.val_preprocess)
    indices = split_training(train, protocol["calibration_sizes"], protocol["seed"])
    save_json(out / "training_indices.json", indices)
    q = original.model.visual.transformer.resblocks[11].attn
    q0 = anchor.model.visual.transformer.resblocks[11].attn
    width = q.embed_dim
    assert width == 768
    normal = original.model.visual.transformer.resblocks[11].ln_1.weight.double().reciprocal()
    normal /= normal.norm()
    confirmation = collect_moments(
        original, make_loader(train, indices["confirmation"], 64), ["V:11"], token_index=0
    )["V:11"]
    models, artifacts, geometry = {}, {}, {}
    for name in ["defender", "attacker"]:
        moment = collect_moments(
            original, make_loader(train, indices[name], 64), ["V:11"], token_index=0
        )["V:11"]
        basis, stats = fit_basis(moment, protocol["retained_energy"])
        u = basis["u"]
        extra = normal - u @ (u.T @ normal)
        if extra.norm() > 1e-8:
            u = torch.cat([u, (extra / extra.norm())[:, None]], dim=1)
        assert (u.T @ u - torch.eye(u.shape[1])).norm() < 1e-8
        basis.update(u=u, rank=u.shape[1], complement_rank=width-u.shape[1])
        update = q.in_proj_weight[:width].double() - q0.in_proj_weight[:width].double()
        removed = update - (update @ u) @ u.T
        cleaned = copy.deepcopy(original).cpu().eval()
        a = cleaned.model.visual.transformer.resblocks[11].attn
        a.in_proj_weight[:width].copy_((q.in_proj_weight[:width].double() - removed).float())
        a.in_proj_bias[:width].copy_((q.in_proj_bias[:width].double() + removed @ basis["mean"]).float())
        actual = q.in_proj_weight[:width].double() - a.in_proj_weight[:width].double()
        stats.update(retained_rank_including_ln_null=u.shape[1],
                     usable_complement_rank=width-u.shape[1],
                     removed_task_update_energy_fraction=float(actual.square().sum()/update.square().sum()),
                     task_space_leakage=float((actual @ u).norm()/actual.norm()))
        cmean = confirmation["mean"] - basis["mean"]
        csecond = confirmation["covariance"] + cmean[:, None] @ cmean[None, :]
        stats["confirmation_query_change_rms"] = float(
            (actual @ csecond @ actual.T).trace().clamp_min(0).sqrt()/width**0.5)
        assert torch.equal(a.in_proj_weight[width:], q.in_proj_weight[width:])
        assert torch.equal(a.in_proj_bias[width:], q.in_proj_bias[width:])
        changed = {k for k, v in original.state_dict().items() if not torch.equal(v, cleaned.state_dict()[k])}
        assert changed == {"model.visual.transformer.resblocks.11.attn.in_proj_weight",
                           "model.visual.transformer.resblocks.11.attn.in_proj_bias"}
        models[name] = cleaned
        artifacts[name] = dict(basis=basis, moment=moment,
                               completed_weight=a.in_proj_weight[:width].clone(),
                               completed_bias=a.in_proj_bias[:width].clone())
        geometry[name] = stats
    capsule = dict(spaces=artifacts, confirmation_moment=confirmation,
                   query_weight=q.in_proj_weight[:width].clone(), query_bias=q.in_proj_bias[:width].clone(),
                   anchor_query_weight=q0.in_proj_weight[:width].clone(), ln1_normal=normal)
    torch.save(capsule, out / "artifacts/capsule.pt")
    save_json(out / "geometry.json", geometry)
    frozen = dict(protocol_sha256=digest(out / "protocol.json"), runner_sha256=digest(__file__),
                  source_sha256=protocol["source_sha256"], checkpoint_sha256=hashes.copy(),
                  state_fingerprints={"anchor": fingerprint(anchor), "original": fingerprint(original),
                                      **{k: fingerprint(v) for k, v in models.items()}},
                  capsule_sha256=digest(out / "artifacts/capsule.pt"), future_partner_loaded=False,
                  source_parent=protocol["source_parent"], new_protected_weights=0)
    save_json(out / "freeze.json", frozen)
    print("FROZEN", json.dumps(geometry), flush=True)

    # Only now read partner weights and historical evaluation records.
    for name in ["DTD/finetuned.pt", "head_DTD.pt"]:
        verify_checkpoint(name)
    partner = load(cp / "DTD/finetuned.pt").cpu().eval()
    ref = root / "outputs/query_replacement"
    ref_freeze = json.loads((ref / "freeze.json").read_text())
    ref_exec = json.loads((ref / "execution.json").read_text())
    validation = json.loads((ref / "independent_validation.json").read_text())
    assert validation["status"] == "passed"
    assert digest(ref / "artifacts/predictions.pt") == ref_exec["predictions_sha256"]
    for name in ["anchor", "original"]:
        assert frozen["state_fingerprints"][name] == ref_freeze["state_fingerprints"][name]
    assert fingerprint(partner) == ref_freeze["state_fingerprints"]["partner"]
    for name, pin in hashes.items():
        assert pin == ref_exec["checkpoint_sha256"][name]
    cached = torch.load(ref / "artifacts/predictions.pt", map_location="cpu")
    old_rows = json.loads((ref / "results.json").read_text())
    rows = [dict(r, cached_from="outputs/query_replacement", fresh_evaluation=False)
            for r in old_rows if r["preset"] == "original"]
    predictions = {"/".join(r[k] for k in ["preset", "stage", "dataset"]):
                   cached["/".join(r[k] for k in ["preset", "stage", "dataset"])] for r in rows}
    heads = {task: load(cp / f"head_{task}.pt").cpu().eval() for task in ["MNIST", "DTD"]}
    datasets = dict(MNIST=MNIST(root / "data", train=False, transform=original.val_preprocess),
                    DTD=DTD(root / "data", split="test", partition=1, transform=original.val_preprocess))
    assert {k: len(v) for k, v in datasets.items()} == {"MNIST": 10000, "DTD": 1880}
    for preset, model in models.items():
        fused = merge(anchor, model, partner, protocol["coefficient"])
        for stage, task, measured in [("standalone", "MNIST", model),
                                      ("direct", "MNIST", fused), ("direct", "DTD", fused)]:
            chunks = []
            metrics = evaluate(measured, heads[task], make_loader(datasets[task]), "cuda",
                               prediction_sink=lambda p, y: chunks.append((p, y)))
            pred, y = [torch.cat([v[i] for v in chunks]) for i in [0, 1]]
            old = predictions["original/" + stage + "/" + task]
            assert torch.equal(y, old["target"])
            key = preset + "/" + stage + "/" + task
            predictions[key] = dict(prediction=pred, target=y)
            paired = paired_counts(old["prediction"].eq(y), pred.eq(y))
            discordant = paired["lost"] + paired["gained"]
            pvalue = binomtest(paired["lost"], discordant).pvalue if discordant else 1.0
            row = dict(preset=preset, stage=stage, dataset=task, **metrics, paired=paired,
                       changed_predictions=int(pred.ne(old["prediction"]).sum()),
                       paired_two_sided_p=pvalue, fresh_evaluation=True)
            rows.append(row)
            save_json(out / "results.json", rows)
            torch.save(predictions, out / "artifacts/predictions.pt")
            print("RESULT", json.dumps(row), flush=True)
        del fused
    assert fingerprint(original) == frozen["state_fingerprints"]["original"]
    save_json(out / "execution.json", dict(completed=True, new_rows=6, cached_rows=4,
        predictions_sha256=digest(out / "artifacts/predictions.pt"),
        capsule_sha256=digest(out / "artifacts/capsule.pt"), checkpoint_sha256=hashes,
        protocol_sha256=digest(out / "protocol.json"), runner_sha256=digest(__file__),
        baseline_prediction_sha256=ref_exec["predictions_sha256"], gpu=torch.cuda.get_device_name(),
        optimizer_steps=0, new_protected_weights=0, AGA_run=False,
        scope="Useful public-completion cost, not a defense or exhaustive attack evaluation"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/root/autodl-tmp/merge_security_20261009")
    run(parser.parse_args())
