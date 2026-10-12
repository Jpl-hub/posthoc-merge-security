"""One middle-layer complement diagnostic; no noise or protected candidate.

Freeze one target-only input PCA, then measure real merging and public completion.
An independent public training subset checks whether the attacker needs that PCA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, digest, evaluate, merge, paired_counts
from subspace_probe import (
    clean_complement,
    collect_moments,
    complement_power,
    fit_basis,
    make_loader,
    save_json,
    split_training,
)
from torchvision.datasets import DTD, MNIST


@torch.no_grad()
def task_basis(model, dataset, indices):
    moment = collect_moments(model, make_loader(dataset, indices, 64), ["FC:5"])["FC:5"]
    basis, stats = fit_basis(moment, 0.9999)
    # Centered LayerNorm outputs satisfy n^T(x-mu)=0, n=1/gamma. Do not
    # interpret this architecture-wide affine null as a task-specific complement.
    normal = model.model.visual.transformer.resblocks[5].ln_2.weight.double().reciprocal()
    normal /= normal.norm()
    u = basis["u"]
    residual = normal - u @ (u.T @ normal)
    if residual.norm() > 1e-8:
        u = torch.cat([u, (residual / residual.norm())[:, None]], dim=1)
    basis.update(u=u, rank=u.shape[1], complement_rank=768 - u.shape[1])
    stats.update(
        retained_rank_including_ln_null=u.shape[1],
        usable_complement_rank=768 - u.shape[1],
        layernorm_null_leakage=float((normal - u @ (u.T @ normal)).norm()),
        orthogonality_error=float((u.T @ u - torch.eye(u.shape[1])).norm()),
    )
    assert stats["orthogonality_error"] < 1e-8
    return basis, stats, moment


@torch.no_grad()
def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = dict(
        question="Is a target-inactive middle input complement activated by real merging, and does public completion lose useful capability?",
        target="MNIST",
        partner="DTD",
        site="FC:5",
        location_rule="Block 5 is the end of the first half of 12 blocks; c_fc is the first nonlinear MLP input. Structural choice, no performance-based search.",
        retained_energy=0.9999,
        rank_rule="Historical fixed PCA energy rule, all tokens; additionally retain LayerNorm affine-null normal 1/gamma to exclude a universal null direction.",
        data="256 MNIST training images construct PCA; 256 disjoint public training images construct attacker PCA; 256 further disjoint MNIST training images confirm. Labels unused in construction.",
        seed=20261013,
        completion="D=(W-W0)(I-UU^T); Wc=W-D; bc=b+D mu. Preserves the estimated affine task subspace, not all unseen inputs.",
        independent_attack="Recompute PCA from separate public MNIST training images. Prefix before edited c_fc is unchanged, hence the same estimate is available from any release limited to this site.",
        diagnostics="Same confirmation MNIST inputs under native/merged models; 256 DTD training inputs are measured only after basis/weights freeze. No partner-based selection.",
        coefficient=0.5,
        metrics="Primary: MNIST native and merged capability. Also report DTD merged capability; partner loss alone is not AGA Eq.3 success.",
        stop="If native-to-merged complement activation is absent or public completion retains useful fusion capability, stop this one region; no noise construction or alternate layer/rank/amplitude sweep.",
        limitation="One location and one existing pair, observed test splits; PCA energy is not a classification subspace theorem or full NUFILT replication.",
        new_defense_candidates=0,
    )
    save_json(out / "protocol.json", protocol)
    root = Path(args.checkpoints)
    pins = json.loads(Path("configs/checkpoints.json").read_bytes())["files"]
    hashes = {}
    for name in ["zeroshot.pt", "MNIST/finetuned.pt", "head_MNIST.pt"]:
        hashes[name] = digest(root / name)
        assert hashes[name] == next(p["sha256"] for p in pins if p["path"] == "ViT-B-32/" + name)
    load = checkpoint_loader(Path(args.author_repo))
    anchor = load(str(root / "zeroshot.pt")).cpu().eval()
    original = load(str(root / "MNIST/finetuned.pt")).cpu().eval()
    train = MNIST(args.data, train=True, transform=original.val_preprocess)
    indices = split_training(train, dict(defender=256, attacker=256, confirmation=256), 20261013)
    save_json(out / "training_indices.json", indices)
    spaces, geometry, moments, models = {}, {}, {}, {"original": original}
    for name in ["defender", "attacker"]:
        spaces[name], geometry[name], moments[name] = task_basis(original, train, indices[name])
        models[name] = clean_complement(original, anchor, "FC:5", spaces[name], "completion")
        before = original.model.visual.transformer.resblocks[5].mlp.c_fc
        after = models[name].model.visual.transformer.resblocks[5].mlp.c_fc
        delta = before.weight.double() - after.weight.double()
        update = (
            before.weight.double()
            - anchor.model.visual.transformer.resblocks[5].mlp.c_fc.weight.double()
        )
        geometry[name].update(
            removed_task_update_energy_fraction=float(delta.square().sum() / update.square().sum()),
            affine_preservation_error=float((delta @ spaces[name]["u"]).norm() / delta.norm()),
        )
        torch.save(
            dict(basis=spaces[name], moment=moments[name], completed=after.state_dict()),
            out / f"artifacts/{name}.pt",
        )
    a, b = spaces["defender"]["u"], spaces["attacker"]["u"]
    geometry["independent_projector"] = dict(
        relative_frobenius_difference=float((a @ a.T - b @ b.T).norm() / (a @ a.T).norm())
    )
    save_json(out / "geometry.json", geometry)
    save_json(
        out / "freeze.json",
        dict(
            checkpoint_sha256=hashes,
            protocol_sha256=digest(out / "protocol.json"),
            source_sha256={p.name: digest(p) for p in Path(__file__).parent.glob("*.py")},
            artifacts_sha256={p.name: digest(p) for p in (out / "artifacts").glob("*.pt")},
            future_partner_loaded=False,
        ),
    )
    print("FROZEN", json.dumps(geometry), flush=True)
    for name in ["DTD/finetuned.pt", "head_DTD.pt"]:
        hashes[name] = digest(root / name)
        assert hashes[name] == next(p["sha256"] for p in pins if p["path"] == "ViT-B-32/" + name)
    partner = load(str(root / "DTD/finetuned.pt")).cpu().eval()
    merged = merge(anchor, original, partner, 0.5)
    dtd_train = DTD(args.data, split="train", partition=1, transform=original.val_preprocess)
    dtd_indices = split_training(dtd_train, dict(diagnostic=256), 20261013)["diagnostic"]
    save_json(
        out / "partner_indices.json", dict(DTD=dtd_indices, use="Post-freeze diagnostics only")
    )
    responses, diagnostic_moments = [], {}
    for context, dataset, selection in [
        ("MNIST", train, indices["confirmation"]),
        ("DTD", dtd_train, dtd_indices),
    ]:
        for stage, model in [("native", original), ("merged", merged)]:
            moment = collect_moments(model, make_loader(dataset, selection, 64), ["FC:5"])["FC:5"]
            diagnostic_moments[context + "/" + stage] = moment
            for name, basis in spaces.items():
                responses.append(
                    dict(
                        dataset=context, stage=stage, basis=name, **complement_power(moment, basis)
                    )
                )
    torch.save(diagnostic_moments, out / "artifacts/diagnostic_moments.pt")
    save_json(out / "responses.json", responses)
    heads = {task: load(str(root / f"head_{task}.pt")).cpu().eval() for task in ["MNIST", "DTD"]}
    tests = dict(
        MNIST=MNIST(args.data, train=False, transform=original.val_preprocess),
        DTD=DTD(args.data, split="test", partition=1, transform=original.val_preprocess),
    )
    rows, predictions = [], {}
    for preset, model in models.items():
        combined = merged if preset == "original" else merge(anchor, model, partner, 0.5)
        for stage, task, measured in [
            ("native", "MNIST", model),
            ("direct", "MNIST", combined),
            ("direct", "DTD", combined),
        ]:
            chunks = []
            metric = evaluate(
                measured,
                heads[task],
                make_loader(tests[task]),
                "cuda",
                lambda p, y: chunks.append((p, y)),
            )
            pred, labels = [torch.cat([v[i] for v in chunks]) for i in range(2)]
            key = "/".join([preset, stage, task])
            predictions[key] = dict(prediction=pred, target=labels)
            row = dict(preset=preset, stage=stage, dataset=task, **metric)
            if preset != "original":
                old = predictions["/".join(["original", stage, task])]
                assert torch.equal(old["target"], labels)
                row.update(
                    paired=paired_counts(old["prediction"].eq(labels), pred.eq(labels)),
                    changed_predictions=int(old["prediction"].ne(pred).sum()),
                )
            rows.append(row)
            save_json(out / "results.json", rows)
            torch.save(predictions, out / "artifacts/predictions.pt")
            print("RESULT", json.dumps(row), flush=True)
    save_json(
        out / "execution.json",
        dict(
            completed=True,
            gpu=torch.cuda.get_device_name(),
            checkpoint_sha256=hashes,
            predictions_sha256=digest(out / "artifacts/predictions.pt"),
            new_defense_candidates=0,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--data", default="data")
    run(parser.parse_args())
