"""Cost of public completion outside the final classifier's numerator space.

One necessary-condition diagnostic, not a protected release. The target head,
post LayerNorm and projection define the space analytically. No PCA rank,
noise amplitude, optimization or partner is used to construct it.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader, digest, evaluate, merge, paired_counts
from subspace_probe import make_loader, save_json, split_training
from torchvision.datasets import DTD, MNIST


@torch.no_grad()
def classifier_space(model, head):
    assert head.normalize and torch.count_nonzero(head.bias) == 0
    visual = model.model.visual
    contrasts = head.weight.double() - head.weight.double().mean(0)
    readout = visual.proj.double() @ contrasts.T
    matrix = visual.ln_post.weight.double()[:, None] * readout
    matrix -= matrix.mean(0)
    offset = visual.ln_post.bias.double() @ readout
    mean_axis = torch.ones(len(matrix), 1, dtype=torch.float64) / len(matrix) ** 0.5
    u, s, _ = torch.linalg.svd(torch.cat([matrix, mean_axis], dim=1), full_matrices=False)
    rank = int((s > s[0] * 1e-10).sum())
    p = u[:, :rank] @ u[:, :rank].T
    error = float(((torch.eye(len(p)) - p) @ matrix).norm() / matrix.norm())
    assert error < 1e-10
    return dict(projector=p, matrix=matrix, offset=offset, rank=rank, span_error=error)


@torch.no_grad()
def complete(model, anchor, projector):
    result = copy.deepcopy(model).cpu().eval()
    layer = result.model.visual.transformer.resblocks[11].mlp.c_proj
    base = anchor.model.visual.transformer.resblocks[11].mlp.c_proj
    c = torch.eye(len(projector), dtype=torch.float64) - projector
    for name in ["weight", "bias"]:
        value, reference = getattr(layer, name), getattr(base, name)
        removed = c @ (value.double() - reference.double())
        value.copy_((value.double() - removed).float())
    return result


@torch.no_grad()
def interpolate(model, anchor, beta):
    result = copy.deepcopy(model).cpu().eval()
    for key, value in result.state_dict().items():
        reference = anchor.state_dict()[key]
        if value.is_floating_point():
            value.copy_(reference + beta * (value - reference))
    return result


@torch.no_grad()
def capture(model, loader):
    raw, features = [], []
    handle = model.model.visual.ln_post.register_forward_pre_hook(
        lambda _m, args: raw.append(args[0].detach().cpu())
    )
    model.cuda().eval()
    try:
        for images, _ in loader:
            features.append(model(images.cuda(non_blocking=True)).detach().cpu())
    finally:
        handle.remove()
        model.cpu()
    return torch.cat(raw).double(), torch.cat(features).double()


@torch.no_grad()
def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "artifacts").mkdir()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    protocol = dict(
        question="Does publicly completing the final MLP output complement remove useful target information in a real merger?",
        targets=["MNIST", "DTD"],
        edited_site="Final block (11) MLP c_proj weight and bias only",
        retained_space="Span of mean axis and centered class readout after target ln_post gamma; exact numerator space, not an input PCA or the NUFILT theorem",
        completion="Wc=P W+(I-P)W0; bc=P b+(I-P)b0; target head and public anchor only",
        limitation="Preserves mean and centered class numerator; LayerNorm variance and its class-dependent beta offset can change classification. No exact accuracy guarantee.",
        calibration="256 official training images check the equation; 256 disjoint training images confirm it. No data fitting or labels in the projector.",
        beta=0.5,
        coefficient=0.5,
        partner_load="Only after both target projectors, completion weights and fingerprints are frozen",
        stop="If completion preserves native and merged capability, stop this fixed public output-complement family before constructing noise; no rank or strength changes",
        attack_scope="Public completion only, not AGA or functional reconstruction; a diagnostic of unprotected experts, not an attacked protected candidate",
        prior_test_observation=True,
        new_defense_candidates=0,
        source_comparison="Old final CLS candidate changed c_fc input-PCA complement. This measures c_proj OUTPUT complement defined by actual class contrasts; no old noise or matching objective is reused.",
        seed=20261013,
    )
    save_json(output / "protocol.json", protocol)
    root = Path(args.checkpoints)
    pins = json.loads(Path("configs/checkpoints.json").read_bytes())["files"]
    hashes = {}
    for name in ["zeroshot.pt", "MNIST/finetuned.pt", "DTD/finetuned.pt", "head_MNIST.pt", "head_DTD.pt"]:
        hashes[name] = digest(root / name)
        assert hashes[name] == next(p["sha256"] for p in pins if p["path"] == "ViT-B-32/" + name)
    load = checkpoint_loader(Path(args.author_repo))
    anchor = load(str(root / "zeroshot.pt")).cpu().eval()
    rows, predictions, indices, spaces, evidence = [], {}, {}, {}, {}
    for task in protocol["targets"]:
        model = load(str(root / task / "finetuned.pt")).cpu().eval()
        head = load(str(root / f"head_{task}.pt")).cpu().eval()
        spaces[task] = classifier_space(model, head)
        completed = complete(model, anchor, spaces[task]["projector"])
        torch.save(
            dict(space=spaces[task], completed=completed.model.visual.transformer.resblocks[11].mlp.c_proj.state_dict()),
            output / f"artifacts/{task}.pt",
        )
        train = (MNIST(args.data, train=True, transform=model.val_preprocess)
                 if task == "MNIST" else DTD(args.data, split="train", partition=1, transform=model.val_preprocess))
        indices[task] = split_training(train, dict(calibration=256, confirmation=256), protocol["seed"])
        save_json(output / "training_indices.json", indices)
        details = []
        for split in ["calibration", "confirmation"]:
            loader = make_loader(train, indices[task][split], 64)
            before, features = capture(model, loader)
            after, changed = capture(completed, loader)
            matrix, offset = spaces[task]["matrix"], spaces[task]["offset"]
            variance = before.var(-1, unbiased=False) + model.model.visual.ln_post.eps
            predicted = before @ matrix / variance.sqrt()[:, None] + offset
            observed = features @ (head.weight.double() - head.weight.double().mean(0)).T
            equation_error = float((predicted - observed).norm() / observed.norm())
            assert equation_error < 2e-5
            details.append(dict(
                split=split, n=len(before), equation_relative_error=equation_error,
                numerator_relative_change=float(((after-before) @ matrix).norm() / (before @ matrix).norm()),
                mean_max_change=float((after.mean(-1)-before.mean(-1)).abs().max()),
                native_changed_predictions=int(((features @ head.weight.double().T).argmax(-1) != (changed @ head.weight.double().T).argmax(-1)).sum()),
                pre_ln_relative_change=float((after-before).norm()/before.norm()),
            ))
        own = model.model.visual.transformer.resblocks[11].mlp.c_proj.weight.double()
        base = anchor.model.visual.transformer.resblocks[11].mlp.c_proj.weight.double()
        c = torch.eye(len(own), dtype=torch.float64)-spaces[task]["projector"]
        proxy = interpolate(model, anchor, protocol["beta"])
        proxy_space = classifier_space(proxy, head)
        evidence[task] = dict(
            retained_rank=spaces[task]["rank"], complement_rank=len(c)-spaces[task]["rank"],
            span_error=spaces[task]["span_error"],
            removed_task_update_energy_fraction=float((c @ (own-base)).square().sum()/(own-base).square().sum()),
            proxy_readout_complement_fraction=float((c @ proxy_space["matrix"]).norm()/proxy_space["matrix"].norm()),
            class_dependent_ln_offset_norm=float(spaces[task]["offset"].norm()),
            checks=details,
        )
        del model, completed, proxy, head
    save_json(output / "geometry.json", evidence)
    save_json(output / "freeze.json", dict(
        checkpoint_sha256=hashes, protocol_sha256=digest(output/"protocol.json"),
        source_sha256={p.name:digest(p) for p in Path(__file__).parent.glob("*.py")},
        artifacts_sha256={p.name:digest(p) for p in (output/"artifacts").glob("*.pt")},
        future_partner_loaded=False,
    ))
    print("FROZEN", json.dumps(evidence), flush=True)
    # Each completion was constructed independently before this shared pair is loaded.
    models = {task:load(str(root/task/"finetuned.pt")).cpu().eval() for task in protocol["targets"]}
    heads = {task:load(str(root/f"head_{task}.pt")).cpu().eval() for task in models}
    tests = {task:(MNIST(args.data, train=False, transform=models[task].val_preprocess)
                   if task=="MNIST" else DTD(args.data, split="test", partition=1, transform=models[task].val_preprocess)) for task in models}

    def record(model, task, target, stage, preset):
        chunks=[]
        metric=evaluate(model, heads[task], make_loader(tests[task]), "cuda", lambda p,y:chunks.append((p,y)))
        pred, labels=[torch.cat([v[i] for v in chunks]) for i in range(2)]
        key="/".join([target,stage,preset,task])
        predictions[key]=dict(prediction=pred,target=labels)
        row=dict(target=target,stage=stage,preset=preset,dataset=task,**metric)
        if preset=="completion":
            before=predictions["/".join([target,stage,"original",task])]
            assert torch.equal(labels,before["target"])
            row["paired"]=paired_counts(before["prediction"].eq(labels),pred.eq(labels))
        rows.append(row)
        save_json(output/"results.json",rows)
        torch.save(predictions,output/"artifacts/predictions.pt")
        print("COMPLETION_RESULT",json.dumps(row),flush=True)

    for target in models:
        partner=next(t for t in models if t!=target)
        completed=complete(models[target],anchor,spaces[target]["projector"])
        saved = torch.load(output / f"artifacts/{target}.pt", map_location="cpu")
        assert all(torch.equal(value, saved["completed"][key]) for key, value in
                   completed.model.visual.transformer.resblocks[11].mlp.c_proj.state_dict().items())
        for preset,model in [("original",models[target]),("completion",completed)]:
            record(model,target,target,"standalone",preset)
            merged=merge(anchor,model,models[partner],protocol["coefficient"])
            for task in models: record(merged,task,target,"direct",preset)
            del merged
        del completed
    save_json(output/"execution.json",dict(completed=True,gpu=torch.cuda.get_device_name(),new_defense_candidates=0,predictions_sha256=digest(output/"artifacts/predictions.pt")))


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--checkpoints",default="checkpoints/ViT-B-32")
    parser.add_argument("--author-repo",default="external/secure-merge-attack")
    parser.add_argument("--data",default="data")
    run(parser.parse_args())
