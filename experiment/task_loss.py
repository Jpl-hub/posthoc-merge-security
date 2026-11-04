"""Fixed synthetic-input comparison of native and joint native/scaled losses.

Adapted objective: Trap2 ef65a265, 8vision_training_full_trap2.py.
This post-hoc A/B experiment is not a reproduction of its supervised training.
"""
import argparse
import copy
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import functional_call

from _impl.io import check_files, digest, new_output, read_json, save_json
from _impl.metrics import logical_parameter_distance
from _impl.models import apply, checked_models, initialize


def outside_task(delta, left, right):
    delta = delta - left @ (left.T @ delta)
    return delta - (delta @ right) @ right.T


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    args = parser.parse_args()
    cfg = read_json(args.protocol)
    check_files(cfg["source_sha256"])
    initialize()
    out = new_output(cfg["output"], cfg)
    (out / "source.py").write_bytes(Path(__file__).read_bytes())
    base, expert, head, pins = checked_models("MNIST")
    parent = apply(expert, cfg["parent"])
    ref = cfg["synthetic_source"]
    check_files({ref["path"]: ref["sha256"]})
    raw = torch.load(ref["path"], map_location="cpu", weights_only=False)
    norm = expert.val_preprocess.transforms[-1]
    mean = torch.tensor(norm.mean)[None, :, None, None]
    std = torch.tensor(norm.std)[None, :, None, None]
    images = []
    for pixels in raw["final"].split(8):
        x = F.interpolate(pixels, (224, 224), mode="bilinear", align_corners=False)
        images.append((torch.cat([torch.roll(x, s, (2, 3)) for s in
                                 [(0, 0), (7, 0), (-7, 0), (0, 7), (0, -7)]]) - mean) / std)
    images = torch.cat(images)
    del raw
    expert.cuda(); head.cuda()
    with torch.no_grad():
        teacher = torch.cat([head(expert(x.cuda())).cpu() for x in images.split(16)])
    labels = teacher.argmax(1)
    expert.cpu()
    original = expert.state_dict()
    public = {k: v.cuda() for k, v in base.named_parameters()}
    bases = {}
    for i in range(12):
        key = f"model.visual.transformer.resblocks.{i}.mlp.c_fc.weight"
        delta = (original[key] - base.state_dict()[key]).cuda().double()
        u, _, vh = torch.linalg.svd(delta, full_matrices=False)
        bases[key] = (u[:, :cfg["rank"]].float(), vh[:cfg["rank"]].T.float())
    del base, delta, u, vh
    releases = {}
    for name, strength in cfg["conditions"].items():
        folder = out / name
        folder.mkdir()
        torch.manual_seed(cfg["seed"])
        rng = np.random.RandomState(cfg["seed"])
        model = copy.deepcopy(parent).cuda().eval()
        params = dict(model.named_parameters())
        trainable = {k: p for k, p in params.items() if ".mlp." in k and not k.endswith("c_fc.bias")}
        for p in trainable.values():
            p.requires_grad_(True)
        originals = {k: original[k].cuda() for k in trainable}
        initials = {k: parent.state_dict()[k].cuda() for k in bases}
        native_scores = []
        with torch.no_grad():
            for x in images.split(16):
                native_scores.append(head(model(x.cuda())).cpu())
        before = torch.cat(native_scores)
        optimizer = torch.optim.AdamW(trainable.values(), lr=cfg["learning_rate"],
                                     weight_decay=0., betas=(.9, .98), eps=1e-6)
        start = time.monotonic()
        records = []
        order = torch.randperm(len(images))
        for step in range(cfg["steps"]):
            offset = (step * cfg["batch_size"]) % len(images)
            if offset == 0:
                order = torch.randperm(len(images))
            ids = order[offset:offset + cfg["batch_size"]]
            x, y = images[ids].cuda(), labels[ids].cuda()
            warm = min(1., (step + 1) / cfg["warmup_steps"])
            phase = max(0., (step - cfg["warmup_steps"]) / (cfg["steps"] - cfg["warmup_steps"]))
            lr = cfg["learning_rate"] * warm * .5 * (1 + math.cos(math.pi * phase))
            optimizer.param_groups[0]["lr"] = lr
            optimizer.zero_grad(set_to_none=True)
            effective = dict(params)
            for key in bases:
                bias = key.replace("weight", "bias")
                beta = key.replace("mlp.c_fc.weight", "ln_2.bias")
                effective[bias] = parent.state_dict()[bias].cuda() - (params[key] - initials[key]) @ params[beta]
            native = F.cross_entropy(head(functional_call(model, effective, (x,))), y)
            native.backward()
            scale = rng.uniform(.05, 2.)
            while .95 < scale < 1.05:
                scale = rng.uniform(.05, 2.)
            # Rebuild the affine coupling after the first backward released its graph.
            for key in bases:
                bias = key.replace("weight", "bias")
                beta = key.replace("mlp.c_fc.weight", "ln_2.bias")
                effective[bias] = parent.state_dict()[bias].cuda() - (params[key] - initials[key]) @ params[beta]
            scaled = {k: public[k] + scale * (p - public[k]) for k, p in effective.items()}
            off = F.cross_entropy(head(functional_call(model, scaled, (x,))), y)
            if strength:
                (-strength / scale * off).backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in trainable.values())
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable.values(), 1.)
            optimizer.step()
            # Project finite A updates, then derive the matching affine bias.
            with torch.no_grad():
                for key, (left, right) in bases.items():
                    a = params[key]
                    a.copy_(initials[key] + outside_task(a - initials[key], left, right))
                    delta = a - originals[key]
                    a.copy_(originals[key] + delta * min(1., cfg["A_limit"] * originals[key].norm().item() / max(delta.norm().item(), 1e-30)))
                    bias = key.replace("weight", "bias")
                    beta = key.replace("mlp.c_fc.weight", "ln_2.bias")
                    params[bias].copy_(parent.state_dict()[bias].cuda() - (a - initials[key]) @ params[beta])
                    bkey = key.replace("c_fc", "c_proj")
                    b = params[bkey]; delta = b - originals[bkey]
                    b.copy_(originals[bkey] + delta * min(1., cfg["B_limit"] * originals[bkey].norm().item() / max(delta.norm().item(), 1e-30)))
            records.append(dict(step=step + 1, native_CE=float(native.detach()),
                                scaled_CE=float(off.detach()), scale=scale, lr=lr,
                                gradient_norm=float(grad_norm)))
            if (step + 1) % 25 == 0:
                print(name, records[-1], flush=True)
            del scaled, effective, native, off
        seconds = time.monotonic() - start
        model.requires_grad_(False)
        with torch.no_grad():
            after = torch.cat([head(model(x.cuda())).cpu() for x in images.split(16)])
        model.cpu()
        edits = {k: v.clone() for k, v in model.state_dict().items() if not torch.equal(v, original[k])}
        assert all(".mlp." in k for k in edits)
        path = folder / "point.pt"
        torch.save(edits, path)
        release = dict(path=str(path), sha256=digest(path))
        torch.save(dict(before=before, after=after, teacher=teacher, labels=labels), folder / "source_scores.pt")
        svd_error = max(float(((edits.get(k, original[k]) - parent.state_dict()[k]).cuda() @ right).abs().max())
                        for k, (_, right) in bases.items())
        result = dict(release=release, updates=cfg["steps"], seconds=seconds,
                      source_n=len(labels), before_correct=int((before.argmax(1)==labels).sum()),
                      after_correct=int((after.argmax(1)==labels).sum()), task_right_error=svd_error,
                      parameter_distance=logical_parameter_distance(edits, original),
                      original_expert_updates=0, real_construction_images=0,
                      source_scores_sha256=digest(folder / "source_scores.pt"), checkpoint_sha256=pins,
                      records=records)
        save_json(folder / "result.json", result)
        releases[name] = release
        save_json(out / "releases.json", releases)
        print("FROZEN", name, {k:v for k,v in result.items() if k not in {"records","checkpoint_sha256"}}, flush=True)
        del model, optimizer, params, trainable, originals, initials
    save_json(out / "completed.json", dict(releases=releases, status="source_completed"))


if __name__ == "__main__":
    main()
