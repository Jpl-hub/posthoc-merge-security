"""Fixed full-objective teacher-output and scaled-loss comparison.

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
        # Every closure uses the same images and scales, as required by line search.
        scales = []
        for _ in images.split(cfg["batch_size"]):
            scale = rng.uniform(.05, 2.)
            while .95 < scale < 1.05:
                scale = rng.uniform(.05, 2.)
            scales.append(float(scale))
        records = []
        start = time.monotonic()
        keys = list(trainable)
        variables = [torch.nn.Parameter(p.detach().clone()) for p in trainable.values()]
        optimizer = torch.optim.LBFGS(variables, lr=1., max_iter=20, max_eval=30,
                                     history_size=5, line_search_fn="strong_wolfe")

        def effective_parameters():
            effective = dict(params)
            effective.update(zip(keys, variables))
            for key, (left, right) in bases.items():
                a = initials[key] + outside_task(effective[key] - initials[key], left, right)
                delta = a - originals[key]
                # Limits are part of the differentiated parameterization.
                a = originals[key] + delta * (cfg["A_limit"] * originals[key].norm() / delta.norm().clamp_min(1e-30)).clamp(max=1.)
                effective[key] = a
                bias = key.replace("weight", "bias")
                beta = key.replace("mlp.c_fc.weight", "ln_2.bias")
                effective[bias] = parent.state_dict()[bias].cuda() - (a - initials[key]) @ params[beta]
                bkey = key.replace("c_fc", "c_proj")
                delta = effective[bkey] - originals[bkey]
                effective[bkey] = originals[bkey] + delta * (cfg["B_limit"] * originals[bkey].norm() / delta.norm().clamp_min(1e-30)).clamp(max=1.)
            return effective

        def closure():
            optimizer.zero_grad(set_to_none=True)
            native_total, scaled_total = 0., 0.
            for offset, scale in zip(range(0, len(images), cfg["batch_size"]), scales):
                end = offset + cfg["batch_size"]
                x = images[offset:end].cuda()
                target = teacher[offset:end].cuda()
                y = labels[offset:end].cuda()
                weight = len(x) / len(images)
                effective = effective_parameters()
                logits = head(functional_call(model, effective, (x,)))
                native = cfg["temperature"] ** 2 * F.kl_div(
                    F.log_softmax(logits / cfg["temperature"], dim=1),
                    F.softmax(target / cfg["temperature"], dim=1), reduction="batchmean")
                (weight * native).backward()
                native_total += weight * float(native.detach())
                if strength:
                    effective = effective_parameters()
                    scaled = {k: public[k] + scale * (p - public[k]) for k, p in effective.items()}
                    off = F.cross_entropy(head(functional_call(model, scaled, (x,))), y)
                    (-weight * strength / scale * off).backward()
                    scaled_total += weight * float(off.detach()) / scale
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in variables)
            objective = native_total - strength * scaled_total
            records.append(dict(evaluation=len(records)+1, native_KL=native_total,
                                inverse_scale_weighted_CE=scaled_total, objective=objective))
            print(name, records[-1], flush=True)
            save_json(folder / "progress.json", dict(records=records))
            return torch.tensor(objective, device="cuda")

        optimizer.step(closure)
        updates = int(optimizer.state[variables[0]]["n_iter"])
        with torch.no_grad():
            effective = effective_parameters()
            for key in params:
                params[key].copy_(effective[key])
        # Read the final point, never select a previous point from test scores.
        closure()
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
        result = dict(release=release, updates=updates, seconds=seconds, objective_evaluations=len(records), scales=scales,
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
        del model, optimizer, params, trainable, originals, initials, variables, effective
    save_json(out / "completed.json", dict(releases=releases, status="source_completed"))


if __name__ == "__main__":
    main()
