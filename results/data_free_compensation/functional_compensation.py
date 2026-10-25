"""One Data-Free classification fit with joint A/B task-response constraints."""
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _impl.io import digest
from _impl.metrics import logical_parameter_distance
from _impl.models import checked_models, initialize


@torch.no_grad()
def joint_constraint(state, entries, block, prefix):
    A, B, a = [state[prefix + "mlp." + k].double().cuda()
               for k in ["c_fc.weight", "c_proj.weight", "c_fc.bias"]]
    ba, bb = [entries[prefix + "mlp." + k + ".weight"] for k in ["c_fc", "c_proj"]]
    ua, va, vb = [v.cuda() for v in [ba["u"], ba["v"], bb["v"]]]
    columns = torch.cat([ua, A @ va, vb, a[:, None]], dim=1)
    columns = columns[:, columns.norm(dim=0) > 0]
    columns /= columns.norm(dim=0)
    C, s, _ = torch.linalg.svd(columns, full_matrices=False)
    C = C[:, s > s[0] * 1e-10]
    Ap = (2 * C @ (C.T @ A) - A).float()
    Bt = (2 * (B @ C) @ C.T - B).float().double()
    r = torch.cat([va.T, -va.T])
    r -= r.mean(1, keepdim=True)
    r *= math.sqrt(768) / r.norm(dim=1, keepdim=True)
    gamma, beta = [state[prefix + "ln_2." + k].double().cuda()
                   for k in ["weight", "bias"]]
    X = F.layer_norm(r, (768,), gamma, beta, block.ln_2.eps)
    phi = lambda z: z * torch.sigmoid(1.702 * z)
    H = phi(F.linear(X, Ap.double(), a)).T
    Y = B @ phi(F.linear(X, A, a)).T
    U, s, vh = torch.linalg.svd(H, full_matrices=False)
    keep = s > torch.finfo(torch.double).eps * max(H.shape) * s[0]
    Q = U[:, keep]
    inverse = (vh[keep].T / s[keep]) @ Q.T
    affine = B + (Y - B @ H) @ inverse
    initial = Bt + (Y - Bt @ H) @ inverse
    assert float((initial @ H - Y).norm() / (1 + Y.norm())) < 1e-10
    radius = ((4 * B.norm()).square() - (affine - B).square().sum()).sqrt()
    return Ap, initial.float(), dict(
        affine=affine.float(), Q=Q.float(), radius=radius.float(),
        H=H, Y=Y, original=B, vb=vb, X=X, a=a,
    )


def main():
    initialize()
    start = time.monotonic()
    path = Path("results/data_free_compensation/functional_compensation_protocol.json")
    cfg = json.loads(path.read_text())
    assert digest(__file__) == cfg["implementation_sha256"]
    out = Path(cfg["output"])
    out.mkdir(exist_ok=False)
    (out / "protocol.json").write_text(path.read_text())
    for k in ["source", "basis"]:
        assert digest(cfg[k]["path"]) == cfg[k]["sha256"]
    base, expert, head, pins = checked_models(cfg["target"])
    del base
    original = {k: v.clone() for k, v in expert.state_dict().items()}
    published = copy.deepcopy(expert).cuda().eval().requires_grad_(False)
    basis = torch.load(cfg["basis"]["path"], map_location="cpu", weights_only=False)
    entries = {v["source_key"]: v for v in basis.values()}
    rules, parameters = [], []
    for i in cfg["layers"]:
        prefix = f"model.visual.transformer.resblocks.{i}."
        block = published.model.visual.transformer.resblocks[i]
        Ap, Bp, rule = joint_constraint(original, entries, block, prefix)
        with torch.no_grad():
            block.mlp.c_fc.weight.copy_(Ap)
            block.mlp.c_proj.weight.copy_(Bp)
        parameters.append(block.mlp.c_proj.weight)
        rules.append(rule)
    optimizer = torch.optim.Adam(parameters, lr=cfg["publication_lr"])
    raw = torch.load(cfg["source"]["path"], map_location="cpu", weights_only=False)
    pixels = raw["final"].clone()
    del raw
    groups = [torch.nn.Parameter(x.clone().cuda())
              for x in pixels.split(cfg["source"]["group_size"])]
    pixel_optimizers = [torch.optim.Adam([x], lr=cfg["source"]["pixel_lr"]) for x in groups]
    history = [pixels]
    norm = expert.val_preprocess.transforms[-1]
    mean = torch.tensor(norm.mean, device="cuda")[None, :, None, None]
    std = torch.tensor(norm.std, device="cuda")[None, :, None, None]
    expert.cuda().eval()
    head.cuda().eval()
    counters = dict(generated_full_image_forwards=0, input_gradient_model_forwards=0,
                    pixel_optimizer_steps=0, publication_B_gradient_steps=0)
    temperature = cfg["source"]["temperature"]

    def views(x):
        x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
        return (torch.cat([torch.roll(x, tuple(s), (2, 3))
                           for s in cfg["source"]["views"]]) - mean) / std

    def logits(model, x):
        counters["generated_full_image_forwards"] += len(x)
        if torch.is_grad_enabled() and x.requires_grad:
            counters["input_gradient_model_forwards"] += len(x)
        return head(model(x))

    def divergence(student, teacher):
        return temperature ** 2 * F.kl_div(
            F.log_softmax(student / temperature, dim=1),
            F.softmax(teacher / temperature, dim=1), reduction="batchmean")

    @torch.no_grad()
    def image_pool(x):
        return torch.cat([views(v.cuda()).cpu()
                          for v in x.split(cfg["source"]["group_size"])])

    @torch.no_grad()
    def project():
        for B, rule in zip(parameters, rules):
            free = B - rule["affine"]
            free -= (free @ rule["Q"]) @ rule["Q"].T
            free *= (rule["radius"] / free.norm().clamp_min(1e-30)).clamp(max=1)
            B.copy_(rule["affine"] + free)

    pools, rounds = [image_pool(pixels)], []
    for cycle in range(cfg["source"]["rounds"]):
        for B in parameters:
            B.requires_grad_(False)
            B.grad = None
        first, last = [], []
        for group_index, (group, actor) in enumerate(zip(groups, pixel_optimizers)):
            for step in range(cfg["source"]["pixel_steps_per_round"]):
                x = views(group)
                loss = divergence(logits(published, x), logits(expert, x))
                if step == 0:
                    first.append(float(loss.detach()))
                actor.zero_grad(set_to_none=True)
                (-loss).backward()
                assert group.grad is not None and torch.isfinite(group.grad).all()
                actor.step()
                counters["pixel_optimizer_steps"] += 1
                with torch.no_grad():
                    group.clamp_(0, 1)
            with torch.no_grad():
                x = views(group)
                last.append(float(divergence(logits(published, x), logits(expert, x))))
            if group_index % 5 == 0 or group_index == len(groups) - 1:
                print("INPUT_SEARCH", cycle + 1, group_index, first[-1], last[-1], flush=True)
        history.append(torch.cat([x.detach().cpu() for x in groups]))
        pools.append(image_pool(history[-1]))
        images = torch.cat(pools)
        batches = list(images.split(cfg["source"]["batch_size"]))
        with torch.no_grad():
            targets = [logits(expert, x.cuda()).detach() for x in batches]
        for B in parameters:
            B.requires_grad_(True)
        epoch_losses = []
        for epoch in range(cfg["publication_epochs_per_round"]):
            total = 0
            for x, target in zip(batches, targets):
                loss = divergence(logits(published, x.cuda()), target)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                assert torch.isfinite(loss) and all(
                    p.grad is not None and torch.isfinite(p.grad).all() for p in parameters)
                optimizer.step()
                project()
                counters["publication_B_gradient_steps"] += 1
                total += float(loss.detach()) * len(x)
            epoch_losses.append(total / len(images))
            print("CLASSIFICATION_FIT", cycle + 1, epoch + 1, epoch_losses[-1], flush=True)
        with torch.no_grad():
            after = torch.cat([logits(published, x.cuda()).cpu() for x in batches])
            before = torch.cat([x.cpu() for x in targets])
        row = dict(cycle=cycle + 1, source_views=len(images),
                   critic_KL_before=sum(first) / len(first),
                   critic_KL_after=sum(last) / len(last), epoch_losses=epoch_losses,
                   source_prediction_changes=int((after.argmax(1) != before.argmax(1)).sum()),
                   source_KL_after=float(divergence(after, before)))
        rounds.append(row)
        (out / "progress.json").write_text(json.dumps(dict(rounds=rounds, counters=counters), indent=2))
        print("ROUND", json.dumps(row), flush=True)
    with torch.no_grad():
        rows = []
        for i, B, rule in zip(cfg["layers"], parameters, rules):
            delta = B.double() - rule["original"]
            error = float((B.double() @ rule["H"] - rule["Y"]).norm() / (1 + rule["Y"].norm()))
            distance = float(delta.norm() / rule["original"].norm())
            assert error < 1e-4 and distance <= 4 + 1e-5
            Akey = f"model.visual.transformer.resblocks.{i}.mlp.c_fc.weight"
            rows.append(dict(layer=i, relative_A_distance=float(
                                 (published.state_dict()[Akey].double() - original[Akey].cuda()).norm()
                                 / original[Akey].norm()), relative_B_distance=distance,
                             joint_probe_relative_error=error,
                             B_parameter_task_energy_ratio=float(
                                 (delta @ rule["vb"]).square().sum() / delta.square().sum()),
                             B_parameter_task_relative_response=float(
                                 (delta @ rule["vb"]).norm() / (rule["original"] @ rule["vb"]).norm())))
        assert all(p.grad is None for p in expert.parameters())
        assert all(torch.equal(v.cpu(), expert.state_dict()[k].cpu()) for k, v in original.items())
        edits = {k: v.detach().cpu() for k, v in published.state_dict().items()
                 if not torch.equal(v.cpu(), original[k])}
    torch.save(edits, out / "point.pt")
    torch.save(history, out / "inputs.pt")
    torch.save(dict(before=before, after=after), out / "source_scores.pt")
    result = dict(status="unique_functional_constraint_point_frozen",
                  point_sha256=digest(out / "point.pt"), source=Path(__file__).read_text(),
                  source_sha256=digest(__file__), protocol_sha256=digest(path),
                  checkpoint_sha256=pins, rounds=rounds, per_layer=rows, counters=counters,
                  input_history_sha256=digest(out / "inputs.pt"),
                  source_scores_sha256=digest(out / "source_scores.pt"),
                  parameter_distance=logical_parameter_distance(edits, original),
                  seconds=time.monotonic() - start, real_construction_images=0,
                  original_expert_parameter_updates=0, new_AGA_calls=0,
                  scope=cfg["scope"])
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("POINT_FROZEN", result["point_sha256"], flush=True)


if __name__ == "__main__":
    main()
