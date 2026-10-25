"""One matched source-generation comparison using a frozen CLIP expert."""
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

from _impl.io import digest
from _impl.models import checked_models, initialize


def patch_entropy(features):
    total = 0.
    for feature in features:
        patches = F.normalize(feature.mean(1)[:, 1:, :], dim=-1, eps=1e-8)
        sims = (patches @ patches.transpose(1, 2)).flatten(1)
        grid = torch.linspace(float(sims.detach().min()), float(sims.detach().max()), 10, device=sims.device)
        density = torch.exp(-.5 * ((grid[None, :, None] - sims[:, None]) / .01).square()).mean(-1)
        density = density / (math.sqrt(2 * math.pi) * .01) + 1e-4
        total -= torch.trapezoid(-density * density.log(), grid, dim=-1).mean()
    return total


def variation(x):
    return sum(v.norm() for v in [x[:, :, :, 1:] - x[:, :, :, :-1],
                                 x[:, :, 1:] - x[:, :, :-1],
                                 x[:, :, 1:, :-1] - x[:, :, :-1, 1:],
                                 x[:, :, 1:, 1:] - x[:, :, :-1, :-1]])


def main():
    initialize()
    start = time.monotonic()
    protocol = Path("results/data_free_compensation/source_prior_protocol.json")
    cfg = json.loads(protocol.read_text())
    assert digest(__file__) == cfg["source_sha256"]
    out = Path(cfg["output"])
    out.mkdir(exist_ok=False)
    (out / "protocol.json").write_text(protocol.read_text())
    base, expert, head, pins = checked_models("MNIST")
    base, expert, head = [x.cuda().eval() for x in [base, expert, head]]
    norm = expert.val_preprocess.transforms[-1]
    mean = torch.tensor(norm.mean, device="cuda")[None, :, None, None]
    std = torch.tensor(norm.std, device="cuda")[None, :, None, None]
    lower, upper = -mean / std, (1 - mean) / std
    initial = torch.randn(32, 3, 224, 224, generator=torch.Generator().manual_seed(cfg["seed"]))
    labels = torch.arange(32, device="cuda") % 10
    torch.save(dict(normalized_images=initial, labels=labels.cpu()), out / "initial.pt")
    rng = random.Random(cfg["seed"])
    jitter = [rng.randint(-15 if i < 500 else -30, 15 if i < 500 else 30) for i in range(1000)]
    features, native_errors = [], []
    verify_native = True

    def capture(module, args, kwargs, output):
        x = args[0]
        assert kwargs.get("attn_mask") is None
        qkv = F.linear(x, module.in_proj_weight, module.in_proj_bias)
        q, k, v = [z.permute(1, 0, 2).reshape(x.shape[1], x.shape[0], module.num_heads, -1).transpose(1, 2)
                   for z in qkv.chunk(3, dim=-1)]
        weighted = ((q @ k.transpose(-1, -2)) / math.sqrt(q.shape[-1])).softmax(-1) @ v
        if verify_native:
            projected = F.linear(weighted.transpose(1, 2).reshape(x.shape[1], x.shape[0], -1),
                                 module.out_proj.weight, module.out_proj.bias).transpose(0, 1)
            native_errors.append(float((projected - output[0]).detach().abs().max()))
            torch.testing.assert_close(projected, output[0], rtol=5e-5, atol=5e-5)
        features.append(weighted)

    hooks = [block.attn.register_forward_hook(capture, with_kwargs=True)
             for block in expert.model.visual.transformer.resblocks]
    counters = dict(synthetic_model_forwards=0, input_gradient_model_forwards=0,
                    pixel_steps=0, real_images=0, expert_updates=0, defense_points=0, AGA_calls=0)

    def scores(model, images):
        if model is expert:
            features.clear()
        counters["synthetic_model_forwards"] += len(images)
        if torch.is_grad_enabled() and images.requires_grad:
            counters["input_gradient_model_forwards"] += len(images)
        return head(model(images))

    with torch.no_grad():
        logits = scores(expert, initial.cuda())
        initial_metrics = dict(pseudo_correct=int((logits.argmax(1) == labels).sum()),
                               CE=float(F.cross_entropy(logits, labels)),
                               negative_patch_entropy=float(patch_entropy(features)),
                               TV=float(variation(initial.cuda())))
    assert len(native_errors) == 12
    verify_native = False
    results = []
    for variant in ["confidence_only", "paper_priors"]:
        images = torch.nn.Parameter(initial.cuda().clone())
        optimizer = torch.optim.Adam([images], lr=.2, betas=(.5, .9), eps=1e-8)
        trace = []
        try:
            for step in range(1000):
                local_step = step % 500
                lr = .2 * (local_step + 1) / 100 if local_step < 100 else .1 * (1 + math.cos(math.pi * (local_step - 100) / 400))
                optimizer.param_groups[0]["lr"] = lr
                rolled = torch.roll(images, (jitter[step], jitter[step]), (2, 3))
                logits = scores(expert, rolled)
                ce, pse, tv = F.cross_entropy(logits, labels), patch_entropy(features), variation(rolled)
                loss = ce if variant == "confidence_only" else ce + pse + .05 * (tv - cfg["tv_target"]).abs()
                assert torch.isfinite(loss)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                counters["pixel_steps"] += 1
                with torch.no_grad():
                    images.copy_(torch.maximum(lower, torch.minimum(upper, images)))
                if (step + 1) % 100 == 0:
                    row = dict(step=step + 1, CE=float(ce.detach()), negative_patch_entropy=float(pse.detach()),
                               TV=float(tv.detach()), pseudo_correct=int((logits.argmax(1) == labels).sum()))
                    trace.append(row)
                    print(variant, json.dumps(row), flush=True)
        except Exception:
            torch.save(images.detach().cpu(), out / (variant + ".failed.pt"))
            raise
        with torch.no_grad():
            final = images.detach()
            expert_scores = []
            for shift in cfg["assessment_shifts"]:
                expert_scores.append(scores(expert, torch.roll(final, tuple(shift), (2, 3))).cpu())
                if shift == [0, 0]:
                    final_pse, final_tv = float(patch_entropy(features)), float(variation(final))
            expert_scores = torch.stack(expert_scores)
            base_scores = scores(base, torch.cat([torch.roll(final, tuple(s), (2, 3)) for s in cfg["assessment_shifts"]])).cpu().reshape(5, 32, 10)
            row = dict(variant=variant, final_pseudo_correct_by_shift=[int((x.argmax(1) == labels.cpu()).sum()) for x in expert_scores],
                       all_shift_pseudo_correct=int((expert_scores.argmax(-1) == labels.cpu()).all(0).sum()),
                       base_pseudo_correct_by_shift=[int((x.argmax(1) == labels.cpu()).sum()) for x in base_scores],
                       negative_patch_entropy=final_pse, TV=final_tv, trace=trace)
            raw = dict(normalized_images=final.cpu(), mean=mean.cpu(), std=std.cpu(), labels=labels.cpu(),
                       expert_scores=expert_scores, base_scores=base_scores)
            torch.save(raw, out / (variant + ".pt"))
            row["raw_sha256"] = digest(out / (variant + ".pt"))
            rgb = ((final * std + mean).clamp(0, 1).permute(0, 2, 3, 1).cpu().numpy() * 255).round().astype(np.uint8)
            sheet = Image.new("RGB", (8 * 160, 4 * 180), "white")
            draw = ImageDraw.Draw(sheet)
            predictions = expert_scores[0].argmax(1).tolist()
            for i, pixel in enumerate(rgb):
                x, y = (i % 8) * 160, (i // 8) * 180
                sheet.paste(Image.fromarray(pixel).resize((160, 160)), (x, y))
                draw.text((x + 4, y + 162), f"target {i % 10}  prediction {predictions[i]}", fill="black")
            sheet.save(out / (variant + ".png"))
            results.append(row)
            print("SOURCE_DONE", json.dumps({k: v for k, v in row.items() if k != "trace"}), flush=True)
    for hook in hooks:
        hook.remove()
    assert all(p.grad is None for model in [expert, base, head] for p in model.parameters())
    result = dict(rows=results, initial=initial_metrics, maximum_native_attention_error=max(native_errors),
                  checkpoint_sha256=pins, initial_sha256=digest(out / "initial.pt"), counters=counters,
                  source_sha256=digest(__file__), source=Path(__file__).read_text(), seconds=time.monotonic()-start,
                  scope=cfg["scope"])
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
