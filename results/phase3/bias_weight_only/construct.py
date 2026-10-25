"""Defense1 direction and amplitude from weights only; no data loader."""
import hashlib
import json
import math
from pathlib import Path

import torch

from _impl.defenses import parameters
from _impl.io import digest
from _impl.merge import merge
from _impl.models import checked_models, initialize


@torch.no_grad()
def main():
    initialize()
    protocol = Path("results/phase3/bias_weight_only/protocol.json")
    cfg = json.loads(protocol.read_text())
    out = Path(cfg["output"])
    out.mkdir(exist_ok=False)
    (out / "protocol.json").write_text(protocol.read_text())
    base, expert, head, pins = checked_models(cfg["target"])
    P = expert.model.visual.proj.double()
    U, s, _ = torch.linalg.svd(P, full_matrices=True)
    rank = int((s > s[0] * cfg["numerical_relative_tolerance"]).sum())
    null = U[:, rank:]
    proxy = merge(base, expert, base, cfg["proxy_coefficient"])
    p = parameters(proxy, head)
    W = head.weight.double()
    contrast = W - W.mean(0)
    u, s, vh = torch.linalg.svd(null.T @ p["projection"] @ contrast.T,
                                 full_matrices=False)
    keep = s > s[0] * cfg["numerical_relative_tolerance"]
    generator = torch.Generator().manual_seed(cfg["noise_seed"])
    requested = int(torch.randint(len(W), (1,), generator=generator))
    desired = torch.zeros(len(W), dtype=torch.double)
    desired[requested] = 1
    desired -= desired.mean()
    direction = null @ (u[:, keep] @ ((vh[keep] @ desired) / s[keep]))
    c = int((direction @ p["projection"] @ contrast.T).argmax())
    rivals = torch.arange(len(W)) != c
    t = p["projection"] @ (W[c] - W[rivals]).T
    gaps = direction @ t
    assert torch.all(gaps > 1e-10)
    q = p["gamma"][:, None] * t
    q -= q.mean(0, keepdim=True)
    worst = math.sqrt(len(P)) * q.norm(dim=0)
    required = ((worst - p["beta"] @ t) /
                (cfg["proxy_coefficient"] * gaps)).clamp_min(0)
    amplitude = cfg["amplitude_multiplier"] * float(required.max())
    assert amplitude > 0 and math.isfinite(amplitude)
    bias = expert.model.visual.ln_post.bias
    changed = (bias.double() + amplitude * direction).float()
    delta = changed.double() - bias.double()
    actual_lower = (p["beta"] + cfg["proxy_coefficient"] * delta) @ t - worst
    assert torch.isfinite(changed).all()
    key = "model.visual.ln_post.bias"
    torch.save({key: changed}, out / "point.pt")
    source = Path(__file__).read_text()
    result = dict(
        status="unique_weight_only_noise_frozen", point_sha256=digest(out / "point.pt"),
        checkpoint_sha256=pins, requested_class=requested, proxy_class=c,
        projection_rank=rank, null_dimension=null.shape[1], amplitude=amplitude,
        ideal_native_embedding_error=float((direction @ P).norm()),
        FP32_native_constant_embedding_offset=float((delta @ P).norm()),
        actual_proxy_margin_lower_bounds=actual_lower.tolist(),
        actual_proxy_all_input_class_certificate=bool(torch.all(actual_lower > 0)),
        delta_norm=float(delta.norm()), real_construction_images=0,
        synthetic_construction_images=0, teacher_training_updates=0, optimizer_steps=0,
        source=source, source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        protocol_sha256=digest(protocol))
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "source"}), flush=True)


if __name__ == "__main__":
    main()
