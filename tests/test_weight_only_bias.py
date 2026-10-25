"""Weight-only noise preserves the native projection and controls the proxy."""
import copy

import torch
import torch.nn.functional as F

from _impl.defenses import construct_bias
from _impl.merge import explicit_merge


def test_weight_only_bias_needs_no_calibration_and_preserves_native(monkeypatch):
    generator = torch.Generator().manual_seed(110)
    base = torch.nn.Module()
    base.model = torch.nn.Module()
    base.model.visual = torch.nn.Module()
    visual = base.model.visual
    visual.proj = torch.nn.Parameter(torch.randn(8, 4, generator=generator))
    visual.ln_post = torch.nn.LayerNorm(8)
    expert = copy.deepcopy(base)
    with torch.no_grad():
        expert.model.visual.proj.add_(0.2 * torch.randn(8, 4, generator=generator))
    head = torch.nn.Linear(4, 3, bias=False)
    with torch.no_grad():
        head.weight.copy_(torch.randn(3, 4, generator=generator))
    before = {k: v.clone() for k, v in expert.state_dict().items()}
    monkeypatch.setattr("_impl.defenses.merge", explicit_merge)
    cfg = dict(
        noise_seed=20261014,
        numerical_relative_tolerance=1e-10,
        proxy_beta=0.5,
        bias_safety=2.0,
    )
    edits, detail = construct_bias(expert, base, head, cfg)
    assert all(torch.equal(v, before[k]) for k, v in expert.state_dict().items())
    assert list(edits) == ["model.visual.ln_post.bias"]
    assert detail["real_construction_images"] == detail["synthetic_construction_images"] == 0
    assert detail["ideal_embedding_null_error"] < 1e-10
    assert detail["fp32_embedding_constant_error"] < 2e-4
    published = copy.deepcopy(expert)
    published.load_state_dict(expert.state_dict() | edits)
    probes = torch.randn(64, 8, generator=generator)
    probes *= torch.logspace(-3, 3, 64)[:, None]
    native = expert.model.visual.ln_post(probes) @ expert.model.visual.proj
    changed = published.model.visual.ln_post(probes) @ published.model.visual.proj
    torch.testing.assert_close(native, changed, rtol=0, atol=2e-4)
    proxy = explicit_merge(base, published, base, 0.5).model.visual
    features = proxy.ln_post(probes) @ proxy.proj
    predictions = head(F.normalize(features, dim=1)).argmax(1)
    assert torch.all(predictions == detail["proxy_collapse_class"])
