"""Freeze a compensated CLIP readout and its A/B ablation without images."""

import argparse
from pathlib import Path

import torch

from _impl.io import digest, new_output, read_json, save_json
from _impl.models import checked_models, initialize


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="results/data_free_compensation/readout_channels/protocol.json")
    cfg = read_json(parser.parse_args().protocol)
    initialize()
    base, expert, head, pins = checked_models(cfg["target"])
    ref = cfg["parent"]
    assert digest(ref["path"]) == ref["sha256"]
    parent = torch.load(ref["path"], map_location="cpu", weights_only=True)
    g, b, p = ("model.visual." + k for k in ("ln_post.weight", "ln_post.bias", "proj"))
    state = expert.state_dict()
    assert not set(parent).intersection((g, b, p))
    gain, bias, projection = (state[k].double() for k in (g, b, p))
    delta = projection - base.state_dict()[p].double()
    basis = torch.linalg.svd(delta, full_matrices=False).U[:, :32]
    outside = (1 - basis.square().sum(1)).clamp_min(0).sqrt()
    classes = head.weight.double()
    contrast = projection @ (classes - classes.mean(0)).T
    channel = int((outside * gain.abs() * contrast.norm(dim=1)).argmax())
    rates = cfg["interpolation_coefficients"]
    c = 2 / sum(t * (1 - t) for t in rates)
    background = ((2 - c) - ((2 - c) ** 2 - 4) ** 0.5) / 2
    scales = torch.full_like(gain, background)
    scales[channel] = 1 + min(v.norm() / v[channel].abs() for v in (gain, bias) if v[channel] != 0)
    edits = {g: (gain * scales).to(state[g]), b: (bias * scales).to(state[b]),
             p: (projection / scales[:, None]).to(state[p])}
    relative = {k: float((v.double() - state[k].double()).norm() / state[k].double().norm()) for k, v in edits.items()}
    assert relative[g] <= 3.2 and relative[b] <= 3.2 and relative[p] <= 2
    # These two products determine the affine readout for every normalized hidden state.
    changed_p = edits[p].double()
    weight_error = (edits[g].double()[:, None] * changed_p - gain[:, None] * projection).abs().max()
    offset_error = (edits[b].double() @ changed_p - bias @ projection).abs().max()
    out = new_output(cfg["output"], cfg)
    releases = {}
    for name, tensors in [("ReadoutOnly", edits), ("ABReadout", parent | edits)]:
        path = out / (name + ".pt")
        torch.save(tensors, path)
        releases[name] = dict(path=str(path), sha256=digest(path))
    # A location-aware attacker can set each published LN gain to the public
    # gain, compensating projection rows to preserve the standalone map.
    canonical = {}
    for name, overrides in (("CleanCanonical", {}), ("ABCanonical", parent | edits)):
        published = state | overrides
        public_gain = base.state_dict()[g].double()
        ratio = public_gain / published[g].double()
        assert torch.isfinite(ratio).all() and (ratio != 0).all()
        corrected = overrides | {
            g: public_gain.to(state[g]),
            b: (published[b].double() * ratio).to(state[b]),
            p: (published[p].double() / ratio[:, None]).to(state[p]),
        }
        path = out / (name + ".pt")
        torch.save(corrected, path)
        canonical[name] = dict(path=str(path), sha256=digest(path))
    save_json(out / "freeze.json", dict(
        releases=releases, canonical=canonical, checkpoints=pins, channel=channel,
        channel_scale=float(scales[channel]), background_scale=background,
        relative_edits=relative, native_weight_max_error=float(weight_error),
        native_offset_max_error=float(offset_error), real_construction_images=0,
        source_sha256=digest(Path(__file__)),
    ))
    print("Frozen readouts and matched recovery inputs", flush=True)


if __name__ == "__main__":
    main()
