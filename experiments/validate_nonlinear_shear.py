"""Replay the fixed shear screen; check inverse, derivative and public deletion."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from critical_subspace import capture
from eight_task_benchmark import digest
from local_erasure import rescale
from neuron_projection import activation
from nonlinear_shear import activation_prime, construct, screen, spaces
from path_subspace import context, read
from subspace_probe import save_json


@torch.no_grad()
def validate(args):
    started, out = time.perf_counter(), Path(args.output)
    cfg, meta = read(args.config), read(out / "metadata.json")
    assert cfg == read(out / "protocol.json") and meta["completed"]
    assert not meta["defense_success_claimed"] and not meta["future_partner_loaded"]
    assert meta["source_sha256"] == digest("experiments/nonlinear_shear.py")
    assert meta["config_sha256"] == digest(args.config)
    for name, expected in meta["helper_sha256"].items():
        assert digest(Path("experiments") / name) == expected
    for name, expected in meta["input_sha256"].items():
        assert digest(name) == expected
    for name, expected in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == expected
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, load, _, holders = context(cfg)
    for name, expected in meta["checkpoint_sha256"].items():
        assert digest(root / name) == expected
    original, anchor = (
        load(str(root / cfg["task"] / "finetuned.pt")).cpu().eval(),
        holders["anchor"],
    )
    captured = torch.load(
        Path(cfg["parent"]) / "artifacts/inputs.pt", map_location="cpu", weights_only=True
    )
    rebuilt, proof, checks = construct(original, anchor, captured, cfg)
    assert checks == read(out / "construction_checks.json")
    saved_proof = torch.load(out / "artifacts/directions.pt", map_location="cpu", weights_only=True)
    states = {}
    for name in rebuilt:
        saved = torch.load(
            out / f"artifacts/{name}_edits.pt", map_location="cpu", weights_only=True
        )
        current = copy.deepcopy(original)
        for key, value in saved.items():
            current.state_dict()[key].copy_(value)
        assert all(
            torch.equal(v, current.state_dict()[k]) for k, v in rebuilt[name].state_dict().items()
        )
        states[name] = current
    rows = []
    for layer in cfg["layers"]:
        chosen, expected = saved_proof[str(layer)], proof[str(layer)]
        assert torch.equal(chosen["u"], expected["u"]) and torch.equal(chosen["v"], expected["v"])
        for name in ["ridge", "eigenvalue", "native_gram_sha256", "path_gram_sha256"]:
            assert chosen[name] == expected[name]
        u, v = chosen["u"].double().cuda(), chosen["v"].double().cuda()
        identity = torch.eye(len(u), dtype=torch.float64, device="cuda")
        t, inv = identity + u[:, None] @ v[None, :], identity - u[:, None] @ v[None, :]
        assert float((t @ inv - identity).abs().max()) < 1e-10
        base, mlp = [m.model.visual.transformer.resblocks[layer].mlp for m in [anchor, original]]
        anchor_errors = [
            float(
                (t @ base.c_fc.weight.double().cuda() - base.c_fc.weight.double().cuda())
                .abs()
                .max()
            ),
            float(
                (t @ base.c_fc.bias.double().cuda() - base.c_fc.bias.double().cuda()).abs().max()
            ),
            float(
                (base.c_proj.weight.double().cuda() @ inv - base.c_proj.weight.double().cuda())
                .abs()
                .max()
            ),
        ]
        assert max(anchor_errors) < 1e-10
        current = states["release"].model.visual.transformer.resblocks[layer].mlp
        wanted = [
            t @ mlp.c_fc.weight.double().cuda(),
            t @ mlp.c_fc.bias.double().cuda(),
            mlp.c_proj.weight.double().cuda() @ inv,
        ]
        for actual, target in zip(
            [current.c_fc.weight, current.c_fc.bias, current.c_proj.weight], wanted, strict=True
        ):
            assert float((actual.double().cuda() - target).abs().max()) < 1e-6
        pw, _, _ = spaces([base.c_fc.weight, base.c_fc.bias[:, None]], cfg["anchor_rank_rcond"])
        pv, _, _ = spaces([base.c_proj.weight.T], cfg["anchor_rank_rcond"])
        pw, pv = pw @ pw.T, pv @ pv.T
        assert float((pv @ u).abs().max()) < 1e-10 and float((v @ pw).abs().max()) < 1e-10
        for name, source in [("public_original", mlp), ("public_protected", current)]:
            found = states[name].model.visual.transformer.resblocks[layer].mlp
            a, b, c = [
                p.double().cuda() for p in [base.c_fc.weight, base.c_fc.bias, base.c_proj.weight]
            ]
            wanted = [
                a + pv @ (source.c_fc.weight.double().cuda() - a),
                b + pv @ (source.c_fc.bias.double().cuda() - b),
                c + (source.c_proj.weight.double().cuda() - c) @ pw,
            ]
            for actual, target in zip(
                [found.c_fc.weight, found.c_fc.bias, found.c_proj.weight], wanted, strict=True
            ):
                assert float((actual.double().cuda() - target).abs().max()) < 1e-6
            assert torch.equal(found.c_proj.bias, mlp.c_proj.bias)
        x = captured["expert"][f"up:{layer}"][:4].double().cuda()
        z = x @ mlp.c_fc.weight.double().cuda().T + mlp.c_fc.bias.double().cuda()
        h, down = activation(z), mlp.c_proj.weight.double().cuda()
        k = (z @ v)[:, None] * activation_prime(z) - (h @ v)[:, None]
        derivative = (k * u) @ down.T
        eps = 1e-4
        plus = (
            activation(z @ (identity + eps * u[:, None] @ v[None, :]).T)
            @ (down @ (identity - eps * u[:, None] @ v[None, :])).T
        )
        minus = (
            activation(z @ (identity - eps * u[:, None] @ v[None, :]).T)
            @ (down @ (identity + eps * u[:, None] @ v[None, :])).T
        )
        relative = float(
            ((plus - minus) / (2 * eps) - derivative).norm() / derivative.norm().clamp_min(1e-20)
        )
        assert relative < 1e-3
        rows.append(
            dict(
                layer=layer,
                anchor_fixed_max_absolute_errors=anchor_errors,
                direct_matrix_public_projection_verified=True,
                central_derivative_relative_error=relative,
            )
        )
    indices = read(out / "training_indices.json")
    assert indices == read(Path(cfg["parent"]) / "training_indices.json")
    assert len(indices["calibration"]) == 64 and len(indices["confirmation"]) == 24
    assert not set(indices["calibration"]).intersection(indices["confirmation"])
    from dataset.registry import get_dataset

    data = get_dataset(cfg["task"], original.val_preprocess, "data", batch_size=128, num_workers=2)
    batches = {
        split: torch.stack([data.train_dataset[i][0] for i in indices[split]])
        for split in ["calibration", "confirmation"]
    }
    for name, beta in [("expert", 1.0)] + [(str(b), b) for b in cfg["betas"]]:
        model = original if beta == 1 else rescale(original, anchor, beta)
        values, _ = capture(model, batches["calibration"], cfg["layers"])
        assert all(torch.equal(values[k], v) for k, v in captured[name].items())
    report, tensors = screen(
        original, anchor, states, load(str(root / f"head_{cfg['task']}.pt")), batches, cfg
    )
    assert report == read(out / "functions.json")
    saved = torch.load(out / "artifacts/features.pt", map_location="cpu", weights_only=True)
    assert saved.keys() == tensors.keys() and all(
        torch.equal(v, saved[k]) for k, v in tensors.items()
    )
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - started,
            source_sha256=digest(__file__),
            results_sha256=digest(out / "functions.json"),
            metadata_sha256=digest(out / "metadata.json"),
            all_six_eigenproblems_replayed=True,
            release_and_two_public_states_replayed=True,
            all_schur_energy_and_taylor_bounds_replayed=True,
            twenty_four_feature_tensors_replayed=True,
            all_head_screen_counts_recomputed=True,
            parent_inputs_recaptured=True,
            independent_direct_checks=rows,
            no_accuracy_or_security_claim=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/nonlinear_shear.json")
    parser.add_argument("--output", default="outputs/2026-10-11/nonlinear_shear")
    validate(parser.parse_args())
