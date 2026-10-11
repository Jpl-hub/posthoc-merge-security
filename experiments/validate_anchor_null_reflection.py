"""Replay the anchor-null screen and independently check its compiled algebra."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from anchor_null_reflection import anchor_basis, direction
from critical_subspace import capture
from eight_task_benchmark import digest
from local_erasure import rescale
from path_subspace import context, read, response
from subspace_probe import save_json


@torch.no_grad()
def validate(args):
    started, out = time.perf_counter(), Path(args.output)
    cfg, meta = read(args.config), read(out / "metadata.json")
    assert cfg == read(out / "protocol.json")
    assert meta["completed"] and not meta["defense_success_claimed"]
    assert digest("experiments/anchor_null_reflection.py") == meta["source_sha256"]
    assert digest(args.config) == meta["config_sha256"]
    for name, expected in meta["helper_sha256"].items():
        assert digest(Path("experiments") / name) == expected
    for name, expected in meta["input_sha256"].items():
        assert digest(name) == expected
    for name, expected in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == expected
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    root, load, _, models = context(cfg)
    for name, expected in meta["checkpoint_sha256"].items():
        assert digest(root / name) == expected
    original = load(str(root / cfg["task"] / "finetuned.pt")).cpu().eval()
    anchor = models["anchor"]
    directions = torch.load(out / "artifacts/directions.pt", map_location="cpu", weights_only=True)
    captured = torch.load(
        Path(cfg["parent"]) / "artifacts/inputs.pt", map_location="cpu", weights_only=True
    )
    sources, edits = {}, {}
    for name in ["release", "public_original", "public_protected"]:
        edits[name] = torch.load(
            out / f"artifacts/{name}_edits.pt", map_location="cpu", weights_only=True
        )
        model = copy.deepcopy(original)
        state = model.state_dict()
        for key, value in edits[name].items():
            state[key].copy_(value)
        assert all(
            torch.equal(v, original.state_dict()[k])
            for k, v in state.items()
            if k not in edits[name]
        )
        sources[name] = model
    rows = []
    for layer in cfg["layers"]:
        mlp = original.model.visual.transformer.resblocks[layer].mlp
        base = anchor.model.visual.transformer.resblocks[layer].mlp
        span, null, _ = anchor_basis(base, cfg["anchor_rank_rcond"])
        vector, _, evidence, _ = direction(mlp, base, captured, layer, cfg, null)
        saved = directions[str(layer)]
        assert abs(float(vector.cpu() @ saved["vector"]) - 1) < 1e-9
        assert all(torch.equal(evidence[k], saved[k]) for k in ["native_gram", "path_gram"])
        assert evidence["ridge"] == saved["ridge"] and evidence["eigenvalue"] == saved["eigenvalue"]
        v = saved["vector"].double().cuda()
        identity = torch.eye(len(v), dtype=torch.float64, device="cuda")
        q = identity - 2 * v[:, None] @ v[None, :]
        projector = span @ span.T
        assert float((q.T @ q - identity).abs().max()) < 1e-10
        assert float((projector @ q - projector).abs().max()) < 1e-10
        anchor_errors = []
        for tensor in [base.c_fc.weight, base.c_fc.bias[:, None], base.c_proj.weight.T]:
            t = tensor.double().cuda()
            anchor_errors.append(float((q @ t - t).norm() / t.norm()))
        assert max(anchor_errors) < 1e-10
        w, b, down = [
            t.double().cuda() for t in [mlp.c_fc.weight, mlp.c_fc.bias, mlp.c_proj.weight]
        ]
        expected = [q @ w, q @ b, down @ q]
        compiled = sources["release"].model.visual.transformer.resblocks[layer].mlp
        rounding = []
        for actual, wanted in zip(
            [compiled.c_fc.weight, compiled.c_fc.bias, compiled.c_proj.weight],
            expected,
            strict=True,
        ):
            error = float((actual.double().cuda() - wanted).abs().max())
            assert error < 1e-6
            rounding.append(error)
        for name, teacher in [("public_original", mlp), ("public_protected", compiled)]:
            current = sources[name].model.visual.transformer.resblocks[layer].mlp
            wanted = [
                projector @ teacher.c_fc.weight.double().cuda(),
                projector @ teacher.c_fc.bias.double().cuda(),
                teacher.c_proj.weight.double().cuda() @ projector,
            ]
            for actual, target in zip(
                [current.c_fc.weight, current.c_fc.bias, current.c_proj.weight], wanted, strict=True
            ):
                assert float((actual.double().cuda() - target).abs().max()) < 1e-6
            assert torch.equal(current.c_proj.bias, mlp.c_proj.bias)
        rows.append(
            dict(
                layer=layer,
                anchor_fixed_relative_errors=anchor_errors,
                direct_reflection_compilation_max_errors=rounding,
                public_projector_algebra_verified=True,
            )
        )
    indices = read(out / "training_indices.json")
    assert indices == read(Path(cfg["parent"]) / "training_indices.json")
    assert not set(indices["calibration"]).intersection(indices["confirmation"])
    from dataset.registry import get_dataset

    data = get_dataset(cfg["task"], original.val_preprocess, "data", batch_size=128, num_workers=2)
    batches = {
        split: torch.stack([data.train_dataset[i][0] for i in indices[split]])
        for split in ["calibration", "confirmation"]
    }
    for name, beta in [("expert", 1.0)] + [(str(t), t) for t in cfg["betas"]]:
        model = original if beta == 1 else rescale(original, anchor, beta)
        inputs, _ = capture(model, batches["calibration"], cfg["layers"])
        assert all(torch.equal(inputs[k], v) for k, v in captured[name].items())
    functions = read(out / "functions.json")
    for name, model in sources.items():
        for path, beta in [("expert", 1.0)] + [(str(t), t) for t in cfg["betas"]]:
            before = original if beta == 1 else rescale(original, anchor, beta)
            after = model if beta == 1 else rescale(model, anchor, beta)
            for split, images in batches.items():
                actual = response(before, after, images)
                expected = functions[name][path][split]
                assert abs(actual["actual_feature_rms"] - expected["actual_feature_rms"]) < 1e-10
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - started,
            source_sha256=digest(__file__),
            results_sha256=digest(out / "functions.json"),
            metadata_sha256=digest(out / "metadata.json"),
            three_directions_replayed=True,
            all_eigh_grams_replayed=True,
            parent_inputs_recaptured=True,
            eighteen_end_to_end_responses_replayed=True,
            direct_matrix_checks=rows,
            no_accuracy_or_security_claim=True,
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/anchor_null_reflection.json")
    parser.add_argument("--output", default="outputs/2026-10-11/anchor_null_reflection")
    validate(parser.parse_args())
