"""Fixed existing-method compression control for irreversible-defense hypotheses.

Reuse the pinned original GPTQ core without installing its LLM dependencies.
No bit sweep, gradients, partner access during construction or novelty claim.
"""

from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional, Tuple

import numpy as np
import torch
import torch.distributed as dist
from clip_benchmark import evaluate
from cross_span_cost import edited_state, public_delete
from eight_task_benchmark import digest, merge_all
from head_collapse import truth
from joint_posthoc import flatten_tokens
from local_erasure import cache, rescale
from operator_diagnostics import paired_counts
from path_subspace import context, read, response
from subspace_probe import make_loader, save_json
from torch import Tensor, nn
from torch.nn.modules.conv import _ConvNd


def tensor_digest(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def author_core(cfg):
    """Load unchanged AST definitions in private dictionaries, never legacy src."""
    audit = read(cfg["source_audit"])
    assert audit["commit"] == cfg["upstream_commit"]
    root = Path(cfg["upstream_root"])
    shared = dict(torch=torch, nn=nn, dist=dist, Tensor=Tensor, _ConvNd=_ConvNd, np=np)
    shared.update(Optional=Optional, Tuple=Tuple)

    def extract(name, selected):
        path = root / "src" / name
        assert digest(path) == audit["files"]["src/" + name]["sha256"]
        nodes = [
            n
            for n in ast.parse(path.read_text()).body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in selected
        ]
        assert {n.name for n in nodes} == set(selected)
        namespace = dict(shared)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
        return namespace

    shared["dist_utils"] = SimpleNamespace(
        **{
            k: v
            for k, v in extract(
                "dist_utils.py", ["is_dist_available_and_initialized", "get_rank", "is_main"]
            ).items()
            if not k.startswith("__")
        }
    )
    shared["linalg_utils"] = SimpleNamespace(**extract("linalg_utils.py", ["inv_sym"]))
    shared["model_utils"] = SimpleNamespace(
        **extract("model_utils.py", ["get_number_of_rows_and_cols"])
    )
    quant = extract(
        "quant_utils_gptq.py", ["quantize_dequantize", "quantize", "dequantize", "Quantizer"]
    )
    shared["Quantizer"] = quant["Quantizer"]
    return extract("gptq.py", ["GPTQ"])["GPTQ"], quant["dequantize"]


@torch.no_grad()
def inputs_for(model, name, images):
    raw = []
    module = model.get_submodule(name)
    handle = module.register_forward_pre_hook(lambda _m, args: raw.append(flatten_tokens(args[0])))
    model.cuda().eval()
    try:
        for batch in images.split(16):
            model(batch.cuda())
    finally:
        handle.remove()
        model.cpu()
    return torch.cat(raw)


def decode(code, dequantize):
    q, scale, zero = (code[k] for k in ["qweight", "scale", "zero"])
    weight = dequantize(
        q.view(q.shape[0], scale.shape[1], -1), scale[:, :, None], zero[:, :, None]
    ).view_as(q)
    if code["perm"] is not None:
        weight = weight[:, code["perm"].argsort()]
    return weight.float()


@torch.no_grad()
def construct(original, images, cfg):
    GPTQ, dequantize = author_core(cfg)
    release = copy.deepcopy(original).cpu().eval()
    codes, checks = {}, []
    for layer in cfg["layers"]:
        for kind in ["c_fc", "c_proj"]:
            name = f"model.visual.transformer.resblocks.{layer}.mlp.{kind}"
            x = inputs_for(release, name, images)
            module = release.get_submodule(name)
            old = module.weight.detach().clone()
            quantizer = GPTQ(module.cuda(), **cfg["gptq"])
            quantizer.update(x.cuda())
            q, scale, zero, perm = quantizer.quantize()
            code = {
                k: v.cpu() if isinstance(v, Tensor) else v
                for k, v in dict(qweight=q, scale=scale, zero=zero, perm=perm).items()
            }
            weight = decode(code, dequantize)
            assert weight.shape == old.shape and torch.isfinite(weight).all()
            module.cpu().weight.copy_(weight)
            difference = (weight.double() - old.double()).cuda()
            output = x.double().cuda() @ difference.T
            native = x.double().cuda() @ old.double().cuda().T
            row = dict(
                module=name,
                calibration_input_sha256=tensor_digest(x),
                calibration_tokens=len(x),
                calibration_linear_relative_rms=float(output.norm() / native.norm()),
                weight_relative_error=float(difference.norm() / old.double().norm()),
                changed_weight_elements=int(weight.ne(old).sum()),
                decoded_weight_sha256=tensor_digest(weight),
            )
            checks.append(row)
            codes[name] = code
            print("QUANTIZATION", row, flush=True)
            del quantizer, difference, output, native, x
    edits = edited_state(original, release)
    assert len(edits) == 2 * len(cfg["layers"])
    assert all(k.endswith(".weight") and ".mlp." in k for k in edits)
    return release, codes, edits, checks


@torch.no_grad()
def support_reset(reference, anchor, layers):
    result = copy.deepcopy(reference).cpu().eval()
    for layer in layers:
        result.model.visual.transformer.resblocks[layer].mlp.load_state_dict(
            anchor.model.visual.transformer.resblocks[layer].mlp.state_dict()
        )
    return result


def sampling(cfg, data):
    calibration = read(cfg["calibration_indices"])[cfg["task"]]["selection"][
        : cfg["calibration_samples"]
    ]
    confirmation = read(cfg["confirmation_indices"])[cfg["task"]]["confirmation"]
    assert len(calibration) == cfg["calibration_samples"]
    assert not set(calibration).intersection(confirmation)
    images = {
        key: torch.stack([data.train_dataset[i][0] for i in indices])
        for key, indices in dict(calibration=calibration, confirmation=confirmation).items()
    }
    return dict(calibration=calibration, confirmation=confirmation), images


def runtime():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


@torch.no_grad()
def run(args):
    start, cfg, out = time.perf_counter(), read(args.config), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    artifacts = out / "artifacts"
    artifacts.mkdir()
    runtime()
    root, load, fusion, models = context(cfg)
    assert cfg["coefficient"] == fusion["coefficient"] == 0.2
    original = load(str(root / cfg["task"] / "finetuned.pt")).cpu().eval()
    from dataset.registry import get_dataset

    data = get_dataset(cfg["task"], original.val_preprocess, "data", batch_size=128, num_workers=2)
    indices, images = sampling(cfg, data)
    save_json(out / "protocol.json", cfg)
    save_json(out / "training_indices.json", indices)
    release, codes, edits, checks = construct(original, images["calibration"], cfg)
    torch.save(edits, artifacts / "release_edits.pt")
    torch.save(codes, artifacts / "quantization_codes.pt")
    save_json(out / "construction_checks.json", checks)
    save_json(
        out / "release_freeze.json",
        dict(
            edits_sha256=digest(artifacts / "release_edits.pt"),
            future_partner_checkpoints_loaded=False,
            labels_used=False,
            optimizer_steps=0,
            additional_response_fits=0,
            new_method_or_defense_claim=False,
        ),
    )
    functions = {}
    for name, beta in [("expert", 1.0)] + [(str(b), b) for b in cfg["betas"]]:
        before = original if beta == 1 else rescale(original, models["anchor"], beta)
        after = release if beta == 1 else rescale(release, models["anchor"], beta)
        functions[name] = {k: response(before, after, x) for k, x in images.items()}
    save_json(out / "functions.json", functions)
    deleted = support_reset(release, models["anchor"], cfg["layers"])
    same = support_reset(original, models["anchor"], cfg["layers"])
    assert all(torch.equal(v, same.state_dict()[k]) for k, v in deleted.state_dict().items())
    a, _ = public_delete(original, models["anchor"], cfg["layers"])
    b, _ = public_delete(release, models["anchor"], cfg["layers"])
    presets = dict(
        original=original,
        quantized=release,
        public_cross_span_original=a,
        public_cross_span_quantized=b,
        full_selected_mlp_reset=deleted,
    )
    for name, model in presets.items():
        if name not in ("original", "quantized"):
            torch.save(edited_state(original, model), artifacts / f"{name}_edits.pt")
    models.update({t: load(str(root / t / "finetuned.pt")).cpu().eval() for t in fusion["tasks"]})
    tasks = cfg["evaluation_tasks"]
    data = {
        t: get_dataset(t, original.val_preprocess, "data", batch_size=128, num_workers=2)
        for t in tasks
    }
    loaders, labels = (
        {t: cache(d) for t, d in data.items()},
        {t: truth(d, t) for t, d in data.items()},
    )
    heads = {t: load(str(root / f"head_{t}.pt")) for t in tasks}
    raw, rows = {}, []
    for name, model in presets.items():
        merged = merge_all(dict(models, **{cfg["task"]: model}), root, cfg["coefficient"])
        for scope, actual, measured in [
            ("standalone", model, [cfg["task"]]),
            ("fusion", merged, tasks),
        ]:
            for task in measured:
                assert time.perf_counter() - start < cfg["time_limit_seconds"]
                p, y = [], []
                score = evaluate(
                    actual,
                    heads[task],
                    loaders[task],
                    "cuda",
                    prediction_sink=lambda a, b: (p.append(a), y.append(b)),
                )
                key = f"{name}/{scope}/{task}"
                value = dict(prediction=torch.cat(p), labels=torch.cat(y))
                assert torch.equal(value["labels"], labels[task])
                assert score["n"] == fusion["expected_test_counts"][task]
                assert score["correct"] == int(value["prediction"].eq(value["labels"]).sum())
                raw[key] = value
                row = dict(key=key, preset=name, scope=scope, dataset=task, reused=False, **score)
                if name != "original":
                    source = f"original/{scope}/{task}"
                    row.update(
                        matched_reference=source,
                        matched_change=paired_counts(
                            raw[source]["prediction"].eq(raw[source]["labels"]),
                            value["prediction"].eq(value["labels"]),
                        ),
                    )
                if name == "public_cross_span_quantized":
                    source = f"public_cross_span_original/{scope}/{task}"
                    row["same_attack_reference"] = source
                    row["same_attack_change"] = paired_counts(
                        raw[source]["prediction"].eq(raw[source]["labels"]),
                        value["prediction"].eq(value["labels"]),
                    )
                rows.append(row)
                print("EVALUATION", key, score["correct"], score["n"], flush=True)
    assert len(rows) == 25
    torch.save(raw, artifacts / "predictions.pt")
    save_json(out / "results.json", rows)
    launch = read(out / "launch.json")
    checkpoint_pins = {str(p.relative_to(root)): digest(p) for p in sorted(root.rglob("*.pt"))}
    assert checkpoint_pins == {
        row["path"].split("/", 1)[1]: row["sha256"]
        for row in read("configs/checkpoints.json")["files"]
    }
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            stage="existing_method_mechanism_control",
            new_method_or_defense_claim=False,
            elapsed_seconds=time.perf_counter() - start,
            evaluations=25,
            source_manifest=launch["source_manifest"],
            checkpoint_sha256=checkpoint_pins,
            upstream_commit=cfg["upstream_commit"],
            full_selected_mlp_reset_states_identical=True,
            aga_or_query_run=False,
            raw_sha256={p.name: digest(p) for p in artifacts.iterdir()},
        ),
    )


@torch.no_grad()
def validate(args):
    start, out = time.perf_counter(), Path(args.output)
    cfg, meta = read(out / "protocol.json"), read(out / "metadata.json")
    assert meta["completed"] and cfg == read(args.config)
    for name, expected in meta["source_manifest"].items():
        assert digest(name) == expected
    for name, expected in meta["raw_sha256"].items():
        assert digest(out / "artifacts" / name) == expected
    runtime()
    root, load, fusion, models = context(cfg, all_tasks=True)
    for name, expected in meta["checkpoint_sha256"].items():
        assert digest(root / name) == expected
    original = models[cfg["task"]]
    from dataset.registry import get_dataset

    data = get_dataset(cfg["task"], original.val_preprocess, "data", batch_size=128, num_workers=2)
    indices, images = sampling(cfg, data)
    assert indices == read(out / "training_indices.json")
    release, codes, edits, checks = construct(original, images["calibration"], cfg)
    saved = torch.load(
        out / "artifacts/quantization_codes.pt", map_location="cpu", weights_only=True
    )
    for name, code in codes.items():
        assert all(torch.equal(v, saved[name][k]) for k, v in code.items() if isinstance(v, Tensor))
        assert int(code["qweight"].max()) < 2 ** cfg["gptq"]["bitwidth"]
        assert float(code["scale"].min()) > 0
        assert torch.equal(code["perm"].sort().values, torch.arange(len(code["perm"])))
        independent = (
            (
                code["qweight"]
                .view(code["qweight"].shape[0], -1, cfg["gptq"]["group_size"])
                .float()
                - code["zero"][:, :, None]
            )
            * code["scale"][:, :, None]
        ).view_as(code["qweight"])[:, code["perm"].argsort()]
        assert torch.equal(independent, release.get_submodule(name).weight)
    stored = torch.load(out / "artifacts/release_edits.pt", map_location="cpu", weights_only=True)
    assert edits.keys() == stored.keys() and all(
        torch.equal(v, stored[k]) for k, v in edits.items()
    )
    assert checks == read(out / "construction_checks.json")
    a, _ = public_delete(original, models["anchor"], cfg["layers"])
    b, _ = public_delete(release, models["anchor"], cfg["layers"])
    reset = support_reset(release, models["anchor"], cfg["layers"])
    reset_original = support_reset(original, models["anchor"], cfg["layers"])
    assert all(
        torch.equal(v, reset_original.state_dict()[k]) for k, v in reset.state_dict().items()
    )
    presets = dict(
        original=original,
        quantized=release,
        public_cross_span_original=a,
        public_cross_span_quantized=b,
        full_selected_mlp_reset=reset,
    )
    for name, model in presets.items():
        if name in ("original", "quantized"):
            continue
        actual = edited_state(original, model)
        expected = torch.load(
            out / "artifacts" / f"{name}_edits.pt", map_location="cpu", weights_only=True
        )
        assert actual.keys() == expected.keys()
        assert all(torch.equal(v, expected[k]) for k, v in actual.items())
    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu", weights_only=True)
    rows = read(out / "results.json")
    assert len(rows) == len(raw) == 25
    for task in cfg["evaluation_tasks"]:
        data = get_dataset(task, original.val_preprocess, "data", batch_size=128, num_workers=2)
        labels = truth(data, task)
        for row in [r for r in rows if r["dataset"] == task]:
            value = raw[row["key"]]
            assert torch.equal(value["labels"], labels)
            assert row["correct"] == int(value["prediction"].eq(labels).sum())
            assert row["n"] == len(labels) == fusion["expected_test_counts"][task]
            assert row["top1"] == row["correct"] / row["n"]
            if "matched_reference" in row:
                assert row["matched_change"] == paired_counts(
                    raw[row["matched_reference"]]["prediction"].eq(labels),
                    value["prediction"].eq(labels),
                )
            if "same_attack_reference" in row:
                assert row["same_attack_change"] == paired_counts(
                    raw[row["same_attack_reference"]]["prediction"].eq(labels),
                    value["prediction"].eq(labels),
                )
        head = load(str(root / f"head_{task}.pt"))
        loader = make_loader(data.test_dataset, list(range(128)), 128)
        for name, model in presets.items():
            merged = merge_all(dict(models, **{cfg["task"]: model}), root, cfg["coefficient"])
            pairs = [("fusion", merged)] + ([("standalone", model)] if task == cfg["task"] else [])
            for scope, actual in pairs:
                predicted = []
                evaluate(
                    actual, head, loader, "cuda", prediction_sink=lambda p, _y: predicted.append(p)
                )
                assert torch.equal(
                    torch.cat(predicted), raw[f"{name}/{scope}/{task}"]["prediction"][:128]
                )
    functions = read(out / "functions.json")
    for name, beta in [("expert", 1.0)] + [(str(b), b) for b in cfg["betas"]]:
        before = original if beta == 1 else rescale(original, models["anchor"], beta)
        after = release if beta == 1 else rescale(release, models["anchor"], beta)
        assert {k: response(before, after, x) for k, x in images.items()} == functions[name]
    save_json(
        out / "validation.json",
        dict(
            status="passed",
            elapsed_seconds=time.perf_counter() - start,
            quantized_weights_and_codes_replayed=True,
            independent_decoding_checked=True,
            sequential_calibration_inputs_replayed=True,
            full_count_and_paired_checks=25,
            official_prediction_prefixes_replayed=25,
            path_response_checks_replayed=True,
            reset_states_bit_identical=True,
            no_defense_claim=True,
            metadata_sha256=digest(out / "metadata.json"),
            results_sha256=digest(out / "results.json"),
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/quantization_probe.json")
    parser.add_argument("--output", default="outputs/2026-10-11/quantization_probe")
    parser.add_argument("--validate", action="store_true")
    args = parser.parse_args()
    (validate if args.validate else run)(args)
