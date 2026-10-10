"""Diagnose frozen response transfer using numerical least squares, without a release."""

from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path

import torch
from clip_benchmark import checkpoint_loader
from joint_posthoc import capture_blocks, digest
from subspace_probe import save_json
from torchvision.datasets import MNIST


def relative(a, b):
    return float((a - b).norm() / b.norm().clamp_min(1e-30))


def matrix_error(design, coefficient, target, teacher):
    residual = design @ coefficient - target
    return dict(
        residual_squared=float(residual.square().sum()),
        relative_target_error=float(residual.norm() / target.norm()),
        relative_teacher_error=float(residual.norm() / teacher.norm()),
    )


@torch.no_grad()
def run(args):
    config = json.loads(Path(args.config).read_bytes())
    parent, out = Path(config["parent"]), Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    prior = json.loads((Path(config["parent_diagnostics"]) / "metadata.json").read_bytes())
    for path, key in [
        ("artifacts/release.pt", "parent_release_sha256"),
        ("protocol.json", "parent_protocol_sha256"),
        ("training_indices.json", "parent_indices_sha256"),
    ]:
        assert digest(parent / path) == prior[key]
    indices = json.loads((parent / "training_indices.json").read_bytes())
    subsets = {
        "calibration": indices["calibration"],
        "confirmation_prefix": indices["confirmation"]["MNIST"][: config["confirmation_prefix"]],
    }
    assert len(subsets["calibration"]) == 256
    assert not set(subsets["calibration"]) & set(subsets["confirmation_prefix"])
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path(args.author_repo))
    root = Path(args.checkpoints)
    original = load(str(root / "MNIST/finetuned.pt"))
    anchor = load(str(root / "zeroshot.pt"))
    release = copy.deepcopy(original).cpu().eval()
    states = torch.load(parent / "artifacts/release.pt", map_location="cpu", weights_only=True)
    for layer, state in states.items():
        block = release.model.visual.transformer.resblocks[layer]
        current = block.state_dict()
        current.update(state)
        block.load_state_dict(current, strict=True)
    save_json(out / "protocol.json", config)
    signs = []
    for layer in range(12):
        base = anchor.model.visual.transformer.resblocks[layer].attn.out_proj
        teacher = original.model.visual.transformer.resblocks[layer].attn.out_proj
        released = release.model.visual.transformer.resblocks[layer].attn.out_proj
        delta_ratio = float((teacher.weight.double() - base.weight).norm() / base.weight.norm())
        correlation = float(
            (released.weight.double() * base.weight).sum() / base.weight.double().square().sum()
        )
        sign = -1 if correlation < 0 else 1
        signs.append(
            dict(
                layer=layer,
                task_delta_to_anchor_norm=delta_ratio,
                public_gain_estimate=correlation,
                sign_certificate=delta_ratio < 1,
                output_weight_recovered=torch.equal(sign * released.weight, teacher.weight),
                output_bias_recovered=torch.equal(sign * released.bias, teacher.bias),
            )
        )
    save_json(out / "gain_certificate.json", signs)
    dataset = MNIST(args.data, train=True, transform=original.val_preprocess)
    captures = {}
    start = time.perf_counter()
    for name, selected in subsets.items():
        images = torch.stack([dataset[i][0] for i in selected])
        captures[name] = (
            capture_blocks(original, images, config["layers"]),
            capture_blocks(release, images, config["layers"], details=True),
        )
    rows, spectra, artifacts = [], [], {}
    for layer in config["layers"]:
        mlp = copy.deepcopy(release.model.visual.transformer.resblocks[layer].mlp).double().cuda()
        fitted = torch.cat([mlp.c_proj.weight.T, mlp.c_proj.bias[None]], dim=0)
        data = {}
        for subset, (targets, current) in captures.items():
            item = current[layer]
            hidden = mlp.gelu(mlp.c_fc(item["mlp_input"].double().cuda()))
            h = torch.cat([hidden, torch.ones_like(hidden[:, :1])], dim=1)
            teacher = targets[layer]["output"].double().cuda()
            t = teacher - item["post_attention"].double().cuda()
            data[subset] = h, t, teacher
        h, t, _ = data["calibration"]
        u, singular, vh = torch.linalg.svd(h, full_matrices=False, driver=config["svd_driver"])
        cutoff_ratio = torch.finfo(h.dtype).eps ** 0.5
        keep = singular > cutoff_ratio * singular[0]
        coefficients = vh[keep].T @ ((u[:, keep].T @ t) / singular[keep, None])
        projection_residual = t - u[:, keep] @ (u[:, keep].T @ t)
        consistency = relative(h @ coefficients - t, projection_residual)
        assert torch.isfinite(coefficients).all() and consistency < 1e-5
        spectra.append(
            dict(
                layer=layer,
                calibration_tokens=len(h),
                features=h.shape[1],
                retained_rank=int(keep.sum()),
                relative_singular_cutoff=cutoff_ratio,
                largest_singular=float(singular[0]),
                smallest_retained_singular=float(singular[keep][-1]),
                coefficient_norm_ratio=float(coefficients.norm() / fitted.norm()),
                projection_residual_consistency=consistency,
            )
        )
        projected_target = u.T @ t
        orthogonal_target_squared = float((t - u @ projected_target).square().sum())
        raw = dict(
            singular=singular.cpu(),
            vh=vh.cpu(),
            retained=keep.cpu(),
            fitted=fitted.cpu(),
            least_squares=coefficients.cpu(),
            projected_target=projected_target.cpu(),
            orthogonal_target_squared=orthogonal_target_squared,
            moments={},
        )
        for subset, (h, t, teacher) in data.items():
            rows.append(
                dict(
                    layer=layer,
                    subset=subset,
                    tokens=len(h),
                    frozen=matrix_error(h, fitted, t, teacher),
                    least_squares=matrix_error(h, coefficients, t, teacher),
                )
            )
            raw["moments"][subset] = dict(
                gram=(h.T @ h).cpu(),
                cross=(h.T @ t).cpu(),
                target_squared=float(t.square().sum()),
                teacher_squared=float(teacher.square().sum()),
            )
        path = out / "artifacts" / f"layer_{layer}.pt"
        torch.save(raw, path)
        artifacts[path.name] = digest(path)
        save_json(out / "results.json", rows)
        save_json(out / "spectra.json", spectra)
        print("TRANSFER_PROBE_LAYER", layer, rows[-2:], flush=True)
        del (
            mlp,
            fitted,
            data,
            h,
            t,
            teacher,
            u,
            singular,
            vh,
            coefficients,
            raw,
            projection_residual,
        )
        del projected_target
        torch.cuda.empty_cache()
    save_json(
        out / "metadata.json",
        dict(
            completed=True,
            source_sha256=digest(__file__),
            config_sha256=digest(args.config),
            parent_release_sha256=prior["parent_release_sha256"],
            parent_indices_sha256=prior["parent_indices_sha256"],
            artifacts_sha256=artifacts,
            checkpoint_sha256={p: digest(root / p) for p in ["MNIST/finetuned.pt", "zeroshot.pt"]},
            helpers_sha256={
                p: digest(Path(__file__).with_name(p))
                for p in ["joint_posthoc.py", "clip_benchmark.py", "subspace_probe.py"]
            },
            elapsed_seconds=time.perf_counter() - start,
            optimizer_steps=0,
            diagnostic_fits=3,
            fitted_coefficients_installed=False,
            new_defense_validated=False,
            confirmation_previously_inspected=True,
        ),
    )


@torch.no_grad()
def audit(root):
    torch.set_num_threads(4)
    root = Path(root)
    metadata = json.loads((root / "metadata.json").read_bytes())
    rows = json.loads((root / "results.json").read_bytes())
    spectra = json.loads((root / "spectra.json").read_bytes())
    assert metadata["completed"] and len(rows) == 6 and len(spectra) == 3
    assert digest(__file__) == metadata["source_sha256"]
    assert all(
        digest(Path(__file__).with_name(p)) == h for p, h in metadata["helpers_sha256"].items()
    )
    for item in spectra:
        layer = item["layer"]
        path = root / "artifacts" / f"layer_{layer}.pt"
        assert digest(path) == metadata["artifacts_sha256"][path.name]
        raw = torch.load(path, map_location="cpu", weights_only=True)
        assert all(
            torch.isfinite(raw[k]).all() for k in ["singular", "vh", "fitted", "least_squares"]
        )
        expected_keep = raw["singular"] > item["relative_singular_cutoff"] * raw["singular"][0]
        assert torch.equal(expected_keep, raw["retained"])
        g = raw["moments"]["calibration"]["gram"]
        v = raw["vh"]
        assert relative(v @ v.T, torch.eye(len(v), dtype=v.dtype)) < 1e-8
        assert relative(g, v.T @ (raw["singular"].square()[:, None] * v)) < 1e-8
        cal = raw["moments"]["calibration"]
        assert (
            relative(cal["cross"], v.T @ (raw["singular"][:, None] * raw["projected_target"]))
            < 1e-8
        )
        assert (
            abs(
                float(raw["projected_target"].square().sum())
                + raw["orthogonal_target_squared"]
                - cal["target_squared"]
            )
            / cal["target_squared"]
            < 1e-8
        )
        for row in [r for r in rows if r["layer"] == layer]:
            moments = raw["moments"][row["subset"]]
            for label, key in [("frozen", "fitted"), ("least_squares", "least_squares")]:
                d = raw[key]
                if row["subset"] == "calibration":
                    residual = raw["singular"][:, None] * (v @ d) - raw["projected_target"]
                    error = float(residual.square().sum()) + raw["orthogonal_target_squared"]
                else:
                    error = float(
                        (d * (moments["gram"] @ d)).sum()
                        - 2 * (d * moments["cross"]).sum()
                        + moments["target_squared"]
                    )
                recorded = row[label]["residual_squared"]
                assert abs(error - recorded) / max(recorded, 1) < 1e-5
                for scale, denom in [
                    ("relative_target_error", "target_squared"),
                    ("relative_teacher_error", "teacher_squared"),
                ]:
                    assert abs((max(error, 0) / moments[denom]) ** 0.5 - row[label][scale]) < 1e-6
    signs = json.loads((root / "gain_certificate.json").read_bytes())
    assert len(signs) == 12 and all(
        r["sign_certificate"] and r["output_weight_recovered"] and r["output_bias_recovered"]
        for r in signs
    )
    save_json(
        root / "validation.json",
        dict(
            status="passed",
            rows=len(rows),
            raw_moments_checked=True,
            decomposition_and_truncation_checked=True,
            results_sha256=digest(root / "results.json"),
            gain_certificate_sha256=digest(root / "gain_certificate.json"),
            validator_source_sha256=digest(__file__),
            accuracy_or_new_defense_claim=False,
        ),
    )
    print("RESPONSE_TRANSFER_AUDIT_PASSED", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/response_transfer_probe.json")
    parser.add_argument("--output", default="outputs/2026-10-10/response_transfer_probe")
    parser.add_argument("--audit-output")
    parser.add_argument("--author-repo", default="external/secure-merge-attack")
    parser.add_argument("--checkpoints", default="checkpoints/ViT-B-32")
    parser.add_argument("--data", default="data")
    args = parser.parse_args()
    audit(args.audit_output) if args.audit_output else run(args)
