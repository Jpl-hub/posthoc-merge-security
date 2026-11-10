"""Fixed NBP component and amplitude controls; reuse the common evaluator."""

import argparse
import json
import math
from pathlib import Path

import torch

from _impl.evaluate import run, verify
from _impl.defenses import construct_bias
from _impl.io import check_files, digest, new_output, save_json
from _impl.models import checked_models, initialize
from cfg import require_root


@torch.no_grad()
def construct(cfg):
    initialize()
    torch.set_num_threads(2)
    anchor, expert, head, _ = checked_models(cfg["target"])
    key = "model.visual.ln_post.bias"
    bias = expert.state_dict()[key].double()
    ref = cfg["publication"]
    check_files({ref["path"]: ref["sha256"]})
    main = torch.load(ref["path"], map_location="cpu", weights_only=False)
    assert list(main) == [key]
    delta = main[key].double() - bias
    projection = expert.model.visual.proj.double()
    left, singular, _ = torch.linalg.svd(projection, full_matrices=True)
    rank = int((singular > singular[0] * cfg["svd_tolerance"]).sum())
    assert rank == 512 and projection.shape == (768, 512)
    null = left[:, rank:]
    random = torch.randn(768, dtype=torch.double,
                         generator=torch.Generator().manual_seed(cfg["random_seed"]))
    random_null = null @ (null.T @ random)
    controls = {f"Scale_{scale:g}": scale * delta for scale in cfg["relative_amplitudes"]}
    construction_details = {}
    for seed in cfg.get("nbp_seeds", []):
        edits, details = construct_bias(expert, anchor, head,
            cfg["nbp_construction"] | {"noise_seed": seed})
        name = f"NBP_seed_{seed}"
        controls[name] = edits[key].double() - bias
        construction_details[name] = details
    if "random_seeds" in cfg:
        for seed in cfg["random_seeds"]:
            sample = torch.randn(768, dtype=torch.double,
                generator=torch.Generator().manual_seed(seed))
            projected = null @ (null.T @ sample)
            controls[f"Random_null_{seed}"] = projected / projected.norm() * delta.norm()
    else:
        controls["Random_null"] = random_null / random_null.norm() * delta.norm()
    if cfg.get("ambient_control", True):
        controls["Ambient"] = random / random.norm() * delta.norm()
    if "class_contrast_seed" in cfg:
        weights = head.weight.double()
        requested = int(torch.randint(len(weights), (1,), generator=
            torch.Generator().manual_seed(cfg["class_contrast_seed"])))
        proxy = (anchor.model.visual.proj.double() + projection) / 2
        contrast = weights[requested] - weights.mean(0)
        projected = null @ (null.T @ (proxy @ contrast))
        controls["Projected_contrast"] = projected / projected.norm() * delta.norm()
        if cfg.get("class_contrast_own_bound", False):
            from _impl.defenses import parameters
            from _impl.merge import merge
            params = parameters(merge(anchor, expert, anchor, .5), head)
            winner = int((projected @ params["projection"] @ weights.T).argmax())
            rivals = torch.arange(len(weights)) != winner
            q = params["projection"] @ (weights[winner] - weights[rivals]).T
            gaps = projected @ q
            assert torch.all(gaps > 1e-10)
            normal = params["gamma"][:, None] * q
            normal -= normal.mean(0, keepdim=True)
            worst = math.sqrt(len(bias)) * normal.norm(dim=0)
            magnitude = 2 * float(((worst - params["beta"] @ q).clamp_min(0) / (.5 * gaps)).max())
            update = magnitude * projected
            actual = (bias + update).float().double() - bias
            assert torch.all((params["beta"] + .5 * actual) @ q - worst > 0)
            controls["Projected_contrast"] = update
    out = new_output(cfg["output"], cfg)
    specs = ({"NBP": {"target": cfg["target"], "release": ref, "AGA_edits": ref}}
             if cfg.get("include_nbp", True) else {})
    records = []
    for name, update in controls.items():
        changed = (bias + update).float()
        actual = changed.double() - bias
        assert torch.isfinite(changed).all()
        path = out / (name + ".pt")
        torch.save({key: changed}, path)
        release = {"path": str(path), "sha256": digest(path)}
        specs[name] = {"target": cfg["target"], "release": release, "AGA_edits": release}
        records.append({"name": name, "publication": release,
                        "actual_delta_norm": float(actual.norm()),
                        "native_embedding_residual_norm": float((actual @ projection).norm()),
                        "construction": construction_details.get(name)})
    evaluation = json.loads(Path(cfg["evaluation_template"]).read_text())
    original = json.loads(Path(cfg["original_evaluation"]).read_text())
    evaluation.update(tasks=original["tasks"], AGA_references=original["AGA_references"],
        output=str(out / "evaluation"), mergers=["TA"], TA_coefficient=0.5,
        attack_scopes=["None", "Protected expert"], defenses=specs,
        cases=[{"target": cfg["target"], "partner": cfg.get("partner", "SVHN"), "defenses": list(specs)}])
    evaluation["source_sha256"].update(cfg["source_sha256"])
    legacy = {}
    for name, old_name in ([("Unprotected", "Unprotected"), ("NBP", "Defense1")]
                           if "cached_scores" in cfg else []):
        legacy["MNIST_SVHN/" + name] = {
            stage: {**cfg["cached_scores"], "key_template": old_name + "/" + key_template}
            for stage, key_template in [("standalone", "Standalone/None/{task}"),
                ("direct", "TA/None/{task}"), ("AGA_target_only", "TA/Protected expert/{task}")]}
    evaluation["legacy"] = cfg.get("legacy", legacy)
    if "nbp_seeds" in cfg:
        evaluation["tasks"] = {t: original["tasks"][t]
            for t in (cfg["target"], cfg["partner"])}
    save_json(out / "evaluation.json", evaluation)
    save_json(out / "freeze.json", {"controls": records,
        "evaluation_sha256": digest(out / "evaluation.json"),
        "main_publication_unchanged": ref, "real_images": 0, "synthetic_images": 0,
        "optimizer_steps": 0, "expert_updates": 0})
    print("FROZEN_CONTROLS", len(records), flush=True)


def evaluate(cfg):
    out = Path(cfg["output"])
    freeze = json.loads((out / "freeze.json").read_text())
    assert digest(out / "evaluation.json") == freeze["evaluation_sha256"]
    for pilot in [True, False]:
        run(str(out / "evaluation.json"), 0, pilot)
        case = json.loads((out / "evaluation.json").read_text())["cases"][0]
        verify(out / "evaluation" / ("pilot" if pilot else "full")
               / f"00_{case['target']}_{case['partner']}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    require_root()
    cfg = json.loads(Path(args.protocol).read_text())
    check_files(cfg["source_sha256"] | cfg["input_sha256"])
    action = evaluate if args.evaluate else construct
    if "targets" not in cfg:
        action(cfg)
        return
    original = json.loads(Path(cfg["original_evaluation"]).read_text())
    for target in cfg["targets"]:
        case = next(c for c in original["cases"] if c["target"] == target)
        action(cfg | {"target": target, "partner": case["partner"],
            "publication": original["defenses"][f"NBP_{target}"]["release"],
            "output": str(Path(cfg["output"]) / target)})


if __name__ == "__main__":
    main()
