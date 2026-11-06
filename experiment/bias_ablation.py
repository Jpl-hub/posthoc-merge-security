"""Fixed NBP component and amplitude controls; reuse the common evaluator."""

import argparse
import json
from pathlib import Path

import torch

from _impl.evaluate import run, verify
from _impl.io import check_files, digest, new_output, save_json
from _impl.models import checked_models, initialize
from cfg import require_root


@torch.no_grad()
def construct(cfg):
    initialize()
    torch.set_num_threads(2)
    _, expert, _, _ = checked_models(cfg["target"])
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
    controls["Random_null"] = random_null / random_null.norm() * delta.norm()
    controls["Ambient"] = random / random.norm() * delta.norm()
    out = new_output(cfg["output"], cfg)
    specs = {"NBP": {"target": cfg["target"], "release": ref, "AGA_edits": ref}}
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
                        "native_embedding_residual_norm": float((actual @ projection).norm())})
    evaluation = json.loads(Path(cfg["evaluation_template"]).read_text())
    original = json.loads(Path(cfg["original_evaluation"]).read_text())
    evaluation.update(tasks=original["tasks"], AGA_references=original["AGA_references"],
        output=str(out / "evaluation"), mergers=["TA"], TA_coefficient=0.5,
        attack_scopes=["None", "Protected expert"], defenses=specs,
        cases=[{"target": cfg["target"], "partner": "SVHN", "defenses": list(specs)}])
    evaluation["source_sha256"].update(cfg["source_sha256"])
    legacy = {}
    for name, old_name in [("Unprotected", "Unprotected"), ("NBP", "Defense1")]:
        legacy["MNIST_SVHN/" + name] = {
            stage: {**cfg["cached_scores"], "key_template": old_name + "/" + key_template}
            for stage, key_template in [("standalone", "Standalone/None/{task}"),
                ("direct", "TA/None/{task}"), ("AGA_target_only", "TA/Protected expert/{task}")]}
    evaluation["legacy"] = legacy
    save_json(out / "evaluation.json", evaluation)
    save_json(out / "freeze.json", {"controls": records,
        "evaluation_sha256": digest(out / "evaluation.json"),
        "main_publication_unchanged": ref, "real_images": 0, "synthetic_images": 0,
        "optimizer_steps": 0, "expert_updates": 0})
    print("FROZEN_CONTROLS", len(records), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--evaluate", action="store_true")
    args = parser.parse_args()
    require_root()
    cfg = json.loads(Path(args.protocol).read_text())
    check_files(cfg["source_sha256"] | cfg["input_sha256"])
    if not args.evaluate:
        construct(cfg)
        return
    out = Path(cfg["output"])
    freeze = json.loads((out / "freeze.json").read_text())
    assert digest(out / "evaluation.json") == freeze["evaluation_sha256"]
    for pilot in [True, False]:
        run(str(out / "evaluation.json"), 0, pilot)
        verify(out / "evaluation" / ("pilot" if pilot else "full") / "00_MNIST_SVHN")


if __name__ == "__main__":
    main()
