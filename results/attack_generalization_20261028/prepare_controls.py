"""Freeze the declared Gaussian and deletion controls; no image loading."""
import json
import time
from pathlib import Path

import torch

from _impl.attacks import author_recover, load_author
from _impl.io import digest, save_json
from _impl.metrics import logical_parameter_distance
from _impl.models import apply, initialize, load_checkpoint

initialize()
root = Path("results/attack_generalization_20261028")
definition = json.loads((root / "baseline_definition.json").read_text())
cfg = json.loads((root / "protocol.json").read_text())
cfg.update(output=definition["output"], attack_scopes=["None", "Protected expert", "All experts"], cases=[], defenses={}, legacy={})
cfg["attacker_permissions"] = dict(category=["weight-only"], operation="Fixed original-author AGA; no public replacement in this batch")
cfg.pop("no_target_control", None)
cfg["deduplication"] = "Reuse complete unprotected scores; MNIST A/B deletion controls already exist and are not rerun"
start = time.perf_counter()
out = Path(definition["output"])
out.mkdir(parents=True, exist_ok=False)
base = load_checkpoint("zeroshot.pt")
author, sha = load_author("external/secure-merge-attack/" + cfg["AGA"]["file"], True)
assert sha == cfg["AGA"]["sha256"]
publications = json.loads((root / "results.json").read_text())["publications"]
frozen = []
for target, specs in publications.items():
    expert = load_checkpoint(target + "/finetuned.pt")
    original = expert.state_dict()
    reference = apply(expert, cfg["AGA_references"][target], full=True)
    spec = specs["main"]
    assert digest(spec["path"]) == spec["sha256"]
    edits = torch.load(spec["path"], map_location="cpu", weights_only=False)
    generator = torch.Generator().manual_seed(definition["seed"])
    noise = {}
    norms = {}
    for key in sorted(edits):
        delta = edits[key].double() - original[key].double()
        draw = torch.randn(delta.shape, dtype=torch.float64, generator=generator)
        noise[key] = (original[key].double() + draw * (delta.norm() / draw.norm())).float()
        actual = float((noise[key].double() - original[key].double()).norm())
        norms[key] = dict(main_delta_norm=float(delta.norm()), random_delta_norm=actual)
        assert abs(actual - float(delta.norm())) <= 1e-5 * max(1., float(delta.norm()))
    controls = {"Random_noise": noise}
    if target != "MNIST":
        controls["A_only"] = {k: v for k, v in edits.items() if ".c_fc." in k}
        controls["B+b_only"] = {k: v for k, v in edits.items() if ".c_proj." in k}
    names = []
    for name, values in controls.items():
        folder = out / "publications" / target / name
        folder.mkdir(parents=True)
        torch.save(values, folder / "release.pt")
        release = dict(path=str(folder / "release.pt"), sha256=digest(folder / "release.pt"))
        model = apply(expert, release)
        attack_start = time.perf_counter()
        restored = author_recover(model, base, author).state_dict()
        aga = {k: v for k, v in restored.items() if not torch.equal(v, reference.state_dict()[k])}
        torch.save(aga, folder / "AGA_edits.pt")
        label = name + "_" + target
        names.append(label)
        cfg["defenses"][label] = dict(target=target, release=release, AGA_edits=dict(path=str(folder / "AGA_edits.pt"), sha256=digest(folder / "AGA_edits.pt")))
        frozen.append(dict(control=name, logical_distance=logical_parameter_distance(values, original), tensors=list(values), norm_comparison=norms if name == "Random_noise" else None, AGA_seconds=time.perf_counter()-attack_start, **cfg["defenses"][label]))
    partner = "SVHN" if target == "MNIST" else "MNIST"
    cfg["cases"].append(dict(target=target, partner=partner, defenses=names))
    main_path = specs["main"]["path"].split("/point/")[0]
    stage = "observation" if target == "DTD" else "evaluation"
    score_path = main_path + f"/{stage}/full/00_{target}_{partner}/artifacts/predictions.pt"
    cfg["legacy"][target + "_" + partner + "/Unprotected"] = {
        stage_name: dict(path=score_path, sha256=digest(score_path), key_template="Unprotected/" + model_stage + "/{task}")
        for stage_name, model_stage in [("standalone", "Standalone/None"), ("direct", "TA/None"), ("AGA_target_only", "TA/Protected expert"), ("AGA_both", "TA/All experts")]
    }
save_json(out / "construction.json", dict(controls=frozen, seconds=time.perf_counter()-start, real_construction_images=0, expert_updates=0, new_full_AGA_calls=len(frozen), source_sha256=digest(__file__), definition_sha256=digest(root / "baseline_definition.json")))
save_json(out / "evaluation_protocol.json", cfg)
print("CONTROLS_FROZEN", len(frozen), flush=True)
