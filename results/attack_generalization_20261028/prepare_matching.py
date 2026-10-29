"""Freeze public MLP weight matching and its known-permutation control."""
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from _impl.attacks import public_mlp_matching
from _impl.data import dataset, make_loader
from _impl.evaluate import evaluate
from _impl.io import digest, save_json
from _impl.models import apply, initialize, load_checkpoint

initialize()
start = time.perf_counter()
root = Path("results/attack_generalization_20261028")
definition = json.loads((root / "matching_definition.json").read_text())
cfg = json.loads((root / "protocol.json").read_text())
cfg.update(output="outputs/attack_generalization_20261028/matching", cases=[], defenses={}, legacy={}, attack_scopes=["None"])
cfg.pop("no_target_control", None)
cfg["attacker_permissions"] = definition["attacker_permissions"]
cfg["source_sha256"] = {k: digest(k) for k in cfg["source_sha256"]}
out = Path(cfg["output"])
out.mkdir(parents=True, exist_ok=False)
assert digest(out.parent / "official/weight_matching.py") == definition["sha256"]
base = load_checkpoint("zeroshot.pt")
state = base.state_dict()
published = dict(state)
g = torch.Generator().manual_seed(definition["seed_known_control"])
for layer in definition["layers"]:
    prefix = f"model.visual.transformer.resblocks.{layer}.mlp."
    order = torch.randperm(state[prefix + "c_fc.weight"].shape[0], generator=g)
    for suffix in ["c_fc.weight", "c_fc.bias"]:
        published[prefix + suffix] = state[prefix + suffix][order]
    published[prefix + "c_proj.weight"] = state[prefix + "c_proj.weight"][:, order]
restored, control_detail = public_mlp_matching(published, state)
assert all(torch.equal(v, restored[k]) for k, v in state.items())
permuted = copy.deepcopy(base)
permuted.load_state_dict(published, strict=True)
data = dataset("MNIST", "data", base.val_preprocess)
ids = np.linspace(0, len(data)-1, 64, dtype=int).tolist()
head = load_checkpoint("head_MNIST.pt")
scores = []
for model in [base, permuted]:
    sink = []
    evaluate(model, head, make_loader(data, ids), "cuda", logit_sink=sink.append)
    scores.append(torch.cat(sink))
error = float((scores[0]-scores[1]).abs().max())
assert error <= cfg["score_tolerance"]
torch.save(dict(before=scores[0], permuted=scores[1], indices=ids), out / "known_permutation.pt")
save_json(out / "known_permutation.json", dict(status="passed", layers=control_detail, exact_restored_tensors=len(state), full_forward_maximum_score_error=error, evaluation_images=128, fitting_images=0))
inputs = json.loads((root / "results.json").read_text())["publications"]
records = []
for target, specs in inputs.items():
    expert = load_checkpoint(target + "/finetuned.pt")
    names = []
    for name, model in [("Unprotected", expert), ("Defense2_"+target, apply(expert, specs["main"])), ("NoSVD_"+target, apply(expert, specs["no_SVD"]))]:
        tick = time.perf_counter()
        recovered, detail = public_mlp_matching(model.state_dict(), state)
        edits = {k: v for k, v in recovered.items() if not torch.equal(v, expert.state_dict()[k])}
        folder = out / "publications" / target / name
        folder.mkdir(parents=True)
        torch.save(edits, folder / "edits.pt")
        spec = dict(target=target, release=dict(path=str(folder / "edits.pt"), sha256=digest(folder / "edits.pt")))
        # Each task has its own config because Unprotected is a matched operation.
        cfg["defenses"][name] = spec
        if name != "Unprotected":
            names.append(name)
        records.append(dict(name=name, seconds=time.perf_counter()-tick, layers=detail, **spec))
    per_task = copy.deepcopy(cfg)
    per_task["cases"] = [dict(target=target, partner="SVHN" if target=="MNIST" else "MNIST", defenses=names)]
    per_task["output"] = str(out / target)
    save_json(out / (target + "_protocol.json"), per_task)
save_json(out / "construction.json", dict(records=records, seconds=time.perf_counter()-start, real_fitting_images=0, synthetic_fitting_images=0, expert_updates=0, source_sha256=digest(__file__), definition_sha256=digest(root / "matching_definition.json")))
print("MATCHING_FROZEN", len(records), flush=True)
