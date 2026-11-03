"""One paper-motivated pruning check; reuse frozen releases and native evaluator."""
import copy
import json
from pathlib import Path
import time

import numpy as np
import torch

from _impl.evaluate import run, verify
from _impl.io import digest, save_json
from _impl.models import apply, initialize, load_checkpoint

root = Path("outputs/params_pruning_20261104")
plan = json.loads(Path("results/comparison/params_pruning/protocol.json").read_text())
assert digest(__file__) == plan["driver_sha256"]
initialize()
started = time.monotonic()
expert = load_checkpoint("MNIST/finetuned.pt")
state = expert.state_dict()
suffixes = ("attn.in_proj_weight", "attn.out_proj.weight", "mlp.c_fc.weight", "mlp.c_proj.weight")
keys = sorted(k for k in state if k.startswith("model.visual.transformer.resblocks.") and k.endswith(suffixes))
assert len(keys) == 48
generator = torch.Generator().manual_seed(plan["seed"])
masks = {k: torch.rand(state[k].shape, generator=generator) >= plan["prune_probability"] for k in keys}
mask_record = {k: dict(total=v.numel(), removed=int((~v).sum())) for k,v in masks.items()}
cfg = json.loads(Path("outputs/params_composition_20261103/evaluation.json").read_text())
cfg.update(defenses={}, mergers=[], attack_scopes=["None"], output=str(root/"native"))
cfg["source_sha256"] = {p:digest(p) for p in cfg["source_sha256"]}
releases = {}
for name, ref in plan["inputs"].items():
    model = copy.deepcopy(expert) if ref is None else apply(expert, ref, ref.get("full_state", False))
    changed = model.state_dict()
    for key, mask in masks.items():
        before = changed[key].clone()
        changed[key] = before * mask
        np.testing.assert_array_equal(changed[key].numpy(), np.where(mask.numpy(), before.numpy(), np.float32(0)))
    edits = {k:v for k,v in changed.items() if not torch.equal(v, state[k])}
    path = root/(name+".pt")
    assert not path.exists()
    torch.save(edits, path)
    release = dict(path=str(path), sha256=digest(path))
    releases[name] = release
    cfg["defenses"][name] = dict(release=release)
    del model, changed, edits
save_json(root/"construction.json", dict(releases=releases, masks=mask_record, seed=plan["seed"],
    numpy_mask_check=True, real_construction_images=0, optimization_steps=0,
    construction_seconds=time.monotonic()-started))
cfg["cases"] = [dict(target="MNIST", partner="SVHN", defenses=list(releases))]
cfg["experiment"] = "Fixed 10 percent matrix pruning: native utility before any recovery; paper-motivated adaptation"
save_json(root/"native.json", cfg)
del masks, state, expert
run(root/"native.json", 0, False)
verify(root/"native/full/00_MNIST_SVHN")
save_json(root/"completed.json", dict(status="native_complete", wall_seconds=time.monotonic()-started,
    new_AGA_calls=0, distillation_steps=0, new_releases=len(releases)))
