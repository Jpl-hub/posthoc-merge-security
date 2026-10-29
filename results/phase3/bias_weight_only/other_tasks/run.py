"""Two missing Defense1 cases; unchanged construction and existing evaluators."""

import copy
import json
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from _impl.io import check_files, digest, new_output, save_json
from _impl.models import initialize, load_checkpoint


def command(module, *args):
    subprocess.run([sys.executable, "-m", module, *map(str, args)], check=True)


protocol = Path("results/phase3/bias_weight_only/other_tasks/protocol.json")
cfg = json.loads(protocol.read_text())
check_files(cfg["source_sha256"])
initialize()
out = new_output(cfg["output"], cfg)
for path in [*cfg["source_sha256"], str(protocol)]:
    destination = out / "source_archive" / path
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, destination)
for ref in cfg["unprotected_references"].values():
    check_files({ref["path"]: ref["sha256"]})
for target, ref in cfg["evaluation"]["AGA_references"].items():
    check_files({ref["path"]: ref["sha256"]})
    expert = load_checkpoint(f"{target}/finetuned.pt")
    recovered = torch.load(ref["path"], map_location="cpu", weights_only=False)
    key = next(k for k in expert.state_dict() if k.endswith("ln_post.bias"))
    assert torch.equal(expert.state_dict()[key], recovered[key]), target
    del expert, recovered
save_json(out / "AGA_transport.json", {"final_bias_unchanged": True, "new_AGA_calls": 0})

# Both publications are fixed before either task's image evaluation.
releases = {}
for target in cfg["targets"]:
    command("experiment.construct", "--method", "bias", "--target", target,
            "--protocol", protocol, "--output", out / target / "construction")
    path = out / target / "construction/release.pt"
    releases[target] = {"path": str(path), "sha256": digest(path)}
save_json(out / "frozen_before_images.json", releases)
print("BOTH_RELEASES_FROZEN", releases, flush=True)

for case in cfg["cases"]:
    target, partner = case["target"], case["partner"]
    ref = releases[target]
    native = out / target / "native"
    command("experiment.native", "--target", target, "--publication", ref["path"],
            "--sha256", ref["sha256"], "--test-only", "--output", native)
    command("experiment.native", "--verify", native)
    e = copy.deepcopy(cfg["evaluation"])
    e.update(output=str(out / target / "evaluation"), cases=[case], legacy={},
             source_sha256=cfg["source_sha256"],
             defenses={"Defense1": {"release": ref, "AGA_edits": ref}})
    raw = {"path": str(native / "full.pt"), "sha256": digest(native / "full.pt")}
    old = cfg["unprotected_references"][target]
    base = {"standalone": dict(raw, native_score_field="before")}
    for stage, scope in [("direct", "None"), ("AGA_target_only", "Protected expert"),
                         ("AGA_both", "All experts")]:
        base[stage] = dict(old, key_template=f"Unprotected/TA/{scope}/{{task}}")
    e["legacy"][f"{target}_{partner}/Unprotected"] = base
    e["legacy"][f"{target}_{partner}/Defense1"] = {
        "standalone": dict(raw, native_score_field="after")}
    evaluation_protocol = out / target / "evaluation_protocol.json"
    save_json(evaluation_protocol, e)
    for stage in ["pilot", "full"]:
        command("experiment.main_table", "--protocol", evaluation_protocol,
                *(["--pilot"] if stage == "pilot" else []))
        command("experiment.main_table", "--verify",
                Path(e["output"]) / stage / f"00_{target}_{partner}")
    print("CASE_COMPLETED", target, flush=True)
