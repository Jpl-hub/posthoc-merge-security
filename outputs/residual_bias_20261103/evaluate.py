"""Evaluate the one frozen residual-bias point through the existing experiment entries."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import torch

from _impl.evaluate import run, verify
from _impl.io import digest, save_json
from _impl.models import apply, load_checkpoint

root = Path("outputs/residual_bias_20261103")
source = json.loads((root/"source.json").read_text())
peer = Path("outputs/params_composition_20261103")
parent = json.loads((peer/"evaluation.json").read_text())
release = dict(path=str(root/"release.pt"),sha256=source["release_sha256"])
assert digest(release["path"]) == release["sha256"]
name = "Residual_bias"

def measure(label, cfg):
    cfg["output"] = str(root/label)
    cfg["experiment"] = "Fixed residual-bias compensation, "+label
    cfg["cases"] = [dict(target="MNIST",partner="SVHN",defenses=[name])]
    if label.endswith("reset"):
        cfg["cases"][0]["public_replacement"] = dict(layer=11,side="B+b")
    path = root/(label+".json")
    save_json(path,cfg)
    run(path,0,False)
    folder = root/label/"full/00_MNIST_SVHN"
    verify(folder)
    return json.loads((folder/"results.json").read_text())

def target_accuracy(rows, merger, scope="None"):
    return next(r["accuracy_percent"] for r in rows if r["defense"]==name and r["dataset"]=="MNIST" and r["merger"]==merger and r["attack_scope"]==scope)

reset = copy.deepcopy(parent)
reset.update(defenses={name:{"release":release}},legacy={},attack_scopes=["None"])
local = measure("local_reset",reset)
aga = root/"AGA"
subprocess.run([sys.executable,"-m","experiment.recover","--method","aga","--target","MNIST","--aligned-reference","--protocol",str(peer/"evaluation.json"),"--publication",release["path"],"--sha256",release["sha256"],"--output",str(aga)],check=True)
ref = dict(path=str(aga/"edits.pt"),sha256=json.loads((aga/"execution.json").read_text())["output_sha256"])
native = copy.deepcopy(parent)
native["defenses"] = {name:dict(release=release,AGA_edits=ref)}
standard = measure("standard",native)

expert = load_checkpoint("MNIST/finetuned.pt")
base = load_checkpoint("zeroshot.pt")
clean_ref = parent["AGA_references"]["MNIST"]
clean_aga = apply(expert,clean_ref,full=True)
protected_aga = apply(clean_aga,ref)
# Compare both AGA outputs when all seven known locations are replaced identically.
keys = set(source["known_keys"])
original, changed = clean_aga.state_dict(), protected_aga.state_dict()
assert keys <= set(original) and set(original)==set(changed)
different_outside = [k for k in original if k not in keys and not torch.equal(original[k],changed[k])]
save_json(root/"all_known.json",dict(keys=sorted(keys),post_AGA_different_outside=different_outside,matched_all_known_reset_equal=not different_outside,states=len(original)))
full = root/"aligned.pt"
torch.save(changed, full)
del original,changed,base,expert,clean_aga,protected_aga
after = copy.deepcopy(reset)
after["defenses"] = {"Unprotected":dict(release=clean_ref,full_state=True),name:dict(release=dict(path=str(full),sha256=digest(full)),full_state=True)}
sequential = measure("AGA_then_reset",after)
# Reuse the previously registered matched raw recovery result as the decision threshold.
previous=json.loads(Path("results/data_free_compensation/finite_classification_result.json").read_text())["tasks"]["MNIST"]["recovery"]["replacement"]["rows"]
before=next(r["accuracy_percent"] for r in previous if r["defense"]!="Unprotected" and r["merger"]=="TA" and r["dataset"]=="MNIST")
eligible = target_accuracy(standard,"Standalone")>48.25 and target_accuracy(local,"TA")<before
save_json(root/"recovery_result.json",dict(native=target_accuracy(standard,"Standalone"),TA=target_accuracy(standard,"TA"),AGA_target=target_accuracy(standard,"TA","Protected expert"),AGA_both=target_accuracy(standard,"TA","All experts"),reset=target_accuracy(local,"TA"),AGA_then_reset=target_accuracy(sequential,"TA"),previous_reset=before,prodistill_eligible=eligible))
if eligible:
    cfg=json.loads((peer/"prodistill.json").read_text())
    cfg["cases"][0]["releases"]={name:release}
    cfg["output"]=str(root/"prodistill")
    save_json(root/"prodistill.json",cfg)
    subprocess.run([sys.executable,"-m","experiment.distill","--prodistill","--protocol",str(root/"prodistill.json")],check=True)
save_json(root/"completed.json",dict(status="completed",prodistill_executed=eligible,expanded_reset_equal=not different_outside))
