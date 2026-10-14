"""Reconstruct the frozen release, recount official labels, replay TA/AGA predictions."""

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import checkpoint_loader,digest,paired_counts
from covariance_noise import dataset
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_probe_candidate import explicit_merge


@torch.no_grad()
def verify(output):
    output=Path(output);cfg=json.loads((output/"protocol.json").read_text());meta=json.loads((output/"execution.json").read_text());case=json.loads((output/"case.json").read_text())
    assert meta["completed"] and meta["fusion_run"] and meta["optimizer_steps"]==64 and meta["evaluation_optimizer_steps"]==0
    for file,key in [("release.pt","release_sha256"),("calibration.pt","calibration_sha256"),("predictions.pt","predictions_sha256"),("released_AGA_edits.pt","AGA_edits_sha256")]:
        assert digest(output/"artifacts"/file)==meta[key]
    assert digest("experiments/covariance_noise.py")==meta["runner_sha256"]
    for file,sha in cfg["source_sha256"].items():assert digest(file)==sha,file
    author=Path("external/secure-merge-attack/utils_attack_B32.py")
    author_text_sha=hashlib.sha256(author.read_text(encoding="utf-8").encode()).hexdigest()
    assert author_text_sha==json.loads(Path("configs/sources.json").read_text())["AGA"]["sha256"]
    freeze=json.loads((output/"release_freeze.json").read_text());assert not freeze["partner_loaded"] and not freeze["test_loaded"]
    assert not any(case["partner"] in k for k in freeze["checkpoint_sha256"])
    assert freeze["release_sha256"]==meta["release_sha256"]
    root=Path("checkpoints/ViT-B-32");load=checkpoint_loader(Path("external/secure-merge-attack"))
    for file,sha in meta["checkpoint_sha256"].items():assert digest(root/file)==sha,file
    target,peer=case["target"],case["partner"]
    anchor=load(str(root/"zeroshot.pt")).cpu().eval();own=load(str(root/target/"finetuned.pt")).cpu().eval()
    other=load(str(root/peer/"finetuned.pt")).cpu().eval()
    heads={t:load(str(root/f"head_{t}.pt")).cpu().eval() for t in [target,peer]}
    source=Path(cfg["publication_point"]);assert digest(source/"candidate.pt")==cfg["candidate_sha256"]
    point=torch.load(source/"candidate.pt",map_location="cpu")
    edits=torch.load(output/"artifacts/release.pt",map_location="cpu");protected=copy.deepcopy(own)
    prefix="model.visual.transformer.resblocks.11."
    keys=cfg.get("publication_keys",{"O":prefix+"attn.out_proj.weight","B":prefix+"mlp.c_proj.weight"});assert set(edits)==set(keys.values())
    for site,key in keys.items():
        expected=(own.state_dict()[key].double()+point["release_deltas"][site]).float()
        assert torch.equal(expected,edits[key]);protected.state_dict()[key].copy_(edits[key])
    def recovered(model,t):
        ref=cfg["AGA_references"][t];assert digest(ref["path"])==ref["sha256"]
        result=copy.deepcopy(model);result.load_state_dict(torch.load(ref["path"],map_location="cpu"),strict=True);return result
    own_a=recovered(own,target);other_a=recovered(other,peer);protected_a=copy.deepcopy(own_a)
    for key,value in torch.load(output/"artifacts/released_AGA_edits.pt",map_location="cpu").items():protected_a.state_dict()[key].copy_(value)
    raw=torch.load(output/"artifacts/predictions.pt",map_location="cpu");rows=json.loads((output/"results.json").read_text())
    ids=json.loads((output/"training_indices.json").read_text());hold=json.loads((output/"holdout_protocol.json").read_text())["indices"]
    data={t:dataset(t,"data",own.val_preprocess) for t in [target,peer]};training=dataset(target,"data",own.val_preprocess,True)
    labels={t:official_labels(t) for t in [target,peer]};train_labels=official_labels(target,True)
    checks=[]
    for row in rows:
        preset,stage,task=[row[k] for k in ["preset","stage","dataset"]];key="/".join([preset,stage,task]);record=raw[key]
        pred,y=record["prediction"].numpy(),record["target"].numpy()
        group=hold if stage=="holdout" else ids["confirmation"] if stage=="confirmation" else None
        expected=train_labels[group] if group is not None else labels[task]
        np.testing.assert_array_equal(y,expected)
        assert int((pred==y).sum())==row["correct"] and len(y)==row["n"]
        assert abs(row["top1"]-row["correct"]/row["n"])<1e-12
        for cls in row["per_class"]:
            k=cls["class_id"];assert cls["n"]==int((y==k).sum()) and cls["correct"]==int(((pred==y)&(y==k)).sum())
        if preset=="released":
            ref=raw["/".join(["original",stage,task])]["prediction"]
            assert row["paired"]==paired_counts(ref.eq(record["target"]),record["prediction"].eq(record["target"]))
            assert row["changed_predictions"]==int(record["prediction"].ne(ref).sum())
        if stage=="partner_standalone":model=other
        elif stage in ["holdout","confirmation","standalone"]:model=own if preset=="original" else protected
        elif stage=="AGA_standalone":model=own_a if preset=="original" else protected_a
        else:
            first=own if preset=="original" else protected
            second=other
            if stage=="AGA":first=own_a if preset=="original" else protected_a;second=other_a
            assert stage in ["direct","AGA"]
            model=explicit_merge(anchor,first,second,cfg["coefficient"])
        chosen=np.linspace(0,len(y)-1,min(16,len(y)),dtype=int)
        samples=[group[i] for i in chosen] if group is not None else chosen.tolist()
        source_data=training if group is not None else data[task]
        images=torch.stack([source_data[i][0] for i in samples]).cuda()
        model.cuda();heads[task].cuda();actual=heads[task](model(images)).argmax(1).cpu().numpy()
        model.cpu();heads[task].cpu();np.testing.assert_array_equal(actual,pred[chosen])
        checks.append({"key":key,"official_n":len(y),"replayed_predictions":len(chosen),"prediction_match":True})
    result={"status":"passed","source_sha256":digest(__file__),"release_sha256":meta["release_sha256"],"predictions_sha256":meta["predictions_sha256"],
            "original_checkpoint_sha256":meta["checkpoint_sha256"],"official_count_rows":len(rows),"per_class_and_paired_counts":"passed",
            "explicit_TA_formula":True,"author_AGA_source_pin_verified":True,"fresh_prediction_instances":sum(c["replayed_predictions"] for c in checks),"checks":checks,
            "author_AGA_text_sha256":author_text_sha,"author_AGA_raw_sha256":digest(author),"author_pin_encoding":"read_text UTF-8 with universal newline normalization; same as unchanged author loader",
            "construction_optimizer_steps":64,"evaluation_optimizer_steps":0,"new_candidates":0,"first_phase_changed":False}
    save_json(output/"independent_validation.json",result);print(json.dumps(result),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--output",required=True);args=parser.parse_args()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;verify(args.output)
