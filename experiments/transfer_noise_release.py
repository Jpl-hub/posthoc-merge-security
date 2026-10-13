"""Evaluate a new partner with identical verified release tensors; no construction."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import checkpoint_loader,digest,evaluate,merge,paired_counts
from covariance_noise import dataset
from subspace_probe import make_loader,save_json
from verify_covariance_noise import official_labels


@torch.no_grad()
def run(args):
    source,out=Path(args.source),Path(args.output)
    validation=json.loads((source/"independent_validation.json").read_text());assert validation["status"]=="passed"
    meta=json.loads((source/"execution.json").read_text());cfg=json.loads((source/"protocol.json").read_text());case=json.loads((source/"case.json").read_text());target=case["target"]
    assert args.partner!=target
    assert digest(source/"artifacts/release.pt")==meta["release_sha256"]
    assert digest(source/"artifacts/released_AGA_edits.pt")==meta["AGA_edits_sha256"]
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path("checkpoints/ViT-B-32");load=checkpoint_loader(Path("external/secure-merge-attack"))
    pins={p["path"].removeprefix("ViT-B-32/"):p["sha256"] for p in json.loads(Path("configs/checkpoints.json").read_text())["files"]}
    names=["zeroshot.pt",f"{target}/finetuned.pt",f"{args.partner}/finetuned.pt",f"head_{target}.pt",f"head_{args.partner}.pt"]
    hashes={name:digest(root/name) for name in names};assert all(hashes[name]==pins[name] for name in names)
    anchor,original,partner=[load(str(root/name)).cpu().eval() for name in names[:3]]
    heads={t:load(str(root/f"head_{t}.pt")).cpu().eval() for t in [target,args.partner]}
    released=copy.deepcopy(original)
    edits=torch.load(source/"artifacts/release.pt",map_location="cpu")
    for key,value in edits.items():released.state_dict()[key].copy_(value)
    def recovered(model,task):
        ref=cfg["AGA_references"][task];assert digest(ref["path"])==ref["sha256"]
        m=copy.deepcopy(model);m.load_state_dict(torch.load(ref["path"],map_location="cpu"));return m
    ao=recovered(original,target);other=recovered(partner,args.partner);ar=copy.deepcopy(ao)
    for key,value in torch.load(source/"artifacts/released_AGA_edits.pt",map_location="cpu").items():ar.state_dict()[key].copy_(value)
    data={t:dataset(t,"data",original.val_preprocess) for t in heads}
    models=dict(original=original,released=released,partner=partner,original_AGA=ao,released_AGA=ar)
    for stage,left,right in [("direct",[("original",original),("released",released)],partner),("AGA",[("original",ao),("released",ar)],other)]:
        for name,model in left:
            if args.verify:
                base=anchor.state_dict();a=model.state_dict();b=right.state_dict();state={k:v+cfg["coefficient"]*((a[k]-v)+(b[k]-v)) if v.is_floating_point() else v.clone() for k,v in base.items()}
                merged=copy.deepcopy(anchor);merged.load_state_dict(state)
            else:merged=merge(anchor,model,right,cfg["coefficient"])
            models[name+"/"+stage]=merged
    prior_rows=json.loads((source/"results.json").read_text());prior_raw=torch.load(source/"artifacts/predictions.pt",map_location="cpu")
    if not args.verify:
        out.mkdir(parents=True,exist_ok=False);(out/"artifacts").mkdir()
        protocol=dict(source=str(source),target=target,partner=args.partner,same_release=True,
            release_sha256=meta["release_sha256"],source_predictions_sha256=meta["predictions_sha256"],
            source_AGA_edits_sha256=meta["AGA_edits_sha256"],source_validation_sha256=digest(source/"independent_validation.json"),
            runner_sha256=digest(__file__),checkpoint_sha256=hashes,coefficient=cfg["coefficient"],
            new_candidates=0,calibration_samples=0,parameter_search=False,public_cleaning_run=False)
        save_json(out/"protocol.json",protocol)
        rows=[dict(r,reused=True) for r in prior_rows if r["stage"] in ["standalone","AGA_standalone"]]
        raw={}
        def record(model,preset,stage,task):
            batches=[];score=evaluate(model,heads[task],make_loader(data[task]),"cuda",prediction_sink=lambda p,y:batches.append((p,y)))
            prediction,truth=[torch.cat([b[i] for b in batches]) for i in [0,1]]
            key="/".join([preset,stage,task]);raw[key]=dict(prediction=prediction,target=truth)
            classes=heads[task].weight.shape[0]
            row=dict(preset=preset,stage=stage,dataset=task,reused=False,**score,
                per_class=[dict(class_id=k,n=int(truth.eq(k).sum()),correct=int((prediction.eq(truth)&truth.eq(k)).sum())) for k in range(classes)])
            if preset=="released":
                baseline=raw["original/"+stage+"/"+task]["prediction"]
                row["paired"]=paired_counts(baseline.eq(truth),prediction.eq(truth));row["changed_predictions"]=int(prediction.ne(baseline).sum())
            rows.append(row);save_json(out/"results.json",rows);torch.save(raw,out/"artifacts/predictions.pt")
            print("TRANSFER_RESULT",json.dumps({k:row[k] for k in ["preset","stage","dataset","correct","n"]}),flush=True)
        record(partner,"original","partner_standalone",args.partner)
        for stage in ["direct","AGA"]:
            for name in ["original","released"]:
                for task in heads:record(models[name+"/"+stage],name,stage,task)
        save_json(out/"execution.json",dict(completed=True,rows=len(rows),new_candidates=0,same_release=True,
            release_sha256=meta["release_sha256"],predictions_sha256=digest(out/"artifacts/predictions.pt"),
            protocol_sha256=digest(out/"protocol.json"),runner_sha256=digest(__file__),new_attacks=0))
        return
    execution=json.loads((out/"execution.json").read_text());assert execution["runner_sha256"]==digest(__file__)
    assert execution["release_sha256"]==meta["release_sha256"] and digest(out/"artifacts/predictions.pt")==execution["predictions_sha256"]
    raw=torch.load(out/"artifacts/predictions.pt",map_location="cpu");rows=json.loads((out/"results.json").read_text());checks=[]
    for row in rows:
        preset,stage,task=[row[k] for k in ["preset","stage","dataset"]];key="/".join([preset,stage,task])
        stored=(prior_raw if row["reused"] else raw)[key];pred=stored["prediction"].numpy();truth=stored["target"].numpy()
        np.testing.assert_array_equal(truth,official_labels(task));assert row["correct"]==int((pred==truth).sum()) and row["n"]==len(truth)
        for c in row["per_class"]:
            chosen=truth==c["class_id"];assert c["n"]==int(chosen.sum()) and c["correct"]==int(((pred==truth)&chosen).sum())
        if stage=="standalone":model=models[preset]
        elif stage=="AGA_standalone":model=models[preset+"_AGA"]
        elif stage=="partner_standalone":model=partner
        else:model=models[preset+"/"+stage]
        image=torch.stack([data[task][i][0] for i in range(args.probe_n)]).cuda();model.cuda();classifier=heads[task].cuda()
        actual=classifier(model(image)).argmax(1).cpu().numpy();np.testing.assert_array_equal(actual,pred[:args.probe_n]);model.cpu();classifier.cpu()
        checks.append(dict(key=key,correct=row["correct"],n=row["n"],replay_n=args.probe_n))
    save_json(out/"independent_validation.json",dict(status="passed",same_release=True,release_sha256=meta["release_sha256"],
        new_candidates=0,new_attacks=0,fresh_predictions=len(rows)*args.probe_n,checks=checks,verifier_sha256=digest(__file__),explicit_merge_formula=True))
    print("TRANSFER_VERIFIED",len(rows),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--source",required=True);p.add_argument("--partner",required=True);p.add_argument("--output",required=True);p.add_argument("--verify",action="store_true");p.add_argument("--probe-n",type=int,default=32)
    run(p.parse_args())
