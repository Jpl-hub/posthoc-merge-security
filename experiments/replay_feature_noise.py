"""Reuse one verified release for DTD; no calibration or protection construction."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from clip_benchmark import author_recover, checkpoint_loader, digest, evaluate, merge, paired_counts
from mechanisms import load_author
from subspace_probe import make_loader, save_json
from torchvision.datasets import DTD, MNIST


@torch.no_grad()
def run(args):
    source, out = Path(args.source), Path(args.output)
    validation = json.loads((source / "independent_validation.json").read_text())
    assert validation["status"] == "passed"
    meta = json.loads((source / "execution.json").read_text())
    state_hashes = json.loads((source / "AGA_states.json").read_text())
    assert digest(source / "artifacts/release.pt") == meta["release_sha256"]
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path("external/secure-merge-attack"))
    root = Path("checkpoints/ViT-B-32")
    files = json.loads(Path("configs/checkpoints.json").read_text())["files"]
    hashes = {}
    for name in ["zeroshot.pt", "MNIST/finetuned.pt", "DTD/finetuned.pt", "head_MNIST.pt", "head_DTD.pt"]:
        hashes[name] = digest(root / name)
        assert hashes[name] == next(p["sha256"] for p in files if p["path"] == "ViT-B-32/" + name)
    anchor, original, partner = [load(str(root / name)).cpu().eval()
                                 for name in ["zeroshot.pt", "MNIST/finetuned.pt", "DTD/finetuned.pt"]]
    released = copy.deepcopy(original)
    artifact = torch.load(source / "artifacts/release.pt", map_location="cpu")
    layer = released.model.visual.transformer.resblocks[11].mlp.c_proj
    layer.weight.copy_(artifact["released_weight"])
    layer.bias.copy_(artifact["released_bias"])
    cfg = json.loads((source / "protocol.json").read_text())
    models = {"original": original, "released": released, "partner": partner}
    for name in ["original", "released"]:
        assert digest(source / f"artifacts/{name}_AGA.pt") == state_hashes[name]
        model = copy.deepcopy(original)
        model.load_state_dict(torch.load(source / f"artifacts/{name}_AGA.pt", map_location="cpu"))
        models["attacked/" + name] = model
    datasets = dict(MNIST=MNIST("data", train=False, transform=original.val_preprocess),
                    DTD=DTD("data", split="test", partition=1, transform=original.val_preprocess))
    heads = {task:load(str(root / f"head_{task}.pt")).cpu().eval() for task in datasets}
    if not args.verify:
        out.mkdir(parents=True, exist_ok=False)
        (out / "artifacts").mkdir()
        save_json(out / "protocol.json", dict(
            coefficient=cfg["coefficient"], partner="DTD", release_sha256=meta["release_sha256"],
            source=str(source), same_release=True, new_candidates=0, calibration_samples=0,
            runner_sha256=digest(__file__), checkpoint_sha256=hashes,
            original_AGA_sha256=state_hashes["original"],
            released_AGA_sha256=state_hashes["released"],
            selection="One previously verified MNIST release; no DTD-specific choices",
            public_cleaning_run=False))
        prior = torch.load(source / "artifacts/predictions.pt", map_location="cpu")
        prior_rows = json.loads((source / "results.json").read_text())
        rows = [dict(r, reused=True) for r in prior_rows if r["stage"] in ["standalone", "AGA_standalone"]]
        raw = {"/".join(r[k] for k in ["preset", "stage", "dataset"]):
               prior["/".join(r[k] for k in ["preset", "stage", "dataset"])] for r in rows}
        def record(model, preset, stage, task):
            batches = []
            score = evaluate(model, heads[task], make_loader(datasets[task]), "cuda",
                             prediction_sink=lambda p,y:batches.append((p,y)))
            pred, y = [torch.cat([b[i] for b in batches]) for i in [0,1]]
            key = "/".join([preset,stage,task]);raw[key] = dict(prediction=pred,target=y)
            row = dict(preset=preset,stage=stage,dataset=task,reused=False,**score)
            if preset == "released":
                ref = raw["original/" + stage + "/" + task]
                row["paired"] = paired_counts(ref["prediction"].eq(y), pred.eq(y))
                row["changed_predictions"] = int(pred.ne(ref["prediction"]).sum())
            rows.append(row);save_json(out / "results.json",rows)
            torch.save(raw,out / "artifacts/predictions.pt")
            print("DTD_REPLAY_RESULT",json.dumps(row),flush=True)
        record(partner,"original","partner_standalone","DTD")
        author,pin = load_author(Path("external/secure-merge-attack/utils_attack_B32.py"),True)
        assert pin == json.loads(Path("configs/sources.json").read_text())["AGA"]["sha256"]
        models["attacked/partner"] = author_recover(partner,anchor,author)
        torch.save(models["attacked/partner"].state_dict(),out / "artifacts/partner_AGA.pt")
        for stage in ["direct","AGA"]:
            for name in ["original","released"]:
                model = merge(anchor,models[name if stage=="direct" else "attacked/"+name],
                              models["partner" if stage=="direct" else "attacked/partner"],cfg["coefficient"])
                for task in datasets:record(model,name,stage,task)
        save_json(out / "execution.json",dict(completed=True,rows=len(rows),new_candidates=0,
            release_sha256=meta["release_sha256"],predictions_sha256=digest(out / "artifacts/predictions.pt"),
            partner_AGA_sha256=digest(out / "artifacts/partner_AGA.pt"),
            protocol_sha256=digest(out / "protocol.json"),runner_sha256=digest(__file__),
            source_validation_sha256=digest(source / "independent_validation.json"),
            optimizer_steps=0,public_cleaning_run=False))
        return

    execution = json.loads((out / "execution.json").read_text())
    assert execution["completed"] and digest(__file__) == execution["runner_sha256"]
    assert digest(out / "artifacts/predictions.pt") == execution["predictions_sha256"]
    assert digest(out / "artifacts/partner_AGA.pt") == execution["partner_AGA_sha256"]
    model = copy.deepcopy(partner)
    model.load_state_dict(torch.load(out / "artifacts/partner_AGA.pt",map_location="cpu"))
    models["attacked/partner"] = model
    base = anchor.state_dict()
    for stage in ["direct","AGA"]:
        for name in ["original","released"]:
            left = models[name if stage=="direct" else "attacked/"+name].state_dict()
            right = models["partner" if stage=="direct" else "attacked/partner"].state_dict()
            state = {k:v+cfg["coefficient"]*((left[k]-v)+(right[k]-v))
                     if v.is_floating_point() else v.clone() for k,v in base.items()}
            model = copy.deepcopy(anchor);model.load_state_dict(state)
            models[stage+"/"+name] = model
    # Official list order and raw IDX labels are independent of evaluation counters.
    b=Path("data/MNIST/raw/t10k-labels-idx1-ubyte").read_bytes()
    labels_m=np.frombuffer(b[8:],dtype=np.uint8).astype("int64")
    folder=Path("data/dtd/dtd");classes=sorted(p.name for p in (folder/"images").iterdir() if p.is_dir())
    labels_d=np.array([classes.index(t.split('/')[0]) for t in (folder/"labels/test1.txt").read_text().splitlines()])
    raw=torch.load(out / "artifacts/predictions.pt",map_location="cpu")
    rows=json.loads((out / "results.json").read_text());checks=[]
    for r in rows:
        key="/".join(r[k] for k in ["preset","stage","dataset"])
        pred,y=[raw[key][k].numpy() for k in ["prediction","target"]]
        np.testing.assert_array_equal(y,labels_m if r['dataset']=='MNIST' else labels_d)
        assert int(np.sum(pred==y))==r['correct'] and len(y)==r['n']
        assert r['top1']==r['correct']/r['n']
        preset,stage,task=[r[k] for k in ['preset','stage','dataset']]
        name=('partner' if stage=='partner_standalone' else
              ('attacked/'+preset if stage=='AGA_standalone' else
               (stage+'/'+preset if stage in ['direct','AGA'] else preset)))
        model=models[name].cuda().eval();h=heads[task].cuda().eval()
        images=torch.stack([datasets[task][i][0] for i in range(128)]).cuda()
        features=model(images)
        if h.normalize:features=features/features.norm(dim=-1,keepdim=True)
        fresh=torch.nn.functional.linear(features,h.weight,h.bias).argmax(-1).cpu().numpy()
        np.testing.assert_array_equal(fresh,pred[:128])
        model.cpu();h.cpu();checks.append(dict(key=key,correct=r['correct'],n=r['n'],replay_n=128))
    save_json(out / "independent_validation.json",dict(status="passed",checks=checks,
        independent_merge="Explicit tensor formula",same_release=True,
        release_sha256=meta['release_sha256'],replayed_examples=128*len(rows),
        verifier_sha256=digest(__file__),new_candidates=0,public_cleaning_run=False))
    print("DTD_VALIDATION_PASSED",len(rows),flush=True)


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source",default="outputs/feature_noise")
    parser.add_argument("--output",required=True)
    parser.add_argument("--verify",action="store_true")
    args=parser.parse_args()
    torch.set_grad_enabled(False)
    run(args)
