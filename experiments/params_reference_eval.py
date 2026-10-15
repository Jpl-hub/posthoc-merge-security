"""Two-task official PaRaMS reference: fixed TA and both original AGA scopes."""

import argparse
import contextlib
import copy
import io
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import author_recover, checkpoint_loader, digest, evaluate, merge, paired_counts
from covariance_noise import dataset
from mechanisms import load_author
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels
from verify_probe_candidate import explicit_merge


def setup(out, cfg):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    assert "3090" in torch.cuda.get_device_name()
    for p, sha in cfg["evaluation_source_sha256"].items(): assert digest(p) == sha, p
    freeze = json.loads((out/"release_freeze.json").read_text())
    assert freeze["status"] == "frozen_before_any_data_or_peer"
    assert digest(out/"artifacts/PaRaMS.pt") == freeze["release_sha256"]
    root = Path("checkpoints/ViT-B-32"); load = checkpoint_loader(Path("external/secure-merge-attack"))
    pins = {x["path"].removeprefix("ViT-B-32/"):x["sha256"]
            for x in json.loads(Path("configs/checkpoints.json").read_text())["files"]}

    def checked(name):
        assert digest(root/name) == pins[name], name
        m = load(str(root/name)).cpu().eval(); m.requires_grad_(False); return m

    anchor, own = checked("zeroshot.pt"), checked("MNIST/finetuned.pt")
    released = copy.deepcopy(own); released.load_state_dict(torch.load(out/"artifacts/PaRaMS.pt", map_location="cpu"), strict=True)
    return anchor, own, released, checked, root


def fresh_split(cfg, out):
    excluded = set(); sources = {}
    for folder in Path("outputs").iterdir():
        if not folder.is_dir() or folder == out: continue
        case = folder/"case.json"; own = "mnist" in folder.name.lower()
        if case.exists(): own = json.loads(case.read_text()).get("target") == "MNIST"
        if not own: continue
        for name in ["training_indices.json", "holdout_protocol.json", "confirmation_split.json"]:
            p = folder/name
            if not p.exists(): continue
            data = json.loads(p.read_text())
            if name == "holdout_protocol.json" and data.get("split") != "train": continue
            groups = list(data.values()) if name == "training_indices.json" else [data["indices"]]
            for group in groups:
                if isinstance(group, list) and all(isinstance(i, int) for i in group): excluded.update(group)
            sources[str(p)] = digest(p)
    pool = sorted(set(range(60000))-excluded); assert len(pool) >= cfg["confirmation_n"]
    ids = np.random.default_rng(cfg["confirmation_seed"]).choice(pool, cfg["confirmation_n"], replace=False).tolist()
    assert not set(ids)&excluded
    save_json(out/"confirmation_split.json", dict(indices=ids, seed=cfg["confirmation_seed"],
              excluded_count=len(excluded), excluded_metadata_sha256=sources, split="train",
              used_for_optimization=False, scope="Disjoint registered JSON indices; no data used by PaRaMS construction. Not a proof of never appearing in all historical cache files."))
    return ids


@torch.no_grad()
def probes(anchor, own, released, cfg, out):
    start = time.monotonic(); a = own.model.visual.transformer.resblocks[-1].mlp.c_fc.weight
    a0 = anchor.model.visual.transformer.resblocks[-1].mlp.c_fc.weight
    delta = (a-a0).double().cuda(); u, s, vh = torch.linalg.svd(delta, full_matrices=False)
    rank = cfg["probe_rank"]; v = vh[:rank].float(); x = v.reshape(rank//4,4,768).contiguous()
    error = float((delta@vh[:rank].T-u[:,:rank]*s[:rank]).abs().max())
    records = []
    for i, (b, p) in enumerate(zip(own.model.visual.transformer.resblocks, released.model.visual.transformer.resblocks)):
        b.cuda(); p.cuda()
        mlp, mp = b.mlp(x), p.mlp(x)
        att, ap = b.attn(x,x,x,need_weights=False)[0], p.attn(x,x,x,need_weights=False)[0]
        records.append(dict(layer=i, MLP_relative=float((mlp-mp).double().norm()/mlp.double().norm()),
                            Attention_relative=float((att-ap).double().norm()/att.double().norm()),
                            MLP_max_abs=float((mlp-mp).abs().max()), Attention_max_abs=float((att-ap).abs().max())))
        b.cpu(); p.cpu()
    save_json(out/"svd_function_probes.json", dict(task_matrix_shape=list(delta.shape), right_probe_shape=list(v.shape),
              probe_location="768-dimensional module input; no flat task-vector SVD", task_SVD_identity_error=error,
              probe_rank=rank, records=records, selection_after_results=False, optimization_steps=0,
              interpretation="Algebraic module-equivalence probes, not a task-importance or whole-input distribution certificate.",
              wall_seconds=time.monotonic()-start))


@torch.no_grad()
def run(protocol, output):
    cfg = json.loads(Path(protocol).read_text()); out = Path(output)
    assert not (out/"results.json").exists()
    save_json(out/"evaluation_protocol.json", cfg)
    start = time.monotonic(); anchor, own, released, checked, root = setup(out,cfg)
    # Release hash fixed before probes, training data, test data, or peer loading.
    probes(anchor,own,released,cfg,out)
    train = dataset("MNIST", "data", own.val_preprocess, True); ids = fresh_split(cfg,out)
    heads = {t:checked(f"head_{t}.pt") for t in ["MNIST","SVHN"]}
    rows, raw = [], {}

    def record(model, task, preset, stage, data, indices=None):
        sink, logits = [], []
        m = evaluate(model, heads[task], make_loader(data,indices), "cuda",
                     prediction_sink=lambda p,y:sink.append((p,y)), logit_sink=logits.append)
        pred,y = [torch.cat([b[i] for b in sink]) for i in [0,1]]
        scores = torch.cat(logits); key=f"{preset}/{stage}/{task}"
        raw[key] = dict(prediction=pred,target=y,logits=scores)
        row = dict(preset=preset,stage=stage,dataset=task,**m,
                   per_class=[dict(class_id=c,n=int(y.eq(c).sum()),correct=int((pred.eq(y)&y.eq(c)).sum())) for c in range(10)])
        if preset == "released":
            ref=raw[f"original/{stage}/{task}"]; assert torch.equal(y,ref["target"])
            row.update(paired=paired_counts(ref["prediction"].eq(y),pred.eq(y)), changed_predictions=int(pred.ne(ref["prediction"]).sum()))
            if stage == "confirmation":
                teacher=ref["logits"].double(); after=scores.double(); yy=teacher.argmax(1,keepdim=True)
                gaps=teacher.gather(1,yy)-teacher; valid=gaps>0; ds=after-teacher
                values=(ds.gather(1,yy)-ds)[valid]/gaps[valid]
                row["worst_relative_task_margin_drift"]=float(values.abs().max())
        rows.append(row); save_json(out/"results.json",rows)
        torch.save(raw,out/"artifacts/predictions.pt")
        print("PARAMS_RESULT",json.dumps(row),flush=True)

    for name,model in [("original",own),("released",released)]: record(model,"MNIST",name,"confirmation",train,ids)
    # Original method is a declared reference even when outside our parameter budget.
    peer=checked("SVHN/finetuned.pt"); data={t:dataset(t,"data",own.val_preprocess) for t in heads}
    for name,model in [("original",own),("released",released)]: record(model,"MNIST",name,"standalone",data["MNIST"])
    record(peer,"SVHN","original","partner_standalone",data["SVHN"])
    for name,model in [("original",own),("released",released)]:
        fused=merge(anchor,model,peer,.5)
        for t in heads: record(fused,t,name,"direct",data[t])
        del fused
    author,pin=load_author(Path("external/secure-merge-attack/utils_attack_B32.py"),True)
    assert pin==cfg["AGA_text_sha256"]
    refs={}
    for task,model in [("MNIST",own),("SVHN",peer)]:
        ref=cfg["AGA_references"][task]; assert digest(ref["path"])==ref["sha256"]
        refs[task]=copy.deepcopy(model);refs[task].load_state_dict(torch.load(ref["path"],map_location="cpu"),strict=True)
    log=io.StringIO()
    with contextlib.redirect_stdout(log): protected_a=author_recover(released,anchor,author)
    text=log.getvalue();(out/"AGA.log").write_text(text)
    assert not any(w in text for w in ["Skipping","Warning:","Error"])
    assert all(f"Attacking Layer {i}..." in text for i in range(12))
    assert ">>> MLP-only attack completed." in text
    diff={k:v.clone() for k,v in protected_a.state_dict().items() if not torch.equal(v,refs["MNIST"].state_dict()[k])}
    torch.save(diff,out/"artifacts/AGA_release_edits.pt")
    for name,model in [("original",refs["MNIST"]),("released",protected_a)]:
        record(model,"MNIST",name,"AGA_standalone",data["MNIST"])
        for scope,other in [("AGA_target_only",peer),("AGA_both",refs["SVHN"])]:
            fused=merge(anchor,model,other,.5)
            for t in heads: record(fused,t,name,scope,data[t])
            del fused
    aligned=[]
    for k,v in refs["MNIST"].state_dict().items():
        if v.ndim==2 and v.is_floating_point():aligned.append(float(((protected_a.state_dict()[k].double()-v.double()).norm()/own.state_dict()[k].double().norm())**2))
    save_json(out/"execution.json",dict(completed=True,new_defense_methods=0,original_expert_updated=False,
              source_sha256=cfg["evaluation_source_sha256"],evaluation_wall_seconds=time.monotonic()-start,
              GPU=torch.cuda.get_device_name(),evaluation_optimizer_steps=0,
              reference_outside_budget=json.loads((out/"release_freeze.json").read_text())["is_different_budget_reference"],
              AGA_aligned_matrix_distance=sum(aligned)**.5,
              predictions_sha256=digest(out/"artifacts/predictions.pt"),AGA_edits_sha256=digest(out/"artifacts/AGA_release_edits.pt"),
              release_sha256=digest(out/"artifacts/PaRaMS.pt"),no_public_parameter_replacement_run=True))


if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--protocol",default="configs/params_reference_eval.json")
    p.add_argument("--output",required=True);args=p.parse_args();run(args.protocol,args.output)
