"""Independent algebra, official labels and explicit-fusion replays for PaRaMS."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import digest
from covariance_noise import dataset
from params_reference_eval import setup
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_probe_candidate import explicit_merge


def arr(t): return t.detach().cpu().numpy()


@torch.no_grad()
def verify(output):
    start=time.monotonic();out=Path(output);cfg=json.loads((out/"evaluation_protocol.json").read_text())
    meta=json.loads((out/"execution.json").read_text());freeze=json.loads((out/"release_freeze.json").read_text())
    assert meta["completed"] and meta["evaluation_optimizer_steps"]==0
    assert digest(out/"artifacts/predictions.pt")==meta["predictions_sha256"]
    assert digest(out/"artifacts/AGA_release_edits.pt")==meta["AGA_edits_sha256"]
    anchor,own,protected,checked,root=setup(out,cfg)
    peer=checked("SVHN/finetuned.pt");heads={t:checked(f"head_{t}.pt") for t in ["MNIST","SVHN"]}
    own_a=copy.deepcopy(own);peer_a=copy.deepcopy(peer)
    for task,model in [("MNIST",own_a),("SVHN",peer_a)]:
        ref=cfg["AGA_references"][task];assert digest(ref["path"])==ref["sha256"]
        model.load_state_dict(torch.load(ref["path"],map_location="cpu"),strict=True)
    protected_a=copy.deepcopy(own_a)
    for k,v in torch.load(out/"artifacts/AGA_release_edits.pt",map_location="cpu").items():protected_a.state_dict()[k].copy_(v)
    # Recover exact row permutations against the original expert, independently
    # of the author's base-anchor Hungarian protection and AGA solvers.
    perms=[];generator=torch.Generator().manual_seed(0);scaling_error=0.;matrix_squares=[];mlp_squares=[]
    old,new=own.state_dict(),protected.state_dict()
    for i,(b,p) in enumerate(zip(own.model.visual.transformer.resblocks,protected.model.visual.transformer.resblocks)):
        x=b.mlp.c_fc.weight.cuda();z=p.mlp.c_fc.weight.cuda()
        sim=torch.nn.functional.normalize(z,dim=1)@torch.nn.functional.normalize(x,dim=1).T
        perm=sim.argmax(1).cpu().numpy();assert len(np.unique(perm))==len(perm)
        for name,expected in [("c_fc.weight",arr(b.mlp.c_fc.weight)[perm]),("c_fc.bias",arr(b.mlp.c_fc.bias)[perm]),("c_proj.weight",arr(b.mlp.c_proj.weight)[:,perm])]:
            np.testing.assert_array_equal(arr(p.mlp.state_dict()[name]),expected)
        low=cfg["effective_defaults"]["scale_min"];high=cfg["effective_defaults"]["scale_max"]
        alpha=arr(low+(high-low)*torch.rand(768,generator=generator));inverse=np.float32(1.)/alpha
        w=arr(b.attn.in_proj_weight);bias=arr(b.attn.in_proj_bias)
        expected=np.concatenate([w[:768]*alpha[:,None],w[768:1536]*inverse[:,None],w[1536:]*alpha[:,None]])
        expected_bias=np.concatenate([bias[:768]*alpha,bias[768:1536]*inverse,bias[1536:]*alpha])
        np.testing.assert_array_equal(arr(p.attn.in_proj_weight),expected)
        np.testing.assert_array_equal(arr(p.attn.in_proj_bias),expected_bias)
        np.testing.assert_array_equal(arr(p.attn.out_proj.weight),arr(b.attn.out_proj.weight)*inverse[None,:])
        perms.append(dict(layer=i,permutation_is_bijection=True,permutation_nonidentity=int((perm!=np.arange(len(perm))).sum()),min_scale=float(alpha.min()),max_scale=float(alpha.max())))
    for k,w in old.items():
        if w.ndim==2 and w.is_floating_point():
            delta=arr(new[k]).astype(np.float64)-arr(w).astype(np.float64)
            value=float(np.linalg.norm(delta)**2/np.linalg.norm(arr(w).astype(np.float64))**2)
            matrix_squares.append(value)
            if ".mlp." in k:mlp_squares.append(value)
    matrix_distance=sum(matrix_squares)**.5;mlp_lower=sum(mlp_squares)**.5
    assert abs(matrix_distance-freeze["parameter_distance_matrices"])<1e-9
    assert abs(mlp_lower-freeze["MLP_matrices_distance_lower_bound"])<1e-9
    if cfg.get("reference_variant")=="scaling_only":
        assert mlp_lower==0 and freeze["parameter_distance_all_changed_tensors"]<=.5
    else:assert mlp_lower>.5
    # Frozen O/B source-only control. No old confirmation values or peer output
    # is used to construct or select any defense in this comparison.
    ob_path=Path("outputs/boundary_mnist/artifacts/release.pt")
    assert digest(ob_path)=="953007daee0d6ba6f4c40de8aa738cfb5052db65270436aa4f0fe5e04aefaa46"
    ob=torch.load(ob_path,map_location="cpu");stem="model.visual.transformer.resblocks.11.attn.out_proj.weight"
    base=arr(anchor.state_dict()[stem]).astype(np.float64);reference=arr(old[stem]).astype(np.float64)
    controls={"PaRaMS":arr(new[stem]).astype(np.float64),"O_B":arr(ob[stem]).astype(np.float64)}
    geometry={}
    for name,weight in controls.items():
        records=[]
        for h in range(12):
            sl=slice(h*64,(h+1)*64);x=reference[:,sl];xp=weight[:,sl];x0=base[:,sl]
            q=np.linalg.qr(x)[0];qp=np.linalg.qr(xp)[0]
            projector_difference=np.sqrt(max(0.,2*(64-np.linalg.norm(q.T@qp)**2)))
            canonical=q@(q.T@x0);cp=qp@(qp.T@x0)
            records.append(dict(head=h,column_projector_frobenius=projector_difference,
                                unregularized_AGA_difference_relative=float(np.linalg.norm(cp-canonical)/np.linalg.norm(canonical))))
        geometry[name]=records
    data={t:dataset(t,"data",own.val_preprocess) for t in heads};train=dataset("MNIST","data",own.val_preprocess,True)
    split=json.loads((out/"confirmation_split.json").read_text());ids=split["indices"]
    raw=torch.load(out/"artifacts/predictions.pt",map_location="cpu");rows=json.loads((out/"results.json").read_text());replays=0
    for row in rows:
        preset,stage,t=[row[k] for k in ["preset","stage","dataset"]]
        record=raw[f"{preset}/{stage}/{t}"];y=official_labels(t,stage=="confirmation")
        if stage=="confirmation":y=y[ids]
        pred=arr(record["prediction"]);logits=arr(record["logits"])
        np.testing.assert_array_equal(arr(record["target"]),y);np.testing.assert_array_equal(logits.argmax(1),pred)
        assert np.isfinite(logits).all() and row["n"]==len(y) and row["correct"]==int((pred==y).sum())
        for c in row["per_class"]:
            mask=y==c["class_id"];assert c["n"]==int(mask.sum()) and c["correct"]==int(((pred==y)&mask).sum())
        if preset=="released":
            r=raw[f"original/{stage}/{t}"];bp=arr(r["prediction"])
            assert row["changed_predictions"]==int((pred!=bp).sum())
            assert row["paired"]["lost"]==int(((bp==y)&(pred!=y)).sum())
            assert row["paired"]["gained"]==int(((bp!=y)&(pred==y)).sum())
            if stage=="confirmation":
                before=arr(r["logits"]).astype(np.float64);after=logits.astype(np.float64);yy=before.argmax(1)
                gap=before[np.arange(len(y)),yy,None]-before;delta=after-before
                ds=delta[np.arange(len(y)),yy,None]-delta;valid=gap>0
                rho=float(np.abs(ds[valid]/gap[valid]).max())
                assert abs(rho-row["worst_relative_task_margin_drift"])<1e-10
        if stage=="partner_standalone":model=peer
        elif stage in ["confirmation","standalone"]:model=own if preset=="original" else protected
        elif stage=="AGA_standalone":model=own_a if preset=="original" else protected_a
        else:
            assert stage in ["direct","AGA_target_only","AGA_both"]
            left=(own if preset=="original" else protected) if stage=="direct" else (own_a if preset=="original" else protected_a)
            right=peer_a if stage=="AGA_both" else peer;model=explicit_merge(anchor,left,right,.5)
        chosen=np.linspace(0,len(y)-1,16,dtype=int);source=train if stage=="confirmation" else data[t]
        samples=[ids[int(i)] for i in chosen] if stage=="confirmation" else chosen
        images=torch.stack([source[int(i)][0] for i in samples]).cuda();model.cuda();heads[t].cuda()
        actual=arr(heads[t](model(images)).argmax(1));model.cpu();heads[t].cpu()
        np.testing.assert_array_equal(actual,pred[chosen]);replays+=len(chosen)
    aligned=[]
    for k,w in own_a.state_dict().items():
        if w.ndim==2 and w.is_floating_point():aligned.append(float(np.linalg.norm(arr(protected_a.state_dict()[k]).astype(np.float64)-arr(w).astype(np.float64))**2/np.linalg.norm(arr(old[k]).astype(np.float64))**2))
    assert abs(sum(aligned)**.5-meta["AGA_aligned_matrix_distance"])<1e-9
    save_json(out/"independent_validation.json",dict(status="passed",rows=len(rows),official_labels_and_per_class=True,
              explicit_formula_prediction_replays=replays,independent_numpy_margin_and_distance=True,
              exact_MLP_permutation_and_attention_scaling=True,parameter_distance_matrices=matrix_distance,
              MLP_matrices_distance_lower_bound=mlp_lower,AGA_aligned_matrix_distance=sum(aligned)**.5,
              source_only_geometry=geometry,per_layer_checks=perms,release_sha256=meta["release_sha256"],
              predictions_sha256=meta["predictions_sha256"],source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
              same_budget_PaRaMS_claim=False,new_defense_methods=0))
    print("PARAMS_INDEPENDENT_VALIDATION_PASSED",len(rows),replays,flush=True)


def aligned_control(output):
    """Recount frozen O/B's surviving matrix distance without images or peers."""
    out=Path(output);freeze=json.loads((out/"release_freeze.json").read_text())
    reference=json.loads((out/"evaluation_protocol.json").read_text())["AGA_references"]["MNIST"]
    edits_meta=json.loads(Path("results/boundary_eval/AGA_edits.json").read_text())
    path=Path("outputs/boundary_mnist/artifacts/released_AGA_edits.pt")
    assert digest(path)==edits_meta["sha256"] and digest(reference["path"])==reference["sha256"]
    a=torch.load(reference["path"],map_location="cpu");edits=torch.load(path,map_location="cpu")
    norms={r["key"]:r["original_norm"] for r in freeze["edited_tensors"]}
    rows=[]
    for k,w in edits.items():
        assert w.ndim==2 and k in norms
        relative=float(np.linalg.norm(arr(w).astype(np.float64)-arr(a[k]).astype(np.float64))/norms[k])
        rows.append(dict(key=k,relative_to_original_expert=relative))
    save_json(out/"frozen_OB_aligned_distance.json",dict(status="passed",matrix_distance=sum(r["relative_to_original_expert"]**2 for r in rows)**.5,
              records=rows,AGA_reference_sha256=reference["sha256"],OB_AGA_edits_sha256=digest(path),
              denominator_source_sha256=digest(out/"release_freeze.json"),source_sha256=digest(__file__),
              images_or_peers_loaded=False,new_publication_parameters=False))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--output",required=True)
    p.add_argument("--aligned-control-only",action="store_true");args=p.parse_args()
    aligned_control(Path(args.output)) if args.aligned_control_only else verify(Path(args.output))
