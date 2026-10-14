"""NumPy score-direction and decision-boundary audit of a frozen publication point."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest, merge
from subspace_probe import save_json
from task_probe_distance import checked_models, parameters
from verify_probe_candidate import array, features


def audit(output):
    output=Path(output); result=json.loads((output/"results.json").read_text())
    path=output/"candidate.pt"; assert digest(path)==result["candidate_sha256"]
    cal=torch.load(path,map_location="cpu"); cfg=cal["protocol"]
    anchor,expert,head,hashes=checked_models(result["target"])
    assert hashes==result["checkpoint_sha256"]
    own=parameters(expert,head); proxy=merge(anchor,expert,anchor,cfg["proxy_beta"])
    pp=parameters(proxy,head); x,h=[array(cal["proxy"][k]) for k in ["x","h"]]
    a,b,bias=[array(pp[k]) for k in ["A","B","a"]]
    pre=x@a.T+bias; z=pre*expit(1.702*pre)
    weights=array(head.weight)
    baseline=features(h,proxy.model.visual)@weights.T
    winner=baseline.argmax(1); rows=np.arange(len(winner)); k=baseline.shape[1]
    def directions(s):
        centered=s-s.mean(1,keepdims=True)
        return centered/np.linalg.norm(centered,axis=1,keepdims=True)
    u0=directions(baseline)
    def gap(s):
        competitors=s.copy(); competitors[rows,winner]=-np.inf
        return s[rows,winner]-competitors.max(1)
    g0=gap(u0); unit=np.partition(g0,(len(g0)-1)//2)[(len(g0)-1)//2]
    counts=np.bincount(winner,minlength=k); w=1./counts[winner]/int((counts>0).sum())
    def forward(ea,eb):
        beta=cfg["proxy_beta"]; pre1=x@(a+beta*ea).T+bias; z1=pre1*expit(1.702*pre1)
        return features(h+z1@(b+beta*eb).T-z@b.T,proxy.model.visual)@weights.T
    # Reconstruct the exact training-only initialization used by the frozen optimizer.
    if result["method"]=="conditional_composite_distillation_primal_dual":
        fraction=max(t["seed_fraction"] for t in result["initialization_trials"] if t["feasible"])
        ea0=cal["alpha"]*fraction*np.linalg.norm(array(own["A"]))*array(cal["ray"])
        xx=np.concatenate([array(cal[g]["x"]) for g in ["native","scale"]])
        aa,bb,abias=[array(own[s]) for s in ["A","B","a"]]
        pre0=xx@aa.T+abias; zz0=pre0*expit(1.702*pre0)
        pre1=xx@(aa+ea0).T+abias; zz1=pre1*expit(1.702*pre1)
        vb=array(cal["basis"]["B"]["vh"]); q0=zz0@vb.T; q1=zz1@vb.T
        ridge=cfg["ridge_relative_floor"]*np.linalg.eigvalsh(q0.T@q0)[-1]
        eb0=np.linalg.solve(q1.T@q1+ridge*np.eye(cfg["probe_rank"]),q1.T@(zz0@bb.T-zz1@bb.T)).T@vb
    else:
        ea0=cal["alpha"]*np.linalg.norm(array(own["A"]))*array(cal["ray"])
        eb0=array(cal["coefficient_B"])@array(cal["basis"]["B"]["vh"])
    initial=forward(ea0,eb0)
    final=forward(array(cal["release_deltas"]["A"]),array(cal["release_deltas"]["B"]))
    summaries={}
    for name,s,expected in [("initial",initial,result["initial"]),("selected",final,result["selected"])]:
        u=directions(s); g=gap(u); delta=np.linalg.norm(u-u0,axis=1)
        changed=s.argmax(1)!=winner
        angular_stress=float(-np.sum(w*g)/unit)
        assert abs(angular_stress-expected["angular_stress"])<1e-7
        assert int(changed.sum())==expected["proxy_changed"]
        # For all competitors j, |(u'_p-u'_j)-(u_p-u_j)| <= sqrt(2)||u'-u||.
        certificate=g0>np.sqrt(2)*delta
        assert not np.any(certificate & changed)
        centered=s-s.mean(1,keepdims=True); c0=baseline-baseline.mean(1,keepdims=True)
        scale=np.sum(centered*c0,axis=1)/np.sum(c0*c0,axis=1)
        affine_residual=centered-scale[:,None]*c0
        summaries[name]={"angular_stress":angular_stress,"prediction_changes":int(changed.sum()),
                         "changed_by_teacher_class":np.bincount(winner[changed],minlength=k).tolist(),
                         "cosine_to_teacher_median":float(np.median(np.sum(u*u0,axis=1))),
                         "positive_affine_fit_scale_median":float(np.median(scale)),
                         "negative_affine_fit_scales":int((scale<=0).sum()),
                         "non_affine_score_rms":float(np.sqrt(np.mean(affine_residual**2))),
                         "score_direction_change_p10_median_p90":np.quantile(delta,[.1,.5,.9]).tolist(),
                         "relative_angular_gap_loss_p10_median_p90":np.quantile((g0-g)/g0,[.1,.5,.9]).tolist(),
                         "remaining_positive_teacher_gaps":int((g>0).sum()),
                         "certified_same_prediction":int(certificate.sum()),"n":len(winner)}
    raw=output/"score_direction_audit.pt"
    torch.save({"teacher_scores":torch.from_numpy(baseline),"initial_scores":torch.from_numpy(initial),
                "selected_scores":torch.from_numpy(final),"teacher_predictions":torch.from_numpy(winner)},raw)
    report={"target":result["target"],"source_sha256":digest(__file__),"candidate_sha256":digest(path),
            "raw_sha256":digest(raw),"class_balance_counts":counts.tolist(),"results":summaries,
            "only_cached_target_train_and_public_anchor":True,"fresh_model_forwards":0,
            "test_used":False,"future_partner_used":False,"new_release_weights":0,
            "certificate":"teacher angular gap > sqrt(2)*score direction change; sufficient per-sample condition"}
    save_json(output/"score_direction_audit.json",report); print(json.dumps(report),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",required=True)
    args=parser.parse_args(); torch.set_num_threads(4); audit(args.output)
