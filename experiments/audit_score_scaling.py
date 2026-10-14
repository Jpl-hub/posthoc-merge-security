"""NumPy score-order audit of a frozen displacement, using training caches only."""

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


@torch.no_grad()
def audit(root, target):
    root = Path(root); geometry = json.loads((root/(target+".json")).read_text())
    path = root/(target+"_diagnostic.pt"); assert digest(path) == geometry["diagnostic_sha256"]
    cal = torch.load(path, map_location="cpu")
    anchor, expert, head, _ = checked_models(target); p = parameters(expert, head)
    assert all(torch.equal(p[k], cal["parameter_reference"][k]) for k in ["A", "B"])
    ea = cal["alpha"]*float(p["A"].norm())*array(cal["ray"])
    eb = array(cal["coefficient_B"])@array(cal["basis"]["B"]["vh"])
    beta = cal["protocol"]["proxy_beta"]; proxy = merge(anchor, expert, anchor, beta)
    records = {}
    for name, model, cache, factor in [("native", expert, cal["native"], 1.),
                                       ("proxy", proxy, cal["proxy"], beta)]:
        mlp = model.model.visual.transformer.resblocks[11].mlp
        a, b, bias = array(mlp.c_fc.weight), array(mlp.c_proj.weight), array(mlp.c_fc.bias)
        x, h = array(cache["x"]), array(cache["h"])
        pre = x@a.T+bias; z = pre*expit(1.702*pre)
        altered = x@(a+factor*ea).T+bias; zn = altered*expit(1.702*altered)
        dh = zn@(b+factor*eb).T-z@b.T
        s0 = features(h, model.model.visual)@array(head.weight).T
        s1 = features(h+dh, model.model.visual)@array(head.weight).T
        c0, c1 = s0-s0.mean(1, keepdims=True), s1-s1.mean(1, keepdims=True)
        scale = np.sum(c0*c1, 1)/np.maximum(np.sum(c0*c0, 1), 1e-20)
        error = c1-scale[:, None]*c0
        winner, changed = s0.argmax(1), s1.argmax(1)!=s0.argmax(1)
        def gap(scores):
            other = scores.copy(); other[np.arange(len(scores)), winner] = -np.inf
            return scores[np.arange(len(scores)), winner]-other.max(1)
        g0, g1 = gap(s0), gap(s1)
        certificate = (scale>0)&(2*np.max(np.abs(error), axis=1)<scale*g0)
        unit0 = c0/np.maximum(np.linalg.norm(c0, axis=1, keepdims=True), 1e-20)
        unit1 = c1/np.maximum(np.linalg.norm(c1, axis=1, keepdims=True), 1e-20)
        records[name] = {"n": len(s0), "changed_predictions": int(changed.sum()),
                         "scale_median": float(np.median(scale)), "positive_scale_count": int((scale>0).sum()),
                         "non_scaling_error_rms": float(np.sqrt(np.mean(np.sum(error**2, 1)))),
                         "positive_affine_certificate_count": int(certificate.sum()),
                         "teacher_gap_median": float(np.median(g0)), "altered_teacher_gap_median": float(np.median(g1)),
                         "score_direction_change_rms": float(np.sqrt(np.mean(np.sum((unit1-unit0)**2, 1)))),
                         "class_counts": {str(k): {"teacher_count": int((winner==k).sum()),
                             "changed": int(((winner==k)&changed).sum())} for k in range(s0.shape[1])}}
    report = {"target": target, "source_sha256": digest(__file__),
              "diagnostic_sha256": geometry["diagnostic_sha256"], "training_cache_only": True,
              "new_weights": 0, "optimizer_updates": 0, "records": records}
    save_json(root/(target+"_score_audit.json"), report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostic-root", required=True); parser.add_argument("--target", required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    audit(args.diagnostic_root, args.target)
