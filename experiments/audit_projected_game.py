"""Audit local rejection causes; do not optimize or create release weights."""

import argparse
import json
from pathlib import Path

import torch

from clip_benchmark import digest, merge
from probe_distance_candidate import on_device
from probe_margin_distance import margin_drift
from projected_distance_game import pack, predict
from subspace_probe import save_json
from task_probe_distance import checked_models, parameters


def audit(target, cfg):
    output = Path(cfg["diagnostic_output"])
    meta = json.loads((output/(target+".json")).read_text())
    path = output/(target+"_diagnostic.pt")
    assert digest(path) == meta["diagnostic_sha256"]
    cal = torch.load(path, map_location="cpu")
    anchor, expert, head, _ = checked_models(target)
    p = on_device(parameters(expert, head))
    basis = {s: on_device(b) for s, b in cal["basis"].items()}
    ray = cal["ray"].cuda(); na, nb = p["A"].norm(), p["B"].norm()
    native = [pack(on_device(cal[k]), p, basis, ray, na, cfg, True) for k in ["native", "scale"]]
    pp = on_device(parameters(merge(anchor, expert, anchor, cfg["proxy_beta"]), head))
    proxy = pack(on_device(cal["proxy"]), pp, basis, ray, na, cfg, False)
    teacher = proxy["teacher"]; winner = teacher.argmax(1)
    wrong = teacher.clone(); wrong[torch.arange(len(winner), device="cuda"), winner] = -torch.inf
    unit = (teacher.gather(1, winner[:, None]).flatten()-wrong.max(1).values).median()
    counts = torch.bincount(winner, minlength=teacher.shape[1])
    weights = 1./counts[winner].double()/int((counts>0).sum())
    t = torch.tensor(cal["alpha"], dtype=torch.float64, device="cuda", requires_grad=True)
    c = cal["optimized_coefficient"].cuda().clone().requires_grad_(True)
    coords = torch.cat([t.detach().reshape(1), c.detach().flatten()])
    radius, limit = cfg["parameter_budget"], cfg["margin_fraction"]
    floor = meta["parameter_distance_floor"]**2

    def evaluate(tt, cc):
        drifts = torch.cat([margin_drift(n["teacher"], predict(n, p, basis, nb, tt, cc, 1.)) for n in native])
        s = predict(proxy, pp, basis, nb, tt, cc, cfg["proxy_beta"])
        other = s.clone(); other[torch.arange(len(winner), device="cuda"), winner] = -torch.inf
        stress = (weights*(other.max(1).values-s.gather(1, winner[:, None]).flatten())).sum()/unit
        distance = tt.square()+cc.square().sum()
        return distance/radius**2+stress, drifts, distance

    objective, drifts, distance = evaluate(t, c)
    def gradient(value):
        gt, gc = torch.autograd.grad(value, [t, c], retain_graph=True)
        return torch.cat([gt.reshape(1), gc.flatten()])
    raw = gradient(objective)
    single = raw.clone()
    if float(distance) >= .99*radius**2:
        single -= (single@coords).clamp_min(0)/coords.square().sum()*coords
    if float(distance) <= 1.01*floor:
        single -= (single@coords).clamp_max(0)/coords.square().sum()*coords
    normal = gradient(drifts.max())
    single -= (single@normal).clamp_min(0)/normal.square().sum().clamp_min(1e-20)*normal
    # A conservative equality-tangent diagnostic, capped at32 active witnesses.
    # This does not assert that the full nonlinear feasible cone is represented.
    active_ids = (drifts >= cfg["active_fraction"]*limit).nonzero().flatten()
    selected = active_ids[torch.argsort(drifts[active_ids], descending=True)[:32]]
    normals = torch.stack([coords/coords.norm()]+[gradient(drifts[j]) for j in selected])
    normals /= normals.norm(dim=1, keepdim=True).clamp_min(1e-20)
    gram = normals@normals.T
    joint = raw-normals.T@(torch.linalg.pinv(gram, rtol=1e-10)@(normals@raw))
    trials = {}
    with torch.no_grad():
        for name, g in [("single_witness", single), ("joint_tangent_diagnostic", joint)]:
            g = g/g.norm().clamp_min(1e-20); rows = []
            for k in range(cfg["backtracking_steps"]):
                trial = coords+cfg["normalized_step"]*.5**k*g
                trial *= min(1., radius/float(trial.norm()))
                obj, dd, dist = evaluate(trial[0], trial[1:].reshape_as(c))
                rows.append({"backtrack": k, "critic": float(dd.max()), "distance": float(dist.sqrt()),
                             "objective": float(obj), "native_ok": float(dd.max()) <= limit,
                             "distance_floor_ok": float(dist) >= floor*(1-1e-8),
                             "ascent_ok": float(obj) >= float(objective)})
            trials[name] = rows
    report = {"target": target, "new_release_weights": 0, "optimized_updates": 0,
              "diagnostic_sha256": meta["diagnostic_sha256"], "source_sha256": digest(__file__),
              "active_instances": len(active_ids), "selected_active_witnesses": len(selected),
              "joint_gradient_norm_fraction": float(joint.norm()/raw.norm()),
              "single_gradient_radial_inner_product": float(single@coords),
              "joint_gradient_radial_inner_product": float(joint@coords),
              "joint_tangent_max_error": float((normals@joint).abs().max()),
              "full_feasible_cone_claim": False, "trials": trials}
    save_json(output/(target+"_local_audit.json"), report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True); parser.add_argument("--target", required=True)
    args = parser.parse_args(); cfg = json.loads(Path(args.protocol).read_text())
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    audit(args.target, cfg)
