"""Projected updates of publication variables, with a worst-task-probe critic."""

import argparse
import json
from pathlib import Path

import torch

from clip_benchmark import digest, merge
from probe_distance_candidate import on_device
from probe_margin_distance import margin_drift, scores, task_probes
from subspace_probe import save_json
from task_probe_distance import checked_models, gelu, parameters


@torch.no_grad()
def pack(cache, params, basis, ray, own_a_norm, cfg, with_probes):
    x, h = cache["x"], cache["h"]
    z = gelu(x@params["A"].T+params["a"])
    inputs, bases, offsets, teacher = [x], [h-z@params["B"].T], [torch.zeros_like(z)], [scores(h, params)]
    if with_probes:
        probe = task_probes(cache, params, basis, cfg)
        n, count = probe["teacher_a"].shape[:2]
        base = probe["base"][:, None, :].expand(-1, count, -1).reshape(n*count, -1)
        inputs += [probe["xa"], probe["x"][:, None, :].expand(-1, count, -1).reshape(n*count, -1)]
        bases += [base, base]
        offsets += [torch.zeros(n*count, z.shape[1], dtype=x.dtype, device=x.device),
                    probe["shifts_b"][None, :, :].expand(n, -1, -1).reshape(n*count, -1)]
        teacher += [probe["teacher_a"].flatten(0, 1), probe["teacher_b"].flatten(0, 1)]
    x = torch.cat(inputs)
    return {"pre": x@params["A"].T+params["a"], "ray_pre": own_a_norm*(x@ray.T),
            "base": torch.cat(bases), "offset": torch.cat(offsets), "teacher": torch.cat(teacher)}


def predict(packed, params, basis, own_b_norm, t, c, factor):
    z = gelu(packed["pre"]+factor*t*packed["ray_pre"])+packed["offset"]
    h = packed["base"]+z@params["B"].T+factor*own_b_norm*((z@basis["B"]["vh"].T)@c.T)
    return scores(h, params)


def diagnose(target, cfg):
    root = Path(cfg["margin_reference"]); source = root/(target+"_diagnostic.pt")
    geometry = json.loads((root/(target+".json")).read_text())
    assert digest(source) == geometry["diagnostic_sha256"]
    assert json.loads((root/(target+"_probe_validation.json")).read_text())["status"] == "passed"
    cal = torch.load(source, map_location="cpu")
    anchor, expert, head, hashes = checked_models(target)
    p_cpu = parameters(expert, head); p = on_device(p_cpu)
    assert all(torch.equal(p_cpu[k], cal["parameter_reference"][k]) for k in ["A", "B"])
    basis = {s: on_device(b) for s, b in cal["basis"].items()}; ray = cal["ray"].cuda()
    na, nb = p["A"].norm(), p["B"].norm()
    native = [pack(on_device(cal[k]), p, basis, ray, na, cfg, True) for k in ["native", "scale"]]
    proxy_model = merge(anchor, expert, anchor, cfg["proxy_beta"])
    pp = on_device(parameters(proxy_model, head)); del proxy_model
    proxy = pack(on_device(cal["proxy"]), pp, basis, ray, na, cfg, False)
    teacher = proxy["teacher"]; winner = teacher.argmax(1)
    wrong = teacher.clone(); wrong[torch.arange(len(winner), device="cuda"), winner] = -torch.inf
    gap = teacher.gather(1, winner[:, None]).flatten()-wrong.max(1).values
    unit = gap.median().detach(); counts = torch.bincount(winner, minlength=teacher.shape[1])
    weights = 1./counts[winner].double()/int((counts>0).sum())
    t = torch.tensor(cal["alpha"], dtype=torch.float64, device="cuda", requires_grad=True)
    c = (cal["coefficient_B"].cuda()/nb).clone().requires_grad_(True)
    radius, limit = cfg["parameter_budget"], cfg["margin_fraction"]

    def evaluate(tt, cc):
        candidate = [predict(n, p, basis, nb, tt, cc, 1.) for n in native]
        critic = torch.stack([margin_drift(n["teacher"], s).max() for n, s in zip(native, candidate)]).max()
        sp = predict(proxy, pp, basis, nb, tt, cc, cfg["proxy_beta"])
        competitors = sp.clone(); competitors[torch.arange(len(winner), device="cuda"), winner] = -torch.inf
        # This ordering term is invariant to a common score offset.
        signed_gap = competitors.max(1).values-sp.gather(1, winner[:, None]).flatten()
        stress = (weights*signed_gap).sum()/unit
        distance = tt.square()+cc.square().sum()
        objective = distance/radius**2+stress
        drifts = torch.cat([margin_drift(n["teacher"], s) for n, s in zip(native, candidate)])
        return objective, critic, distance, stress, int((sp.argmax(1)!=winner).sum()), drifts

    initial = evaluate(t, c); distance_floor = float(initial[2])
    best = (float(initial[0]), t.detach().clone(), c.detach().clone())
    history = []; accepted = 0; corrective_gradients = 0

    def retract(trial):
        """Local nonlinear feasibility correction; every result is checked again."""
        nonlocal corrective_gradients
        for _ in range(cfg.get("retraction_steps", 0)):
            with torch.enable_grad():
                tt = trial[0].detach().requires_grad_(True)
                cc = trial[1:].reshape_as(c).detach().requires_grad_(True)
                critic = torch.stack([margin_drift(n["teacher"], predict(n, p, basis, nb, tt, cc, 1.)).max() for n in native]).max()
                if float(critic) <= limit: break
                ht, hc = torch.autograd.grad(critic, [tt, cc]); corrective_gradients += 1
                normal = torch.cat([ht.reshape(1), hc.flatten()]).detach()
            # Stay tangent to the active parameter sphere while restoring the task constraint.
            if float(trial.norm()) >= .99*radius:
                normal -= (normal@trial)/trial.square().sum()*trial
            if float(normal.square().sum()) < 1e-20: break
            trial = trial-(float(critic)-limit+cfg["retraction_interior_margin"])*normal/normal.square().sum()
            trial *= min(1., radius/float(trial.norm()))
        return trial

    for step in range(cfg["optimization_steps"]):
        objective, critic, distance, stress, changed, drifts = evaluate(t, c)
        gt, gc = torch.autograd.grad(objective, [t, c], retain_graph=True)
        g = torch.cat([gt.reshape(1), gc.flatten()])
        coordinates = torch.cat([t.detach().reshape(1), c.detach().flatten()])
        if float(distance) >= .99*radius**2:
            g -= (torch.dot(g, coordinates).clamp_min(0)/coordinates.square().sum())*coordinates
        if float(distance) <= 1.01*distance_floor:
            g -= (torch.dot(g, coordinates).clamp_max(0)/coordinates.square().sum())*coordinates
        active_witnesses = 0
        if cfg.get("projection_mode", "single") == "joint" and float(critic) >= cfg["active_fraction"]*limit:
            active = (drifts >= cfg["active_fraction"]*limit).nonzero().flatten()
            selected = active[torch.argsort(drifts[active], descending=True)[:cfg["maximum_active_witnesses"]]]
            normals = []
            if float(distance) >= .99*radius**2 or float(distance) <= 1.01*distance_floor:
                normals.append(coordinates/coordinates.norm())
            for witness in selected:
                ht, hc = torch.autograd.grad(drifts[witness], [t, c], retain_graph=True)
                normals.append(torch.cat([ht.reshape(1), hc.flatten()]))
            normals = torch.stack(normals)
            normals /= normals.norm(dim=1, keepdim=True).clamp_min(1e-20)
            # Joint equality tangent is conservative; backtracking checks all constraints.
            gram = normals@normals.T
            g -= normals.T@(torch.linalg.pinv(gram, rtol=cfg["active_gram_rtol"])@(normals@g))
            active_witnesses = len(selected)
        elif float(critic) >= cfg["active_fraction"]*limit:
            ht, hc = torch.autograd.grad(critic, [t, c])
            normal = torch.cat([ht.reshape(1), hc.flatten()])
            g -= (torch.dot(g, normal).clamp_min(0)/normal.square().sum().clamp_min(1e-20))*normal
        g = g/g.norm().clamp_min(1e-20)
        chosen = None
        with torch.no_grad():
            for backtrack in range(cfg["backtracking_steps"]):
                trial = coordinates+cfg["normalized_step"]*(.5**backtrack)*g
                trial *= min(1., radius/float(trial.norm()))
                trial = retract(trial)
                tt, cc = trial[0], trial[1:].reshape_as(c)
                values = evaluate(tt, cc)
                if (float(values[1]) <= limit and float(values[2]) >= distance_floor*(1-1e-8)
                        and float(values[0]) >= float(objective)):
                    chosen = (tt.clone(), cc.clone(), values, backtrack); break
            if chosen is not None:
                t.copy_(chosen[0]); c.copy_(chosen[1]); accepted += 1
                if float(chosen[2][0]) > best[0]: best = (float(chosen[2][0]), t.detach().clone(), c.detach().clone())
        history.append({"step": step+1, "accepted": chosen is not None,
                        "objective": float(objective), "critic": float(critic), "distance": float(distance.sqrt()),
                        "stress": float(stress), "proxy_changed": changed,
                        "joint_active_witnesses": active_witnesses,
                        "cumulative_corrective_gradient_evaluations": corrective_gradients,
                        "backtracking_evaluations": cfg["backtracking_steps"] if chosen is None else chosen[3]+1})
        if (step+1)%8 == 0: print(json.dumps({"target": target, **history[-1]}), flush=True)
    with torch.no_grad():
        t, c = best[1], best[2]; final = evaluate(t, c)
        confirmation = pack(on_device(cal["confirmation"]), p, basis, ray, na, cfg, True)
        confirmed = predict(confirmation, p, basis, nb, t, c, 1.)
        confirmation_critic = float(margin_drift(confirmation["teacher"], confirmed).max())
        ea = t*na*ray; eb = nb*c@basis["B"]["vh"]
        numerator = (ea*p["A"]).sum(1)-(eb*p["B"]).sum(0)
        denominator = p["A"].square().sum(1)+p["B"].square().sum(0)
        gauge = float((numerator.square()/denominator.clamp_min(1e-20)).sum()/(ea.square().sum()+eb.square().sum()))
    output = Path(cfg["diagnostic_output"]); output.mkdir(parents=True, exist_ok=True)
    path = output/(target+"_diagnostic.pt"); assert not path.exists()
    torch.save({**cal, "protocol": cfg, "alpha": float(t), "coefficient_B": (nb*c).cpu(),
                "optimized_coefficient": c.cpu(), "optimizer_history": history, "optimizer_steps": cfg["optimization_steps"],
                "accepted_gradient_updates": accepted, "parent_diagnostic_sha256": geometry["diagnostic_sha256"]}, path)
    report = {"target": target, "method": "projected_publication_distance_game", "optimizer_steps": cfg["optimization_steps"],
              "accepted_gradient_updates": accepted, "parameter_variables": ["task_ray_t", "c_proj_singular_input_coefficient"],
              "corrective_gradient_evaluations": corrective_gradients,
              "original_expert_parameters_updated": False, "supervised_task_training": False, "labels_used": False,
              "future_partner_used": False, "test_used": False, "new_release_weights": 0,
              "initial_objective": float(initial[0]), "final_objective": float(final[0]),
              "initial_proxy_changed": initial[4], "final_proxy_changed": final[4], "proxy_n": len(winner),
              "parameter_distance_floor": distance_floor**.5,
              "parameter_distance": float(final[2].sqrt()), "critic": float(final[1]), "confirmation_critic": confirmation_critic,
              "neuron_scaling_explanation": gauge, "source_sha256": digest(__file__), "checkpoint_sha256": hashes,
              "diagnostic_sha256": digest(path), "head_class_balance_counts": counts.cpu().tolist(), "global_optimum_claim": False}
    save_json(output/(target+".json"), report); save_json(output/(target+"_history.json"), history)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--target", required=True)
    parser.add_argument("--protocol", required=True); args = parser.parse_args()
    cfg = json.loads(Path(args.protocol).read_text())
    for name, pin in cfg["source_sha256"].items(): assert digest(name) == pin, name
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    diagnose(args.target, cfg)
