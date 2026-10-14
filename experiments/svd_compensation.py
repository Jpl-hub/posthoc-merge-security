"""Bounded task-vector displacement with closed-form composite-block distillation."""

import argparse
import copy
import json
from pathlib import Path

import torch

from clip_benchmark import digest, merge
from covariance_noise import run
from decision_noise import holdout_loader
from probe_distance_candidate import on_device
from subspace_probe import save_json
from task_probe_distance import PREFIX, capture, checked_models, gelu, parameters, tail


@torch.no_grad()
def build(original, anchor, head, training, indices, cfg, target, diagnose_only=False):
    output = Path(cfg["diagnostic_output"])
    if not diagnose_only:
        report = output / (target + ".json")
        geometry = json.loads(report.read_text())
        path = output / (target + "_diagnostic.pt")
        assert digest(path) == geometry["diagnostic_sha256"]
        cal = torch.load(path, map_location="cpu")
        assert cal["protocol"] == cfg and cal["indices"] == indices
        assert geometry["source_sha256"] == digest(__file__)
        assert geometry["neuron_scaling_explanation"] < cfg["maximum_scaling_explanation"]
        span = geometry["span_diagnostic"]
        assert span["full_output_residual_fraction"] < span["restricted_residual_fraction"]
        assert geometry["diagnostics"]["proxy_fit"]["changed_predictions"] > 0, "No proxy decision change; stop before generating release"
        p_cpu = parameters(original, head)
        for s in ["A", "B"]:
            assert torch.equal(p_cpu[s], cal["parameter_reference"][s])
        ea = cal["alpha"] * p_cpu["A"].norm() * cal["ray"]
        eb = cal["coefficient_B"] @ cal["basis"]["B"]["vh"]
        released = copy.deepcopy(original).cpu().eval(); edits = {}
        for site, change, suffix in [("A", ea, "c_fc.weight"), ("B", eb, "c_proj.weight")]:
            key = PREFIX+suffix; value = (p_cpu[site]+change).float()
            released.state_dict()[key].copy_(value); edits[key] = value.clone()
        geometry["new_release_weights"] = 1
        return released, edits, cal, geometry
    reference = Path(cfg["matrix_references"][target])
    meta = json.loads((reference / "execution.json").read_text())
    assert meta["completed"] and digest(reference / "artifacts/calibration.pt") == meta["calibration_sha256"]
    assert indices == json.loads((reference / "training_indices.json").read_text())
    stored = torch.load(reference / "artifacts/calibration.pt", map_location="cpu")
    probe = Path(cfg["probe_references"][target])
    probe_meta = json.loads((probe / "results.json").read_text())
    assert digest(probe / "artifacts/probes.pt") == probe_meta["probes_sha256"]
    svd = torch.load(probe / "artifacts/probes.pt", map_location="cpu")["svd"]
    p_cpu = parameters(original, head); p = on_device(p_cpu)
    basis = {s: on_device(b) for s, b in stored["basis"].items()}
    selected = svd["A"]["selected"]
    task_a = (svd["A"]["u"][:, selected] * svd["A"]["s"][selected]) @ svd["A"]["vh"][selected]
    ray = task_a.cuda() / task_a.norm().cuda()
    native = on_device(stored["native"])
    scale_cpu = capture(original, training, indices["scale"])
    scale = on_device(scale_cpu)
    proxy = merge(anchor, original, anchor, cfg["proxy_beta"])
    pp = on_device(parameters(proxy, head)); del proxy
    proxy_cache = on_device(stored["proxy"])
    norms = {s: p[s].norm() for s in ["A", "B"]}
    response_scale = torch.tensor(stored["scales"]["B"], dtype=torch.float64, device="cuda")
    v = basis["B"]["vh"]
    z = gelu(native["x"] @ p["A"].T + p["a"])
    target_output = z @ p["B"].T
    pre = native["x"] @ p["A"].T + p["a"]
    prob = torch.sigmoid(1.702 * pre)
    derivative = prob + 1.702 * pre * prob * (1-prob)
    first_change = ((native["x"] @ (norms["A"] * ray).T) * derivative) @ p["B"].T
    x = z @ v.T
    gram = x.T @ x
    ridge = cfg["ridge_relative_floor"] * float(torch.linalg.eigvalsh(gram)[-1])
    linear_k = -torch.linalg.solve(gram + ridge * torch.eye(len(v), device="cuda"), x.T @ first_change).T
    restricted_k = basis["B"]["u"] @ (basis["B"]["u"].T @ linear_k)
    linear_residual = first_change + x @ linear_k.T
    restricted_residual = first_change + x @ restricted_k.T
    span_report = {"upstream_energy": float(first_change.square().sum()),
                   "restricted_residual_fraction": float(restricted_residual.square().sum()/first_change.square().sum()),
                   "full_output_residual_fraction": float(linear_residual.square().sum()/first_change.square().sum()),
                   "ridge": ridge, "gram_condition": float(torch.linalg.eigvalsh(gram)[-1]/torch.linalg.eigvalsh(gram)[0])}

    def compensate(alpha):
        ea = alpha * norms["A"] * ray
        znew = gelu(native["x"] @ (p["A"]+ea).T + p["a"])
        inputs = znew @ v.T
        residual = target_output - znew @ p["B"].T
        g = inputs.T @ inputs
        lam = cfg["ridge_relative_floor"] * torch.linalg.eigvalsh(g)[-1]
        k = torch.linalg.solve(g + lam * torch.eye(len(v), device="cuda"), inputs.T @ residual).T
        return ea, k @ v, k, float(lam)

    def measure(cache, params, ea, eb, factor=1.):
        x = cache["x"]
        old = gelu(x @ params["A"].T + params["a"])
        new = gelu(x @ (params["A"]+factor*ea).T + params["a"])
        dh = new @ (params["B"]+factor*eb).T - old @ params["B"].T
        _, f0 = tail(cache["h"], params); _, f1 = tail(cache["h"]+dh, params)
        feature_loss = (f1-f0).square().sum(1).mean()
        probe_loss = ((dh @ basis["B"]["u"]) / response_scale).square().sum(1).mean()
        contrast = params["head"] - params["head"].mean(0)
        s0, s1 = f0 @ contrast.T, f1 @ contrast.T
        winner = s0.argmax(1); wrong = s0.clone()
        wrong[torch.arange(len(winner), device="cuda"), winner] = -torch.inf
        gap = s0.gather(1, winner[:, None]).flatten() - wrong.max(1).values
        ds = s1-s0
        return {"feature_loss": float(feature_loss), "probe_loss": float(probe_loss),
                "combined_loss": float(feature_loss+probe_loss), "n": len(winner),
                "changed_predictions": int((s1.argmax(1)!=winner).sum()),
                "score_rms": float(ds.square().sum(1).mean().sqrt()),
                "certified_unchanged": int((2*ds.abs().max(1).values<gap).sum()),
                "gap_median": float(gap.median())}

    trials = []
    def feasible(alpha):
        ea, eb, _, _ = compensate(alpha)
        distance = float((ea.norm()/norms["A"]).square()+(eb.norm()/norms["B"]).square())
        groups = {name: measure(c, p, ea, eb) for name, c in [("fit", native), ("scale", scale)]}
        valid = distance <= cfg["parameter_budget"]**2 and all(
            d["combined_loss"] <= cfg["response_budget"]**2 for d in groups.values())
        trials.append({"alpha": alpha, "distance_squared": distance, "groups": groups, "feasible": valid})
        return valid

    upper = cfg["parameter_budget"]
    if feasible(upper):
        alpha = upper
    else:
        low, high = 0., upper
        for _ in range(cfg["scalar_solver_limit"]):
            mid = (low+high)/2
            if feasible(mid): low = mid
            else: high = mid
        valid = [t for t in trials if t["feasible"]]
        assert valid, "No nonzero feasible displacement on the frozen task-vector ray"
        alpha = max(valid, key=lambda t: t["distance_squared"])["alpha"]
    ea, eb, k, lam = compensate(alpha)
    diagnostic = {name: measure(c, params, ea, eb, factor) for name, c, params, factor in
                  [("native_fit", native, p, 1.), ("native_scale", scale, p, 1.),
                   ("proxy_fit", proxy_cache, pp, cfg["proxy_beta"])]}
    numerator = (ea*p["A"]).sum(1) - (eb*p["B"]).sum(0)
    denominator = p["A"].square().sum(1) + p["B"].square().sum(0)
    explained = float((numerator.square()/denominator.clamp_min(1e-20)).sum()/(ea.square().sum()+eb.square().sum()))
    geometry = {"method": "task_singular_ray_composite_distillation", "span_diagnostic": span_report,
        "alpha": alpha, "parameter_distance": float(((ea.norm()/norms["A"]).square()+(eb.norm()/norms["B"]).square()).sqrt()),
        "layer_relative_distance": {"A": float(ea.norm()/norms["A"]), "B": float(eb.norm()/norms["B"])},
        "absolute_parameter_distance": float((ea.square().sum()+eb.square().sum()).sqrt()),
        "native_preprobe_rms": float(((native["x"]@ea.T)@basis["A"]["u"]).square().sum(1).mean().sqrt())/stored["scales"]["A"],
        "neuron_scaling_explanation": explained, "diagnostics": diagnostic, "scalar_solver_evaluations": len(trials),
        "optimizer_steps": 0, "gradient_parameter_updates": 0, "labels_used": False, "partner_used": False,
        "phase1_bias_used": False, "global_optimum_claim": False, "new_release_weights": 0,
        "source_sha256": digest(__file__), "new_teacher_forward_images": len(indices["scale"]),
        "fit_cache_reused": True, "scale_is_construction_data": True}
    cal = {"basis": stored["basis"], "native": stored["native"], "scale": scale_cpu, "proxy": stored["proxy"],
           "ray": ray.cpu(), "coefficient_B": k.cpu(), "alpha": alpha, "ridge": lam, "trials": trials,
           "source_cache_sha256": meta["calibration_sha256"], "probe_sha256": probe_meta["probes_sha256"],
           "protocol": cfg, "indices": indices, "parameter_reference": p_cpu}
    output.mkdir(parents=True, exist_ok=True)
    path = output/(target+"_diagnostic.pt")
    assert not path.exists(), "Diagnostic already saved; reuse instead of rerunning"
    torch.save(cal, path)
    geometry["diagnostic_sha256"] = digest(path)
    save_json(output/(target+".json"), geometry)
    save_json(output/(target+"_trials.json"), trials)
    return geometry


def construct(original, anchor, head, training, indices, cfg, target):
    return build(original, anchor, head, training, indices, cfg, target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True); parser.add_argument("--partner", required=True)
    parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--diagnose-only", action="store_true")
    args = parser.parse_args()
    if args.diagnose_only:
        from covariance_noise import dataset
        from subspace_probe import split_training
        cfg = json.loads(Path(args.protocol).read_text())
        for path, pin in cfg["source_sha256"].items(): assert digest(path) == pin, path
        torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
        anchor, expert, head, _ = checked_models(args.target)
        training = dataset(args.target, "data", expert.val_preprocess, True)
        n = min(cfg["maximum_calibration_group"], len(training)//3)
        indices = split_training(training, {k: n for k in ["fit", "scale", "confirmation"]}, cfg["seed"])
        print(json.dumps(build(expert, anchor, head, training, indices, cfg, args.target, True)), flush=True)
    else:
        run(args, constructor=construct, holdout_loader=holdout_loader)
