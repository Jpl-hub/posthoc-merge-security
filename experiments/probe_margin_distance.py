"""Task-SVD distance optimization with finite, margin-relative teacher probes."""

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
from task_probe_distance import PREFIX, checked_models, gelu, parameters, tail


def scores(h, params):
    return tail(h, params)[1] @ params["head"].T


def margin_drift(before, after):
    winner = before.argmax(-1)
    gap = before.gather(-1, winner[..., None]) - before
    change = (after-before).gather(-1, winner[..., None]) - (after-before)
    mask = torch.nn.functional.one_hot(winner, before.shape[-1]).bool()
    ratio = change.abs()/gap.masked_fill(mask, 1.).clamp_min(1e-12)
    ratio.masked_fill_(mask, 0.)
    return ratio.max(-1).values


@torch.no_grad()
def task_probes(cache, params, basis, cfg):
    """Right singular vectors perturb real task activations, not images."""
    n = cfg["probe_contexts"]
    x, h = cache["x"][:n], cache["h"][:n]
    z = gelu(x@params["A"].T+params["a"])
    va, vb = basis["A"]["vh"], basis["B"]["vh"]
    rms_a = (cache["x"]@va.T).square().mean(0).sqrt()
    all_z = gelu(cache["x"]@params["A"].T+params["a"])
    rms_b = (all_z@vb.T).square().mean(0).sqrt()
    signs = torch.tensor([-1., 1.], device=x.device)
    shifts_a = (cfg["finite_step"]*signs[:, None, None]*rms_a[None, :, None]*va[None, :, :]).flatten(0, 1)
    shifts_b = (cfg["finite_step"]*signs[:, None, None]*rms_b[None, :, None]*vb[None, :, :]).flatten(0, 1)
    xa = (x[:, None, :]+shifts_a[None, :, :]).flatten(0, 1)
    zb = (z[:, None, :]+shifts_b[None, :, :]).flatten(0, 1)
    base = h-z@params["B"].T
    ha = base[:, None, :]+(gelu(xa@params["A"].T+params["a"])@params["B"].T).reshape(n, len(shifts_a), -1)
    hb = base[:, None, :]+(zb@params["B"].T).reshape(n, len(shifts_b), -1)
    return {"x": x, "z": z, "base": base, "xa": xa, "shifts_b": shifts_b,
            "teacher_a": scores(ha, params), "teacher_b": scores(hb, params)}


@torch.no_grad()
def responses(cache, params, ea, eb, probe=None, factor=1.):
    x, h = cache["x"], cache["h"]
    old = gelu(x@params["A"].T+params["a"])
    new = gelu(x@(params["A"]+factor*ea).T+params["a"])
    dh = new@(params["B"]+factor*eb).T-old@params["B"].T
    before, after = scores(h, params), scores(h+dh, params)
    drift = margin_drift(before, after)
    result = {"n": len(x), "changed_predictions": int((before.argmax(1)!=after.argmax(1)).sum()),
              "margin_drift_max": float(drift.max()),
              "score_rms": float((after-before).square().sum(1).mean().sqrt()),
              "feature_rms": float((tail(h+dh, params)[1]-tail(h, params)[1]).square().sum(1).mean().sqrt())}
    if probe is not None:
        n = len(probe["x"]); count = probe["teacher_a"].shape[1]
        ba = params["B"]+eb
        altered_a = gelu(probe["xa"]@(params["A"]+ea).T+params["a"])@ba.T
        changed_a = scores(probe["base"][:, None, :]+altered_a.reshape(n, count, -1), params)
        altered_z = gelu(probe["x"]@(params["A"]+ea).T+params["a"])
        altered_b = (altered_z[:, None, :]+probe["shifts_b"][None, :, :])@ba.T
        changed_b = scores(probe["base"][:, None, :]+altered_b, params)
        da = margin_drift(probe["teacher_a"], changed_a)
        db = margin_drift(probe["teacher_b"], changed_b)
        # A finite worst-case witness on all sampled task-vector probes.
        result.update(probe_margin_drift_max=float(torch.maximum(da.max(), db.max())),
                      probe_predictions_changed=int((changed_a.argmax(-1)!=probe["teacher_a"].argmax(-1)).sum()
                          +(changed_b.argmax(-1)!=probe["teacher_b"].argmax(-1)).sum()),
                      probe_prediction_instances=2*n*count)
    return result


@torch.no_grad()
def diagnose(target, cfg):
    ref = Path(cfg["compensation_references"][target])
    meta = json.loads((ref/"execution.json").read_text())
    assert meta["completed"] and digest(ref/"artifacts/calibration.pt") == meta["calibration_sha256"]
    old = torch.load(ref/"artifacts/calibration.pt", map_location="cpu")
    anchor, expert, head, hashes = checked_models(target)
    p_cpu = parameters(expert, head); p = on_device(p_cpu)
    assert all(torch.equal(p_cpu[s], old["parameter_reference"][s]) for s in ["A", "B"])
    basis = {s: on_device(b) for s, b in old["basis"].items()}
    native, scale, proxy = [on_device(old[k]) for k in ["native", "scale", "proxy"]]
    interpolation = merge(anchor, expert, anchor, cfg["proxy_beta"])
    pp = on_device(parameters(interpolation, head)); del interpolation
    norms = {s: p[s].norm() for s in ["A", "B"]}; ray = old["ray"].cuda(); v = basis["B"]["vh"]
    teacher_output = gelu(native["x"]@p["A"].T+p["a"])@p["B"].T
    probes = {g: task_probes(c, p, basis, cfg) for g, c in [("fit", native), ("scale", scale)]}

    def compensate(alpha):
        ea = alpha*norms["A"]*ray
        znew = gelu(native["x"]@(p["A"]+ea).T+p["a"]); inputs = znew@v.T
        gram = inputs.T@inputs
        ridge = cfg["ridge_relative_floor"]*torch.linalg.eigvalsh(gram)[-1]
        k = torch.linalg.solve(gram+ridge*torch.eye(len(v), device="cuda"), inputs.T@(teacher_output-znew@p["B"].T)).T
        return ea, k@v, k, float(ridge)

    trials = []
    def feasible(alpha):
        ea, eb, _, _ = compensate(alpha)
        distance = float((ea.norm()/norms["A"]).square()+(eb.norm()/norms["B"]).square())
        groups = {g: responses(c, p, ea, eb, probes[g]) for g, c in [("fit", native), ("scale", scale)]}
        valid = distance <= cfg["parameter_budget"]**2 and all(max(d["margin_drift_max"], d["probe_margin_drift_max"]) <= cfg["margin_fraction"] for d in groups.values())
        trials.append({"alpha": alpha, "distance_squared": distance, "groups": groups, "feasible": valid})
        return valid

    # Both signs belong to the one scalar optimization domain. No peer/test selection.
    for sign in [-1., 1.]:
        upper = cfg["parameter_budget"]
        if not feasible(sign*upper):
            low, high = 0., upper
            for _ in range(cfg["scalar_solver_limit"]):
                mid = (low+high)/2
                if feasible(sign*mid): low = mid
                else: high = mid
    choices = [t for t in trials if t["feasible"]]
    assert choices, "No nonzero feasible point under the frozen margin/probe constraints"
    chosen = max(choices, key=lambda t: t["distance_squared"]); alpha = chosen["alpha"]
    ea, eb, k, ridge = compensate(alpha)
    diagnostic = {g: responses(c, p, ea, eb, probes[g]) for g, c in [("fit", native), ("scale", scale)]}
    diagnostic["proxy"] = responses(proxy, pp, ea, eb, factor=cfg["proxy_beta"])
    # Confirmation is checked once after the sole training-selected point is fixed.
    pref = Path(cfg["probe_references"][target]); pin = json.loads((pref/"results.json").read_text())["probes_sha256"]
    assert digest(pref/"artifacts/probes.pt") == pin
    confirmation = torch.load(pref/"artifacts/probes.pt", map_location="cpu")["groups"]["confirmation"]
    confirmation_gpu = on_device(confirmation)
    diagnostic["confirmation"] = responses(confirmation_gpu, p, ea, eb, task_probes(confirmation_gpu, p, basis, cfg))
    numerator = (ea*p["A"]).sum(1)-(eb*p["B"]).sum(0)
    denominator = p["A"].square().sum(1)+p["B"].square().sum(0)
    gauge = float((numerator.square()/denominator.clamp_min(1e-20)).sum()/(ea.square().sum()+eb.square().sum()))
    geometry = {"method": "margin_relative_task_singular_probes", "alpha": alpha,
        "parameter_distance": chosen["distance_squared"]**.5, "margin_fraction": cfg["margin_fraction"],
        "diagnostics": diagnostic, "neuron_scaling_explanation": gauge, "scalar_solver_evaluations": len(trials),
        "optimizer_steps": 0, "gradient_parameter_updates": 0, "labels_used": False, "partner_used": False,
        "test_used": False, "new_release_weights": 0, "source_sha256": digest(__file__),
        "task_probe_rank": cfg["probe_rank"], "probe_contexts": cfg["probe_contexts"], "finite_step": cfg["finite_step"],
        "full_visual_feature_preservation_claim": False, "global_optimum_claim": False,
        "checkpoint_sha256": hashes, "old_cache_sha256": meta["calibration_sha256"]}
    cal = {**old, "protocol": cfg, "parameter_reference": p_cpu, "alpha": alpha,
           "coefficient_B": k.cpu(), "ridge": ridge, "trials": trials, "confirmation": confirmation,
           "source_cache_sha256": meta["calibration_sha256"], "source_cache_path": str(ref/"artifacts/calibration.pt")}
    out = Path(cfg["diagnostic_output"]); out.mkdir(parents=True, exist_ok=True)
    path = out/(target+"_diagnostic.pt"); assert not path.exists(), "Existing diagnostic must be reused"
    torch.save(cal, path); geometry["diagnostic_sha256"] = digest(path)
    save_json(out/(target+".json"), geometry); save_json(out/(target+"_trials.json"), trials)
    print(json.dumps(geometry), flush=True)


@torch.no_grad()
def construct(original, anchor, head, training, indices, cfg, target):
    out = Path(cfg["diagnostic_output"]); geometry = json.loads((out/(target+".json")).read_text())
    path = out/(target+"_diagnostic.pt"); assert digest(path) == geometry["diagnostic_sha256"]
    assert geometry["source_sha256"] == digest(__file__)
    cal = torch.load(path, map_location="cpu"); assert cal["protocol"] == cfg and cal["indices"] == indices
    assert geometry["neuron_scaling_explanation"] < cfg["maximum_scaling_explanation"]
    confirm = geometry["diagnostics"]["confirmation"]
    assert max(confirm["margin_drift_max"], confirm["probe_margin_drift_max"]) <= cfg["margin_fraction"], "Held task-probe constraint failed; no retuning"
    assert geometry["diagnostics"]["proxy"]["changed_predictions"] > 0, "No proxy prediction change; stop before release"
    params = parameters(original, head)
    assert all(torch.equal(params[s], cal["parameter_reference"][s]) for s in ["A", "B"])
    ea = cal["alpha"]*params["A"].norm()*cal["ray"]
    eb = cal["coefficient_B"]@cal["basis"]["B"]["vh"]
    released = copy.deepcopy(original).cpu().eval(); edits = {}
    for suffix, key, delta in [("c_fc.weight", "A", ea), ("c_proj.weight", "B", eb)]:
        name = PREFIX+suffix; value = (params[key]+delta).float()
        released.state_dict()[name].copy_(value); edits[name] = value.clone()
    geometry["new_release_weights"] = 1
    return released, edits, cal, geometry


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True); parser.add_argument("--partner", required=True)
    parser.add_argument("--protocol", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--diagnose-only", action="store_true"); args = parser.parse_args()
    cfg = json.loads(Path(args.protocol).read_text())
    for path, pin in cfg["source_sha256"].items(): assert digest(path) == pin, path
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.diagnose_only: diagnose(args.target, cfg)
    else: run(args, constructor=construct, holdout_loader=holdout_loader)
