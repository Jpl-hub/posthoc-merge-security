"""Cached task-only diagnosis of separate versus composite MLP probe constraints.

No release weights, partner models, test inputs, or parameter optimizer.
"""

import argparse
import json
from pathlib import Path

import torch

from clip_benchmark import digest
from probe_distance_candidate import displacement, nonlinear, on_device
from task_probe_distance import checked_models, mlp_delta, parameters, tail
from subspace_probe import save_json


@torch.no_grad()
def diagnose(target, output):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    source = Path("outputs/probe_matrix_" + target.lower())
    meta = json.loads((source / "execution.json").read_text())
    cfg = json.loads((source / "protocol.json").read_text())
    assert meta["completed"] and digest(source / "artifacts/calibration.pt") == meta["calibration_sha256"]
    for path, pin in cfg["source_sha256"].items():
        assert digest(path) == pin, path
    cal = torch.load(source / "artifacts/calibration.pt", map_location="cpu")
    anchor, expert, head, hashes = checked_models(target)
    p = on_device(parameters(expert, head))
    p0 = parameters(anchor, head)
    pp = {k: (p0[k].cuda() + cfg["proxy_beta"] * (p[k] - p0[k].cuda()))
          if torch.is_tensor(p[k]) else p[k] for k in p}
    # A task-arithmetic interpolation leaves the target head fixed.
    pp["head"] = p["head"]
    basis = {s: on_device(b) for s, b in cal["basis"].items()}
    native, proxy = [on_device(cal[k]) for k in ["native", "proxy"]]
    norms = {s: p[s].norm() for s in ["A", "B"]}
    scales = {k: torch.tensor(v, dtype=torch.float64, device="cuda") for k, v in cal["scales"].items()}
    h, hf, gp = [cal[k].cuda() for k in ["gram_native", "gram_native_features", "gram_proxy"]]
    r = cfg["probe_rank"]
    ax = native["x"] @ basis["A"]["vh"].T
    ga = torch.zeros_like(h)
    ga[:r*r, :r*r] = (norms["A"] / scales["A"]).square() * torch.kron(torch.eye(r, device="cuda"), ax.T @ ax / len(ax))
    hb = (h - hf - ga + (h - hf - ga).T) / 2
    assert float(torch.linalg.eigvalsh(hb)[0]) >= -1e-8 * float(h.norm())
    composite = (hf + hb + (hf + hb).T) / 2
    vals, q = torch.linalg.eigh((gp - hf + gp.T - hf.T) / 2)
    keep = vals > 1e-10 * max(float(gp.norm()), float(hf.norm()))
    q = q[:, keep]
    assert len(q.T), "No visibility subspace"
    report = {"target": target, "cache_sha256": meta["calibration_sha256"],
              "checkpoint_sha256": hashes, "source_sha256": digest(__file__),
              "new_release_weights": 0, "new_model_forwards": 0, "partner_used": False,
              "test_used": False, "optimizer_steps": 0, "visibility_dimension": int(keep.sum()),
              "internal_probe_gram_min_eigenvalue": float(torch.linalg.eigvalsh(hb)[0]), "constraints": {}}
    for name, metric in [("separate", h), ("composite", composite)]:
        ev, vectors = torch.linalg.eigh(q.T @ metric @ q)
        d = q @ vectors[:, 0]
        curvature = float(d @ metric @ d)
        assert curvature > 0
        upper = min(cfg["parameter_budget"], cfg["response_budget"] / curvature ** .5)
        trials = []

        def feasible(alpha):
            total, parts, changed = nonlinear(native, p, basis, norms, scales, alpha * d)
            constrained = total if name == "separate" else parts[0] + parts[2]
            trials.append({"alpha": alpha, "constrained_loss": constrained, "all_parts": parts})
            return constrained <= cfg["response_budget"] ** 2

        if feasible(upper):
            alpha = upper
        else:
            low, high = 0., upper
            for _ in range(cfg["scalar_solver_limit"]):
                middle = (low + high) / 2
                if feasible(middle): low = middle
                else: high = middle
            alpha = low
        c = alpha * d
        _, selected_parts, _ = nonlinear(native, p, basis, norms, scales, c)
        edits = displacement(c, basis, norms)
        scored = {}
        for context, cache, params, factor in [("native", native, p, 1.), ("proxy", proxy, pp, cfg["proxy_beta"])]:
            _, f0 = tail(cache["h"], params)
            dh = mlp_delta(cache, params, factor * edits["A"], factor * edits["B"])
            _, f1 = tail(cache["h"] + dh, params)
            wc = params["head"] - params["head"].mean(0)
            s0, s1 = f0 @ wc.T, f1 @ wc.T
            winner = s0.argmax(1)
            wrong = s0.clone(); wrong[torch.arange(len(winner), device="cuda"), winner] = -torch.inf
            margin = s0.gather(1, winner[:, None]).flatten() - wrong.max(1).values
            ds = s1 - s0
            scored[context] = {"n": len(winner), "feature_rms": float((f1-f0).square().sum(1).mean().sqrt()),
                               "score_rms": float(ds.square().sum(1).mean().sqrt()), "gap_median": float(margin.median()),
                               "certified_unchanged": int((2*ds.abs().max(1).values < margin).sum()),
                               "changed_predictions": int((s1.argmax(1) != winner).sum())}
        ea, eb = edits["A"], edits["B"]
        zeroa, zerob = torch.zeros_like(ea), torch.zeros_like(eb)
        _, f0 = tail(native["h"], p)
        def feature_energy(a, b):
            _, f = tail(native["h"] + mlp_delta(native, p, a, b), p)
            return float((f-f0).square().sum(1).mean())
        numerator = (ea*p["A"]).sum(1) - (eb*p["B"]).sum(0)
        denominator = p["A"].square().sum(1) + p["B"].square().sum(0)
        gauge = float((numerator.square()/denominator.clamp_min(1e-20)).sum() / (ea.square().sum()+eb.square().sum()))
        report["constraints"][name] = {"curvature": curvature, "distance": float(c.norm()),
            "native_separate_preprobe_rms": selected_parts[1] ** .5,
            "selected_nonlinear_parts": selected_parts,
            "last_solver_trial": trials[-1], "selected_alpha": alpha, "scalar_evaluations": len(trials),
            "linear_native_feature_energy": float(c @ hf @ c), "linear_proxy_feature_energy": float(c @ gp @ c),
            "nonlinear_feature_energy_A_only": feature_energy(ea, zerob),
            "nonlinear_feature_energy_B_only": feature_energy(zeroa, eb),
            "nonlinear_feature_energy_combined": feature_energy(ea, eb),
            "neuron_scaling_explanation": gauge, "decision": scored}
        if name == "separate":
            assert abs(curvature - cal["curvature"]) <= 1e-4 * cal["curvature"] + 1e-7
    output = Path(output); output.mkdir(parents=True, exist_ok=False)
    save_json(output / (target + ".json"), report)
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(); diagnose(args.target, args.output)
