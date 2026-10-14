"""Independent NumPy forward and analytic native-JVP verification."""

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


def verify(output):
    output = Path(output); report = json.loads((output/"direction.json").read_text())
    assert digest(output/"direction.pt") == report["direction_sha256"]
    saved = torch.load(output/"direction.pt", map_location="cpu"); cfg = saved["cfg"]
    parent = Path(cfg["margin_reference"])/(report["target"]+"_diagnostic.pt")
    assert digest(parent) == report["parent_diagnostic_sha256"]
    cal = torch.load(parent, map_location="cpu")
    anchor, expert, head, _ = checked_models(report["target"]); p = parameters(expert, head)
    theta, direction = array(saved["theta"]), array(saved["direction"])
    basis = {s: {k: array(v) for k, v in b.items()} for s, b in cal["basis"].items()}
    rank = len(basis["A"]["vh"]); size = 1 if cfg["coordinate_form"] == "ray" else rank**2
    norms = {s: np.linalg.norm(array(p[s])) for s in ["A", "B"]}

    def changes(coeff):
        ea = norms["A"]*coeff[0]*array(cal["ray"]) if size == 1 else norms["A"]*basis["A"]["u"]@coeff[:size].reshape(rank, rank)@basis["A"]["vh"]
        eb = norms["B"]*coeff[size:].reshape(p["B"].shape[0], rank)@basis["B"]["vh"]
        return ea, eb

    ea, eb = changes(theta); da, db = changes(direction)
    a, b, bias = array(p["A"]), array(p["B"]), array(p["a"])

    def logits(h, model):
        return features(h, model.model.visual)@array(head.weight).T

    def score_jvp(h, dh, model):
        pp = parameters(model, head); gamma, beta, projection = [array(pp[k]) for k in ["gamma", "beta", "projection"]]
        centered = h-h.mean(1, keepdims=True); sigma = np.sqrt(np.mean(centered**2, 1, keepdims=True)+pp["eps"])
        dc = dh-dh.mean(1, keepdims=True)
        dn = dc/sigma-centered*np.mean(centered*dc, 1, keepdims=True)/sigma**3
        t = (centered/sigma*gamma+beta)@projection; f = t/np.linalg.norm(t, axis=1, keepdims=True)
        dt = (dn*gamma)@projection
        df = (dt-f*np.sum(dt*f, 1, keepdims=True))/np.linalg.norm(t, axis=1, keepdims=True)
        return df@array(head.weight).T

    rr, jj, finite_max = [], [], []
    finite_index = next((i for i, x in enumerate(report["finite_steps"]) if x["feasible"]), 0)
    feasible = report["finite_steps"][finite_index]
    if "finite_thetas" in saved:
        trial = array(saved["finite_thetas"][finite_index])
    else:
        trial = theta+feasible["step"]*direction
        trial *= min(1., cfg["parameter_budget"]/np.linalg.norm(trial))
    conditional_error = None
    if "conditional_A" in saved:
        inputs = np.concatenate([array(cal[k]["x"]) for k in ["native", "scale"]])
        a0, b0, bias0 = [array(p[k]) for k in ["A", "B", "a"]]
        pre0 = inputs@a0.T+bias0; z0 = pre0*expit(1.702*pre0)
        q0 = z0@basis["B"]["vh"].T
        ridge = cfg["ridge_relative_floor"]*np.linalg.eigvalsh(q0.T@q0)[-1]
        assert abs(ridge-saved["ridge"]) < 1e-9
        errors = []
        for coeff in [theta, trial]:
            delta_a, _ = changes(coeff)
            pre = inputs@(a0+delta_a).T+bias0; z = pre*expit(1.702*pre)
            q = z@basis["B"]["vh"].T
            k = np.linalg.solve(q.T@q+ridge*np.eye(rank), q.T@(z0@b0.T-z@b0.T)).T
            error = float(np.max(np.abs(k/norms["B"]-coeff[size:].reshape(b0.shape[0], rank))))
            assert error < 1e-7
            errors.append(error)
        conditional_error = max(errors)
    tea, teb = changes(trial)
    for group in ["native", "scale"]:
        x, h = array(cal[group]["x"]), array(cal[group]["h"])
        pre0 = x@a.T+bias; z0 = pre0*expit(1.702*pre0); n = cfg["probe_contexts"]
        va, vb = basis["A"]["vh"], basis["B"]["vh"]
        sa = (cfg["finite_step"]*np.array([-1., 1.])[:, None, None]*np.sqrt(np.mean((x@va.T)**2, 0))[None, :, None]*va[None, :, :]).reshape(-1, a.shape[1])
        sb = (cfg["finite_step"]*np.array([-1., 1.])[:, None, None]*np.sqrt(np.mean((z0@vb.T)**2, 0))[None, :, None]*vb[None, :, :]).reshape(-1, b.shape[1])
        count = len(sa); xx = np.concatenate([x, (x[:n, None, :]+sa[None, :, :]).reshape(-1, x.shape[1]), np.repeat(x[:n], count, axis=0)])
        base = h-z0@b.T; bases = np.concatenate([base, np.repeat(base[:n], count, axis=0), np.repeat(base[:n], count, axis=0)])
        offset = np.concatenate([np.zeros((len(x)+n*count, b.shape[1])), np.tile(sb, (n, 1))])
        def hidden(aa, bb):
            pre = xx@(a+aa).T+bias
            return bases+(pre*expit(1.702*pre)+offset)@(b+bb).T
        hh = hidden(ea, eb); pre = xx@(a+ea).T+bias; probability = expit(1.702*pre)
        z = pre*probability+offset
        dz = (probability+1.702*pre*probability*(1-probability))*(xx@da.T)
        dh = dz@(b+eb).T+z@db.T
        before, after, derivative = logits(hidden(np.zeros_like(a), np.zeros_like(b)), expert), logits(hh, expert), score_jvp(hh, dh, expert)
        winner = before.argmax(1); rows = np.arange(len(before)); gap = before[rows, winner, None]-before; gap[rows, winner] = 1.
        ds = after-before; residual = (ds[rows, winner, None]-ds)/np.maximum(gap, 1e-12); residual[rows, winner] = 0.
        jvp = (derivative[rows, winner, None]-derivative)/np.maximum(gap, 1e-12); jvp[rows, winner] = 0.
        delta = logits(hidden(tea, teb), expert)-before
        finite = (delta[rows, winner, None]-delta)/np.maximum(gap, 1e-12); finite[rows, winner] = 0.
        finite_max.append(float(np.abs(finite).max())); rr.append(residual.flatten()); jj.append(jvp.flatten())
    residual, jvp = np.concatenate(rr), np.concatenate(jj)
    np.testing.assert_allclose(residual, array(saved["native_residual"]), atol=1e-7, rtol=1e-7)
    np.testing.assert_allclose(jvp, array(saved["native_directional_derivative"]), atol=1e-7, rtol=1e-7)
    assert abs(max(finite_max)-feasible["native_critic"]) < 1e-7
    result = {"status": "passed", "source_sha256": digest(__file__), "direction_sha256": report["direction_sha256"],
              "native_signed_constraints": len(residual), "native_jvp_maximum_absolute_error": float(np.max(np.abs(jvp-array(saved["native_directional_derivative"])))),
              "residual_maximum_absolute_error": float(np.max(np.abs(residual-array(saved["native_residual"])))),
              "finite_step": feasible["step"], "finite_step_feasible": feasible["feasible"],
              "finite_native_critic": max(finite_max), "conditional_numpy_coefficient_error": conditional_error,
              "fresh_model_forwards": 0, "training_cache_only": True, "new_weights": 0}
    save_json(output/"independent_validation.json", result); print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", required=True)
    args = parser.parse_args(); torch.set_num_threads(4); verify(args.output)
