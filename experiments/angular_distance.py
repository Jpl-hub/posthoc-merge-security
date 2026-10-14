"""Task-SVD publication distance with affine-invariant classification directions."""

import argparse
import json
import time
from pathlib import Path

import torch

from clip_benchmark import digest, merge
from probe_distance_candidate import on_device
from probe_margin_distance import scores, task_probes
from projected_distance_game import pack
from subspace_probe import save_json
from task_probe_distance import checked_models, gelu, parameters


def angular_scores(s):
    centered = s-s.mean(-1, keepdim=True)
    return centered/centered.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def signed_residual(before, after):
    winner = before.argmax(-1, keepdim=True)
    gap = before.gather(-1, winner)-before
    mask = torch.nn.functional.one_hot(winner.squeeze(-1), before.shape[-1]).bool()
    ds = after-before
    return ((ds.gather(-1, winner)-ds)/gap.masked_fill(mask, 1.).clamp_min(1e-12)).masked_fill(mask, 0.)


class Problem:
    def __init__(self, target, cfg):
        self.cfg, self.target = cfg, target
        root = Path(cfg["margin_reference"]); path = root/(target+"_diagnostic.pt")
        meta = json.loads((root/(target+".json")).read_text())
        assert digest(path) == meta["diagnostic_sha256"]
        self.parent_sha = meta["diagnostic_sha256"]
        self.cal = torch.load(path, map_location="cpu")
        self.anchor, self.expert, self.head, self.hashes = checked_models(target)
        self.p_cpu = parameters(self.expert, self.head); self.p = on_device(self.p_cpu)
        assert all(torch.equal(self.p_cpu[k], self.cal["parameter_reference"][k]) for k in ["A", "B"])
        self.basis = {s: on_device(b) for s, b in self.cal["basis"].items()}
        self.ray = self.cal["ray"].cuda(); self.na, self.nb = self.p["A"].norm(), self.p["B"].norm()
        self.r = self.basis["A"]["vh"].shape[0]
        self.form = cfg["coordinate_form"]
        self.a_size = 1 if self.form == "ray" else self.r**2
        ca = torch.tensor([self.cal["alpha"]], dtype=torch.float64, device="cuda")
        if self.form == "matrix":
            ca = self.cal["alpha"]*(self.basis["A"]["u"].T@self.ray@self.basis["A"]["vh"].T)
            restored = self.basis["A"]["u"]@ca@self.basis["A"]["vh"]
            assert torch.allclose(restored, self.cal["alpha"]*self.ray, atol=1e-12, rtol=1e-10)
        self.initial = torch.cat([ca.flatten(), (self.cal["coefficient_B"].cuda()/self.nb).flatten()])
        self.native = [self.prepare(self.cal[k], self.p, True) for k in ["native", "scale"]]
        proxy = merge(self.anchor, self.expert, self.anchor, cfg["proxy_beta"])
        self.pp = on_device(parameters(proxy, self.head)); del proxy
        self.proxy = self.prepare(self.cal["proxy"], self.pp, False)
        teacher = self.proxy["teacher"]
        self.winner = teacher.argmax(1); self.rows = torch.arange(len(teacher), device="cuda")
        counts = torch.bincount(self.winner, minlength=teacher.shape[1])
        self.weights = 1./counts[self.winner].double()/int((counts>0).sum())
        self.class_counts = counts.cpu().tolist()
        self.unit = self.gap(angular_scores(teacher)).median().detach()
        assert self.unit > 0
        self.radius = cfg["parameter_budget"]
        self.provenance = self.check_svd()

    @torch.no_grad()
    def prepare(self, cache_cpu, params, probes):
        cache = on_device(cache_cpu)
        packed = pack(cache, params, self.basis, self.ray, self.na, self.cfg, probes)
        xx = [cache["x"]]
        if probes:
            probe = task_probes(cache, params, self.basis, self.cfg)
            n, count = probe["teacher_a"].shape[:2]
            xx += [probe["xa"], probe["x"][:, None, :].expand(-1, count, -1).reshape(n*count, -1)]
        packed["task_input"] = torch.cat(xx)@self.basis["A"]["vh"].T
        return packed

    @torch.no_grad()
    def check_svd(self):
        anchor = on_device(parameters(self.anchor, self.head)); result = {}
        for site in ["A", "B"]:
            u, vh = self.basis[site]["u"], self.basis[site]["vh"]
            task = self.p[site]-anchor[site]; singular = torch.diag(u.T@task@vh.T)
            left = task@vh.T; right = task.T@u
            result[site] = {"input_dimension": vh.shape[1], "output_dimension": u.shape[0],
                            "rank": len(vh), "left_equation_relative_error": float((left-u*singular).norm()/left.norm()),
                            "right_equation_relative_error": float((right-vh.T*singular).norm()/right.norm()),
                            "u_orthogonality_error": float((u.T@u-torch.eye(len(vh), device="cuda")).norm()),
                            "v_orthogonality_error": float((vh@vh.T-torch.eye(len(vh), device="cuda")).norm())}
            assert max(result[site][k] for k in ["left_equation_relative_error", "right_equation_relative_error"]) < 1e-8
        return result

    def predict(self, packed, params, theta, factor=1.):
        ca = theta[:self.a_size]; cb = theta[self.a_size:].reshape(self.p["B"].shape[0], self.r)
        if self.form == "ray":
            delta = ca[0]*packed["ray_pre"]
        else:
            delta = self.na*(packed["task_input"]@ca.reshape(self.r, self.r).T)@self.basis["A"]["u"].T
        z = gelu(packed["pre"]+factor*delta)+packed["offset"]
        h = packed["base"]+z@params["B"].T+factor*self.nb*((z@self.basis["B"]["vh"].T)@cb.T)
        return scores(h, params)

    def gap(self, s):
        competitors = s.clone(); competitors[self.rows, self.winner] = -torch.inf
        return s[self.rows, self.winner]-competitors.max(1).values

    def stress(self, s):
        return -(self.weights*self.gap(angular_scores(s))).sum()/self.unit

    def residuals(self, theta):
        return torch.cat([signed_residual(n["teacher"], self.predict(n, self.p, theta)).flatten() for n in self.native])

    def objective(self, theta):
        proxy = self.predict(self.proxy, self.pp, theta, self.cfg["proxy_beta"])
        return theta.square().sum()/self.radius**2+self.stress(proxy)

    @torch.no_grad()
    def metrics(self, theta):
        s = self.predict(self.proxy, self.pp, theta, self.cfg["proxy_beta"])
        changed = s.argmax(1)!=self.winner
        return {"objective": float(self.objective(theta)), "angular_stress": float(self.stress(s)),
                "parameter_distance": float(theta.norm()), "native_critic": float(self.residuals(theta).abs().max()),
                "proxy_changed": int(changed.sum()), "proxy_n": len(changed),
                "changed_by_teacher_class": torch.bincount(self.winner[changed], minlength=s.shape[1]).cpu().tolist(),
                "minimum_centered_score_norm": float((s-s.mean(1, keepdim=True)).norm(dim=1).min())}


def direction_diagnostic(target, cfg):
    output = Path(cfg["output"]); output.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    problem = Problem(target, cfg); theta = problem.initial.clone().requires_grad_(True)
    s = problem.predict(problem.proxy, problem.pp, theta, cfg["proxy_beta"])
    loss = problem.stress(s)
    score_gradient = torch.autograd.grad(loss, s, retain_graph=True)[0]
    gradient = torch.autograd.grad(loss, theta)[0]
    centered = s.detach()-s.detach().mean(1, keepdim=True)
    relative_radial = (score_gradient*centered).sum(1).abs()/(score_gradient.norm(dim=1)*centered.norm(dim=1)).clamp_min(1e-20)
    direction = gradient-(gradient@theta.detach())/theta.detach().square().sum()*theta.detach()
    norm_fraction = float(direction.norm()/gradient.norm()); direction = direction/direction.norm()
    residual, derivative = torch.autograd.functional.jvp(problem.residuals, theta, direction)
    bounds = torch.full_like(derivative, torch.inf)
    positive, negative = derivative>0, derivative<0
    bounds[positive] = (cfg["margin_fraction"]-residual[positive])/derivative[positive]
    bounds[negative] = (-cfg["margin_fraction"]-residual[negative])/derivative[negative]
    linear_step_bound = float(bounds.min())
    h = cfg["derivative_step"]
    with torch.no_grad():
        fd = float((problem.objective(theta+h*direction)-problem.objective(theta-h*direction))/(2*h))
        directional = float(gradient@direction)
        baseline = problem.metrics(theta); rows = []
        for k in range(cfg["backtracking_steps"]):
            step = cfg["normalized_step"]*.5**k
            trial = theta.detach()+step*direction
            trial *= min(1., problem.radius/float(trial.norm()))
            row = {"step": step, **problem.metrics(trial)}
            row["feasible"] = row["native_critic"] <= cfg["margin_fraction"] and row["parameter_distance"] <= problem.radius*(1+1e-12)
            row["linear_native_prediction"] = float((residual+step*derivative).abs().max())
            row["objective_directional_slope"] = (row["objective"]-baseline["objective"])/step
            rows.append(row)
        invariance = float((angular_scores(.7*s+1.3)-angular_scores(s)).abs().max())
    torch.cuda.synchronize()
    report = {"target": target, "coordinate_form": problem.form, "parameter_variables": len(theta),
              "status": "diagnostic_only", "new_release_weights": 0, "optimizer_updates": 0,
              "reverse_gradient_calls": 2, "native_jvp_calls": 1,
              "gradient_norm": float(gradient.norm()), "tangent_gradient_norm_fraction": norm_fraction,
              "score_gradient_constant_maximum": float(score_gradient.sum(1).abs().max()),
              "score_gradient_radial_relative_maximum": float(relative_radial.max()),
              "affine_invariance_maximum": invariance, "directional_derivative": directional,
              "central_difference": fd, "directional_relative_error": abs(fd-directional)/max(abs(directional), 1e-20),
              "linear_feasible_step_bound": linear_step_bound, "baseline": baseline, "finite_steps": rows,
              "probe_matrix_certificate": problem.provenance, "training_indices": problem.cal["indices"],
              "data_use": "target training fit/scale and fixed target-anchor proxy only; no confirmation, peer or test",
              "parent_diagnostic_sha256": problem.parent_sha, "checkpoint_sha256": problem.hashes,
              "source_sha256": digest(__file__), "protocol_sha256": digest(cfg["protocol_path"]),
              "wall_seconds": time.monotonic()-started, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    assert invariance < 1e-10 and float(relative_radial.max()) < 1e-10
    assert report["directional_relative_error"] < 1e-3
    save_json(output/"direction.json", report)
    print(json.dumps({k: v for k, v in report.items() if k!="training_indices"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--target", required=True)
    parser.add_argument("--protocol", required=True); args = parser.parse_args()
    cfg = json.loads(Path(args.protocol).read_text())
    for f, pin in cfg["source_sha256"].items(): assert digest(f) == pin, f
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cfg["protocol_path"] = args.protocol
    direction_diagnostic(args.target, cfg)
