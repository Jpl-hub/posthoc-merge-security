"""Eliminate c_proj by exact conditional block-output distillation."""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import Problem, fresh_confirmation, signed_residual
from clip_benchmark import digest
from subspace_probe import save_json
from task_probe_distance import gelu


class DistilledProblem:
    def __init__(self, target, cfg):
        self.problem = Problem(target, cfg); p = self.problem
        self.solve_calls = 0
        assert p.form == "matrix"
        counts = [len(p.cal[k]["x"]) for k in ["native", "scale"]]
        self.pre = torch.cat([n["pre"][:count] for n, count in zip(p.native, counts)])
        self.inputs = torch.cat([n["task_input"][:count] for n, count in zip(p.native, counts)])
        self.teacher_output = gelu(self.pre)@p.p["B"].T
        q0 = gelu(self.pre)@p.basis["B"]["vh"].T
        self.ridge = cfg["ridge_relative_floor"]*torch.linalg.eigvalsh(q0.T@q0)[-1].detach()
        self.eye = torch.eye(p.r, dtype=torch.float64, device="cuda")
        seed = p.initial[:p.a_size].clone()
        self.initialization_trials = []
        with torch.no_grad():
            if self.feasible(seed): self.initial = seed
            else:
                low, high = 0., 1.
                for _ in range(cfg["initialization_bisection_steps"]):
                    mid = (low+high)/2
                    if self.feasible(mid*seed): low = mid
                    else: high = mid
                self.initial = low*seed
        assert float(self.initial.norm()) > 0, "No nonzero feasible initialization in the fixed seed path"

    def coefficients(self, ca):
        self.solve_calls += 1
        p = self.problem
        pre = self.pre+p.na*(self.inputs@ca.reshape(p.r, p.r).T)@p.basis["A"]["u"].T
        z = gelu(pre); q = z@p.basis["B"]["vh"].T
        residual = self.teacher_output-z@p.p["B"].T
        k = torch.linalg.solve(q.T@q+self.ridge*self.eye, q.T@residual).T
        return torch.cat([ca.flatten(), (k/p.nb).flatten()])

    @torch.no_grad()
    def feasible(self, ca):
        p = self.problem; theta = self.coefficients(ca)
        rho = float(p.residuals(theta).abs().max()); distance = float(theta.norm())
        ok = rho <= p.cfg["margin_fraction"] and distance <= p.radius
        self.initialization_trials.append({"seed_fraction": float(ca.norm()/p.initial[:p.a_size].norm()),
                                           "distance": distance, "native_critic": rho, "feasible": ok})
        return ok

    def stress(self, ca):
        p = self.problem; theta = self.coefficients(ca)
        return p.stress(p.predict(p.proxy, p.pp, theta, p.cfg["proxy_beta"]))

    def residuals(self, ca):
        return self.problem.residuals(self.coefficients(ca))

    def publication_deltas(self, ca, theta):
        p = self.problem
        return {"A": p.na*p.basis["A"]["u"]@ca.reshape(p.r,p.r)@p.basis["A"]["vh"],
                "B": p.nb*theta[p.a_size:].reshape(p.p["B"].shape[0],p.r)@p.basis["B"]["vh"]}

    @torch.no_grad()
    def retract_distance(self, ca):
        """Return a verified point on the zero-to-trial ray, not a closest projection."""
        p = self.problem
        if float(self.coefficients(ca).norm()) <= p.radius: return ca, 0
        low, high = 0., 1.
        for _ in range(p.cfg["distance_retraction_steps"]):
            mid = (low+high)/2
            if float(self.coefficients(mid*ca).norm()) <= p.radius: low = mid
            else: high = mid
        result = low*ca
        assert float(self.coefficients(result).norm()) <= p.radius
        return result, p.cfg["distance_retraction_steps"]

    @torch.no_grad()
    def fit_residual(self, ca):
        p = self.problem; theta = self.coefficients(ca)
        pre = self.pre+p.na*(self.inputs@ca.reshape(p.r, p.r).T)@p.basis["A"]["u"].T
        z = gelu(pre); k = p.nb*theta[p.a_size:].reshape(p.p["B"].shape[0], p.r)
        difference = z@p.p["B"].T+(z@p.basis["B"]["vh"].T)@k.T-self.teacher_output
        return float(difference.norm()/self.teacher_output.norm())


def diagnose(target, cfg, problem_class=DistilledProblem):
    output = Path(cfg["output"]); output.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    coupled = problem_class(target, cfg); p = coupled.problem
    ca = coupled.initial.clone().requires_grad_(True)
    theta = coupled.coefficients(ca); loss = coupled.stress(ca)
    gradient = torch.autograd.grad(loss, ca, retain_graph=True)[0]
    normal = torch.autograd.grad(theta.square().sum(), ca)[0]
    # Study ordering changes tangent to the same physical-distance level set.
    direction = gradient-(gradient@normal)/normal.square().sum().clamp_min(1e-20)*normal
    fraction = float(direction.norm()/gradient.norm())
    _, physical = torch.autograd.functional.jvp(coupled.coefficients, ca, direction)
    direction = direction/physical.norm().clamp_min(1e-20)
    _, physical = torch.autograd.functional.jvp(coupled.coefficients, ca, direction)
    residual, derivative = torch.autograd.functional.jvp(coupled.residuals, ca, direction)
    bounds = torch.full_like(derivative, torch.inf); pos, neg = derivative>0, derivative<0
    bounds[pos] = (cfg["margin_fraction"]-residual[pos])/derivative[pos]
    bounds[neg] = (-cfg["margin_fraction"]-residual[neg])/derivative[neg]
    baseline = p.metrics(theta); rows, finite_thetas = [], []
    h = cfg["derivative_step"]
    with torch.no_grad():
        fd = float((coupled.stress(ca+h*direction)-coupled.stress(ca-h*direction))/(2*h))
        directional = float(gradient@direction)
        for k in range(cfg["backtracking_steps"]):
            step = cfg["normalized_step"]*.5**k
            trial = coupled.coefficients(ca+step*direction)
            row = {"step": step, **p.metrics(trial), "full_output_fit_relative_residual": coupled.fit_residual(ca+step*direction)}
            row["feasible"] = row["native_critic"] <= cfg["margin_fraction"] and row["parameter_distance"] <= p.radius
            row["linear_native_prediction"] = float((residual+step*derivative).abs().max())
            row["angular_directional_slope"] = (row["angular_stress"]-baseline["angular_stress"])/step
            rows.append(row); finite_thetas.append(trial.cpu())
    torch.cuda.synchronize()
    artifact = output/"direction.pt"
    torch.save({"theta": theta.detach().cpu(), "direction": physical.detach().cpu(), "cfg": cfg,
                "native_residual": residual.detach().cpu(), "native_directional_derivative": derivative.detach().cpu(),
                "conditional_A": ca.detach().cpu(), "conditional_direction_A": direction.detach().cpu(),
                "ridge": float(coupled.ridge), "finite_thetas": finite_thetas}, artifact)
    report = {"target": target, "coordinate_form": "matrix", "method": cfg.get("diagnostic_method", "conditional_block_distillation_angular_direction"),
              "optimization_variables": p.a_size, "resolved_publication_coefficients": len(theta),
              "new_release_weights": 0, "optimizer_updates": 0, "explicit_reverse_gradient_calls": 2,
              "coefficient_jvp_calls": 2, "native_jvp_calls": 1, "jvp_method": "reverse-over-reverse; two reverse passes each",
              "gradient_norm": float(gradient.norm()), "distance_tangent_norm_fraction": fraction,
              "physical_direction_norm": float(physical.norm()), "linear_feasible_step_bound": float(bounds.min()),
              "directional_derivative": directional, "central_difference": fd,
              "directional_relative_error": abs(fd-directional)/max(abs(directional), 1e-20),
              "initialization_trials": coupled.initialization_trials, "ridge": float(coupled.ridge),
              "baseline": baseline, "full_output_fit_relative_residual": coupled.fit_residual(ca), "finite_steps": rows,
              "direction_sha256": digest(artifact), "parent_diagnostic_sha256": p.parent_sha,
              "source_sha256": digest(__file__), "protocol_sha256": digest(cfg["protocol_path"]),
              "primary_entrypoint_sha256": digest(cfg.get("entrypoint", __file__)),
              "checkpoint_sha256": p.hashes, "training_indices": p.cal["indices"],
              "confirmation_used": False, "future_partner_used": False, "test_used": False,
              "wall_seconds": time.monotonic()-started, "peak_gpu_bytes": torch.cuda.max_memory_allocated()}
    assert report["directional_relative_error"] < 1e-3
    save_json(output/"direction.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!="training_indices"}), flush=True)


def optimize(target, cfg, problem_class=DistilledProblem):
    output = Path(cfg["output"]); output.mkdir(parents=True, exist_ok=False)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    coupled = problem_class(target, cfg); p = coupled.problem
    ca = coupled.initial.clone().requires_grad_(True); limit = cfg["margin_fraction"]
    initial = p.metrics(coupled.coefficients(ca)); best = (initial["objective"], ca.detach().clone(), 0)
    dual, history, feasible_count = 0., [], 1
    for step in range(cfg["optimization_steps"]):
        theta = coupled.coefficients(ca)
        residual = p.residuals(theta); witness = int(residual.abs().argmax())
        primal = p.objective(theta)-dual*(residual[witness].abs()/limit-1.)
        gradient = torch.autograd.grad(primal, ca, retain_graph=True)[0]
        normal = torch.autograd.grad(theta.square().sum(), ca)[0]
        # Remove only the outward component when the distance bound is active.
        outward = gradient@normal
        direction = gradient
        if float(theta.norm()) >= p.radius*cfg["distance_boundary_fraction"] and float(outward)>0:
            direction = gradient-outward/normal.square().sum().clamp_min(1e-20)*normal
        _, physical = torch.autograd.functional.jvp(coupled.coefficients, ca, direction)
        with torch.no_grad():
            trial = ca+cfg["normalized_step"]*direction/physical.norm().clamp_min(1e-20)
            trial, retraction_steps = coupled.retract_distance(trial)
            trial_theta = coupled.coefficients(trial); metrics = p.metrics(trial_theta)
            dual = max(0., dual+cfg["dual_step"]*(metrics["native_critic"]/limit-1.))
            ca.copy_(trial)
            feasible = metrics["native_critic"] <= limit and metrics["parameter_distance"] <= p.radius
            if feasible:
                feasible_count += 1
                if metrics["objective"] > best[0]: best = (metrics["objective"], ca.detach().clone(), step+1)
        history.append({"step": step+1, "dual_multiplier": dual, "native_witness_index": witness,
                        "feasible": feasible, "gradient_norm": float(gradient.norm()),
                        "physical_direction_norm_before_normalization": float(physical.norm()),
                        "distance_retraction_bisections": retraction_steps, **metrics})
        if (step+1)%8 == 0: print(json.dumps(history[-1]), flush=True)
    ca = best[1]; theta = coupled.coefficients(ca).detach(); final = p.metrics(theta)
    confirmation, labels, split = fresh_confirmation(p, cfg, capture_function=getattr(coupled, "confirmation_capture", None))
    packed = p.prepare(confirmation, p.p, True)
    with torch.no_grad():
        after = p.predict(packed, p.p, theta)
        confirmed_rho = float(signed_residual(packed["teacher"], after).abs().max())
        before_real = packed["teacher"][:len(labels)].argmax(1).cpu()
        after_real = after[:len(labels)].argmax(1).cpu()
        deltas = coupled.publication_deltas(ca, theta)
    artifact = output/"candidate.pt"
    torch.save({**p.cal, "theta": theta.cpu(), "release_deltas": {k:v.cpu() for k,v in deltas.items()},
                "protocol": cfg, "confirmation": confirmation, "confirmation_indices": split["indices"],
                "confirmation_labels": labels, "confirmation_original_predictions": before_real,
                "confirmation_candidate_predictions": after_real, "parent_diagnostic_sha256": p.parent_sha}, artifact)
    torch.cuda.synchronize()
    fit_residual = coupled.fit_residual(ca)
    result = {"target": target, "method": cfg.get("optimization_method", "conditional_composite_distillation_primal_dual"),
              "optimization_variables": p.a_size, "publication_coefficients": len(theta),
              "initial": initial, "selected": final, "selected_step": best[2],
              "optimizer_updates": cfg["optimization_steps"], "explicit_reverse_gradient_calls": 2*cfg["optimization_steps"],
              "coefficient_jvp_calls": cfg["optimization_steps"], "jvp_method": "two internal reverse passes per JVP",
              "dual_updates": cfg["optimization_steps"], "conditional_linear_solve_calls": coupled.solve_calls,
              "feasible_training_iterates": feasible_count,
              "temporarily_infeasible_optimization_iterates": sum(not r["feasible"] for r in history),
              "distance_retraction_bisections": sum(r["distance_retraction_bisections"] for r in history),
              "original_expert_parameters_updated": False, "gold_labels_used_for_optimization": False,
              "future_partner_used": False, "test_used": False, "new_release_weights": 0,
              "full_output_fit_relative_residual": fit_residual,
              "confirmation": {"n": len(labels), "original_correct": int((before_real==labels).sum()),
                               "candidate_correct": int((after_real==labels).sum()),
                               "changed_predictions": int((before_real!=after_real).sum()), "worst_margin_drift": confirmed_rho},
              "confirmation_passed": confirmed_rho <= limit,
              "candidate_sha256": digest(artifact), "parent_diagnostic_sha256": p.parent_sha,
              "source_sha256": digest(__file__), "protocol_sha256": digest(cfg["protocol_path"]),
              "primary_entrypoint_sha256": digest(cfg.get("entrypoint", __file__)),
              "checkpoint_sha256": p.hashes, "initialization_trials": coupled.initialization_trials,
              "wall_seconds": time.monotonic()-started, "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
              "global_saddle_optimum_claim": False}
    save_json(output/"results.json", result); save_json(output/"history.json", history)
    save_json(output/"confirmation_split.json", split); save_json(output/"protocol.json", cfg)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--target", required=True)
    parser.add_argument("--protocol", required=True); args = parser.parse_args()
    cfg = json.loads(Path(args.protocol).read_text())
    for f, pin in cfg["source_sha256"].items(): assert digest(f) == pin, f
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    cfg["protocol_path"] = args.protocol
    if cfg.get("mode") == "optimize": optimize(args.target, cfg)
    else: diagnose(args.target, cfg)
