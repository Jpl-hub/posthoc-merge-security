"""One Data-Free joint A/B construction with nonlinear task response compensation."""
import copy
import json
import time
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.func import functional_call

from _impl.io import digest
from _impl.metrics import logical_parameter_distance
from _impl.models import checked_models, initialize


def response_compensation(Ap, B, a, b, X, Y, Z):
    z = F.linear(X, Ap, a)
    H = (z * torch.sigmoid(1.702 * z)).double().T
    factor = torch.linalg.cholesky(H.T @ H)
    coefficients = torch.cholesky_solve(((Y.double().T - b[:, None]) - B @ H).T, factor).T
    correction = coefficients @ H.T
    free = Z - torch.cholesky_solve((Z @ H).T, factor).T @ H.T
    free = free * (B.norm() / free.norm().clamp_min(1e-30)).clamp(max=1)
    return B + correction + free


def main():
    initialize()
    start = time.monotonic()
    protocol = Path("results/data_free_compensation/response_compensation_protocol.json")
    cfg = json.loads(protocol.read_text())
    assert digest(__file__) == cfg["source_sha256"]
    check_only = "--geometry-only" in sys.argv
    out = Path(cfg["output"] + ("_geometry" if check_only else ""))
    out.mkdir(exist_ok=False)
    (out / "protocol.json").write_text(protocol.read_text())
    for key in ["source", "basis"]:
        assert digest(cfg[key]["path"]) == cfg[key]["sha256"]
    base, expert, head, pins = checked_models(cfg["target"])
    del base
    original = {k: v.clone() for k, v in expert.state_dict().items()}
    published = copy.deepcopy(expert).cuda().eval().requires_grad_(False)
    basis = torch.load(cfg["basis"]["path"], map_location="cpu", weights_only=False)
    entries = {r["source_key"]: r for r in basis.values()}
    variables, geometry, keys = [], [], []
    for i in range(12):
        prefix = f"model.visual.transformer.resblocks.{i}."
        A, B, a, b = [original[prefix + "mlp." + k].double().cuda() for k in
                      ["c_fc.weight", "c_proj.weight", "c_fc.bias", "c_proj.bias"]]
        V = entries[prefix + "mlp.c_fc.weight"]["v"].cuda()
        N = torch.linalg.qr(V, mode="complete").Q[:, 32:]
        variables.extend([
            torch.nn.Parameter(torch.zeros(736, 736, device="cuda")),
            torch.nn.Parameter(torch.zeros_like(B, dtype=torch.float)),
        ])
        keys.extend([prefix + "mlp.c_fc.weight", prefix + "mlp.c_proj.weight"])
        gamma, beta = [original[prefix + "ln_2." + k].double().cuda()
                       for k in ["weight", "bias"]]
        G = V / gamma[:, None]
        _, _, vh = torch.linalg.svd(G.sum(0, keepdim=True), full_matrices=True)
        task_Q = torch.linalg.qr(G @ vh[1:].T, mode="reduced").Q
        sobol = torch.quasirandom.SobolEngine(31, scramble=True, seed=cfg["task_seed"] + i)
        coefficients = torch.erfinv(2 * sobol.draw(cfg["task_probes"]).double() - 1)
        assert torch.isfinite(coefficients).all()
        residual = coefficients.cuda() @ task_Q.T
        residual *= 768 ** .5 / residual.norm(dim=1, keepdim=True)
        eps = published.model.visual.transformer.resblocks[i].ln_2.eps
        X = F.layer_norm(residual, (768,), gamma, beta, eps).float()
        with torch.no_grad():
            z = F.linear(X, A.float(), a.float())
            Y = F.linear(z * torch.sigmoid(1.702 * z), B.float(), b.float())
            assert float((Y - Y.mean(0)).square().mean()) > 0
        geometry.append(dict(A=A, B=B, a=a, b=b, V=V, N=N, AN=A @ N,
                             X=X, Y=Y, task_variance=(Y - Y.mean(0)).square().mean(),
                             I=torch.eye(736, dtype=torch.double, device="cuda")))
    counters = dict(generated_image_forwards=0, input_gradient_model_forwards=0,
                    pixel_optimizer_steps=0, joint_AB_library_iterations=0,
                    full_objective_evaluations=0, local_task_vector_forwards=12 * cfg["task_probes"])

    def pairs():
        result = []
        for T, Z, g in zip(variables[::2], variables[1::2], geometry):
            skew = (T.double() - T.double().T) / 2
            C = torch.linalg.solve(g["I"] - skew, g["I"] + skew)
            K = (C - C.T) / 2
            S = -torch.linalg.solve(g["I"] - K, g["I"] + K)
            Ap = g["A"] + g["AN"] @ (S - g["I"]) @ g["N"].T
            Bp = response_compensation(Ap.float(), g["B"], g["a"].float(),
                                       g["b"], g["X"], g["Y"], Z.double())
            counters["local_task_vector_forwards"] += len(g["X"])
            result.extend([Ap.float(), Bp.float()])
        return result

    @torch.no_grad()
    def materialize():
        for key, value in zip(keys, pairs()):
            published.state_dict()[key].copy_(value)

    if check_only:
        with torch.no_grad():
            matrices = pairs()
            rows = []
            for i, (Ap, Bp, g) in enumerate(zip(matrices[::2], matrices[1::2], geometry)):
                z = F.linear(g["X"], Ap, g["a"].float())
                H = (z * torch.sigmoid(1.702 * z)).double().T
                singular = torch.linalg.svdvals(H)
                actual = F.linear(H.T.float(), Bp, g["b"].float())
                error = float((actual - g["Y"]).norm() / (1 + g["Y"].norm()))
                row = dict(layer=i, rank=int((singular > torch.finfo(torch.double).eps * max(H.shape) * singular[0]).sum()),
                           condition_number=float(singular[0] / singular[-1]),
                           relative_A_change=float((Ap.double() - g["A"]).norm() / g["A"].norm()),
                           relative_B_change=float((Bp.double() - g["B"]).norm() / g["B"].norm()),
                           nonlinear_task_response_relative_error=error)
                assert row["rank"] == cfg["task_probes"] and error < 1e-4
                if i in [0, 11]:
                    import numpy as np
                    hn = H.cpu().numpy()
                    E = (g["Y"].double().T - g["b"][:, None] - g["B"] @ H).cpu().numpy()
                    correction = np.linalg.lstsq(hn.T, E.T, rcond=np.finfo(float).eps * max(hn.shape))[0].T
                    row["numpy_relative_B_change_error"] = abs(float(np.linalg.norm(correction) / float(g["B"].norm())) - row["relative_B_change"])
                    assert row["numpy_relative_B_change_error"] < 1e-6
                rows.append(row)
            result = dict(rows=rows, source_sha256=digest(__file__),source=Path(__file__).read_text(),
                          source_images=0,new_defense_points=0,parameter_updates=0,new_AGA_calls=0,
                          seconds=time.monotonic()-start,task_seed=cfg["task_seed"],checkpoint_sha256=pins,
                          local_task_vector_forwards=counters["local_task_vector_forwards"] + 12 * cfg["task_probes"])
            (out / "results.json").write_text(json.dumps(result,indent=2)+"\n")
            print("GEOMETRY",json.dumps(rows),flush=True)
        return

    norm = expert.val_preprocess.transforms[-1]
    mean = torch.tensor(norm.mean, device="cuda")[None, :, None, None]
    std = torch.tensor(norm.std, device="cuda")[None, :, None, None]
    raw = torch.load(cfg["source"]["path"], map_location="cpu", weights_only=False)
    pixels = raw["final"].clone()
    del raw
    groups = [torch.nn.Parameter(x.cuda()) for x in pixels.split(8)]
    pixel_optimizers = [torch.optim.Adam([x], lr=.05) for x in groups]
    history = [pixels]
    expert.cuda().eval()
    head.cuda().eval()

    def views(x):
        x = F.interpolate(x, (224, 224), mode="bilinear", align_corners=False)
        return (torch.cat([torch.roll(x, tuple(s), (2, 3)) for s in cfg["source"]["views"]]) - mean) / std

    def score(model, x):
        counters["generated_image_forwards"] += len(x)
        if torch.is_grad_enabled() and x.requires_grad:
            counters["input_gradient_model_forwards"] += len(x)
        return head(model(x))

    def KL(student, teacher):
        return 16 * F.kl_div(F.log_softmax(student / 4, dim=1),
                             F.softmax(teacher / 4, dim=1), reduction="batchmean")

    @torch.no_grad()
    def image_pool(x):
        return torch.cat([views(v.cuda()).cpu() for v in x.split(8)])

    pools, rounds, trace = [image_pool(pixels)], [], []
    materialize()
    for cycle in range(3):
        first, last = [], []
        for index, (group, optimizer) in enumerate(zip(groups, pixel_optimizers)):
            for step in range(32):
                x = views(group)
                loss = KL(score(published, x), score(expert, x))
                if step == 0:
                    first.append(float(loss.detach()))
                optimizer.zero_grad(set_to_none=True)
                (-loss).backward()
                assert group.grad is not None and torch.isfinite(group.grad).all()
                optimizer.step()
                counters["pixel_optimizer_steps"] += 1
                with torch.no_grad():
                    group.clamp_(0, 1)
            with torch.no_grad():
                x = views(group)
                last.append(float(KL(score(published, x), score(expert, x))))
            if index % 5 == 0 or index == len(groups) - 1:
                print("INPUT", cycle + 1, index, first[-1], last[-1], flush=True)
        history.append(torch.cat([x.detach().cpu() for x in groups]))
        pools.append(image_pool(history[-1]))
        images = torch.cat(pools)
        batches = list(images.split(40))
        with torch.no_grad():
            targets = [score(expert, x.cuda()).detach() for x in batches]
        optimizer = torch.optim.LBFGS(variables, lr=1., max_iter=40, max_eval=60,
                                      history_size=5, line_search_fn="strong_wolfe")

        def closure():
            optimizer.zero_grad(set_to_none=True)
            actual = pairs()
            leaves = [x.detach().requires_grad_() for x in actual]
            overrides = dict(zip(keys, leaves))
            classification = 0.
            for x, target in zip(batches, targets):
                counters["generated_image_forwards"] += len(x)
                logits = head(functional_call(published, overrides, (x.cuda(),)))
                loss = KL(logits, target) * len(x) / len(images)
                assert torch.isfinite(loss)
                loss.backward()
                classification += float(loss.detach())
            torch.autograd.backward(actual, grad_tensors=[p.grad for p in leaves])
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in variables)
            counters["full_objective_evaluations"] += 1
            row = dict(cycle=cycle + 1, evaluation=counters["full_objective_evaluations"],
                       classification_KL=classification, total=classification)
            trace.append(row)
            (out / "progress.json").write_text(json.dumps(dict(trace=trace, rounds=rounds, counters=counters), indent=2))
            print("JOINT_FIT", json.dumps(row), flush=True)
            return torch.tensor(classification, device="cuda")

        optimizer.step(closure)
        counters["joint_AB_library_iterations"] += int(optimizer.state[variables[0]]["n_iter"])
        materialize()
        with torch.no_grad():
            before = torch.cat([x.cpu() for x in targets])
            after = torch.cat([score(published, x.cuda()).cpu() for x in batches])
        row = dict(cycle=cycle + 1, source_views=len(images),
                   critic_KL_before=sum(first) / len(first), critic_KL_after=sum(last) / len(last),
                   source_KL=float(KL(after, before)),
                   teacher_prediction_changes=int((after.argmax(1) != before.argmax(1)).sum()))
        rounds.append(row)
        print("ROUND", json.dumps(row), flush=True)
        del optimizer
    materialize()
    with torch.no_grad():
        rows = []
        state = published.state_dict()
        for i, g in enumerate(geometry):
            p = f"model.visual.transformer.resblocks.{i}.mlp."
            A, B = [state[p + k].double() for k in ["c_fc.weight", "c_proj.weight"]]
            z = F.linear(g["X"], A.float(), g["a"].float())
            actual = F.linear(z * torch.sigmoid(1.702 * z), B.float(), g["b"].float())
            error = float((actual - g["Y"]).norm() / (1 + g["Y"].norm()))
            counters["local_task_vector_forwards"] += len(z)
            da, db = float((A - g["A"]).norm() / g["A"].norm()), float((B - g["B"]).norm() / g["B"].norm())
            lower = 2 ** .5 * float(g["AN"].norm() / g["A"].norm())
            assert error < 1e-4 and da >= lower - 1e-5
            rows.append(dict(layer=i,relative_A_distance=da,relative_B_distance=db,
                             guaranteed_A_lower_bound=lower,nonlinear_task_response_relative_error=error,
                             task_A_parameter_error=float(((A-g["A"])@g["V"]).norm()/(A-g["A"]).norm())))
        assert all(p.grad is None for p in expert.parameters())
        assert all(torch.equal(v, expert.state_dict()[k].cpu()) for k, v in original.items())
        edits = {key: state[key].cpu().clone() for key in keys}
    torch.save(edits, out / "point.pt")
    torch.save(history, out / "inputs.pt")
    torch.save(dict(before=before, after=after), out / "source_scores.pt")
    result = dict(status="one_nonlinear_response_point_frozen",point_sha256=digest(out / "point.pt"),
                  source_sha256=digest(__file__),source=Path(__file__).read_text(),
                  protocol_sha256=digest(protocol),checkpoint_sha256=pins,
                  parameter_distance=logical_parameter_distance(edits, original),
                  per_layer=rows,rounds=rounds,trace=trace,counters=counters,
                  input_history_sha256=digest(out / "inputs.pt"),source_scores_sha256=digest(out / "source_scores.pt"),
                  original_expert_updates=0,real_construction_images=0,new_AGA_calls=0,
                  seconds=time.monotonic()-start,scope=cfg["scope"])
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("POINT_FROZEN", result["point_sha256"], flush=True)


if __name__ == "__main__":
    main()
