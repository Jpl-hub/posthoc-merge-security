"""Weight-only feasibility of joint nonlinear SVD probes; not a release."""
import hashlib
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _impl.io import digest
from _impl.models import initialize, load_checkpoint


def main():
    initialize()
    start = time.monotonic()
    out = Path("outputs/functional_probe_20261025")
    out.mkdir(exist_ok=False)
    expert = load_checkpoint("MNIST/finetuned.pt")
    ref = Path("outputs/phase2_task_subspace_comparison_20261020/basis.pt")
    assert digest(ref) == "99b5f25e7868dd8fd1e9712cf2cca37600f470e8345f5782869e9a0d67d71344"
    basis = torch.load(ref, map_location="cpu", weights_only=False)
    entries = {v["source_key"]: v for v in basis.values()}
    state = expert.state_dict()
    rows = []
    for i in range(12):
        p = f"model.visual.transformer.resblocks.{i}."
        A, B, a, b = [state[p + "mlp." + k].double().cuda() for k in
                      ["c_fc.weight", "c_proj.weight", "c_fc.bias", "c_proj.bias"]]
        ba, bb = [entries[p + "mlp." + k + ".weight"] for k in ["c_fc", "c_proj"]]
        ua, va, vb = [v.cuda() for v in [ba["u"], ba["v"], bb["v"]]]
        columns = torch.cat([ua, A @ va, vb, a[:, None]], dim=1)
        columns = columns[:, columns.norm(dim=0) > 0]
        columns /= columns.norm(dim=0)
        C, s, _ = torch.linalg.svd(columns, full_matrices=False)
        C = C[:, s > s[0] * 1e-10]
        Ap = (2 * C @ (C.T @ A) - A).float().double()
        Bt = (2 * (B @ C) @ C.T - B).float().double()
        r = torch.cat([va.T, -va.T])
        r -= r.mean(1, keepdim=True)
        assert torch.all(r.norm(dim=1) > 1e-10)
        r *= math.sqrt(768) / r.norm(dim=1, keepdim=True)
        gamma, beta = [state[p + "ln_2." + k].double().cuda()
                       for k in ["weight", "bias"]]
        eps = expert.model.visual.transformer.resblocks[i].ln_2.eps
        X = F.layer_norm(r, (768,), gamma, beta, eps)
        phi = lambda z: z * torch.sigmoid(1.702 * z)
        H = phi(F.linear(X, Ap, a)).T
        Y = B @ phi(F.linear(X, A, a)).T
        U, s, vh = torch.linalg.svd(H, full_matrices=False)
        tolerance = torch.finfo(torch.double).eps * max(H.shape) * s[0]
        keep = s > tolerance
        Q = U[:, keep]
        inverse = (vh[keep].T / s[keep]) @ Q.T
        minimal = (Y - B @ H) @ inverse
        correction = (Y - Bt @ H) @ inverse
        affine = B + minimal
        initial = Bt + correction
        free = initial - affine
        residual = float((initial @ H - Y).norm() / (1 + Y.norm()))
        numpy_correction = np.linalg.lstsq(
            H.cpu().numpy().T, (Y - Bt @ H).cpu().numpy().T,
            rcond=np.finfo(float).eps * max(H.shape),
        )[0].T
        relative = float(np.linalg.norm(numpy_correction - correction.cpu().numpy()) /
                         (1 + np.linalg.norm(numpy_correction)))
        assert residual < 1e-10 and relative < 1e-10
        old32 = F.linear(phi(F.linear(X.float(), A.float(), a.float())), B.float(), b.float())
        new32 = F.linear(phi(F.linear(X.float(), Ap.float(), a.float())), initial.float(), b.float())
        delta = initial - B
        row = dict(
            layer=i, probes=len(X), rank=int(keep.sum()),
            condition_number=float(s[0] / s[keep][-1]),
            relative_A_distance=float((Ap - A).norm() / A.norm()),
            minimum_relative_B_change=float(minimal.norm() / B.norm()),
            initial_relative_B_change=float(delta.norm() / B.norm()),
            joint_probe_error_before=float((Bt @ H - Y).norm() / (1 + Y.norm())),
            joint_probe_error_after_FP64=residual,
            joint_probe_error_after_FP32=float((new32.double() - old32.double()).norm() /
                                               (1 + old32.double().norm())),
            maximum_FP32_probe_error=float((new32 - old32).abs().max()),
            B_parameter_task_energy_ratio=float((delta @ vb).square().sum() / delta.square().sum()),
            B_parameter_task_relative_response=float((delta @ vb).norm() / (B @ vb).norm()),
            affine_free_inner_product=float((minimal * free).sum() /
                                            (1 + minimal.norm() * free.norm())),
            free_probe_residual=float((free @ Q).norm() / (1 + free.norm())),
            independent_numpy_correction_relative_error=relative,
        )
        rows.append(row)
        print("FUNCTIONAL_PROBE", i, row["initial_relative_B_change"],
              row["joint_probe_error_after_FP32"], flush=True)
    source = Path(__file__).read_text()
    result = dict(
        status="finite_joint_probe_geometry_only", rows=rows, source=source,
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        basis_sha256=digest(ref), target="MNIST", seconds=time.monotonic() - start,
        real_images=0, synthetic_images=0, local_matrix_probes=12 * 64,
        new_defense_publications=0, model_parameter_updates=0, new_AGA_calls=0,
        scope="Finite local nonlinear SVD inputs only. No full-model, native-task or fusion claim. "
              "Raw B task-direction changes are reported, not called zero. "
              "Standard pseudoinverse/affine projections are not novelty claims.",
    )
    (out / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print("FUNCTIONAL_PROBE_DONE", result["seconds"], flush=True)


if __name__ == "__main__":
    main()
