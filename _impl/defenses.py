"""Post-hoc bias protection, numerical compensation and permutation controls."""

import copy
import json
from pathlib import Path

import torch

from .data import make_loader
from .io import digest
from .merge import merge


def compensate_after_alignment(expert, recovered, anchor, basis, cfg):
    """Finite O rotations erased by ridge alignment, with joint nonlinear B compensation."""
    import torch.nn.functional as F
    from torch.func import functional_call

    from .attacks import load_author
    from .merge import parameter_hash

    visual, attacked = expert.model.visual, recovered.model.visual
    layer = cfg["layer"]
    block, attacked_block = [v.transformer.resblocks[layer] for v in [visual, attacked]]
    O, B = block.attn.out_proj.weight.detach(), block.mlp.c_proj.weight.detach()
    key = f"model.visual.transformer.resblocks.{layer}."
    entries = {v["source_key"]: v for v in basis.values()}
    vo = entries[key + "attn.out_proj.weight"]["v"].double().cuda()
    vb = entries[key + "mlp.c_proj.weight"]["v"].float().cuda()
    eye = torch.eye(64, dtype=torch.double, device="cuda")
    projectors = []
    for v in vo.split(64):
        u, s, _ = torch.linalg.svd(v, full_matrices=False)
        u = u[:, s > s[0] * 1e-10]
        projectors.append(eye - u @ u.T)
    P = torch.stack(projectors)
    R0 = eye - 2 * P
    heads = O.reshape(768, 12, 64).permute(1, 0, 2).double()
    z = torch.nn.Parameter(torch.zeros_like(P))
    delta = torch.nn.Parameter(torch.zeros_like(B))
    norm = B.norm()
    author, _ = load_author(cfg["author_source"])
    anchor_block = anchor.model.visual.transformer.resblocks[layer]
    perm = author["solve_permutation"](block.mlp.c_fc.weight,
                                       anchor_block.mlp.c_fc.weight.cuda())
    torch.testing.assert_close(attacked_block.mlp.c_fc.weight,
                               block.mlp.c_fc.weight[perm], rtol=0, atol=0)
    mean = O.new_tensor([.48145466, .4578275, .40821073])[None, :, None, None]
    std = O.new_tensor([.26862954, .26130258, .27577711])[None, :, None, None]

    def weights():
        R = R0 @ torch.matrix_exp(P @ (z - z.transpose(1, 2)) @ P)
        op = (heads @ R).permute(1, 0, 2).reshape(768, 768).float()
        db = delta - (delta @ vb) @ vb.T
        return op, B + norm * db, R

    def pixels(gen, step):
        side = 8 if step % 2 == 0 else 224
        x = torch.rand(cfg["batch_size"], 3, side, side, generator=gen, device="cuda")
        return F.interpolate(x, size=224, mode="bilinear", align_corners=False)

    def prefix(v, x):
        h = v.conv1((x-mean)/std).flatten(2).permute(0, 2, 1)
        cls = v.class_embedding[None, None].expand(h.shape[0], 1, -1)
        h = v.ln_pre(torch.cat([cls, h], 1)+v.positional_embedding).permute(1, 0, 2)
        for b in v.transformer.resblocks[:layer]:
            h = b(h)
        return h

    def finish(v, b, h, edited=None):
        h = functional_call(b, edited, (h,)) if edited else b(h)
        return F.normalize(v.ln_post(h[0]) @ v.proj, dim=-1)

    def native(x, input_gradient=False):
        with torch.enable_grad() if input_gradient else torch.no_grad():
            h = prefix(visual, x)
            ref = finish(visual, block, h)
        op, bp, _ = weights()
        out = finish(visual, block, h, {"attn.out_proj.weight": op, "mlp.c_proj.weight": bp})
        return (out-ref).square().sum(-1)

    gen = torch.Generator(device="cuda").manual_seed(cfg["seed"])
    x = pixels(gen, 0)
    with torch.no_grad():
        control_error = max(float((finish(v, b, prefix(v, x))-
                                   F.normalize(v((x-mean)/std), dim=-1)).abs().max())
                            for v, b in [(visual, block), (attacked, attacked_block)])
    assert control_error < 1e-5
    # One constrained ridge initialization uses generated pixels and the actual
    # modified Attention/LN input. It does not fit real training examples.
    gram = torch.zeros(3072, 3072, dtype=torch.double, device="cuda")
    cross = torch.zeros(3072, 768, dtype=torch.double, device="cuda")
    buffer = []
    hook = block.mlp.gelu.register_forward_hook(lambda _m, _a, out: buffer.append(out.detach()))
    try:
        with torch.no_grad():
            op, _, _ = weights()
            for step in range(cfg["source_n"] // cfg["batch_size"]):
                h = prefix(visual, pixels(gen, step))
                ref = block(h)
                buffer.clear()
                out = functional_call(block, {"attn.out_proj.weight": op}, (h,))
                features = buffer.pop().flatten(0, 1).double()
                features -= (features @ vb.double()) @ vb.double().T
                error = (ref-out).flatten(0, 1).double()
                gram += features.T @ features
                cross += features.T @ error
            ridge = cfg["damping"] * gram.diagonal().mean()
            system = gram + ridge * torch.eye(3072, dtype=torch.double, device="cuda")
            solution = torch.linalg.solve(system, cross)
            solve_error = float((system @ solution-cross).norm()/cross.norm())
            assert solve_error < 1e-8
            solution -= vb.double() @ (vb.double().T @ solution)
            delta.copy_((solution.T/norm).float())
            unclipped = float(delta.norm())
            delta.mul_(min(1., cfg["B_bound"] / max(unclipped, 1e-12)))
            init = dict(unclipped_B_distance=unclipped, B_distance=float(delta.norm()),
                        solve_relative_error=solve_error, damping=float(ridge))
    finally:
        hook.remove()
    del gram, cross, system, solution
    optimizer = torch.optim.Adam([dict(params=[z], lr=cfg["rotation_lr"]),
                                  dict(params=[delta], lr=cfg["B_lr"])])
    trace = []
    for step in range(cfg["steps"]):
        x = pixels(gen, step)
        for _ in range(cfg["input_steps"]):
            x.requires_grad_(True)
            grad, = torch.autograd.grad(native(x, True).mean(), x)
            x = (x + cfg["input_step_size"] * grad.sign()).detach().clamp(0, 1)
        optimizer.zero_grad(set_to_none=True)
        keep = native(x).mean()
        with torch.no_grad():
            h = prefix(attacked, x)
            ref = finish(attacked, attacked_block, h)
        op, bp, _ = weights()
        rec = finish(attacked, attacked_block, h, {"mlp.c_proj.weight": bp[:, perm]})
        separate = (rec-ref).square().sum(-1).mean()
        distance = ((op-O)/O.norm()).square().sum() + delta.square().sum()
        loss = cfg["native_weight"]*keep - separate - cfg["distance_weight"]*distance
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite objective at step{step}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            delta.sub_((delta @ vb) @ vb.T)
            delta.mul_(min(1., cfg["B_bound"] / max(float(delta.norm()), 1e-12)))
        if step % 32 == 0 or step + 1 == cfg["steps"]:
            row = dict(step=step+1, native_unit_RMS=float(keep.sqrt()),
                       recovered_unit_RMS=float(separate.sqrt()),
                       O_distance=float((op-O).norm()/O.norm()), B_distance=float(delta.norm()))
            trace.append(row)
            print("COMPENSATION", json.dumps(row), flush=True)
    with torch.no_grad():
        op, bp, R = weights()
        orthogonal = float((R.transpose(1, 2) @ R-eye).abs().max())
        task_error = [float(((w-ref).double() @ v).norm()/ref.double().norm())
                      for w, ref, v in [(op, O, vo), (bp, B, vb.double())]]
        recovery_error = 0.
        for head in range(12):
            x, y = op[:, 64*head:64*(head+1)], anchor_block.attn.out_proj.weight[:, 64*head:64*(head+1)].cuda()
            actual = x @ author["solve_linear_transformation"](x, y)
            recovery_error = max(recovery_error, float(
                (actual-attacked_block.attn.out_proj.weight[:, 64*head:64*(head+1)]).abs().max()))
        assert orthogonal < 1e-10 and max(task_error) < 1e-6 and recovery_error < 2e-5
        edits = {key+"attn.out_proj.weight": op.cpu(), key+"mlp.c_proj.weight": bp.cpu()}
        frozen_hash = parameter_hash(edits)
        held = torch.Generator(device="cuda").manual_seed(cfg["confirmation_seed"])
        errors = []
        for k in range(cfg["confirmation_batches"]):
            errors.extend(native(pixels(held, k)).cpu().tolist())
    return edits, dict(initialization=init, trace=trace, orthogonality_max_error=orthogonal,
                       explicit_visual_max_error=control_error,
                       task_relative_errors=task_error, originalAGA_O_max_error=recovery_error,
                       heldout_native_unit_RMS=(sum(errors)/len(errors))**.5,
                       heldout_max_unit_error=max(errors)**.5, heldout_n=len(errors),
                       parameter_content_sha256_before_confirmation=frozen_hash,
                       gradient_steps=cfg["steps"], real_construction_images=0,
                       teacher_training_updates=0, final_O_B_distances=[float((op-O).norm()/O.norm()), float(delta.norm())])


@torch.no_grad()
def _fit_nonlinear_mlp_layer(A, B, a, ba, bb, X, target, phi, cfg):
    """Fixed task-complement change and one finite output reconstruction."""
    import torch.nn.functional as F

    ua, va, vb = ba["u"].cuda(), ba["v"].cuda(), bb["v"].cuda()
    reference_va, reference_vb = va, vb
    if not cfg.get("task_constraints", True):
        ua, va, vb = ua[:, :0], va[:, :0], vb[:, :0]
    columns = torch.cat([ua, A @ va, vb, a[:, None]], dim=1)
    columns = columns[:, columns.norm(dim=0) > 0]
    columns /= columns.norm(dim=0)
    C, singular, _ = torch.linalg.svd(columns, full_matrices=False)
    C = C[:, singular > singular[0] * 1e-10]
    Ap = (2 * C @ (C.T @ A) - A).float()
    Bt = (2 * (B @ C) @ C.T - B).float()

    width = B.shape[1]
    gram = torch.zeros(width, width, dtype=torch.double, device="cuda")
    cross = torch.zeros(width, B.shape[0], dtype=torch.double, device="cuda")
    error_energy = torch.zeros((), dtype=torch.double, device="cuda")
    target_energy = error_energy.clone()
    for x, y in zip(X.flatten(0, 1).split(4096), target.flatten(0, 1).split(4096)):
        x = x.cuda()
        hp = phi(F.linear(x, Ap, a.float()))
        y = y.cuda()
        error = (y - F.linear(hp, Bt)).double()
        error_energy += error.square().sum()
        target_energy += y.double().square().sum()
        hp = hp.double()
        z = hp - (hp @ vb) @ vb.T
        gram += z.T @ z
        cross += z.T @ error
    damping = cfg["damping"] * gram.diagonal().mean()
    if not float(damping) > 0:
        raise ValueError("No active complementary calibration directions")
    system = gram + damping * torch.eye(width, dtype=torch.double, device="cuda")
    delta = torch.linalg.solve(system, cross)
    solve_error = float((system @ delta - cross).norm() / cross.norm())
    if not solve_error < 1e-8:
        raise ValueError(f"Ridge solve residual: {solve_error}")
    delta -= vb @ (vb.T @ delta)
    fitted_error = error_energy - 2 * (delta * cross).sum() + (delta * (gram @ delta)).sum()
    Bp = (Bt.double() + delta.T).float()
    da, db = Ap.double() - A, Bp.double() - B
    distances = [float(da.norm() / A.norm()), float(db.norm() / B.norm())]
    details = dict(
        task_rank=reference_va.shape[1],
        task_constraints_enabled=cfg.get("task_constraints", True),
        fixed_hidden_dimension=C.shape[1],
        relative_A_distance=distances[0],
        relative_B_distance=distances[1],
        logical_distance=sum(v * v for v in distances) ** 0.5,
        task_A_response_relative=float((da @ reference_va).norm() / da.norm()),
        task_B_response_relative=float((db @ reference_vb).norm() / db.norm()),
        ridge_damping=float(damping),
        ridge_solve_residual=solve_error,
        output_relative_RMS_before_fit=float((error_energy / target_energy).sqrt()),
        output_relative_RMS_after_ridge=float((fitted_error.clamp_min(0) / target_energy).sqrt()),
    )
    details["parameter_range_passed"] = all(
        d <= limit + 1e-6 for d, limit in zip(distances, cfg["matrix_bounds"])
    )
    return Ap.cpu(), Bp.cpu(), details


@torch.no_grad()
def nonlinear_mlp_compensation(expert, head, training, indices, basis, cfg, anchor=None):
    """Reconstruct teacher block outputs using the actual modified prefix.

    The teacher is frozen. Only publication A/B tensors are computed by linear
    solves; all LayerNorm and Attention responses come from complete forwards.
    """
    from .data import official_labels
    from .evaluate import evaluate
    from .metrics import logical_parameter_distance, task_change

    if cfg.get("pre_ln_compensation") or cfg.get("local_teacher"):
        raise ValueError("Closed reconstruction controls: reproduce from Git ad5800c")
    layers = cfg["layers"]
    loader = make_loader(training, indices)
    blocks = expert.model.visual.transformer.resblocks
    teacher_outputs = {i: [] for i in layers}
    teacher_residuals = {i: [] for i in layers}
    hooks = [
        blocks[i].register_forward_hook(
            lambda _m, _args, output, i=i: teacher_outputs[i].append(
                output.detach().permute(1, 0, 2).cpu()
            )
        )
        for i in layers
    ]
    if not cfg.get("prefix_compensation", True):
        hooks += [
            blocks[i].ln_2.register_forward_pre_hook(
                lambda _m, args, i=i: teacher_residuals[i].append(
                    args[0].detach().permute(1, 0, 2).cpu()
                )
            )
            for i in layers
        ]
    original_scores, changed_scores = [], []
    try:
        evaluate(expert, head, loader, "cuda", logit_sink=original_scores.append)
    finally:
        for hook in hooks:
            hook.remove()
    published = copy.deepcopy(expert).cpu().eval()
    state = expert.state_dict()
    entries = {v["source_key"]: v for v in basis.values()}
    edits, per_layer = {}, []
    if cfg.get("public_backbone", False):
        keep = {"model.visual.proj", "model.visual.ln_post.weight", "model.visual.ln_post.bias"}
        edits = {
            k: v
            for k, v in anchor.state_dict().items()
            if ".mlp." not in k and k not in keep and not torch.equal(v, state[k])
        }
        published.load_state_dict(state | edits, strict=True)
    fixed_reset_distance = logical_parameter_distance(edits, state)
    fixed_reset_tensors = len(edits)
    forward_count = len(indices)
    for i in layers:
        block = published.model.visual.transformer.resblocks[i]
        inputs, residuals = [], []
        hooks = [
            block.mlp.c_fc.register_forward_pre_hook(
                lambda _m, args: inputs.append(args[0].detach().permute(1, 0, 2).cpu())
            ),
            block.ln_2.register_forward_pre_hook(
                lambda _m, args: residuals.append(args[0].detach().permute(1, 0, 2).cpu())
            ),
        ]
        try:
            evaluate(published, head, loader, "cuda")
            forward_count += len(indices)
        finally:
            for hook in hooks:
                hook.remove()
        prefix = f"model.visual.transformer.resblocks.{i}.mlp."
        A, B = [state[prefix + k + ".weight"].double().cuda() for k in ["c_fc", "c_proj"]]
        a = state[prefix + "c_fc.bias"].double().cuda()
        b = state[prefix + "c_proj.bias"]
        X = torch.cat(inputs)
        reference = (
            torch.cat(residuals)
            if cfg.get("prefix_compensation", True)
            else torch.cat(teacher_residuals.pop(i))
        )
        target = torch.cat(teacher_outputs.pop(i)) - reference - b
        ba, bb = [entries[prefix + k + ".weight"] for k in ["c_fc", "c_proj"]]
        Ap, Bp, row = _fit_nonlinear_mlp_layer(A, B, a, ba, bb, X, target, block.mlp.gelu, cfg)
        row["layer"] = i
        per_layer.append(row)
        changed = {prefix + "c_fc.weight": Ap, prefix + "c_proj.weight": Bp}
        for key, value in changed.items():
            published.state_dict()[key].copy_(value)
        edits.update(changed)
        print("NONLINEAR_LAYER", i, row, flush=True)
        if not row["parameter_range_passed"] and not cfg.get("ablation", False):
            return (
                edits,
                dict(indices=indices),
                dict(
                    source_passed=False,
                    stop_reason="Declared matrix range exceeded",
                    per_layer=per_layer,
                    image_forwards=forward_count,
                ),
            )
    evaluate(published, head, loader, "cuda", logit_sink=changed_scores.append)
    forward_count += len(indices)
    before, after = [torch.cat(v).double().numpy() for v in [original_scores, changed_scores]]
    labels = official_labels(cfg["target"], True)[indices]
    details = dict(
        per_layer=per_layer,
        logical_distance=sum(r["logical_distance"] ** 2 for r in per_layer) ** 0.5,
        gradient_steps=0,
        expert_training_updates=0,
        linear_solves=len(per_layer),
        image_forwards=forward_count,
        finite_function_identity=False,
        parameter_range_passed=all(r["parameter_range_passed"] for r in per_layer),
    )
    if cfg.get("public_backbone", False):
        details.update(
            logical_distance=logical_parameter_distance(edits, state),
            fixed_public_reset_distance=fixed_reset_distance,
            fixed_public_reset_tensors=fixed_reset_tensors,
            task_probe_scope="MLP A/B constraints retained;other task updates explicitly removed",
        )
    details["source"] = task_change(before, after)
    details["source"].update(
        before_correct=int((before.argmax(1) == labels).sum()),
        after_correct=int((after.argmax(1) == labels).sum()),
    )
    details["source"]["accuracy_loss_pp"] = (
        100
        * (details["source"]["before_correct"] - details["source"]["after_correct"])
        / len(indices)
    )
    details["source_passed"] = details["source"]["accuracy_loss_pp"] <= cfg["source_loss_pp"]
    calibration = dict(indices=indices, before=before, after=after)
    return edits, calibration, details


def residual_permutation(state, order):
    """Known global residual-coordinate symmetry; keep QKV head outputs fixed.

    Native LayerNorm gains/biases move with their coordinates. This deliberately
    differs from rotating a native learned-gain LayerNorm without rewriting it.
    """
    edits = {}
    for k, v in state.items():
        if not k.startswith("model.visual."):
            continue
        name = k.removeprefix("model.visual.")
        if name in ["conv1.weight", "class_embedding", "proj"]:
            axis = 0
        elif name == "positional_embedding" or name.endswith(
            ("attn.in_proj_weight", "mlp.c_fc.weight")
        ):
            axis = 1
        elif name.endswith(
            ("attn.out_proj.weight", "attn.out_proj.bias", "mlp.c_proj.weight", "mlp.c_proj.bias")
        ):
            axis = 0
        elif any(x in name for x in ["ln_pre.", "ln_post.", ".ln_1.", ".ln_2."]):
            axis = 0
        else:
            assert name.endswith(("attn.in_proj_bias", "mlp.c_fc.bias")), name
            continue
        assert v.shape[axis] == len(order), (k, v.shape)
        changed = v.index_select(axis, order).contiguous()
        if not torch.equal(changed, v):
            edits[k] = changed
    return edits


def classify(h, params):
    centered = h - h.mean(1, keepdim=True)
    sigma = (centered.square().mean(1, keepdim=True) + params["eps"]).sqrt()
    t = (centered / sigma * params["gamma"] + params["beta"]) @ params["projection"]
    f = t / t.norm(dim=1, keepdim=True)
    return f @ params["head"].T, t, f


def parameters(model, head):
    v = model.model.visual
    return dict(
        gamma=v.ln_post.weight.double(),
        beta=v.ln_post.bias.double(),
        eps=v.ln_post.eps,
        projection=v.proj.double(),
        head=head.weight.double(),
    )


@torch.no_grad()
def capture(model, loader):
    """Actual post-QuickGELU CLS input and final residual, not preactivation PCA."""
    hidden, residual = [], []
    block = model.model.visual.transformer.resblocks[11]
    handles = [
        block.mlp.c_proj.register_forward_pre_hook(
            lambda _m, args: hidden.append(args[0][0].detach().cpu())
        ),
        model.model.visual.ln_post.register_forward_pre_hook(
            lambda _m, args: residual.append(args[0].detach().cpu())
        ),
    ]
    try:
        model.cuda().eval()
        for images, _ in loader:
            model(images.cuda(non_blocking=True))
    finally:
        for handle in handles:
            handle.remove()
        model.cpu()
    return torch.cat(hidden).double(), torch.cat(residual).double()


@torch.no_grad()
def construct_bias(original, anchor, head, training, indices, cfg, target):
    cache_ref = cfg.get("native_cache", {}).get(target)
    cache_metadata = {}
    if cache_ref:
        ref = Path(cache_ref)
        meta = json.loads((ref / "execution.json").read_text())
        assert json.loads((ref / "independent_validation.json").read_text())["status"] == "passed"
        assert indices == json.loads((ref / "training_indices.json").read_text())
        assert digest(ref / "artifacts/calibration.pt") == meta["calibration_sha256"]
        cache = torch.load(ref / "artifacts/calibration.pt", map_location="cpu")
        proxy_residual = cache["proxy_residual"]
        cache_metadata = dict(
            cache_path=str(ref / "artifacts/calibration.pt"),
            cache_sha256=meta["calibration_sha256"],
        )
    projection = original.model.visual.proj.double()
    u, s, _ = torch.linalg.svd(projection, full_matrices=True)
    rank = int((s > s[0] * cfg["numerical_relative_tolerance"]).sum())
    assert rank == projection.shape[1] and projection.shape[0] > rank
    null = u[:, rank:]
    proxy = merge(anchor, original, anchor, cfg["proxy_beta"])
    if not cache_ref:
        _, proxy_residual = capture(proxy, make_loader(training, indices["fit"], 64))
        cache_metadata = dict(
            proxy_residual=proxy_residual, calibration_source="fresh_target_training"
        )
    params = parameters(proxy, head)
    contrast = head.weight.double() - head.weight.double().mean(0)
    leakage = null.T @ params["projection"] @ contrast.T
    left, values, right = torch.linalg.svd(leakage, full_matrices=False)
    keep = values > values[0] * cfg["numerical_relative_tolerance"]
    assert keep.any()
    generator = torch.Generator().manual_seed(cfg["noise_seed"])
    requested_class = int(torch.randint(len(contrast), (1,), generator=generator))
    desired = torch.zeros(len(contrast), dtype=torch.float64)
    desired[requested_class] = 1
    desired -= desired.mean()
    direction = null @ (left[:, keep] @ ((right[keep] @ desired) / values[keep]))
    assert float((direction @ projection).norm()) < 1e-9
    class_bias = direction @ params["projection"] @ contrast.T
    collapse_class = int(class_bias.argmax())
    rivals = torch.arange(len(contrast)) != collapse_class
    gaps = class_bias[collapse_class] - class_bias[rivals]
    assert torch.all(gaps > 1e-10), "No unique proxy class bias; stop rather than reseed"
    _, features, _ = classify(proxy_residual, params)
    logits = features @ head.weight.double().T
    required = (logits[:, rivals] - logits[:, collapse_class, None]) / (cfg["proxy_beta"] * gaps)
    first_bound = max(0.0, float(required.max()))
    assert first_bound > 0, "Proxy already collapsed; not a useful candidate"
    amplitude = cfg["bias_safety"] * first_bound
    original_bias = original.model.visual.ln_post.bias
    delta = amplitude * direction
    released = copy.deepcopy(original).cpu().eval()
    released.model.visual.ln_post.bias.copy_((original_bias.double() + delta).float())
    key = "model.visual.ln_post.bias"
    changed = {
        k
        for k in original.state_dict()
        if not torch.equal(original.state_dict()[k], released.state_dict()[k])
    }
    assert changed == {key}
    actual = released.state_dict()[key].double() - original_bias.double()
    after_params = params.copy()
    beta0 = anchor.model.visual.ln_post.bias
    after_params["beta"] = (
        beta0 + cfg["proxy_beta"] * (released.model.visual.ln_post.bias - beta0)
    ).double()
    protected_scores = classify(proxy_residual, after_params)[0]
    assert torch.all(protected_scores.argmax(1).eq(collapse_class))
    labels_all = getattr(
        training, "targets", getattr(training, "_labels", getattr(training, "labels", None))
    )
    labels = torch.tensor([int(labels_all[i]) for i in indices["fit"]])
    before = classify(proxy_residual, params)[0].argmax(1)
    after = protected_scores.argmax(1)
    per_class = [
        dict(
            class_id=k,
            n=int(labels.eq(k).sum()),
            original_correct=int((before.eq(labels) & labels.eq(k)).sum()),
            protected_correct=int((after.eq(labels) & labels.eq(k)).sum()),
        )
        for k in range(len(contrast))
    ]
    cal = dict(
        **cache_metadata,
        null_basis=null,
        projection_rank=rank,
        direction=direction,
        desired=desired,
        requested_class=requested_class,
        collapse_class=collapse_class,
        class_bias=class_bias,
        first_bound=first_bound,
        amplitude=amplitude,
        delta=delta,
    )
    geometry = dict(
        noise_family="projection_bias",
        edited_key=key,
        projection_rank=rank,
        exact_complement_dimension=null.shape[1],
        noise_seed=cfg["noise_seed"],
        requested_class=requested_class,
        proxy_collapse_class=collapse_class,
        amplitude=amplitude,
        delta_norm=float(actual.norm()),
        ideal_embedding_null_error=float((direction @ projection).norm()),
        fp32_embedding_constant_error=float((actual @ projection).norm()),
        partner_used=False,
        labels_used_in_direction=False,
        labels_used_in_amplitude=False,
        optimizer_steps=0,
        parameter_search=False,
        proxy_fit=dict(
            n=len(labels),
            original_correct=int(before.eq(labels).sum()),
            protected_correct=int(after.eq(labels).sum()),
            per_class=per_class,
        ),
        limitation="Author AGA does not edit final LN beta or visual projection. No claim of resistance to public nullspace cleanup or parameter replacement.",
    )
    return released, {key: released.state_dict()[key].clone()}, cal, geometry
