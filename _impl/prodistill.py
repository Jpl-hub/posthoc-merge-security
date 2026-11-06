"""Pinned ProDistill coefficient kernel; two-expert native CLIP adaptation.

Source: c773fcae, Vision/main_prodistill.py and merge_models.py.
The existing Oct18 adapter is reused; final LayerNorm/projection stay fixed.
"""

import ast
import copy
import hashlib
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from .data import dataset
from .io import digest, save_json


def author(cfg):
    env = dict(torch=torch, nn=nn, np=np, copy=copy)
    for name, ref in cfg["author_files"].items():
        assert digest(ref["path"]) == ref["sha256"]
        tree = ast.parse(Path(ref["path"]).read_text())
        nodes = [
            n
            for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))
            and (
                name == "merge_models"
                or getattr(n, "name", "") == "create_preresblock_activation_extractor"
            )
        ]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), ref["path"], "exec"), env)
    return env


def check(cfg):
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=True)
    env = author(cfg)
    torch.manual_seed(0)
    base = nn.Linear(3, 2).double()
    left = copy.deepcopy(base)
    right = copy.deepcopy(base)
    with torch.no_grad():
        left.weight.add_(0.1)
        right.bias.sub_(0.2)
    wrapper = env["MergedModel"](base, [left, right], "elementwise")
    x = torch.arange(15, dtype=torch.float64).reshape(5, 3) / 10
    expected = F.linear(
        x,
        base.weight + 0.3 * (left.weight - base.weight) + 0.3 * (right.weight - base.weight),
        base.bias + 0.3 * (left.bias - base.bias) + 0.3 * (right.bias - base.bias),
    )
    actual = wrapper(x)
    np.testing.assert_allclose(actual.detach().numpy(), expected.numpy(), atol=1e-12, rtol=1e-12)
    actual.square().sum().backward()
    alpha = wrapper.alphas[0][0]
    gradient = float(alpha.grad[0, 0])
    values = []
    with torch.no_grad():
        for sign in [-1, 1]:
            alpha[0, 0] = 0.3 + sign * 1e-06
            values.append(float(wrapper(x).square().sum()))
    error = abs((values[1] - values[0]) / 2e-06 - gradient)
    assert error < 1e-07
    save_json(
        out / "kernel_validation.json",
        dict(
            status="passed_official_coefficient_formula_and_gradient",
            gradient_error=error,
            source_sha256=digest(__file__),
            author_files=cfg["author_files"],
            images=0,
            expert_updates=0,
        ),
    )


@torch.no_grad()
def initial(anchor, experts):
    result = copy.deepcopy(anchor)
    state = result.state_dict()
    base = anchor.state_dict()
    for expert in experts:
        for key, _ in anchor.named_parameters():
            state[key].add_(0.3 * (expert.state_dict()[key] - base[key]))
    return result


def train(cfg, case, out, anchor, experts, own, peer, env, *, feature_teacher=None):
    """Use own for the target weight direction; its block outputs normally teach the student."""
    if feature_teacher is None:
        feature_teacher = own
    start = time.monotonic()
    torch.manual_seed(0)
    torch.cuda.reset_peak_memory_stats()
    tasks = [case.get("target", "MNIST"), case.get("partner", "SVHN")]
    if "synthetic_source" in case:
        ref = case["synthetic_source"]
        assert digest(ref["path"]) == ref["sha256"]
        raw = torch.load(ref["path"], map_location="cpu", weights_only=False)
        selected = ref["indices"]
        assert len(selected) == len(set(selected)) == 64
        pixels = raw["final"][selected]
        partner_ref = case.get("synthetic_partner_source")
        if partner_ref:
            assert digest(partner_ref["path"]) == partner_ref["sha256"]
            partner_raw = torch.load(partner_ref["path"], map_location="cpu", weights_only=False)
            partner_ids = partner_ref["indices"]
            assert len(partner_ids) == len(set(partner_ids)) == 64
            partner_pixels = partner_raw[partner_ref["key"]][partner_ids]
            del partner_raw
        else:
            partner_ids, partner_pixels = selected, pixels
        pixels = torch.cat((pixels, partner_pixels)).cuda()
        assert pixels.shape == (128, 3, 32, 32) and bool(((pixels >= 0) & (pixels <= 1)).all())
        norm = own.val_preprocess.transforms[-1]
        mean = torch.tensor(norm.mean, device="cuda")[None, :, None, None]
        std = torch.tensor(norm.std, device="cuda")[None, :, None, None]
        images = (F.interpolate(pixels, (224, 224), mode="bilinear", align_corners=False) - mean) / std
        ids = dict(zip(tasks, [selected, partner_ids]))
        del raw, pixels
    else:
        data = {t: dataset(t, "data", experts[t].val_preprocess, True) for t in tasks}
        ids = {
            t: torch.randperm(len(data[t]), generator=torch.Generator().manual_seed(0))[-64:].tolist()
            for t in data
        }
        images = torch.stack([data[t][i][0] for t in ids for i in ids[t]]).cuda()
        del data
    save_json(out / "calibration_indices.json", ids)
    if "calibration_indices" in case:
        assert ids == case["calibration_indices"]
    image_sha = hashlib.sha256(images.cpu().numpy().tobytes()).hexdigest()
    if "calibration_image_sha256" in case:
        assert image_sha == case["calibration_image_sha256"]
    source = torch.tensor([0] * 64 + [1] * 64, device="cuda")
    merged = initial(anchor, [own, peer])

    def prefix(model, images):
        model.cuda()
        with torch.no_grad():
            value = env["create_preresblock_activation_extractor"](model)(images).detach().cpu()
        model.cpu()
        return value

    xm = prefix(merged, images)
    (a, b) = (prefix(feature_teacher, images), prefix(peer, images))
    xt = torch.where(source.cpu()[None, :, None] == 0, a, b)
    del images, a, b
    log = []
    steps = 0
    for i in range(12):
        layers = [
            copy.deepcopy(m.model.visual.transformer.resblocks[i]).cuda().eval()
            for m in [own, peer]
        ]
        base = copy.deepcopy(anchor.model.visual.transformer.resblocks[i]).cuda()
        teacher_layers = list(layers)
        if feature_teacher is not own:
            teacher_layers[0] = copy.deepcopy(feature_teacher.model.visual.transformer.resblocks[i]).cuda().eval()
        wrapper = env["MergedModel"](base, layers, "elementwise")
        wrapper.train()
        params = [p for p in wrapper.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(params, lr=0.01)
        (x, y) = (xm.cuda(), xt.cuda())
        losses = []
        for epoch in range(100):
            order = torch.randperm(128, device="cuda")
            (xx, yy) = (x[:, order], y[:, order])
            s = source[order]
            optimizer.zero_grad()
            student = wrapper.get_merged_model()(xx).permute(1, 0, 2).reshape(128, -1)
            loss = 0
            for j, layer in enumerate(teacher_layers):
                with torch.no_grad():
                    teacher = layer(yy).permute(1, 0, 2).reshape(128, -1)
                loss += (F.mse_loss(student, teacher, reduction="none").sum(1) * s.eq(j)).sum()
            assert torch.isfinite(loss)
            loss.backward()
            optimizer.step()
            steps += 1
            if epoch in [0, 99]:
                losses.append(float(loss.detach()))
        maximum = 0.0
        coefficient_hash = hashlib.sha256()
        state = anchor.model.visual.transformer.resblocks[i].state_dict()
        with torch.no_grad():
            fun = wrapper.get_merged_model()
            for k, name in enumerate(wrapper.names):
                value = fun
                for part in name.split("."):
                    value = getattr(value, part)
                p0 = base.state_dict()[name].cpu().numpy()
                expected = np.zeros_like(p0)
                for j, layer in enumerate(layers):
                    alpha = wrapper.alphas[j][k].detach().cpu().numpy()
                    coefficient_hash.update(alpha.tobytes())
                    expected += alpha * (layer.state_dict()[name].cpu().numpy() - p0)
                expected += p0
                actual = value.detach().cpu()
                np.testing.assert_allclose(actual.numpy(), expected, atol=2e-05, rtol=2e-05)
                maximum = max(maximum, float(np.abs(actual.numpy() - expected).max()))
                state[name] = actual
            merged.model.visual.transformer.resblocks[i].load_state_dict(state, strict=True)
            xm = fun(x).cpu()
            xt = torch.where(source[None, :, None] == 0, teacher_layers[0](y), teacher_layers[1](y)).cpu()
        log.append(
            dict(
                layer=i,
                first_last_loss=losses,
                trainable_coefficients=sum((p.numel() for p in params)),
                coefficient_sha256=coefficient_hash.hexdigest(),
                NumPy_weight_error=maximum,
            )
        )
        print("PRODISTILL_LAYER", case, i, losses, flush=True)
        del wrapper, optimizer, layers, teacher_layers, base, fun, x, y, params
    torch.save(merged.state_dict(), out / "model.pt")
    save_json(
        out / "training.json",
        dict(
            completed=True,
            wall_seconds=time.monotonic() - start,
            backward_steps=steps,
            gradients="Only attacker elementwise merging coefficients; experts and defense checkpoints frozen",
            layers=log,
            model_sha256=digest(out / "model.pt"),
            indices_sha256=digest(out / "calibration_indices.json"),
            source_sha256=digest(__file__),
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
            expert_updates=0,
            defense_updates=0,
            selection_from_test=False,
            image_tensor_sha256=image_sha,
            synthetic_source=case.get("synthetic_source"),
            synthetic_partner_source=case.get("synthetic_partner_source"),
        ),
    )
    return merged
