"""Independent NumPy construction, official-label recount and fresh replay."""

import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat

from clip_benchmark import checkpoint_loader, digest
from covariance_noise import dataset
from feature_noise import capture
from subspace_probe import make_loader, save_json


def official_labels(task, training=False):
    if task == "MNIST":
        name = "train" if training else "t10k"
        n = 60000 if training else 10000
        raw = Path(f"data/MNIST/raw/{name}-labels-idx1-ubyte").read_bytes()
        assert struct.unpack(">II", raw[:8]) == (2049, n)
        return np.frombuffer(raw[8:], dtype=np.uint8).astype(np.int64)
    if task == "DTD":
        root = Path("data/dtd/dtd")
        classes = sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
        names = (root / "labels" / ("train1.txt" if training else "test1.txt")).read_text().splitlines()
        return np.array([classes.index(name.split("/")[0]) for name in names])
    labels = loadmat("data/train_32x32.mat" if training else "data/test_32x32.mat")["y"].reshape(-1)
    return np.where(labels == 10, 0, labels).astype(np.int64)


@torch.no_grad()
def verify(out):
    cfg = json.loads((out / "protocol.json").read_text())
    meta = json.loads((out / "execution.json").read_text())
    case = json.loads((out / "case.json").read_text())
    assert meta["completed"] and meta["new_candidates"] == 1
    assert not (out / "independent_validation.json").exists()
    for filename, key in [("release.pt", "release_sha256"), ("calibration.pt", "calibration_sha256"),
                          ("predictions.pt", "predictions_sha256")]:
        assert digest(out / "artifacts" / filename) == meta[key]
    assert digest("configs/covariance_noise.json") == meta["protocol_sha256"]
    assert cfg == json.loads(Path("configs/covariance_noise.json").read_text())
    assert digest("experiments/covariance_noise.py") == meta["runner_sha256"]
    for path, pin in cfg["source_sha256"].items():
        assert digest(path) == pin
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    target, partner = case["target"], case["partner"]
    load = checkpoint_loader(Path("external/secure-merge-attack"))
    root = Path("checkpoints/ViT-B-32")
    for path, pin in meta["checkpoint_sha256"].items():
        assert digest(root / path) == pin
    anchor = load(str(root / "zeroshot.pt")).cpu().eval()
    original = load(str(root / target / "finetuned.pt")).cpu().eval()
    head = load(str(root / f"head_{target}.pt")).cpu().eval()
    cal = torch.load(out / "artifacts/calibration.pt", map_location="cpu")
    parent = cal
    if "reference" in cal:
        assert digest(cal["reference"]["path"]) == cal["reference"]["sha256"]
        parent = torch.load(cal["reference"]["path"], map_location="cpu")
        assert json.loads(Path(cal["reference"]["path"]).parent.parent.joinpath("independent_validation.json").read_text())["status"] == "passed"
    z, zs, hp, zp = [v.numpy() for v in
                      [parent["fit_hidden"], parent["scale_hidden"], cal["proxy_residual"], cal["proxy_hidden"]]]
    mean, basis = cal["mean"].numpy(), cal["basis"].numpy()
    np.testing.assert_allclose(mean, z.mean(0), rtol=0, atol=1e-11)
    np.testing.assert_allclose(basis.T @ basis, np.eye(basis.shape[1]), rtol=0, atol=1e-9)
    if "reference" not in cal:
        _, singular, vh = np.linalg.svd(z - mean, full_matrices=False)
        fractions = np.cumsum(singular ** 2) / np.sum(singular ** 2)
        rank = int(np.searchsorted(fractions, cfg["retained_energy"])) + 1
        assert rank == basis.shape[1]
        reference_basis = vh[:rank].T
        pca_error = float(np.linalg.norm(basis - reference_basis @ (reference_basis.T @ basis)) / np.sqrt(rank))
        assert pca_error < 1e-6
    else:
        np.testing.assert_array_equal(basis, parent["basis"].numpy())
        pca_error = 0.0

    sa, se = anchor.state_dict(), original.state_dict()
    def proxy_parameter(name):
        return (sa[name] + cfg["proxy_beta"] * (se[name] - sa[name])).double().numpy()
    contrast = head.weight.double().numpy()
    contrast -= contrast.mean(0)
    readout = proxy_parameter("model.visual.ln_post.weight")[:, None] * (
        proxy_parameter("model.visual.proj") @ contrast.T)
    readout -= readout.mean(0)
    q, singular, _ = np.linalg.svd(readout, full_matrices=False)
    q = q[:, singular > singular[0] * cfg["numerical_relative_tolerance"]]
    coeff = (hp - hp.mean(0)) @ q
    values, vectors = np.linalg.eigh(coeff.T @ coeff / len(hp))
    keep = values > values[-1] * cfg["numerical_relative_tolerance"]
    output = q @ vectors[:, keep]
    white = ((hp - hp.mean(0)) @ output / np.sqrt(values[keep])) @ output.T
    incoming = zp - zp.mean(0)
    incoming -= (incoming @ basis) @ basis.T
    cross = white.T @ incoming / len(zp)
    operator = -cross / np.linalg.norm(cross)
    operator_error = float(np.linalg.norm(operator - cal["operator"].numpy()))
    assert operator_error < 1e-7
    alignment_error = abs(float(np.sum(white * ((zp - mean) @ operator.T)) / len(zp)) + np.linalg.norm(cross))
    assert alignment_error < 1e-8
    span_error = float(np.linalg.norm(operator @ basis))
    assert span_error < 1e-8
    rms = max(float(np.sqrt(np.mean(np.sum(((x - mean) @ operator.T) ** 2, axis=1)))) for x in [z, zs])
    h_rms = min(float(np.sqrt(np.mean(np.sum(parent[name].numpy() ** 2, axis=1))))
                for name in ["fit_residual", "scale_residual"])
    layer = original.model.visual.transformer.resblocks[11].mlp.c_proj
    alpha = min(cfg["native_residual_fraction"] * h_rms / rms,
                cfg["weight_relative_cap"] * np.linalg.norm(layer.weight.double().numpy()))
    assert abs(alpha - cal["amplitude"]) < 1e-7
    released = copy.deepcopy(original)
    edits = torch.load(out / "artifacts/release.pt", map_location="cpu")
    prefix = "model.visual.transformer.resblocks.11.mlp.c_proj."
    assert set(edits) == {prefix + "weight", prefix + "bias"}
    expected_w = (layer.weight.double().numpy() + alpha * operator).astype(np.float32)
    expected_b = (layer.bias.double().numpy() - alpha * operator @ mean).astype(np.float32)
    np.testing.assert_allclose(edits[prefix + "weight"].numpy(), expected_w, atol=2e-6, rtol=0)
    np.testing.assert_allclose(edits[prefix + "bias"].numpy(), expected_b, atol=2e-6, rtol=0)
    state = released.state_dict()
    for key, value in edits.items():
        state[key].copy_(value)
    indices = json.loads((out / "training_indices.json").read_text())
    assert len(set(sum(indices.values(), []))) == sum(map(len, indices.values()))
    assert {k: len(v) for k, v in indices.items()} == case["calibration_sizes"]
    labels = {target: official_labels(target)}
    train_labels = official_labels(target, True)
    data = {target: dataset(target, "data", original.val_preprocess)}
    training = dataset(target, "data", original.val_preprocess, True)
    heads = {target: head}
    models = {"original": original, "released": released}
    if meta["fusion_run"]:
        labels[partner] = official_labels(partner)
        data[partner] = dataset(partner, "data", original.val_preprocess)
        heads[partner] = load(str(root / f"head_{partner}.pt")).cpu().eval()
        models["partner"] = load(str(root / partner / "finetuned.pt")).cpu().eval()
        for name, task in [("original", target), ("partner", partner)]:
            ref = cfg["AGA_references"][task]
            assert digest(ref["path"]) == ref["sha256"]
            model = copy.deepcopy(models[name])
            model.load_state_dict(torch.load(ref["path"], map_location="cpu"), strict=True)
            models["attacked/" + name] = model
        assert digest(out / "artifacts/released_AGA_edits.pt") == meta["AGA_edits_sha256"]
        model = copy.deepcopy(models["attacked/original"])
        state = model.state_dict()
        for key, value in torch.load(out / "artifacts/released_AGA_edits.pt", map_location="cpu").items():
            state[key].copy_(value)
        models["attacked/released"] = model

    def fused(preset, stage):
        left = models[preset if stage == "direct" else "attacked/" + preset].state_dict()
        right = models["partner" if stage == "direct" else "attacked/partner"].state_dict()
        state = {k: v + cfg["coefficient"] * ((left[k] - v) + (right[k] - v))
                 if v.is_floating_point() else v.clone() for k, v in sa.items()}
        model = copy.deepcopy(anchor)
        model.load_state_dict(state, strict=True)
        return model

    raw = torch.load(out / "artifacts/predictions.pt", map_location="cpu")
    checks, cached_key, cached_model = [], None, None
    for row in json.loads((out / "results.json").read_text()):
        preset, stage, task = [row[k] for k in ["preset", "stage", "dataset"]]
        key = "/".join([preset, stage, task])
        prediction, y = [raw[key][k].numpy() for k in ["prediction", "target"]]
        truth = train_labels[indices["confirmation"]] if stage == "confirmation" else labels[task]
        np.testing.assert_array_equal(y, truth)
        assert len(y) == row["n"] and int((prediction == y).sum()) == row["correct"]
        assert row["top1"] == row["correct"] / row["n"]
        for c in row["per_class"]:
            selected = y == c["class_id"]
            assert c["n"] == int(selected.sum())
            assert c["correct"] == int((prediction[selected] == y[selected]).sum())
        if preset == "released":
            before = raw["/".join(["original", stage, task])]["prediction"].numpy() == y
            after = prediction == y
            for field, expected in {"lost": (before & ~after).sum(), "gained": (~before & after).sum(),
                                    "both_correct": (before & after).sum(), "both_wrong": (~before & ~after).sum()}.items():
                assert row["paired"][field] == int(expected)
        if stage in ["direct", "AGA"]:
            if cached_key != (preset, stage):
                cached_model = None
                cached_model, cached_key = fused(preset, stage), (preset, stage)
            model = cached_model
        else:
            model = models["partner" if stage == "partner_standalone" else
                           "attacked/" + preset if stage == "AGA_standalone" else preset]
        source = training if stage == "confirmation" else data[task]
        selected = indices["confirmation"][:128] if stage == "confirmation" else range(128)
        images = torch.stack([source[i][0] for i in selected]).cuda()
        model.cuda().eval()
        classifier = heads[task].cuda().eval()
        features = model(images)
        if classifier.normalize:
            features /= features.norm(dim=-1, keepdim=True)
        fresh = torch.nn.functional.linear(features, classifier.weight, classifier.bias).argmax(1).cpu().numpy()
        np.testing.assert_array_equal(fresh, prediction[:len(selected)])
        model.cpu()
        classifier.cpu()
        checks.append(dict(key=key, correct=row["correct"], n=row["n"], replay_n=len(selected)))

    cached_model = None
    confirmation = make_loader(training, indices["confirmation"][:128], 64)
    def proxy_model(model):
        state = {k: v + cfg["proxy_beta"] * (model.state_dict()[k] - v)
                 if v.is_floating_point() else v.clone() for k, v in sa.items()}
        result = copy.deepcopy(anchor)
        result.load_state_dict(state, strict=True)
        return result

    permutation = None
    if meta["fusion_run"]:
        own_fc = original.model.visual.transformer.resblocks[11].mlp.c_fc
        attacked_fc = models["attacked/original"].model.visual.transformer.resblocks[11].mlp.c_fc
        row_bytes = lambda w, b: w.numpy().tobytes() + b.numpy().tobytes()
        rows = {hashlib.sha256(row_bytes(w, b)).digest(): i
                for i, (w, b) in enumerate(zip(own_fc.weight.detach(), own_fc.bias.detach()))}
        permutation = [rows[hashlib.sha256(row_bytes(w, b)).digest()]
                       for w, b in zip(attacked_fc.weight.detach(), attacked_fc.bias.detach())]
        assert len(set(permutation)) == len(permutation)
        np.testing.assert_array_equal(attacked_fc.weight.detach().numpy(), own_fc.weight.detach().numpy()[permutation])
    finite = []
    for context in (["native", "proxy", "direct", "AGA"] if meta["fusion_run"] else ["native", "proxy"]):
        if context == "native":
            before, after = original, released
        elif context == "proxy":
            before, after = proxy_model(original), proxy_model(released)
        else:
            before, after = fused("original", context), fused("released", context)
        keys = {k for k, v in after.state_dict().items() if not torch.equal(v, before.state_dict()[k])}
        assert keys <= {prefix + "weight", prefix + "bias"}
        z_before, h_before = capture(before, confirmation)
        _, h_after = capture(after, confirmation)
        bw, aw = [m.state_dict()[prefix + "weight"].double() for m in [before, after]]
        bb, ab = [m.state_dict()[prefix + "bias"].double() for m in [before, after]]
        expected = z_before @ (aw - bw).T + ab - bb
        actual = h_after - h_before
        error = float((actual - expected).norm() / expected.norm())
        assert error < 1e-3
        v = before.model.visual
        a = v.ln_post.weight.double()[:, None] * (v.proj.double() @ torch.from_numpy(contrast.T.copy()))
        a -= a.mean(0)
        q, singular, _ = torch.linalg.svd(a, full_matrices=False)
        q = q[:, singular > singular[0] * cfg["numerical_relative_tolerance"]]
        f = (h_before - h_before.mean(0)) @ q
        response = (actual - actual.mean(0)) @ q
        denom = f.norm() * response.norm()
        correlation = float((f * response).sum() / denom) if denom > 0 else None
        task_basis = torch.from_numpy(basis[permutation].copy() if context == "AGA" else basis)
        task_mean = torch.from_numpy(mean[permutation].copy() if context == "AGA" else mean)
        outside = z_before - task_mean
        outside -= (outside @ task_basis) @ task_basis.T
        finite.append(dict(context=context, n=len(z_before), relative_error=error,
                           residual_response_rms=float(actual.square().sum(1).mean().sqrt()),
                           complement_input_rms=float(outside.square().sum(1).mean().sqrt()),
                           mean_response_norm=float(actual.mean(0).norm()),
                           classification_span_centered_correlation=correlation))
        if context != "native":
            del before, after

    validation = dict(status="passed", target=target, partner=partner, pca_subspace_error=pca_error,
                      independent_math="NumPy whitening, cross covariance, amplitude and explicit tensor merging",
                      operator_error=operator_error, alignment_error=float(alignment_error), span_error=span_error,
                      recounts=checks, replayed_examples=sum(c["replay_n"] for c in checks),
                      finite_response=finite, AGA_noise_only_down_projection=meta["fusion_run"],
                      verifier_sha256=digest(__file__), raw_predictions_sha256=meta["predictions_sha256"],
                      release_sha256=meta["release_sha256"], optimizer_steps=0, new_candidates=0,
                      public_cleaning_run=False)
    save_json(out / "independent_validation.json", validation)
    print("COVARIANCE_VERIFIED", target, json.dumps(validation), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    verify(Path(args.output))
