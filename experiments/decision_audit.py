"""Audit frozen noise on training splits; no new protection or attack."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import checkpoint_loader, digest, merge
from covariance_noise import dataset
from feature_noise import capture
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels


PREFIX = "model.visual.transformer.resblocks.11.mlp.c_proj."


def quantiles(x):
    x = x.detach().double().cpu().flatten()
    return {str(q): float(torch.quantile(x, q)) for q in [0.05, 0.5, 0.95]}


def classify(h, params):
    centered = h - h.mean(1, keepdim=True)
    sigma = (centered.square().mean(1, keepdim=True) + params["eps"]).sqrt()
    t = (centered / sigma * params["gamma"] + params["beta"]) @ params["projection"]
    f = t / t.norm(dim=1, keepdim=True)
    return f @ params["head"].T, t, f


def decision_metrics(before, after, labels, t0, t1, f0, f1):
    rows = torch.arange(len(labels))
    p0, p1 = before.argmax(1), after.argmax(1)
    c0, c1 = p0.eq(labels), p1.eq(labels)
    wrong = before.clone()
    wrong[rows, labels] = -torch.inf
    rival = wrong.argmax(1)
    m0 = before[rows, labels] - wrong.max(1).values
    wrong_after = after.clone()
    wrong_after[rows, labels] = -torch.inf
    m1 = after[rows, labels] - wrong_after.max(1).values
    fixed_m1 = after[rows, labels] - after[rows, rival]
    a0, a1 = before - before.mean(1, keepdim=True), after - after.mean(1, keepdim=True)
    scale = (a0 * a1).sum(1) / a0.square().sum(1)
    remainder = a1 - scale[:, None] * a0
    winner_scores = before.clone()
    winner_scores[rows, p0] = -torch.inf
    winner_gap = before[rows, p0] - winner_scores.max(1).values
    certificate = scale.gt(0) & (2 * remainder.abs().max(1).values < scale * winner_gap)
    assert torch.equal(p0[certificate], p1[certificate])
    dt = t1 - t0
    radial = (dt * f0).sum(1, keepdim=True) * f0
    radial_scale = (t1 * t0).sum(1) / t0.square().sum(1)
    radial_only = t0 + radial
    radial_scores = radial_only / radial_only.norm(dim=1, keepdim=True)
    # The head scores for a positive radial change equal the original scores.
    positive = radial_scale.gt(0)
    assert torch.allclose(radial_scores[positive], f0[positive], atol=1e-10, rtol=0)
    per_class = [dict(class_id=k, n=int(labels.eq(k).sum()),
                      original_correct=int((c0 & labels.eq(k)).sum()),
                      protected_correct=int((c1 & labels.eq(k)).sum()))
                 for k in range(before.shape[1])]
    return dict(n=len(labels), original_correct=int(c0.sum()), protected_correct=int(c1.sum()),
                lost=int((c0 & ~c1).sum()), gained=int((~c0 & c1).sum()),
                changed_predictions=int(p0.ne(p1).sum()), per_class=per_class,
                correct_margin_before=quantiles(m0[c0]), correct_margin_after=quantiles(m1[c0]),
                fixed_rival_margin_change=quantiles((fixed_m1 - m0)[c0]),
                fixed_rival_crossings=int((c0 & fixed_m1.le(0)).sum()),
                score_scale=quantiles(scale), positive_scale_count=int(scale.gt(0).sum()),
                order_certificate_count=int(certificate.sum()),
                order_certificate_violations=0,
                residual_to_scaled_margin=quantiles(2 * remainder.abs().max(1).values /
                    (scale.abs() * winner_gap).clamp_min(1e-15)),
                feature_cosine=quantiles((f0 * f1).sum(1)),
                feature_radial_energy_fraction=float(radial.square().sum() / dt.square().sum()),
                positive_feature_radial_count=int(positive.sum()),
                normalized_score_delta_rms=float((after - before).square().sum(1).mean().sqrt()))


@torch.no_grad()
def pca(z, energy):
    mean = z.mean(0)
    _, singular, vh = torch.linalg.svd((z - mean).cuda(), full_matrices=False)
    fractions = singular.square().cumsum(0) / singular.square().sum()
    rank = int(torch.searchsorted(fractions, energy).item()) + 1
    return dict(mean=mean, basis=vh[:rank].T.cpu(), rank=rank,
                retained=float(fractions[rank - 1]), singular=singular.cpu())


@torch.no_grad()
def run(args):
    cfg = json.loads(Path(args.protocol).read_text())
    for name, pin in cfg["source_sha256"].items():
        assert digest(name) == pin, name
    assert digest(__file__) == cfg["runner_sha256"]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    (out / "artifacts").mkdir()
    save_json(out / "protocol.json", cfg)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path("external/secure-merge-attack"))
    root = Path("checkpoints/ViT-B-32")
    pins = {p["path"].removeprefix("ViT-B-32/"): p["sha256"]
            for p in json.loads(Path("configs/checkpoints.json").read_text())["files"]}
    hashes = {}

    def checked(name):
        hashes[name] = digest(root / name)
        assert hashes[name] == pins[name], name
        return load(str(root / name)).cpu().eval()

    def recovered(model, task):
        ref = cfg["AGA_references"][task]
        assert digest(ref["path"]) == ref["sha256"]
        assert json.loads(Path(ref["validation"]).read_text())["status"] == "passed"
        result = copy.deepcopy(model)
        result.load_state_dict(torch.load(ref["path"], map_location="cpu"), strict=True)
        return result

    anchor = checked("zeroshot.pt")
    raw, summaries = {}, {}
    for case in cfg["cases"]:
        task, partner_name = case["target"], case["partner"]
        source = Path(case["source"])
        assert json.loads((source / "independent_validation.json").read_text())["status"] == "passed"
        meta = json.loads((source / "execution.json").read_text())
        assert digest(source / "execution.json") == case["execution_sha256"]
        assert digest(source / "artifacts/calibration.pt") == meta["calibration_sha256"]
        cal = torch.load(source / "artifacts/calibration.pt", map_location="cpu")
        parent = cal
        if "reference" in cal:
            assert digest(cal["reference"]["path"]) == cal["reference"]["sha256"]
            parent = torch.load(cal["reference"]["path"], map_location="cpu")
        indices = json.loads((source / "training_indices.json").read_text())
        assert len(set(sum(indices.values(), []))) == sum(map(len, indices.values()))
        expert = checked(f"{task}/finetuned.pt")
        head = checked(f"head_{task}.pt")
        assert head.normalize and torch.count_nonzero(head.bias) == 0
        training = dataset(task, "data", expert.val_preprocess, True)
        labels_all = torch.from_numpy(official_labels(task, True))
        confirmation = make_loader(training, indices["confirmation"], 64)
        zc, hc = capture(expert, confirmation)
        folds = dict(fit=parent["fit_hidden"], scale=parent["scale_hidden"], confirmation=zc)
        spaces = {name: pca(z, cfg["retained_energy"]) for name, z in folds.items()}
        fitted_basis = cal["basis"]
        fitted_mean = cal["mean"]
        fit_space_error = float((fitted_basis - spaces["fit"]["basis"] @
                                (spaces["fit"]["basis"].T @ fitted_basis)).norm() /
                               fitted_basis.shape[1] ** 0.5)
        assert fit_space_error < 1e-6
        geometry = dict(ranks={k: v["rank"] for k, v in spaces.items()}, comparisons=[], cross_energy=[])
        for left, right in [("fit", "scale"), ("fit", "confirmation"), ("scale", "confirmation")]:
            a, b = spaces[left]["basis"], spaces[right]["basis"]
            cosine = torch.linalg.svdvals(a.T @ b).clamp(0, 1)
            geometry["comparisons"].append(dict(left=left, right=right,
                mean_squared_principal_cosine=float(cosine.square().mean()), cosine=quantiles(cosine)))
        for name, space in spaces.items():
            u, mean = space["basis"], space["mean"]
            for group, z in folds.items():
                centered = z - mean
                outside = centered - (centered @ u) @ u.T
                shift = outside.mean(0)
                variance = float((outside - shift).square().sum(1).mean())
                geometry["cross_energy"].append(dict(space=name, group=group,
                    outside_fraction=float(outside.square().sum() / centered.square().sum()),
                    outside_variance=variance, outside_mean_squared=float(shift.square().sum())))

        families = {}
        assert digest(source / "artifacts/release.pt") == meta["release_sha256"]
        native_edits = torch.load(source / "artifacts/release.pt", map_location="cpu")
        assert set(native_edits) == {PREFIX + "weight", PREFIX + "bias"}
        assert digest(source / "artifacts/released_AGA_edits.pt") == meta["AGA_edits_sha256"]
        aga_edits = torch.load(source / "artifacts/released_AGA_edits.pt", map_location="cpu")
        assert set(aga_edits) == set(native_edits)
        families["covariance"] = dict(native=native_edits, AGA=aga_edits)
        if task == "MNIST":
            best = cfg["best_reference"]
            assert digest(best["release"]) == best["release_sha256"]
            assert digest(best["AGA"]) == best["AGA_sha256"]
            frozen = torch.load(best["release"], map_location="cpu")
            assert torch.equal(fitted_basis, frozen["basis"])
            assert indices == json.loads(Path(best["indices"]).read_text())
            families["rank_one"] = dict(native={PREFIX + "weight": frozen["released_weight"],
                PREFIX + "bias": frozen["released_bias"]},
                AGA={k: v for k, v in torch.load(best["AGA"], map_location="cpu").items()
                     if k.startswith(PREFIX)})
        task_raw = dict(indices=indices, folds={k: v.float() for k, v in folds.items()},
                        spaces=spaces, labels=labels_all[indices["confirmation"]], contexts={})
        summary = dict(PCA=geometry, native_responses={}, decisions={})
        ew, eb = [expert.state_dict()[PREFIX + name].double() for name in ["weight", "bias"]]
        for family, parameters in families.items():
            dw = parameters["native"][PREFIX + "weight"].double() - ew
            db = parameters["native"][PREFIX + "bias"].double() - eb
            summary["native_responses"][family] = {}
            for group, z in folds.items():
                response = z @ dw.T + db
                shift = response.mean(0)
                summary["native_responses"][family][group] = dict(n=len(z),
                    variance=float((response - shift).square().sum(1).mean()),
                    mean_squared=float(shift.square().sum()),
                    rms=float(response.square().sum(1).mean().sqrt()))

        partner = checked(f"{partner_name}/finetuned.pt")
        attacked_expert = recovered(expert, task)
        attacked_partner = recovered(partner, partner_name)
        for context in cfg["contexts"]:
            if context == "native":
                model, latent, h = expert, zc, hc
            elif context == "proxy":
                model = merge(anchor, expert, anchor, cfg["proxy_beta"])
                latent, h = capture(model, confirmation)
            else:
                model = merge(anchor, expert if context == "direct" else attacked_expert,
                              partner if context == "direct" else attacked_partner, cfg["coefficient"])
                latent, h = capture(model, confirmation)
            visual = model.model.visual
            params = dict(gamma=visual.ln_post.weight.double().clone(), beta=visual.ln_post.bias.double().clone(),
                          eps=visual.ln_post.eps, projection=visual.proj.double().clone(),
                          head=head.weight.double().clone())
            s0, t0, f0 = classify(h, params)
            context_raw = dict(latent=latent.float(), h=h.float(), params=params, before_scores=s0,
                before_fields={PREFIX + part: model.state_dict()[PREFIX + part].clone()
                               for part in ["weight", "bias"]}, families={})
            labels = task_raw["labels"]
            for family, parameters in families.items():
                own = parameters["native" if context != "AGA" else "AGA"]
                after_fields = {}
                for part in ["weight", "bias"]:
                    key = PREFIX + part
                    if context == "native":
                        after_fields[key] = own[key]
                    elif context == "proxy":
                        base = anchor.state_dict()[key]
                        after_fields[key] = base + cfg["proxy_beta"] * (own[key] - base)
                    else:
                        base = anchor.state_dict()[key]
                        other = (partner if context == "direct" else attacked_partner).state_dict()[key]
                        after_fields[key] = base + cfg["coefficient"] * ((own[key] - base) + (other - base))
                dw = after_fields[PREFIX + "weight"].double() - model.state_dict()[PREFIX + "weight"].double()
                db = after_fields[PREFIX + "bias"].double() - model.state_dict()[PREFIX + "bias"].double()
                delta = latent @ dw.T + db
                s1, t1, f1 = classify(h + delta, params)
                stats = decision_metrics(s0, s1, labels, t0, t1, f0, f1)
                mean_free_scores = classify(h + delta - delta.mean(1, keepdim=True), params)[0]
                assert torch.allclose(s1, mean_free_scores, atol=1e-10, rtol=0)
                changed = copy.deepcopy(model)
                for key, value in after_fields.items():
                    changed.state_dict()[key].copy_(value)
                probe_ids = indices["confirmation"][:cfg["probe_n"]]
                images = torch.stack([training[i][0] for i in probe_ids]).cuda()
                changed.cuda().eval()
                features = changed(images)
                features /= features.norm(dim=-1, keepdim=True)
                measured = features @ head.weight.cuda().T + head.bias.cuda()
                error = float((measured.double().cpu() - s1[:len(probe_ids)]).abs().max())
                assert error < 1e-4
                stats["fresh_score_max_error"] = error
                stats["fresh_prediction_mismatches"] = int(measured.argmax(1).cpu().ne(s1[:len(probe_ids)].argmax(1)).sum())
                assert stats["fresh_prediction_mismatches"] == 0
                stats["fresh_probe_n"] = len(probe_ids)
                changed.cpu()
                del changed
                key = family + "/" + context
                summary["decisions"][key] = stats
                context_raw["families"][family] = dict(delta_h=delta, after_scores=s1,
                    after_fields=after_fields, fresh_scores=measured.cpu())
                print("DECISION_AUDIT", task, key,
                      json.dumps({k: stats[k] for k in ["n", "original_correct", "protected_correct",
                          "changed_predictions", "order_certificate_count", "score_scale", "feature_radial_energy_fraction"]}), flush=True)
            task_raw["contexts"][context] = context_raw
            if context != "native":
                del model
        raw[task], summaries[task] = task_raw, summary
        torch.save(raw, out / "artifacts/observations.pt")
        save_json(out / "results.json", summaries)
        del expert, partner, attacked_expert, attacked_partner, head, cal, parent, families
    save_json(out / "execution.json", dict(completed=True, source_commit=cfg["source_commit"],
        runner_sha256=digest(__file__), protocol_sha256=digest(args.protocol), checkpoint_sha256=hashes,
        raw_sha256=digest(out / "artifacts/observations.pt"), new_protection_weights=0,
        optimizer_steps=0, new_attacks=0, test_set_used=False, contexts=cfg["contexts"], gpu=torch.cuda.get_device_name()))


def verify(output):
    """Independent NumPy recount from raw observations; no primary metric helpers."""
    out = Path(output)
    execution = json.loads((out / "execution.json").read_text())
    assert digest(out / "artifacts/observations.pt") == execution["raw_sha256"]
    raw = torch.load(out / "artifacts/observations.pt", map_location="cpu")
    results = json.loads((out / "results.json").read_text())
    checks = dict(status="passed", raw_sha256=execution["raw_sha256"], tasks={})

    def array(t):
        return t.numpy().astype(np.float64, copy=False)

    def numpy_scores(h, params):
        x = h - h.mean(1, keepdims=True)
        x /= np.sqrt(np.mean(x * x, axis=1, keepdims=True) + params["eps"])
        t = (x * array(params["gamma"]) + array(params["beta"])) @ array(params["projection"])
        f = t / np.linalg.norm(t, axis=1, keepdims=True)
        return f @ array(params["head"]).T, t, f

    for task, record in raw.items():
        labels = record["labels"].numpy()
        summary = results[task]
        task_checks = dict(decisions={}, PCA={})
        for name, space in record["spaces"].items():
            z = array(record["folds"][name])
            u, mu = array(space["basis"]), array(space["mean"])
            x = z - z.mean(0)
            np.testing.assert_allclose(mu, z.mean(0), atol=1e-12, rtol=0)
            np.testing.assert_allclose(u.T @ u, np.eye(u.shape[1]), atol=1e-10, rtol=0)
            # Gram eigenvalues give the PCA spectrum independently of torch's SVD.
            spectrum = np.linalg.eigvalsh(x @ x.T)[::-1].clip(0)
            fraction = np.cumsum(spectrum) / spectrum.sum()
            protocol = json.loads((out / "protocol.json").read_text())
            rank = int(np.searchsorted(fraction, protocol["retained_energy"])) + 1
            assert rank == space["rank"]
            energy = np.square(x @ u).sum() / np.square(x).sum()
            np.testing.assert_allclose(energy, fraction[rank-1], atol=1e-10, rtol=0)
            task_checks["PCA"][name] = dict(rank=rank, retained=energy)
        for row in summary["PCA"]["comparisons"]:
            a, b = [array(record["spaces"][k]["basis"]) for k in [row["left"], row["right"]]]
            cosine = np.linalg.svd(a.T @ b, compute_uv=False).clip(0, 1)
            np.testing.assert_allclose(np.mean(cosine**2), row["mean_squared_principal_cosine"], atol=1e-10)
        for row in summary["PCA"]["cross_energy"]:
            space = record["spaces"][row["space"]]
            x = array(record["folds"][row["group"]]) - array(space["mean"])
            u = array(space["basis"])
            residual = x - (x @ u) @ u.T
            np.testing.assert_allclose(np.square(residual).sum()/np.square(x).sum(), row["outside_fraction"], atol=1e-10)
        for context, observed in record["contexts"].items():
            h, z = array(observed["h"]), array(observed["latent"])
            s0, t0, f0 = numpy_scores(h, observed["params"])
            np.testing.assert_allclose(s0, array(observed["before_scores"]), atol=1e-10, rtol=0)
            rows = np.arange(len(labels))
            p0 = s0.argmax(1)
            wrong = s0.copy()
            wrong[rows, labels] = -np.inf
            rival = wrong.argmax(1)
            m0 = s0[rows, labels] - wrong.max(1)
            for family, saved in observed["families"].items():
                fields, original = saved["after_fields"], observed["before_fields"]
                dw = array(fields[PREFIX+"weight"]) - array(original[PREFIX+"weight"])
                db = array(fields[PREFIX+"bias"]) - array(original[PREFIX+"bias"])
                dh = z @ dw.T + db
                np.testing.assert_allclose(dh, array(saved["delta_h"]), atol=1e-10, rtol=0)
                s1, t1, f1 = numpy_scores(h + dh, observed["params"])
                np.testing.assert_allclose(s1, array(saved["after_scores"]), atol=1e-10, rtol=0)
                p1 = s1.argmax(1)
                c0, c1 = p0 == labels, p1 == labels
                a0, a1 = s0-s0.mean(1, keepdims=True), s1-s1.mean(1, keepdims=True)
                scale = (a0*a1).sum(1)/(a0*a0).sum(1)
                rem = a1-scale[:,None]*a0
                not_winner = s0.copy()
                not_winner[rows,p0] = -np.inf
                gap = s0[rows,p0]-not_winner.max(1)
                certificate = (scale>0) & (2*np.abs(rem).max(1)<scale*gap)
                assert np.all(p0[certificate] == p1[certificate])
                expected = summary["decisions"][family+"/"+context]
                recounted = dict(original_correct=int(c0.sum()), protected_correct=int(c1.sum()),
                    lost=int((c0&~c1).sum()), gained=int((~c0&c1).sum()),
                    changed_predictions=int((p0!=p1).sum()), order_certificate_count=int(certificate.sum()),
                    fixed_rival_crossings=int((c0 & ((s1[rows,labels]-s1[rows,rival])<=0)).sum()))
                assert all(expected[k]==v for k,v in recounted.items())
                for row in expected["per_class"]:
                    selected = labels == row["class_id"]
                    assert row["n"]==int(selected.sum())
                    assert row["original_correct"]==int((selected & c0).sum())
                    assert row["protected_correct"]==int((selected & c1).sum())
                dt = t1-t0
                radial = (dt*f0).sum(1,keepdims=True)*f0
                fraction = float(np.square(radial).sum()/np.square(dt).sum())
                np.testing.assert_allclose(fraction, expected["feature_radial_energy_fraction"], atol=1e-10)
                np.testing.assert_allclose(np.quantile(scale,.5), expected["score_scale"]["0.5"],atol=1e-10)
                measured = array(saved["fresh_scores"])
                score_error = float(np.abs(s1[:len(measured)]-measured).max())
                assert score_error<1e-4 and np.array_equal(measured.argmax(1),p1[:len(measured)])
                task_checks["decisions"][family+"/"+context] = dict(**recounted, fresh_score_max_error=score_error)
        checks["tasks"][task] = task_checks
    save_json(out / "independent_validation.json", checks)
    print(json.dumps(checks), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", default="configs/decision_audit.json")
    parser.add_argument("--output", default="outputs/decision_audit")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    verify(args.output) if args.verify else run(args)
