"""Training-only finite margin decomposition from frozen audit observations."""

import hashlib
import json
import struct
from pathlib import Path

import numpy as np
import torch


out = Path("outputs/decision_audit")
meta = json.loads((out / "execution.json").read_text())
assert hashlib.sha256((out / "artifacts/observations.pt").read_bytes()).hexdigest() == meta["raw_sha256"]
assert json.loads((out / "independent_validation.json").read_text())["status"] == "passed"
raw = torch.load(out / "artifacts/observations.pt", map_location="cpu")
results = json.loads((out / "results.json").read_text())


def arr(t):
    return t.numpy().astype(np.float64, copy=False)


def quantile(x):
    return {str(q): float(np.quantile(x, q)) for q in [.05, .5, .95]}


report = dict(raw_sha256=meta["raw_sha256"], analysis_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              test_set_used=False, new_protection_weights=0, tasks={})
for task, saved in raw.items():
    if task == "MNIST":
        label_file = Path("data/MNIST/raw/train-labels-idx1-ubyte").read_bytes()
        assert struct.unpack(">II", label_file[:8]) == (2049, 60000)
        official = np.frombuffer(label_file[8:], np.uint8)
    else:
        root = Path("data/dtd/dtd")
        classes = sorted(p.name for p in (root / "images").iterdir() if p.is_dir())
        names = (root / "labels/train1.txt").read_text().splitlines()
        official = np.array([classes.index(name.split("/")[0]) for name in names])
    labels = saved["labels"].numpy()
    np.testing.assert_array_equal(labels, official[saved["indices"]["confirmation"]])
    rows = np.arange(len(labels))
    task_report = {}
    for context, observed in saved["contexts"].items():
        params = observed["params"]
        gamma, beta, projection, head = [arr(params[k]) for k in ["gamma", "beta", "projection", "head"]]
        h = arr(observed["h"])
        # Remove class-common terms; the common positive denominator does not affect ranking.
        contrast = head - head.mean(0)
        readout = gamma[:, None] * (projection @ contrast.T)
        readout -= readout.mean(0)
        offset = beta @ projection @ contrast.T
        sigma = np.sqrt(np.var(h, axis=1) + params["eps"])
        numerator = h @ readout + sigma[:, None] * offset
        s0 = arr(observed["before_scores"])
        np.testing.assert_array_equal(numerator.argmax(1), s0.argmax(1))
        wrong = s0.copy()
        wrong[rows, labels] = -np.inf
        rival = wrong.argmax(1)
        correct = s0.argmax(1) == labels
        margin = numerator[rows, labels] - numerator[rows, rival]
        # sigma(h) is 1/sqrt(d)-Lipschitz; this radius preserves the winner
        # for EVERY residual perturbation direction, not just the observed noise.
        gram = readout.T @ readout
        squared_distance = np.diag(gram)[:, None] + np.diag(gram)[None, :] - 2*gram
        lipschitz = np.sqrt(squared_distance.clip(0)) + np.abs(offset[:, None]-offset[None, :])/np.sqrt(h.shape[1])
        winner = numerator.argmax(1)
        pair_gap = numerator[rows, winner, None] - numerator
        radius_by_rival = np.divide(pair_gap, lipschitz[winner], out=np.full_like(pair_gap, np.inf), where=lipschitz[winner]>1e-14)
        radius_by_rival[rows, winner] = np.inf
        radius = radius_by_rival.min(1)
        for family, record in observed["families"].items():
            dh = arr(record["delta_h"])
            sigma_after = np.sqrt(np.var(h + dh, axis=1) + params["eps"])
            linear_change = dh @ readout
            ln_offset_change = (sigma_after-sigma)[:, None] * offset
            change = linear_change + ln_offset_change
            s1 = arr(record["after_scores"])
            np.testing.assert_array_equal((numerator+change).argmax(1), s1.argmax(1))
            fixed_margin_change = change[rows, labels] - change[rows, rival]
            ratio = -fixed_margin_change[correct]/margin[correct]
            # This ratio only tests the original strongest rival; another rival can win instead.
            stats = dict(fixed_rival_noise_to_margin=quantile(ratio),
                fixed_rival_crossings=int((ratio >= 1).sum()),
                perturbation_to_safe_radius=quantile(np.linalg.norm(dh, axis=1)/radius),
                all_direction_safe_count=int((np.linalg.norm(dh, axis=1)<radius).sum()),
                all_direction_safe_correct_count=int(((np.linalg.norm(dh, axis=1)<radius)&correct).sum()),
                layernorm_offset_delta_energy_fraction=float(np.square(ln_offset_change).sum()/np.square(change).sum()),
                class_numerator_delta_rms=float(np.sqrt(np.square(change).sum(1).mean())))
            safe = np.linalg.norm(dh, axis=1)<radius
            assert np.array_equal(winner[safe], s1.argmax(1)[safe])
            assert stats["fixed_rival_crossings"] == results[task]["decisions"][family+"/"+context]["fixed_rival_crossings"]
            task_report[family+"/"+context] = stats
    # Check whether the existing 1% raw-residual budget could be replaced by
    # a rigorous direction-specific training margin bound. This selects no
    # new release amplitude and uses no partner model or evaluation test set.
    native = saved["contexts"]["native"]
    p = native["params"]
    contrast = arr(p["head"])-arr(p["head"]).mean(0)
    a = arr(p["gamma"])[:, None]*(arr(p["projection"]) @ contrast.T)
    a -= a.mean(0)
    b = arr(p["beta"]) @ arr(p["projection"]) @ contrast.T
    parent = torch.load(Path("outputs/covariance_noise_"+task.lower()+"/artifacts/calibration.pt"),map_location="cpu")
    if "reference" in parent:
        parent = torch.load(parent["reference"]["path"],map_location="cpu")
    directional_bounds = {}
    prefix = "model.visual.transformer.resblocks.11.mlp.c_proj."
    for family, record in native["families"].items():
        dw = arr(record["after_fields"][prefix+"weight"])-arr(native["before_fields"][prefix+"weight"])
        db = arr(record["after_fields"][prefix+"bias"])-arr(native["before_fields"][prefix+"bias"])
        bounds = {}
        for fold in ["fit","scale","confirmation"]:
            z = arr(saved["folds"][fold])
            h = arr(native["h"]) if fold=="confirmation" else arr(parent[fold+"_residual"])
            dh = z @ dw.T+db
            n = h @ a+np.sqrt(np.var(h,axis=1)+p["eps"])[:,None]*b
            winner=n.argmax(1); row=np.arange(len(h))
            gap=n[row,winner,None]-n
            linear=dh @ a
            pair_slope=linear[row,winner,None]-linear
            error=np.abs(b[winner,None]-b)*np.linalg.norm(dh-dh.mean(1,keepdims=True),axis=1,keepdims=True)/np.sqrt(h.shape[1])
            rate=error-pair_slope
            cap=np.divide(gap,rate,out=np.full_like(gap,np.inf),where=rate>1e-14)
            cap[row,winner]=np.inf
            bounds[fold]=float(cap.min())
        directional_bounds[family]=dict(relative_to_existing_noise=bounds,
            fit_scale_sufficient_bound=min(bounds["fit"],bounds["scale"]),
            release_amplitude_changed=False, no_accuracy_tuning=True,
            interpretation="Sufficient whole-fold prediction preservation bound; below 1 does not prove existing noise unsafe.")
    report.setdefault("native_directional_bounds",{})[task]=directional_bounds
    report["tasks"][task] = task_report
(out / "margin_analysis.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report), flush=True)
