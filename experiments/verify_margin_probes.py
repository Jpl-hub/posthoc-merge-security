"""Independent NumPy finite task probes, with actual CLIP hook verification."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from covariance_noise import dataset
from subspace_probe import save_json
from task_probe_distance import checked_models
from verify_probe_candidate import array, features
from verify_svd_compensation import verify


def logits(h, expert, head):
    shape = h.shape[:-1]
    f = features(h.reshape(-1, h.shape[-1]), expert.model.visual)
    return (f@array(head.weight).T).reshape(*shape, len(head.weight))


def drift(before, after):
    winner = np.argmax(before, axis=-1)[..., None]
    gaps = np.take_along_axis(before, winner, axis=-1)-before
    ds = after-before; changes = np.take_along_axis(ds, winner, axis=-1)-ds
    np.put_along_axis(gaps, winner, 1., axis=-1)
    ratios = np.abs(changes)/np.maximum(gaps, 1e-12)
    np.put_along_axis(ratios, winner, 0., axis=-1)
    return np.max(ratios, axis=-1)


def gelu(x):
    return x*expit(1.702*x)


@torch.no_grad()
def constraints(expert, head, cal, cfg, ea, eb):
    block = expert.model.visual.transformer.resblocks[11].mlp
    a, b, bias = array(block.c_fc.weight), array(block.c_proj.weight), array(block.c_fc.bias)
    va, vb = [array(cal["basis"][s]["vh"]) for s in ["A", "B"]]
    report = {}; teachers = {}
    for group in ["native", "scale", "confirmation"]:
        x, h = array(cal[group]["x"]), array(cal[group]["h"])
        z, new_z = gelu(x@a.T+bias), gelu(x@(a+ea).T+bias)
        dh = new_z@(b+eb).T-z@b.T
        before, after = logits(h, expert, head), logits(h+dh, expert, head)
        n = cfg["probe_contexts"]; xx, hh, zz = x[:n], h[:n], z[:n]
        rms_a = np.sqrt(np.mean((x@va.T)**2, axis=0)); rms_b = np.sqrt(np.mean((z@vb.T)**2, axis=0))
        sa = (cfg["finite_step"]*np.array([-1., 1.])[:, None, None]*rms_a[None, :, None]*va[None, :, :]).reshape(-1, a.shape[1])
        sb = (cfg["finite_step"]*np.array([-1., 1.])[:, None, None]*rms_b[None, :, None]*vb[None, :, :]).reshape(-1, b.shape[1])
        base = hh-zz@b.T; xp = xx[:, None, :]+sa[None, :, :]
        ha = base[:, None, :]+gelu(xp@a.T+bias)@b.T
        hb = base[:, None, :]+(zz[:, None, :]+sb[None, :, :])@b.T
        ya, yb = logits(ha, expert, head), logits(hb, expert, head)
        altered_a = base[:, None, :]+gelu(xp@(a+ea).T+bias)@(b+eb).T
        nz = gelu(xx@(a+ea).T+bias)
        altered_b = base[:, None, :]+(nz[:, None, :]+sb[None, :, :])@(b+eb).T
        pa, pb = logits(altered_a, expert, head), logits(altered_b, expert, head)
        maximum = float(max(drift(before, after).max(), drift(ya, pa).max(), drift(yb, pb).max()))
        assert maximum <= cfg["margin_fraction"]*(1+1e-6), (group, maximum)
        assert np.array_equal(np.argmax(before, axis=-1), np.argmax(after, axis=-1))
        assert np.array_equal(np.argmax(ya, axis=-1), np.argmax(pa, axis=-1))
        assert np.array_equal(np.argmax(yb, axis=-1), np.argmax(pb, axis=-1))
        report[group] = {"maximum_margin_drift": maximum, "n": len(x), "probe_instances": 2*n*len(sa)}
        teachers[group] = {"A": ya, "B": yb, "shift_A": sa, "shift_B": sb}
    # The task singular vectors are injected into the real unmodified expert.
    training = dataset(cfg["verification_target"], "data", expert.val_preprocess, True)
    fresh = []; originals = {"A": block.c_fc.weight.clone(), "B": block.c_proj.weight.clone()}
    for group, group_ids in [("native", "fit"), ("scale", "scale")]:
        images = torch.stack([training[i][0] for i in cal["indices"][group_ids][:4]]).cuda()
        expert.cuda(); head.cuda()
        for site, module in [("A", block.c_fc), ("B", block.c_proj)]:
            for j in [0, cfg["probe_rank"]-1, cfg["probe_rank"], 2*cfg["probe_rank"]-1]:
                delta = torch.from_numpy(teachers[group]["shift_"+site][j]).cuda().float()
                def inject(_module, args):
                    changed = args[0].clone(); changed[0] += delta
                    return (changed,)
                hook = module.register_forward_pre_hook(inject)
                try:
                    actual = head(expert(images)).double().cpu().numpy()
                finally:
                    hook.remove()
                expected = teachers[group][site][:4, j]
                np.testing.assert_allclose(actual, expected, atol=5e-4, rtol=5e-5)
                fresh.append({"group": group, "site": site, "direction_and_sign": j,
                              "n": len(images), "max_score_error": float(np.max(np.abs(actual-expected)))})
        expert.cpu(); head.cpu()
    assert torch.equal(block.c_fc.weight, originals["A"]) and torch.equal(block.c_proj.weight, originals["B"])
    report["actual_teacher_probe_checks"] = fresh
    report["actual_teacher_probe_instances"] = sum(r["n"] for r in fresh)
    report["probe_validator_sha256"] = digest(__file__)
    return report


def candidate_constraints(expert, head, cal, cfg, ea, eb):
    target = next(t for t in cfg["targets"] if cal["source_cache_path"].endswith(t.lower()+"/artifacts/calibration.pt"))
    return constraints(expert, head, cal, {**cfg, "verification_target": target}, ea, eb)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True); parser.add_argument("--target")
    parser.add_argument("--output"); parser.add_argument("--probe-n", type=int, default=16)
    args = parser.parse_args(); cfg = json.loads(Path(args.protocol).read_text())
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    if args.output:
        verify(args.output, args.probe_n, native_constraint=candidate_constraints)
    else:
        p = Path(cfg["diagnostic_output"]); geometry = json.loads((p/(args.target+".json")).read_text())
        path = p/(args.target+"_diagnostic.pt"); assert digest(path) == geometry["diagnostic_sha256"]
        cal = torch.load(path, map_location="cpu"); assert cal["protocol"] == cfg
        anchor, expert, head, _ = checked_models(args.target)
        a = array(expert.model.visual.transformer.resblocks[11].mlp.c_fc.weight)
        ea = cal["alpha"]*np.linalg.norm(a)*array(cal["ray"])
        eb = array(cal["coefficient_B"])@array(cal["basis"]["B"]["vh"])
        result = constraints(expert, head, cal, {**cfg, "verification_target": args.target}, ea, eb)
        save_json(p/(args.target+"_probe_validation.json"), {"status": "passed", **result})
        print(json.dumps({"target": args.target, "status": "passed", "maximum_drift": {g: result[g]["maximum_margin_drift"] for g in ["native", "scale", "confirmation"]}, "fresh_teacher_probe_instances": result["actual_teacher_probe_instances"]}), flush=True)
