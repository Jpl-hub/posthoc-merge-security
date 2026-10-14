"""Independent NumPy reconstruction, official-label counts and real merge replay."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import checkpoint_loader, digest
from covariance_noise import dataset
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_probe_candidate import PREFIX, array, explicit_merge, features


@torch.no_grad()
def verify(output, probe_n):
    out = Path(output); cfg = json.loads((out/"protocol.json").read_text())
    meta = json.loads((out/"execution.json").read_text()); case = json.loads((out/"case.json").read_text())
    assert meta["completed"] and meta["optimizer_steps"] == 0
    for path, pin in cfg["source_sha256"].items(): assert digest(path) == pin, path
    for name, key in [("calibration.pt", "calibration_sha256"), ("release.pt", "release_sha256"), ("predictions.pt", "predictions_sha256")]:
        assert digest(out/"artifacts"/name) == meta[key]
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    load = checkpoint_loader(Path("external/secure-merge-attack")); root = Path("checkpoints/ViT-B-32")
    for path, pin in meta["checkpoint_sha256"].items(): assert digest(root/path) == pin, path
    target, peer_name = case["target"], case["partner"]
    anchor = load(str(root/"zeroshot.pt")).cpu().eval(); expert = load(str(root/target/"finetuned.pt")).cpu().eval()
    head = load(str(root/f"head_{target}.pt")).cpu().eval(); cal = torch.load(out/"artifacts/calibration.pt", map_location="cpu")
    release = torch.load(out/"artifacts/release.pt", map_location="cpu")
    reference = Path(cfg["matrix_references"][target]); assert digest(reference/"artifacts/calibration.pt") == cal["source_cache_sha256"]
    old = torch.load(reference/"artifacts/calibration.pt", map_location="cpu")
    probe_path = Path(cfg["probe_references"][target])/"artifacts/probes.pt"
    assert digest(probe_path) == cal["probe_sha256"]
    probe = torch.load(probe_path, map_location="cpu")["svd"]["A"]
    ids = probe["selected"]; u, s, vh = array(probe["u"][:, ids]), array(probe["s"][ids]), array(probe["vh"][ids])
    block = expert.model.visual.transformer.resblocks[11].mlp; base_block = anchor.model.visual.transformer.resblocks[11].mlp
    a, b = array(block.c_fc.weight), array(block.c_proj.weight)
    np.testing.assert_allclose((a-array(base_block.c_fc.weight)) @ vh.T, u*s, rtol=1e-8, atol=1e-9)
    task = (u*s) @ vh; ray = task/np.linalg.norm(task)
    np.testing.assert_allclose(ray, array(cal["ray"]), atol=1e-9)
    ea = cal["alpha"]*np.linalg.norm(a)*ray
    x = array(cal["native"]["x"]); pre = x@a.T+array(block.c_fc.bias); z = pre*expit(1.702*pre)
    newpre = x@(a+ea).T+array(block.c_fc.bias); znew = newpre*expit(1.702*newpre)
    v = array(cal["basis"]["B"]["vh"]); inputs = znew@v.T; gram = inputs.T@inputs
    np.testing.assert_allclose(v@v.T, np.eye(len(v)), atol=1e-8)
    ridge = cfg["ridge_relative_floor"]*np.linalg.eigvalsh(gram)[-1]
    np.testing.assert_allclose(ridge, cal["ridge"], rtol=1e-9)
    k = np.linalg.solve(gram+ridge*np.eye(len(v)), inputs.T@(z@b.T-znew@b.T)).T; eb = k@v
    np.testing.assert_allclose(k, array(cal["coefficient_B"]), rtol=1e-6, atol=1e-8)
    assert set(release) == {PREFIX+"c_fc.weight", PREFIX+"c_proj.weight"}
    for suffix, w, e in [("c_fc.weight", a, ea), ("c_proj.weight", b, eb)]:
        np.testing.assert_allclose((w+e).astype(np.float32), release[PREFIX+suffix].numpy(), rtol=2e-7, atol=2e-8)
    distance = np.sqrt(np.sum(ea*ea)/np.sum(a*a)+np.sum(eb*eb)/np.sum(b*b))
    assert distance <= cfg["parameter_budget"]+1e-10
    residuals = {}
    for group in ["native", "scale"]:
        cache = cal[group]; x, h = array(cache["x"]), array(cache["h"])
        pre = x@a.T+array(block.c_fc.bias); zz = pre*expit(1.702*pre)
        pre_new = x@(a+ea).T+array(block.c_fc.bias); zn = pre_new*expit(1.702*pre_new)
        dh = zn@(b+eb).T-zz@b.T
        df = features(h+dh, expert.model.visual)-features(h, expert.model.visual)
        pr = dh@array(cal["basis"]["B"]["u"])/old["scales"]["B"]
        loss = float(np.mean(np.sum(df*df, 1))+np.mean(np.sum(pr*pr, 1)))
        assert loss <= cfg["response_budget"]**2*(1+1e-6)
        residuals[group] = loss
    chosen = max([t for t in cal["trials"] if t["feasible"]], key=lambda t: t["distance_squared"])
    assert chosen["alpha"] == cal["alpha"]
    released = copy.deepcopy(expert)
    for key, value in release.items(): released.state_dict()[key].copy_(value)
    raw = torch.load(out/"artifacts/predictions.pt", map_location="cpu"); rows = json.loads((out/"results.json").read_text())
    indices = json.loads((out/"training_indices.json").read_text()); hold = json.loads((out/"holdout_protocol.json").read_text())
    train_labels = official_labels(target, True); checks = []; fresh = 0
    models = {"original": expert, "released": released}; tests = {target: dataset(target, "data", expert.val_preprocess)}; heads = {target: head}
    if meta["fusion_run"]:
        peer = load(str(root/peer_name/"finetuned.pt")).cpu().eval()
        heads[peer_name] = load(str(root/f"head_{peer_name}.pt")).cpu().eval(); tests[peer_name] = dataset(peer_name, "data", expert.val_preprocess)
        def recover(model, task):
            ref = cfg["AGA_references"][task]; assert digest(ref["path"]) == ref["sha256"]
            result = copy.deepcopy(model); result.load_state_dict(torch.load(ref["path"], map_location="cpu")); return result
        ao, ap = recover(expert, target), recover(peer, peer_name); ar = copy.deepcopy(ao)
        assert digest(out/"artifacts/released_AGA_edits.pt") == meta["AGA_edits_sha256"]
        for key, value in torch.load(out/"artifacts/released_AGA_edits.pt", map_location="cpu").items(): ar.state_dict()[key].copy_(value)
        for preset, own, attack in [("original", expert, ao), ("released", released, ar)]:
            models[preset+"_direct"] = explicit_merge(anchor, own, peer, cfg["coefficient"])
            models[preset+"_AGA"] = explicit_merge(anchor, attack, ap, cfg["coefficient"])
    for row in rows:
        preset, stage, task = [row[k] for k in ["preset", "stage", "dataset"]]; key = "/".join([preset, stage, task])
        pred, y = [raw[key][k].numpy() for k in ["prediction", "target"]]
        if stage == "holdout" and target == "DTD":
            droot = Path("data/dtd/dtd"); classes = sorted(p.name for p in (droot/"images").iterdir() if p.is_dir())
            truth = np.array([classes.index(n.split("/")[0]) for n in (droot/"labels/val1.txt").read_text().splitlines()])
        elif stage in ["holdout", "confirmation"]: truth = train_labels[hold["indices"] if stage == "holdout" else indices["confirmation"]]
        else: truth = official_labels(task)
        np.testing.assert_array_equal(y, truth); assert int((pred==y).sum()) == row["correct"] and len(y) == row["n"]
        for c in row["per_class"]:
            mask = y == c["class_id"]; assert int(mask.sum()) == c["n"] and int(((pred==y)&mask).sum()) == c["correct"]
        if preset == "released":
            p0 = raw["/".join(["original", stage, task])]["prediction"].numpy()
            assert int((p0!=pred).sum()) == row["changed_predictions"]
            assert int(((p0==y)&(pred!=y)).sum()) == row["paired"]["lost"]
            assert int(((p0!=y)&(pred==y)).sum()) == row["paired"]["gained"]
        if stage in ["standalone", "direct", "AGA"]:
            model = models[preset if stage == "standalone" else preset+"_"+stage]
            images = torch.stack([tests[task][i][0] for i in range(probe_n)]).cuda()
            model.cuda(); classifier = heads[task].cuda(); actual = classifier(model(images)).argmax(1).cpu().numpy()
            np.testing.assert_array_equal(actual, pred[:probe_n]); model.cpu(); classifier.cpu(); fresh += probe_n
        checks.append({"key": key, "correct": row["correct"], "n": row["n"]})
    actual_distance = sum(float((release[PREFIX+suffix].double()-w.double()).square().sum()) for suffix, w in [("c_fc.weight", block.c_fc.weight), ("c_proj.weight", block.c_proj.weight)]) ** .5
    model_norm = sum(float(v.double().square().sum()) for v in expert.state_dict().values() if v.is_floating_point()) ** .5
    report = {"status": "passed", "verifier_sha256": digest(__file__), "coefficient_distance": float(distance),
              "nonlinear_native_losses": residuals, "official_counts": checks, "fresh_predictions": fresh,
              "full_model_relative_distance": actual_distance/model_norm, "explicit_merge_formula": True,
              "new_attacks": 0, "new_release_weights": 0, "release_sha256": meta["release_sha256"],
              "raw_predictions_sha256": meta["predictions_sha256"]}
    save_json(out/"independent_validation.json", report)
    print(json.dumps({k:v for k,v in report.items() if k!="official_counts"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output", required=True)
    parser.add_argument("--probe-n", type=int, default=16)
    args = parser.parse_args(); verify(args.output, args.probe_n)
