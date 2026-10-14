"""Verify publication deltas and fresh task constraints independently of optimization."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from covariance_noise import dataset
from subspace_probe import save_json
from task_probe_distance import checked_models, parameters
from verify_margin_probes import constraints
from verify_probe_candidate import array


def verify(output):
    output = Path(output); report = json.loads((output/"results.json").read_text())
    artifact = output/"candidate.pt"; assert digest(artifact) == report["candidate_sha256"]
    cal = torch.load(artifact, map_location="cpu"); cfg = cal["protocol"]
    anchor, expert, head, hashes = checked_models(report["target"])
    assert hashes == report["checkpoint_sha256"]
    p = parameters(expert, head); theta = array(cal["theta"])
    rank = cfg["probe_rank"]; matrix = cfg["coordinate_form"] == "matrix"
    size = rank**2 if matrix else 1
    basis = cal["basis"]
    a, b, bias = [array(p[k]) for k in ["A", "B", "a"]]
    ea = (np.linalg.norm(a)*array(basis["A"]["u"])@theta[:size].reshape(rank,rank)@array(basis["A"]["vh"])) if matrix else np.linalg.norm(a)*theta[0]*array(cal["ray"])
    eb = np.linalg.norm(b)*theta[size:].reshape(b.shape[0],rank)@array(basis["B"]["vh"])
    np.testing.assert_allclose(ea, array(cal["release_deltas"]["A"]), atol=1e-8, rtol=1e-7)
    np.testing.assert_allclose(eb, array(cal["release_deltas"]["B"]), atol=1e-8, rtol=1e-7)
    distance = float(np.sqrt(np.linalg.norm(ea)**2/np.linalg.norm(a)**2+np.linalg.norm(eb)**2/np.linalg.norm(b)**2))
    assert abs(distance-report["selected"]["parameter_distance"]) < 1e-8
    assert distance <= cfg["parameter_budget"]*(1+1e-12)
    conditional_error = None
    if report["method"] == "conditional_composite_distillation_primal_dual":
        x = np.concatenate([array(cal[k]["x"]) for k in ["native","scale"]])
        pre0 = x@a.T+bias; z0 = pre0*expit(1.702*pre0)
        vb = array(basis["B"]["vh"]); q0 = z0@vb.T
        ridge = cfg["ridge_relative_floor"]*np.linalg.eigvalsh(q0.T@q0)[-1]
        pre = x@(a+ea).T+bias; z = pre*expit(1.702*pre); q = z@vb.T
        k = np.linalg.solve(q.T@q+ridge*np.eye(rank), q.T@(z0@b.T-z@b.T)).T
        conditional_error = float(np.max(np.abs(k/np.linalg.norm(b)-theta[size:].reshape(b.shape[0],rank))))
        assert conditional_error < 1e-7
    split = json.loads((output/"confirmation_split.json").read_text())
    indices = cal["confirmation_indices"]
    assert split["indices"] == indices and len(set(indices)) == len(indices)
    old = set(i for group in cal["indices"].values() for i in group)
    assert not old.intersection(indices)
    for path, sha in split["excluded_metadata_sha256"].items():
        assert digest(path) == sha
        data = json.loads(Path(path).read_text())
        groups = list(data.values()) if Path(path).name == "training_indices.json" else [data["indices"]]
        assert not set(indices).intersection(i for group in groups if isinstance(group,list) for i in group)
    before = array(cal["confirmation_original_predictions"]); after = array(cal["confirmation_candidate_predictions"])
    labels = array(cal["confirmation_labels"])
    actual_counts = {"n":len(labels),"original_correct":int((before==labels).sum()),
                     "candidate_correct":int((after==labels).sum()),"changed_predictions":int((before!=after).sum())}
    for key,value in actual_counts.items(): assert report["confirmation"][key] == value
    try:
        checked = constraints(expert, head, cal, {**cfg,"verification_target":report["target"]}, ea, eb)
        assert report["confirmation_passed"]
        status = "passed"
    except AssertionError as error:
        detail = error.args[0]
        assert isinstance(detail,tuple) and detail[0] == "confirmation", error
        assert not report["confirmation_passed"] and detail[1] > cfg["margin_fraction"]
        assert abs(detail[1]-report["confirmation"]["worst_margin_drift"]) < 1e-6
        checked = {"failed_group":detail[0],"independent_maximum":detail[1]}
        status = "confirmed_constraint_failure"
    # Only a passing candidate warrants true full-model fresh confirmation replays.
    fresh = []
    if status == "passed":
        protected = copy.deepcopy(expert)
        block = protected.model.visual.transformer.resblocks[11].mlp
        with torch.no_grad():
            block.c_fc.weight.add_(torch.from_numpy(ea).float())
            block.c_proj.weight.add_(torch.from_numpy(eb).float())
        training = dataset(report["target"],"data",expert.val_preprocess,True)
        for start in range(0,min(128,len(indices)),32):
            images = torch.stack([training[i][0] for i in indices[start:start+32]]).cuda()
            for name,model,pred in [("original",expert,before),("candidate",protected,after)]:
                with torch.no_grad():
                    model.cuda(); head.cuda(); actual = head(model(images)).argmax(1).cpu().numpy()
                    model.cpu(); head.cpu()
                assert np.array_equal(actual,pred[start:start+len(images)])
                fresh.append({"model":name,"offset":start,"n":len(images),"prediction_match":True})
    result = {"status":status,"source_sha256":digest(__file__),"candidate_sha256":digest(artifact),
              "parameter_distance":distance,"conditional_numpy_coefficient_error":conditional_error,
              "confirmation_counts":actual_counts,"task_constraints":checked,
              "fresh_confirmation_prediction_instances":sum(row["n"] for row in fresh),"fresh_replays":fresh,
              "new_release_weights":0,"test_used":False,"partner_used":False}
    save_json(output/"independent_validation.json",result); print(json.dumps(result),flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--output",required=True)
    args = parser.parse_args(); torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    verify(args.output)
