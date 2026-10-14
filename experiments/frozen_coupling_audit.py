"""Read-only factorial response audit of an already frozen publication point."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from clip_benchmark import checkpoint_loader, digest, merge
from covariance_noise import dataset
from cross_module_distance import capture_attention, extended_parameters, final_hidden, normalized_input
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_probe_distance import checked_models, gelu, tail


def pinned(path, pin):
    assert digest(path) == pin, str(path)
    return torch.load(path, map_location="cpu")


@torch.no_grad()
def audit(cfg):
    out = Path(cfg["output"])
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    anchor, expert, head, hashes = checked_models(cfg["target"])
    point = pinned(Path(cfg["point"]), cfg["point_sha256"])
    release = pinned(Path(cfg["release"]), cfg["release_sha256"])
    released = copy.deepcopy(expert)
    released.load_state_dict(expert.state_dict() | release, strict=True)
    load = checkpoint_loader(Path("external/secure-merge-attack"))
    peer_path = Path(cfg["peer_checkpoint"])
    assert digest(peer_path) == cfg["peer_sha256"]
    peer = load(str(peer_path)).cpu().eval()
    original_aga = copy.deepcopy(expert)
    original_aga.load_state_dict(pinned(Path(cfg["original_aga"]), cfg["original_aga_sha256"]))
    published_aga = copy.deepcopy(expert)
    aga_edits = pinned(Path(cfg["published_aga_edits"]), cfg["published_aga_edits_sha256"])
    published_aga.load_state_dict(original_aga.state_dict() | aga_edits, strict=True)
    peer_aga = copy.deepcopy(peer)
    peer_aga.load_state_dict(pinned(Path(cfg["peer_aga"]), cfg["peer_aga_sha256"]))
    indices = point["indices"]["fit"][:cfg["n"]]
    assert len(indices) == cfg["n"]
    training = dataset(cfg["target"], "data", expert.val_preprocess, True)
    labels = torch.tensor([training[i][1] for i in indices])
    pairs = {
        "native": (expert, released),
        "public_proxy": (merge(anchor, expert, anchor, .5), merge(anchor, released, anchor, .5)),
        "real_TA": (merge(anchor, expert, peer, .5), merge(anchor, released, peer, .5)),
        "real_AGA": (merge(anchor, original_aga, peer_aga, .5), merge(anchor, published_aga, peer_aga, .5)),
    }
    raw = {"indices": indices, "labels": labels, "groups": {}, "cfg": cfg}
    records = []
    peak = 0
    for name, (before, after) in pairs.items():
        torch.cuda.reset_peak_memory_stats()
        c0 = capture_attention(before, training, indices)
        c1 = capture_attention(after, training, indices)
        p0 = on_device(extended_parameters(before, head))
        p1 = on_device(extended_parameters(after, head))
        for key in ["A", "a", "b", "gamma2", "beta2", "gamma", "beta", "projection", "head"]:
            assert torch.equal(p0[key], p1[key]), (name, key)
        a, b = on_device(c0), on_device(c1)
        rounding = a["h"] - final_hidden(a["r"], p0)
        hidden0 = gelu(normalized_input(a["r"], p0) @ p0["A"].T + p0["a"])
        hidden1 = gelu(normalized_input(b["r"], p0) @ p0["A"].T + p0["a"])
        db = p1["B"] - p0["B"]
        hs = {"baseline": a["h"], "attention_branch_only": rounding + final_hidden(b["r"], p0),
              "B_only": a["h"] + hidden0 @ db.T, "joint": rounding + final_hidden(b["r"], p1)}
        hidden_error = float((hs["joint"] - b["h"]).abs().max())
        assert hidden_error < cfg["cache_tolerance"], (name, hidden_error)
        ss = {key: scores(value, p0) for key, value in hs.items()}
        actual0 = (a["actual"] / a["actual"].norm(dim=1, keepdim=True)) @ p0["head"].T
        actual1 = (b["actual"] / b["actual"].norm(dim=1, keepdim=True)) @ p1["head"].T
        score_error = max(float((ss["baseline"]-actual0).abs().max()), float((ss["joint"]-actual1).abs().max()))
        assert score_error < cfg["cache_tolerance"]
        assert torch.equal(ss["joint"].argmax(1), actual1.argmax(1))
        yy = labels.cuda(); winner = ss["baseline"].argmax(1)
        row = torch.arange(len(yy), device="cuda")
        other = ss["baseline"].clone(); other[row, winner] = -torch.inf
        runner = other.argmax(1); gap = ss["baseline"][row, winner] - ss["baseline"][row, runner]
        f0 = tail(hs["baseline"], p0)[1]
        da = tail(hs["attention_branch_only"], p0)[1] - f0
        dbf = tail(hs["B_only"], p0)[1] - f0
        dj = tail(hs["joint"], p0)[1] - f0
        cosine = (da*dbf).sum(1)/(da.norm(dim=1)*dbf.norm(dim=1)).clamp_min(1e-20)
        details = []
        for key, s in ss.items():
            changed = s.argmax(1) != winner
            details.append({"component": key, "correct": int((s.argmax(1)==yy).sum()), "n": len(yy),
                "changed": int(changed.sum()), "changed_by_teacher_class": torch.bincount(winner[changed], minlength=s.shape[1]).cpu().tolist(),
                "worst_relative_margin_drift": float(signed_residual(ss["baseline"], s).abs().max()),
                "median_fixed_runner_margin_change": float(((s[row,winner]-s[row,runner]-gap)/gap).median())})
        interaction = hs["joint"]-hs["attention_branch_only"]-hs["B_only"]+hs["baseline"]
        exact_interaction = (hidden1-hidden0) @ db.T
        interaction_error = float((interaction-exact_interaction).abs().max())
        assert interaction_error < 1e-10
        records.append({"environment": name, "components": details, "feature_compensation_cosine_median": float(cosine.median()),
            "joint_to_individual_feature_response_ratio": float(dj.norm()/(da.norm()+dbf.norm())),
            "interaction_hidden_norm": float(interaction.norm()), "hidden_interaction_identity_error": interaction_error,
            "full_hidden_cache_maximum_error": hidden_error, "full_score_cache_maximum_error": score_error,
            "changed_parameter_keys": [k for k,v in after.state_dict().items() if not torch.equal(v, before.state_dict()[k])]})
        raw["groups"][name] = {"before": c0, "after": c1, "parameters_before": {k:v.cpu() if torch.is_tensor(v) else v for k,v in p0.items()},
            "parameters_after": {k:v.cpu() if torch.is_tensor(v) else v for k,v in p1.items()}, "scores": {k:v.cpu() for k,v in ss.items()}}
        peak = max(peak, torch.cuda.max_memory_allocated())
        print(json.dumps(records[-1]), flush=True)
    path = out / "diagnostic.pt"; torch.save(raw, path)
    save_json(out / "protocol.json", cfg)
    save_json(out / "results.json", {"status": "frozen_candidate_response_decomposition", "records": records, "checkpoint_sha256": hashes,
        "diagnostic_sha256": digest(path), "source_sha256": digest(__file__), "protocol_sha256": digest(cfg["protocol_path"]),
        "wall_seconds": time.monotonic()-start, "peak_gpu_bytes": peak, "new_full_model_forward_instances": 8*len(indices),
        "publication_updates": 0, "new_release_weights": 0, "test_used": False,
        "partner_used": True, "role": "Retrospective interpretation only; no new candidate construction or selection uses these peer results.",
        "limitation": "One frozen target-training slice and one existing peer; no causal generalization, task-uniformity, or recovery robustness claim."})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--protocol", required=True); args = parser.parse_args()
    cfg = json.loads(Path(args.protocol).read_text()); cfg["protocol_path"] = args.protocol
    for file, pin in cfg["source_sha256"].items(): assert digest(file) == pin, file
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    audit(cfg)
