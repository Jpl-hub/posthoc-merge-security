"""Evaluate the unchanged bias and A/B releases together on MNIST."""
import copy
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from _impl.attacks import published_bias_projection
from _impl.data import dataset, make_loader, official_labels
from _impl.evaluate import evaluate
from _impl.io import digest, new_output, read_json, save_json
from _impl.merge import parameter_hash
from _impl.metrics import row_metrics
from _impl.models import apply, checked_models, initialize, load_checkpoint
from _impl.prodistill import initial


def run_module(name, *args):
    subprocess.run([sys.executable, "-u", "-m", "experiment." + name, *map(str, args)], check=True)


def distill_comparison(cfg, out, ref):
    base, expert, target_head, _ = checked_models(cfg["target"])
    peer = load_checkpoint(cfg["partner"] + "/finetuned.pt")
    protected = apply(expert, ref)
    trained_teacher = apply(expert, cfg["trained_teacher"])
    heads = {cfg["target"]: target_head, cfg["partner"]: load_checkpoint("head_" + cfg["partner"] + ".pt")}
    data = {t: dataset(t, "data", expert.val_preprocess) for t in heads}
    frozen = read_json(cfg["prefix_freeze"])
    tail = {"model.visual.ln_post.weight", "model.visual.ln_post.bias", "model.visual.proj"}
    raw, rows, saved = {}, [], {}
    for label, teacher, donor, key in [("Clean", expert, expert, "Unprotected"),
                                       ("Combined", protected, trained_teacher, "ABReadout")]:
        entry = frozen[cfg["target"] + "/" + key]
        assert parameter_hash(donor.state_dict()) == entry["teacher_parameter_sha256"]
        assert all(torch.equal(v, donor.state_dict()[k]) for k,v in teacher.state_dict().items() if k not in tail)
        student = apply(base, entry["publication"], full=True)
        expected = initial(base, [donor, peer]).state_dict()
        assert all(torch.equal(expected[k], student.state_dict()[k]) for k in tail)
        for cleanup in (False, True):
            own = copy.deepcopy(teacher)
            if cleanup:
                own.load_state_dict(published_bias_projection(own.state_dict()))
            mix = initial(base, [own, peer]).state_dict()
            for k in tail:
                student.state_dict()[k].copy_(mix[k])
            group = label + ("_cleaned" if cleanup else "")
            saved[group] = {"donor":entry["publication"], "parameter_sha256":parameter_hash(student.state_dict())}
            for task, head in heads.items():
                scores, labels = [], []
                cost = evaluate(student, head, make_loader(data[task]), "cuda", logit_sink=scores.append,
                                prediction_sink=lambda _,y: labels.append(y))
                a, y = torch.cat(scores), torch.cat(labels)
                np.testing.assert_array_equal(y.numpy(), official_labels(task))
                assert torch.isfinite(a).all()
                raw[group+"/"+task] = {"logits":a, "target":y}
                control = "Clean_cleaned" if cleanup else "Clean"
                metrics = row_metrics(a, y, raw[control+"/"+task]["logits"], a.shape[1])
                assert cost["n"] == metrics["n"] and cost["correct"] == metrics["correct"]
                rows.append(dict(group=group, task=task, **metrics, seconds=cost["evaluation_seconds"]))
                print("ProDistill", group, task, metrics["correct"], len(y), flush=True)
    torch.save(raw, out/"distilled_scores.pt")
    save_json(out/"distilled.json", dict(rows=rows, prefix_reuse=saved, scores_sha256=digest(out/"distilled_scores.pt")))


def main():
    initialize()
    cfg = read_json(Path(__file__).with_name("protocol.json"))
    out = new_output(cfg["output"], cfg)
    edits = []
    for ref in [cfg["bias"], cfg["ab"]]:
        assert digest(ref["path"]) == ref["sha256"]
        edits.append(torch.load(ref["path"], weights_only=True, map_location="cpu"))
    assert set(edits[0]) == {"model.visual.ln_post.bias"} and set(edits[0]).isdisjoint(edits[1])
    path = out/"combined.pt"
    torch.save(edits[0] | edits[1], path)
    ref = {"path":str(path), "sha256":digest(path)}
    save_json(out/"freeze.json", dict(combined=ref, parents=[cfg["bias"],cfg["ab"]], source_sha256=digest(__file__)))
    distill_comparison(cfg, out, ref)
    template = read_json(cfg["ta_template"])
    aga = out/"aga"
    run_module("recover","--target",cfg["target"],"--method","aga","--aligned-reference","--publication",path,
               "--sha256",ref["sha256"],"--protocol",cfg["ta_template"],"--output",aga)
    spec = {"target":cfg["target"], "release":ref, "AGA_edits":{"path":str(aga/"edits.pt"),"sha256":digest(aga/"edits.pt")}}
    for group, intervention in [("ta",{}), ("bias_cleanup",{"published_bias_projection":True}),
                                ("B_reset",{"public_replacement":{"layer":11,"side":"B+b"}})]:
        conf = copy.deepcopy(template)
        conf.update(output=str(out/group), defenses={"Combined":spec})
        conf["cases"] = [{"target":cfg["target"],"partner":cfg["partner"],"defenses":["Combined"], **intervention}]
        conf["legacy"] = {} if intervention else {k:v for k,v in conf["legacy"].items() if k.endswith("/Unprotected")}
        if intervention:
            conf["attack_scopes"] = ["None"]
        protocol = out/(group+".json")
        save_json(protocol, conf)
        run_module("main_table","--protocol",protocol)
        run_module("main_table","--verify",out/group/"full"/"00_MNIST_SVHN")
    print("All combined controls completed",flush=True)


if __name__ == "__main__":
    main()
