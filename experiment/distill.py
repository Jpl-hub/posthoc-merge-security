"""Fixed-budget teacher-label recovery; reuse the established Oct19 training rule.

Extra unlabeled task data and student training; no updates to the defender.
This is a disclosed native adaptation, not the official PaRaMS KD implementation.
"""

import argparse
import ast
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from _impl.attacks import author_recover, load_author
from _impl.data import dataset
from _impl.io import check_files, digest, new_output, save_json
from _impl.merge import parameter_hash
from _impl.models import apply, initialize, load_checkpoint
from cfg import require_root

def train(base, head, images, labels, cfg, pilot=False):
    """Same public initialization and fixed training rule for each label group."""
    torch.manual_seed(cfg['training_seed'])
    torch.cuda.manual_seed_all(cfg['training_seed'])
    student = copy.deepcopy(base).cuda().train().requires_grad_(False)
    student.model.visual.requires_grad_(True)
    head = head.cuda().eval().requires_grad_(False)
    params = [p for p in student.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    path = Path(cfg['schedule']['path'])
    assert digest(path) == cfg['schedule']['sha256']
    names = {'assign_learning_rate', '_warmup_lr', 'cosine_lr'}
    nodes = [n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    env = {'np': np}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), env)
    steps = cfg['epochs'] * ((len(images) + cfg['batch_size'] - 1) // cfg['batch_size'])
    scheduler = env['cosine_lr'](opt, cfg['lr'], cfg['warmup_steps'], steps)
    rng = torch.Generator().manual_seed(cfg['training_seed'])
    history = []
    step = 0
    start = time.monotonic()
    head_before = parameter_hash({k: v.cpu() for (k, v) in head.state_dict().items()})
    torch.cuda.reset_peak_memory_stats()
    for epoch in range(1 if pilot else cfg['epochs']):
        order = torch.randperm(len(images), generator=rng)
        losses = []
        for ids in [order[:8]] if pilot else order.split(cfg['batch_size']):
            scheduler(step)
            opt.zero_grad(set_to_none=True)
            logits = head(student(images[ids].cuda()))
            loss = torch.nn.functional.cross_entropy(logits, labels[ids].cuda())
            assert torch.isfinite(loss)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            assert torch.isfinite(norm) and norm > 0
            assert all((p.grad is None for p in head.parameters()))
            opt.step()
            losses.append(float(loss.detach()))
            step += 1
        history.append({'epoch': epoch + 1, 'mean_pseudo_label_loss': float(np.mean(losses))})
        print('KD_EPOCH', epoch + 1, history[-1]['mean_pseudo_label_loss'], flush=True)
    assert head_before == parameter_hash({k: v.cpu() for (k, v) in head.state_dict().items()})
    torch.cuda.synchronize()
    seconds = time.monotonic() - start
    peak = torch.cuda.max_memory_allocated()
    student.cpu().eval().requires_grad_(False)
    head.cpu()
    del opt
    assert step == (1 if pilot else steps)
    return (student, dict(completed=True, backward_steps=step, training_seconds=seconds, peak_gpu_bytes=peak, trainable_student_parameters=sum((p.numel() for p in params)), history=history, expert_updates=0, defense_updates=0, head_updates=0, selection_from_test=False))


def save_student(model, base, path):
    """Store changed tensors against the public base; reconstruct exactly."""
    state, anchor = model.state_dict(), base.state_dict()
    edits = {k: v for k, v in state.items() if not torch.equal(v, anchor[k])}
    reconstructed = anchor | edits
    assert all(torch.equal(v, reconstructed[k]) for k, v in state.items())
    torch.save(edits, path)
    return dict(path=str(path), sha256=digest(path), reference="public_base")


def run(protocol):
    cfg = json.loads(Path(protocol).read_text())
    check_files(cfg["source_sha256"])
    check_files(cfg.get("historical_inputs", {}))
    template = json.loads(Path(cfg["evaluation_template"]).read_text())
    parent = {k: template[k] for k in ["tasks", "TA_coefficient", "pilot_n", "replay_n",
              "score_tolerance", "AGA_references", "AGA"]}
    out = new_output(cfg["output"], cfg)
    for path in cfg["source_sha256"]:
        dest = out / "source_archive" / path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(Path(path).read_bytes())
    initialize()
    torch.use_deterministic_algorithms(True)
    base = load_checkpoint("zeroshot.pt")
    original = load_checkpoint(f"{cfg['target']}/finetuned.pt")
    head = load_checkpoint(f"head_{cfg['target']}.pt")
    teacher = apply(original, cfg["release"])
    original_hash = parameter_hash(original.state_dict())
    teacher_hash = parameter_hash(teacher.state_dict())
    base_hash = parameter_hash(base.state_dict())
    history = Path(cfg["historical_output"]) if cfg.get("historical_output") else None
    previous = json.loads((history / "queries.json").read_text()) if history else None
    data = dataset(cfg["target"], "data", original.val_preprocess, training=True)
    assert 0 < cfg["unlabeled_n"] <= len(data)
    indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(cfg["data_seed"]))[:cfg["unlabeled_n"]].tolist()
    images = torch.stack([data[i][0] for i in indices])  # Discard all dataset labels.
    image_sha = hashlib.sha256(images.numpy().tobytes()).hexdigest()
    scores, labels = {}, {}
    if history:
        assert original_hash == previous["teacher_parameter_sha256"]["Unprotected"]
        assert indices == json.loads((history / "training_indices.json").read_text())
        assert image_sha == previous["image_tensor_sha256"]
        old_queries = torch.load(history / "teacher_queries.pt", map_location="cpu", weights_only=False)
        labels["Unprotected"] = old_queries["labels"]["Unprotected"]
    start = time.monotonic()
    for name, model in [("Unprotected", original), ("Defense2", teacher)]:
        if name in labels:
            continue
        with torch.no_grad():
            model.cuda().eval()
            head.cuda().eval()
            scores[name] = torch.cat([head(model(x.cuda())).cpu() for x in images.split(cfg["batch_size"])])
            model.cpu()
            head.cpu()
        assert torch.isfinite(scores[name]).all()
        labels[name] = scores[name].argmax(1)
    torch.save(dict(scores=scores, labels=labels, indices=indices), out / "teacher_queries.pt")
    save_json(out / "training_indices.json", indices)
    save_json(out / "queries.json", dict(
        queries=len(indices)*len(scores), queries_per_teacher=len(indices), real_images=len(indices),
        ground_truth_labels_used=False, teacher_parameter_sha256={"Unprotected":original_hash,"Defense2":teacher_hash},
        image_tensor_sha256=image_sha, raw_sha256=digest(out / "teacher_queries.pt"),
        seconds=time.monotonic()-start,
        prediction_disagreements_with_unprotected=int((labels["Defense2"] != labels["Unprotected"]).sum()),
        reused_unprotected_queries=str(history / "teacher_queries.pt") if history else None))
    students, publications, training, trained_labels = {}, {}, {}, {}
    for label_name in ["Unprotected", "Defense2"]:
        name = "KD_" + label_name
        key = hashlib.sha256(labels[label_name].numpy().tobytes()).hexdigest()
        if key in trained_labels:
            other = trained_labels[key]
            students[name], publications[name] = students[other], publications[other]
            training[name] = dict(reused_from=other, new_backward_steps=0, labels_sha256=key)
        elif history and label_name == "Unprotected":
            students[name] = apply(base, cfg["historical_student"], full=True)
            publications[name] = dict(cfg["historical_student"], full_state=True)
            training[name] = dict(reused_from=str(history), new_backward_steps=0, labels_sha256=key)
        else:
            pilot_student, pilot = train(base, head, images, labels[label_name], cfg, pilot=True)
            del pilot_student
            save_json(out / (name + "_pilot.json"), pilot)
            torch.cuda.empty_cache()
            student, report = train(base, head, images, labels[label_name], cfg)
            students[name] = student
            publications[name] = save_student(student, base, out / (name + ".pt"))
            training[name] = dict(report, parameter_sha256=parameter_hash(student.state_dict()), labels_sha256=key)
        trained_labels[key] = name
        save_json(out / "training.json", training)
    del images
    assert original_hash == parameter_hash(original.state_dict())
    assert teacher_hash == parameter_hash(teacher.state_dict())
    assert base_hash == parameter_hash(base.state_dict())
    save_json(out / "freeze.json", dict(publications=publications, teacher_sha256=teacher_hash,
        all_students_frozen_before_test=True, source_sha256=digest(__file__), protocol_sha256=digest(protocol)))
    parent.update(output=str(out / "evaluation"), source_sha256=cfg["source_sha256"],
        tasks={k:parent["tasks"][k] for k in [cfg["target"],cfg["partner"]]},
        cases=[dict(target=cfg["target"], partner=cfg["partner"], defenses=list(students))],
        defenses={}, mergers=["TA"], attack_scopes=["None","Protected expert","All experts"],
        attacker_data=True, attack_permission=cfg["permission"])
    parent["legacy"] = {f"{cfg['target']}_{cfg['partner']}/Unprotected":
                        cfg.get("unprotected_legacy", template["legacy"].get(f"{cfg['target']}_{cfg['partner']}/Unprotected", {}))}
    for name, ref in publications.items():
        if history and ref["sha256"] == cfg["historical_student"]["sha256"]:
            parent["legacy"][f"{cfg['target']}_{cfg['partner']}/{name}"] = {
                stage: dict(path=str(history / "scores.pt"),
                            sha256=cfg["historical_inputs"][str(history / "scores.pt")],
                            key_template="Unprotected/" + old + "/{task}")
                for stage, old in [("standalone", "Standalone"), ("direct", "TA"),
                                   ("AGA_target_only", "AGA-target-only"), ("AGA_both", "AGA-both")]}
    author, sha = load_author(Path("external/secure-merge-attack") / parent["AGA"]["file"], True)
    assert sha == parent["AGA"]["sha256"]
    recovered = {}
    for name, student in students.items():
        key = publications[name]["sha256"]
        if key not in recovered:
            start = time.monotonic()
            aligned = author_recover(student, base, author)
            recovered[key] = dict(save_student(aligned, base, out / (name + "_AGA.pt")),
                                  seconds=time.monotonic()-start)
        parent["defenses"][name] = dict(target=cfg["target"], release=publications[name],
            full_state=publications[name].get("full_state", False), AGA_edits=recovered[key])
    save_json(out / "evaluation_protocol.json", parent)
    save_json(out / "recovery.json", dict(author_calls=len(recovered), recovered=recovered,
        defense_updates=0, original_expert_updates=0, parameter_selection_from_test=False))
    print("KD_FROZEN", cfg["target"], list(publications), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", required=True)
    args = p.parse_args()
    require_root()
    run(args.protocol)
