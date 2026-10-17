"""NUFILT author kernels with an explicit OpenAI CLIP weight adapter.

Fixed two-order MNIST/SVHN evaluation. No images enter merger optimization;
original experts and defenses are frozen. This is not the FusionBench app/table.
"""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from clip_benchmark import digest, evaluate
from covariance_noise import dataset
from phase3_evaluate import apply, row_metrics, setup
from phase3_nufilt_check import load_official
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels


def selection(state):
    """HF registration order: k, v, q, out, fc1; native packed rows are q,k,v."""
    for i in range(12):
        prefix = f'model.visual.transformer.resblocks.{i}.'
        key = prefix+'attn.in_proj_weight'; width = state[key].shape[1]
        for part in [1, 2, 0]: yield key, part*width, (part+1)*width
        for suffix in ['attn.out_proj.weight', 'mlp.c_fc.weight']:
            key = prefix+suffix
            yield key, 0, state[key].shape[0]


def merge_nufilt(anchor, inputs, author, output):
    """Same per-module construction, global Adam and final matrix as upstream."""
    torch.manual_seed(42)
    initial = anchor.state_dict()
    state = {k: v.clone() for k, v in initial.items()}
    selected = list(selection(initial)); assert len(selected) == 60
    evidence, history = [], []
    start = time.monotonic()
    for step, model in enumerate(inputs):
        task_state = model.state_dict(); modules = []
        for key, a, b in selected:
            current = state[key][a:b].cuda()
            original = initial[key][a:b].cuda()
            task = task_state[key][a:b].cuda()-original
            base = nn.Linear(current.shape[1], current.shape[0], bias=False, device='meta')
            base.weight = nn.Parameter(current, requires_grad=False)
            modules.append(author['FilterLoRA'](base, task, True, current-original, 64, 128, 8, 'cuda'))
        if step > 0:
            params = [p for m in modules for n, p in m.named_parameters() if p.requires_grad and 'gate' in n]
            optimizer = torch.optim.Adam(params, lr=.001)
            for iteration in range(50):
                loss = sum(m.solve_lora() for m in modules)
                loss.backward(); optimizer.step(); optimizer.zero_grad()
                if iteration in [0, 49]: history.append(dict(task=step, iteration=iteration, loss=float(loss.detach())))
        gates = []
        with torch.no_grad():
            for (key, a, b), m in zip(selected, modules):
                fused = m.merge_to_base().cpu()
                gates.append({k: v.detach().cpu().clone() for k, v in
                              [('U', m.gate.U), ('A', m.gate.A), ('B', m.gate.B)]})
                # Independent finite matrix computation; no temporary gate forward.
                u, aa, bb = [gates[-1][k].double() for k in ['U', 'A', 'B']]
                transform = torch.eye(aa.shape[1], dtype=torch.double)-u@u.T+bb@aa
                expected = state[key][a:b].double()+(task_state[key][a:b]-initial[key][a:b]).double()@transform
                torch.testing.assert_close(fused.double(), expected, atol=2e-5, rtol=2e-5)
                state[key][a:b] = fused
        evidence.append(gates)
        del modules
    selected_keys = {x[0] for x in selected}
    assert all(torch.equal(v, state[k]) for k, v in initial.items() if k not in selected_keys)
    torch.save(dict(weights={k: state[k] for k in selected_keys}, gates=evidence), output)
    merged = copy.deepcopy(anchor); merged.load_state_dict(state, strict=True)
    return merged, dict(merging_seconds=time.monotonic()-start, optimizer_steps=50,
                        selected_modules=60, skipped_tensors_preserved=True, loss_history=history,
                        artifact_sha256=digest(output), matrix_reconstruction_passed=True)


def run(full=False, protocol='configs/phase3.json'):
    cfg = json.loads(Path(protocol).read_text()); candidate = cfg.get('nufilt_defense', 'A/B')
    author, pins = load_official()
    root = Path(cfg.get('nufilt_output', 'outputs/phase3/nufilt')); root.mkdir(parents=True, exist_ok=True)
    previous = cfg.get('nufilt_controls'); controls_raw = None
    if previous:
        ref = previous['full' if full else 'pilot']; assert digest(ref['path']) == ref['sha256']
        controls_raw = torch.load(ref['path'], map_location='cpu', weights_only=False)
        ref = previous['construction']; assert digest(ref['path']) == ref['sha256']
        old_costs = {c['key']: c for c in json.loads(Path(ref['path']).read_text())['costs']}
    frozen = dict(seed=42, orders=[['MNIST', 'SVHN'], ['SVHN', 'MNIST']],
        defenses=['Unprotected', 'Phase1_MNIST', candidate], scopes=['None', 'Protected expert', 'All experts'],
        defaults=dict(lora_rank=64, null_rank=128, probe_rank=8, steps=50, lr=.001),
        source_sha256={__file__: digest(__file__), 'experiments/phase3_nufilt_check.py': digest('experiments/phase3_nufilt_check.py')},
        author_source_sha256=pins, parent_protocol_sha256=digest(protocol), reused_controls=previous,
        adapter='Selected weights only; packed QKV split/merge, k/v/q/out/fc1 order. All skipped tensors from public anchor.',
        scope='Official kernels and native weight adapter, not full FusionBench execution or original multi-task table.',
        data='No optimization images. Historically exposed evaluation sets; no method/hyperparameter selection.',
        phase1_reuse='Selected inputs identical: assert exact equality before reusing merged model and scores.',
        pilot_n=64, full_replay_n=16, score_tolerance=.0005,
        gradient_role='Attacker merger gates only; 50 global backwards per distinct two-task merge. No expert or defense optimization.')
    protocol = root/'protocol.json'
    if protocol.exists(): assert json.loads(protocol.read_text()) == frozen
    else: save_json(protocol, frozen)
    out = root/('full' if full else 'pilot'); out.mkdir(exist_ok=False)
    if full: assert json.loads((root/'pilot/validation.json').read_text())['status'] == 'passed'
    pilot_costs = {c['key']: c for c in json.loads((root/'pilot/execution.json').read_text())['costs']} if full else {}
    anchor, experts, heads, aligned, _ = setup(cfg)
    start = time.monotonic(); rows, raw, costs = [], {}, []
    data = {t: dataset(t, 'data', experts[t].val_preprocess) for t in ['MNIST', 'SVHN']}
    ids = {t: list(range(len(d))) if full else np.linspace(0, len(d)-1, 64, dtype=int).tolist() for t, d in data.items()}
    # QKV split and repack must preserve every original tensor bit-for-bit.
    for model in [anchor]+list(experts.values()):
        state = model.state_dict()
        for i in range(12):
            w = state[f'model.visual.transformer.resblocks.{i}.attn.in_proj_weight']
            assert torch.equal(torch.cat(w.chunk(3, 0), 0), w)
    for order_id, order in enumerate(frozen['orders']):
        for scope_id, scope in enumerate(frozen['scopes']):
            controls = {}
            for defense in ['Unprotected', candidate, 'Phase1_MNIST']:
                target = experts['MNIST'] if scope == 'None' else aligned['MNIST']
                if defense != 'Unprotected':
                    spec = cfg['defenses'][defense]
                    target = apply(target, spec['release'] if scope == 'None' else spec['AGA_edits'])
                peer = aligned['SVHN'] if scope == 'All experts' else experts['SVHN']
                inputs = {'MNIST': target, 'SVHN': peer}
                key = f'{order_id}_{scope_id}_{defense.replace("/", "_")}'
                artifact = root/f'{order_id}_{scope_id}_{"Unprotected" if defense == "Phase1_MNIST" else defense.replace("/", "_")}.pt'
                if previous and defense in ['Unprotected', 'Phase1_MNIST']:
                    artifact = Path(previous['root'])/f'{order_id}_{scope_id}_Unprotected.pt'
                    assert digest(artifact) == old_costs[f'{order_id}_{scope_id}_Unprotected']['artifact_sha256']
                if defense == 'Phase1_MNIST':
                    ref = experts['MNIST'] if scope == 'None' else aligned['MNIST']
                    assert all(torch.equal(target.state_dict()[k][a:b], ref.state_dict()[k][a:b]) for k, a, b in selection(ref.state_dict()))
                    # Already measured exact same merged model; no fabricated new forward.
                    scores_by_task = controls
                    cost = dict(reused_identical_selected_inputs=True, optimizer_steps=0, artifact_sha256=digest(artifact))
                elif previous and defense == 'Unprotected':
                    prior_key = f'{order_id}_{scope_id}_Unprotected'
                    scores_by_task = {t: controls_raw[prior_key+'/'+t] for t in data}
                    assert all(scores_by_task[t]['indices'] == ids[t] for t in data)
                    cost = dict(reused_frozen_control_scores=True, optimizer_steps=0, artifact_sha256=digest(artifact))
                else:
                    if full:
                        assert digest(artifact) == pilot_costs[key]['artifact_sha256']
                        record = torch.load(artifact, map_location='cpu', weights_only=False)
                        fused = copy.deepcopy(anchor); fused.load_state_dict(anchor.state_dict() | record['weights'], strict=True)
                        cost = dict(reused_frozen_pilot_merger_weights=True, optimizer_steps=0, artifact_sha256=digest(artifact))
                    else:
                        assert not artifact.exists()
                        fused, cost = merge_nufilt(anchor, [inputs[t] for t in order], author, artifact)
                    scores_by_task = {}
                    for t in data:
                        logits, labels = [], []
                        timing = evaluate(fused, heads[t], make_loader(data[t], ids[t]), 'cuda',
                                 prediction_sink=lambda p, y: labels.append(y), logit_sink=logits.append)
                        scores_by_task[t] = dict(logits=torch.cat(logits), target=torch.cat(labels), indices=ids[t],
                                                evaluation_seconds=timing['evaluation_seconds'])
                        assert scores_by_task[t]['logits'].shape == (len(ids[t]), 10)
                        assert torch.isfinite(scores_by_task[t]['logits']).all()
                        np.testing.assert_array_equal(scores_by_task[t]['target'].numpy(), official_labels(t)[ids[t]])
                    del fused
                if defense == 'Unprotected': controls = scores_by_task
                costs.append(dict(key=key, **cost))
                for t, score in scores_by_task.items():
                    entry = dict(defense=defense, target='MNIST', partner='SVHN', merger='NUFILT', order=order,
                        attack_scope=scope, dataset=t, role='target' if t == 'MNIST' else 'partner',
                        evaluation_seconds=0. if defense == 'Phase1_MNIST' or cost.get('reused_frozen_control_scores') else score['evaluation_seconds'],
                        measurement='Reused identical selected-input model' if defense == 'Phase1_MNIST' else 'Reused frozen matched control' if cost.get('reused_frozen_control_scores') else 'Full forward',
                        **row_metrics(score['logits'], score['target'], controls[t]['logits'], 10))
                    rows.append(entry); raw[key+'/'+t] = score
                    print('NUFILT_ROW', order_id, defense, scope, t, entry['correct'], entry['n'], flush=True)
                save_json(out/'results.json', rows)
                torch.save(raw, out/'scores.pt')
    # Independent NumPy class/paired recount and fresh full-model score replays.
    maximum_error = 0.
    for row in rows:
        order_id = frozen['orders'].index(row['order']); scope_id = frozen['scopes'].index(row['attack_scope'])
        key = f'{order_id}_{scope_id}_{row["defense"].replace("/", "_")}/{row["dataset"]}'
        score = raw[key]; pred = score['logits'].numpy().argmax(1); labels = score['target'].numpy()
        assert int((pred == labels).sum()) == row['correct']
        before = raw[f'{order_id}_{scope_id}_Unprotected/{row["dataset"]}']['logits'].numpy().argmax(1)
        assert int((before != pred).sum()) == row['changed_predictions']
        assert int(((before == labels) & (pred != labels)).sum()) == row['paired']['lost']
        assert int(((before != labels) & (pred == labels)).sum()) == row['paired']['gained']
        for c in row['per_class']: assert int(((pred == labels) & (labels == c['class_id'])).sum()) == c['correct']
        name = 'Unprotected' if row['defense'] == 'Phase1_MNIST' else row['defense'].replace('/', '_')
        artifact_root = Path(previous['root']) if previous and name == 'Unprotected' else root
        record = torch.load(artifact_root/f'{order_id}_{scope_id}_{name}.pt', map_location='cpu', weights_only=False)
        fused = copy.deepcopy(anchor); fused.load_state_dict(anchor.state_dict() | record['weights'], strict=True)
        loc = np.linspace(0, len(labels)-1, 16, dtype=int)
        error = 0.
        for block in sorted(set(loc//128)):
            start, stop = block*128, min((block+1)*128, len(labels)); wanted = loc[(loc >= start) & (loc < stop)]
            logits = []; evaluate(fused, heads[row['dataset']], make_loader(data[row['dataset']], score['indices'][start:stop]), 'cuda', logit_sink=logits.append)
            error = max(error, float((torch.cat(logits)[wanted-start]-score['logits'][wanted]).abs().max()))
        maximum_error = max(maximum_error, error)
        assert error <= .0005, (key, error)
        del fused
    save_json(out/'execution.json', dict(wall_seconds=time.monotonic()-start, costs=costs,
        source_sha256=digest(__file__), source_protocol_sha256=digest(protocol), scores_sha256=digest(out/'scores.pt'),
        GPU=torch.cuda.get_device_name(), defense_updates=0, full=full))
    save_json(out/'validation.json', dict(status='passed', rows=len(rows), full_replay_count=len(rows)*16,
        maximum_score_error=maximum_error, numpy_counts=True, selected_weight_identity_checked=True,
        evidence_scope='Native adapter and author kernels; no second complete FusionBench implementation.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--full', action='store_true'); parser.add_argument('--protocol', default='configs/phase3.json'); args = parser.parse_args()
    run(args.full, args.protocol)
