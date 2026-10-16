"""Frozen-defense evaluation: official TA/TIES, with cached original-author AGA.

No defense construction, coefficient selection, training, or test-time adaptation.
"""

import argparse
import ast
import copy
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import checkpoint_loader, digest, evaluate, merge, paired_counts
from covariance_noise import dataset
from frozen_AB_target_only import paired_logp
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels
from verify_probe_candidate import explicit_merge


def load_ties(spec):
    """Execute unchanged pure author functions, excluding unrelated T5 imports."""
    assert digest(spec['path']) == spec['sha256']
    names = {'topk_values_mask', 'resolve_zero_signs', 'normfrac_based_sign',
             'normmass_based_sign', 'resolve_sign', 'disjoint_merge', 'aggregate',
             'merge_methods', 'greater_than_std_mask', 'topk_mask_preserve_normfrac'}
    tree = ast.parse(Path(spec['path']).read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in nodes} == names
    env = dict(torch=torch, logger=logging.getLogger('TIES'))
    exec(compile(ast.Module(body=nodes, type_ignores=[]), spec['path'], 'exec'), env)
    fixture = np.array([[1., -2., 3., -4., 5., -6., 7., -8., 9., -10.],
                        [-2., 1., -4., 3., -6., 5., -8., 7., -10., 9.]])
    k = fixture.shape[1] - int(fixture.shape[1]*spec['topk_percent']/100)
    threshold = np.partition(np.abs(fixture), k-1, axis=1)[:, k-1, None]
    trimmed = fixture * (np.abs(fixture) >= threshold)
    signs = np.sign(trimmed.sum(0)); signs[signs == 0] = np.sign(signs.sum())
    selected = trimmed * np.where(signs[None] > 0, trimmed > 0, trimmed < 0)
    expected = selected.sum(0) / np.maximum((selected != 0).sum(0), 1)
    actual = env['merge_methods']('topk', torch.from_numpy(fixture),
        reset_thresh=spec['topk_percent'], resolve_method='mass', merge_func='dis-mean')
    np.testing.assert_array_equal(actual.numpy(), expected)
    return env['merge_methods']


@torch.no_grad()
def fuse(anchor, left, right, method, cfg, ties, independent=False):
    if method == 'TA':
        fn = explicit_merge if independent else merge
        return fn(anchor, left, right, cfg['TA_coefficient'])
    sa, sl, sr = [m.state_dict() for m in (anchor, left, right)]
    # Official load_vit uses sorted, globally flattened state dictionaries.
    keys = sorted(k for k, v in sa.items() if v.is_floating_point())
    vectors = torch.stack([torch.cat([(s[k] - sa[k]).flatten() for k in keys])
                           for s in (sl, sr)])
    delta = ties('topk', vectors, reset_thresh=cfg['TIES']['topk_percent'],
                 resolve_method='mass', merge_func='dis-mean')
    state = {k: v.clone() for k, v in sa.items()}
    offset = 0
    for k in keys:
        n = sa[k].numel()
        state[k] = sa[k] + cfg['TIES']['coefficient'] * delta[offset:offset+n].reshape_as(sa[k])
        offset += n
    assert offset == delta.numel()
    result = copy.deepcopy(anchor)
    result.load_state_dict(state, strict=True)
    return result


def setup(cfg):
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert '3090' in torch.cuda.get_device_name()
    assert digest('external/secure-merge-attack/utils_attack_B32.py') == cfg['AGA']['sha256']
    for p, sha in cfg['source_sha256'].items():
        assert digest(p) == sha, p
    load = checkpoint_loader(Path('external/secure-merge-attack'))
    root = Path('checkpoints/ViT-B-32')
    pins = {x['path'].removeprefix('ViT-B-32/'): x['sha256']
            for x in json.loads(Path('configs/checkpoints.json').read_text())['files']}

    def checked(name):
        assert digest(root/name) == pins[name]
        model = load(str(root/name)).cpu().eval()
        model.requires_grad_(False)
        return model

    anchor = checked('zeroshot.pt')
    originals = {t: checked(f'{t}/finetuned.pt') for t in cfg['tasks']}
    heads = {t: checked(f'head_{t}.pt') for t in cfg['tasks']}
    aligned = {t: apply(originals[t], cfg['AGA_references'][t], full=True) for t in cfg['tasks']}
    return anchor, originals, heads, aligned, load_ties(cfg['TIES'])


def apply(original, ref, full=False):
    assert digest(ref['path']) == ref['sha256'], ref['path']
    state = torch.load(ref['path'], map_location='cpu', weights_only=False)
    result = copy.deepcopy(original)
    result.load_state_dict(state if full else original.state_dict() | state, strict=True)
    result.requires_grad_(False)
    return result


def row_metrics(scores, labels, before, classes):
    prediction = scores.argmax(1)
    correct = prediction.eq(labels)
    pair = paired_counts(before.argmax(1).eq(labels), correct)
    # Paired normal interval describes image-sampling uncertainty, not seed variation.
    differences = correct.double() - before.argmax(1).eq(labels).double()
    half = 1.96 * float(differences.std(unbiased=True)) / len(labels)**.5
    return dict(n=len(labels), correct=int(correct.sum()), accuracy_percent=100*float(correct.double().mean()),
                changed_predictions=int(prediction.ne(before.argmax(1)).sum()), paired=pair,
                mcnemar_log10_p=paired_logp(pair['lost'], pair['gained']),
                paired_accuracy_delta_95ci_pp=[100*(float(differences.mean())-half),
                                               100*(float(differences.mean())+half)],
                per_class=[dict(class_id=c, n=int(labels.eq(c).sum()),
                                correct=int((correct & labels.eq(c)).sum()),
                                original_correct=int((before.argmax(1).eq(labels) & labels.eq(c)).sum()))
                           for c in range(classes)])


@torch.no_grad()
def run(protocol, case_index, pilot):
    cfg = json.loads(Path(protocol).read_text())
    case = cfg['cases'][case_index]
    out = Path(cfg['output']) / ('pilot' if pilot else 'full') / f'{case_index:02d}_{case["target"]}_{case["partner"]}'
    out.mkdir(parents=True, exist_ok=False)
    (out/'artifacts').mkdir()
    save_json(out/'protocol.json', cfg)
    start = time.monotonic()
    anchor, originals, heads, aligned, ties = setup(cfg)
    target, partner = case['target'], case['partner']
    data = {t: dataset(t, 'data', originals[t].val_preprocess) for t in (target, partner)}
    indices = {t: np.linspace(0, len(d)-1, cfg['pilot_n'], dtype=int).tolist() if pilot else list(range(len(d)))
               for t, d in data.items()}
    rows, raw, baseline = [], {}, {}
    loaded_legacy = {}

    def record(model, defense, method, scope, task, legacy=None):
        key = '/'.join((defense, method, scope, task))
        imported = None
        if not pilot and legacy:
            ref, old_key = legacy
            if ref['path'] not in loaded_legacy:
                assert digest(ref['path']) == ref['sha256']
                loaded_legacy[ref['path']] = torch.load(ref['path'], map_location='cpu', weights_only=False)
            imported = loaded_legacy[ref['path']].get(old_key)
        if imported is not None and 'logits' not in imported:
            imported = None  # Old prediction-only records cannot supply real scores.
        if imported is not None:
            scores, y = imported['logits'], imported['target']
            metric = dict(evaluation_seconds=0., imported_from=legacy[0]['path'], imported_key=legacy[1])
        else:
            sink, logits = [], []
            metric = evaluate(model, heads[task], make_loader(data[task], indices[task]), 'cuda',
                              prediction_sink=lambda p, y: sink.append((p, y)), logit_sink=logits.append)
            scores, y = torch.cat(logits), torch.cat([r[1] for r in sink])
        expected = official_labels(task)[indices[task]]
        np.testing.assert_array_equal(y.numpy(), expected)
        assert scores.shape == (len(expected), cfg['tasks'][task]['classes']) and torch.isfinite(scores).all()
        ref_key = '/'.join((method, scope, task))
        if defense == 'Unprotected': baseline[ref_key] = scores
        row = dict(defense=defense, target=target, partner=partner, merger=method,
                   attack_scope=scope, dataset=task, role='target' if task == target else 'partner',
                   **row_metrics(scores, y, baseline[ref_key], cfg['tasks'][task]['classes']),
                   evaluation_seconds=metric['evaluation_seconds'],
                   provenance={k: v for k, v in metric.items() if k.startswith('imported')})
        raw[key] = dict(logits=scores, target=y, indices=indices[task])
        rows.append(row)
        save_json(out/'results.json', rows)
        torch.save(raw, out/'artifacts/predictions.pt')
        print('PHASE3_ROW', defense, method, scope, task, row['correct'], row['n'], flush=True)

    for name in ['Unprotected'] + case['defenses']:
        spec = cfg['defenses'].get(name)
        own = originals[target] if spec is None else apply(originals[target], spec['release'], spec.get('full_state', False))
        attacked = aligned[target] if spec is None else apply(aligned[target], spec['AGA_edits'])
        legacy = cfg['legacy'].get(f'{target}_{partner}/{name}', {})

        def old(stage, task):
            ref = legacy.get(stage)
            if not ref: return None
            key = ref.get('key_template', '{preset}/{stage}/{task}').format(
                preset=ref.get('preset', 'released'), stage=ref.get('stage', stage), task=task)
            return ref, key

        record(own, name, 'Standalone', 'None', target, old('standalone', target))
        for method in cfg['mergers']:
            for scope, left, right in [('None', own, originals[partner]),
                                       ('Protected expert', attacked, originals[partner]),
                                       ('All experts', attacked, aligned[partner])]:
                fused = fuse(anchor, left, right, method, cfg, ties)
                stage = {'None': 'direct', 'Protected expert': 'AGA_target_only', 'All experts': 'AGA_both'}[scope]
                for t in (target, partner):
                    record(fused, name, method, scope, t, old(stage, t) if method == 'TA' else None)
                del fused
        del own, attacked
    save_json(out/'execution.json', dict(completed=True, pilot=pilot, case_index=case_index,
              rows=len(rows), wall_seconds=time.monotonic()-start, GPU=torch.cuda.get_device_name(),
              new_defense_weights=0, optimizer_steps=0, new_AGA_calls=0,
              cached_author_AGA_weights=True, defense_feedback=False,
              predictions_sha256=digest(out/'artifacts/predictions.pt'), source_sha256=digest(__file__),
              protocol_sha256=digest(out/'protocol.json')))


@torch.no_grad()
def verify(output):
    out = Path(output)
    cfg = json.loads((out/'protocol.json').read_text())
    meta = json.loads((out/'execution.json').read_text())
    assert meta['completed'] and digest(out/'artifacts/predictions.pt') == meta['predictions_sha256']
    anchor, originals, heads, aligned, ties = setup(cfg)
    case = cfg['cases'][meta['case_index']]
    target, partner = case['target'], case['partner']
    data = {t: dataset(t, 'data', originals[t].val_preprocess) for t in (target, partner)}
    raw = torch.load(out/'artifacts/predictions.pt', map_location='cpu', weights_only=False)
    rows = json.loads((out/'results.json').read_text())
    replays, maximum_error = 0, 0.
    for name in ['Unprotected'] + case['defenses']:
        spec = cfg['defenses'].get(name)
        own = originals[target] if spec is None else apply(originals[target], spec['release'], spec.get('full_state', False))
        attacked = aligned[target] if spec is None else apply(aligned[target], spec['AGA_edits'])
        groups = [('Standalone', 'None', own, [target])]
        for method in cfg['mergers']:
            for scope, left, right in [('None', own, originals[partner]), ('Protected expert', attacked, originals[partner]),
                                       ('All experts', attacked, aligned[partner])]:
                groups.append((method, scope, fuse(anchor, left, right, method, cfg, ties, True), [target, partner]))
        for method, scope, model, tasks in groups:
            for task in tasks:
                key = '/'.join((name, method, scope, task)); r = raw[key]
                y = official_labels(task)[r['indices']]; scores = r['logits'].numpy(); pred = scores.argmax(1)
                np.testing.assert_array_equal(y, r['target'].numpy()); assert np.isfinite(scores).all()
                row = next(v for v in rows if (v['defense'], v['merger'], v['attack_scope'], v['dataset']) == (name, method, scope, task))
                before = raw['/'.join(('Unprotected', method, scope, task))]['logits'].numpy().argmax(1)
                lost, gained = int(((before == y) & (pred != y)).sum()), int(((before != y) & (pred == y)).sum())
                assert int((pred == y).sum()) == row['correct']
                assert (lost, gained) == (row['paired']['lost'], row['paired']['gained'])
                assert int((before != pred).sum()) == row['changed_predictions']
                for c in row['per_class']:
                    mask = y == c['class_id']
                    assert int(mask.sum()) == c['n'] and int(((pred == y) & mask).sum()) == c['correct']
                ids = np.linspace(0, len(y)-1, cfg['replay_n'], dtype=int)
                images = torch.stack([data[task][r['indices'][int(i)]][0] for i in ids]).cuda()
                model.cuda(); heads[task].cuda(); actual = heads[task](model(images)).cpu(); model.cpu(); heads[task].cpu()
                np.testing.assert_array_equal(actual.argmax(1).numpy(), pred[ids])
                error = float((actual.double() - r['logits'][ids].double()).abs().max())
                assert error <= cfg['score_tolerance'], (key, error)
                maximum_error = max(maximum_error, error); replays += len(ids)
        del groups, own, attacked
    save_json(out/'independent_validation.json', dict(status='passed', official_label_numpy_recount=True,
              full_image_prediction_replays=replays, maximum_score_error=maximum_error,
              explicit_TA_reconstruction=True, original_TIES_functions=True,
              independent_TIES_small_numpy_fixture=True, independent_full_TIES_reimplementation=False, rows=len(rows),
              raw_sha256=meta['predictions_sha256'], source_sha256=digest(__file__)))
    print('PHASE3_VERIFIED', out, replays, flush=True)


@torch.no_grad()
def audit(protocol):
    """Read-only finite utility/probe audit; exposed samples never repair a release."""
    cfg = json.loads(Path(protocol).read_text()); start = time.monotonic()
    anchor, originals, heads, _, _ = setup(cfg)
    out = Path(cfg['output'])/'utility'
    out.mkdir(parents=True, exist_ok=False)
    save_json(out/'protocol.json', cfg)
    raw, rows = {}, []
    for task, task_spec in cfg['tasks'].items():
        own, head = originals[task], heads[task]
        train = dataset(task, 'data', own.val_preprocess, True)
        ids = np.random.default_rng(cfg['utility_seed']).choice(len(train), cfg['utility_n'], replace=False).tolist()
        loader = make_loader(train, ids)
        scores = []; evaluate(own, head, loader, 'cuda', logit_sink=scores.append)
        teacher = torch.cat(scores).double(); yy = teacher.argmax(1, keepdim=True)
        gaps = teacher.gather(1, yy)-teacher; positive = gaps > 0
        bases = {}
        for suffix in ['attn.out_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']:
            key = 'model.visual.transformer.resblocks.11.'+suffix
            delta = (own.state_dict()[key]-anchor.state_dict()[key]).double().cuda()
            _, singular, vh = torch.linalg.svd(delta, full_matrices=False)
            bases[key] = (vh[:cfg['probe_rank']].T.cpu(), singular[:cfg['probe_rank']].cpu())
        raw[task] = dict(indices=ids, teacher_logits=teacher)
        for name, spec in cfg['defenses'].items():
            if spec['target'] != task: continue
            release = apply(own, spec['release'], spec.get('full_state', False))
            scores = []; evaluate(release, head, loader, 'cuda', logit_sink=scores.append)
            after = torch.cat(scores).double(); ds = after-teacher
            relative = torch.zeros_like(gaps)
            relative[positive] = ((ds.gather(1, yy)-ds)[positive]/gaps[positive]).abs()
            matrix_cost, tensor_cost, zero_denominators, probes = [], [], [], []
            for key, value in own.state_dict().items():
                difference = release.state_dict()[key].double()-value.double()
                if not bool(difference.any()): continue
                norm = float(value.double().norm())
                if norm == 0: zero_denominators.append(key); continue
                cost = float(difference.norm())**2/norm**2
                tensor_cost.append(cost)
                if value.ndim == 2: matrix_cost.append(cost)
                if key in bases:
                    v, singular = bases[key]
                    action = difference@v
                    probes.append(dict(key=key, task_matrix_shape=list(value.shape),
                        input_probe_shape=list(v.T.shape), singular_values=singular.tolist(),
                        relative_linear_probe_action=float(action.norm()/(value.double()@v).norm()),
                        probe_action_frobenius=float(action.norm())))
            raw[task][name] = after
            row = dict(defense=name, target=task, n=len(ids), split='train',
                       purpose='Read-only finite audit, not new confirmation or tuning; historical overlap possible.',
                       parameter_distance_matrices=sum(matrix_cost)**.5,
                       parameter_distance_changed_tensors=sum(tensor_cost)**.5 if not zero_denominators else None,
                       zero_original_norm_tensors=zero_denominators,
                       worst_relative_task_margin_drift=float(relative.max()),
                       n_inputs_over_task_budget=int((relative.max(1).values>cfg['task_budget']).sum()),
                       changed_teacher_predictions=int(teacher.argmax(1).ne(after.argmax(1)).sum()),
                       probes=probes, probe_scope='Selected final-block linear task-SVD actions only; not full functional preservation.',
                       prior_budget_status=spec['budget_status'])
            rows.append(row); save_json(out/'results.json', rows)
            print('PHASE3_UTILITY', name, row['worst_relative_task_margin_drift'], flush=True)
    torch.save(raw, out/'scores.pt')
    # Independent NumPy calculation from the saved FP64 teacher/release scores.
    for row in rows:
        record = raw[row['target']]; a = record['teacher_logits'].numpy(); b = record[row['defense']].numpy()
        k = a.argmax(1); gaps = a[np.arange(len(a)), k, None]-a
        changes = (b-a)[np.arange(len(a)), k, None]-(b-a)
        value = float(np.max(np.abs(changes[gaps>0]/gaps[gaps>0])))
        assert abs(value-row['worst_relative_task_margin_drift']) < 1e-10
    save_json(out/'independent_validation.json', dict(status='passed_numpy_score_recount',
              new_weights=0, optimizer_steps=0, source_sha256=digest(__file__),
              scores_sha256=digest(out/'scores.pt'), wall_seconds=time.monotonic()-start,
              global_preservation_proved=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol', default='configs/phase3.json'); p.add_argument('--case', type=int, default=0)
    p.add_argument('--pilot', action='store_true'); p.add_argument('--verify')
    p.add_argument('--audit', action='store_true')
    args = p.parse_args()
    if args.verify: verify(args.verify)
    elif args.audit: audit(args.protocol)
    else: run(args.protocol, args.case, args.pilot)
