"""Recount frozen results and replay selected images in their original batch context.

The original 16-image replay changed GEMM shapes relative to evaluation at 128.
Keep all predictions and the original 5e-4 tolerance; do not edit source results.
"""

import argparse
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import digest
from phase3_evaluate import apply, fuse, setup
from subspace_probe import make_loader, save_json
from covariance_noise import dataset
from verify_covariance_noise import official_labels


@torch.no_grad()
def verify(output):
    out = Path(output); cfg = json.loads((out/'protocol.json').read_text())
    meta = json.loads((out/'execution.json').read_text())
    assert meta['completed'] and digest(out/'artifacts/predictions.pt') == meta['predictions_sha256']
    anchor, experts, heads, aligned, ties = setup(cfg)
    case = cfg['cases'][meta['case_index']]; target, partner = case['target'], case['partner']
    data = {t: dataset(t, 'data', experts[t].val_preprocess) for t in (target, partner)}
    raw = torch.load(out/'artifacts/predictions.pt', map_location='cpu', weights_only=False)
    rows = json.loads((out/'results.json').read_text())
    batch = inspect.signature(make_loader).parameters['batch_size'].default
    maximum, replays, context_images = 0., 0, 0
    for name in ['Unprotected']+case['defenses']:
        spec = cfg['defenses'].get(name)
        own = experts[target] if spec is None else apply(experts[target], spec['release'], spec.get('full_state', False))
        attacked = aligned[target] if spec is None else apply(aligned[target], spec['AGA_edits'])
        groups = [('Standalone', 'None', own, [target])]
        for method in cfg['mergers']:
            for scope, left, right in [('None', own, experts[partner]), ('Protected expert', attacked, experts[partner]), ('All experts', attacked, aligned[partner])]:
                groups.append((method, scope, fuse(anchor, left, right, method, cfg, ties, True), [target, partner]))
        for method, scope, model, tasks in groups:
            for task in tasks:
                key = '/'.join((name, method, scope, task)); score = raw[key]
                logits = score['logits'].numpy(); y = official_labels(task)[score['indices']]; pred = logits.argmax(1)
                np.testing.assert_array_equal(y, score['target'].numpy()); assert np.isfinite(logits).all()
                row = next(r for r in rows if (r['defense'], r['merger'], r['attack_scope'], r['dataset']) == (name, method, scope, task))
                control = raw['/'.join(('Unprotected', method, scope, task))]['logits'].numpy().argmax(1)
                assert int((pred == y).sum()) == row['correct']
                assert int((pred != control).sum()) == row['changed_predictions']
                assert int(((control == y) & (pred != y)).sum()) == row['paired']['lost']
                assert int(((control != y) & (pred == y)).sum()) == row['paired']['gained']
                for c in row['per_class']:
                    mask = y == c['class_id']
                    assert int(mask.sum()) == c['n'] and int(((pred == y) & mask).sum()) == c['correct']
                ids = np.linspace(0, len(y)-1, cfg['replay_n'], dtype=int)
                model.cuda(); heads[task].cuda()
                for block in sorted(set(ids//batch)):
                    start, end = block*batch, min((block+1)*batch, len(y))
                    images = torch.stack([data[task][score['indices'][i]][0] for i in range(start, end)]).cuda()
                    actual = heads[task](model(images)).cpu()
                    wanted = ids[(ids >= start) & (ids < end)]
                    subset = actual[wanted-start]
                    np.testing.assert_array_equal(subset.argmax(1).numpy(), pred[wanted])
                    error = float((subset.double()-score['logits'][wanted].double()).abs().max())
                    assert error <= cfg['score_tolerance'], (key, error)
                    maximum = max(maximum, error); replays += len(wanted); context_images += len(images)
                model.cpu(); heads[task].cpu()
        del groups, own, attacked
    save_json(out/'independent_validation.json', dict(status='passed', rows=len(rows),
        official_label_numpy_recount=True, full_image_prediction_replays=replays, context_forward_images=context_images,
        original_evaluation_batch_size=batch, maximum_score_error=maximum, score_tolerance=cfg['score_tolerance'],
        explicit_TA_reconstruction='TA' in cfg['mergers'], original_TIES_functions='TIES' in cfg['mergers'], independent_TIES_small_numpy_fixture='TIES' in cfg['mergers'],
        independent_full_TIES_reimplementation=False, raw_sha256=meta['predictions_sha256'],
        DARE_NumPy_mask_arithmetic='DARE+TA' in cfg['mergers'], DARE_independent_RNG=False,
        source_sha256=digest(__file__), original_evaluation_source_sha256=meta['source_sha256'],
        reason='Preserve original batch/GEMM context after observed 16-vs-128 numerical discrepancy; no scores, weights or tolerance changed.'))
    print('PHASE3_CONTEXT_VERIFIED', out, replays, maximum, flush=True)


@torch.no_grad()
def utility_witnesses():
    """Replay budget counterexamples; exposed inputs never repair a defense."""
    out = Path('outputs/phase3/utility')
    cfg = json.loads((out/'protocol.json').read_text())
    proof = json.loads((out/'independent_validation.json').read_text())
    assert proof['status'] == 'passed_numpy_score_recount'
    assert digest(out/'scores.pt') == proof['scores_sha256']
    _, experts, heads, _, _ = setup(cfg)
    raw = torch.load(out/'scores.pt', map_location='cpu', weights_only=False)
    records = []
    for row in json.loads((out/'results.json').read_text()):
        if row['defense'] not in ['O/B', 'A/B']: continue
        task, name = row['target'], row['defense']; r = raw[task]
        before, after = r['teacher_logits'].numpy(), r[name].numpy()
        y = before.argmax(1); gap = before[np.arange(len(y)), y, None]-before
        ds = after-before; delta = ds[np.arange(len(y)), y, None]-ds
        ratio = np.divide(delta, gap, out=np.zeros_like(delta), where=gap>0)
        i, c = np.unravel_index(np.abs(ratio).argmax(), ratio.shape)
        own = experts[task]; released = apply(own, cfg['defenses'][name]['release'])
        train = dataset(task, 'data', own.val_preprocess, True)
        batch = inspect.signature(make_loader).parameters['batch_size'].default
        start, end = i//batch*batch, min((i//batch+1)*batch, len(y))
        images = torch.stack([train[r['indices'][k]][0] for k in range(start, end)]).cuda()
        own.cuda(); released.cuda(); heads[task].cuda()
        a = heads[task](own(images)).cpu().double().numpy()
        b = heads[task](released(images)).cpu().double().numpy()
        error = max(float(np.abs(a-before[start:end]).max()), float(np.abs(b-after[start:end]).max()))
        assert error <= cfg['score_tolerance']
        a, b = a[i-start], b[i-start]; teacher = int(y[i])
        actual = ((b[teacher]-b[c])-(a[teacher]-a[c]))/(a[teacher]-a[c])
        assert abs(actual-ratio[i,c]) < 1e-5 and abs(actual) > cfg['task_budget']
        records.append(dict(defense=name, training_index=int(r['indices'][i]), label=int(train[r['indices'][i]][1]),
            teacher_class=teacher, competitor_class=int(c), teacher_gap=float(a[teacher]-a[c]),
            release_gap=float(b[teacher]-b[c]), signed_relative_drift=float(actual),
            teacher_prediction=int(a.argmax()), release_prediction=int(b.argmax()),
            complete_before_scores=a.tolist(), complete_after_scores=b.tolist(), full_model_score_error=error,
            release_sha256=cfg['defenses'][name]['release']['sha256']))
        own.cpu(); released.cpu(); heads[task].cpu()
    save_json(out/'witnesses.json', dict(status='counterexamples_replayed_in_frozen_full_models',
        records=records, budget=cfg['task_budget'], raw_scores_sha256=digest(out/'scores.pt'),
        source_sha256=digest(__file__), gradients=0, new_weights=0, repair_or_selection=False,
        scope='Read-only exposed audit witnesses; accuracy results and earlier finite confirmations retained.'))
    print('PHASE3_UTILITY_WITNESSES', json.dumps(records), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output', nargs='?')
    parser.add_argument('--utility', action='store_true'); args = parser.parse_args()
    utility_witnesses() if args.utility else verify(args.output)
