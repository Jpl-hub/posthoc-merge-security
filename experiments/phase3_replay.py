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
        explicit_TA_reconstruction=True, original_TIES_functions=True, independent_TIES_small_numpy_fixture=True,
        independent_full_TIES_reimplementation=False, raw_sha256=meta['predictions_sha256'],
        source_sha256=digest(__file__), original_evaluation_source_sha256=meta['source_sha256'],
        reason='Preserve original batch/GEMM context after observed 16-vs-128 numerical discrepancy; no scores, weights or tolerance changed.'))
    print('PHASE3_CONTEXT_VERIFIED', out, replays, maximum, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('output')
    verify(parser.parse_args().output)
