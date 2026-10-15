"""Replay one independently identified task-drift witness; no model selection or repair."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import digest
from covariance_noise import dataset
from subspace_probe import save_json
from task_probe_distance import checked_models, initialize
from verify_levelset_audits import convert, raw_scores
from verify_finite_comparison import suffix


@torch.no_grad()
def run(output):
    start = time.monotonic(); root = Path(output)
    cfg = json.loads((root/'protocol.json').read_text())
    data = torch.load(root/'artifacts/diagnostic.pt', map_location='cpu')
    base = data['environments']['source']; group = data['cases']['O_B']['source']
    p, q = convert(base['parameters']), convert(group['parameters'])
    h = base['cache']['h'].numpy(); H = base['cache']['H'].numpy()
    h0, _ = suffix(H, p); h1, _ = suffix(H, q)
    s0, s1 = raw_scores(h, p), raw_scores(h+h1-h0, p)
    y = s0.argmax(1); gaps = s0[np.arange(len(y)), y, None]-s0
    ds = s1-s0
    rho = np.divide(ds[np.arange(len(y)), y, None]-ds, gaps,
                    out=np.zeros_like(gaps), where=gaps > 0)
    i, competitor = np.unravel_index(np.abs(rho).argmax(), rho.shape)
    index = int(data['indices'][i]); predicted = int(y[i])
    initialize(); assert '3090' in torch.cuda.get_device_name()
    _, teacher, head, hashes = checked_models('MNIST')
    protected = copy.deepcopy(teacher)
    release = cfg['cases'][0]['release']
    assert digest(release) == cfg['inputs'][release]
    protected.load_state_dict(teacher.state_dict() | torch.load(release, map_location='cpu'), strict=True)
    train = dataset('MNIST', 'data', teacher.val_preprocess, True)
    image, label = train[index]; image = image[None].cuda(); head.cuda().eval()
    teacher.cuda().eval(); protected.cuda().eval()
    before = head(teacher(image)).cpu().double().numpy()[0]
    after = head(protected(image)).cpu().double().numpy()[0]
    actual = ((after[predicted]-before[predicted])-(after[competitor]-before[competitor]))/(before[predicted]-before[competitor])
    score_error = float(max(abs(before-s0[i]).max(), abs(after-s1[i]).max()))
    assert score_error < cfg['forward_atol']
    assert abs(actual) > cfg['task_budget']
    result = dict(status='frozen_OB_source_task_budget_counterexample_replayed',
        training_index=index, diagnostic_position=int(i), teacher_class=predicted,
        competitor_class=int(competitor), label=int(label),
        NumPy_signed_relative_drift=float(rho[i, competitor]),
        full_model_signed_relative_drift=float(actual), budget=cfg['task_budget'],
        full_model_prediction_before=int(before.argmax()), full_model_prediction_after=int(after.argmax()),
        full_model_vs_cached_score_max_abs=score_error,
        source_scores_before=before.tolist(), source_scores_after=after.tolist(),
        release_sha256=digest(release), original_checkpoint_sha256=hashes,
        diagnostic_sha256=digest(root/'artifacts/diagnostic.pt'), source_sha256=digest(__file__),
        full_model_forwards=2, gradients=0, new_weights=0, confirmation_used=False,
        test_used=False, partner_loaded=False, wall_seconds=time.monotonic()-start,
        limitation='Exposed construction data, retrospective counterexample only. Original finite confirmation and recorded TA/AGA results remain intact. No repair or new candidate.')
    save_json(root/'OB_task_budget_witness.json', result)
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', required=True)
    run(ap.parse_args().output)
