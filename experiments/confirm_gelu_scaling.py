"""Fresh target-only confirmation of the already frozen scaling transform."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from angular_distance import signed_residual
from boundary_game import risk_first
from clip_benchmark import digest
from confirm_temperature_compensation import unused
from covariance_noise import dataset
from finite_gelu_scaling import cpu, pack
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import make_loader, save_json
from task_probe_distance import PREFIX, capture, checked_models, gelu, initialize
from verify_levelset_audits import convert, raw_scores, residual


@torch.no_grad()
def confirm(cfg):
    start = time.monotonic(); out = Path(cfg['output']); report = json.loads((out/'results.json').read_text())
    validation = json.loads((out/'independent_validation.json').read_text())
    assert report['task_gate_passed'] and validation['status'] == 'passed_task_calibration'
    assert digest(out/'point.pt') == report['point_sha256'] == validation['point_sha256']
    data = torch.load(out/'diagnostic.pt', map_location='cpu'); point = torch.load(out/'point.pt', map_location='cpu')
    anchor, expert, head, pins = checked_models(cfg['target']); assert pins == report['checkpoint_sha256']
    p = on_device(data['parameters']); basis = {k: on_device(v) for k, v in data['basis'].items()}; gain = point['gain'].cuda()
    training = dataset(cfg['target'], 'data', expert.val_preprocess, True); split_file = out/'confirmation_split.json'
    split = json.loads(split_file.read_text()) if split_file.exists() else unused(training, data['indices'], cfg)
    assert split['seed'] == cfg['confirmation_seed'] and not set(split['indices']) & set(data['indices'])
    save_json(split_file, split)
    cached = capture(expert, training, split['indices']); cached, ids, ncore = risk_first(cached, split['indices'], cfg['contexts_per_class'], p)
    pack0 = pack(on_device(cached), torch.arange(ncore, device='cuda'), p, basis, cfg)
    x, off, base = pack0['x'], pack0['offset'], pack0['base']; pre = x@p['A'].T+p['a']
    before = scores(base+(gelu(pre)+off)@p['B'].T, p)
    after = scores(base+(gelu(pre*gain)+off)@(p['B']/gain).T, p)
    drift = float(signed_residual(before, after).abs().max()); n = len(ids)
    # Independent NumPy actual-matrix confirmation, including both probe sites.
    pn = convert(cpu(p)); xx = x.cpu().numpy(); oo = off.cpu().numpy(); bb = base.cpu().numpy(); gg = gain.cpu().numpy()
    bs, aa = [], []
    for start0 in range(0, len(xx), cfg['verification_batch_size']):
        end = start0+cfg['verification_batch_size']; u = xx[start0:end]@pn['A'].T+pn['a']
        ug = xx[start0:end]@(pn['A']*gg[:, None]).T+pn['a']*gg
        bs.append(raw_scores(bb[start0:end]+(u*expit(1.702*u)+oo[start0:end])@pn['B'].T, pn))
        aa.append(raw_scores(bb[start0:end]+(ug*expit(1.702*ug)+oo[start0:end])@(pn['B']/gg).T, pn))
    bs, aa = np.concatenate(bs), np.concatenate(aa)
    np.testing.assert_allclose(bs, before.cpu().numpy(), atol=1e-9, rtol=1e-9)
    np.testing.assert_allclose(aa, after.cpu().numpy(), atol=1e-9, rtol=1e-9)
    numpy_drift = float(np.abs(residual(bs, aa)).max()); assert abs(numpy_drift-drift) < 1e-6
    labels = torch.tensor([int(training[i][1]) for i in ids]); pred0 = before[:n].argmax(1).cpu(); pred1 = after[:n].argmax(1).cpu()
    released = copy.deepcopy(expert); state = expert.state_dict()
    changed = {'A': p['A']*(gain-1)[:, None], 'a': p['a']*(gain-1), 'B': p['B']*(1/gain-1)}
    keys = {'A': PREFIX+'c_fc.weight', 'a': PREFIX+'c_fc.bias', 'B': PREFIX+'c_proj.weight'}
    for site, key in keys.items():
        assert torch.equal(state[key].double(), p[site].cpu())
        released.state_dict()[key].copy_((state[key].double()+changed[site].cpu()).float())
    formal_distance = float(sum((changed[k].norm()/p[k].norm())**2 for k in keys)**.5)
    checks = {}
    for name, model, expected in [('original', expert, pred0), ('released', released, pred1)]:
        values = []; model.cuda().eval(); head.cuda().eval()
        for images, _ in make_loader(training, ids, 64): values.append(head(model(images.cuda())).argmax(1).cpu())
        model.cpu(); head.cpu(); actual = torch.cat(values); assert torch.equal(actual, expected), name
        checks[name] = dict(n=n, correct=int((actual == labels).sum()), all_predictions_match=True)
    passed = formal_distance <= cfg['parameter_budget'] and drift <= cfg['task_budget'] and torch.equal(pred0, pred1)
    torch.save(dict(cache=cpu(cached), indices=ids, teacher=before.cpu(), released_scores=after.cpu(), labels=labels), out/'confirmation.pt')
    result = dict(status='passed' if passed else 'failed_task_confirmation', confirmation_passed=passed, target=cfg['target'], natural_n=n,
                  all_task_inputs=len(before), parameter_distance=formal_distance, worst_margin_drift=drift, numpy_worst_margin_drift=numpy_drift,
                  natural_changes=int((pred1 != pred0).sum()), original_correct=int((pred0 == labels).sum()), released_correct=int((pred1 == labels).sum()),
                  full_model_replay=checks, new_model_forwards=3*n, confirmation_sha256=digest(out/'confirmation.pt'),
                  point_sha256=report['point_sha256'], source_sha256=digest(__file__), wall_seconds=time.monotonic()-start,
                  partner_used=False, test_used=False, selected_after_confirmation=False, original_expert_updates=0)
    if passed:
        torch.save(dict(release_deltas=cpu(changed), gain=point['gain'], confirmation_indices=ids, point_sha256=report['point_sha256'], cfg=cfg), out/'candidate.pt')
        result['candidate_sha256'] = digest(out/'candidate.pt')
        report.update(confirmation_passed=True, candidate_sha256=result['candidate_sha256'], selected_step=0, optimizer_updates=0,
                      explicit_reverse_gradient_calls=0, analytic_task_jacobian_calls=1, nonlinear_correction_updates=0,
                      confirmation=result, new_release_weights=1, status='passed_independent_fresh_confirmation')
        validation = dict(status='passed', source_calibration=validation, confirmation=result,
                          new_model_forwards=3*n, candidate_sha256=result['candidate_sha256'])
        save_json(out/'results.json', report); save_json(out/'independent_validation.json', validation)
    save_json(out/'confirmation_validation.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True); args = ap.parse_args()
    cfg = json.loads(Path(args.protocol).read_text())
    for path, pin in cfg['source_sha256'].items(): assert digest(path) == pin, path
    initialize(); confirm(cfg)
