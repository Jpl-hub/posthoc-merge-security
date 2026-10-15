"""One fresh target-training confirmation of an already frozen joint point."""

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
from verify_downstream_first import convert
from verify_levelset_audits import raw_scores, residual


@torch.no_grad()
def confirm(cfg):
    start = time.monotonic(); out = Path(cfg['output']); report = json.loads((out/'results.json').read_text())
    validation = json.loads((out/'independent_validation.json').read_text())
    assert report['confirmation_eligible'] and validation['status'] == 'passed_task_calibration'
    assert digest(out/'training_point.pt') == report['training_point_sha256'] == validation['point_sha256']
    point = torch.load(out/'training_point.pt', map_location='cpu'); data = torch.load(out/'diagnostic.pt', map_location='cpu')
    anchor, expert, head, pins = checked_models(cfg['target']); assert pins == report['checkpoint_sha256']
    p = on_device(data['parameters']); basis = {k: on_device(v) for k, v in data['basis'].items()}
    keys = dict(A=PREFIX+'c_fc.weight', B=PREFIX+'c_proj.weight'); changed, edits = {}, {}
    released = copy.deepcopy(expert)
    for name, key in keys.items():
        original = expert.state_dict()[key]; assert torch.equal(original.double(), data['parameters'][name])
        edited = (original.double()+point['release_deltas'][name]).float()
        edits[key] = edited; changed[name] = (edited.double()-original.double()).cuda()
        released.state_dict()[key].copy_(edited)
    distance = float(sum((changed[key].norm()/p[key].norm())**2 for key in keys)**.5)
    def paired(packed):
        x, off, base = packed['x'], packed['offset'], packed['base']
        pre = x@p['A'].T+p['a']; before = scores(base+(gelu(pre)+off)@p['B'].T, p)
        after = scores(base+(gelu(x@(p['A']+changed['A']).T+p['a'])+off)@(p['B']+changed['B']).T, p)
        return before, after
    # FP32 export is checked against all old source constraints BEFORE drawing
    # any confirmation data. A quantization failure is not repaired.
    source_pack = pack(on_device(data['cache']), data['core'].cuda(), p, basis, cfg)
    bs, aa = paired(source_pack); source_drift = float(signed_residual(bs, aa).abs().max())
    assert source_drift <= cfg['task_budget'] and distance <= cfg['parameter_budget'], 'FP32 export failed original source budget'
    training = dataset(cfg['target'], 'data', expert.val_preprocess, True); split_file = out/'confirmation_split.json'
    split = json.loads(split_file.read_text()) if split_file.exists() else unused(training, data['indices'], cfg)
    assert split['seed'] == cfg['confirmation_seed'] and not set(split['indices']) & set(data['indices'])
    save_json(split_file, split)
    cached = capture(expert, training, split['indices']); cached, ids, ncore = risk_first(cached, split['indices'], cfg['contexts_per_class'], p)
    packed = pack(on_device(cached), torch.arange(ncore, device='cuda'), p, basis, cfg)
    before, after = paired(packed); drift = float(signed_residual(before, after).abs().max()); n = len(ids)
    pn = convert(cpu(p)); xx, oo, bb = [packed[key].cpu().numpy() for key in ['x', 'offset', 'base']]
    da, db = changed['A'].cpu().numpy(), changed['B'].cpu().numpy(); nbs, nas = [], []
    for lo in range(0, len(xx), cfg['verification_batch_size']):
        hi = lo+cfg['verification_batch_size']; u = xx[lo:hi]@pn['A'].T+pn['a']; un = xx[lo:hi]@(pn['A']+da).T+pn['a']
        nbs.append(raw_scores(bb[lo:hi]+(u*expit(1.702*u)+oo[lo:hi])@pn['B'].T, pn))
        nas.append(raw_scores(bb[lo:hi]+(un*expit(1.702*un)+oo[lo:hi])@(pn['B']+db).T, pn))
    nbs, nas = np.concatenate(nbs), np.concatenate(nas)
    np.testing.assert_allclose(nbs, before.cpu().numpy(), atol=1e-9, rtol=1e-9); np.testing.assert_allclose(nas, after.cpu().numpy(), atol=1e-9, rtol=1e-9)
    independent_drift = float(np.abs(residual(nbs, nas)).max()); assert abs(independent_drift-drift) < 1e-6
    labels = torch.tensor([int(training[i][1]) for i in ids]); pred0, pred1 = before[:n].argmax(1).cpu(), after[:n].argmax(1).cpu()
    replay = {}
    for name, model, expected in [('original', expert, pred0), ('released', released, pred1)]:
        predictions = []; model.cuda().eval(); head.cuda().eval()
        for images, _ in make_loader(training, ids, 64): predictions.append(head(model(images.cuda())).argmax(1).cpu())
        model.cpu(); head.cpu(); actual = torch.cat(predictions); assert torch.equal(actual, expected), name
        replay[name] = dict(n=n, correct=int((actual == labels).sum()), all_predictions_match=True)
    passed = bool(drift <= cfg['task_budget'] and distance <= cfg['parameter_budget'] and torch.equal(pred0, pred1))
    torch.save(dict(cache=cpu(cached), indices=ids, teacher=before.cpu(), released_scores=after.cpu(), labels=labels,
                    source_FP32_worst_drift=source_drift, release_deltas=cpu(changed)), out/'confirmation.pt')
    result = dict(status='passed' if passed else 'failed_task_confirmation', confirmation_passed=passed, target=cfg['target'],
                  natural_n=n, all_task_inputs=len(before), parameter_distance=distance, worst_margin_drift=drift,
                  numpy_worst_margin_drift=independent_drift, source_FP32_worst_drift=source_drift,
                  natural_changes=int((pred0 != pred1).sum()), original_correct=int((pred0 == labels).sum()), released_correct=int((pred1 == labels).sum()),
                  full_model_replay=replay, new_model_forwards=3*n, confirmation_sha256=digest(out/'confirmation.pt'),
                  point_sha256=report['training_point_sha256'], source_sha256=digest(__file__), wall_seconds=time.monotonic()-start,
                  partner_used=False, test_used=False, selected_after_confirmation=False, original_expert_updates=0)
    if passed:
        torch.save(dict(release_deltas=cpu(changed), C=point['C'], K=point['K'], confirmation_indices=ids,
                        training_point_sha256=report['training_point_sha256'], cfg=cfg), out/'candidate.pt')
        result['candidate_sha256'] = digest(out/'candidate.pt')
        report.update(status='passed_independent_fresh_confirmation', confirmation_passed=True, independent_confirmation_run=True,
                      candidate_sha256=result['candidate_sha256'], confirmation=result, new_release_weights=1)
        save_json(out/'results.json', report)
        save_json(out/'independent_validation.json', dict(status='passed', source_calibration=validation, confirmation=result,
                                                        new_model_forwards=3*n, candidate_sha256=result['candidate_sha256']))
    save_json(out/'confirmation_validation.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True); args = ap.parse_args()
    initialize(); confirm(json.loads(Path(args.protocol).read_text()))
