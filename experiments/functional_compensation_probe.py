"""Source-only finite B-input probes from the classifier-weighted complement.

The directions are internal interventions, not assumed future merge shifts.
All frozen points stay unchanged. NumPy independently reconstructs each result.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from attention_cascade import norm, suffix
from clip_benchmark import digest
from finite_compensation_transport import geometry, sigma
from probe_distance_candidate import on_device
from subspace_probe import save_json
from task_probe_distance import gelu, initialize
from verify_finite_comparison import N as numpy_N, states as numpy_states
from verify_levelset_audits import convert


def cpu(p): return {k: v.cpu() if torch.is_tensor(v) else v for k, v in p.items()}


def num(h, p):
    m, b = geometry(p); return h@m+sigma(h, p)*b


def counts(before, after, teacher, classes):
    stable = (before.argmax(-1) == teacher[:, None, None]).all(-1)
    change = after.argmax(-1) != teacher[:, None, None]; both = stable & change.all(-1); either = stable & change.any(-1)
    return dict(teacher_stable_pairs=int(stable.sum()), total_pairs=stable.numel(), protected_both_signs_changed=int(both.sum()),
                protected_either_sign_changed=int(either.sum()), images_with_both_sign_failure=int(both.any(1).sum()),
                both_sign_failures_by_class=[int(both[teacher == c].sum()) for c in range(classes)],
                stable_pairs_by_class=[int(stable[teacher == c].sum()) for c in range(classes)])


@torch.no_grad()
def audit(cfg):
    start = time.monotonic(); out = Path(cfg['output']); out.mkdir(parents=True, exist_ok=False)
    for path, pin in cfg['inputs'].items(): assert digest(path) == pin, path
    cal = torch.load(cfg['source_prefix'], map_location='cpu'); original = cal['parameters']; original.pop('D', None); p = on_device(original)
    point = torch.load(cfg['index_point'], map_location='cpu'); ids = point['indices']['fit'][:cfg['n']]
    lookup = {v: i for i, v in enumerate(cal['indices']['fit'])}; take0 = [lookup[i] for i in ids]
    H = cal['native']['H'][take0].cuda().double(); h0 = cal['native']['h'][take0].cuda().double()
    basis = torch.load(cfg['basis_reference'], map_location='cpu')['basis']['B']['vh'].cuda()
    raw = torch.load(cfg['boundary_reference'], map_location='cpu'); r = raw['screen']['r'].cuda()
    source_x = norm(r, p['gamma2'], p['beta2'], p['eps2']); source_z = gelu(source_x@p['A'].T+p['a'])
    n0 = num(h0, p); teacher = n0.argmax(1); classes = n0.shape[1]; chosen = []; id_tensor = torch.tensor(ids, device='cuda')
    for c in range(classes):
        members = (teacher == c).nonzero().flatten(); assert len(members) >= cfg['images_per_class']
        chosen.extend(members[id_tensor[members].argsort(stable=True)[:cfg['images_per_class']]].tolist())
    take = torch.tensor(chosen, device='cuda'); u = n0/n0.norm(dim=1, keepdim=True)
    weight = 1./torch.bincount(teacher, minlength=classes)[teacher].double()/classes
    tangent = torch.eye(classes, device='cuda', dtype=u.dtype)-torch.einsum('n,ni,nj->ij', weight, u, u)
    eig, v = torch.linalg.eigh(tangent); tangent_sqrt = (v*eig.clamp_min(0).sqrt())@v.T
    original_out = suffix(H, p)[0]; records = {}; saved = {}
    mapping = {'attn.in_proj_weight': 'W', 'attn.in_proj_bias': 'w', 'attn.out_proj.weight': 'O', 'attn.out_proj.bias': 'ob',
               'mlp.c_fc.weight': 'A', 'mlp.c_fc.bias': 'a', 'mlp.c_proj.weight': 'B'}
    for case in cfg['cases']:
        q = dict(p); release = torch.load(case['release'], map_location='cpu')
        for key, value in release.items(): q[mapping[key.removeprefix('model.visual.transformer.resblocks.11.')]] = value.cuda().double()
        h1 = h0+suffix(H, q)[0]-original_out; db = q['B']-p['B']; m, _ = geometry(p)
        K = db.T@m; perpendicular = K-basis.T@(basis@K); operator = perpendicular@tangent_sqrt
        left, singular, right = torch.linalg.svd(operator, full_matrices=False)
        tol = torch.finfo(singular.dtype).eps*max(operator.shape)*singular[0]; rank = int((singular > tol).sum())
        assert 0 < rank <= classes-1; directions = left[:, :rank].T
        rms = (source_z@directions.T).square().mean(0).sqrt()
        shifts = torch.stack([-cfg['amplitude']*rms[:, None]*directions, cfg['amplitude']*rms[:, None]*directions], 1).flatten(0, 1)
        a = h0[take, None]+(shifts@p['B'].T)[None]; b = h1[take, None]+(shifts@q['B'].T)[None]
        before = num(a.flatten(0, 1), p).reshape(len(take), rank, 2, classes)
        after = num(b.flatten(0, 1), p).reshape_as(before); row = counts(before, after, teacher[take], classes)
        # Independent native-prefix, classifier geometry, singular equations,
        # RMS and every finite LayerNorm decision: no generator function reused.
        pn, qn = convert(cpu(p)), convert(cpu(q)); hn, HH = h0.cpu().numpy(), H.cpu().numpy()
        st = numpy_states(HH, hn, pn, qn); np.testing.assert_allclose(st['joint'], h1.cpu().numpy(), atol=1e-8, rtol=1e-8)
        nn = numpy_N(hn, pn); un = nn/np.linalg.norm(nn, axis=1, keepdims=True); yy = nn.argmax(1)
        wn = 1./np.bincount(yy, minlength=classes)[yy]/classes
        gn = np.eye(classes)-np.einsum('n,ni,nj->ij', wn, un, un)
        np.testing.assert_allclose(gn, tangent.cpu().numpy(), atol=1e-10, rtol=1e-10)
        en, vn = np.linalg.eigh(gn); sqrt_n = (vn*np.sqrt(np.maximum(en, 0)))@vn.T
        head = pn['head']-pn['head'].mean(0); mn = pn['gamma'][:, None]*(pn['projection']@head.T); mn -= mn.mean(0)
        kn = (qn['B']-pn['B']).T@mn; vb = basis.cpu().numpy(); opn = (kn-vb.T@(vb@kn))@sqrt_n; dirs = directions.cpu().numpy()
        error = float(np.max(np.abs(opn@(opn.T@dirs.T)-dirs.T*singular[:rank].cpu().numpy()**2)))
        assert error < 1e-7 and np.max(np.abs(vb@dirs.T)) < 1e-8
        sx = source_x.cpu().numpy(); pre = sx@pn['A'].T+pn['a']; zn = pre*expit(1.702*pre)
        rms_n = np.sqrt(np.mean((zn@dirs.T)**2, axis=0)); np.testing.assert_allclose(rms_n, rms.cpu().numpy(), atol=1e-9, rtol=1e-9)
        sn = np.stack([-cfg['amplitude']*rms_n[:, None]*dirs, cfg['amplitude']*rms_n[:, None]*dirs], 1).reshape(shifts.shape)
        ii = take.cpu().numpy(); bn = numpy_N((hn[ii, None]+(sn@pn['B'].T)[None]).reshape(-1, hn.shape[1]), pn).reshape(before.shape)
        an = numpy_N((st['joint'][ii, None]+(sn@qn['B'].T)[None]).reshape(-1, hn.shape[1]), pn).reshape(after.shape)
        np.testing.assert_allclose(bn, before.cpu().numpy(), atol=1e-8, rtol=1e-8); np.testing.assert_allclose(an, after.cpu().numpy(), atol=1e-8, rtol=1e-8)
        stable = (bn.argmax(-1) == yy[ii, None, None]).all(-1); changed = an.argmax(-1) != yy[ii, None, None]; both = stable & changed.all(-1)
        assert int(stable.sum()) == row['teacher_stable_pairs'] and int(both.sum()) == row['protected_both_signs_changed']
        row.update(rank=rank, singular_values=singular[:rank].cpu().tolist(), independent_numpy_passed=True, singular_equation_maximum_error=error,
                   native_prediction_changes=int((num(h1, p).argmax(1) != teacher).sum()),
                   original_task_mode_overlap=float((basis@directions.T).norm()), source_only=True, no_candidate_or_evaluation_gate=True)
        records[case['name']] = row; saved[case['name']] = dict(directions=directions.cpu(), rms=rms.cpu(), before=before.cpu(), after=after.cpu(), source_parameters=cpu(q))
    torch.save(dict(indices=ids, take=take.cpu(), parameters=cpu(p), H=H.cpu(), h=h0.cpu(), cases=saved, cfg=cfg), out/'diagnostic.pt')
    report = dict(status='source_only_functional_complement_probes_verified', records=records, n=len(ids), images_per_class=cfg['images_per_class'],
                  source_sha256=digest(__file__), protocol_sha256=digest(cfg['protocol_path']), diagnostic_sha256=digest(out/'diagnostic.pt'),
                  wall_seconds=time.monotonic()-start, new_full_model_forward_instances=0, new_candidates=0, new_release_weights=0, publication_updates=0,
                  partner_used=False, test_used=False, information_boundary='Reads only frozen target releases, target/source-training caches and source-task SVD basis. No peer weights, real fusion artifacts or test inputs.',
                  limitation='Classifier-weighted complement directions are defined by the frozen release, not an unknown-partner distribution. One RMS B-input interventions may be off the QuickGELU image. Not a new defense or transfer guarantee.')
    save_json(out/'results.json', report); save_json(out/'protocol.json', cfg); print(json.dumps(report), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol', required=True); args = ap.parse_args()
    cfg = json.loads(Path(args.protocol).read_text()); cfg['protocol_path'] = args.protocol
    for path, pin in cfg['source_sha256'].items(): assert digest(path) == pin, path
    initialize(); audit(cfg)
