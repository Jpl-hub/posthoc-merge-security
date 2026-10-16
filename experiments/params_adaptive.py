"""PaRaMS paper-objective diagonal recovery; official MLP kernel, not author attack code."""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import time
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])

import jax
import numpy as np
import torch

SOURCE = Path('external/PaRaMS/src/PaRaMS/PaRaMS_algorithm.py')
SOURCE_SHA = '40c114a121907d5633d29a49984efa9be2ac03e29e649e801fed2dccad89fb60'
LABEL = 'PaRaMS diagonal adaptation (paper objective)'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def official():
    assert sha(SOURCE) == SOURCE_SHA
    spec = importlib.util.spec_from_file_location('official_params_adaptive', SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@torch.no_grad()
def recover(state, anchor, author, layers=12):
    """No input images or peer; fit Q/V row scales and apply reciprocal pairs."""
    result = {k: v.detach().cpu().clone() for k, v in state.items()}
    spec = author.vit_permutation_spec_MLP(layers)
    keys = list(spec.axes_to_perm)
    permutation, _, _, _ = author.parameter_rearrangement(
        jax.random.PRNGKey(0), spec, {k: anchor[k] for k in keys},
        {k: result[k] for k in keys}, max_iter=200, obj='matching', tolerance=1)
    result.update({k: torch.as_tensor(np.array(v), dtype=state[k].dtype)
                   for k, v in author.apply_permutation(spec, permutation, {k: result[k] for k in keys}).items()})
    records = []
    for i in range(layers):
        p = f'model.visual.transformer.resblocks.{i}.attn.'
        weight, bias, output = [result[p+k].double() for k in
                                ['in_proj_weight', 'in_proj_bias', 'out_proj.weight']]
        ref = anchor[p+'in_proj_weight'].double()
        d = weight.shape[1]
        assert weight.shape == (3*d, d) and output.shape == (d, d)
        factors = []
        for part in [0, 2]:
            a, b = part*d, (part+1)*d
            denominator = weight[a:b].square().sum(1)
            assert (denominator > 0).all(), 'Zero Q/V row: diagonal recovery is not identifiable.'
            scale = (weight[a:b]*ref[a:b]).sum(1)/denominator
            assert torch.isfinite(scale).all() and (scale != 0).all(), 'Singular row scale; stop without clipping.'
            weight[a:b] *= scale[:, None]; bias[a:b] *= scale
            if part == 0:
                weight[d:2*d] /= scale[:, None]; bias[d:2*d] /= scale
            else:
                output /= scale[None, :]
            factors.append(scale)
        for key, value in [('in_proj_weight', weight), ('in_proj_bias', bias), ('out_proj.weight', output)]:
            result[p+key] = value.to(state[p+key].dtype)
            assert torch.isfinite(result[p+key]).all()
        records.append(dict(layer=i, Q_inverse_scale=factors[0].tolist(), V_inverse_scale=factors[1].tolist()))
    return result, dict(layer_scales=records, permutations={k: v.tolist() for k, v in permutation.items()},
                        source_sha256=SOURCE_SHA, images=0, gradients=0,
                        scope='Official MLP kernel plus paper-objective diagonal Attention adaptation; not a full official attack implementation.')


@torch.no_grad()
def check(output):
    assert not output.exists()
    torch.set_num_threads(1); torch.manual_seed(0); start = time.monotonic(); author = official()
    p = 'model.visual.transformer.resblocks.0.'
    attention = torch.nn.MultiheadAttention(12, 3, dropout=0.)
    base = {p+'attn.'+k: v.clone() for k, v in attention.state_dict().items()}
    for name, layer in [('c_fc', torch.nn.Linear(12, 6)), ('c_proj', torch.nn.Linear(6, 12))]:
        base.update({p+'mlp.'+name+'.'+k: v.clone() for k, v in layer.state_dict().items()})
    protected = author.params(base, base, num_layers=1, rng_seed=0)
    restored, evidence = recover(protected, base, author, 1)
    x = torch.randn(7, 2, 12)

    def forward(state):
        attention.load_state_dict({k.removeprefix(p+'attn.'): v for k, v in state.items() if '.attn.' in k})
        h = x+attention(x, x, x, need_weights=False)[0]
        z = h@state[p+'mlp.c_fc.weight'].T+state[p+'mlp.c_fc.bias']
        return h+(z*torch.sigmoid(1.702*z))@state[p+'mlp.c_proj.weight'].T+state[p+'mlp.c_proj.bias']

    response = {n: forward(s) for n, s in [('base', base), ('protected', protected), ('recovered', restored)]}
    errors = {n: float((response[n]-response['base']).abs().max()) for n in ['protected', 'recovered']}
    assert max(errors.values()) < 5e-5
    matrix_error = max(float((restored[k]-base[k]).abs().max()) for k in base)
    assert matrix_error < 1e-6
    numpy_error = 0.
    for k, value in restored.items():
        if '.mlp.' in k:
            perm = evidence['permutations']['P_0']; a = protected[k].numpy()
            expected = a[:, perm] if k.endswith('c_proj.weight') else a[perm] if not k.endswith('c_proj.bias') else a
        else:
            a = protected[k].numpy().astype('float64').copy(); d = 12
            q, v = [np.array(evidence['layer_scales'][0][n]) for n in ['Q_inverse_scale', 'V_inverse_scale']]
            if k.endswith('in_proj_weight'): a *= np.concatenate([q, 1/q, v])[:, None]
            if k.endswith('in_proj_bias'): a *= np.concatenate([q, 1/q, v])
            if k.endswith('out_proj.weight'): a /= v[None, :]
            expected = a.astype('float32')
        numpy_error = max(numpy_error, float(np.abs(value.numpy()-expected).max()))
    assert numpy_error < 1e-6
    result = dict(status='synthetic_full_Attention_and_MLP_chain_verified_not_CLIP_evaluation',
                  method=LABEL, official_commit='4159a9a8bf253088694154d944f4583db8eca60e',
                  official_source_sha256=SOURCE_SHA, implementation_sha256=sha(__file__),
                  fixture='One seeded 12-dimensional 3-head Attention plus QuickGELU MLP; author default protection.',
                  full_response_max_abs=errors, reconstructed_parameter_max_abs=matrix_error,
                  independent_NumPy_parameter_reconstruction_max_abs=numpy_error,
                  K_recovery='reciprocal of fitted Q scale', O_recovery='reciprocal of fitted V scale',
                  paper_scope='Section6 objective with explicit paired-inverse diagonal completion; not the paper multi-task table.',
                  images=0, gradients=0, GPU=False, real_CLIP_evaluations=0,
                  wall_seconds=time.monotonic()-start,
                  runtime=dict(torch=torch.__version__, jax=jax.__version__, numpy=np.__version__))
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


def inputs(protocol):
    from clip_benchmark import checkpoint_loader
    cfg = json.loads(Path(protocol).read_text())
    assert '4090' in torch.cuda.get_device_name(), 'This new protocol is registered for RTX4090 only.'
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    for path, pin in cfg['source_sha256'].items(): assert sha(path) == pin, path
    assert sha(cfg['frozen_protocol']) == cfg['frozen_protocol_sha256']
    frozen = json.loads(Path(cfg['frozen_protocol']).read_text())
    pins = {r['path']: r['sha256'] for r in json.loads(Path('configs/checkpoints.json').read_text())['files']}
    load = checkpoint_loader(Path('external/secure-merge-attack'))
    def checked(name):
        path = Path('checkpoints/ViT-B-32')/name; assert sha(path) == pins['ViT-B-32/'+name]
        model = load(str(path)).cpu().eval(); model.requires_grad_(False); return model
    return cfg, frozen, checked


@torch.no_grad()
def run(protocol, full=False, verify=False):
    from clip_benchmark import digest, evaluate, merge
    from covariance_noise import dataset
    from phase3_evaluate import apply, row_metrics
    from subspace_probe import make_loader, save_json
    from verify_covariance_noise import official_labels
    from verify_probe_candidate import explicit_merge
    cfg, frozen, checked = inputs(protocol); root = Path(cfg['output']); start = time.monotonic()
    anchor, own, peer = checked('zeroshot.pt'), checked('MNIST/finetuned.pt'), checked('SVHN/finetuned.pt')
    methods = ['Unprotected']+cfg['defenses']
    if not full and not verify:
        root.mkdir(parents=True, exist_ok=False); (root/'artifacts').mkdir()
        save_json(root/'protocol.json', cfg); author = official(); records = {}
        for name in methods+['SVHN_control']:
            model = peer if name == 'SVHN_control' else own if name == 'Unprotected' else apply(
                own, frozen['defenses'][name]['release'], frozen['defenses'][name].get('full_state', False))
            state, evidence = recover(model.state_dict(), anchor.state_dict(), author)
            edits = {k: v for k, v in state.items() if not torch.equal(v, model.state_dict()[k])}
            path = root/'artifacts'/(name.replace('/', '_')+'.pt'); torch.save(edits, path)
            records[name] = dict(path=str(path), sha256=digest(path), **evidence)
        save_json(root/'attack_freeze.json', dict(records=records, constructed_before_images=True,
            method=LABEL, protocol_sha256=sha(root/'protocol.json'), defense_updates=0))
    else:
        assert cfg == json.loads((root/'protocol.json').read_text())
        if full: assert json.loads((root/'pilot/independent_validation.json').read_text())['status'] == 'passed'
    freeze = json.loads((root/'attack_freeze.json').read_text())
    assert freeze['protocol_sha256'] == sha(root/'protocol.json') and freeze['constructed_before_images']
    attacked_peer = apply(peer, freeze['records']['SVHN_control'])
    heads = {t: checked('head_'+t+'.pt') for t in ['MNIST', 'SVHN']}
    data = {t: dataset(t, 'data', own.val_preprocess) for t in heads}
    assert all(len(data[t]) == frozen['tasks'][t]['test_n'] for t in data)
    out = root/('full' if full else 'pilot')
    if not verify: out.mkdir(exist_ok=False)
    raw = torch.load(out/'predictions.pt', map_location='cpu', weights_only=False) if verify else {}
    rows = json.loads((out/'results.json').read_text()) if verify else []
    if verify:
        meta = json.loads((out/'execution.json').read_text()); assert meta['completed']
        assert sha(out/'predictions.pt') == meta['predictions_sha256']
    baseline, controls, replays, error = {}, [], 0, 0.
    if not full and not verify:
        ids = np.linspace(0, len(data['SVHN'])-1, cfg['pilot_n'], dtype=int).tolist()
        before, after = [], []
        evaluate(peer, heads['SVHN'], make_loader(data['SVHN'], ids), 'cuda', logit_sink=before.append)
        evaluate(attacked_peer, heads['SVHN'], make_loader(data['SVHN'], ids), 'cuda', logit_sink=after.append)
        difference = float((torch.cat(before)-torch.cat(after)).abs().max())
        controls.append(dict(defense='SVHN_control', score_error=difference))
        save_json(out/'function_controls.json', controls); assert difference <= cfg['score_tolerance']
    for name in methods:
        spec = frozen['defenses'].get(name)
        release = own if spec is None else apply(own, spec['release'], spec.get('full_state', False))
        attacked = apply(release, freeze['records'][name])
        fusion = explicit_merge if verify else merge
        groups = [('Standalone', 'Protected expert', attacked, ['MNIST']),
                  ('TA', 'Protected expert', fusion(anchor, attacked, peer, .5), list(heads)),
                  ('TA', 'All experts', fusion(anchor, attacked, attacked_peer, .5), list(heads))]
        for method, scope, model, tasks in groups:
            for task in tasks:
                ids = list(range(len(data[task]))) if full else np.linspace(0, len(data[task])-1, cfg['pilot_n'], dtype=int).tolist()
                key = '/'.join((name, method, scope, task)); control_key = '/'.join(('Unprotected', method, scope, task))
                if verify:
                    record = raw[key]; values = record['logits'].numpy(); y = official_labels(task)[ids]; prediction = values.argmax(1)
                    np.testing.assert_array_equal(y, record['target'].numpy())
                    before = raw[control_key]['logits'].numpy().argmax(1)
                    row = next(r for r in rows if r['key'] == key)
                    assert row['changed_predictions'] == int((prediction != before).sum())
                    assert row['correct'] == int((prediction == y).sum())
                    assert row['paired']['lost'] == int(((before == y) & (prediction != y)).sum())
                    assert row['paired']['gained'] == int(((before != y) & (prediction == y)).sum())
                    for c in row['per_class']:
                        mask = y == c['class_id']; assert c['n'] == int(mask.sum()) and c['correct'] == int(((prediction == y) & mask).sum())
                    for pos in np.linspace(0, len(ids)-1, cfg['replay_n'], dtype=int):
                        lo = int(pos)//128*128; hi = min(lo+128, len(ids))
                        images = torch.stack([data[task][i][0] for i in ids[lo:hi]]).cuda()
                        model.cuda(); heads[task].cuda(); actual = heads[task](model(images)).cpu()[int(pos)-lo]; model.cpu(); heads[task].cpu()
                        e = float((actual.double()-record['logits'][int(pos)].double()).abs().max()); error = max(error, e)
                        assert e <= cfg['score_tolerance']; assert int(actual.argmax()) == int(prediction[int(pos)]); replays += 1
                else:
                    logits, labels = [], []
                    metric = evaluate(model, heads[task], make_loader(data[task], ids), 'cuda',
                        prediction_sink=lambda p, y: labels.append(y), logit_sink=logits.append)
                    values, y = torch.cat(logits), torch.cat(labels)
                    np.testing.assert_array_equal(y.numpy(), official_labels(task)[ids]); assert torch.isfinite(values).all()
                    if name == 'Unprotected': baseline[(method, scope, task)] = values
                    reference = baseline[(method, scope, task)].double(); winner = reference.argmax(1, keepdim=True)
                    gaps = reference.gather(1, winner)-reference; ds = values.double()-reference
                    ratio = torch.zeros_like(gaps); positive = gaps > 0
                    ratio[positive] = (ds.gather(1, winner)-ds)[positive]/gaps[positive]
                    rows.append(dict(key=key, defense=name, recovery=LABEL, merger=method, attack_scope=scope,
                        dataset=task, role='target' if task == 'MNIST' else 'partner',
                        **row_metrics(values, y, baseline[(method, scope, task)], frozen['tasks'][task]['classes']),
                        worst_relative_margin_drift_to_matched_control=float(ratio.abs().max()),
                        maximum_signed_margin_harm_to_matched_control=float((-ratio).max()),
                        evaluation_seconds=metric['evaluation_seconds']))
                    raw[key] = dict(logits=values, target=y, indices=ids)
                    save_json(out/'results.json', rows); torch.save(raw, out/'predictions.pt')
                    if not full and method == 'Standalone':
                        reference = []; evaluate(release, heads[task], make_loader(data[task], ids), 'cuda', logit_sink=reference.append)
                        difference = float((values-torch.cat(reference)).abs().max())
                        controls.append(dict(defense=name, score_error=difference)); save_json(out/'function_controls.json', controls)
                        assert difference <= cfg['score_tolerance'], 'Recovery module function control failed; do not tune or omit this case.'
        del groups, attacked, release
    assert len(rows) == len(methods)*5
    if verify:
        save_json(out/'independent_validation.json', dict(status='passed', official_labels=True, NumPy_per_class_and_pairs=True,
            explicit_TA_reconstruction=True, complete_model_selected_scores=replays, maximum_score_error=error,
            source_sha256=sha(__file__), predictions_sha256=sha(out/'predictions.pt')))
    else:
        for t in ['MNIST', 'SVHN']:
            assert sha(Path('checkpoints/ViT-B-32')/t/'finetuned.pt') == next(r['sha256'] for r in json.loads(Path('configs/checkpoints.json').read_text())['files'] if r['path'] == f'ViT-B-32/{t}/finetuned.pt')
        save_json(out/'execution.json', dict(completed=True, rows=len(rows), GPU=torch.cuda.get_device_name(),
            method=LABEL, gradients=0, defense_updates=0, test_parameter_selection=False,
            full_from_same_frozen_attack_edits=full, source_sha256=sha(__file__), protocol_sha256=sha(root/'protocol.json'),
            predictions_sha256=sha(out/'predictions.pt'), wall_seconds=time.monotonic()-start))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    for name in ['check', 'pilot', 'full', 'verify']: group.add_argument('--'+name, action='store_true')
    parser.add_argument('--protocol', default='configs/params_adaptive.json')
    parser.add_argument('--full-output', action='store_true', help='Verify full instead of pilot results.')
    args = parser.parse_args()
    if args.check: check(Path('results/phase3/params_adaptive_check.json'))
    else: run(args.protocol, args.full or args.full_output, args.verify)
