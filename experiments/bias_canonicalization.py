"""Architecture-aware LN-bias recovery; supplementary adaptation, not author AGA."""

import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import checkpoint_loader, digest, evaluate, merge
from covariance_noise import dataset
from phase3_evaluate import apply, row_metrics
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels
from verify_probe_candidate import explicit_merge


def inputs(protocol):
    cfg=json.loads(Path(protocol).read_text())
    assert cfg['GPU'].removeprefix('RTX') in torch.cuda.get_device_name()
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    for path,pin in cfg['source_sha256'].items(): assert digest(path)==pin,path
    assert digest(cfg['frozen_protocol'])==cfg['frozen_protocol_sha256']
    frozen=json.loads(Path(cfg['frozen_protocol']).read_text())
    pins={r['path']:r['sha256'] for r in json.loads(Path('configs/checkpoints.json').read_text())['files']}
    load=checkpoint_loader(Path('external/secure-merge-attack'))
    def checked(name):
        path=Path('checkpoints/ViT-B-32')/name; assert digest(path)==pins['ViT-B-32/'+name]
        model=load(str(path)).cpu().eval(); model.requires_grad_(False); return model
    return cfg,frozen,checked


@torch.no_grad()
def canonicalize(model, independent=False):
    """Minimum-norm bias with the same published visual projection response."""
    p = model.model.visual.proj.detach().cpu().double()
    beta = model.model.visual.ln_post.bias.detach().cpu().double()
    q, r = torch.linalg.qr(p, mode='reduced')
    assert torch.linalg.matrix_rank(r) == p.shape[1]
    ideal = torch.from_numpy(np.linalg.lstsq(p.numpy().T, (beta@p).numpy(), rcond=None)[0]) if independent else q@(q.T@beta)
    error = float(((ideal-beta)@p).abs().max())
    assert error < 1e-8, 'Ideal function-preservation identity failed.'
    result = copy.deepcopy(model)
    result.model.visual.ln_post.bias.copy_(ideal.float())
    actual = result.model.visual.ln_post.bias.double()
    return result, dict(ideal_projection_error=error,
        FP32_projection_error=float(((actual-beta)@p).abs().max()),
        bias_norm_before=float(beta.norm()), bias_norm_after=float(actual.norm()))


@torch.no_grad()
def run(protocol, full=False, verify=False):
    start = time.monotonic()
    cfg, frozen, checked = inputs(protocol)
    root = Path(cfg['output']); tasks = sorted({t for case in cfg['cases'] for t in case.values()})
    if not full and not verify:
        root.mkdir(parents=True, exist_ok=False); (root/'artifacts').mkdir()
        save_json(root/'protocol.json', cfg); records = {}
        if cfg.get('frozen_recovery_reference'):
            ref=cfg['frozen_recovery_reference']; assert digest(ref['path'])==ref['sha256']
            previous=json.loads(Path(ref['path']).read_text()); assert previous['constructed_before_images']
            records=previous['records']
        for task in ([] if records else tasks):
            own = checked(task+'/finetuned.pt')
            for name in ['Unprotected', 'Phase1_'+task]:
                model = own if name == 'Unprotected' else apply(own, frozen['defenses'][name]['release'])
                clean, geometry = canonicalize(model)
                ref = root/'artifacts'/(task+'_'+name+'.pt')
                torch.save({'model.visual.ln_post.bias': clean.model.visual.ln_post.bias.cpu()}, ref)
                records[task+'/'+name] = dict(path=str(ref), sha256=digest(ref), **geometry)
        save_json(root/'freeze.json', dict(records=records, constructed_before_images=True,
            protocol_sha256=digest(root/'protocol.json'), attack_data=0, defense_updates=0))
    else:
        assert cfg == json.loads((root/'protocol.json').read_text())
        if full: assert json.loads((root/'pilot/validation.json').read_text())['status'] == 'passed'
    freeze = json.loads((root/'freeze.json').read_text())
    assert freeze['constructed_before_images'] and digest(root/'protocol.json') == freeze['protocol_sha256']
    out = root/('full' if full else 'pilot')
    if not verify: out.mkdir(exist_ok=False)
    raw = torch.load(out/'predictions.pt', map_location='cpu', weights_only=False) if verify else {}
    rows = json.loads((out/'results.json').read_text()) if verify else []
    if verify: assert digest(out/'predictions.pt') == json.loads((out/'execution.json').read_text())['predictions_sha256']
    controls, replays, maximum_error = [], 0, 0.
    anchor = checked('zeroshot.pt')
    for ci, case in enumerate(cfg['cases']):
        target, partner = case['target'], case['partner']
        own, peer = [checked(t+'/finetuned.pt') for t in (target, partner)]
        heads = {t: checked('head_'+t+'.pt') for t in (target, partner)}
        data = {t: dataset(t, 'data', own.val_preprocess) for t in heads}
        for t in data: assert len(data[t]) == frozen['tasks'][t]['test_n']
        clean_peer = canonicalize(peer, True)[0] if verify else apply(peer, freeze['records'][partner+'/Unprotected'])
        for name in ['Unprotected', 'Phase1_'+target]:
            release = own if name == 'Unprotected' else apply(own, frozen['defenses'][name]['release'])
            clean = canonicalize(release, True)[0] if verify else apply(release, freeze['records'][target+'/'+name])
            fusion = explicit_merge if verify else merge
            groups = [('Standalone', 'Protected expert', clean, [target]),
                ('TA', 'Protected expert', fusion(anchor, clean, peer, .5), list(heads)),
                ('TA', 'All experts', fusion(anchor, clean, clean_peer, .5), list(heads))]
            baseline = {} if name == 'Unprotected' else baseline
            for method, scope, model, evaluated in groups:
                for task in evaluated:
                    ids = list(range(len(data[task]))) if full else np.linspace(0,len(data[task])-1,cfg['pilot_n'],dtype=int).tolist()
                    key = '/'.join((str(ci),name,method,scope,task))
                    if verify:
                        record = raw[key]; logits = record['logits']; labels = official_labels(task)[ids]
                        np.testing.assert_array_equal(labels,record['target'].numpy())
                        row = next(r for r in rows if r['key']==key); prediction = logits.numpy().argmax(1)
                        assert int((prediction==labels).sum()) == row['correct']
                        control = raw['/'.join((str(ci),'Unprotected',method,scope,task))]['logits'].numpy().argmax(1)
                        assert int((prediction!=control).sum()) == row['changed_predictions']
                        assert int(((control==labels)&(prediction!=labels)).sum()) == row['paired']['lost']
                        assert int(((control!=labels)&(prediction==labels)).sum()) == row['paired']['gained']
                        for c in row['per_class']:
                            mask = labels==c['class_id']; assert int(mask.sum())==c['n'] and int(((prediction==labels)&mask).sum())==c['correct']
                        for pos in np.linspace(0,len(ids)-1,cfg['replay_n'],dtype=int):
                            lo=int(pos)//128*128; hi=min(lo+128,len(ids))
                            x=torch.stack([data[task][i][0] for i in ids[lo:hi]]).cuda()
                            model.cuda(); heads[task].cuda(); actual=heads[task](model(x)).cpu()[int(pos)-lo]; model.cpu(); heads[task].cpu()
                            err=float((actual-logits[int(pos)]).abs().max()); maximum_error=max(maximum_error,err)
                            assert err<=cfg['score_tolerance'] and int(actual.argmax())==int(prediction[int(pos)]); replays+=1
                    else:
                        scores, labels = [], []
                        metric=evaluate(model,heads[task],make_loader(data[task],ids),'cuda',logit_sink=scores.append,prediction_sink=lambda p,y:labels.append(y))
                        scores, labels=torch.cat(scores),torch.cat(labels)
                        np.testing.assert_array_equal(labels.numpy(),official_labels(task)[ids]); assert torch.isfinite(scores).all()
                        if name=='Unprotected': baseline[method,scope,task]=scores
                        rows.append(dict(key=key,target=target,partner=partner,defense=name,merger=method,attack_scope=scope,
                            recovery='Published-projection LN-bias canonicalization',dataset=task,role='target' if task==target else 'partner',
                            **row_metrics(scores,labels,baseline[method,scope,task],frozen['tasks'][task]['classes']),evaluation_seconds=metric['evaluation_seconds']))
                        raw[key]=dict(logits=scores,target=labels,indices=ids)
                        save_json(out/'results.json',rows); torch.save(raw,out/'predictions.pt')
                        if not full and method=='Standalone':
                            before=[]; evaluate(release,heads[task],make_loader(data[task],ids),'cuda',logit_sink=before.append)
                            err=float((scores-torch.cat(before)).abs().max()); controls.append(dict(case=ci,defense=name,max_score_error=err))
                            save_json(out/'function_controls.json',controls); assert err<=cfg['score_tolerance']
            del groups,release,clean
    assert len(rows)==len(cfg['cases'])*10
    if verify:
        save_json(out/'validation.json',dict(status='passed',NumPy_labels_classes_pairs=True,NumPy_lstsq_bias_reconstruction=True,
            explicit_TA=True,selected_score_replays=replays,maximum_score_error=maximum_error,predictions_sha256=digest(out/'predictions.pt'),source_sha256=digest(__file__)))
    else:
        save_json(out/'execution.json',dict(completed=True,rows=len(rows),wall_seconds=time.monotonic()-start,GPU=torch.cuda.get_device_name(),
            defense_updates=0,gradients=0,test_parameter_selection=False,full=full,predictions_sha256=digest(out/'predictions.pt'),source_sha256=digest(__file__)))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__); ap.add_argument('--protocol',default='configs/bias_canonicalization.json')
    ap.add_argument('--full',action='store_true'); ap.add_argument('--verify',action='store_true'); args=ap.parse_args()
    run(args.protocol,args.full,args.verify)
