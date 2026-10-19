"""One fixed LayerNorm-input affine gauge; no training or response tuning."""
import argparse
import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import digest
from confirm_temperature_compensation import unused
from covariance_noise import dataset
from local_residual_transport import metric
from mechanisms import load_author
from subspace_probe import make_loader, save_json
from task_probe_distance import checked_models, initialize

PREFIX = 'model.visual.transformer.resblocks.11.attn.'
KEYS = [PREFIX+'in_proj_weight', PREFIX+'in_proj_bias']


@torch.no_grad()
def build_matching(expert,anchor,cfg):
    pin=cfg['matching_geometry'];assert digest(pin['path'])==pin['sha256']
    raw=torch.load(pin['path'],map_location='cpu');keys=list(raw['original'])
    assert all(torch.equal(expert.state_dict()[k],v) for k,v in raw['original'].items())
    block=expert.model.visual.transformer.resblocks[11]
    assert torch.equal(raw['gamma'],block.ln_2.weight.double()) and torch.equal(raw['beta'],block.ln_2.bias.double())
    task=expert.state_dict()[keys[0]].double()-anchor.state_dict()[keys[0]].double()
    _,singular,vh=torch.linalg.svd(task,full_matrices=False)
    delta=raw['edits'][keys[0]].double()-raw['original'][keys[0]].double()
    inner=delta@vh[:32].T@vh[:32]
    distance=sum(float((raw['edits'][k].double()-v.double()).norm().square()/v.double().norm().square()) for k,v in raw['original'].items())**.5
    geometry=dict(parameter_distance=distance,logical_A_and_bias_distance=distance,
        LN_constant=float(raw['constant']),MLP_task_outside_energy_fraction=float((delta-inner).square().sum()/delta.square().sum()),
        task_probe_rank=32,task_singular_values=singular[:32].tolist(),
        changed_AGA_assignment_positions=int((raw['original_restore']!=raw['changed_restore']).sum()),
        scope='One weight-selected affine fingerprint point; changed assignment is not classification protection.')
    point={k:raw[k] for k in ['normal','constant','original','gamma','beta']}
    point.update(keys=keys,site='MLP',vh=vh[:32],assignment_geometry=geometry)
    return raw['edits'],point,geometry


@torch.no_grad()
def build(expert, anchor, cfg):
    if cfg.get('matching_geometry'):return build_matching(expert,anchor,cfg)
    block = expert.model.visual.transformer.resblocks[11]
    original = {k: expert.state_dict()[k].double().clone() for k in KEYS}
    w, b = [original[k] for k in KEYS]
    gamma, beta = block.ln_1.weight.double(), block.ln_1.bias.double()
    assert float(gamma.abs().min()) > 1e-8
    normal = gamma.reciprocal(); normal /= normal.norm()
    constant = normal @ beta
    task = w[:768]-anchor.state_dict()[KEYS[0]][:768].double()
    axes = []
    for h in range(12):
        u, _, _ = torch.linalg.svd(task[64*h:64*(h+1)], full_matrices=False)
        v = u[:, 0]; axes.append(v*torch.sign(v[v.abs().argmax()]))
    axis = torch.cat(axes); axis /= axis.norm()
    dw = torch.zeros_like(w); db = torch.zeros_like(b)
    dw[:768] = axis[:, None]*normal[None]; db[:768] = -constant*axis
    cost = float(dw.norm().square()/w.norm().square()+db.norm().square()/b.norm().square())
    scale = cfg['parameter_budget']*(1-cfg['distance_reserve'])/np.sqrt(cost)
    edits = dict(zip(KEYS, [(w+scale*dw).float(), (b+scale*db).float()]))
    delta = edits[KEYS[0]][:768].double()-w[:768]
    _, singular, vh = torch.linalg.svd(task, full_matrices=False)
    inner = delta@vh[:32].T@vh[:32]
    geometry = dict(parameter_distance=sum(float((edits[k].double()-original[k]).norm().square()/original[k].norm().square()) for k in KEYS)**.5,
        scale=scale, LN_constant=float(constant), minimum_abs_LN_gain=float(gamma.abs().min()),
        task_probe_rank=32, task_singular_values=singular[:32].tolist(),
        Q_task_outside_energy_fraction=float((delta-inner).norm().square()/delta.norm().square()),
        per_tensor_distance={k:float((edits[k].double()-original[k]).norm()/original[k].norm()) for k in KEYS})
    return edits, dict(normal=normal,constant=constant,axis=axis,vh=vh[:32],original=original,gamma=gamma,beta=beta), geometry


@torch.no_grad()
def recovery_geometry(expert, anchor, edits, point, cfg):
    if point.get('site')=='MLP':return dict(assignment_is_not_classification_damage=True)
    author, pin = load_author(Path(cfg['author_path'])); assert pin == cfg['author_text_sha256']
    base = anchor.state_dict()[KEYS[0]][:768].double(); original=point['original'][KEYS[0]][:768]
    changed=edits[KEYS[0]][:768].double(); gamma=point['gamma']; numerator=denominator=0.; errors=[]
    for i in range(0,768,64):
        a=base[i:i+64]; rec=[]
        for w in [original[i:i+64],changed[i:i+64]]:
            t=author['solve_linear_transformation'](w.T.float(),a.T.float())
            wn,an=w.numpy(),a.numpy(); tn=np.linalg.lstsq(wn@wn.T+1e-8*np.eye(64),wn@an.T,rcond=None)[0]
            errors.append(float(abs(tn-t.double().numpy()).max()))
            functional=(t.double().T@original[i:i+64])*gamma
            rec.append(functional-functional.mean(1,keepdim=True))
        numerator+=float((rec[1]-rec[0]).norm().square());denominator+=float(rec[0].norm().square())
    assert max(errors)<=cfg['author_weight_tolerance']
    return dict(Q_functional_operator_relative_change=(numerator/denominator)**.5,
        NumPy_author_transform_error=max(errors),scope='Q function after author head recovery, not classification harm or a security guarantee.')


@torch.no_grad()
def observe(model, head, train, ids, capture=False):
    values, inputs = [], []
    hook = model.model.visual.transformer.resblocks[11].register_forward_pre_hook(
        lambda _m,args: inputs.append(args[0].transpose(0,1).detach().cpu())) if capture else None
    try:
        model.cuda().eval(); head.cuda().eval()
        for images, _ in make_loader(train, ids, 64): values.append(head(model(images.cuda())).cpu().double())
    finally:
        model.cpu(); head.cpu()
        if hook is not None: hook.remove()
    return torch.cat(values), torch.cat(inputs) if capture else None


@torch.no_grad()
def probe_scores(model, head, inputs):
    visual = model.model.visual; scores = []
    visual.cuda().eval(); head.cuda().eval()
    for part in inputs.split(64):
        h = visual.transformer.resblocks[11](part.cuda().transpose(0,1))[0]
        scores.append(head(visual.ln_post(h)@visual.proj).cpu().double())
    visual.cpu(); head.cpu()
    return torch.cat(scores)


@torch.no_grad()
def run(protocol, confirm=False):
    start = time.monotonic(); cfg = json.loads(Path(protocol).read_text()); out = Path(cfg['output'])
    for path, pin in cfg['source_sha256'].items(): assert digest(path) == pin, path
    assert '3090' in torch.cuda.get_device_name()
    anchor, expert, head, pins = checked_models('MNIST')
    if not confirm:
        out.mkdir(parents=True, exist_ok=False); save_json(out/'protocol.json', cfg)
        edits, point, geometry = build(expert, anchor, cfg)
        assert geometry['parameter_distance'] <= cfg['parameter_budget']
        geometry.update(recovery_geometry(expert,anchor,edits,point,cfg))
        torch.save(edits, out/'frozen_release.pt'); torch.save(point, out/'point.pt')
        save_json(out/'before_images.json', dict(release_sha256=digest(out/'frozen_release.pt'),point_sha256=digest(out/'point.pt'),
            geometry=geometry,checkpoint_sha256=pins,source_sha256=digest(__file__),protocol_sha256=digest(protocol)))
        name, seed = 'source', cfg['construction_seed']
    else:
        assert json.loads((out/'results.json').read_text())['source_feasible']
        assert json.loads((out/'independent_validation.json').read_text())['status'] == 'passed'
        name, seed = 'confirmation', cfg['confirmation_seed']
    assert not (out/(name+'.pt')).exists(), 'No response-driven repeat.'
    freeze = json.loads((out/'before_images.json').read_text()); geometry = freeze['geometry']
    for file, field in [('frozen_release.pt','release_sha256'),('point.pt','point_sha256')]: assert digest(out/file) == freeze[field]
    edits = torch.load(out/'frozen_release.pt',map_location='cpu'); point = torch.load(out/'point.pt',map_location='cpu')
    release = copy.deepcopy(expert)
    for k, value in edits.items(): release.state_dict()[k].copy_(value)
    train = dataset('MNIST','data',expert.val_preprocess,True); excluded = set(cfg['extra_excluded_indices'])
    for path in cfg['extra_index_exclusions']: excluded.update(json.loads(Path(path).read_text())['indices'])
    split = unused(train,excluded,cfg|dict(confirmation_n=cfg['check_n'],confirmation_seed=seed))
    save_json(out/(name+'_split.json'),split)
    save_json(out/('training_indices.json' if not confirm else 'confirmation_split.json'),dict(check=split['indices']) if not confirm else split)
    before, x = observe(expert,head,train,split['indices'],True); after, _ = observe(release,head,train,split['indices'])
    y = before.argmax(1); take = torch.stack([(y==k).nonzero().flatten()[0] for k in range(10)])
    v = point['vh'].float(); rms = (x[:,0]@v.T).square().mean(0).sqrt()
    shifts = (cfg['finite_step']*torch.tensor([-1.,1.])[:,None,None]*rms[None,:,None]*v[None]).flatten(0,1)
    probes = x[take].repeat_interleave(len(shifts),0); probes[:,0] += shifts.repeat(len(take),1)
    pb, pa = probe_scores(expert,head,probes), probe_scores(release,head,probes)
    control = float((probe_scores(expert,head,x)-before).abs().max()); assert control <= cfg['score_tolerance']
    natural, finite = metric(before,after,len(before)),metric(pb,pa,len(pb))
    feasible = all(m['worst_task_drift']<=cfg['task_budget'] and m['changed']==0 for m in [natural,finite])
    labels = torch.tensor([int(train[i][1]) for i in split['indices']])
    torch.save(dict(before=before,after=after,probe_before=pb,probe_after=pa,inputs=x,probe_inputs=probes,
        indices=split['indices'],labels=labels),out/(name+'.pt'))
    report = dict(status='passed' if feasible else 'closed_task_budget_failure',source_feasible=feasible,
        geometry=geometry,natural=natural,finite_probes=finite,full_block_score_control=control,
        release_sha256=freeze['release_sha256'],raw_sha256=digest(out/(name+'.pt')),source_sha256=digest(__file__),
        checkpoint_sha256=pins,protocol_sha256=digest(protocol),expert_updates=0,gradients=0,
        partner_used=False,test_used=False,wall_seconds=time.monotonic()-start)
    save_json(out/('confirmation_validation.json' if confirm else 'results.json'),report)
    print(json.dumps(report),flush=True)


def verify(folder, confirm=False):
    from audit_hermite_splits import exclusions
    from verify_covariance_noise import official_labels
    out=Path(folder); name='confirmation' if confirm else 'source'
    report=json.loads((out/('confirmation_validation.json' if confirm else 'results.json')).read_text())
    assert digest(out/(name+'.pt'))==report['raw_sha256']
    raw=torch.load(out/(name+'.pt'),map_location='cpu'); point=torch.load(out/'point.pt',map_location='cpu')
    edits=torch.load(out/'frozen_release.pt',map_location='cpu'); n=point['normal'].numpy(); g=point['gamma'].numpy(); beta=point['beta'].numpy()
    keys=point.get('keys',KEYS)
    dw=(edits[keys[0]].double()-point['original'][keys[0]].double()).numpy()
    db=(edits[keys[1]].double()-point['original'][keys[1]].double()).numpy()
    linear=dw*g; centered=linear-linear.mean(1,keepdims=True)
    identity=max(float(abs(centered).max()),float(abs(dw@beta+db).max())); assert identity<1e-6
    distance=sum(float(np.linalg.norm((edits[k].double()-v).numpy())**2/np.linalg.norm(v.numpy())**2) for k,v in point['original'].items())**.5
    assert abs(distance-report['geometry']['parameter_distance'])<1e-10
    for (ka,kb), field in [(('before','after'),'natural'),(('probe_before','probe_after'),'finite_probes')]:
        a,b=raw[ka].numpy(),raw[kb].numpy(); y=a.argmax(1); ix=np.arange(len(a)); gap=a[ix,y,None]-a; gap[ix,y]=1
        delta=b-a; r=abs((delta[ix,y,None]-delta)/np.maximum(gap,1e-12));r[ix,y]=0
        assert abs(float(r.max())-report[field]['worst_task_drift'])<1e-9
        assert int((a.argmax(1)!=b.argmax(1)).sum())==report[field]['changed']
    cfg=json.loads((out/'protocol.json').read_text()); split=json.loads((out/(name+'_split.json')).read_text()); initial=set(cfg['extra_excluded_indices'])
    for path in cfg['extra_index_exclusions']: initial.update(json.loads(Path(path).read_text())['indices'])
    excluded,exposed=exclusions(split,initial); assert len(excluded)==split['excluded_count']
    draw=np.random.default_rng(split['seed']).choice(sorted(set(range(60000))-excluded),cfg['check_n'],replace=False).tolist()
    assert draw==raw['indices']==split['indices'] and not set(draw)&exposed
    np.testing.assert_array_equal(official_labels('MNIST',True)[draw],raw['labels'])
    save_json(out/('confirmation_independent_validation.json' if confirm else 'independent_validation.json'),
        dict(status='passed',raw_sha256=report['raw_sha256'],parameter_distance=distance,NumPy_affine_identity_error=identity,
            natural_and_probe_margin_recount=True,source_indices_and_labels_checked=True,source_sha256=digest(__file__)))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol');ap.add_argument('--confirm',action='store_true');ap.add_argument('--verify')
    args=ap.parse_args();initialize()
    verify(args.verify,args.confirm) if args.verify else run(args.protocol,args.confirm)
