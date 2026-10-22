"""Frozen LN-related publication controls; no training or response tuning."""
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
def build_mean_compensation(expert, anchor, cfg):
    """One weight-selected finite mean-equivalent coordinate transform."""
    from author_protocol_audit import logical_parameter_distance
    ref=cfg['mean_compensation_geometry'];assert digest(ref['path'])==ref['sha256']
    root=Path(ref['path']).parent;assert json.loads((root/'independent_validation.json').read_text())['status']=='passed'
    p=torch.load(ref['path'],map_location='cpu');u=p['axis'];q=torch.eye(768,dtype=torch.double)-2*u[:,None]*u[None]
    state=expert.state_dict();edits={};visual='model.visual.'
    for key in ['conv1.weight','class_embedding','positional_embedding']:
        w=state[visual+key].double();edits[visual+key]=(w@q if key=='positional_embedding' else (q@w.flatten(1)).reshape_as(w) if w.ndim>1 else q@w).float()
    edits[visual+'ln_pre.bias']=(q@state[visual+'ln_pre.bias'].double()).float()
    for i in range(12):
        prefix=f'{visual}transformer.resblocks.{i}.'
        for ln,read,write in [('ln_1.','attn.in_proj_','attn.out_proj.'),('ln_2.','mlp.c_fc.','mlp.c_proj.')]:
            gain,beta=[state[prefix+ln+k].double() for k in ['weight','bias']]
            w,b=[state[prefix+read+k].double() for k in ['weight','bias']]
            wp=((w*gain[None])@q)/gain[None];t=p['affine_shares'][prefix+ln]
            edits[prefix+read+'weight']=wp.float()
            edits[prefix+read+'bias']=(b+(1-t)*(w-wp)@beta).float()
            edits[prefix+ln+'bias']=(beta+t*(gain*(q@(beta/gain))-beta)).float()
            for name in ['weight','bias']:edits[prefix+write+name]=(q@state[prefix+write+name].double()).float()
    gain,beta=[state[visual+'ln_post.'+k].double() for k in ['weight','bias']]
    edits[visual+'proj']=((q@(gain[:,None]*state[visual+'proj'].double()))/gain[:,None]).float()
    edits[visual+'ln_post.bias']=(gain*(q@(beta/gain))).float()
    edits={k:v for k,v in edits.items() if not torch.equal(v,state[k])}
    basis=torch.load(cfg['task_basis']['path'],map_location='cpu');assert digest(cfg['task_basis']['path'])==cfg['task_basis']['sha256']
    inside=outside=0.;vh=None
    for item in basis.values():
        k=item['source_key'];lo,hi=item['row_start'],item['row_end'];w=state[k][lo:hi].double().flatten(1)
        e=edits.get(k,state[k])[lo:hi].double().flatten(1)-w;v=item['v'];ein=e@v
        inside+=float(ein.square().sum()/w.square().sum());outside+=float((e-ein@v.T).square().sum()/w.square().sum())
        if k==KEYS[0] and lo==0:vh=v.T
    maximum=max(float((v.double()-state[k].double()).norm()/state[k].double().norm()) for k,v in edits.items() if k.endswith('bias'))
    assert inside**.5<=cfg['task_direction_parameter_limit'] and maximum<=cfg['individual_bias_limit']
    geometry=dict(parameter_distance=logical_parameter_distance(edits,state),task_inside_distance=inside**.5,task_outside_distance=outside**.5,
        maximum_relative_bias=maximum,selected_index=p['selected_index'],changed_tensors=len(edits),original_total_D0_5_comparison=False,
        no_method_novelty_claim=True,public_orthogonal_alignment_is_a_separate_permission=True)
    return edits,dict(site='MEAN',axis=u,keys=list(edits),affine_shares=p['affine_shares'],vh=vh),geometry


@torch.no_grad()
def build_structural_unit(expert,anchor,cfg):
    ref=cfg['structural_unit_geometry'];assert digest(ref['path'])==ref['sha256']
    root=Path(ref['path']).parent;check=json.loads((root/'results.json').read_text())
    assert check['necessary_source_passed'] and json.loads((root/'independent_validation.json').read_text())['status']=='passed'
    d=torch.load(ref['path'],map_location='cpu');j=d['unit'];prefix='model.visual.transformer.resblocks.11.mlp.'
    keys=[prefix+'c_fc.weight',prefix+'c_fc.bias',prefix+'c_proj.weight'];original={k:expert.state_dict()[k].double().clone() for k in keys}
    edits={k:v.float() for k,v in original.items()}
    edits[keys[0]][j]=d['edits']['A_row'];edits[keys[1]][j]=d['edits']['a_value'];edits[keys[2]][:,j]=d['edits']['B_column']
    block=expert.model.visual.transformer.resblocks[11]
    assert torch.equal(block.ln_2.weight.double(),d['gamma2']) and torch.equal(block.ln_2.bias.double(),d['beta2'])
    D=sum(float((edits[k].double()-v).square().sum()/v.square().sum()) for k,v in original.items())**.5
    assert abs(D-check['parameter_distance'])<1e-10
    geometry=dict(parameter_distance=D,unit=j,eta=d['eta'],known_components_not_original_defense=True,
        source_pruning_cost_not_zero=True,task_probe_rank=32,
        per_tensor_distance=check['per_tensor_distance'],A_task32_inside_distance=check['A_task32_inside_distance'],
        B_task32_inside_distance=check['B_task32_inside_distance'],data_free_canonical_cleanup_disclosed=True)
    point=dict(site='STRUCTURAL_UNIT',original=original,keys=keys,unit=j,eta=d['eta'],u=d['u'],
        gamma=d['gamma2'],beta=d['beta2'],normal=d['n'],constant=d['c'],vh=d['Va'],B_shifts=d['shifts'],
        structural_source=ref)
    return edits,point,geometry


@torch.no_grad()
def build_readout_compensation(expert,anchor,cfg):
    """One known affine gauge with a nonzero finite interpolation cross term."""
    bk,pk='model.visual.ln_post.bias','model.visual.proj';state=expert.state_dict()
    beta,p,g=[state[k].double() for k in [bk,pk,'model.visual.ln_post.weight']]
    assert float(g.abs().min())>1e-8
    n=1/g;n=n/n.norm();c=beta@n;assert abs(float(c))>1e-8
    a=float(c.square()/beta.square().sum());b=float((n@p).square().sum()/p.square().sum())
    lo,hi=0.,1.;radius=cfg['parameter_budget']*(1-cfg['distance_reserve'])
    for _ in range(64):
        r=(lo+hi)/2
        if a*r*r+b*(r/(1-r))**2>radius**2:hi=r
        else:lo=r
    r=lo;edits={bk:(beta-r*c*n).float(),pk:(p+(r/(1-r))*n[:,None]*(n@p)[None]).float()}
    original={bk:state[bk],pk:state[pk]};dp=edits[pk].double()-p;db=edits[bk].double()-beta
    u,sv,_=torch.linalg.svd(p-anchor.state_dict()[pk].double(),full_matrices=False);inside=u[:,:32]@(u[:,:32].T@dp)
    # The same canonicalization is available to an attacker without the secret r.
    def canonical(bb,pp):
        offset=bb@n;return bb+(1-offset)*n,pp+(offset-1)*n[:,None]*(n@pp)[None]
    cb,cp=canonical(beta,p);rb,rp=canonical(edits[bk].double(),edits[pk].double())
    geometry=dict(parameter_distance=sum(float((edits[k].double()-v.double()).square().sum()/v.double().square().sum()) for k,v in original.items())**.5,
        r=r,LN_constant=float(c),per_tensor_distance={k:float((edits[k].double()-v.double()).norm()/v.double().norm()) for k,v in original.items()},
        P_task_outside_energy_fraction=float((dp-inside).square().sum()/dp.square().sum()),task_probe_rank=32,
        probe_coordinates='Left singular vectors of the row-vector projection task matrix; perturb actual final LN inputs.',
        task_singular_values=sv[:32].tolist(),self_midpoint_feature_shift_norm=float((c*r*r/(4*(1-r))*(n@p)).norm()),
        canonical_parameter_max_abs_error=max(float((cb-rb).abs().max()),float((cp-rp).abs().max())),
        known_affine_symmetry_not_novel=True,adaptive_cleanup_disclosed_before_images=True)
    return edits,dict(site='READOUT',normal=n,constant=c,original=original,gamma=g,beta=beta,projection=p,eps=expert.model.visual.ln_post.eps,vh=u[:,:32].T),geometry


@torch.no_grad()
def build_head_identity(expert,anchor,cfg):
    pin=cfg['head_identity_geometry'];assert digest(pin['path'])==pin['sha256']
    raw=torch.load(pin['path'],map_location='cpu')
    assert all(torch.equal(expert.state_dict()[k],v) for k,v in raw['original'].items())
    from author_protocol_audit import logical_parameter_distance
    distance=logical_parameter_distance(raw['edits'],raw['original'])
    key='model.visual.transformer.resblocks.11.mlp.c_fc.weight'
    _,singular,vh=torch.linalg.svd(expert.state_dict()[key].double()-anchor.state_dict()[key].double(),full_matrices=False)
    geometry=dict(parameter_distance=distance,original_0_5_parameter_budget_passed=distance<=.5,
        separately_declared_parameter_budget=cfg['parameter_budget'],known_symmetry_control_not_new_defense=True,
        task_probe_rank=32,task_singular_values=singular[:32].tolist())
    return raw['edits'],dict(site='HEAD',original=raw['original'],head_order=raw['head_order'],vh=vh[:32]),geometry


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
    if cfg.get('mean_compensation_geometry'):return build_mean_compensation(expert,anchor,cfg)
    if cfg.get('residual_coordinate_control'):
        from aga_coordinate_audit import residual_permutation
        from author_protocol_audit import logical_parameter_distance
        state=expert.state_dict();order=torch.arange(767,-1,-1);edits=residual_permutation(state,order)
        task=state[KEYS[0]][:768].double()-anchor.state_dict()[KEYS[0]][:768].double()
        _,_,vh=torch.linalg.svd(task,full_matrices=False)
        geometry=dict(parameter_distance=logical_parameter_distance(edits,state),original_budget=0.5,
            declared_control_budget=cfg['parameter_budget'],changed_tensors=len(edits),known_symmetry_not_new_defense=True,
            task_subspace_preserved_by_covariant_function_probes=True,parameter_subspace_invariance_not_claimed=True,
            order_rule='reverse all768 residual coordinates once;no response selection',blind_public_patch_matching_disclosed=True)
        return edits,dict(site='RESIDUAL',order=order,keys=list(edits),vh=vh[:32].clone()),geometry
    if cfg.get('structural_unit_geometry'):return build_structural_unit(expert,anchor,cfg)
    if cfg.get('readout_compensation'):return build_readout_compensation(expert,anchor,cfg)
    if cfg.get('head_identity_geometry'):return build_head_identity(expert,anchor,cfg)
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
    if point.get('site')=='MEAN':return dict(finite_mean_equivalence_not_AGA_security=True,full_AGA_not_yet_run=True)
    if point.get('site')=='RESIDUAL':return dict(global_residual_symmetry_known=True,original_AGA_not_global_residual_matching=True,full_AGA_not_yet_run=True)
    if point.get('site')=='STRUCTURAL_UNIT':return dict(native_zero_unit_preserved_by_MLP_permutation_in_real_arithmetic=True,
        adaptive_canonical_cleanup_disclosed=True,actual_full_AGA_not_yet_run=True)
    if point.get('site')=='READOUT':return dict(original_AGA_does_not_edit_final_LN_or_projection=True,not_adaptive_attack_robust=True)
    if point.get('site')=='HEAD':return dict(known_head_symmetry_not_a_recovery_guarantee=True)
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
    layer=model.model.visual.ln_post if capture=='readout' else model.model.visual.transformer.resblocks[11]
    hook = layer.register_forward_pre_hook(
        lambda _m,args: inputs.append((args[0] if capture=='readout' else args[0].transpose(0,1)).detach().cpu())) if capture else None
    try:
        model.cuda().eval(); head.cuda().eval()
        for images, _ in make_loader(train, ids, 64): values.append(head(model(images.cuda())).cpu().double())
    finally:
        model.cpu(); head.cpu()
        if hook is not None: hook.remove()
    return torch.cat(values), torch.cat(inputs) if capture else None


@torch.no_grad()
def probe_scores(model, head, inputs, readout=False):
    visual = model.model.visual; scores = []
    visual.cuda().eval(); head.cuda().eval()
    for part in inputs.split(64):
        h = part.cuda() if readout else visual.transformer.resblocks[11](part.cuda().transpose(0,1))[0]
        scores.append(head(visual.ln_post(h)@visual.proj).cpu().double())
    visual.cpu(); head.cpu()
    return torch.cat(scores)


@torch.no_grad()
def structural_B_probes(model,head,inputs,shifts):
    """Keep the old post-QuickGELU intervention, including the dormant unit."""
    at=0
    def inject(_m,args):
        nonlocal at
        z=args[0].clone();n=z.shape[1];z[0]+=shifts[at:at+n].to(z);at+=n
        return (z,)
    hook=model.model.visual.transformer.resblocks[11].mlp.c_proj.register_forward_pre_hook(inject)
    try:result=probe_scores(model,head,inputs)
    finally:hook.remove()
    assert at==len(shifts);return result


@torch.no_grad()
def mean_components(folder, protocol=None):
    """Source-only counterfactuals of one frozen, jointly compensated release."""
    out=Path(folder);cfg=json.loads((out/'protocol.json').read_text());raw=torch.load(out/'source.pt',map_location='cpu')
    assert digest(out/'source.pt')==json.loads((out/'results.json').read_text())['raw_sha256']
    assert not (out/'component_scores.pt').exists()
    if protocol:
        audit=json.loads(Path(protocol).read_text());assert audit['source_directory']==folder
        for path,pin in audit['input_sha256'].items():assert digest(path)==pin,path
        assert digest(__file__)==audit['source_sha256']
    _,expert,head,_=checked_models(cfg['target']);edits=torch.load(out/'frozen_release.pt',map_location='cpu')
    assert digest(out/'frozen_release.pt')==json.loads((out/'before_images.json').read_text())['release_sha256']
    original={k:expert.state_dict()[k].clone() for k in edits};inner={};outer={}
    coordinate=cfg.get('residual_coordinate_control',False)
    if coordinate:
        assert protocol,'Register the fixed coordinate-component comparison first.'
        def group(k):
            if k.startswith('model.visual.transformer.'):return 'transformer_only'
            if k.startswith('model.visual.ln_post.') or k=='model.visual.proj':return 'readout_only'
            return 'input_only'
        variants={name:{k:v for k,v in edits.items() if group(k)==name}
                  for name in ['input_only','transformer_only','readout_only']}
        assert set().union(*(set(v) for v in variants.values()))==set(edits)
        assert sum(len(v) for v in variants.values())==len(edits)
        variants['complete']=edits
    else:
        assert digest(cfg['task_basis']['path'])==cfg['task_basis']['sha256']
        for item in torch.load(cfg['task_basis']['path'],map_location='cpu').values():
            key=item['source_key'];lo,hi=item['row_start'],item['row_end'];v=item['v']
            w=original[key][lo:hi].double().flatten(1);e=edits[key][lo:hi].double().flatten(1)-w;ein=(e@v)@v.T
            if key not in inner:inner[key]=original[key].clone();outer[key]=original[key].clone()
            inner[key][lo:hi]=(w+ein).reshape_as(original[key][lo:hi]).float()
            outer[key][lo:hi]=(w+e-ein).reshape_as(original[key][lo:hi]).float()
        variants=dict(task_inside=inner,task_outside=outer,all_matrices={k:edits[k] for k in inner},
            other_parameters={k:v for k,v in edits.items() if k not in inner},complete=edits)
    save_json(out/'component_protocol.json',dict(source_sha256=digest(__file__),source_split_sha256=digest(out/'source_split.json'),
        release_sha256=digest(out/'frozen_release.pt'),variants=list(variants),edited_keys={k:list(v) for k,v in variants.items()},rank=None if coordinate else 32,source_only=True,
        separately_quantized_counterfactuals=True,no_candidate_or_attack=True))
    data=dataset(cfg['target'],'data',expert.val_preprocess,True);scores={};rows=[];start=time.monotonic()
    for name,values in variants.items():
        for key,value in (original|values).items():expert.state_dict()[key].copy_(value)
        result,_=observe(expert,head,data,raw['indices']);scores[name]=result
        m=metric(raw['before'],result,len(result));before=raw['before'].argmax(1);after=result.argmax(1);labels=raw['labels']
        rows.append(dict(name=name,metric=m,correct=int((after==labels).sum()),n=len(labels),
            per_class=[dict(class_id=k,n=int((labels==k).sum()),lost=int(((before==labels)&(after!=labels)&(labels==k)).sum()),
                gained=int(((before!=labels)&(after==labels)&(labels==k)).sum())) for k in range(10)]))
    error=float((scores['complete']-raw['after']).abs().max());assert error<=cfg['score_tolerance']
    torch.save(dict(before=raw['before'],scores=scores,indices=raw['indices'],labels=raw['labels']),out/'component_scores.pt')
    save_json(out/'components.json',dict(status='completed_source_counterfactuals',rows=rows,full_source_replay_error=error,
        raw_sha256=digest(out/'component_scores.pt'),source_sha256=digest(__file__),new_image_forwards=len(variants)*len(raw['indices']),
        new_AGA_calls=0,new_candidate=False,wall_seconds=time.monotonic()-start))


def verify_components(folder):
    """Independent NumPy recount of fixed source component scores and tables."""
    from verify_covariance_noise import official_labels
    out=Path(folder);meta=json.loads((out/'components.json').read_text())
    assert digest(out/'component_scores.pt')==meta['raw_sha256']
    raw=torch.load(out/'component_scores.pt',map_location='cpu');before=raw['before'].numpy()
    y=official_labels('MNIST',True)[raw['indices']];np.testing.assert_array_equal(y,raw['labels'])
    winner=before.argmax(1);ix=np.arange(len(y));gap=before[ix,winner,None]-before
    mask=np.ones_like(gap,dtype=bool);mask[ix,winner]=False;assert (gap[mask]>0).all()
    rows=[]
    for row in meta['rows']:
        scores=raw['scores'][row['name']].numpy();assert np.isfinite(scores).all();pred=scores.argmax(1)
        assert int((pred==y).sum())==row['correct'] and int((pred!=winner).sum())==row['metric']['changed']
        delta=scores-before;ratio=(delta[ix,winner,None]-delta)/np.where(mask,gap,1.);ratio[ix,winner]=0
        assert abs(float(abs(ratio).max())-row['metric']['worst_task_drift'])<1e-10
        for c in row['per_class']:
            take=y==c['class_id'];assert int(take.sum())==c['n']
            assert int(((winner==y)&(pred!=y)&take).sum())==c['lost']
            assert int(((winner!=y)&(pred==y)&take).sum())==c['gained']
        rows.append([row['name'].replace('_',' '),f"{row['correct']}/{len(y)}",str(row['metric']['changed']),f"{row['metric']['worst_task_drift']:.6g}"])
    md=['# Frozen-release component comparison','',
        'Same512 original construction images. Separate counterfactuals, no new defense selection, training or attack. The original full-model TA/AGA results remain unchanged.','',
        '| Components applied | Correct | Changed predictions | Relative class-gap change |','|---|---:|---:|---:|']
    md+=['| '+' | '.join(row)+' |' for row in rows];(out/'components.md').write_text('\n'.join(md)+'\n')
    tex=['\\begin{table}[t]','\\centering','\\small','\\begin{tabular}{lrrr}','\\toprule',
         'Components & Correct & Changed & Gap change \\\\','\\midrule']
    tex+=[' & '.join(row)+' \\\\' for row in rows]
    tex+=['\\bottomrule','\\end{tabular}',
          '\\caption{Fixed source component counterfactuals on512 training images. These are standalone responses, not fusion or attack results.}',
          '\\label{tab:coordinate-components}','\\end{table}','']
    (out/'components.tex').write_text('\n'.join(tex))
    save_json(out/'components_independent_validation.json',dict(status='passed_NumPy_score_class_and_label_recount',
        rows=len(rows),source_images=len(y),raw_sha256=meta['raw_sha256'],new_GPU_forwards=0,
        source_sha256=digest(__file__),MD_and_TeX_same_rows=True))


@torch.no_grad()
def cross_task(protocol, target, attack=False):
    """Validate a frozen structural rule on another expert; never retune it."""
    from aga_coordinate_audit import residual_permutation
    from author_protocol_audit import logical_parameter_distance
    from clip_benchmark import author_recover, evaluate
    from verify_covariance_noise import official_labels
    import contextlib
    import io
    cfg=json.loads(Path(protocol).read_text());assert target in cfg['targets']
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    anchor,expert,head,pins=checked_models(target);out=Path(cfg['output'])/target
    if attack:
        check=json.loads((out/'independent_validation.json').read_text())
        assert check['passed'] or (cfg.get('reference_utility_observation') and check['complete_native_observation'])
        assert not (out/'AGA_edits.pt').exists()
        freeze=json.loads((out/'before_images.json').read_text());assert digest(out/'release.pt')==freeze['release_sha256']
        expert.load_state_dict(expert.state_dict()|torch.load(out/'release.pt',map_location='cpu'),strict=True)
        author,sha=load_author(Path(cfg['author_path']),True);assert sha==cfg['AGA_text_sha256']
        log=io.StringIO();start=time.monotonic()
        with contextlib.redirect_stdout(log):recovered=author_recover(expert,anchor,author)
        text=log.getvalue();(out/'AGA.log').write_text(text)
        assert not any(x in text for x in ['Skipping','Warning:','Error','[Error]'])
        assert all(f'Attacking Layer {i}...' in text for i in range(12))
        assert all(f'Block {i}: MLP alignment applied.' in text for i in [0,4,8])
        assert '>>> MLP-only attack completed.' in text
        ref=cfg['AGA_references'][target];assert digest(ref['path'])==ref['sha256']
        before=torch.load(ref['path'],map_location='cpu')
        edits={k:v for k,v in recovered.state_dict().items() if not torch.equal(v,before[k])}
        torch.save(edits,out/'AGA_edits.pt')
        save_json(out/'AGA.json',dict(status='completed',AGA_edits_sha256=digest(out/'AGA_edits.pt'),
            release_sha256=freeze['release_sha256'],seconds=time.monotonic()-start,author_AGA_calls=1,
            source_sha256=digest(__file__),no_optimizer_or_result_selection=True))
        return
    out.mkdir(parents=True,exist_ok=False);save_json(out/'protocol.json',cfg)
    if cfg.get('prebuilt_releases'):
        ref=cfg['prebuilt_releases'][target];assert digest(ref['path'])==ref['sha256']
        edits=torch.load(ref['path'],map_location='cpu');(out/'release.pt').symlink_to(Path(ref['path']).resolve())
    else:
        edits=residual_permutation(expert.state_dict(),torch.arange(767,-1,-1));torch.save(edits,out/'release.pt')
    distance=logical_parameter_distance(edits,expert.state_dict())
    if cfg['parameter_budget'] is not None:assert distance<=cfg['parameter_budget']
    save_json(out/'before_images.json',dict(target=target,release_sha256=digest(out/'release.pt'),
        checkpoint_sha256=pins,parameter_distance=distance,protocol_sha256=digest(protocol),
        source_sha256=digest(__file__),construction_images=0,optimizer_steps=0,fixed_reverse768=not bool(cfg.get('prebuilt_releases')),
        baseline_reference=cfg.get('reference_family'),reference_utility_observation=cfg.get('reference_utility_observation',False)))
    released=copy.deepcopy(expert);released.load_state_dict(expert.state_dict()|edits,strict=True)
    train=dataset(target,'data',expert.val_preprocess,True)
    order=np.random.default_rng(cfg['seed']).permutation(len(train)).tolist();n=cfg['check_n']
    assert len(train)>=2*n
    raw={};rows=[];start=time.monotonic()
    for stage,ids in [('source',order[:n]),('confirmation',order[n:2*n]),('full',None)]:
        data=train if stage!='full' else dataset(target,'data',expert.val_preprocess)
        ids=list(range(len(data))) if ids is None else ids;scores=[]
        for model in [expert,released]:
            sink=[];evaluate(model,head,make_loader(data,ids),'cuda',logit_sink=sink.append)
            scores.append(torch.cat(sink).double())
        before,after=scores;assert torch.isfinite(before).all() and torch.isfinite(after).all()
        y=official_labels(target,stage!='full')[ids]
        winner=before.argmax(1);gap=before.max(1).values[:,None]-before
        mask=~torch.nn.functional.one_hot(winner,before.shape[1]).bool();assert (gap[mask]>0).all()
        delta=after-before;ratio=((delta.gather(1,winner[:,None])-delta)[mask]/gap[mask]).abs()
        row=dict(stage=stage,n=len(ids),original_correct=int(winner.eq(torch.as_tensor(y)).sum()),
            released_correct=int(after.argmax(1).eq(torch.as_tensor(y)).sum()),changed=int(winner.ne(after.argmax(1)).sum()),R=float(ratio.max()))
        row['passed']=row['changed']==0 and row['R']<=cfg['task_budget'];rows.append(row)
        raw[stage]=dict(before=before,after=after,indices=ids,labels=torch.as_tensor(y))
        torch.save(raw,out/'native_scores.pt');save_json(out/'native.json',dict(rows=rows,passed=all(r['passed'] for r in rows),
            raw_sha256=digest(out/'native_scores.pt'),image_forwards=sum(2*r['n'] for r in rows),seconds=time.monotonic()-start))
        print('COORDINATE_NATIVE',target,json.dumps(row),flush=True)
        if not row['passed'] and not cfg.get('reference_utility_observation'):break


def verify_cross_task(folder):
    """Independent NumPy coordinate rule and native gap/count checks."""
    from verify_covariance_noise import official_labels
    out=Path(folder);cfg=json.loads((out/'protocol.json').read_text());target=out.name
    freeze=json.loads((out/'before_images.json').read_text());meta=json.loads((out/'native.json').read_text())
    assert digest(out/'release.pt')==freeze['release_sha256'] and digest(out/'native_scores.pt')==meta['raw_sha256']
    _,expert,_,pins=checked_models(target);assert pins==freeze['checkpoint_sha256']
    edits=torch.load(out/'release.pt',map_location='cpu');checked=0
    if cfg.get('reference_family')=='PaRaMS':
        from params_reference import verify_edits
        algebra=verify_edits(expert,edits);checked=algebra['all_state_tensors_checked']
    else:
        for k,value in expert.state_dict().items():
            name=k.removeprefix('model.visual.');a=value.numpy()
            fixed=not k.startswith('model.visual.') or name.endswith(('attn.in_proj_bias','mlp.c_fc.bias'))
            axis=1 if name=='positional_embedding' or name.endswith(('attn.in_proj_weight','mlp.c_fc.weight')) else 0
            expected=a if fixed else np.flip(a,axis=axis)
            np.testing.assert_array_equal(edits.get(k,value),expected);checked+=1
    raw=torch.load(out/'native_scores.pt',map_location='cpu');seen=set()
    for row in meta['rows']:
        r=raw[row['stage']];a,b=r['before'].numpy(),r['after'].numpy();assert np.isfinite(a).all() and np.isfinite(b).all()
        ids=r['indices'];y=official_labels(target,row['stage']!='full')[ids];np.testing.assert_array_equal(y,r['labels'])
        if row['stage']!='full':assert not seen.intersection(ids);seen.update(ids)
        ix=np.arange(len(y));winner=a.argmax(1);mask=np.ones_like(a,dtype=bool);mask[ix,winner]=False
        gap=a[ix,winner,None]-a;assert (gap[mask]>0).all();e=b-a;R=float(abs((e[ix,winner,None]-e)[mask]/gap[mask]).max())
        assert abs(R-row['R'])<1e-10 and int((winner!=b.argmax(1)).sum())==row['changed']
        assert int((winner==y).sum())==row['original_correct'] and int((b.argmax(1)==y).sum())==row['released_correct']
    save_json(out/'independent_validation.json',dict(status='passed_NumPy_checks',passed=meta['passed'] and len(meta['rows'])==3,
        native_raw_sha256=meta['raw_sha256'],release_sha256=freeze['release_sha256'],all_state_tensors_checked=checked,
        complete_native_observation=len(meta['rows'])==3,reference_family=cfg.get('reference_family'),
        source_confirmation_disjoint=True,historically_exposed_data_not_claimed_blind=True,source_sha256=digest(__file__)))


@torch.no_grad()
def run(protocol, confirm=False):
    start = time.monotonic(); cfg = json.loads(Path(protocol).read_text()); out = Path(cfg['output'])
    for path, pin in cfg['source_sha256'].items(): assert digest(path) == pin, path
    assert '3090' in torch.cuda.get_device_name()
    anchor, expert, head, pins = checked_models('MNIST')
    if not confirm:
        if cfg.get('resume_frozen_setup'):
            assert out.exists() and not any((out/f).exists() for f in ['source_split.json','source.pt','results.json'])
            freeze=json.loads((out/'before_images.json').read_text())
            for name,key in [('frozen_release.pt','release_sha256'),('point.pt','point_sha256')]:
                assert digest(out/name)==freeze[key]==cfg['resume_frozen_setup'][key]
            assert pins==freeze['checkpoint_sha256'];geometry=freeze['geometry']
            save_json(out/'setup_resume.json',dict(original_before_images=freeze,new_protocol_sha256=digest(protocol),images_before_failure=0))
            save_json(out/'protocol.json',cfg)
        else:
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
        save_json(out/'confirmation_protocol.json',cfg)
        name, seed = 'confirmation', cfg['confirmation_seed']
    assert not (out/(name+'.pt')).exists(), 'No response-driven repeat.'
    freeze = json.loads((out/'before_images.json').read_text()); geometry = freeze['geometry']
    for file, field in [('frozen_release.pt','release_sha256'),('point.pt','point_sha256')]: assert digest(out/file) == freeze[field]
    edits = torch.load(out/'frozen_release.pt',map_location='cpu'); point = torch.load(out/'point.pt',map_location='cpu')
    release = copy.deepcopy(expert)
    for k, value in edits.items(): release.state_dict()[k].copy_(value)
    train = dataset('MNIST','data',expert.val_preprocess,True); excluded = set(cfg['extra_excluded_indices'])
    for path in cfg['extra_index_exclusions']:
        record=json.loads(Path(path).read_text());assert 'indices' in record or list(record)==['fit']
        excluded.update(record['indices'] if 'indices' in record else record['fit'])
    split = unused(train,excluded,cfg|dict(confirmation_n=cfg['check_n'],confirmation_seed=seed))
    save_json(out/(name+'_split.json'),split)
    save_json(out/('training_indices.json' if not confirm else 'confirmation_split.json'),dict(check=split['indices']) if not confirm else split)
    readout=point.get('site')=='READOUT'
    if readout:assert head.normalize and torch.count_nonzero(head.bias)==0
    before, x = observe(expert,head,train,split['indices'],'readout' if readout else True); after, _ = observe(release,head,train,split['indices'])
    y = before.argmax(1); take = torch.stack([(y==k).nonzero().flatten()[0] for k in range(10)])
    v = point['vh'].float(); rms = ((x if readout else x[:,0])@v.T).square().mean(0).sqrt()
    shifts = (cfg['finite_step']*torch.tensor([-1.,1.])[:,None,None]*rms[None,:,None]*v[None]).flatten(0,1)
    probes = x[take].repeat_interleave(len(shifts),0)
    if readout:probes+=shifts.repeat(len(take),1)
    else:probes[:,0] += shifts.repeat(len(take),1)
    probe_release=probes[...,point['order']] if point.get('site')=='RESIDUAL' else probes
    if point.get('site')=='MEAN':
        axis=point['axis'].float();probe_release=probes-2*(probes@axis)[...,None]*axis
    pb, pa = probe_scores(expert,head,probes,readout), probe_scores(release,head,probe_release,readout)
    B_probe_n=0
    if point.get('site') in ['STRUCTURAL_UNIT','RESIDUAL','MEAN']:
        for s in [before,pb]:
            gap=s.max(1).values[:,None]-s;mask=~torch.nn.functional.one_hot(s.argmax(1),10).bool()
            assert (gap[mask]>1e-12).all(),'No active denominator floor.'
    if point.get('site')=='STRUCTURAL_UNIT':
        offset=point['B_shifts'].repeat(len(take),1);binputs=x[take].repeat_interleave(len(point['B_shifts']),0)
        bb=structural_B_probes(expert,head,binputs,offset);ba=structural_B_probes(release,head,binputs,offset)
        B_probe_n=len(bb);pb=torch.cat([pb,bb]);pa=torch.cat([pa,ba])
        for s in [before,pb]:
            gaps=s.max(1).values[:,None]-s;mask=~torch.nn.functional.one_hot(s.argmax(1),10).bool()
            assert (gaps[mask]>1e-12).all(), 'No active denominator floor in this protocol.'
    control = float((probe_scores(expert,head,x,readout)-before).abs().max()); assert control <= cfg['score_tolerance']
    natural, finite = metric(before,after,len(before)),metric(pb,pa,len(pb))
    feasible = all(m['worst_task_drift']<=cfg['task_budget'] and m['changed']==0 for m in [natural,finite])
    labels = torch.tensor([int(train[i][1]) for i in split['indices']])
    branches={}
    if readout:
        for branch_name,keep in [('bias_only','model.visual.ln_post.bias'),('projection_only','model.visual.proj')]:
            for key,value in point['original'].items():release.state_dict()[key].copy_(edits[key] if key==keep else value)
            branches[branch_name]=probe_scores(release,head,x,True)
        for key,value in edits.items():release.state_dict()[key].copy_(value)
    torch.save(dict(before=before,after=after,probe_before=pb,probe_after=pa,inputs=x,probe_inputs=probes,branches=branches,
        head_weight=head.weight.detach().cpu().double() if readout else None,
        indices=split['indices'],labels=labels),out/(name+'.pt'))
    report = dict(status='passed' if feasible else 'closed_task_budget_failure',source_feasible=feasible,
        branch_response={k:metric(before,v,len(before)) for k,v in branches.items()},
        geometry=geometry,natural=natural,finite_probes=finite,full_block_score_control=control,
        structural_B_probe_n=B_probe_n,
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
    edits=torch.load(out/'frozen_release.pt',map_location='cpu');head_control=point.get('site')=='HEAD'
    readout=point.get('site')=='READOUT'
    residual_control=point.get('site')=='RESIDUAL'
    mean_control=point.get('site')=='MEAN'
    if mean_control:
        _,expert,_,_=checked_models('MNIST');s={k:v.double().numpy() for k,v in expert.state_dict().items()}
        point['original']={k:expert.state_dict()[k] for k in point['keys']}
        u=point['axis'].numpy();q=np.eye(768)-2*np.outer(u,u);visual='model.visual.';expected={}
        for key in ['conv1.weight','class_embedding','positional_embedding']:
            w=s[visual+key]
            expected[visual+key]=w-2*np.outer(w@u,u) if key=='positional_embedding' else (w.reshape(768,-1)-2*np.outer(u,u@w.reshape(768,-1))).reshape(w.shape)
        expected[visual+'ln_pre.bias']=q@s[visual+'ln_pre.bias']
        for i in range(12):
            prefix=f'{visual}transformer.resblocks.{i}.'
            for ln,read,write in [('ln_1.','attn.in_proj_','attn.out_proj.'),('ln_2.','mlp.c_fc.','mlp.c_proj.')]:
                gain,beta=[s[prefix+ln+k] for k in ['weight','bias']];t=point['affine_shares'][prefix+ln]
                w,b=[s[prefix+read+k] for k in ['weight','bias']];dw=-2*np.outer(w@(gain*u),u/gain)
                expected[prefix+read+'weight']=w+dw;expected[prefix+read+'bias']=b-(1-t)*dw@beta
                expected[prefix+ln+'bias']=beta-2*t*gain*u*((u/gain)@beta)
                w,b=[s[prefix+write+k] for k in ['weight','bias']]
                expected[prefix+write+'weight']=w-2*np.outer(u,u@w);expected[prefix+write+'bias']=b-2*u*(u@b)
        gain,beta=[s[visual+'ln_post.'+k] for k in ['weight','bias']];w=s[visual+'proj']
        expected[visual+'proj']=w-2*np.outer(u/gain,(gain*u)@w)
        expected[visual+'ln_post.bias']=beta-2*gain*u*((u/gain)@beta)
        identity=0.
        for key,values in expected.items():
            actual=edits.get(key,expert.state_dict()[key]).numpy();rounded=values.astype(np.float32)
            identity=max(identity,float(abs(actual-rounded).max()))
            assert np.all(abs(actual-rounded)<=np.abs(np.spacing(rounded))+1e-12),key
        assert set(edits)<=set(expected)
    elif residual_control:
        _,expert,_,_=checked_models('MNIST');point['original']={k:expert.state_dict()[k] for k in point['keys']}
        ix=point['order'].numpy();assert np.array_equal(ix,np.arange(767,-1,-1))
        for k,v in point['original'].items():
            parameter_name=k.removeprefix('model.visual.');axis=1 if parameter_name=='positional_embedding' or parameter_name.endswith(('attn.in_proj_weight','mlp.c_fc.weight')) else 0
            np.testing.assert_array_equal(edits[k].numpy(),np.take(v.numpy(),ix,axis=axis))
        identity=0.
    elif readout:
        g=point['gamma'].numpy();beta=point['beta'].numpy();p=point['projection'].numpy()
        bp=edits['model.visual.ln_post.bias'].double().numpy();pp=edits['model.visual.proj'].double().numpy()
        linear=g[:,None]*(pp-p);linear-=linear.mean(0)
        identity=max(float(abs(linear).max()),float(abs(bp@pp-beta@p).max()));assert identity<1e-6
        h=raw['inputs'].double().numpy();norm=(h-h.mean(1,keepdims=True))/np.sqrt(((h-h.mean(1,keepdims=True))**2).mean(1,keepdims=True)+point['eps'])
        maximum=0.
        for label,bb,pr in [('before',beta,p),('after',bp,pp),('bias_only',bp,p),('projection_only',beta,pp)]:
            f=(norm*g+bb)@pr;s=f/np.linalg.norm(f,axis=1,keepdims=True)@raw['head_weight'].numpy().T
            actual=raw['branches'][label].numpy() if label in raw['branches'] else raw[label].numpy()
            maximum=max(maximum,float(abs(s-actual).max()))
        assert maximum<=json.loads((out/'protocol.json').read_text())['score_tolerance']
    elif head_control:
        ix=(64*point['head_order'].numpy()[:,None]+np.arange(64)[None]).flatten();full=np.concatenate([ix+768*j for j in range(3)])
        for k,v in point['original'].items():
            expected=v.numpy()[:,ix] if k.endswith('out_proj.weight') else v.numpy()[full]
            np.testing.assert_array_equal(edits[k],expected)
        identity=0.
    elif point.get('site')=='STRUCTURAL_UNIT':
        keys=point['keys'];j=point['unit'];g=point['gamma'].numpy();beta=point['beta'].numpy()
        n=1/g;n/=np.linalg.norm(n);c=beta@n;eta=point['eta'];A=point['original'][keys[0]].numpy();B=point['original'][keys[2]].numpy()
        expected={k:v.numpy().astype(np.float32) for k,v in point['original'].items()}
        expected[keys[0]][j]=(eta*np.linalg.norm(A)*n).astype(np.float32)
        expected[keys[1]][j]=np.float32(-eta*np.linalg.norm(A)*c)
        expected[keys[2]][:,j]=(B[:,j]+eta*np.linalg.norm(B)*point['u'].numpy()).astype(np.float32)
        for key in keys:np.testing.assert_allclose(edits[key],expected[key],atol=1e-7,rtol=0)
        linear=edits[keys[0]][j].double().numpy()*g;linear-=linear.mean()
        identity=max(float(abs(linear).max()),abs(float(edits[keys[0]][j].double().numpy()@beta+edits[keys[1]][j])))
        assert identity<1e-6
    else:
        g=point['gamma'].numpy();beta=point['beta'].numpy();keys=point.get('keys',KEYS)
        dw=(edits[keys[0]].double()-point['original'][keys[0]].double()).numpy();db=(edits[keys[1]].double()-point['original'][keys[1]].double()).numpy()
        linear=dw*g;centered=linear-linear.mean(1,keepdims=True)
        identity=max(float(abs(centered).max()),float(abs(dw@beta+db).max()));assert identity<1e-6
    cost=0.
    for k,v in point['original'].items():
        a,b=edits[k].double().numpy(),v.double().numpy()
        pieces=zip(np.split(a,3),np.split(b,3)) if (head_control or residual_control or mean_control) and k.endswith(('in_proj_weight','in_proj_bias')) else [(a,b)]
        for changed,before in pieces:
            if np.linalg.norm(before)>0:cost+=float(np.linalg.norm(changed-before)**2/np.linalg.norm(before)**2)
            else:np.testing.assert_array_equal(changed,before)
    distance=cost**.5
    assert abs(distance-report['geometry']['parameter_distance'])<1e-10
    for (ka,kb), field in [(('before','after'),'natural'),(('probe_before','probe_after'),'finite_probes')]:
        a,b=raw[ka].numpy(),raw[kb].numpy(); y=a.argmax(1); ix=np.arange(len(a)); gap=a[ix,y,None]-a; gap[ix,y]=1
        structural=point.get('site') in ['STRUCTURAL_UNIT','RESIDUAL','MEAN']
        if structural:assert (gap>0).all()
        delta=b-a; r=abs((delta[ix,y,None]-delta)/(gap if structural else np.maximum(gap,1e-12)));r[ix,y]=0
        assert abs(float(r.max())-report[field]['worst_task_drift'])<1e-9
        assert int((a.argmax(1)!=b.argmax(1)).sum())==report[field]['changed']
    cfg=json.loads((out/'protocol.json').read_text()); split=json.loads((out/(name+'_split.json')).read_text()); initial=set(cfg['extra_excluded_indices'])
    for path in cfg['extra_index_exclusions']:
        record=json.loads(Path(path).read_text());assert 'indices' in record or list(record)==['fit']
        initial.update(record['indices'] if 'indices' in record else record['fit'])
    excluded,exposed=exclusions(split,initial); assert len(excluded)==split['excluded_count']
    draw=np.random.default_rng(split['seed']).choice(sorted(set(range(60000))-excluded),cfg['check_n'],replace=False).tolist()
    assert draw==raw['indices']==split['indices'] and not set(draw)&exposed
    np.testing.assert_array_equal(official_labels('MNIST',True)[draw],raw['labels'])
    save_json(out/('confirmation_independent_validation.json' if confirm else 'independent_validation.json'),
        dict(status='passed',raw_sha256=report['raw_sha256'],parameter_distance=distance,NumPy_affine_identity_error=None if head_control else identity,
            NumPy_complete_readout_and_nonzero_branches_error=maximum if readout else None,
            NumPy_whole_head_permutation_exact=head_control,
            NumPy_native_residual_permutation_exact=residual_control,covariant_task_probe_inputs=residual_control or mean_control,
            NumPy_mean_compensation_rounded_parameters_checked=mean_control,
            natural_and_probe_margin_recount=True,source_indices_and_labels_checked=True,source_sha256=digest(__file__)))


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol');ap.add_argument('--confirm',action='store_true');ap.add_argument('--verify');ap.add_argument('--components');ap.add_argument('--component-protocol');ap.add_argument('--verify-components')
    ap.add_argument('--cross-task');ap.add_argument('--target');ap.add_argument('--attack',action='store_true');ap.add_argument('--verify-cross-task')
    args=ap.parse_args();initialize()
    if args.cross_task:cross_task(args.cross_task,args.target,args.attack);raise SystemExit
    if args.verify_cross_task:verify_cross_task(args.verify_cross_task);raise SystemExit
    if args.verify_components:verify_components(args.verify_components);raise SystemExit
    if args.components:mean_components(args.components,args.component_protocol);raise SystemExit
    verify(args.verify,args.confirm) if args.verify else run(args.protocol,args.confirm)
