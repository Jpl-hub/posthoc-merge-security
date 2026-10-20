"""Extract exact row permutations from existing AGA artifacts; no attack rerun."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import digest
from mechanisms import load_author
from subspace_probe import save_json
from task_probe_distance import checked_models


def residual_permutation(state, order):
    """Known global residual-coordinate symmetry; keep QKV head outputs fixed.

    Native LayerNorm gains/biases move with their coordinates. This deliberately
    differs from rotating a native learned-gain LayerNorm without rewriting it.
    """
    edits={}
    for k,v in state.items():
        if not k.startswith('model.visual.'):continue
        name=k.removeprefix('model.visual.')
        if name in ['conv1.weight','class_embedding','proj']:axis=0
        elif name=='positional_embedding' or name.endswith(('attn.in_proj_weight','mlp.c_fc.weight')):axis=1
        elif name.endswith(('attn.out_proj.weight','attn.out_proj.bias','mlp.c_proj.weight','mlp.c_proj.bias')):axis=0
        elif any(x in name for x in ['ln_pre.','ln_post.','.ln_1.','.ln_2.']):axis=0
        else:
            assert name.endswith(('attn.in_proj_bias','mlp.c_fc.bias')),name
            continue
        assert v.shape[axis]==len(order),(k,v.shape)
        changed=v.index_select(axis,order).contiguous()
        if not torch.equal(changed,v):edits[k]=changed
    return edits


@torch.no_grad()
def residual_recovery(output):
    """Data-free public patch-row matching; original expert only validates it."""
    from scipy.optimize import linear_sum_assignment
    out=Path(output);start=time.monotonic();cfg=json.loads((out/'protocol.json').read_text())
    meta=json.loads((out/'execution.json').read_text());assert meta['completed']
    assert json.loads((out/'independent_validation.json').read_text())['status']=='passed'
    anchor,expert,_,pins=checked_models('MNIST');original,public=expert.state_dict(),anchor.state_dict()
    assert digest(cfg['release']['path'])==cfg['release']['sha256']
    published=original|torch.load(cfg['release']['path'],map_location='cpu');key='model.visual.conv1.weight';records=[]
    def recover(state):
        x,y=[s[key].double().flatten(1) for s in [state,public]]
        cost=(x/x.norm(dim=1,keepdim=True))@(y/y.norm(dim=1,keepdim=True)).T
        a,b=x.numpy(),y.numpy();ind=(a/np.linalg.norm(a,axis=1,keepdims=True))@(b/np.linalg.norm(b,axis=1,keepdims=True)).T
        error=float(abs(cost.numpy()-ind).max());assert error<1e-10
        rows,cols=linear_sum_assignment(-ind);order=np.empty(len(cols),dtype=np.int64);order[cols]=rows
        other=ind.copy();other[rows,cols]=-np.inf;margin=float((ind[rows,cols]-other.max(1)).min())
        records.append(dict(public_coordinate_to_published_row=order.tolist(),strict_row_margin=margin,numpy_cost_error=error))
        return state|residual_permutation(state,torch.tensor(order))
    recovered,control=recover(published),recover(original)
    diffs=[[k for k,v in s.items() if not torch.equal(v,original[k])] for s in [recovered,control]]
    exact=not any(diffs);baseline=[r for r in json.loads((out/'results.json').read_text()) if r['preset']=='original']
    result=dict(status='exact_original_state_recovered' if exact else 'nonidentical_recovery_no_prediction_reuse',tensor_differences=diffs,records=records,
        attack_inputs=['published_state','public_base'],secret_order_used=False,images_used=0,original_expert_used_only_for_validation=True,
        recovered_tensor_count=len(original),release_sha256=cfg['release']['sha256'],checkpoint_sha256=pins,
        matched_control_included=True,reused_original_rows=baseline if exact else [],new_AGA_calls=0,new_image_forwards=0,
        reuse_reason='Only exact whole-state equality licenses reuse of independently replayed original TA/AGA scores.',
        source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,novelty_claim=False)
    save_json(out/'residual_recovery.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['records','reused_original_rows']}),flush=True)


def permutation(before, after):
    fingerprint=lambda row:hashlib.sha256(row.contiguous().numpy().tobytes()).digest()
    index={fingerprint(row):i for i,row in enumerate(before)}
    assert len(index)==len(before)
    p=torch.tensor([index[fingerprint(row)] for row in after])
    assert len(set(p.tolist()))==len(before) and torch.equal(before[p],after)
    return p


def head_costs(state,reference,layer):
    """Native CLIP layout of GLMC's two published head invariants."""
    prefix=f'model.visual.transformer.resblocks.{layer}.attn.'
    def features(s):
        w,b,o=[s[prefix+k].double() for k in ['in_proj_weight','in_proj_bias','out_proj.weight']]
        q,k,v=torch.cat([w,b[:,None]],1).reshape(3,12,64,769)
        return q.transpose(1,2)@k,o.reshape(768,12,64).permute(1,0,2)@v
    a,b=features(state),features(reference)
    cost=torch.stack([torch.stack([sum((x[i]-y[j]).norm() for x,y in zip(a,b)) for j in range(12)]) for i in range(12)])
    independent=np.array([[sum(np.linalg.norm(x[i].numpy()-y[j].numpy()) for x,y in zip(a,b)) for j in range(12)] for i in range(12)])
    error=float(abs(cost.numpy()-independent).max());assert error<1e-8
    return independent,error


@torch.no_grad()
def head_identity_recovery(output):
    """Recover head identity using only actual published/public-anchor tensors."""
    from scipy.optimize import linear_sum_assignment
    out=Path(output);assert not (out/'head_recovery.json').exists();start=time.monotonic();torch.set_num_threads(4)
    cfg=json.loads((out/'protocol.json').read_text());meta=json.loads((out/'execution.json').read_text())
    assert meta['completed'] and json.loads((out/'independent_validation.json').read_text())['status']=='passed'
    anchor,expert,_,pins=checked_models('MNIST');original=expert.state_dict();public=anchor.state_dict()
    assert digest(cfg['release']['path'])==cfg['release']['sha256'];published=original|torch.load(cfg['release']['path'],map_location='cpu')
    def recover(state,reference):
        changed={};rows=[];maximum=0.
        for layer in range(12):
            prefix=f'model.visual.transformer.resblocks.{layer}.attn.';wk,bk,ok=[prefix+n for n in ['in_proj_weight','in_proj_bias','out_proj.weight']]
            independent,error=head_costs(state,reference,layer);maximum=max(maximum,error)
            row,col=linear_sum_assignment(independent);order=torch.empty(12,dtype=torch.long);order[torch.tensor(col)]=torch.tensor(row)
            other=independent.copy();other[row,col]=np.inf;margin=float((other.min(1)-independent[row,col]).min());assert margin>0
            ix=(64*order[:,None]+torch.arange(64)[None]).flatten();full=torch.cat([ix+768*j for j in range(3)])
            changed.update({wk:state[wk][full],bk:state[bk][full],ok:state[ok][:,ix]})
            rows.append(dict(layer=layer,output_head_to_published_head=order.tolist(),strict_row_minimum_margin=margin))
        return state|changed,rows,maximum
    recovered,protected_rows,error=recover(published,public);control,control_rows,control_error=recover(original,public)
    protected_differences=[k for k,v in recovered.items() if not torch.equal(v,original[k])]
    control_differences=[k for k,v in control.items() if not torch.equal(v,original[k])]
    assert not protected_differences and not control_differences
    for pin in cfg['AGA_references'].values():assert digest(pin['path'])==pin['sha256']
    assert digest(cfg['reference_predictions']['path'])==cfg['reference_predictions']['sha256']
    rows=[r for r in json.loads((out/'results.json').read_text()) if r['preset']=='original']
    save_json(out/'head_recovery.json',dict(status='actual_CLIP_head_recovery_exact_all_tensors',protected_rows=protected_rows,matched_control_rows=control_rows,
        tensor_count=len(original),protected_tensor_differences=protected_differences,matched_control_tensor_differences=control_differences,
        independent_NumPy_cost_error=max(error,control_error),attack_inputs=['actual_published_state','public_anchor_state'],
        secret_permutation_used_by_recovery=False,original_expert_used_only_for_validation=True,
        release_sha256=cfg['release']['sha256'],checkpoint_sha256=pins,AGA_reference_sha256={k:v['sha256'] for k,v in cfg['AGA_references'].items()},
        reference_predictions_sha256=cfg['reference_predictions']['sha256'],reused_original_rows=rows,
        reuse_reason='Recovered whole state equals original before unchanged author AGA. Cached original pipelines were independently replayed in this batch; no new full adaptive prediction run is claimed.',
        author_AGA_new_calls=0,new_model_forwards=0,gradients=0,source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        novelty_claim=False,scope='Known head matching then original AGA, paper-kernel CLIP native adaptation, not full official GLMC. This frozen control violates original parameter0.5.'))


@torch.no_grad()
def head_identity_geometry(output):
    """One known whole-head permutation control; not a new defense mechanism."""
    from scipy.optimize import linear_sum_assignment
    from author_protocol_audit import logical_parameter_distance
    out=Path(output);assert not out.exists();out.mkdir(parents=True);torch.set_num_threads(4);start=time.monotonic()
    anchor,expert,_,pins=checked_models('MNIST');s,b=expert.state_dict(),anchor.state_dict()
    order=torch.tensor([1,0]+list(range(2,12)));ix=(64*order[:,None]+torch.arange(64)[None]).flatten()
    edits={};original={};records=[];restored={};maximum=0.
    for layer in range(12):
        p=f'model.visual.transformer.resblocks.{layer}.attn.';wk,bk,ok=[p+n for n in ['in_proj_weight','in_proj_bias','out_proj.weight']]
        full=torch.cat([ix+768*j for j in range(3)])
        for key,v in [(wk,s[wk][full]),(bk,s[bk][full]),(ok,s[ok][:,ix])]:original[key]=s[key];edits[key]=v.clone()
        cn,error=head_costs(s,b,layer);maximum=max(maximum,error)
        base_row,base_col=linear_sum_assignment(cn);row,col=linear_sum_assignment(cn[order.numpy()]);restore=torch.empty(12,dtype=torch.long);restore[torch.tensor(col)]=torch.tensor(row)
        chosen=cn[order.numpy()][row,col];other=np.array(cn[order.numpy()],copy=True);other[row,col]=np.inf
        margin=float((other.min(1)-chosen).min());assert margin>0 and np.array_equal(base_col,np.arange(12))
        reindex=(64*restore[:,None]+torch.arange(64)[None]).flatten();allrows=torch.cat([reindex+768*j for j in range(3)])
        for key,v in [(wk,edits[wk][allrows]),(bk,edits[bk][allrows]),(ok,edits[ok][:,reindex])]:assert torch.equal(v,s[key]);restored[key]=v
        records.append(dict(layer=layer,original_matching=base_col.tolist(),release_recovery_order=restore.tolist(),minimum_strict_row_cost_margin=margin))
    distance=logical_parameter_distance(edits,s);nd=0.
    for key,v in edits.items():
        a,before=v.double().numpy(),s[key].double().numpy();pieces=zip(np.split(a,3),np.split(before,3)) if key.endswith(('in_proj_weight','in_proj_bias')) else [(a,before)]
        for x,y in pieces:
            if np.linalg.norm(y)>0:nd+=float(np.linalg.norm(x-y)**2/np.linalg.norm(y)**2)
            else:np.testing.assert_array_equal(x,y)
    assert abs(nd**.5-distance)<1e-10
    torch.save(dict(original=original,edits=edits,head_order=order,rows=records),out/'head_identity.pt')
    save_json(out/'geometry.json',dict(status='known_symmetry_control_weight_checks_passed' if distance<=6 else 'closed_declared_parameter_budget_failed',
        logical_parameter_distance=distance,original_parameter_budget=.5,original_parameter_budget_passed=distance<=.5,
        separately_predeclared_parameter_budget=6.,task_budget=.5,NumPy_distance_error=abs(nd**.5-distance),NumPy_head_cost_error=maximum,
        all_tensors_exactly_restored_by_public_anchor_head_matching=True,matched_unprotected_head_assignment_identity=True,
        head_matching='GeneralizedLMC5e343c0 Qaug.T@Kaug / O@Vaug Frobenius costs, SciPy assignment native adaptation; not full official GLMC.',
        rows=records,raw_sha256=digest(out/'head_identity.pt'),checkpoint_sha256=pins,source_sha256=digest(__file__),
        source_images=0,gradients=0,new_released_models=0,novelty_claim=False,wall_seconds=time.monotonic()-start))


@torch.no_grad()
def affine_matching_geometry(output):
    """One weight-only, fixed-budget certificate for a changed MLP assignment."""
    out=Path(output);assert not out.exists();out.mkdir(parents=True);start=time.monotonic()
    anchor,expert,_,pins=checked_models('MNIST');block=expert.model.visual.transformer.resblocks[11]
    prefix='model.visual.transformer.resblocks.11.mlp.'
    A,a=block.mlp.c_fc.weight.cpu(),block.mlp.c_fc.bias.cpu()
    base=anchor.state_dict()[prefix+'c_fc.weight'].cpu()
    gamma,beta=block.ln_2.weight.double().cpu(),block.ln_2.bias.double().cpu()
    assert float(gamma.abs().min())>1e-8
    normal=gamma.reciprocal();normal/=normal.norm();constant=normal@beta
    coefficient=float(A.double().norm().reciprocal().square()+constant.square()/a.double().norm().square())
    radius=.5*(1-1e-6)/coefficient**.5;shift=radius/2**.5
    ref=json.loads(Path('configs/ln_affine_eval.json').read_text())['AGA_references']['MNIST']
    assert digest(ref['path'])==ref['sha256']
    historical=torch.load(ref['path'],map_location='cpu')[prefix+'c_fc.weight']
    p0=permutation(A,historical);order=torch.argsort(p0)
    author,pin=load_author(Path('external/secure-merge-attack/utils_attack_B32.py'))
    assert pin=='4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778'
    original_author=author['solve_permutation'](A,base).cpu()
    assert torch.equal(original_author,p0),'Fixed-author baseline must match the frozen original AGA.'
    x,y,n=[v.double().cuda() for v in [A,base[order],normal]]
    y=y/(y.norm(dim=1,keepdim=True)+1e-8);dot=x@y.T
    length=x.norm(dim=1);along=x@n;target_along=y@n
    plus=(length.square()+2*shift*along+shift**2).sqrt()+1e-8
    minus=(length.square()-2*shift*along+shift**2).sqrt()+1e-8
    differences=dot-dot.diag()[:,None];tilt=shift*(target_along[None,:]-target_along[:,None])
    gains=(differences+tilt)/plus[:,None]+((differences-tilt)/minus[:,None]).T
    gains.fill_diagonal_(-torch.inf);flat=int(gains.argmax());i,k=divmod(flat,len(A));gain=float(gains[i,k])
    # NumPy independently repeats the global fixed-rule pair selection.
    xn,yn,nn=A.double().numpy(),base[order].double().numpy(),normal.numpy()
    yn=yn/(np.linalg.norm(yn,axis=1,keepdims=True)+1e-8);dn=xn@yn.T;an=xn@nn;bn=yn@nn
    norms=np.linalg.norm(xn,axis=1);pp=np.sqrt(norms**2+2*shift*an+shift**2)+1e-8
    mm=np.sqrt(norms**2-2*shift*an+shift**2)+1e-8;dd=dn-np.diag(dn)[:,None]
    tt=shift*(bn[None,:]-bn[:,None]);gn=(dd+tt)/pp[:,None]+((dd-tt)/mm[:,None]).T
    np.fill_diagonal(gn,-np.inf);assert divmod(int(np.argmax(gn)),len(A))==(i,k)
    error=abs(float(gn[i,k])-gain);assert error<1e-10
    t=torch.zeros(len(A),dtype=torch.float64);t[i]=shift;t[k]=-shift
    edits={prefix+'c_fc.weight':(A.double()+t[:,None]*normal).float(),prefix+'c_fc.bias':(a.double()-constant*t).float()}
    dw=edits[prefix+'c_fc.weight'].double()-A.double();db=edits[prefix+'c_fc.bias'].double()-a.double()
    distance=float((dw.norm().square()/A.double().norm().square()+db.norm().square()/a.double().norm().square()).sqrt())
    assert distance<=.5
    linear=dw*gamma;centered=linear-linear.mean(1,keepdim=True);offset=dw@beta+db
    identity_bound=float((768**.5*centered.norm(dim=1)+offset.abs()).max())
    changed=None;materialized_gain=None;p1=None
    if gain>1e-6:
        p1=author['solve_permutation'](edits[prefix+'c_fc.weight'],base).cpu()
        changed=int((p1!=p0).sum())
        actual=edits[prefix+'c_fc.weight'].numpy();bn32=base[order].numpy()
        actual=actual/(np.linalg.norm(actual,axis=1,keepdims=True)+1e-8)
        bn32=bn32/(np.linalg.norm(bn32,axis=1,keepdims=True)+1e-8)
        materialized_gain=float(actual[i]@(bn32[k]-bn32[i])+actual[k]@(bn32[i]-bn32[k]))
    supported=gain>1e-6 and materialized_gain is not None and materialized_gain>1e-6 and changed>0
    torch.save(dict(t=t,normal=normal,constant=constant,original={prefix+'c_fc.weight':A,prefix+'c_fc.bias':a},
        edits=edits,gamma=gamma,beta=beta,original_restore=p0,changed_restore=p1),out/'counterfactual.pt')
    report=dict(status='matching_boundary_certificate_passed' if supported else 'closed_fixed_pair_rule_no_boundary_certificate',
        pair=[i,k],logical_A_and_bias_distance=distance,LN_constant=float(constant),shift=shift,
        maximum_fixed_rule_swap_gain=gain,materialized_FP32_swap_gain=materialized_gain,
        original_AGA_changed_assignment_positions=changed,independent_NumPy_gain_error=error,
        uniform_real_input_preactivation_rounding_bound=identity_bound,
        raw_sha256=digest(out/'counterfactual.pt'),source_sha256=digest(__file__),checkpoint_sha256=pins,
        author_text_sha256=pin,source_images=0,new_model_forwards=0,gradients=0,new_released_models=0,
        classification_protection_claim=False,adaptive_canonicalization_limitation_disclosed=True,
        scope='One fixed equal/opposite two-row affine-input gauge at logicalA/bias0.5. Pair chosen by weight-only finite cycle gain, not images or fusion scores. A changed assignment is not classification harm. No alternate pair allocation, scale or layer retry.',
        wall_seconds=time.monotonic()-start)
    save_json(out/'affine_matching_geometry.json',report);print(json.dumps(report),flush=True)


def audit(output):
    out=Path(output);cfg=json.loads((out/'protocol.json').read_text());execution=json.loads((out/'execution.json').read_text())
    _,expert,_,pins=checked_models('MNIST')
    ref=cfg['AGA_references']['MNIST'];assert digest(ref['path'])==ref['sha256']
    baseline=torch.load(ref['path'],map_location='cpu')
    release=Path(cfg['release']['path']) if isinstance(cfg.get('release'),dict) else out/'artifacts/release.pt'
    attack=out/'artifacts/AGA_release_edits.pt'
    if not attack.exists():attack=out/'artifacts/released_AGA_edits.pt'
    assert digest(release)==execution['release_sha256'] and digest(attack)==execution['AGA_edits_sha256']
    published=expert.state_dict()|torch.load(release,map_location='cpu')
    recovered=baseline|torch.load(attack,map_location='cpu')
    prefix='model.visual.transformer.resblocks.11.mlp.'
    p0=permutation(expert.state_dict()[prefix+'c_fc.weight'],baseline[prefix+'c_fc.weight'])
    p1=permutation(published[prefix+'c_fc.weight'],recovered[prefix+'c_fc.weight'])
    for source,dest,p in [(expert.state_dict(),baseline,p0),(published,recovered,p1)]:
        assert torch.equal(source[prefix+'c_fc.bias'][p],dest[prefix+'c_fc.bias'])
        assert torch.equal(source[prefix+'c_proj.weight'][:,p],dest[prefix+'c_proj.weight'])
    result={'status':'passed_exact_AGA_row_permutation_extraction','neurons':len(p0),'changed_permutation_positions':int((p0!=p1).sum()),
        'original_to_AGA_permutation':p0.tolist(),'published_to_AGA_permutation':p1.tolist(),
        'bias_and_B_column_consistency':'passed','checkpoint_sha256':pins,'release_sha256':digest(release),'AGA_edits_sha256':digest(attack),
        'source_sha256':digest(__file__),'new_model_forwards':0,'new_attack_calls':0,'new_release_weights':0,'test_used':False,'partner_used':False,
        'scope':'Exact algebra of the existing author MLP permutation. A changed assignment is a coordinate mismatch explanation, not a quantified causal attribution of the entire TA/AGA accuracy gap.'}
    save_json(out/'coordinate_audit.json',result)
    print(json.dumps({k:v for k,v in result.items() if not isinstance(v,list)}),flush=True)


@torch.no_grad()
def head_geometry(output):
    """Separate removable head coordinates from changed column spaces; no images."""
    start=time.monotonic();out=Path(output);assert not out.exists()
    cfg=json.loads(Path('configs/frozen_coupling_audit.json').read_text())
    anchor,expert,_,pins=checked_models('MNIST')
    def checked(path,pin):
        assert digest(path)==pin,path
        return torch.load(path,map_location='cpu')
    released=expert.state_dict()|checked(cfg['release'],cfg['release_sha256'])
    recovered=checked(cfg['original_aga'],cfg['original_aga_sha256'])
    restored=recovered|checked(cfg['published_aga_edits'],cfg['published_aga_edits_sha256'])
    source=Path('external/secure-merge-attack/utils_attack_B32.py')
    spec=json.loads(Path('configs/phase3.json').read_text())
    assert digest(source)==spec['AGA']['raw_sha256']
    author,_=load_author(source,True)
    key='model.visual.transformer.resblocks.11.attn.out_proj.weight'
    original,changed,base=[s[key] for s in [expert.state_dict(),released,anchor.state_dict()]]
    records=[];maximum=0.
    for head in range(12):
        sl=slice(head*64,(head+1)*64)
        w,d,a=[x.double() for x in [original[:,sl],changed[:,sl]-original[:,sl],base[:,sl]]]
        q=torch.linalg.qr(w,mode='reduced').Q
        tangent=q@(q.T@d);normal=d-tangent
        u=np.linalg.svd(w.numpy(),full_matrices=False)[0]
        np.testing.assert_allclose(tangent.numpy(),u@(u.T@d.numpy()),atol=1e-11,rtol=1e-10)
        def projection(x):
            z=torch.linalg.qr(x,mode='reduced').Q
            return z@(z.T@a)
        coordinate=torch.linalg.lstsq(w,w+tangent).solution
        minimum=float(torch.linalg.svdvals(coordinate).min());assert minimum>1e-8
        erased=float((projection(w+tangent)-projection(w)).abs().max());assert erased<1e-10
        ideal=projection(w+d)-projection(w)
        for state,wanted in [(original,recovered),(changed,restored)]:
            x=state[:,sl];actual=x@author['solve_linear_transformation'](x,base[:,sl])
            err=float((actual-wanted[key][:,sl]).abs().max());maximum=max(maximum,err)
            assert err<2e-5,err
        observed=restored[key][:,sl].double()-recovered[key][:,sl].double()
        records.append(dict(head=head,delta_squared=float(d.square().sum()),
            normal_squared=float(normal.square().sum()),tangent_squared=float(tangent.square().sum()),
            ideal_recovered_delta_squared=float(ideal.square().sum()),
            author_recovered_delta_squared=float(observed.square().sum()),
            removable_coordinate_minimum_singular=minimum,tangent_projection_maximum_error=erased,
            ideal_to_author_delta_maximum_error=float((ideal-observed).abs().max())))
    sums={k:sum(r[k] for r in records) for k in ['delta_squared','normal_squared','tangent_squared','ideal_recovered_delta_squared','author_recovered_delta_squared']}
    result=dict(status='source_only_head_geometry_checked_not_defense',records=records,totals=sums,
        normal_energy_fraction=sums['normal_squared']/sums['delta_squared'],
        restored_delta_norm_ratio=(sums['author_recovered_delta_squared']/sums['delta_squared'])**.5,
        original_author_CPU_to_frozen_GPU_maximum_weight_error=maximum,
        scope='Only frozen MNIST O/B, public anchor and existing target-only parameter recovery. No peer, image, test, confirmation, gradient, new release or accuracy claim.',
        hypothesis='If head-coordinate erasure dominates the old O/B compensation loss, most original O change should lie in unchanged head column spaces and disappear on recovery. Energy and finite restored operator must be reported, not treated as classification damage.',
        source_sha256=digest(__file__),author_sha256=digest(source),checkpoint_sha256=pins,
        frozen_inputs={cfg[k]:cfg[k+'_sha256'] for k in ['release','original_aga','published_aga_edits']},
        old_protocol_sha256=digest('configs/frozen_coupling_audit.json'),wall_seconds=time.monotonic()-start,
        images_loaded=0,new_release_weights=0,gradients=0,partner_used=False)
    save_json(out,result);print(json.dumps(result,indent=2),flush=True)


@torch.no_grad()
def head_response(output):
    """Measure the two head-space components in the frozen source computation."""
    from angular_distance import signed_residual
    from cross_module_distance import extended_parameters, final_hidden
    from probe_margin_distance import scores
    from verify_levelset_audits import convert, raw_scores
    from scipy.special import expit
    start=time.monotonic();out=Path(output);assert not out.exists()
    cfg=json.loads(Path('configs/frozen_coupling_audit.json').read_text())
    _,expert,head,pins=checked_models('MNIST');p=extended_parameters(expert,head)
    for field in ['point','release']:assert digest(cfg[field])==cfg[field+'_sha256']
    point=torch.load(cfg['point'],map_location='cpu');release=torch.load(cfg['release'],map_location='cpu')
    c={k:v[:512].double() for k,v in point['native'].items()};pn=convert(p)
    key='model.visual.transformer.resblocks.11.'
    d=release[key+'attn.out_proj.weight'].double()-p['O'];db=release[key+'mlp.c_proj.weight'].double()-p['B']
    tangent=torch.empty_like(d)
    for h in range(12):
        sl=slice(h*64,(h+1)*64);q=torch.linalg.qr(p['O'][:,sl],mode='reduced').Q
        tangent[:,sl]=q@(q.T@d[:,sl])
    rounding=c['h']-final_hidden(c['r'],p);before=scores(c['h'],p);winner=before.argmax(1)
    records=[];error=0.
    for name,delta in [('none',torch.zeros_like(d)),('full',d),('tangent',tangent),('normal',d-tangent)]:
        r=c['r']+c['o_input']@delta.T;rn=r.numpy();center=rn-rn.mean(1,keepdims=True)
        pre=(center/np.sqrt((center**2).mean(1,keepdims=True)+p['eps2'])*pn['gamma2']+pn['beta2'])@pn['A'].T+pn['a']
        for compensate in [False,True]:
            pp=p|{'B':p['B']+(db if compensate else 0)}
            hidden=rounding+final_hidden(r,pp);after=scores(hidden,p)
            hn=rounding.numpy()+rn+(pre*expit(1.702*pre))@pp['B'].numpy().T+pn['b']
            independent=raw_scores(hn,pn);err=float(abs(independent-after.numpy()).max());assert err<1e-8;error=max(error,err)
            changed=after.argmax(1)!=winner
            records.append(dict(O_component=name,B_changed=compensate,teacher_prediction_changes=int(changed.sum()),
                changed_by_teacher_class=torch.bincount(winner[changed],minlength=10).tolist(),
                maximum_relative_margin_drift=float(signed_residual(before,after).abs().max())))
    result=dict(status='frozen_source_finite_counterfactual_not_actual_AGA_or_candidate',source_n=512,rows=records,
        source_sha256=digest(__file__),point_sha256=cfg['point_sha256'],release_sha256=cfg['release_sha256'],
        source_indices_sha256=hashlib.sha256(json.dumps(point['indices']['fit'][:512]).encode()).hexdigest(),
        checkpoint_sha256=pins,maximum_independent_NumPy_score_error=error,wall_seconds=time.monotonic()-start,
        images_loaded=0,gradients=0,new_release_weights=0,partner_used=False,test_used=False,
        scope='Native cached source activations; exact LN2, QuickGELU, final LN and feature normalization. Removing a head-space component is a counterfactual, not the complete finite AGA map. No attack-effect or universal guarantee.')
    save_json(out,result);print(json.dumps(result,indent=2),flush=True)


@torch.no_grad()
def fixedpoint(output):
    """Independent CPU recount of one closed construction, without image responses."""
    from audit_hermite_splits import exclusions
    out=Path(output);start=time.monotonic();cfg=json.loads((out/'protocol.json').read_text())
    report=json.loads((out/'results.json').read_text());assert report['closed'] and not report['source_feasible']
    assert digest(out/'diagnostic.pt')==report['diagnostic_sha256']
    assert digest(out/'executed_source.py')==report['source_sha256']==cfg['source_sha256']['experiments/alignment_fixedpoint.py']
    raw=torch.load(out/'diagnostic.pt',map_location='cpu');p,q=raw['p'],raw['q'];rows={}
    for k,v in p.items():
        if k in ['O','B']:continue
        if torch.is_tensor(v):assert torch.equal(v,q[k]),k
        else:assert v==q[k],k
    for k,basis in [('O',raw['vo']),('B',raw['vb'])]:
        delta=(q[k]-p[k]).numpy();v=basis.numpy();np.testing.assert_allclose(v@v.T,np.eye(len(v)),atol=1e-10)
        inside=(delta@v.T)@v;outside=delta-inside;den=np.linalg.norm(p[k])
        rows[k]=dict(relative_distance=float(np.linalg.norm(delta)/den),
            frozen_probe_subspace_relative_distance=float(np.linalg.norm(inside)/den),
            frozen_probe_complement_relative_distance=float(np.linalg.norm(outside)/den),
            frozen_probe_subspace_energy_fraction=float(np.linalg.norm(inside)**2/np.linalg.norm(delta)**2))
        assert abs(rows[k]['relative_distance']-report['per_matrix_distance'][k])<1e-10
    db=(q['B']-p['B']).numpy();den=np.linalg.norm(p['B'])
    readout=p['gamma'].numpy()[:,None]*(p['projection'].numpy()@p['head'].numpy().T)
    readout-=readout.mean(0,keepdims=True);readout-=readout.mean(1,keepdims=True)
    u,s,_=np.linalg.svd(readout,full_matrices=False);rank=int((s>s[0]*1e-10).sum());u=u[:,:rank]
    output_geometry=dict(zero_mean_B_relative_distance=float(np.linalg.norm(db-db.mean(0,keepdims=True))/den),
        class_linear_readout_rank=rank,class_linear_B_relative_distance=float(np.linalg.norm(u.T@db)/den),
        class_linear_complement_B_relative_distance=float(np.linalg.norm(db-u@(u.T@db))/den),
        scope='Decomposition of the closed regression only; neither a necessary task-budget bound nor an evaluated replacement.')
    distance=sum(v['relative_distance']**2 for v in rows.values())**.5
    assert abs(distance-report['parameter_distance'])<1e-10 and distance>cfg['parameter_budget']
    anchor,expert,head,pins=checked_models(cfg['target']);assert pins==report['geometry']['checkpoint_sha256']
    base=anchor.state_dict()['model.visual.transformer.resblocks.11.attn.out_proj.weight'].double().numpy()
    errors=[]
    for i in range(0,768,64):
        x=q['O'][:,i:i+64].numpy();a=base[:,i:i+64]
        t=np.linalg.solve(x.T@x+cfg['ridge']*np.eye(64),x.T@a)
        errors.append(float(abs((x.astype('float32')@t.astype('float32')).astype('float64')-x).max()))
    assert max(errors)<cfg['author_weight_tolerance']
    split=json.loads((out/'construction_split.json').read_text());initial=set(cfg['extra_excluded_indices'])
    for path in cfg['extra_index_exclusions']:initial.update(json.loads(Path(path).read_text())['indices'])
    excluded,confirmed=exclusions(split,initial);assert len(excluded)==split['excluded_count']
    ids=np.random.default_rng(cfg['construction_seed']).choice(sorted(set(range(60000))-excluded),cfg['fit_n']+cfg['check_n'],replace=False).tolist()
    assert ids==split['indices']==raw['indices']['fit']+raw['indices']['check']
    assert not set(ids)&(excluded|confirmed) and len(set(ids))==len(ids)
    result=dict(status='passed_independent_closed_point_recount',parameter_distance=distance,by_matrix=rows,
        parameter_budget=cfg['parameter_budget'],source_feasible=False,closed=True,output_geometry=output_geometry,
        NumPy_FP32_fixedpoint_maximum_weight_error=max(errors),historical_excluded_n=len(excluded),
        construction_n=len(ids),fit_n=len(raw['indices']['fit']),source_check_n=len(raw['indices']['check']),
        historical_exposed_indices_used=0,exact_construction_draw_verified=True,unchanged_tensors_verified=True,
        diagnostic_sha256=report['diagnostic_sha256'],executed_source_sha256=report['source_sha256'],
        split_sha256=digest(out/'construction_split.json'),verifier_sha256=digest(__file__),
        images_loaded=0,gradients=0,new_releases=0,confirmation_draws=0,new_TA_or_AGA_evaluations=0,
        fit_residual_independently_replayed=False,wall_seconds=time.monotonic()-start,
        scope='Parameter and input-index recount only. Probe-space fractions are not functional preservation. The least-squares residual is not re-evaluated here; no classification claim or candidate repair.')
    path=out/'independent_validation.json'
    if path.exists():save_json(out/'parameter_recount_initial.json',json.loads(path.read_text()))
    save_json(path,result);print(json.dumps(result,indent=2),flush=True)


@torch.no_grad()
def fixedpoint_cost(output):
    """Replay only the closed fit's spectral cost; never construct another B."""
    from scipy.linalg import svd
    from scipy.special import expit
    from covariance_noise import dataset
    from cross_module_distance import capture_attention
    from task_probe_distance import initialize
    initialize();assert '3090' in torch.cuda.get_device_name()
    out=Path(output);start=time.monotonic();report=json.loads((out/'results.json').read_text())
    assert report['closed'] and digest(out/'diagnostic.pt')==report['diagnostic_sha256']
    raw=torch.load(out/'diagnostic.pt',map_location='cpu');cfg=raw['cfg'];p,q=raw['p'],raw['q']
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    anchor,expert,head,pins=checked_models(cfg['target']);assert pins==report['geometry']['checkpoint_sha256']
    training=dataset(cfg['target'],'data',expert.val_preprocess,True);ids=raw['indices']['fit']
    cache=capture_attention(expert,training,ids);r=cache['r'].numpy();H=cache['o_input'].numpy()
    pn={k:v.numpy() if torch.is_tensor(v) else v for k,v in p.items()}
    def activation(x):
        c=x-x.mean(1,keepdims=True)
        pre=(c/np.sqrt((c*c).mean(1,keepdims=True)+pn['eps2'])*pn['gamma2']+pn['beta2'])@pn['A'].T+pn['a']
        return pre*expit(1.702*pre)
    changed=r+H@(q['O']-p['O']).numpy().T;z=activation(changed)
    target=r-changed+(activation(r)-z)@pn['B'].T
    u,s,vh=svd(z,full_matrices=False,lapack_driver='gesdd',check_finite=True)
    keep=s>s[0]*cfg['regression_rcond'];assert int(keep.sum())==report['fit_rank']
    modes=u.T@target;energy=(modes*modes).sum(1);cost=energy[keep]/s[keep]**2
    den=np.linalg.norm(pn['B']);distance=float(np.sqrt(cost.sum())/den)
    saved=(q['B']-p['B']).numpy().T;residual=float(np.linalg.norm(z@saved-target)/np.linalg.norm(target))
    assert abs(distance-report['per_matrix_distance']['B'])<1e-6
    assert abs(residual-report['fit_relative_hidden_residual'])<1e-6
    projected=vh[keep]@saved;wanted=modes[keep]/s[keep,None]
    coefficient_error=float(np.linalg.norm(projected-wanted)/np.linalg.norm(wanted));assert coefficient_error<1e-6
    rows=[];relative=s/s[0];total=float((target*target).sum())
    for low,high in [(0.1,1.01),(0.01,0.1),(0.001,0.01),(cfg['regression_rcond'],0.001)]:
        mask=(relative>=low)&(relative<high)&keep;values=energy[mask]/s[mask]**2
        rows.append(dict(relative_singular_interval=[low,min(high,1.)],modes=int(mask.sum()),
            target_energy_fraction=float(energy[mask].sum()/total),parameter_cost_fraction=float(values.sum()/cost.sum())))
    np.savez_compressed(out/'fit_spectrum.npz',singular_values=s,target_mode_energy=energy,retained=keep)
    result=dict(status='closed_fit_cost_independently_replayed',n=len(ids),fit_rank=int(keep.sum()),
        B_relative_distance=distance,hidden_relative_residual=residual,coefficient_relative_error=coefficient_error,
        rows=rows,source_point_sha256=report['diagnostic_sha256'],verifier_sha256=digest(__file__),
        spectrum_sha256=digest(out/'fit_spectrum.npz'),source_split_sha256=digest(out/'construction_split.json'),
        original_model_forwards=len(ids),new_training_images=0,new_releases=0,gradients=0,expert_updates=0,
        confirmation_or_test_used=False,wall_seconds=time.monotonic()-start,
        scope='Independent NumPy LN/QuickGELU and SciPy SVD of the original fixed construction. No alternative coefficient, cutoff, direction, release or task-feasibility claim.')
    save_json(out/'fit_cost.json',result);print(json.dumps(result,indent=2),flush=True)


@torch.no_grad()
def matching_response(output):
    """Frozen 2x2 assignment/gauge counterfactual; never select another release."""
    import copy
    from scipy.special import expit
    from clip_benchmark import merge
    from cross_module_distance import capture_attention, extended_parameters, final_hidden
    from covariance_noise import dataset
    from evaluate_local_residual_transport import inputs
    from local_residual_transport import metric
    from probe_margin_distance import scores
    from verify_levelset_audits import convert, raw_scores
    out=Path(output);assert not (out/'matching_response.json').exists();start=time.monotonic()
    cfg=json.loads((out/'protocol.json').read_text());meta=json.loads((out/'execution.json').read_text());assert meta['completed']
    anchor,own,peer,release,heads,_,aligned=inputs(cfg);folder=Path(cfg['release']['path']).parent
    point=torch.load(folder/'point.pt',map_location='cpu');source=torch.load(folder/'source.pt',map_location='cpu')
    source_meta=json.loads((folder/'results.json').read_text());assert digest(folder/'source.pt')==source_meta['raw_sha256']
    proto=json.loads((folder/'protocol.json').read_text());pin=proto['matching_geometry']
    assert digest(pin['path'])==pin['sha256'];g=torch.load(pin['path'],map_location='cpu')
    p0,p1=g['original_restore'],g['changed_restore'];state=own.state_dict();prefix='model.visual.transformer.resblocks.11.mlp.'
    actual=aligned['MNIST'].state_dict()|torch.load(out/'artifacts/AGA_release_edits.pt',map_location='cpu')
    control=merge(anchor,aligned['MNIST'],peer,.5);head=heads['MNIST'];p=extended_parameters(control,head)
    cache=capture_attention(control,dataset('MNIST','data',own.val_preprocess,True),source['indices'])
    rounding=cache['h']-final_hidden(cache['r'],p);before=scores(cache['h'],p);records=[];all_scores={};error=0.
    for gauge in [False,True]:
        for changed_order in [False,True]:
            order=p1 if changed_order else p0;model=copy.deepcopy(aligned['MNIST']);s=model.state_dict();edited=release.state_dict() if gauge else state
            for name in ['c_fc.weight','c_fc.bias']:s[prefix+name].copy_(edited[prefix+name][order])
            s[prefix+'c_proj.weight'].copy_(state[prefix+'c_proj.weight'][:,order])
            if gauge and changed_order:
                assert all(torch.equal(v,actual[k]) for k,v in s.items()),'Factorial endpoint must equal executed author AGA.'
            fused=merge(anchor,model,peer,.5);q=extended_parameters(fused,head)
            for k,v in p.items():
                if k not in ['A','a','B'] and torch.is_tensor(v):assert torch.equal(v,q[k]),k
            h=rounding+final_hidden(cache['r'],q);after=scores(h,q);qn=convert(q);r=cache['r'].numpy();center=r-r.mean(1,keepdims=True)
            pre=(center/np.sqrt((center*center).mean(1,keepdims=True)+qn['eps2'])*qn['gamma2']+qn['beta2'])@qn['A'].T+qn['a']
            hn=rounding.numpy()+r+(pre*expit(1.702*pre))@qn['B'].T+qn['b']
            err=float(abs(raw_scores(hn,qn)-after.numpy()).max());assert err<1e-8;error=max(error,err)
            name=f'gauge{int(gauge)}_assignment{int(changed_order)}';all_scores[name]=after
            records.append(dict(gauge=gauge,changed_assignment=changed_order,**metric(before,after,len(before)),
                score_change_RMS=float((after-before).square().mean().sqrt())))
            del model,fused
    torch.save(dict(scores=all_scores,indices=source['indices']),out/'artifacts/matching_response.pt')
    save_json(out/'matching_response.json',dict(status='frozen_retrospective_factorial_passed',rows=records,
        source_n=len(before),NumPy_complete_tail_maximum_error=error,source_raw_sha256=source_meta['raw_sha256'],
        evaluation_predictions_sha256=meta['predictions_sha256'],release_sha256=cfg['release']['sha256'],
        raw_sha256=digest(out/'artifacts/matching_response.pt'),source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        new_training_images=0,source_image_reuses=len(before),test_images=0,gradients=0,new_releases=0,
        partner_used_for_retrospective_evaluation_only=True,
        scope='Only frozen512 construction indices and this one real target-only merger. Assignment/gauge finite effects are not new defense selection or an unknown-partner guarantee.'))


@torch.no_grad()
def normalization_geometry(output):
    """Decompose the frozen point and recount its evaluated class losses; no fitting."""
    from scipy.special import logsumexp
    out=Path(output);cfg=json.loads((out/'protocol.json').read_text())
    pointdir=Path(cfg['release']['path']).parent
    anchor,expert,head,pins=checked_models('MNIST');state=expert.state_dict();base=anchor.state_dict()
    edits=torch.load(cfg['release']['path'],map_location='cpu');assert digest(cfg['release']['path'])==cfg['release']['sha256']
    point=torch.load(pointdir/'point.pt',map_location='cpu');prefix='model.visual.transformer.resblocks.11.attn.'
    geometry={}
    for key in ([] if point.get('site') in ['MLP','HEAD','READOUT'] else ['in_proj_weight','out_proj.weight']):
        k=prefix+key;original=state[k].double().numpy();delta=edits[k].double().numpy()-original
        if key=='in_proj_weight':
            task=(state[k][1536:]-base[k][1536:]).double().numpy();_,_,vh=np.linalg.svd(task,full_matrices=False);v=vh[:32];d=delta[1536:]
        else:v=point['vo'].numpy();d=delta
        inside=(d@v.T)@v;outside=d-inside;den=np.linalg.norm(original)
        geometry[key]=dict(relative_distance=float(np.linalg.norm(delta)/den),task32_inside_distance=float(np.linalg.norm(inside)/den),
            task32_outside_distance=float(np.linalg.norm(outside)/den),outside_squared_energy_fraction=float(np.linalg.norm(outside)**2/np.linalg.norm(d)**2),
            orthogonality_error=float(abs(np.sum(inside*outside))),denominator='Full edited tensor, including unchanged Q/K where applicable.')
    if point.get('site')=='MLP':
        for k,changed in edits.items():
            original=state[k].double().numpy();delta=changed.double().numpy()-original;den=np.linalg.norm(original)
            geometry[k]=dict(relative_distance=float(np.linalg.norm(delta)/den),denominator='Whole logical A matrix or its own bias, float64.')
            if delta.ndim==2:
                v=point['vh'].numpy();inside=(delta@v.T)@v;outside=delta-inside
                geometry[k].update(task32_inside_distance=float(np.linalg.norm(inside)/den),task32_outside_distance=float(np.linalg.norm(outside)/den),
                    outside_squared_energy_fraction=float(np.linalg.norm(outside)**2/np.linalg.norm(delta)**2))
    if point.get('site')=='HEAD':
        for k,changed in edits.items():
            a,b=changed.double().numpy(),state[k].double().numpy()
            pieces=zip(['Q','K','V'],np.split(a,3),np.split(b,3)) if k.endswith(('in_proj_weight','in_proj_bias')) else [('O',a,b)]
            for name,x,y in pieces:geometry[k+':'+name]=dict(relative_distance=float(np.linalg.norm(x-y)/np.linalg.norm(y)) if np.linalg.norm(y)>0 else 0.,
                denominator='Whole separate logical projection or its bias, float64; known head-permutation control, not functional distance.')
    if point.get('site')=='READOUT':
        for k,changed in edits.items():
            delta=changed.double().numpy()-state[k].double().numpy();den=np.linalg.norm(state[k].double().numpy())
            geometry[k]=dict(relative_distance=float(np.linalg.norm(delta)/den),denominator='Own full final LN bias or projection tensor, float64.')
            if delta.ndim==2:
                v=point['vh'].numpy();inside=v.T@(v@delta);outside=delta-inside
                geometry[k].update(task32_inside_distance=float(np.linalg.norm(inside)/den),task32_outside_distance=float(np.linalg.norm(outside)/den),
                    outside_squared_energy_fraction=float(np.linalg.norm(outside)**2/np.linalg.norm(delta)**2))
        source=torch.load(pointdir/'source.pt',map_location='cpu');h=source['inputs'].double().numpy();gamma=point['gamma'].numpy()
        z=(h-h.mean(1,keepdims=True))/np.sqrt(((h-h.mean(1,keepdims=True))**2).mean(1,keepdims=True)+point['eps'])
        beta,p=point['beta'].numpy(),point['projection'].numpy();head=source['head_weight'].numpy()
        bp,pp=[edits[k].double().numpy() for k in ['model.visual.ln_post.bias','model.visual.proj']]
        f=(z*gamma+beta)@p;mid=(z*gamma+(beta+bp)/2)@((p+pp)/2);shift=mid-f
        n=point['normal'].numpy();c=float(point['constant']);r=json.loads((pointdir/'before_images.json').read_text())['geometry']['r']
        ideal=c*r*r/(4*(1-r))*(n@p);error=float(abs(shift-ideal).max());assert error<1e-6
        scores=f/np.linalg.norm(f,axis=1,keepdims=True)@head.T;after=mid/np.linalg.norm(mid,axis=1,keepdims=True)@head.T
        y=scores.argmax(1);ix=np.arange(len(y));numer=f@head.T;dn=shift@head.T;gap=numer[ix,y,None]-numer;gap[ix,y]=1
        harm=-(dn[ix,y,None]-dn)/np.maximum(gap,1e-12);harm[ix,y]=0
        geometry['fixed_source_self_midpoint']=dict(n=len(y),changed=int((y!=after.argmax(1)).sum()),feature_shift_norm=float(np.linalg.norm(ideal)),
            finite_constant_identity_error=error,maximum_signed_numerator_margin_harm=float(harm.max()),
            maximum_harm_per_image_quantiles=np.quantile(harm.max(1),[.5,.95,1]).tolist(),
            scope='Fixed source-only self-chord after freezing. Not a true partner/AGA evaluation, new candidate, or universal no-change certificate.')
    meta=json.loads((out/'execution.json').read_text());assert meta['completed']
    rawpath=out/'artifacts/predictions.pt';assert digest(rawpath)==meta['predictions_sha256']
    raw=torch.load(rawpath,map_location='cpu');rows=[]
    for key,data in raw.items():
        if not key.startswith('released/'):continue
        ref=raw[key.replace('released/','original/',1)];a,b=[x['logits'].double().numpy() for x in [ref,data]]
        y=data['target'].numpy();np.testing.assert_array_equal(y,ref['target']);ix=np.arange(len(y))
        la=logsumexp(a,axis=1)-a[ix,y];lb=logsumexp(b,axis=1)-b[ix,y];pa,pb=a.argmax(1),b.argmax(1)
        teacher=a.argmax(1);gap=a[ix,teacher,None]-a;gap[ix,teacher]=1;ds=b-a
        harm=-(ds[ix,teacher,None]-ds)/np.maximum(gap,1e-12);harm[ix,teacher]=0
        absolute_drift=float(abs(harm).max())
        if key=='released/standalone/MNIST':
            gate=json.loads((out/'full_native_gate.json').read_text())
            assert abs(absolute_drift-gate['worst_relative_margin_drift'])<1e-9
        rows.append(dict(key=key,original_accuracy=float((pa==y).mean()),released_accuracy=float((pb==y).mean()),
            maximum_absolute_relative_margin_drift=absolute_drift,
            signed_relative_teacher_margin_harm_quantiles=np.quantile(harm.max(1),[.5,.95,1]).tolist(),
            classes=[dict(class_id=c,n=int((y==c).sum()),original_mean_CE=float(la[y==c].mean()),released_mean_CE=float(lb[y==c].mean()),
                lost=int(((pa==y)&(pb!=y)&(y==c)).sum()),gained=int(((pa!=y)&(pb==y)&(y==c)).sum())) for c in range(10)]))
    save_json(out/'mechanism_and_class_loss.json',dict(status='passed_read_only_recount',parameter_geometry=geometry,rows=rows,
        predictions_sha256=meta['predictions_sha256'],release_sha256=cfg['release']['sha256'],verifier_sha256=digest(__file__),
        gradients=0,new_images=0,new_weights=0,scope='Parameter task-SVD geometry is not a functional nullspace. Classification effects and source preservation are measured separately.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument('--head-geometry',action='store_true');mode.add_argument('--head-response',action='store_true')
    mode.add_argument('--fixedpoint',action='store_true')
    mode.add_argument('--fixedpoint-cost',action='store_true')
    mode.add_argument('--normalization',action='store_true')
    mode.add_argument('--matching-response',action='store_true')
    mode.add_argument('--head-identity',action='store_true')
    mode.add_argument('--head-recovery',action='store_true')
    mode.add_argument('--residual-recovery',action='store_true')
    mode.add_argument('--affine-matching',action='store_true')
    args=parser.parse_args();torch.set_num_threads(4)
    (residual_recovery if args.residual_recovery else head_identity_recovery if args.head_recovery else head_identity_geometry if args.head_identity else matching_response if args.matching_response else affine_matching_geometry if args.affine_matching else normalization_geometry if args.normalization else fixedpoint_cost if args.fixedpoint_cost else fixedpoint if args.fixedpoint else head_geometry if args.head_geometry else head_response if args.head_response else audit)(args.output)
