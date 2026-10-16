"""One source-only local reflection: test the missing skip, without new optimizers."""
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit
from clip_benchmark import digest, merge
from confirm_temperature_compensation import unused
from covariance_noise import dataset
from cross_module_distance import capture_attention, extended_parameters, final_hidden, normalized_input
from probe_distance_candidate import on_device
from probe_margin_distance import margin_drift, scores
from subspace_probe import make_loader, save_json
from task_probe_distance import checked_models, gelu, initialize
from verify_levelset_audits import raw_scores

FIELDS={'O':'attn.out_proj.weight','ob':'attn.out_proj.bias','A':'mlp.c_fc.weight',
        'a':'mlp.c_fc.bias','B':'mlp.c_proj.weight','b':'mlp.c_proj.bias'}

def cpu(d):
    return {k:v.detach().cpu() if torch.is_tensor(v) else v for k,v in d.items()}

@torch.no_grad()
def capture(model, data, ids):
    values=[]; hook=model.model.visual.transformer.resblocks[11].ln_1.register_forward_pre_hook(
        lambda _m,args:values.append(args[0][0].detach().cpu()))
    try: cache=capture_attention(model,data,ids)
    finally: hook.remove()
    cache['skip']=torch.cat(values).double(); return cache

@torch.no_grad()
def pack(cache,p,vo,vb,cfg):
    c=on_device(cache); r,H=c['r'],c['o_input']; n=len(r)
    teacher=scores(c['h'],p).argmax(1); take=[]
    for k in range(len(p['head'])):
        ix=(teacher==k).nonzero().flatten(); assert len(ix)>=cfg['contexts_per_class']
        take.extend(ix[:cfg['contexts_per_class']].tolist())
    take=torch.tensor(take,device='cuda'); z=gelu(normalized_input(r,p)@p['A'].T+p['a'])
    shifts=[]
    for x,v in [(H,vo),(z,vb)]:
        rms=(x@v.T).square().mean(0).sqrt()
        shifts.append((cfg['finite_step']*torch.tensor([-1.,1.],device='cuda')[:,None,None]*rms[None,:,None]*v[None]).flatten(0,1))
    so,sb=shifts; m=len(so); rounding=c['h']-final_hidden(r,p)
    return dict(r=torch.cat([r,(r[take,None]+(so@p['O'].T)[None]).flatten(0,1),r[take].repeat_interleave(m,0)]),
        H=torch.cat([H,(H[take,None]+so[None]).flatten(0,1),H[take].repeat_interleave(m,0)]),
        rounding=torch.cat([rounding,rounding[take].repeat_interleave(m,0),rounding[take].repeat_interleave(m,0)]),
        offset=torch.cat([torch.zeros_like(z),torch.zeros((len(take)*m,z.shape[1]),device='cuda'),sb.repeat(len(take),1)]),natural_n=n)

@torch.no_grad()
def forward(data,p,q):
    r=data['r']+data['H']@(q['O']-p['O']).T+q['ob']-p['ob']
    z=gelu(normalized_input(r,q)@q['A'].T+q['a'])+data['offset']
    return scores(data['rounding']+r+z@q['B'].T+q['b'],q)

def metric(before,after,n):
    changed=before[:n].argmax(1)!=after[:n].argmax(1)
    return dict(natural_n=n,task_inputs=len(before),changed=int(changed.sum()),
        changed_by_teacher_class=torch.bincount(before[:n].argmax(1)[changed],minlength=before.shape[1]).cpu().tolist(),
        worst_task_drift=float(margin_drift(before,after).max()),
        natural_worst_task_drift=float(margin_drift(before[:n],after[:n]).max()))

@torch.no_grad()
def run(cfg):
    start=time.monotonic();out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=True)
    assert not (out/'diagnostic.pt').exists() and not (out/'results.json').exists(),'no replay after responses'
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    assert '3090' in torch.cuda.get_device_name();torch.cuda.reset_peak_memory_stats()
    anchor,expert,head,pins=checked_models(cfg['target']); p=on_device(extended_parameters(expert,head));p0=on_device(extended_parameters(anchor,head))
    train=dataset(cfg['target'],'data',expert.val_preprocess,True); excluded=set(cfg['extra_excluded_indices'])
    for path in cfg['extra_index_exclusions']:
        excluded.update(json.loads(Path(path).read_text())['indices'])
    splitfile=out/'construction_split.json'
    if splitfile.exists():
        assert (out/'failed_dtype_protocol.json').exists(),'only the recorded pre-response dtype repair may resume'
        split=json.loads(splitfile.read_text());assert split['seed']==cfg['construction_seed'] and len(split['indices'])==cfg['fit_n']+cfg['scale_n']
    else:split=unused(train,excluded,cfg|dict(confirmation_n=cfg['fit_n']+cfg['scale_n'],confirmation_seed=cfg['construction_seed']))
    split.update(used_for_optimization=True,role='construction only, never independent confirmation')
    ids={'fit':split['indices'][:cfg['fit_n']],'scale':split['indices'][cfg['fit_n']:]}
    save_json(out/'construction_split.json',split);save_json(out/'training_indices.json',ids);save_json(out/'protocol.json',cfg)
    cache={k:capture(expert,train,v) for k,v in ids.items()}; f=on_device(cache['fit'])
    d=len(p['gamma']);assert float(p['gamma2'].abs().min())>1e-10
    readout=torch.cat([torch.ones((d,1),device='cuda'),p['gamma'][:,None]*p['projection']],1)
    u,s,_=torch.linalg.svd(readout,full_matrices=True);rank=int((s>s[0]*1e-10).sum());N=u[:,rank:];assert N.shape[1]>0
    _,task_s,vh=torch.linalg.svd(((p['A']-p0['A'])*p['gamma2'][None])@N,full_matrices=False)
    axis=N@vh[0];axis*=torch.sign(axis[axis.abs().argmax()]);R=torch.eye(d,device='cuda')-2*axis[:,None]*axis[None]
    _,_,vo=torch.linalg.svd(p['O']-p0['O'],full_matrices=False);vo=vo[:cfg['probe_rank']]
    _,_,vb=torch.linalg.svd(p['B']-p0['B'],full_matrices=False);vb=vb[:cfg['probe_rank']]
    complement=torch.linalg.qr(vo.T,mode='complete').Q[:,cfg['probe_rank']:]
    H=f['o_input'];y=f['skip']@axis;meanH=H.mean(0);ym=y.mean();X=(H-meanH)@complement
    gram=X.T@X;ridge=cfg['ridge_relative']*torch.linalg.eigvalsh(gram)[-1]
    v=complement@torch.linalg.solve(gram+ridge*torch.eye(len(gram),device='cuda'),X.T@(y-ym));intercept=ym-meanH@v
    q=dict(p);q['O']=R@p['O']-2*axis[:,None]*v[None];q['ob']=p['ob']@R-2*intercept*axis
    q['A']=((p['A']*p['gamma2'][None])@R)/p['gamma2'][None];q['a']=p['a']+(p['A']-q['A'])@p['beta2']
    q['B']=R@p['B'];q['b']=p['b']@R
    for k in FIELDS:q[k]=q[k].float().double()
    distance=float(sum(((q[k]-p[k]).norm()/p[k].norm()).square() for k in FIELDS).sqrt())
    matrices=float(sum(((q[k]-p[k]).norm()/p[k].norm()).square() for k in ['O','A','B']).sqrt())
    middle=p|{k:((p[k]+q[k])*.5).float().double() for k in FIELDS}
    report=dict(status='local_residual_transport_source_diagnosis',checkpoint_sha256=pins,parameter_distance=distance,
        matrix_distance=matrices,readout_null_dimension=N.shape[1],readout_null_error=float((axis@readout).abs().max()),
        orthogonality_error=float((R.T@R-torch.eye(d,device='cuda')).abs().max()),task_generator_singular_value=float(task_s[0]),
        task_generator_is_not_functional_importance_proof=True,probe_input_orthogonality_error=float((vo@v).abs().max()),
        original_expert_updated=False,gradients=0,optimizer_steps=0,new_release_weights=0,partner_used=False,test_used=False,confirmation_used=False,groups={})
    saved={}
    for name,c in cache.items():
        data=pack(c,p,vo,vb,cfg);before=forward(data,p,p);after=forward(data,p,q);half=forward(data,p,middle)
        cn=on_device(c);error=cn['skip']@axis-cn['o_input']@v-intercept
        report['groups'][name]=dict(source=metric(before,after,len(c['h'])),source_midpoint=metric(before,half,len(c['h'])),
            bypass_error_rms=float(error.square().mean().sqrt()),bypass_error_max=float(error.abs().max()),
            ideal_reflection_score_error=float((scores(cn['h']@R,p)-scores(cn['h'],p)).abs().max()))
        saved[name]=cpu(data)|dict(before=before.cpu(),after=after.cpu(),middle=half.cpu())
    feasible=distance<=cfg['parameter_budget'] and all(g['source']['worst_task_drift']<=cfg['task_budget'] and g['source']['changed']==0 for g in report['groups'].values())
    scale=report['groups']['scale']['source_midpoint'];signal=scale['changed']>=cfg['minimum_finite_changes'] and sum(n>0 for n in scale['changed_by_teacher_class'])>=cfg['minimum_changed_classes']
    release=copy.deepcopy(expert)
    for k,key in FIELDS.items():release.state_dict()['model.visual.transformer.resblocks.11.'+key].copy_(q[k].cpu().float())
    controls={}
    for name,model in [('original',expert),('diagnostic',release)]:
        actual=[];model.cuda();head.cuda()
        for images,_ in make_loader(train,ids['scale'][:cfg['full_replay_n']],64):actual.append(head(model(images.cuda())).cpu())
        model.cpu();head.cpu();actual=torch.cat(actual).double();wanted=saved['scale']['before' if name=='original' else 'after'][:len(actual)]
        err=float((actual-wanted).abs().max());controls[name]=dict(n=len(actual),score_error=err,predictions_match=bool(torch.equal(actual.argmax(1),wanted.argmax(1))))
        assert err<=cfg['replay_tolerance'] and controls[name]['predictions_match'],'fixed full-model control failed'
    report.update(source_feasible=feasible,finite_source_midpoint_signal=signal,full_model_controls=controls,
        status='source_condition_supported_not_a_release' if feasible and signal else 'closed_local_residual_transport_fixed_direction',
        stop_reason='none' if feasible and signal else 'source_budget_failure' if not feasible else 'no_predicted_multiclass_finite_source_signal',
        source_sha256=digest(__file__),protocol_sha256=digest(cfg['protocol_path']),new_full_model_forwards=len(split['indices'])+2*cfg['full_replay_n'],
        peak_gpu_bytes=torch.cuda.max_memory_allocated(),wall_seconds=time.monotonic()-start)
    torch.save(dict(p=cpu(p),q=cpu(q),axis=axis.cpu(),R=R.cpu(),v=v.cpu(),intercept=intercept.cpu(),vo=vo.cpu(),vb=vb.cpu(),groups=saved,indices=ids,cfg=cfg),out/'diagnostic.pt')
    report['diagnostic_sha256']=digest(out/'diagnostic.pt');save_json(out/'results.json',report);print(json.dumps(report),flush=True)

def verify(folder):
    out=Path(folder);start=time.monotonic();rep=json.loads((out/'results.json').read_text());assert digest(out/'diagnostic.pt')==rep['diagnostic_sha256']
    raw=torch.load(out/'diagnostic.pt',map_location='cpu');p,q=[{k:v.numpy() if torch.is_tensor(v) else v for k,v in raw[t].items()} for t in ['p','q']]
    maximum=0.; rows={}
    for name,data in raw['groups'].items():
        c={k:v.numpy() if torch.is_tensor(v) else v for k,v in data.items()}
        for label,params in [('before',p),('after',q),('middle',p|{k:((p[k]+q[k])*.5).astype('float32').astype('float64') for k in FIELDS})]:
            r=c['r']+c['H']@(params['O']-p['O']).T+params['ob']-p['ob'];center=r-r.mean(1,keepdims=True)
            z=(center/np.sqrt((center**2).mean(1,keepdims=True)+params['eps2'])*params['gamma2']+params['beta2'])@params['A'].T+params['a']
            h=c['rounding']+r+(z*expit(1.702*z)+c['offset'])@params['B'].T+params['b'];actual=raw_scores(h,params)
            maximum=max(maximum,float(abs(actual-c[label]).max()));np.testing.assert_allclose(actual,c[label],atol=1e-8,rtol=1e-9)
            np.testing.assert_array_equal(actual.argmax(1),c[label].argmax(1))
        y=c['before'].argmax(1);i=np.arange(len(y));gap=c['before'][i,y,None]-c['before'];gap[i,y]=1
        delta=c['after']-c['before'];drift=abs((delta[i,y,None]-delta)/gap);drift[i,y]=0
        rows[name]=float(drift.max());assert abs(rows[name]-rep['groups'][name]['source']['worst_task_drift'])<1e-8
    result=dict(status='passed',numpy_score_max_error=maximum,source_drift=rows,source_feasible=rep['source_feasible'],new_release_weights=0,gradients=0,source_sha256=digest(__file__),wall_seconds=time.monotonic()-start)
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)

@torch.no_grad()
def anchor_check(folder):
    """Complete the stated public-base record; never reselect or promote the point."""
    out=Path(folder);start=time.monotonic();rep=json.loads((out/'results.json').read_text())
    assert not (out/'anchor_interpolation.json').exists() and digest(out/'diagnostic.pt')==rep['diagnostic_sha256']
    raw=torch.load(out/'diagnostic.pt',map_location='cpu');cfg=raw['cfg'];anchor,expert,head,pins=checked_models(cfg['target'])
    assert pins==rep['checkpoint_sha256'];base=merge(anchor,expert,anchor,.5);p=on_device(extended_parameters(base,head));q=dict(p)
    for k in FIELDS:q[k]=(p[k]+.5*(raw['q'][k]-raw['p'][k]).cuda()).float().double()
    train=dataset(cfg['target'],'data',expert.val_preprocess,True);ids=raw['indices']['scale'];c=on_device(capture(base,train,ids))
    data=dict(r=c['r'],H=c['o_input'],rounding=c['h']-final_hidden(c['r'],p),offset=torch.zeros((len(ids),p['A'].shape[0]),device='cuda'))
    before,after=forward(data,p,p),forward(data,p,q);result=metric(before,after,len(ids));R={}
    for name,par,wanted in [('original',p,before),('changed',q,after)]:
        model=copy.deepcopy(base)
        for k,key in FIELDS.items():model.state_dict()['model.visual.transformer.resblocks.11.'+key].copy_(par[k].cpu().float())
        actual=[];model.cuda();head.cuda()
        for images,_ in make_loader(train,ids[:cfg['full_replay_n']],64):actual.append(head(model(images.cuda())).cpu())
        model.cpu();head.cpu();a=torch.cat(actual).double();err=float((a-wanted[:len(a)].cpu()).abs().max())
        assert err<=cfg['replay_tolerance'] and torch.equal(a.argmax(1),wanted[:len(a)].cpu().argmax(1));R[name]=dict(n=len(a),score_error=err)
    def numpy_forward(params):
        pn={k:v.cpu().numpy() if torch.is_tensor(v) else v for k,v in params.items()};dn={k:v.cpu().numpy() for k,v in data.items()}
        r=dn['r']+dn['H']@(pn['O']-p['O'].cpu().numpy()).T+pn['ob']-p['ob'].cpu().numpy();cc=r-r.mean(1,keepdims=True)
        pre=(cc/np.sqrt((cc**2).mean(1,keepdims=True)+pn['eps2'])*pn['gamma2']+pn['beta2'])@pn['A'].T+pn['a']
        return raw_scores(dn['rounding']+r+(pre*expit(1.702*pre))@pn['B'].T+pn['b'],pn)
    numpy_error=0.
    for par,wanted in [(p,before),(q,after)]:
        actual=numpy_forward(par);numpy_error=max(numpy_error,float(abs(actual-wanted.cpu().numpy()).max()));np.testing.assert_allclose(actual,wanted.cpu(),atol=1e-8,rtol=1e-9)
    torch.save(dict(data=cpu(data),p=cpu(p),q=cpu(q),before=before.cpu(),after=after.cpu(),indices=ids),out/'anchor_interpolation.pt')
    result.update(status='fixed_public_base_record_not_unknown_peer_test',source_point_unchanged=rep['diagnostic_sha256'],
        full_model_controls=R,numpy_score_max_error=numpy_error,new_candidates=0,point_promoted=False,new_gradients=0,
        protocol_scope='Complete predeclared public-base interpolation record only; no original results overwritten or failed midpoint gate relaxed.',
        source_sha256=digest(__file__),raw_sha256=digest(out/'anchor_interpolation.pt'),wall_seconds=time.monotonic()-start,full_model_forwards=len(ids)+2*cfg['full_replay_n'])
    save_json(out/'anchor_interpolation.json',result);print(json.dumps(result),flush=True)

def inspect_geometry(folder):
    """Read-only finite classification decomposition of the already closed point."""
    out=Path(folder);start=time.monotonic();raw=torch.load(out/'diagnostic.pt',map_location='cpu')
    p,q=[{k:v.numpy() if torch.is_tensor(v) else v for k,v in raw[t].items()} for t in ['p','q']]
    axis=raw['axis'].numpy();mid=p|{k:((p[k]+q[k])*.5).astype('float32').astype('float64') for k in FIELDS}
    m=p['gamma'][:,None]*(p['projection']@p['head'].T);m-=m.mean(0);bias=p['beta']@p['projection']@p['head'].T
    contributions={k:float(np.linalg.norm(q[k]-p[k])**2/np.linalg.norm(p[k])**2) for k in FIELDS};groups={}
    for name,data in raw['groups'].items():
        n=data['natural_n'];c={k:v[:n].numpy() for k,v in data.items() if torch.is_tensor(v)};r0=c['r'];r1=r0+c['H']@(mid['O']-p['O']).T+mid['ob']-p['ob']
        def activation(r,pp):
            cc=r-r.mean(1,keepdims=True);v=(cc/np.sqrt((cc**2).mean(1,keepdims=True)+pp['eps2'])*pp['gamma2']+pp['beta2'])@pp['A'].T+pp['a']
            return v*expit(1.702*v)
        z0=activation(r0,p);z1=activation(r1,mid)
        h0=c['rounding']+r0+z0@p['B'].T+p['b'];h1=c['rounding']+r1+z1@mid['B'].T+mid['b']
        sd=lambda h:np.sqrt(((h-h.mean(1,keepdims=True))**2).mean(1,keepdims=True)+p['eps'])
        n0=h0@m+sd(h0)*bias;n1=h1@m+sd(h1)*bias;y=n0.argmax(1);ii=np.arange(n)
        gap=n0[ii,y,None]-n0;gap[ii,y]=1
        terms={'gate':((z1-z0)@mid['B'].T)@m,'direct':((r1-r0)+z0@(mid['B']-p['B']).T+mid['b']-p['b'])@m,'final_LN_scale':(sd(h1)-sd(h0))*bias}
        err=float(abs(sum(terms.values())-(n1-n0)).max());assert err<1e-8
        damage=-(n1-n0)[ii,y,None]+(n1-n0);damage/=gap;damage[ii,y]=-np.inf;competitor=damage.argmax(1);maximum=damage[ii,competitor]
        changed=n1.argmax(1)!=y;assert np.array_equal(changed,maximum>1)
        centered=r0-r0.mean(1,keepdims=True);energy=(centered@axis)**2/(centered**2).sum(1)
        pieces={key:(-(v[ii,y]-v[ii,competitor])/gap[ii,competitor]) for key,v in terms.items()}
        groups[name]=dict(n=n,changed=int(changed.sum()),reflection_axis_centered_energy_quantiles=np.quantile(energy,[.5,.95,1]).tolist(),
            signed_margin_harm_quantiles=np.quantile(maximum,[.5,.95,1]).tolist(),maximum_signed_harm=float(maximum.max()),finite_numerator_identity_error=err,
            competing_direction_term_quantiles={k:np.quantile(v,[.5,.95,1]).tolist() for k,v in pieces.items()},
            direct_numerator_max=float(abs(terms['direct']).max()),by_class=[dict(class_id=k,n=int((y==k).sum()),changed=int(changed[y==k].sum()),maximum_harm=float(maximum[y==k].max())) for k in range(len(p['head']))])
    result=dict(status='closed_point_finite_geometry_verified',distance_squared_by_tensor=contributions,
        bias_fraction_of_distance_squared=sum(contributions[k] for k in ['ob','a','b'])/sum(contributions.values()),groups=groups,
        source_point_sha256=digest(out/'diagnostic.pt'),source_sha256=digest(__file__),new_images=0,new_gradients=0,new_candidates=0,wall_seconds=time.monotonic()-start)
    save_json(out/'geometry_validation.json',result);print(json.dumps(result),flush=True)

def verify_splits(folder):
    from audit_hermite_splits import exclusions
    out=Path(folder);source=json.loads((out/'construction_split.json').read_text());fresh=json.loads((out/'confirmation_split.json').read_text())
    cfg=json.loads((out/'protocol.json').read_text());initial=set(cfg['extra_excluded_indices']);extra={}
    for path in cfg['extra_index_exclusions']:
        initial.update(json.loads(Path(path).read_text())['indices']);extra[path]=digest(path)
    old,exposed=exclusions(source,initial);assert len(old)==source['excluded_count']
    chosen=np.random.default_rng(source['seed']).choice(sorted(set(range(60000))-old),len(source['indices']),replace=False).tolist();assert chosen==source['indices']
    old2,exposed2=exclusions(fresh,initial|set(chosen));assert len(old2)==fresh['excluded_count']
    chosen2=np.random.default_rng(fresh['seed']).choice(sorted(set(range(60000))-old2),len(fresh['indices']),replace=False).tolist();assert chosen2==fresh['indices']
    assert not set(chosen)&exposed and not set(chosen2)&old2
    result=dict(status='passed_registered_exclusions_and_exact_both_draws',construction_n=len(chosen),confirmation_n=len(chosen2),
        construction_excluded_n=len(old),confirmation_excluded_n=len(old2),exposed_confirmations_used_for_construction=0,construction_confirmation_overlap=0,
        source_split_sha256=digest(out/'construction_split.json'),confirmation_split_sha256=digest(out/'confirmation_split.json'),extra_exclusion_sha256=extra,
        new_images=0,new_gradients=0,new_weights=0,source_sha256=digest(__file__))
    save_json(out/'split_independence_audit.json',result);print(json.dumps(result),flush=True)

def aligned_geometry(folder):
    from aga_coordinate_audit import permutation
    out=Path(folder);cfg=json.loads((out/'protocol.json').read_text());meta=json.loads((out/'execution.json').read_text())
    root=Path(cfg['release']['path']).parent;raw=torch.load(root/'diagnostic.pt',map_location='cpu');p,q=raw['p'],raw['q']
    ref=cfg['AGA_references']['MNIST'];assert digest(ref['path'])==ref['sha256'];original=torch.load(ref['path'],map_location='cpu')
    path=out/'artifacts/AGA_release_edits.pt';assert digest(path)==meta['AGA_edits_sha256'];changed=original|torch.load(path,map_location='cpu')
    key='model.visual.transformer.resblocks.11.mlp.'
    p0=permutation(p['A'].float(),original[key+'c_fc.weight']);p1=permutation(q['A'].float(),changed[key+'c_fc.weight'])
    for par,state,perm in [(p,original,p0),(q,changed,p1)]:
        assert torch.equal(par['a'].float()[perm],state[key+'c_fc.bias'])
        assert torch.equal(par['B'].float()[:,perm],state[key+'c_proj.weight'])
    M=p['gamma'][:,None]*(p['projection']@p['head'].T);M-=M.mean(0);rows={}
    for name,db in [('before_AGA',q['B']-p['B']),('after_AGA',changed[key+'c_proj.weight'].double()-original[key+'c_proj.weight'].double())]:
        rows[name]=dict(B_difference_frobenius=float(db.norm()),B_difference_readout_frobenius=float((db.T@M).norm()),
            normalized_B_readout_ratio=float((db.T@M).norm()/(db.norm()*M.norm())))
    result=dict(status='frozen_point_source_readout_geometry_verified',changed_MLP_permutation_positions=int((p0!=p1).sum()),
        source_readout_axis_norm=float((raw['axis']@M).norm()),operators=rows,
        source_point_sha256=digest(root/'diagnostic.pt'),AGA_edits_sha256=meta['AGA_edits_sha256'],source_sha256=digest(__file__),
        new_images=0,new_attacks=0,new_gradients=0,new_weights=0,partner_loaded=False,
        scope='Read-only source/author-alignment operator geometry; no claim operator norms alone prove fused class outcomes.')
    save_json(out/'readout_geometry.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol');ap.add_argument('--verify');ap.add_argument('--anchor-check');ap.add_argument('--geometry');ap.add_argument('--verify-splits');ap.add_argument('--aligned-geometry');args=ap.parse_args();initialize()
    if args.verify:verify(args.verify)
    elif args.anchor_check:anchor_check(args.anchor_check)
    elif args.geometry:inspect_geometry(args.geometry)
    elif args.verify_splits:verify_splits(args.verify_splits)
    elif args.aligned_geometry:aligned_geometry(args.aligned_geometry)
    else:
        cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol;run(cfg)
