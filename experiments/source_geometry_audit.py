"""Frozen publication chord geometry and exact normalized-readout radius bounds."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from angular_distance import signed_residual
from clip_benchmark import digest
from covariance_noise import dataset
from cross_module_distance import extended_parameters, normalized_input
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import make_loader, save_json
from task_probe_distance import checked_models, gelu
from verify_covariance_noise import official_labels


def class_geometry(h,p):
    """LayerNorm and positive feature normalization preserve numerator ordering."""
    m=p['gamma'][:,None]*(p['projection']@p['head'].T)
    m=m-m.mean(0,keepdim=True);b=p['beta']@p['projection']@p['head'].T
    c=h-h.mean(1,keepdim=True);sigma=(c.square().mean(1,keepdim=True)+p['eps']).sqrt()
    return h@m+sigma*b,m,b,sigma.squeeze(1)


def positive_root(value,linear,quadratic):
    # Positive root of Q*r^2+L*r=value; stable near zero.
    return 2*value.clamp_min(0)/(linear+(linear.square()+4*quadratic*value.clamp_min(0)).sqrt()).clamp_min(1e-30)


def certificates(h0,hm,p):
    v0,m,b,sd0=class_geometry(h0,p);vm,_,_,sdm=class_geometry(hm,p)
    y=v0.argmax(1);rows=torch.arange(len(y),device='cuda');gap0=v0[rows,y,None]-v0;gapm=vm[rows,y,None]-vm
    norm2=m.square().sum(0);gram=m.T@m
    mn=(norm2[y,None]+norm2[None,:]-2*gram[y]).clamp_min(0).sqrt()
    bd=b[y,None]-b;lip=mn+bd.abs()/h0.shape[1]**.5
    hscale=(h0-h0.mean(1,keepdim=True)).norm(dim=1);cm=(hm-hm.mean(1,keepdim=True)).norm(dim=1)
    # A joint, dimensionless representation/readout set. No radius is chosen:
    # ||P eta||<=r*H, ||Delta m_yc||<=r*M_yc,
    # |Delta b_yc|<=r*sqrt(d)*M_yc. Readout contrasts stay centered.
    q=2*mn*hscale[:,None]
    linear0=hscale[:,None]*lip+mn*hscale[:,None]+h0.shape[1]**.5*mn*sd0[:,None]
    linearm=hscale[:,None]*lip+mn*cm[:,None]+h0.shape[1]**.5*mn*sdm[:,None]
    mask=torch.nn.functional.one_hot(y,v0.shape[1]).bool()
    clean_fixed=(gap0/lip.clamp_min(1e-30)).masked_fill(mask,torch.inf).min(1).values
    corrupt_fixed=((-gapm).clamp_min(0)/lip.clamp_min(1e-30)).masked_fill(mask,0).max(1).values
    clean_joint=positive_root(gap0,linear0,q).masked_fill(mask,torch.inf).min(1).values
    corrupt_joint=positive_root(-gapm,linearm,q).masked_fill(mask,0).max(1).values
    fixed=torch.minimum(clean_fixed,corrupt_fixed)/hscale
    joint=torch.minimum(clean_joint,corrupt_joint)
    return {'winner':y,'gap_original':gap0,'gap_midpoint':gapm,'fixed_readout_radius':fixed,'joint_radius':joint,
        'contrast_norm':mn,'contrast_bias':bd,'readout_lipschitz':lip,'hidden_scale':hscale,'sigma0':sd0,'sigmam':sdm}


def describe_radius(values,changed):
    positive=values[changed]
    return {'n':len(values),'positive_count':int((values>0).sum()),'minimum_over_all_inputs':float(values.min()),
        'among_changed_quantiles':torch.quantile(positive,torch.tensor([0.,.25,.5,.75,1.],device='cuda',dtype=torch.float64)).cpu().tolist() if len(positive) else None}


def run(cfg):
    out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();torch.cuda.reset_peak_memory_stats()
    _,expert,head,pins=checked_models(cfg['target']);pc=extended_parameters(expert,head);p=on_device(pc)
    training=dataset(cfg['target'],'data',expert.val_preprocess,True);labels=official_labels(cfg['target'],True)
    raw={'cfg':cfg,'parameters':pc,'cases':{}};records=[];replays=0;readonly_jvps=0
    for case in cfg['cases']:
        path=Path(case['point']);assert digest(path)==case['sha256'];point=torch.load(path,map_location='cpu')
        cache=on_device(point['native']);h0=cache['h'];delta=on_device(point['release_deltas'])
        x0=normalized_input(cache['r'],p);z0=gelu(x0@p['A'].T+p['a']);base=h0-z0@p['B'].T
        def hidden(t):
            if 'A' in delta:
                pre=x0@(p['A']+t*delta['A']).T+p['a']
                return base+gelu(pre)@(p['B']+t*delta['B']).T
            rr=cache['r']+t*cache['o_input']@delta['O'].T
            old=cache['r']+z0@p['B'].T+p['b']
            rounding=h0-old
            pre=normalized_input(rr,p)@p['A'].T+p['a']
            return rounding+rr+gelu(pre)@(p['B']+t*delta['B']).T+p['b']
        zero=torch.tensor(0.,device='cuda',dtype=torch.float64);one=torch.tensor(1.,device='cuda',dtype=torch.float64)
        hm,h1=hidden(.5),hidden(1.);s0,sm,s1=[scores(h,p) for h in [h0,hm,h1]]
        y=s0.argmax(1);changed=sm.argmax(1)!=y
        assert torch.equal(y,s1.argmax(1)), 'Frozen native calibration preservation changed'
        scalar=torch.tensor(0.,device='cuda',dtype=torch.float64)
        first=torch.autograd.functional.jvp(hidden,scalar,one,create_graph=False)[1];readonly_jvps+=1
        derivative=lambda t:torch.autograd.functional.jvp(hidden,t,one,create_graph=True)[1]
        second=torch.autograd.functional.jvp(derivative,scalar,one,create_graph=False)[1];readonly_jvps+=2
        chord=hm-.5*(h0+h1);approx=-second/8
        cert=certificates(h0,hm,p);nom=class_geometry(hm,p)[0]
        assert torch.equal(nom.argmax(1),sm.argmax(1))
        predicted=s0.argmax(1);row=torch.arange(len(y),device='cuda');teacher_gap=s0[row,y,None]-s0
        score_gap=sm[row,y,None]-sm
        chord_score=scores(hm,p)-.5*(s0+s1)
        indices=point['indices']['fit'];gold=torch.tensor(labels[indices],device='cuda')
        per_class=[]
        for k in range(len(p['head'])):
            members=y==k
            per_class.append({'class':k,'n':int(members.sum()),'midpoint_changed':int((changed&members).sum()),
                'fixed_radius_median':float(cert['fixed_readout_radius'][members].median()),'joint_radius_median':float(cert['joint_radius'][members].median())})
        fresh=[]
        for factor in [0.,.5,1.]:
            model=copy.deepcopy(expert);prefix='model.visual.transformer.resblocks.11.'
            for site,d in delta.items():
                key=prefix+('attn.out_proj.weight' if site=='O' else 'mlp.c_fc.weight' if site=='A' else 'mlp.c_proj.weight')
                weight=model.state_dict()[key];weight.copy_((weight.double()+factor*d.cpu()).to(weight.dtype))
            model.cuda().eval();pred=[]
            with torch.no_grad():
                for images,_ in make_loader(training,indices[:cfg['replay_n']],64):
                    f=model(images.cuda());f=f/f.norm(dim=1,keepdim=True);pred.append((f@head.weight.cuda().T).argmax(1).cpu())
            expected=[s0,sm,s1][int(2*factor)][:cfg['replay_n']].argmax(1).cpu()
            assert torch.equal(torch.cat(pred),expected);replays+=cfg['replay_n'];fresh.append({'factor':factor,'n':cfg['replay_n'],'match':True});model.cpu()
        records.append({'case':case['name'],'n':len(y),'native_original_correct':int((y==gold).sum()),'native_endpoint_correct':int((s1.argmax(1)==gold).sum()),
            'source_midpoint_correct':int((sm.argmax(1)==gold).sum()),'source_midpoint_changed':int(changed.sum()),'classes':per_class,
            'endpoint_worst_margin_drift':float(signed_residual(s0,s1).abs().max()),'midpoint_worst_margin_drift':float(signed_residual(s0,sm).abs().max()),
            'hidden_chord_deviation_norm':float(chord.norm()),'quadratic_chord_relative_error':float((chord-approx).norm()/chord.norm().clamp_min(1e-30)),
            'native_initial_hidden_tangent_norm':float(first.norm()),'native_initial_hidden_second_derivative_norm':float(second.norm()),
            'certified_fixed_readout':describe_radius(cert['fixed_readout_radius'],changed),'certified_joint_readout':describe_radius(cert['joint_radius'],changed),
            'fresh_prediction_replays':fresh,'role':'Source-expert publication chord; not a real peer fusion or a new defense.'})
        raw['cases'][case['name']]={'cache':point['native'],'indices':indices,'labels':gold.cpu(),'delta':point['release_deltas'],
            'hidden0':h0.cpu(),'hidden_midpoint':hm.detach().cpu(),'hidden1':h1.detach().cpu(),'first_derivative':first.cpu(),'second_derivative':second.cpu(),
            'certificates':{k:v.cpu() for k,v in cert.items()},'scores0':s0.cpu(),'scores_midpoint':sm.detach().cpu(),'scores1':s1.detach().cpu(),'chord_score':chord_score.detach().cpu()}
        print(json.dumps(records[-1]),flush=True)
    file=out/'diagnostic.pt';torch.save(raw,file)
    report={'status':'source_chord_and_exact_classification_neighborhood_diagnosis','target':cfg['target'],'records':records,
        'diagnostic_sha256':digest(file),'source_sha256':digest(__file__),'protocol_sha256':digest(cfg['protocol_path']),'checkpoint_sha256':pins,
        'wall_seconds':time.monotonic()-start,'peak_gpu_bytes':torch.cuda.max_memory_allocated(),'new_full_model_forward_instances':replays,
        'read_only_JVP_calls':readonly_jvps,'publication_updates':0,'new_release_weights':0,'new_proxy_models':0,'partner_used':False,'test_used':False,'confirmation_used':False,
        'limitation':'Certificates are conditional representation/readout sets around the source chord. Target-only information does not prove that arbitrary future merged models belong to these sets.'}
    save_json(out/'results.json',report);save_json(out/'protocol.json',cfg);print(json.dumps(report),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--protocol',required=True);args=parser.parse_args()
    cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;run(cfg)
