"""Independent NumPy attention, conditional publication solve and task derivatives."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit,softmax

from clip_benchmark import digest
from covariance_noise import dataset
from query_distance import query_parameters
from subspace_probe import save_json
from task_probe_distance import checked_models
from verify_classification_completion import target_numpy
from verify_cross_module import ln,ln_jvp
from verify_probe_candidate import array,features
from verify_covariance_noise import official_labels


class Reconstruction:
    def __init__(self,cal,cfg,p,space):
        self.cal,self.cfg,self.p=cal,cfg,p;self.rank=cfg['probe_rank'];self.size=self.rank**2
        self.u,self.v=[array(cal['basis_Q'][k]) for k in ['u','vh']];self.vb=array(cal['basis_B']['vh']);self.oq=array(space['Q']);self.w=array(space['w'])
        self.a=1.-np.mean((self.w-self.w.mean())**2);self.nq,self.nb=np.linalg.norm(p['Q']),np.linalg.norm(p['B'])
        for basis in [self.u.T,self.v,self.vb]:np.testing.assert_allclose(basis@basis.T,np.eye(self.rank),atol=1e-9)
        np.testing.assert_allclose(self.oq.T@self.oq,np.eye(self.oq.shape[1]),atol=1e-10)
    def hidden(self,r):
        p=self.p;pre=ln(r,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a'];return pre*expit(1.702*pre)
    def attention(self,x,k,v,e=None):
        p=self.p;q=(x@p['Q'].T+p['qb']+(0 if e is None else x@e.T)).reshape(len(x),p['heads'],p['head_dim'])
        a=softmax(np.einsum('nhd,nhsd->nhs',q,k)/p['head_dim']**.5,axis=-1)
        return np.einsum('nhs,nhsd->nhd',a,v).reshape(len(x),-1),a
    def delta(self,theta):
        eq=self.nq*self.u@theta[:self.size].reshape(self.rank,self.rank)@self.v
        eb=self.nb*self.oq@theta[self.size:].reshape(self.oq.shape[1],self.p['B'].shape[1])
        return eq,eb
    def conditional(self,co):
        p=self.p;real={k:np.concatenate([array(self.cal[g][k]) for g in ['native','scale']]) for k in ['r','q_input','keys','values','h']}
        r,x,teacher=[real[k] for k in ['r','q_input','h']];z0=self.hidden(r);rounding=teacher-r-z0@p['B'].T-p['b']
        q0=z0@self.vb.T;ridge=self.cfg['ridge_relative_floor']*np.linalg.eigvalsh(q0.T@q0)[-1]
        eq=self.nq*self.u@co.reshape(self.rank,self.rank)@self.v
        original,_=self.attention(x,real['keys'],real['values']);changed,_=self.attention(x,real['keys'],real['values'],eq)
        r1=r+(changed-original)@p['O'].T;z=self.hidden(r1);base=rounding+r1+z@p['B'].T+p['b']
        linear=p['gamma'][:,None]*p['projection'];linear-=linear.mean(0);bias=p['beta']@p['projection']
        ideal,_,_=target_numpy(base,teacher,self.oq,self.w,self.a,linear,bias,p['eps'],1.-self.cfg['margin_fraction']/2.)
        kb=np.linalg.solve(z@z.T+ridge*np.eye(len(z)),(ideal-base)@self.oq).T@z
        return np.concatenate([co.flatten(),(kb/self.nb).flatten()]),ridge
    def group(self,g,theta,direction=None,trial=None):
        p=self.p;cache=self.cal[g];r,x,h=[array(cache[k]) for k in ['r','q_input','h']];keys,values=[array(cache[k]) for k in ['keys','values']]
        rounding=h-r-self.hidden(r)@p['B'].T-p['b'];base,_=self.attention(x,keys,values)
        n,count=self.cfg['probe_contexts'],2*self.rank
        sq=(self.cfg['finite_step']*np.array([-1.,1.])[:,None,None]*np.sqrt(np.mean((x@self.v.T)**2,0))[None,:,None]*self.v[None,:,:]).reshape(count,-1)
        sb=(self.cfg['finite_step']*np.array([-1.,1.])[:,None,None]*np.sqrt(np.mean((self.hidden(r)@self.vb.T)**2,0))[None,:,None]*self.vb[None,:,:]).reshape(count,-1)
        xx=np.concatenate([x,(x[:n,None,:]+sq[None,:,:]).reshape(n*count,-1),np.repeat(x[:n],count,0)])
        rr=np.concatenate([r,np.repeat(r[:n],count,0),np.repeat(r[:n],count,0)]);cc=np.concatenate([rounding,np.repeat(rounding[:n],count,0),np.repeat(rounding[:n],count,0)])
        aa=np.concatenate([base,np.repeat(base[:n],count,0),np.repeat(base[:n],count,0)]);indices=np.concatenate([np.arange(len(r)),np.repeat(np.arange(n),count),np.repeat(np.arange(n),count)])
        offsets=np.concatenate([np.zeros((len(r)+n*count,p['B'].shape[1])),np.tile(sb,(n,1))]);eq,eb=self.delta(theta)
        dq,db=self.delta(direction) if direction is not None else (None,None);tq,tb=self.delta(trial) if trial is not None else (None,None)
        res,der,fin,real_before,real_after=[],[],[],[],[]
        for first in range(0,len(xx),128):
            sl=slice(first,min(first+128,len(xx)));xz,kz,vz=xx[sl],keys[indices[sl]],values[indices[sl]]
            def forward(qb,bb):
                a,prob=self.attention(xz,kz,vz,qb);rr1=rr[sl]+(a-aa[sl])@p['O'].T;zz=self.hidden(rr1)+offsets[sl]
                out=cc[sl]+rr1+zz@(p['B']+bb).T+p['b'];return out,rr1,zz,prob
            zero=np.zeros_like(eq);zb=np.zeros_like(eb)
            h0,_,_,_=forward(zero,zb);h1,r1,z1,prob=forward(eq,eb)
            before=features(h0,self.visual)@p['head'].T;after=features(h1,self.visual)@p['head'].T
            winner=before.argmax(1);rows=np.arange(len(before));gap=before[rows,winner,None]-before;gap[rows,winner]=1.
            ds=after-before;rho=(ds[rows,winner,None]-ds)/np.maximum(gap,1e-12);rho[rows,winner]=0.;res.append(rho.flatten())
            if first<len(r):
                length=min(len(r)-first,len(before));real_before.append(before[:length].argmax(1));real_after.append(after[:length].argmax(1))
            if direction is not None:
                query_derivative=(xz@dq.T).reshape(len(xz),p['heads'],p['head_dim']);dl=np.einsum('nhd,nhsd->nhs',query_derivative,kz)/p['head_dim']**.5
                da=prob*(dl-np.sum(prob*dl,-1,keepdims=True));dv=np.einsum('nhs,nhsd->nhd',da,vz).reshape(len(xz),-1);dr=dv@p['O'].T
                pre=ln(r1,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a'];prob_g=expit(1.702*pre)
                dz=(prob_g+1.702*pre*prob_g*(1-prob_g))*(ln_jvp(r1,dr,p['gamma2'],p['eps2'])@p['A'].T)
                dh=dr+dz@(p['B']+eb).T+z1@db.T
                t=ln(h1,p['gamma'],p['beta'],p['eps'])@p['projection'];length=np.linalg.norm(t,axis=1,keepdims=True);f=t/length
                dt=ln_jvp(h1,dh,p['gamma'],p['eps'])@p['projection'];df=(dt-f*np.sum(f*dt,1,keepdims=True))/length;ds=df@p['head'].T
                j=(ds[rows,winner,None]-ds)/np.maximum(gap,1e-12);j[rows,winner]=0.;der.append(j.flatten())
            if trial is not None:
                ht,_,_,_=forward(tq,tb);changed=features(ht,self.visual)@p['head'].T-before
                f=(changed[rows,winner,None]-changed)/np.maximum(gap,1e-12);f[rows,winner]=0.;fin.append(f.flatten())
        return np.concatenate(res),np.concatenate(der) if der else None,np.concatenate(fin) if fin else None,(np.concatenate(real_before),np.concatenate(real_after))


def verify(output):
    out=Path(output);r=json.loads((out/'direction.json').read_text());meta=json.loads((out/'mechanism.json').read_text());assert digest(out/'direction.pt')==r['direction_sha256']
    saved=torch.load(out/'direction.pt',map_location='cpu');cfg=saved['cfg'];assert digest(out/'calibration.pt')==meta['calibration_sha256'];cal=torch.load(out/'calibration.pt',map_location='cpu')
    space=torch.load(out/'readout.pt',map_location='cpu');anchor,expert,head,hashes=checked_models(r['target']);assert hashes==r['checkpoint_sha256']
    raw=query_parameters(expert,head);p={k:array(v) if torch.is_tensor(v) else v for k,v in raw.items()};rec=Reconstruction(cal,cfg,p,space);rec.visual=expert.model.visual
    original=query_parameters(anchor,head);np.testing.assert_allclose((p['Q']-array(original['Q']))@rec.v.T,rec.u*array(cal['basis_Q']['s']),atol=1e-9,rtol=1e-7)
    theta,direction=[array(saved[k]) for k in ['theta','direction']];actual,ridge=rec.conditional(array(saved['conditional_A']));assert abs(ridge-saved['ridge'])<1e-9
    coefficient_error=float(np.max(np.abs(actual-theta)));assert coefficient_error<1e-7
    step=1e-5;co=array(saved['conditional_A']);dc=array(saved['conditional_direction_A'])
    plus,_=rec.conditional(co+step*dc);minus,_=rec.conditional(co-step*dc);fd=(plus-minus)/(2*step)
    derivative_error=float(np.linalg.norm(fd-direction)/np.linalg.norm(direction));assert derivative_error<1e-3
    i=next(j for j,row in enumerate(r['finite_steps']) if row['feasible']);trial=array(saved['finite_thetas'][i])
    rr,jj,ff=[],[],[]
    for group in ['native','scale']:
        rho,j,finite,_=rec.group(group,theta,direction,trial);rr.append(rho);jj.append(j);ff.append(finite)
    residual=np.concatenate(rr);jvp=np.concatenate(jj);np.testing.assert_allclose(residual,array(saved['native_residual']),atol=1e-7,rtol=1e-7);np.testing.assert_allclose(jvp,array(saved['native_directional_derivative']),atol=1e-7,rtol=1e-7)
    finite_max=float(np.max(np.abs(np.concatenate(ff))));assert abs(finite_max-r['finite_steps'][i]['native_critic'])<1e-7
    result={'status':'passed','source_sha256':digest(__file__),'direction_sha256':r['direction_sha256'],'calibration_sha256':meta['calibration_sha256'],'conditional_coefficient_maximum_error':coefficient_error,'conditional_JVP_finite_difference_relative_error':derivative_error,'native_JVP_maximum_error':float(np.max(np.abs(jvp-array(saved['native_directional_derivative'])))),'native_residual_maximum_error':float(np.max(np.abs(residual-array(saved['native_residual'])))),'finite_critic':finite_max,'new_release_weights':0,'new_model_forwards':0,'future_partner_used':False,'test_used':False}
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


def verify_game(output):
    out=Path(output);r=json.loads((out/'results.json').read_text());assert digest(out/'candidate.pt')==r['candidate_sha256']
    cal=torch.load(out/'candidate.pt',map_location='cpu');cfg=cal['protocol'];space=torch.load(out/'readout.pt',map_location='cpu')
    _,expert,head,hashes=checked_models(r['target']);assert hashes==r['checkpoint_sha256']
    raw=query_parameters(expert,head);p={k:array(v) if torch.is_tensor(v) else v for k,v in raw.items()};rec=Reconstruction(cal,cfg,p,space);rec.visual=expert.model.visual
    theta=array(cal['theta']);eq,eb=rec.delta(theta)
    for site,delta in [('Q',eq),('B',eb)]:np.testing.assert_allclose(delta,array(cal['release_deltas'][site]),atol=1e-8,rtol=1e-7)
    reconstructed,_=rec.conditional(theta[:rec.size]);error=float(np.max(np.abs(reconstructed-theta)));assert error<1e-7
    distance=float(np.sqrt(np.linalg.norm(eq)**2/rec.nq**2+np.linalg.norm(eb)**2/rec.nb**2));assert distance<=cfg['parameter_budget']*(1+1e-10)
    constraints={};confirmation=None;routing=[]
    for group in ['native','scale','confirmation']:
        rho,_,_,pred=rec.group(group,theta);maximum=float(np.max(np.abs(rho)))
        row,competitor=np.unravel_index(np.abs(rho).argmax(),(len(rho)//len(p['head']),len(p['head'])))
        real_n=len(pred[0]);span=2*cfg['probe_rank']*cfg['probe_contexts']
        domain='natural_input' if row<real_n else 'Query_input_probe' if row<real_n+span else 'B_input_probe'
        local=row if row<real_n else (row-real_n if row<real_n+span else row-real_n-span)//(2*cfg['probe_rank'])
        constraints[group]={'maximum_relative_margin_drift':maximum,'real_n':real_n,'changed_predictions':int((pred[0]!=pred[1]).sum()),'worst_domain':domain,'worst_context_offset':int(local),'worst_competitor_class':int(competitor),'signed_worst_relative_margin_change':float(rho.reshape(-1,len(p['head']))[row,competitor])}
        if group!='confirmation':assert maximum<=cfg['margin_fraction']*(1+1e-7)
        else:confirmation=pred
        cache=cal[group];x,k,v=[array(cache[key]) for key in ['q_input','keys','values']]
        _,old=rec.attention(x,k,v);_,new=rec.attention(x,k,v,eq)
        kl=np.sum(old*(np.log(np.maximum(old,1e-300))-np.log(np.maximum(new,1e-300))),axis=-1)
        routing.append({'group':group,'mean_attention_KL':float(kl.mean()),'maximum_attention_KL':float(kl.max()),'maximum_probability_change':float(np.max(np.abs(new-old)))})
    rho=constraints['confirmation']['maximum_relative_margin_drift'];assert abs(rho-r['confirmation']['worst_margin_drift'])<1e-7
    assert (rho<=cfg['margin_fraction'])==r['confirmation_passed']
    before,after=confirmation;np.testing.assert_array_equal(before,array(cal['confirmation_original_predictions']));np.testing.assert_array_equal(after,array(cal['confirmation_candidate_predictions']))
    ids=cal['confirmation_indices'];labels=official_labels(r['target'],True)[ids];np.testing.assert_array_equal(labels,array(cal['confirmation_labels']))
    for name,pred in [('original',before),('candidate',after)]:assert r['confirmation'][name+'_correct']==int((pred==labels).sum())
    old=set(i for group in cal['indices'].values() for i in group);parent=torch.load(Path(cfg['margin_reference'])/(r['target']+'_diagnostic.pt'),map_location='cpu');old.update(parent['indices']['confirmation'])
    assert len(set(ids))==len(ids) and not old.intersection(ids)
    block=expert.model.visual.transformer.resblocks[11];d=block.attn.embed_dim
    wk=block.attn.in_proj_weight[d:2*d].detach().cpu().double().numpy().reshape(p['heads'],p['head_dim'],d)
    wq=p['Q'].reshape(p['heads'],p['head_dim'],d);qe=eq.reshape(p['heads'],p['head_dim'],d)
    invariant=float(np.sqrt(sum(np.linalg.norm(qe[j].T@wk[j])**2 for j in range(p['heads'])))/np.sqrt(sum(np.linalg.norm(wq[j].T@wk[j])**2 for j in range(p['heads']))))
    replays=[]
    if rho<=cfg['margin_fraction']:
        released=copy.deepcopy(expert);newblock=released.model.visual.transformer.resblocks[11]
        with torch.no_grad():
            newblock.attn.in_proj_weight[:d].copy_((block.attn.in_proj_weight[:d].double()+torch.from_numpy(eq)).float())
            newblock.mlp.c_proj.weight.copy_((block.mlp.c_proj.weight.double()+torch.from_numpy(eb)).float())
        training=dataset(r['target'],'data',expert.val_preprocess,True)
        for start in range(0,min(128,len(ids)),32):
            images=torch.stack([training[i][0] for i in ids[start:start+32]]).cuda()
            for name,model,pred in [('original',expert,before),('candidate',released,after)]:
                with torch.no_grad():model.cuda();head.cuda();actual=head(model(images)).argmax(1).cpu().numpy();model.cpu();head.cpu()
                np.testing.assert_array_equal(actual,pred[start:start+len(images)]);replays.append({'model':name,'start':start,'n':len(images),'prediction_match':True})
    result={'status':'passed' if rho<=cfg['margin_fraction'] else 'confirmed_constraint_failure','source_sha256':digest(__file__),'candidate_sha256':r['candidate_sha256'],'conditional_coefficient_maximum_error':error,'parameter_distance':distance,'Q_relative_distance':float(np.linalg.norm(eq)/rec.nq),'B_relative_distance':float(np.linalg.norm(eb)/rec.nb),'task_constraints':constraints,'attention_routing':routing,'QK_bilinear_relative_change':invariant,'fresh_replays':replays,'new_model_forwards':sum(x['n'] for x in replays),'new_release_weights':0,'future_partner_used':False,'test_used':False,'scope':'Task preservation and representation-change evidence; not true fusion or AGA protection'}
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();torch.set_num_threads(4)
    if (Path(args.output)/'results.json').exists():verify_game(args.output)
    else:verify(args.output)
