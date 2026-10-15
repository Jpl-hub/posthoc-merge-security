"""NumPy/eigen/row-solve verification of the sole output-null bias diagnostic.

Torch is used for loading frozen tensors and replaying the original SVD frame
orientation, whose geometry/derivatives are checked independently in NumPy.
No publication parameters are selected, changed, repaired or optimized.
"""
import argparse,json,os,time,math
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from scipy.special import expit
from clip_benchmark import digest
from cross_module_distance import extended_parameters
from probe_distance_candidate import on_device
from subspace_probe import save_json
from task_boundary_audit import raw_jacobians
from task_probe_distance import checked_models,initialize
from verify_cascade import suffix
from verify_cross_module import ln
from verify_downstream_first import convert,jacobian
from verify_levelset_audits import raw_scores,residual


def phi(z):return z*expit(1.702*z)
def dp(z):
    s=expit(1.702*z);return s+1.702*z*s*(1-s)
def readout(p):
    L=p['gamma'][:,None]*p['projection'];L-=L.mean(0);c=p['beta']@p['projection'];return L,c
def score_stats(before,after):
    y=before.argmax(1);rows=np.arange(len(y));mask=np.ones_like(before,dtype=bool);mask[rows,y]=False
    ss=before-before.mean(1,keepdims=True);aa=after-after.mean(1,keepdims=True);alpha=np.sum(ss*aa,1)/np.sum(ss*ss,1);r=aa-alpha[:,None]*ss
    gap=before[rows,y,None]-before;adverse=r-r[rows,y,None]-alpha[:,None]*gap
    np.testing.assert_array_equal(np.max(np.where(mask,adverse,-np.inf),1)>0,after.argmax(1)!=y)
    return dict(changed=int((after.argmax(1)!=y).sum()),worst=float(np.abs(residual(before,after)).max()),
        excess=float(np.where(mask,adverse,-np.inf).max()),identity=float(np.abs(after[rows,y,None]-after-(alpha[:,None]*gap+r[rows,y,None]-r)).max()))
def bound(h,p,F,rho,constants):
    L,c=readout(p);width=len(h);sd=np.sqrt(np.mean((h-h.mean())**2)+p['eps']);tn=np.linalg.norm(h@L+sd*c);ql,bn,fn=constants
    m1=1.5*bn*fn;m2=1.5*bn*fn**2;ls=sd-m1*rho/np.sqrt(width);lt=tn-ql*m1*rho
    if min(ls,lt)<=0:return None
    wn=np.linalg.norm(p['head'][:,None]-p['head'][None],axis=-1);g=wn*ql/lt;hess=wn*(6*ql**2/lt**2+np.linalg.norm(c)/(width*ls*lt))
    return hess*m1*m1+g*m2


def verify(cfg):
    start=time.monotonic();out=Path(cfg['output']);rep=json.loads((out/'results.json').read_text());assert digest(out/'diagnostic.pt')==rep['diagnostic_sha256']
    for name,pin in cfg['inputs'].items():assert digest(name)==pin,name
    caltorch=torch.load(cfg['calibration_reference'],map_location='cpu');cal=convert(caltorch);data=convert(torch.load(out/'diagnostic.pt',map_location='cpu'));env=convert(torch.load(cfg['environment_cache'],map_location='cpu'))
    anchor,expert,head,pins=checked_models(cfg['target']);assert pins==rep['checkpoint_sha256'];p=cal['parameters'];base=convert(extended_parameters(anchor,head))
    for k,v in extended_parameters(expert,head).items():
        if torch.is_tensor(v):np.testing.assert_array_equal(v.numpy(),p[k])
    # Independent kernel projector from positive-definite row Gram solve.
    B=p['B'];gram=B@B.T;np.linalg.cholesky(gram)
    contrast=p['head']-p['head'].mean(0);M=p['gamma'][:,None]*(p['projection']@contrast.T);M-=M.mean(0)
    q,sv,_=np.linalg.svd(M,full_matrices=False);q=q[:,:int((sv>sv[0]*1e-10).sum())]
    g=q.T@(B-base['B']);g-=np.linalg.solve(gram,B@g.T).T@B
    eig,u=np.linalg.eigh(g@g.T);vec=g.T@u[:,-1]/np.sqrt(eig[-1]);vec*=1 if vec[np.argmax(np.abs(vec))]>0 else -1
    np.testing.assert_allclose(vec,data['direction'],atol=1e-9,rtol=1e-8)
    expected=cfg['parameter_budget']*(1-cfg['export_reserve'])*np.linalg.norm(p['a'])*vec
    np.testing.assert_allclose(expected,data['delta_exact'],atol=1e-8,rtol=1e-7)
    # Quantization is verified from the frozen exact delta, not a reselected vector.
    d=(p['a']+data['delta_exact']).astype(np.float32).astype(np.float64)-p['a'];np.testing.assert_array_equal(d,data['delta_FP32'])
    distance=float(np.linalg.norm(d)/np.linalg.norm(p['a']));assert distance<=.5 and abs(distance-rep['parameter_distance'])<1e-12
    # Source ordering: same predeclared classwise low-gap rule, no release response.
    raw=cal['cache'];ss=raw_scores(raw['h'],p);y=ss.argmax(1);tmp=ss.copy();tmp[np.arange(len(y)),y]=-np.inf;gap=ss[np.arange(len(y)),y]-tmp.max(1);core=[]
    for c in range(len(p['head'])):
        members=np.flatnonzero(y==c);core.extend(members[np.argsort(gap[members],kind='stable')[:cfg['contexts_per_class']]])
    order=core+[i for i in range(len(y)) if i not in set(core)];assert [cal['indices'][i] for i in order]==data['source_indices']
    raw={k:v[order] for k,v in raw.items()};x=ln(raw['r'],p['gamma2'],p['beta2'],p['eps2']);pre=x@p['A'].T+p['a'];z=phi(pre)
    rounding=raw['h']-raw['r']-z@B.T-p['b'];ncore=len(core);rank=cfg['probe_rank'];vo=cal['bases']['O']['vh'];vb=cal['bases']['B']['vh']
    # Frozen historical task-probe code multiplies Python.01 by float32 signs
    # before multiplying double RMS; reproduce that actual arithmetic order.
    scale=float(np.float32(cfg['finite_step']));shifts=[]
    for inputs,v in [(raw['o_input'],vo),(z,vb)]:
        rms=np.sqrt(np.mean((inputs@v.T)**2,axis=0));shifts.append((scale*np.array([-1.,1.])[:,None,None]*rms[None,:,None]*v[None]).reshape(2*rank,-1))
    so,sb=shifts;count=len(so);r=np.r_[raw['r'],(raw['r'][:ncore,None]+(so@p['O'].T)[None]).reshape(-1,768),np.repeat(raw['r'][:ncore],count,0)]
    ro=np.r_[rounding,np.repeat(rounding[:ncore],count,0),np.repeat(rounding[:ncore],count,0)]
    offset=np.r_[np.zeros((len(z)+ncore*count,len(p['a']))),np.tile(sb,(ncore,1))]
    zpre=ln(r,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a'];h=ro+r+(phi(zpre)+offset)@B.T+p['b'];source_before=raw_scores(h,p);source_after=raw_scores(h+(phi(zpre+d)-phi(zpre))@B.T,p)
    errors=[np.abs(source_before-data['source']['before']).max(),np.abs(source_after-data['source']['after']).max()];stats=score_stats(source_before,source_after)
    assert stats['changed']==rep['source']['changed'] and abs(stats['worst']-rep['source']['worst_task_drift'])<1e-8
    pe=env['proxy_parameters'];he,z_env=suffix(env['proxy']['H'].astype(np.float64),pe)
    r_env=he-z_env@pe['B'].T-pe['b'];he+=env['proxy']['rounding']
    ze=ln(r_env,pe['gamma2'],pe['beta2'],pe['eps2'])@pe['A'].T+pe['a'];de=cfg['coefficient']*d
    linear=.5*de@pe['B'].T;even=phi(ze+de)-.5*(ze+de)-phi(ze)+.5*ze;dhe=linear+even@pe['B'].T
    envbefore=raw_scores(he,pe);envafter=raw_scores(he+dhe,pe);errors +=[np.abs(envbefore-data['environment']['before']).max(),np.abs(envafter-data['environment']['after']).max()]
    es=score_stats(envbefore,envafter);assert es['changed']==rep['environment']['changed'] and abs(es['worst']-rep['environment']['worst_task_drift'])<1e-8
    # Recount every class from independent finite scores.
    for before,after,want in [(source_before,source_after,rep['source']),(envbefore,envafter,rep['environment'])]:
        for c in want['per_class']:
            yy=before.argmax(1);changed=after.argmax(1)!=yy;assert int((yy==c['label']).sum())==c['n'] and int((changed&(yy==c['label'])).sum())==c['changed']
    # Check the dual without calling the LP solver a second time.
    Le,ce=readout(pe);Me=Le@(pe['head']-pe['head'].mean(0)).T;be=ce@(pe['head']-pe['head'].mean(0)).T
    sigma=np.sqrt(np.mean((he-he.mean(1,keepdims=True))**2,1)+pe['eps']);numerator=he@Me+sigma[:,None]*be
    yy=envbefore.argmax(1);rows=np.arange(len(yy));other=envbefore.copy();other[rows,yy]=-np.inf;jj=other.argmax(1)
    coeff=-cfg['coefficient']*.5*((Me[:,yy]-Me[:,jj]).T@pe['B'])/(numerator[rows,yy]-numerator[rows,jj])[:,None]
    coeff-=np.linalg.solve(gram,B@coeff.T).T@B;classes=np.stack([coeff[yy==c].mean(0) for c in range(len(p['head']))])
    np.testing.assert_allclose(classes,data['class_rows'],atol=1e-10,rtol=1e-8);pi=data['dual_weights'];assert pi.min()>=-1e-14 and abs(pi.sum()-1)<1e-12
    dualupper=cfg['parameter_budget']*np.linalg.norm(p['a'])*np.linalg.norm(pi@classes);assert dualupper<1e-10
    # Check actual task-matrix SVD and its complete functional normal/tangent frame.
    va=data['basis_A'];np.testing.assert_allclose(va@va.T,np.eye(rank),atol=1e-10);da=p['A']-base['A'];ev,V=np.linalg.eigh(da.T@da)
    np.testing.assert_allclose(va.T@va,V[:,-rank:]@V[:,-rank:].T,atol=1e-8,rtol=1e-7)
    np.testing.assert_allclose(data['rms_A'],np.sqrt(np.mean((x@va.T)**2,0)),atol=1e-10,rtol=1e-10);F=(p['A']@va.T)*data['rms_A'][None]
    L,c=readout(p);constants=(np.sqrt(np.linalg.eigvalsh(L.T@L)[-1])+np.linalg.norm(c)/np.sqrt(768),np.sqrt(np.linalg.eigvalsh(gram)[-1]),np.sqrt(np.linalg.eigvalsh(F.T@F)[-1]))
    pt=on_device(caltorch['parameters']);neighborhood=[];witnesses=[];maxj=0.;maxfd=0.;rho=cfg['neighborhood_radius']
    for i in range(ncore):
        hi=raw['h'][i:i+1];jh,before=jacobian(hi,p);ju=((jh[0]@B)*dp(pre[i])[None])@F;y=int(before.argmax())
        jht,_,_,_=raw_jacobians(dict(r=torch.tensor(raw['r'][i:i+1],device='cuda'),h=torch.tensor(hi,device='cuda')),pt)
        jut=((jht[0]@pt['B'])*torch.tensor(dp(pre[i]),device='cuda')[None])@torch.tensor(F,device='cuda')
        maxj=max(maxj,float(np.abs(jut.cpu().numpy()-ju).max()));assert maxj<1e-8
        normal=jut[y,None]-jut;normal=normal[torch.arange(len(normal),device='cuda')!=y]
        _,svt,frame=torch.linalg.svd(normal,full_matrices=True);frame=frame.cpu().numpy();nrank=data['neighborhood'][i]['normal_rank']
        normalnp=ju[y,None]-ju;normalnp=normalnp[np.arange(len(ju))!=y]
        np.testing.assert_allclose(frame@frame.T,np.eye(rank),atol=1e-10);assert np.max(np.abs(normalnp@frame[nrank:].T))<1e-8
        shift=rho*np.r_[frame,-frame];ps=shift@F.T
        hs=hi+(phi(pre[i]+ps)-phi(pre[i]))@B.T;hr=hi+(phi(pre[i]+ps+d)-phi(pre[i]))@B.T
        bs,rs=raw_scores(hs,p),raw_scores(hr,p);st=score_stats(bs,rs);want=data['neighborhood'][i]
        assert st['changed']==want['changed'] and abs(st['worst']-want['worst_task_drift'])<1e-8
        hrel=hi+(phi(pre[i]+d)-phi(pre[i]))@B.T;jr,_=jacobian(hrel,p);jur=((jr[0]@B)*dp(pre[i]+d)[None])@F
        jg=ju[y,None]-ju;je=jur[y,None]-jur-jg;refbound=bound(hi[0],p,F,rho,constants);relbound=bound(hrel[0],p,F,rho,constants)
        v=data['bounds'][i];cert=False
        if refbound is not None and relbound is not None:
            score=raw_scores(hi,p)[0];snew=raw_scores(hrel,p)[0];g0=score[y]-score;e0=snew[y]-snew-g0;mask=np.arange(len(g0))!=y
            lower=g0-rho*np.linalg.norm(jg,axis=1)-.5*rho*rho*refbound[y];upper=np.abs(e0)+rho*np.linalg.norm(je,axis=1)+.5*rho*rho*(refbound[y]+relbound[y])
            cert=bool((lower[mask]>0).all() and (upper[mask]<=.5*lower[mask]).all())
            assert v['denominator_nonzero_proved'];assert abs(float(lower[mask].min())-v['minimum_teacher_gap_lower'])<1e-7
        else:assert not v['denominator_nonzero_proved']
        assert cert==v['certified'];eps=cfg['derivative_step'];qq=frame[0];hp=hi+(phi(pre[i]+eps*(F@qq))-phi(pre[i]))@B.T;hm=hi+(phi(pre[i]-eps*(F@qq))-phi(pre[i]))@B.T
        maxfd=max(maxfd,float(np.abs((raw_scores(hp,p)-raw_scores(hm,p))[0]/(2*eps)-ju@qq).max()));assert maxfd<cfg['derivative_tolerance']
        neighborhood.append(dict(source_position=i,normal_rank=nrank,tangent_dimension=rank-nrank,changed=st['changed'],worst=st['worst'],certified=cert))
        witnesses.append(dict(frame=frame,before=bs,after=rs))
    assert max(errors)<1e-8
    expected_signal=(stats['worst']<=.5 and max(v['worst'] for v in neighborhood)<=.5 and stats['changed']==0 and es['changed']>=cfg['minimum_environment_changes'] and rep['environment']['classes_changed']>=cfg['minimum_environment_classes'])
    assert expected_signal==rep['mechanism_supported']
    torch.save(dict(neighborhood=witnesses,source_sha256=digest(__file__)),out/'neighborhood_replay.pt')
    result=dict(status='passed_closed_mechanism' if not expected_signal else 'passed_mechanism',mechanism_supported=expected_signal,
        source=stats,environment=es,neighborhood=neighborhood,maximum_numpy_score_error=float(max(errors)),maximum_independent_Jacobian_error=maxj,maximum_directional_fd_error=maxfd,
        linear_class_dual_upper=float(dualupper),dual_verified_without_resolving=True,parameter_distance=distance,
        kernel_solver='independent row-Gram positive-definite solve, then9x9 eigenproblem; no Torch output-matrix SVD',
        frame_orientation='Original Torch SVD orientation replayed as input data; its full orthogonality, normal/tangent geometry and responses independently checked with NumPy',
        diagnostic_sha256=rep['diagnostic_sha256'],source_sha256=digest(__file__),neighborhood_replay_sha256=digest(out/'neighborhood_replay.pt'),
        new_release_weights=0,new_model_forwards=0,new_parameter_updates=0,peer_loaded=False,test_loaded=False,confirmation_loaded=False,wall_seconds=time.monotonic()-start)
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--protocol',required=True);args=a.parse_args();initialize();verify(json.loads(Path(args.protocol).read_text()))
