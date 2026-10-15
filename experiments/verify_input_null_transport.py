"""Independent NumPy finite reconstruction and different class-game solver."""
import argparse,json,os,time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from scipy.optimize import minimize
from scipy.special import expit
from clip_benchmark import digest
from cross_module_distance import extended_parameters
from subspace_probe import save_json
from task_probe_distance import checked_models,initialize
from verify_cascade import suffix
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_downstream_first import convert,jacobian
from verify_levelset_audits import raw_scores,residual


def phi(z):return z*expit(1.702*z)
def slope(z):
    s=expit(1.702*z);return s+1.702*z*s*(1-s)
def readout(p):
    contrast=p['head']-p['head'].mean(0);M=p['gamma'][:,None]*(p['projection']@contrast.T);M-=M.mean(0);b=p['beta']@p['projection']@contrast.T
    return M,b
def stats(a,b):
    y=a.argmax(1);n=len(y);rows=np.arange(n);mask=np.ones_like(a,dtype=bool);mask[rows,y]=False
    aa=a-a.mean(1,keepdims=True);bb=b-b.mean(1,keepdims=True);scale=np.sum(aa*bb,1)/np.sum(aa*aa,1);r=bb-scale[:,None]*aa
    g=a[rows,y,None]-a;excess=r-r[rows,y,None]-scale[:,None]*g;changed=b.argmax(1)!=y
    np.testing.assert_array_equal(np.max(np.where(mask,excess,-np.inf),1)>0,changed)
    return dict(n=n,changed=int(changed.sum()),worst=float(np.abs(residual(a,b)).max()),classes_changed=sum(bool((changed&(y==c)).any()) for c in range(a.shape[1])))
def verify_stats(a,b,w):
    v=stats(a,b);assert v['changed']==w['changed'] and abs(v['worst']-w['worst_task_drift'])<1e-8
    for row in w['per_class']:
        y=a.argmax(1);changed=b.argmax(1)!=y;assert int((y==row['label']).sum())==row['n'] and int((changed&(y==row['label'])).sum())==row['changed']
    return v
def finite(x,pre,h,p,E,factor=1.):return raw_scores(h+(phi(pre+factor*x@E.T)-phi(pre))@p['B'].T,p)


def verify(cfg):
    start=time.monotonic();out=Path(cfg['output']);rep=json.loads((out/'results.json').read_text());assert digest(out/'diagnostic.pt')==rep['diagnostic_sha256']
    for f,pin in cfg['inputs'].items():assert digest(f)==pin,f
    d=convert(torch.load(out/'diagnostic.pt',map_location='cpu'));cal=convert(torch.load(cfg['calibration_reference'],map_location='cpu'));env=convert(torch.load(cfg['environment_cache'],map_location='cpu'))
    anchor,expert,head,pins=checked_models(cfg['target']);assert pins==rep['checkpoint_sha256'];p=cal['parameters'];p0=convert(extended_parameters(anchor,head))
    for k,v in extended_parameters(expert,head).items():
        if torch.is_tensor(v):np.testing.assert_array_equal(v.numpy(),p[k])
    ss=raw_scores(cal['cache']['h'],p);y=ss.argmax(1);tmp=ss.copy();tmp[np.arange(len(y)),y]=-np.inf;gap=ss[np.arange(len(y)),y]-tmp.max(1);core=[]
    for c in range(len(p['head'])):
        ix=np.flatnonzero(y==c);core.extend(ix[np.argsort(gap[ix],kind='stable')[:cfg['contexts_per_class']]])
    order=core+[i for i in range(len(y)) if i not in set(core)];ids=[cal['indices'][i] for i in order];assert ids==d['source_indices']
    raw={k:v[order] for k,v in cal['cache'].items()};ss=ss[order];yy=ss.argmax(1);rows=np.arange(len(yy));ncore=len(core)
    x=ln(raw['r'],p['gamma2'],p['beta2'],p['eps2']);z=x@p['A'].T+p['a'];J,_=jacobian(raw['h'][:cfg['selection_n']],p);K=(J@p['B'])*slope(z[:cfg['selection_n']])[:,None]
    # Independent eigensystem, functional-mode ranking, and kernel constraint rows.
    da=p['A']-p0['A'];ev,va=np.linalg.eigh(da.T@da);sa=np.sqrt(np.maximum(ev[::-1],0));va=va[:,::-1].T;ua=(da@va.T)/sa[None]
    response=(K@ua)*(x[:len(K)]@va.T)[:,None,:]*sa[None,None,:];sy=yy[:len(K)];rr=np.arange(len(K));g=ss[:len(K)][rr,sy,None]-ss[:len(K)];g[rr,sy]=1
    rel=(response[rr,sy,None]-response)/g[:,:,None];rel[rr,sy]=0;counts=np.bincount(sy,minlength=len(p['head']));weights=1/counts[sy]/np.count_nonzero(counts)
    imp=np.sum(np.sum(rel*rel,1)*weights[:,None],0);chosen=np.argsort(-imp,kind='stable')[:cfg['probe_rank']];np.testing.assert_array_equal(chosen,d['selection_modes']);np.testing.assert_allclose(imp,d['selection_importance'],atol=1e-7,rtol=1e-7)
    U,V,C=d['U'],d['V'],d['C'];np.testing.assert_allclose(V.T@V,va[chosen].T@va[chosen],atol=1e-8,rtol=1e-8);np.testing.assert_allclose(U.T@U,np.eye(U.shape[1]),atol=1e-10)
    normals=K[:ncore][np.arange(ncore),yy[:ncore],None]-K[:ncore];normals=normals[np.arange(len(p['head']))[None]!=yy[:ncore,None]]
    W=np.r_[p['B']/np.linalg.norm(p['B'],axis=1,keepdims=True),normals/np.linalg.norm(normals,axis=1,keepdims=True)];np.testing.assert_allclose(W,d['W_rows'],atol=1e-10,rtol=1e-8)
    _,sw,R=np.linalg.svd(W,full_matrices=False);rank=int((sw>np.finfo(float).eps*max(W.shape)*sw[0]).sum());assert rank==rep['kernel_constraint_rank'];R=R[:rank]
    M,_=readout(p);Q,sv,_=np.linalg.svd(M,full_matrices=False);Q=Q[:,:np.count_nonzero(sv>sv[0]*1e-10)];G=Q.T@(p['B']-p0['B']);G-=G@R.T@R
    assert np.abs(W@U).max()<1e-9 and np.linalg.norm(G-G@U@U.T)<1e-9
    np.testing.assert_allclose(d['unit'],np.linalg.norm(p['A'])*U@C@V,atol=1e-9,rtol=1e-8);assert abs(np.linalg.norm(d['unit'])/np.linalg.norm(p['A'])-1)<1e-9
    # Independently reconstruct historical finite task probes.
    rounding=raw['h']-raw['r']-phi(z)@p['B'].T-p['b'];scale=float(np.float32(cfg['finite_step']));shifts=[]
    for incoming,v in [(raw['o_input'],cal['bases']['O']['vh']),(phi(z),cal['bases']['B']['vh'])]:
        rms=np.sqrt(np.mean((incoming@v.T)**2,0));shifts.append((scale*np.array([-1.,1.])[:,None,None]*rms[None,:,None]*v[None]).reshape(2*cfg['probe_rank'],-1))
    so,sb=shifts;count=len(so);r=np.r_[raw['r'],(raw['r'][:ncore,None]+(so@p['O'].T)[None]).reshape(-1,768),np.repeat(raw['r'][:ncore],count,0)]
    ro=np.r_[rounding,np.repeat(rounding[:ncore],count,0),np.repeat(rounding[:ncore],count,0)];offset=np.r_[np.zeros((len(z)+ncore*count,len(p['a']))),np.tile(sb,(ncore,1))]
    sx=ln(r,p['gamma2'],p['beta2'],p['eps2']);sz=sx@p['A'].T+p['a'];h=ro+r+(phi(sz)+offset)@p['B'].T+p['b'];before=raw_scores(h,p)
    rms=np.sqrt(np.mean((x@V.T)**2,0));np.testing.assert_allclose(rms,d['rms'],atol=1e-10);F=p['A']@V.T*rms[None];nx=[];nh=[];maxfd=0.
    for i,item in enumerate(d['frames']):
        frame=item['frame'];np.testing.assert_allclose(frame@frame.T,np.eye(len(frame)),atol=1e-10);ju=K[i]@F;normal=ju[yy[i],None]-ju;normal=normal[np.arange(len(ju))!=yy[i]]
        assert np.abs(normal@frame[item['normal_rank']:].T).max()<1e-8
        q=cfg['neighborhood_radius']*np.r_[frame,-frame];nx.append(x[i]+(q*rms)@V);nh.append(raw['h'][i]+(phi(z[i]+q@F.T)-phi(z[i]))@p['B'].T)
        step=cfg['derivative_step'];dq=F@frame[0];hp=raw['h'][i]+(phi(z[i]+step*dq)-phi(z[i]))@p['B'].T;hm=raw['h'][i]+(phi(z[i]-step*dq)-phi(z[i]))@p['B'].T
        maxfd=max(maxfd,float(np.abs((raw_scores(hp[None],p)-raw_scores(hm[None],p))[0]/(2*step)-ju@frame[0]).max()))
    assert maxfd<cfg['derivative_tolerance'];nx=np.concatenate(nx);nh=np.concatenate(nh);nz=nx@p['A'].T+p['a'];nb=raw_scores(nh,p);ray=[];errors=[]
    for i,trial in enumerate(rep['source_ray']):
        assert trial['alpha']==cfg['source_ray'][i];E=(p['A']+trial['alpha']*d['unit']).astype(np.float32).astype(np.float64)-p['A'];distance=np.linalg.norm(E)/np.linalg.norm(p['A']);assert distance<=.5
        after=finite(sx,sz,h,p,E);na=finite(nx,nz,nh,p,E);sr=verify_stats(before,after,trial['source']);nr=verify_stats(nb,na,trial['neighborhood']);ok=sr['worst']<=cfg['source_reserve'] and nr['worst']<=cfg['source_reserve'] and sr['changed']==nr['changed']==0
        assert bool(ok)==trial['source_feasible'];ray.append(dict(step=i,source=sr,neighborhood=nr,feasible=bool(ok)))
        if i<len(rep['source_ray'])-1:assert not ok
    np.testing.assert_array_equal(E,d['delta_FP32']);np.testing.assert_allclose(d['delta_exact'],trial['alpha']*d['unit'],atol=1e-12);assert abs(distance-rep['parameter_distance'])<1e-12
    for a,b in [(before,d['source']['before']),(after,d['source']['after']),(nb,d['neighborhood_before']),(na,d['neighborhood_after'])]:errors.append(np.abs(a-b).max())
    pe=env['proxy_parameters'];he,ze=suffix(env['proxy']['H'].astype(np.float64),pe);re=he-ze@pe['B'].T-pe['b'];he+=env['proxy']['rounding'];xe=ln(re,pe['gamma2'],pe['beta2'],pe['eps2']);pre=xe@pe['A'].T+pe['a'];eb=raw_scores(he,pe);ea=finite(xe,pre,he,pe,E,cfg['coefficient']);es=verify_stats(eb,ea,rep['environment']);errors.extend([np.abs(eb-d['environment']['before']).max(),np.abs(ea-d['environment']['after']).max()]);assert max(errors)<1e-8
    Me,be=readout(pe);sd=np.sqrt(np.mean((he-he.mean(1,keepdims=True))**2,1)+pe['eps']);N=he@Me+sd[:,None]*be;ey=eb.argmax(1);er=np.arange(len(ey));other=eb.copy();other[er,ey]=-np.inf;ej=other.argmax(1);den=N[er,ey]-N[er,ej]
    left=-.5*cfg['coefficient']*np.linalg.norm(p['A'])*((Me[:,ey]-Me[:,ej]).T@pe['B']@U)/den[:,None];lr=(left[:,:,None]*(xe@V.T)[:,None,:]).reshape(len(xe),-1);classes=np.stack([lr[ey==c].mean(0) for c in range(len(p['head']))]);np.testing.assert_allclose(classes,d['class_rows'],atol=1e-9,rtol=1e-8)
    H=classes@classes.T;H/=np.max(np.abs(H));sol=minimize(lambda pi:.5*pi@H@pi,np.ones(len(classes))/len(classes),jac=lambda pi:H@pi,bounds=[(0,1)]*len(classes),constraints=[dict(type='eq',fun=lambda pi:pi.sum()-1,jac=lambda pi:np.ones(len(pi)))],method='SLSQP',options=dict(maxiter=200,ftol=1e-13))
    assert sol.success;dual=np.linalg.norm(sol.x@classes);assert abs(dual-rep['class_game']['dual_value'])<1e-7
    pi=d['dual_weights'];assert pi.min()>=0 and abs(pi.sum()-1)<1e-12;optimal=(pi@classes)/np.linalg.norm(pi@classes);np.testing.assert_allclose(optimal,C.ravel(),atol=1e-8)
    # Post-freeze signed directional attribution. No alternative point is selected.
    Je,_=jacobian(he,pe);dpre=cfg['coefficient']*xe@d['unit'].T;dh=(slope(pre)*dpre)@pe['B'].T;linear=.5*dpre@pe['B'].T;even=dh-linear
    dhsd=np.mean((he-he.mean(1,keepdims=True))*dh,1)/sd
    components=dict(linear=linear@Me,even=even@Me,variance=dhsd[:,None]*be);signed={k:(v[er,ej]-v[er,ey])/den for k,v in components.items()};signed['joint']=sum(signed.values())
    perclass=[dict(label=c,**{k:float(v[ey==c].mean()) for k,v in signed.items()}) for c in range(len(p['head']))]
    linear_min=min(v['linear'] for v in perclass);assert abs(linear_min-rep['class_game']['primal_value'])<1e-8
    # Final labels are construction labels, not new independent/test accuracies.
    for a,b,indices,w in [(before[:len(ids)],after[:len(ids)],ids,rep['source']),(eb,ea,d['environment_indices'],rep['environment'])]:
        labels=official_labels(cfg['target'],True)[indices];assert int((a.argmax(1)==labels).sum())==w['original_correct'] and int((b.argmax(1)==labels).sum())==w['protected_correct']
    supported=bool(rep['accepted_source_step'] is not None and rep['class_game']['primal_value']>cfg['minimum_game_value'] and rep['environment']['lost']-rep['environment']['gained']>=cfg['minimum_net_losses'] and es['classes_changed']>=cfg['minimum_environment_classes']);assert supported==rep['mechanism_supported']
    result=dict(status='passed_closed_mechanism' if not supported else 'passed_mechanism_requires_fresh_confirmation',mechanism_supported=supported,maximum_numpy_score_error=float(max(errors)),maximum_source_frame_directional_fd_error=maxfd,
        independent_class_dual_value=float(dual),original_class_value=rep['class_game']['primal_value'],class_solver='SLSQP simplex QP; primary enumerated1023active faces',source_ray=ray,environment=es,parameter_distance=float(distance),kernel_constraint_rank=rank,
        fixed_direction_class_first_order=perclass,task_mode_ranking_independently_verified=True,confirmation_loaded=False,peer_loaded=False,test_loaded=False,new_parameter_updates=0,new_release_weights=0,
        diagnostic_sha256=rep['diagnostic_sha256'],source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,scope='Frozen point independently checked; derivative attribution is post-freeze only. No new model/direction/ray selection. Finite samples are not a continuous certificate.')
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--protocol',required=True);args=a.parse_args();initialize();verify(json.loads(Path(args.protocol).read_text()))
