"""Independent finite NumPy completion, all trial replay and branch attribution."""
import argparse,json,os,time
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from scipy.special import expit
from classification_completion import readout
from clip_benchmark import digest
from cross_module_distance import extended_parameters
from subspace_probe import save_json
from task_probe_distance import checked_models,initialize
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_downstream_first import convert,jacobian
from verify_levelset_audits import raw_scores,residual


def phi(z):return z*expit(1.702*z)
def slope(z):
    s=expit(1.702*z);return s+1.702*z*s*(1-s)
def angular(s):
    z=s-s.mean(1,keepdims=True);return z/np.maximum(np.linalg.norm(z,axis=1,keepdims=True),1e-12)
def stats(a,b):
    y=a.argmax(1);rows=np.arange(len(y));aa=angular(a);bb=angular(b);o=bb.copy();o[rows,y]=-np.inf;old=aa.copy();old[rows,y]=-np.inf
    damage=1-(bb[rows,y]-o.max(1))/(aa[rows,y]-old.max(1));np.testing.assert_array_equal(damage>1,b.argmax(1)!=y)
    q=np.array([np.sort(damage[y==c],kind='stable')[int(.25*(sum(y==c)-1))] for c in range(a.shape[1])])
    return dict(changed=int((b.argmax(1)!=y).sum()),worst=float(np.abs(residual(a,b)).max()),class_quantiles=q.tolist()),q
def check(a,b,w):
    st,_=stats(a,b);assert st['changed']==w['changed'] and abs(st['worst']-w['worst_task_drift'])<1e-8
    for c in w['per_class']:
        y=a.argmax(1);changed=b.argmax(1)!=y;assert int((y==c['label']).sum())==c['n'] and int((changed&(y==c['label'])).sum())==c['changed']
    return st


def verify(cfg):
    start=time.monotonic();out=Path(cfg['output']);rep=json.loads((out/'results.json').read_text())
    assert digest(out/'geometry.pt')==rep['geometry_sha256'] and digest(out/'diagnostic.pt')==rep['diagnostic_sha256']
    for f,pin in cfg['inputs'].items():assert digest(f)==pin,f
    cal=convert(torch.load(cfg['calibration_reference'],map_location='cpu'));g=convert(torch.load(out/'geometry.pt',map_location='cpu'));d=convert(torch.load(out/'diagnostic.pt',map_location='cpu'));p=cal['parameters']
    anchor,expert,head,pins=checked_models(cfg['target']);assert pins==rep['checkpoint_sha256'];p0=convert(extended_parameters(anchor,head))
    for key,value in extended_parameters(expert,head).items():
        if torch.is_tensor(value):np.testing.assert_array_equal(value.numpy(),p[key])
    # Reuse only the nonunique original SVD coordinate orientation, never responses.
    sp,_=readout({k:torch.from_numpy(v).cuda() if isinstance(v,np.ndarray) else v for k,v in p.items()});Q=sp['Q'].cpu().numpy();contrast=p['head']-p['head'].mean(0);M=p['gamma'][:,None]*(p['projection']@contrast.T);M-=M.mean(0);bias=p['beta']@p['projection']@contrast.T
    qnp,sm,_=np.linalg.svd(M,full_matrices=False);qnp=qnp[:,:sum(sm>sm[0]*1e-10)];np.testing.assert_allclose(Q@Q.T,qnp@qnp.T,atol=1e-10)
    small=Q.T@M;v=np.linalg.solve(small@small.T,small@bias);w=v@Q.T;qa=1-np.mean((w-w.mean())**2);assert qa>0
    # Recover original source ordering and all natural/task/normal/tangent inputs.
    ss=raw_scores(cal['cache']['h'],p);yy=ss.argmax(1);other=ss.copy();other[np.arange(len(yy)),yy]=-np.inf;gap=ss[np.arange(len(yy)),yy]-other.max(1);core=[]
    for c in range(len(p['head'])):
        ix=np.flatnonzero(yy==c);core.extend(ix[np.argsort(gap[ix],kind='stable')[:cfg['contexts_per_class']]])
    order=core+[i for i in range(len(yy)) if i not in set(core)];ids=[cal['indices'][i] for i in order];assert ids==g['source_indices'];raw={k:v[order] for k,v in cal['cache'].items()};ss=ss[order];yy=yy[order];n=len(core)
    x=ln(raw['r'],p['gamma2'],p['beta2'],p['eps2']);z=x@p['A'].T+p['a'];J,_=jacobian(raw['h'][:cfg['selection_n']],p);KJ=(J@p['B'])*slope(z[:len(J)])[:,None]
    da=p['A']-p0['A'];ev,Vall=np.linalg.eigh(da.T@da);sing=np.sqrt(np.maximum(ev[::-1],0));Vall=Vall[:,::-1].T;Uall=da@Vall.T/sing[None];response=(KJ@Uall)*(x[:len(J)]@Vall.T)[:,None]*sing[None,None]
    sy=yy[:len(J)];rr=np.arange(len(J));gap=ss[:len(J)][rr,sy,None]-ss[:len(J)];gap[rr,sy]=1;rel=(response[rr,sy,None]-response)/gap[:,:,None];rel[rr,sy]=0;count=np.bincount(sy,minlength=len(p['head']));imp=np.sum(np.sum(rel*rel,1)/count[sy,None],0);chosen=np.argsort(-imp,kind='stable')[:cfg['probe_rank']]
    np.testing.assert_array_equal(chosen,g['chosen']);np.testing.assert_allclose(imp,g['importance'],atol=1e-7,rtol=1e-7);U,V,Va=g['U'],g['V'],g['input_V'];np.testing.assert_allclose(U@U.T,Uall[:,chosen]@Uall[:,chosen].T,atol=1e-8,rtol=1e-8);np.testing.assert_allclose(Va.T@Va,Vall[chosen].T@Vall[chosen],atol=1e-8);np.testing.assert_array_equal(V,cal['bases']['B']['vh'])
    rounding=raw['h']-raw['r']-phi(z)@p['B'].T-p['b'];shifts=[]
    for incoming,basis in [(raw['o_input'],cal['bases']['O']['vh']),(phi(z),V)]:
        rms=np.sqrt(np.mean((incoming@basis.T)**2,0));shifts.append((float(np.float32(cfg['finite_step']))*np.array([-1.,1.])[:,None,None]*rms[None,:,None]*basis[None]).reshape(2*cfg['probe_rank'],-1))
    so,sb=shifts;ct=len(so);r=np.r_[raw['r'],(raw['r'][:n,None]+(so@p['O'].T)[None]).reshape(-1,768),np.repeat(raw['r'][:n],ct,0)];ro=np.r_[rounding,np.repeat(rounding[:n],ct,0),np.repeat(rounding[:n],ct,0)];offset=np.r_[np.zeros((len(z)+n*ct,len(p['a']))),np.tile(sb,(n,1))]
    pre=ln(r,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a'];hh=ro+r+(phi(pre)+offset)@p['B'].T+p['b'];rms=np.sqrt(np.mean((x@Va.T)**2,0));np.testing.assert_allclose(rms,g['rms'],atol=1e-10);F=p['A']@Va.T*rms[None];ep=[];eh=[]
    for i,item in enumerate(g['frames']):
        fr=item['frame'];np.testing.assert_allclose(fr@fr.T,np.eye(len(fr)),atol=1e-10);ju=KJ[i]@F;normal=ju[yy[i],None]-ju;normal=normal[np.arange(len(ju))!=yy[i]];assert np.abs(normal@fr[item['rank']:].T).max()<1e-8
        shift=cfg['neighborhood_radius']*np.r_[fr,-fr]@F.T;ep.append(z[i]+shift);eh.append(raw['h'][i]+(phi(z[i]+shift)-phi(z[i]))@p['B'].T)
    pre=np.r_[pre,np.concatenate(ep)];hh=np.r_[hh,np.concatenate(eh)];offset=np.r_[offset,np.zeros_like(np.concatenate(ep))];Z0=phi(pre)+offset;teacher=raw_scores(hh,p);aa=angular(teacher);y=teacher.argmax(1);rows=np.arange(len(y));alt=aa.copy();alt[rows,y]=-np.inf;mg=aa[rows,y]-alt.max(1);cnt=np.bincount(y,minlength=len(p['head']));weights=1/mg**2/cnt[y];weights/=weights.mean();na=np.linalg.norm(p['a']);nb=np.linalg.norm(p['B'])
    def complete(t):
        delta=na*(U@t);Z=phi(pre+delta)+offset;hg=hh+(Z-Z0)@p['B'].T;se=np.sqrt(np.mean((hh-hh.mean(1,keepdims=True))**2,1,keepdims=True)+p['eps']);proj=(hh-hg)@Q;star=hg+proj@Q.T+se*w;center=star-star.mean(1,keepdims=True);b0=np.mean(center*(w-w.mean()),1,keepdims=True);c0=np.mean(center**2,1,keepdims=True)+p['eps'];rho=c0/(np.sqrt(b0*b0+qa*c0)+b0);target=proj+(se-rho)*v;X=Z@V.T;gram=X.T@(weights[:,None]*X);ridge=cfg['ridge_relative']*np.trace(gram)/len(gram);k=np.linalg.solve(gram+ridge*np.eye(len(gram)),X.T@(weights[:,None]*target));return delta,k,Z,hg,X,target,ridge
    envbase=torch.load(cfg['environment_cache'],map_location='cpu');pe=convert(envbase)['proxy_parameters'];envs=[]
    for row in g['environments']:
        cache=row['cache'];pr=cache['x']@pe['A'].T+pe['a'];actual=cache['actual']/np.linalg.norm(cache['actual'],axis=1,keepdims=True)@pe['head'].T;bs=raw_scores(cache['h'],pe);assert np.abs(actual-bs).max()<5e-4;envs.append(dict(name=row['name'],pre=pr,h=cache['h'],teacher=bs))
    def env_forward(delta,db):return [raw_scores(e['h']+(phi(e['pre']+.5*delta)-phi(e['pre']))@pe['B'].T+.5*phi(e['pre']+.5*delta)@db.T,pe) for e in envs]
    errors=[];trace_reports=[]
    center=-U.T@z.mean(0);seed=cfg['initial_gate_norm']*center/np.linalg.norm(center);seed_delta,seed_k,seed_z,seed_hg,_,_,_=complete(seed);check(teacher,raw_scores(seed_hg+(seed_z@V.T@seed_k)@Q.T,p),rep['initial']['source'])
    for item in d['trace']:
        dt,kk,Z,hg,_,_,_=complete(item['t']);np.testing.assert_allclose(dt,item['d'],atol=1e-10);np.testing.assert_allclose(kk,item['K'],atol=1e-8,rtol=1e-7);sa=raw_scores(hg+(Z@V.T@kk)@Q.T,p);st=check(teacher,sa,item['metrics']['source']);errors.append(np.abs(sa-item['source_after']).max());values=env_forward(dt,Q@kk.T@V);qq=[]
        for e,after,want,saved in zip(envs,values,item['metrics']['environment'],item['environment_after']):
            check(e['teacher'],after,want);_,quantile=stats(e['teacher'],after);qq.extend(quantile);errors.append(np.abs(after-saved).max())
        assert abs(min(qq)-item['metrics']['minimum_class_quantile'])<1e-8;trace_reports.append(dict(source=st,minimum_class_quantile=float(min(qq)),source_feasible=item['metrics']['source_feasible']))
    pt=d['selected'];delta,kk,Z,hg,_,_,_=complete(pt['t']);np.testing.assert_allclose(kk,pt['K'],atol=1e-8,rtol=1e-7);unrounded=Q@kk.T@V
    # Validate materialization, then evaluate the saved exact FP32 tensors independently.
    dd,db=pt['d'],pt['delta_B'];np.testing.assert_array_equal((p['a']+dd).astype(np.float32).astype(np.float64)-p['a'],dd);np.testing.assert_array_equal((p['B']+db).astype(np.float32).astype(np.float64)-p['B'],db)
    assert np.max(np.abs(db-unrounded)-.5*np.abs(np.spacing((p['B']+db).astype(np.float32))).astype(float))<1e-10
    distance=np.sqrt(np.linalg.norm(dd)**2/na**2+np.linalg.norm(db)**2/nb**2);assert abs(distance-rep['selected']['distance'])<1e-12 and distance<=.5
    SZ=phi(pre+dd)+offset;sh=hh+(SZ-Z0)@p['B'].T+SZ@db.T;sa=raw_scores(sh,p);source=check(teacher,sa,rep['selected']['source']);errors.append(np.abs(sa-pt['source_after']).max());finite_env=env_forward(dd,db);reports=[]
    def decompose(h,pr,p,delta,db,factor):
        old=phi(pr);new=phi(pr+factor*delta);up=(new-old)@p['B'].T;down=factor*old@db.T;interaction=factor*(new-old)@db.T;joint=up+down+interaction
        b=raw_scores(h,p);a=raw_scores(h+joint,p);contrast=p['head']-p['head'].mean(0);M=p['gamma'][:,None]*(p['projection']@contrast.T);M-=M.mean(0);bias=p['beta']@p['projection']@contrast.T;sd=lambda v:np.sqrt(np.mean((v-v.mean(1,keepdims=True))**2,1)+p['eps']);N=h@M+sd(h)[:,None]*bias;Y=b.argmax(1);ix=np.arange(len(Y));mask=np.ones_like(N,dtype=bool);mask[ix,Y]=False;gap=N[ix,Y,None]-N;components=dict(up=up@M,down=down@M,interaction=interaction@M,variance=(sd(h+joint)-sd(h))[:,None]*bias);harm={k:(v-v[ix,Y,None])/np.where(mask,gap,1) for k,v in components.items()};total=sum(harm.values());j=np.argmax(np.where(mask,total,-np.inf),1);damage=total[ix,j];np.testing.assert_array_equal(damage>1,a.argmax(1)!=Y)
        first=(slope(pr)*(factor*delta))@p['B'].T;remainder=up-first
        return dict(changed=int((a.argmax(1)!=Y).sum()),maximum_numerator_damage=float(damage.max()),median_numerator_damage=float(np.median(damage)),remaining_at_closest_crossing=float(1-damage.max()),
            up_only_changed=int((raw_scores(h+up,p).argmax(1)!=Y).sum()),down_only_changed=int((raw_scores(h+down,p).argmax(1)!=Y).sum()),
            means_at_joint_worst_competitor={k:float(v[ix,j].mean()) for k,v in harm.items()},gate_sign_crossing_fraction=float(np.mean(pr*(pr+factor*delta)<0)),
            first_order_up_rms=float(np.sqrt(np.mean(first**2))),finite_up_remainder_rms=float(np.sqrt(np.mean(remainder**2))),
            identity_error=float(np.abs(sum(components.values())-((h+joint)@M+sd(h+joint)[:,None]*bias-N)).max()),
            per_class=[dict(label=c,n=int(sum(Y==c)),changed=int(sum((a.argmax(1)!=Y)&(Y==c))),maximum_damage=float(damage[Y==c].max())) for c in range(len(p['head']))])
    for i,(env,after,want,saved) in enumerate(zip(envs,finite_env,rep['selected']['environment'],pt['environment'])):
        st=check(env['teacher'],after,want);errors.append(np.abs(after-saved['after']).max());attr=decompose(env['h'],env['pre'],pe,dd,db,.5);reports.append(dict(name=env['name'],**st,attribution=attr));lab=official_labels(cfg['target'],True)[g['environment_indices']];assert int((after.argmax(1)==lab).sum())==want['protected_correct']
    assert max(errors)<1e-8
    # Frozen last trust proposals: distinguish source infeasibility from critic switching.
    last=rep['history'][-1];switching=[];jac=d['jacobians'][-1]
    for trial,item in zip(last['trials'],d['trace'][-len(last['trials']):]):
        dt,kt,_,_,_,_,_=complete(item['t']);vals=env_forward(dt,Q@kt.T@V);fixed=[]
        for critic in jac['critics']:
            index=[e['name'] for e in envs].index(critic['environment']);e=envs[index];base=angular(e['teacher']);new=angular(vals[index]);i=critic['sample'];y=int(e['teacher'][i].argmax());j=critic['competitor'];oth=base[i].copy();oth[y]=-np.inf;fixed.append(1-(new[i,y]-new[i,j])/(base[i,y]-oth.max()))
        switching.append(dict(step=trial['step'],source_feasible=trial['metrics']['source_feasible'],predicted_improvement=trial['prediction'],actual_improvement=trial['actual'],
            frozen_critic_minimum_improvement=float(min(fixed)-np.min(jac['q'])),reordered_full_quantile_improvement=trial['actual']))
    gradient_errors=[];distance_gradient_errors=[]
    for saved in [d['jacobians'][0],d['jacobians'][-1]]:
        step=1e-5;fixed_values=[];distances=[]
        for sign in [-1,1]:
            t=saved['t']+sign*step*saved['direction'];dt,kt,_,_,_,_,_=complete(t);vals=env_forward(dt,Q@kt.T@V);crit=[]
            distances.append(float(t@t+np.sum(kt*kt)/nb**2))
            for cc in saved['critics']:
                ix=[e['name'] for e in envs].index(cc['environment']);e=envs[ix];a=angular(e['teacher']);b=angular(vals[ix]);i=cc['sample'];y=int(e['teacher'][i].argmax());j=cc['competitor'];o=a[i].copy();o[y]=-np.inf
                crit.append(1-(b[i,y]-b[i,j])/(a[i,y]-o.max()))
            fixed_values.append(np.array(crit))
        fd=(fixed_values[1]-fixed_values[0])/(2*step);err=float(np.abs(fd-saved['G']@saved['direction']).max());gradient_errors.append(err)
        derr=abs((distances[1]-distances[0])/(2*step)-saved['distance_gradient']@saved['direction']);distance_gradient_errors.append(float(derr))
        assert err<1e-5 and derr<1e-5,(err,derr)
    result=dict(status='passed_closed_mechanism' if not rep['mechanism_supported'] else 'passed_mechanism_requires_fresh_confirmation',mechanism_supported=rep['mechanism_supported'],source=source,parameter_distance=float(distance),environment=reports,
        maximum_numpy_score_error=float(max(errors)),trial_points_independently_reconstructed=len(d['trace']),all_trial_compensators_verified=True,last_trust_proposals=switching,
        frozen_critic_implicit_gradient_fd_errors=gradient_errors,distance_implicit_gradient_fd_errors=distance_gradient_errors,
        source_natural_original_correct=int((teacher[:len(ids)].argmax(1)==official_labels(cfg['target'],True)[ids]).sum()),source_natural_protected_correct=int((sa[:len(ids)].argmax(1)==official_labels(cfg['target'],True)[ids]).sum()),
        source_completion_basis='Original nonunique CUDA/Torch SVD coordinate orientation only, column projector independently verified; all completion, solve and finite responses are independent NumPy. First verifier CPU orientation had different column signs; only basis-coordinate replay fixed, primary unchanged.',
        diagnostic_sha256=rep['diagnostic_sha256'],geometry_sha256=rep['geometry_sha256'],source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        peer_loaded=False,test_loaded=False,confirmation_loaded=False,original_expert_updated=False,new_parameter_updates=0,new_release_weights=0,
        scope='Only one frozen gate chart and finite trust sequence verified. Failure is not a proof that full-budget finite gate compensation is impossible. Three actual shared prefix nodes do not prove unknown-peer coverage.')
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--protocol',required=True);args=a.parse_args();initialize();verify(json.loads(Path(args.protocol).read_text()))
