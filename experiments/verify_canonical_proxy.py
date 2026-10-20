"""NumPy finite differences of the frozen target-only canonical proxy contrast."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from cross_module_distance import extended_parameters
from subspace_probe import save_json
from task_probe_distance import checked_models
from verify_levelset_audits import convert, raw_scores
from verify_probe_candidate import explicit_merge
from verify_cross_module import ln


def verify(output):
    out=Path(output);report=json.loads((out/'results.json').read_text());assert digest(out/'diagnostic.pt')==report['diagnostic_sha256']
    data=torch.load(out/'diagnostic.pt',map_location='cpu');cfg=data['cfg'];point=torch.load(Path(cfg['publication_point'])/'candidate.pt',map_location='cpu')
    assert digest(Path(cfg['publication_point'])/'candidate.pt')==cfg['candidate_sha256']
    anchor,expert,head,_=checked_models('MNIST');canonical=copy.deepcopy(expert)
    assert digest(cfg['canonical_reference'])==cfg['canonical_reference_sha256'];canonical.load_state_dict(torch.load(cfg['canonical_reference'],map_location='cpu'))
    p=convert(extended_parameters(expert,head));u,v=[point['basis_A'][k].numpy() for k in ['u','vh']];q=point['output_basis'].numpy();r=len(v);size=r*r
    theta=point['theta'].numpy();d=np.random.default_rng(20261016).normal(size=len(theta));d/=np.linalg.norm(d);checks=[]
    for record in report['records']:
        name=record['environment'];proxy=explicit_merge(anchor,expert if name=='ordinary' else canonical,anchor,.5);pp=convert(extended_parameters(proxy,head))
        c=convert(point['proxy'] if name=='ordinary' else data['canonical_cache']);x=ln(c['r'],pp['gamma2'],pp['beta2'],pp['eps2'])
        pre=x@pp['A'].T+pp['a'];z=pre*expit(1.702*pre);base=c['h']-z@pp['B'].T
        def predict(t):
            ea=np.linalg.norm(p['A'])*u@t[:size].reshape(r,r)@v;eb=np.linalg.norm(p['B'])*q@t[size:].reshape(q.shape[1],p['B'].shape[1])
            pre=x@(pp['A']+.5*ea).T+pp['a'];zz=pre*expit(1.702*pre)
            return raw_scores(base+zz@(pp['B']+.5*eb).T,pp)
        before=predict(np.zeros_like(theta));after=predict(theta);saved=data['records'][name]
        np.testing.assert_allclose(before,saved['teacher'].numpy(),atol=1e-8);np.testing.assert_allclose(after,saved['after'].numpy(),atol=1e-8)
        y=before.argmax(1);rows=np.arange(len(y));counts=np.bincount(y,minlength=before.shape[1]);weight=1./counts[y]/int((counts>0).sum())
        def angular(s):
            c=s-s.mean(1,keepdims=True);return c/np.linalg.norm(c,axis=1,keepdims=True)
        def gap(s):
            other=s.copy();other[rows,y]=-np.inf;return s[rows,y]-other.max(1)
        unit=np.sort(gap(angular(before)))[(len(y)-1)//2]
        loss=lambda t:-np.sum(weight*np.maximum(gap(angular(predict(t))),0))/unit
        h=1e-6;fd=(loss(h*d)-loss(-h*d))/(2*h);analytic=saved['gradient'].numpy()@d
        assert abs(fd-analytic)<1e-6
        changed=after.argmax(1)!=y;assert int(changed.sum())==record['frozen_point_changed']
        checks.append({'environment':name,'random_gradient_fd_error':abs(fd-analytic),'normalized_scores_and_changed_class_counts':'passed'})
    a,b=[data['records'][name]['projected_gradient'].numpy() for name in ['ordinary','author_canonical']]
    cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b)));assert abs(cosine-report['task_tangent_gradient_cosine'])<1e-10
    save_json(out/'independent_validation.json',{'status':'passed','checks':checks,'gradient_cosine':cosine,'source_sha256':digest(__file__),
        'diagnostic_sha256':digest(out/'diagnostic.pt'),'new_model_forwards':0,'publication_updates':0,'test_used':False,'partner_used':False,
        'scope':'Independent NumPy normalized scores, finite-difference gradient and stored projected-gradient cosine; no unknown-peer transfer guarantee.'})
    print(json.dumps(checks),flush=True)


def verify_feedback(output):
    """Independent NumPy finite native/probe responses and variable AGA map."""
    from verify_levelset_audits import residual
    from verify_covariance_noise import official_labels
    out=Path(output);rep=json.loads((out/'results.json').read_text())
    assert digest(out/'calibration.pt')==rep['raw_calibration_sha256']
    assert digest(out/'training_point.pt')==rep['raw_point_sha256']
    raw=torch.load(out/'calibration.pt',map_location='cpu');point=torch.load(out/'training_point.pt',map_location='cpu')
    cfg=point['cfg'];p,base,pp=[convert(raw[k]) for k in ['p','base','pp']];cache=raw['cache']
    q=p|convert(point['q']);theta=point['theta'].numpy();r=cfg['probe_rank'];size=r*r
    U,R=[cache['basis_O'][k].numpy() for k in ['u','vh']];vo=raw['task_probe_O'].numpy();vb=cache['basis_B']['vh'].numpy()
    Q=cache['output_basis'].numpy();perm=raw['permutation'].numpy()
    np.testing.assert_allclose(R@vo.T,0,atol=1e-8)
    for a in [U,Q]:np.testing.assert_allclose(a.T@a,np.eye(a.shape[1]),atol=1e-8)
    O=(p['O']+np.linalg.norm(p['O'])*U@theta[:size].reshape(r,r)@R).astype('float32').astype('float64')
    B=(p['B']+np.linalg.norm(p['B'])*Q@theta[size:].reshape(Q.shape[1],3072)).astype('float32').astype('float64')
    # A near-half-ulp quantization discrepancy is bounded in physical distance,
    # not hidden by replacing the stored publication with reconstructed weights.
    quantization_error=max(float(abs(O-q['O']).max()),float(abs(B-q['B']).max()));assert quantization_error<1e-6
    def hidden(rr,par):
        z=ln(rr,par['gamma2'],par['beta2'],par['eps2'])@par['A'].T+par['a']
        return z*expit(1.702*z)
    def prepare(c,probes):
        rr,H,h=[c[k] for k in ['r','o_input','h']];z=hidden(rr,p if probes else pp)
        par=p if probes else pp;rounding=h-rr-z@par['B'].T-par['b']
        rs,hs,bs,off=[rr],[H],[rounding],[np.zeros_like(z)]
        if probes:
            shift=[];n=cfg['probe_contexts']
            for x,v in [(H,vo),(z,vb)]:
                rms=np.sqrt(np.mean((x@v.T)**2,axis=0))
                shift.append((cfg['finite_step']*np.array([-1.,1.])[:,None,None]*rms[None,:,None]*v[None]).reshape(2*r,-1))
            so,sb=shift;m=len(so)
            rs.extend([(rr[:n,None]+(so@p['O'].T)[None]).reshape(-1,768),np.repeat(rr[:n],m,axis=0)])
            hs.extend([(H[:n,None]+so[None]).reshape(-1,768),np.repeat(H[:n],m,axis=0)])
            bs.extend([np.repeat(rounding[:n],m,axis=0)]*2)
            off.extend([np.zeros((n*m,3072)),np.tile(sb,(n,1))])
        return dict(r=np.concatenate(rs),H=np.concatenate(hs),rounding=np.concatenate(bs),offset=np.concatenate(off))
    def forward(d,old,new):
        rr=d['r']+d['H']@(new['O']-old['O']).T
        return raw_scores(d['rounding']+rr+(hidden(rr,new)+d['offset'])@new['B'].T+new['b'],new)
    maximum=0.;source_R=0.;natural=[];ids=[]
    for key,wanted in zip(['native','scale'],point['source_scores']):
        c=convert(cache[key]);d=prepare(c,True);before=forward(d,p,p);after=forward(d,p,q)
        maximum=max(maximum,float(abs(after-wanted.numpy()).max()));np.testing.assert_allclose(after,wanted,atol=1e-8,rtol=1e-9)
        source_R=max(source_R,float(abs(residual(before,after)).max()));natural.append(after[:len(c['h'])])
        ids.extend(cache['indices']['fit' if key=='native' else 'scale'])
    assert abs(source_R-rep['finite_probe_R'])<1e-7
    source_actual=point['actual'].numpy();source_original=point['original'].numpy()
    actual_R=float(abs(residual(source_original,source_actual)).max());assert abs(actual_R-rep['natural_source_R'])<1e-9
    controls=float(abs(source_actual-np.concatenate(natural)).max());assert controls<=cfg['score_tolerance']
    labels=official_labels(cfg['target'],True)[ids];old=source_original.argmax(1);new=source_actual.argmax(1)
    assert int((old!=new).sum())==rep['actual_source_changed']
    classes=[dict(label=c,n=int((labels==c).sum()),original_correct=int(((old==labels)&(labels==c)).sum()),
        released_correct=int(((new==labels)&(labels==c)).sum()),lost=int(((old==labels)&(new!=labels)&(labels==c)).sum()),
        gained=int(((old!=labels)&(new==labels)&(labels==c)).sum())) for c in range(10)]
    def recover(W,quantize=False):
        values=[]
        for j in range(0,768,64):
            A=W[:,j:j+64];C=base['O'][:,j:j+64]
            T=np.linalg.solve(A.T@A+1e-8*np.eye(64),A.T@C)
            values.append((A.astype('float32')@T.astype('float32')).astype('float64') if quantize else A@T)
        return np.concatenate(values,axis=1)
    official_error=max(float(abs(recover(W,True)-V.numpy()).max()) for W,V in zip([p['O'],q['O']],point['official_O']))
    assert official_error<=cfg['author_column_tolerance']
    c=convert(cache['proxy']);d=prepare(c,False);before=forward(d,pp,pp);prototype=pp|convert(point['prototype'])
    after=forward(d,pp,prototype);np.testing.assert_allclose(after,point['environment_scores'],atol=1e-8,rtol=1e-9)
    public_error=float(abs(after-point['environment_actual'].numpy()).max());assert public_error<=cfg['score_tolerance']
    y=before.argmax(1);rows=np.arange(len(y));counts=np.bincount(y,minlength=10);weight=1./counts[y]/int((counts>0).sum())
    def angular(s):
        a=s-s.mean(1,keepdims=True);return a/np.linalg.norm(a,axis=1,keepdims=True)
    def gap(s):
        other=s.copy();other[rows,y]=-np.inf;return s[rows,y]-other.max(1)
    unit=np.sort(gap(angular(before)))[(len(y)-1)//2];F0=recover(p['O'])
    competitors=angular(before);competitors[rows,y]=-np.inf;runner=competitors.argmax(1)
    def objective(t):
        co=t[:size].reshape(r,r);cb=t[size:].reshape(Q.shape[1],3072)
        qo=pp['O']+.5*(recover(p['O']+np.linalg.norm(p['O'])*U@co@R)-F0)
        qb=pp['B']+.5*np.linalg.norm(p['B'])*(Q@cb)[:,perm]
        ss=forward(d,pp,pp|dict(O=qo,B=qb))
        active=angular(ss)
        return (np.dot(t,t)-np.linalg.norm(cb@vb.T)**2)/.25-np.sum(weight*(active[rows,y]-active[rows,runner]))/unit
    g=point['initial_gradient'].numpy();direction=g/np.linalg.norm(g);h=cfg['derivative_step']
    fd=(objective(h*direction)-objective(-h*direction))/(2*h);gradient_error=abs(fd-g@direction)/np.linalg.norm(g)
    assert gradient_error<=cfg['derivative_tolerance']
    D=sum((np.linalg.norm(q[k]-p[k])/np.linalg.norm(p[k]))**2 for k in ['O','B'])**.5
    assert abs(D-rep['parameter_distance'])<1e-10
    changed=after.argmax(1)!=y;perclass=[dict(label=c,n=int((y==c).sum()),changed=int((changed&(y==c)).sum())) for c in range(10)]
    assert perclass==rep['public_after_AGA_per_class']
    result=dict(status='passed',source_sha256=digest(__file__),raw_point_sha256=rep['raw_point_sha256'],
        numpy_source_score_max_error=maximum,quantized_weight_reconstruction_error=quantization_error,
        original_AGA_column_max_error=official_error,variable_recovery_gradient_relative_error=gradient_error,
        natural_source_R=actual_R,finite_probe_R=source_R,source_full_model_score_error=controls,
        public_full_model_score_error=public_error,parameter_distance=D,source_per_class=classes,
        public_per_teacher_class=perclass,new_model_forwards=0,new_candidates=0,
        scope='Source-only independent finite-function,parameter,official recovery-kernel and gradient verification;not confirmation,test or unknown-partner validation.')
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);parser.add_argument('--recovery-feedback',action='store_true');args=parser.parse_args();torch.set_num_threads(4)
    (verify_feedback if args.recovery_feedback else verify)(args.output)
