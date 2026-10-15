"""SymPy exact counterexample and independent NumPy finite CLIP reconstruction."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import sympy as sp
import torch

from clip_benchmark import digest
from subspace_probe import save_json
from verify_finite_comparison import N, geometry, sd, suffix
from verify_levelset_audits import convert, raw_scores


def exact_toy(data):
    a=b=sp.Rational(3,10);c=sp.Rational(4,5);margin=sp.Rational(1,20)
    ae=sp.Matrix([[1,0],[0,1],[0,0]]);f=sp.Matrix([[1,1-margin],[1-margin,1]])
    be=f.row_join(sp.zeros(2,1));v=sp.Matrix([a,a]);d=sp.Matrix([b,-b])
    ar=sp.eye(2).col_join(v.T);br=(f-d*v.T).row_join(d)
    assert br*ar==be*ae==f
    k=c/(1+a*a)
    def align(mat):
        return mat*sp.diag(*[c*mat[i,i]/(mat[:,i].dot(mat[:,i])) for i in range(2)])
    assert align(ar)==k*ar and br*align(ar)==k*f
    n=lambda x:sum(z*z for z in x)
    squared=n(ar-ae)/n(ae)+n(br-be)/n(be)
    assert squared<sp.Rational(1,4) and str(squared)==data['source_squared_distance_exact']
    # Derive the finite mixed formula with symbolic, unrestricted peer parameters.
    q,s,w0,w1=sp.symbols('q s w0 w1');w=sp.Matrix([w0,w1]);ap=s*sp.eye(2).col_join(w.T/s)
    symbolic=((br+be)/2)*((q*ar+ap)/2)-(q+s)*f/2-d*(w-s*v).T/4
    assert sp.simplify(symbolic)==sp.zeros(2,2)
    families={'public_base':c*ae,'same_function_preserved':ae[:2,:].col_join(v.T),
              'same_function_reversed':ae[:2,:].col_join(sp.Matrix([[-c,c]]))}
    checks=[]
    for row in data['exact_certificate']:
        name,mode=row['partner'],row['column_alignment'];peer=families[name]
        left=ar if mode=='none' else align(ar)
        orig=ae if mode=='none' else align(ae)
        if mode=='all_columns':peer=align(peer)
        result=((br+be)/2)*((left+peer)/2);baseline=be*((orig+peer)/2)
        assert result==sp.Matrix([[sp.Rational(z) for z in rr] for rr in row['after']])
        gaps=[result[0,0]-result[1,0],result[1,1]-result[0,1]]
        assert all(sp.Rational(z)==g for z,g in zip(row['margin'],gaps))
        assert baseline[0,0]>baseline[1,0] and baseline[1,1]>baseline[0,1]
        checks.append(dict(partner=name,column_alignment=mode,correct=sum(bool(z>0) for z in gaps)))
    return dict(symbolic_finite_identity=True,exact_source_all_input_function=True,
                exact_squared_distance=str(squared),all9_exact_margin_rows=True,checks=checks,
                scope='Two-module mathematical illustration with column alignment only, not an original defense or real CLIP fusion result.')


def margin_stat(before,after):
    y=before.argmax(1);rows=np.arange(len(y));gaps=before[rows,y,None]-before;valid=gaps>0
    delta=after-before;harm=np.divide(-(delta[rows,y,None]-delta),gaps,out=np.full_like(gaps,-np.inf),where=valid)
    maximum=harm.max(1);changed=after.argmax(1)!=y
    np.testing.assert_array_equal(changed,maximum>1)
    return int(changed.sum()),float(maximum.max()),[int(changed[y==c].sum()) for c in range(before.shape[1])]


def verify(output):
    start=time.monotonic();root=Path(output);cfg=json.loads((root/'protocol.json').read_text());report=json.loads((root/'results.json').read_text())
    assert digest(root/'artifacts/diagnostic.pt')==report['diagnostic_sha256']
    assert digest(root/'counterexample.json')==report['toy_sha256']
    toy=exact_toy(json.loads((root/'counterexample.json').read_text()))
    data=torch.load(root/'artifacts/diagnostic.pt',map_location='cpu');records=[];max_error=0.;full_error=0.
    for env,base in data['environments'].items():
        p=convert(base['parameters']);H=base['cache']['H'].numpy();h=base['cache']['h'].numpy()
        h0,c0=suffix(H,p);n0=N(h,p);basis=data['basis'].numpy()
        if env=='aligned_source':basis=basis[:,data['AGA_permutation'].numpy()]
        for name,groups in data['cases'].items():
            group=groups[env];q=convert(group['parameters']);hq,cq=suffix(H,q);db=q['B']-p['B']
            assert np.array_equal(p['b'],q['b'])
            u=cq['r']-c0['r']+(cq['z']-c0['z'])@p['B'].T
            branch=c0['z']@db.T;cross=(cq['z']-c0['z'])@db.T
            st=dict(upper=h+u,B_only=h+branch,joint=h+hq-h0)
            assert np.max(abs((hq-h0)-u-branch-cross))<1e-9
            for key in ['upper','B_only','joint']:
                score=N(st[key],p);saved=group['numerator'][key].numpy();e=float(abs(score-saved).max());max_error=max(max_error,e)
                np.testing.assert_allclose(score,saved,atol=1e-8,rtol=1e-8)
                changed,maximum,counts=margin_stat(n0,score);rr=report['records'][env][name][key]
                assert changed==rr['changed']
                assert counts==[x['changed'] for x in rr['by_teacher_class']]
                assert abs(maximum-rr['maximum_signed_harm'])<1e-7
                np.testing.assert_array_equal(score.argmax(1),raw_scores(st[key],p).argmax(1))
            s0,s1=raw_scores(h,p),raw_scores(st['joint'],p);y=s0.argmax(1);rows=np.arange(len(y));g=s0[rows,y,None]-s0
            ds=s1-s0;rho=np.divide(ds[rows,y,None]-ds,g,out=np.zeros_like(g),where=g>0)
            rr=report['records'][env][name]
            assert abs(float(abs(rho).max())-rr['normalized_score_worst_margin_drift'])<1e-7
            m,b=geometry(p);terms=dict(upper=u@m,original_gate_B=branch@m,finite_interaction=cross@m,
                                      normalization=(sd(st['joint'],p)-sd(h,p))*b)
            for key,value in terms.items():np.testing.assert_allclose(value,group['terms'][key].numpy(),atol=1e-8,rtol=1e-8)
            assert np.max(abs(sum(terms.values())-(N(st['joint'],p)-n0)))<1e-8
            if np.array_equal(p['A'],q['A']) and np.linalg.norm(db)>0:
                dt=db@basis.T@basis;r1=np.sum(dt**2)/np.sum(db**2)
                r2=np.sum((cq['z']@dt.T@m)**2)/max(1e-30,np.sum((cq['z']@db.T@m)**2))
                assert abs(r1-rr['frozen_task_SVD_B_parameter_energy_ratio'])<1e-8
                assert abs(r2-rr['frozen_task_SVD_B_class_response_energy_ratio'])<1e-8
            actual=group['actual'].double().numpy();e=float(abs(s1[:len(actual)]-actual).max());full_error=max(full_error,e)
            control=e<cfg['forward_atol'];assert control==rr['full_model_control_passed']
            assert abs(e-rr['full_model_score_max_abs'])<1e-7
            records.append(dict(environment=env,method=name,all512_module_counts_and_scores=True,all3_branches=True,full_model_control_passed=control))
            print('INDEPENDENT',env,name,'joint changes',rr['joint']['changed'],flush=True)
    result=dict(status='finite_module_audit_verified_with_preserved_control_failures',symbolic_counterexample=toy,
                NumPy_finite_numerator_max_abs=max_error,NumPy_vs_full_model_score_max_abs=full_error,
                records=records,full_model_controls=sum(r['full_model_replay_n'] for g in report['records'].values() for r in g.values()),
                full_model_control_failures=sum(not r['full_model_control_passed'] for r in records),source_only=True,no_new_candidate=True,real_partner_checkpoint_loaded=False,test_used=False,confirmation_used=False,
                diagnostic_sha256=report['diagnostic_sha256'],results_sha256=digest(root/'results.json'),verifier_sha256=digest(__file__),
                wall_seconds=time.monotonic()-start)
    save_json(root/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True);torch.set_num_threads(4);verify(p.parse_args().output)
