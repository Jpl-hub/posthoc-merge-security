"""Exact finite counterexample and matched source-only frozen-model mechanism audit."""

import argparse
import copy
import json
import time
from fractions import Fraction as Q
from pathlib import Path

import numpy as np
import torch

from attention_cascade import parameters
from clip_benchmark import digest
from compare_finite_protection import capture_prefix, cpu, numerator, parts
from covariance_noise import dataset
from mechanisms import load_author
from probe_distance_candidate import on_device
from probe_margin_distance import scores
from subspace_probe import save_json
from task_probe_distance import checked_models, initialize


def product(a,b):
    return [[sum(x*y for x,y in zip(row,col)) for col in zip(*b)] for row in a]


def mean(a,b): return [[(x+y)/2 for x,y in zip(ra,rb)] for ra,rb in zip(a,b)]
def floats(a): return np.asarray(a,dtype=float)


def counterexample(author):
    a=b=Q(3,10);c=Q(4,5);m=Q(1,20)
    f=[[Q(1),1-m],[1-m,Q(1)]];ae=[[Q(1),Q(0)],[Q(0),Q(1)],[Q(0),Q(0)]]
    be=[row+[Q(0)] for row in f];v=[a,a];d=[b,-b]
    ar=ae[:2]+[v];br=[[f[i][j]-d[i]*v[j] for j in range(2)]+[d[i]] for i in range(2)]
    assert product(be,ae)==f and product(br,ar)==f
    def aligned(mat):
        out=[[],[],[]]
        for j in range(2):
            t=c*mat[j][j]/sum(mat[i][j]**2 for i in range(3))
            for i in range(3):out[i].append(mat[i][j]*t)
        return out
    ah=aligned(ar);aeh=aligned(ae);k=c/(1+a*a)
    assert product(br,ah)==[[k*z for z in row] for row in f]
    norm=lambda z:sum(t*t for row in z for t in row)
    delta=lambda x,y:[[a-b for a,b in zip(u,v)] for u,v in zip(x,y)]
    dist=norm(delta(ar,ae))/norm(ae)+norm(delta(br,be))/norm(be)
    aligned_dist=norm(delta(ah,aeh))/norm(ae)+norm(delta(br,be))/norm(be)
    assert dist<Q(1,4)
    rows=[];exact=[];error=0.
    families=[('public_base',[[c,Q(0)],[Q(0),c],[Q(0),Q(0)]]),('same_function_preserved',ae[:2]+[v]),('same_function_reversed',ae[:2]+[[Q(-4,5),Q(4,5)]])]
    for name,ap in families:
        fp=product(be,ap);normalized=floats(fp)/np.linalg.norm(floats(fp),axis=0,keepdims=True)
        np.testing.assert_allclose(normalized,floats(f)/np.linalg.norm(floats(f),axis=0,keepdims=True),atol=1e-15)
        for attack in ['none','target_columns','all_columns']:
            left=ar if attack=='none' else ah;original=ae if attack=='none' else aeh
            peer=aligned(ap) if attack=='all_columns' else ap
            before=product(mean(be,be),mean(original,peer));after=product(mean(br,be),mean(left,peer))
            gamma=(left[0][0]+peer[0][0])/2
            formula=[[gamma*f[i][j]+d[i]*(peer[2][j]-peer[0][0]*v[j])/4 for j in range(2)] for i in range(2)]
            assert formula==after
            margins=[after[0][0]-after[1][0],after[1][1]-after[0][1]]
            before_margins=[before[0][0]-before[1][0],before[1][1]-before[0][1]]
            assert min(before_margins)>0
            correct=sum(z>0 for z in margins)
            if name=='same_function_preserved':assert correct==2
            if name=='same_function_reversed':assert correct==0
            if name=='public_base' and attack in ['none','target_columns']:assert correct==2
            rows.append(dict(partner=name,column_alignment=attack,correct=correct,n=2,margin=[float(z) for z in margins],unprotected_correct=2))
            exact.append(dict(partner=name,column_alignment=attack,after=[[str(z) for z in row] for row in after],formula_verified=True,margin=[str(z) for z in margins]))
    for mat in [ae,ar]+[p for _,p in families]:
        src=torch.tensor(floats(mat),dtype=torch.float32);ref=float(c)*torch.tensor(floats(ae),dtype=torch.float32)
        result=torch.cat([src[:,j:j+1]@author['solve_linear_transformation'](src[:,j:j+1],ref[:,j:j+1]) for j in range(2)],dim=1)
        error=max(error,float(np.max(abs(result.numpy()-floats(aligned(mat))))))
    assert error<1e-6
    uu,ss,vh=np.linalg.svd((1-float(c))*floats(ae),full_matrices=False)
    svd_error=float(abs(((1-float(c))*floats(ae))@vh.T-uu*ss).max())
    functional_probe_error=float(abs(floats(product(br,ar))@vh.T-floats(f)@vh.T).max())
    return dict(task_SVD_matrix_shape=[3,2],task_SVD_right_probe_shape=list(vh.shape),task_SVD_identity_error=svd_error,task_SVD_functional_probe_error=functional_probe_error,official_column_solver_calls=10,status='exact_finite_counterexample_verified',dimension=dict(input=2,hidden=3,classes=2),
                source_parameter_distance=float(dist)**.5,source_squared_distance_exact=str(dist),
                aligned_parameter_distance=float(aligned_dist)**.5,source_function_equal_for_all_inputs=True,
                normalized_task_and_SVD_response_equal_for_all_nonzero_outputs=True,
                coefficients=dict(a=str(a),b=str(b),c=str(c),margin=str(m)),results=rows,exact_certificate=exact,
                official_column_solver_error=error,scope='Only the author least-squares column component in a two-linear-module illustration; not CLIP, complete AGA, a trained-peer distribution, or defense impossibility for all transforms.')


def margin_stat(n0,n1):
    y=n0.argmax(1);g=n0.gather(1,y[:,None])-n0;valid=g>0
    delta=n1-n0;harm=-(delta.gather(1,y[:,None])-delta)/g.masked_fill(~valid,1.)
    harm=harm.masked_fill(~valid,-torch.inf);maximum=harm.max(1).values
    changed=n1.argmax(1).ne(y);assert torch.equal(changed,maximum>1.)
    return dict(n=len(y),changed=int(changed.sum()),maximum_signed_harm=float(maximum.max()),
                mean_maximum_signed_harm=float(maximum.mean()),
                by_teacher_class=[dict(class_id=c,n=int(y.eq(c).sum()),changed=int((changed&y.eq(c)).sum()),
                    maximum_signed_harm=float(maximum[y.eq(c)].max()) if y.eq(c).any() else None) for c in range(n0.shape[1])])


@torch.no_grad()
def run(protocol):
    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False)
    (out/'artifacts').mkdir();save_json(out/'protocol.json',cfg)
    for p,h in cfg['source_sha256'].items():assert digest(p)==h,p
    for p,h in cfg['inputs'].items():assert digest(p)==h,p
    initialize();assert '3090' in torch.cuda.get_device_name();start=time.monotonic()
    author,ah=load_author(Path('external/secure-merge-attack/utils_attack_B32.py'));assert ah==cfg['AGA_source_sha256']
    toy=counterexample(author);save_json(out/'counterexample.json',toy)
    anchor,own,head,hashes=checked_models('MNIST')
    ids=json.loads(Path(cfg['source_indices']).read_text())['indices'];assert len(ids)==cfg['n']
    save_json(out/'training_indices.json',dict(fit=ids,role='retrospective_construction_data_not_confirmation'))
    train=dataset('MNIST','data',own.val_preprocess,True)
    original_aga=copy.deepcopy(own);original_aga.load_state_dict(torch.load(cfg['original_AGA'],map_location='cpu'),strict=True)
    basis=torch.load(cfg['basis_reference'],map_location='cpu')['basis']['B']['vh']
    native_A=own.model.visual.transformer.resblocks[11].mlp.c_fc.weight
    def key(v):return v.float().contiguous().numpy().tobytes()
    row_map={key(row):i for i,row in enumerate(native_A)}
    perm=torch.tensor([row_map[key(row)] for row in original_aga.model.visual.transformer.resblocks[11].mlp.c_fc.weight])
    records={};raw=dict(indices=ids,cases={},environments={},basis=basis,AGA_permutation=perm,config=cfg)
    for env,baseline in [('source',own),('aligned_source',original_aga)]:
        cache=capture_prefix(baseline,train,ids);p0=parameters(baseline,head);p0.pop('D');p=on_device(p0)
        raw['environments'][env]=dict(cache=cache,parameters=p0);H=cache['H'].cuda();h=cache['h'].cuda();n0=numerator(h,p)
        records[env]={};basis_env=basis if env=='source' else basis[:,perm]
        for case in cfg['cases']:
            model=copy.deepcopy(baseline)
            if case['name']=='head' and env=='aligned_source':
                data=torch.load(case['AGA_raw'],map_location='cpu')['AGA_modified']
                names={'W':'in_proj_weight','bias':'in_proj_bias','O':'out_proj.weight','ob':'out_proj.bias'}
                edits={'model.visual.transformer.resblocks.11.attn.'+names[k]:v.float() for k,v in data.items()}
            else:edits=torch.load(case['release' if env=='source' else 'AGA_edits'],map_location='cpu')
            model.load_state_dict(baseline.state_dict()|edits,strict=True)
            q0=parameters(model,head);q0.pop('D');q=on_device(q0);st=parts(H,h,p,q)
            row={k:margin_stat(n0,numerator(st[k],p)) for k in ['upper','B_only','joint']}
            normal0=scores(h,p);normal1=scores(st['joint'],p);yy=normal0.argmax(1);gaps=normal0.gather(1,yy[:,None])-normal0
            ds=normal1-normal0;valid=gaps>0;rho=((ds.gather(1,yy[:,None])-ds)/gaps.masked_fill(~valid,1.)).masked_fill(~valid,0.)
            row['normalized_score_worst_margin_drift']=float(rho.abs().max())
            from finite_compensation_transport import geometry,sigma
            m,b=geometry(p)
            terms={k:v@m for k,v in [('upper',st['U']),('original_gate_B',st['V']),('finite_interaction',st['I'])]}
            terms['normalization']=(sigma(st['joint'],p)-sigma(h,p))*b
            error=float((sum(terms.values())-(numerator(st['joint'],p)-n0)).abs().max());assert error<1e-8
            row['finite_identity_error']=error
            db=q['B']-p['B'];same_a=torch.equal(p['A'],q['A'])
            row['same_A_coordinates']=same_a
            if same_a and float(db.norm())>0:
                vv=basis_env.cuda();dt=db@vv.T@vv
                response=st['z']@db.T@m;tr=st['z']@dt.T@m
                row['frozen_task_SVD_B_parameter_energy_ratio']=float(dt.square().sum()/db.square().sum())
                row['frozen_task_SVD_B_class_response_energy_ratio']=float(tr.square().sum()/response.square().sum().clamp_min(1e-30))
            else:row['frozen_task_SVD_B_projection']=None
            choose=torch.arange(cfg['full_forward_check_n']);images=torch.stack([train[ids[i]][0] for i in choose]).cuda()
            model.cuda();head.cuda();actual=head(model(images)).cpu();model.cpu();head.cpu()
            expected=scores(st['joint'][choose.cuda()],p).cpu();e=float((actual.double()-expected).abs().max())
            row['full_model_control_passed']=e<cfg['forward_atol']
            row['full_model_control_tolerance']=cfg['forward_atol']
            row['full_model_control_prediction_changes']=int(actual.argmax(1).ne(expected.argmax(1)).sum())
            row['full_model_replay_n']=len(choose);row['full_model_score_max_abs']=e
            raw['cases'].setdefault(case['name'],{})[env]=dict(parameters=q0,states=cpu(st),numerator_before=n0.cpu(),
                numerator={k:numerator(st[k],p).cpu() for k in ['upper','B_only','joint']},terms=cpu(terms),actual=actual)
            records[env][case['name']]=row
            print('FROZEN_SOURCE',env,case['name'],json.dumps(row),flush=True)
    torch.save(raw,out/'artifacts/diagnostic.pt')
    report=dict(status='frozen_module_intervention_audit_with_recorded_full_model_controls',records=records,source_n=len(ids),
                source_only=True,test_used=False,partner_loaded=False,confirmation_used=False,full_model_control_failures=sum(not row['full_model_control_passed'] for group in records.values() for row in group.values()),new_candidates=0,new_full_model_attack_calls=0,toy_author_column_solver_calls=10,
                gradients=0,source_sha256=cfg['source_sha256'],input_sha256=cfg['inputs'],original_checkpoint_sha256=hashes,
                diagnostic_sha256=digest(out/'artifacts/diagnostic.pt'),toy_sha256=digest(out/'counterexample.json'),
                wall_seconds=time.monotonic()-start,GPU=torch.cuda.get_device_name(),
                limitation='Matched last-block interventions on exposed source construction inputs; not fresh confirmation, full-model AGA accuracy, or real merging. Full PaRaMS remains outside budget.')
    save_json(out/'results.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--protocol',required=True);run(p.parse_args().protocol)
