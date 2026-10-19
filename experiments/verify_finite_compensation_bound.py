"""Independent NumPy finite readout, projection and dual-bound reconstruction."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch


def verify(source, report, output):
    assert not output.exists()
    result = json.loads(report.read_text())
    assert hashlib.sha256(source.read_bytes()).hexdigest() == result['source_sha256']
    data = torch.load(source, map_location='cpu', weights_only=False)
    p = {k: v.numpy() if torch.is_tensor(v) else v for k, v in data['parameters'].items()}
    h, star, z, joint = [data['states'][k].numpy() for k in ['h0', 'upper', 'z', 'joint']]
    linear = p['gamma'][:, None]*p['projection']; linear -= linear.mean(0)
    offset = p['beta']@p['projection']; head = p['head']-p['head'].mean(0)
    m, b = linear@head.T, offset@head.T
    def feature(x):
        sigma = np.sqrt(np.mean((x-x.mean(1, keepdims=True))**2, axis=1)+p['eps'])
        return x@linear+sigma[:, None]*offset
    f = feature(h); y = (f@head.T).argmax(1); n, d = h.shape; count = np.bincount(y, minlength=len(head))
    competitors = np.array([[c for c in range(len(head)) if c != t] for t in y])
    mg = m[:, y].T[:, :, None]-m[:, competitors].transpose(1, 0, 2)
    bg = b[y, None]-b[competitors]; numerator = f@head.T
    gaps = numerator[np.arange(n), y, None]-np.take_along_axis(numerator, competitors, axis=1)
    spans = np.stack([gaps, bg], -1)
    projection = np.eye(len(head)-1)-spans@np.linalg.pinv(spans, rcond=1e-10)
    q = np.einsum('nij,nj->ni', projection, np.einsum('nd,ndc->nc', star, mg))
    w = q/(gaps*gaps).sum(1, keepdims=True)/count[y, None]
    if np.linalg.norm(w): w /= np.linalg.norm(w)
    adjoint = np.einsum('ndc,nc->nd', mg, w).T@z
    lhs = abs(np.sum(w*q)); radius = result['remaining_B_distance']; norm = data['B_norm']
    lipschitz = np.linalg.norm(linear, 2)+np.linalg.norm(offset)/np.sqrt(d)
    values, ratios = [], []
    for row in result['rows']:
        restricted = row['family'] == 'B_input32_output_class'
        act = z@data['B_task_basis'].numpy().T if restricted else z
        dual = adjoint@data['B_task_basis'].numpy().T if restricted else adjoint
        tau = (np.linalg.norm(feature(star), axis=1)+lipschitz*norm*radius*np.linalg.norm(act, axis=1))/np.linalg.norm(f, axis=1)
        allowance = .5*np.sum(tau*(abs(w)*gaps).sum(1)); capacity = radius*norm*np.linalg.norm(dual)
        expected = [lhs, allowance, capacity, allowance+capacity]
        actual = [row[k] for k in ['dual_lhs', 'task_tolerance_allowance', 'parameter_capacity', 'budget_upper_rhs']]
        np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-8)
        assert row['infeasible_certificate'] == (lhs > allowance+capacity+1e-8*max(1., lhs))
        values.append(dict(family=row['family'], maximum_scalar_difference=float(np.max(abs(np.array(actual)-expected)))))
        if not restricted:
            true_ratio = np.linalg.norm(feature(joint), axis=1)/np.linalg.norm(f, axis=1)
            assert (tau+1e-10 >= true_ratio).all()
            ratios.append(dict(maximum_bound=float(tau.max()), maximum_frozen_actual_ratio=float(true_ratio.max())))
    bdelta = data['B_delta'].numpy()
    np.testing.assert_allclose(star+z@bdelta.T, joint, atol=1e-8, rtol=1e-8)
    validation = dict(status='passed', independent_NumPy_projection_and_bounds=True, source_n=n,
        scalar_checks=values, normalization_bound_looseness=ratios,
        source_sha256=result['source_sha256'], report_sha256=hashlib.sha256(report.read_bytes()).hexdigest(),
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        gradients=0, images=0, new_weights=0, scope='Frozen source diagnosis only; bounds remain inconclusive.')
    output.write_text(json.dumps(validation, indent=2)+'\n'); print(json.dumps(validation, indent=2))


def verify_projection_cone(root):
    """Rebuild finite cones in NumPy; check a solution or an infeasibility ray."""
    import time
    from scipy import sparse
    from clip_benchmark import digest
    from subspace_probe import save_json
    from task_probe_distance import checked_models,initialize
    from verify_covariance_noise import official_labels
    initialize();start=time.monotonic();root=Path(root)
    assert not (root/'independent_validation.json').exists()
    cfg=json.loads((root/'protocol.json').read_text());meta=json.loads((root/'results.json').read_text())
    for file,key in [('source_data.pt','source_raw_sha256'),('cone_A.npz','cone_A_sha256'),('cone_raw.npz','cone_raw_sha256')]:assert digest(root/file)==meta[key]
    data=torch.load(root/'source_data.pt',map_location='cpu',weights_only=False)
    raw=np.load(root/'cone_raw.npz');A=sparse.load_npz(root/'cone_A.npz');b=raw['b']
    anchor,expert,head,pins=checked_models(cfg['target']);assert pins==meta['checkpoint_sha256']
    P=data['P'].numpy();np.testing.assert_array_equal(P,expert.model.visual.proj.double().numpy())
    H=data['head'].numpy();np.testing.assert_array_equal(H,head.weight.double().numpy())
    ids=data['indices'];assert ids==json.loads((root/'source_split.json').read_text())['indices']
    assert not set(ids)&set(cfg['extra_excluded_indices'])
    split=json.loads((root/'source_split.json').read_text());excluded=set(cfg['extra_excluded_indices'])
    boundary=json.loads(Path('results/task_boundary_audit/split_independence_audit.json').read_text())
    static={}
    for row in boundary['records']:static.setdefault(row['path'],[]).append(row['group'])
    for path,pin in split['excluded_metadata_sha256'].items():
        assert digest(path)==pin,path
        if path in static:
            value=json.loads(Path(path).read_text()) if Path(path).suffix=='.json' else torch.load(path,map_location='cpu',weights_only=False)['indices']
            for group in static[path]:excluded.update(value[group])
        else:
            value=json.loads(Path(path).read_text())
            groups=(list(value.values()) if isinstance(value,dict) else [value]) if Path(path).name=='training_indices.json' else [value.get('indices',[])]
            for group in groups:
                if isinstance(group,list):excluded.update(int(x) for x in group)
    for path in cfg['extra_index_exclusions']:excluded.update(json.loads(Path(path).read_text())['indices'])
    assert len(excluded)==split['excluded_count']
    pool=sorted(set(range(60000))-excluded)
    expected=np.random.default_rng(cfg['construction_seed']).choice(pool,cfg['source_n'],replace=False).tolist()
    assert expected==ids and not set(ids)&excluded
    np.testing.assert_array_equal(data['labels'].numpy(),official_labels(cfg['target'],True)[ids[:cfg['fit_n']]])
    U,Q=data['projection_basis'].numpy(),data['head_basis'].numpy()
    np.testing.assert_allclose(U.T@U,np.eye(len(U.T)),atol=1e-12)
    np.testing.assert_allclose(Q.T@Q,np.eye(len(Q.T)),atol=1e-12)
    teacher=data['before'].numpy().astype(float);old=data['original_X'].numpy().astype(float)@P
    f=old/np.linalg.norm(old,axis=1,keepdims=True);control=float(abs(f@H.T-teacher).max());assert control<=cfg['full_model_score_tolerance']
    base=data['prefix_X'].numpy().astype(float)@P;Z=np.linalg.norm(P)*data['prefix_X'].numpy().astype(float)@U
    y=teacher.argmax(1);gap=teacher[np.arange(len(y)),y,None]-teacher;npar=A.shape[1];q=Q.shape[1];eps=cfg['source_reserve']
    remaining=meta['remaining_P_distance'];errors=[]
    for c in [np.zeros(npar),np.arange(1,npar+1,dtype=float)*remaining/(2*np.linalg.norm(np.arange(1,npar+1)))]:
        v=base+Z@c.reshape(cfg['rank'],q)@Q.T;perp=base-(base@Q)@Q.T
        expected=[];upper=[]
        for i in range(len(y)):
            for k in range(len(H)):
                if k==y[i]:continue
                a=H[y[i]]-H[k];scale=(1-eps)*gap[i,k]
                expected.extend(np.r_[a@v[i],scale*(v[i]@Q),scale*np.linalg.norm(perp[i])])
                upper.append((1+eps)*gap[i,k]*(f[i]@v[i])-a@v[i])
            upper.append(f[i]@v[i]-.5*np.linalg.norm(old[i]))
        expected=np.r_[expected,upper,remaining,c]
        error=float(abs(expected-(b-A@c)).max());assert error<1e-7;errors.append(error)
    B=expert.model.visual.transformer.resblocks[0].mlp.c_proj.weight.double().numpy()
    actual_D=float(np.linalg.norm(data['B_edit'].numpy()-B)/np.linalg.norm(B))
    assert abs(actual_D-meta['prefix_distance'])<1e-10
    B0=anchor.model.visual.transformer.resblocks[0].mlp.c_proj.weight.double().numpy()
    vb=np.linalg.svd(B-B0,full_matrices=False)[2][:cfg['rank']]
    deltaB=data['B_edit'].numpy()-B
    complement_leak=float(np.linalg.norm(deltaB@vb.T)/np.linalg.norm(deltaB))
    pu=np.linalg.svd(P-anchor.model.visual.proj.double().numpy(),full_matrices=False)[0][:,:cfg['rank']]
    basis_error=float(abs(U@U.T-pu@pu.T).max());assert basis_error<1e-8
    prefix=data['prefix_scores'].numpy().astype(float);ds=prefix-teacher
    drift=(ds[np.arange(len(y)),y,None]-ds)/np.where(gap>0,gap,1);drift[np.arange(len(y)),y]=0
    truth=data['labels'].numpy();prediction=prefix.argmax(1)
    report=dict(status='independent_cone_assembly_checked',solver_status=meta['status'],assembly_errors=errors,
        original_head_error=control,prefix_D=actual_D,source_sampling_reproduced=True,source_n=len(ids),excluded_source_indices=len(excluded),
        prefix_right_task32_response_fraction=complement_leak,projection_task32_basis_error=basis_error,
        fit_n=len(y),original_fit_correct=int((y==truth).sum()),prefix_fit_correct=int((prediction==truth).sum()),
        prefix_fit_changed=int((prediction!=y).sum()),prefix_fit_R=float(abs(drift).max()),
        prefix_per_class=[dict(class_id=k,n=int((truth==k).sum()),lost=int(((y==truth)&(prediction!=truth)&(truth==k)).sum()),
            gained=int(((y!=truth)&(prediction==truth)&(truth==k)).sum())) for k in range(len(H))],
        new_model_forwards=0,new_candidates=0,new_TA_calls=0,new_AGA_calls=0,
        source_sha256=digest(__file__),input_result_sha256=digest(root/'results.json'),raw_sha256=meta['cone_raw_sha256'])
    if meta['status']=='PrimalInfeasible':
        z=raw['z'].copy();stationarity=float(abs(A.T@z).max());dot=float(b@z);at=0;violation=0.
        def project_soc(t):
            norm=np.linalg.norm(t[1:])
            if norm<=t[0]:p=t.copy()
            elif norm<=-t[0]:p=np.zeros_like(t)
            else:p=.5*(norm+t[0])*np.r_[1.,t[1:]/norm]
            p[0]+=1e-12*max(1.,np.linalg.norm(p))
            return p
        for _ in range(len(y)*(len(H)-1)):
            t=z[at:at+q+2];violation=max(violation,float(np.linalg.norm(t[1:])-t[0]));z[at:at+q+2]=project_soc(t);at+=q+2
        count=len(y)*len(H);violation=max(violation,float(-z[at:at+count].min()))
        z[at:at+count]=np.maximum(z[at:at+count],0.)+1e-12;at+=count
        t=z[at:];assert len(t)==npar+1;violation=max(violation,float(np.linalg.norm(t[1:])-t[0]));z[at:]=project_soc(t)
        # A compact variable ball gives a certificate even with nonzero A.T z:
        # feasible => b.T z = x.T A.T z + s.T z >= -r*||A.T z||.
        projected_dot=float(b@z);residual_bound=remaining*float(np.linalg.norm(A.T@z))
        margin=-projected_dot-residual_bound
        report.update(dual_certificate_b_dot_z=dot,dual_stationarity_max=stationarity,dual_cone_violation=max(0.,violation),
            projected_dual_b_dot_z=projected_dot,parameter_ball_residual_allowance=residual_bound,bounded_dual_certificate_margin=margin,
            infeasibility_certificate_verified=margin>1e-5*(1+abs(projected_dot)+residual_bound),
            scope='Numerical certificate with explicit stationarity residual allowance on the known parameter ball and projection into self-dual cones. Excludes only this fixed source prefix,288-variable readout family,and sufficient epsilon0.25 cone; not the original0.5 task band or all defenses. Not a formal interval-arithmetic certificate.')
        np.savez_compressed(root/'projected_dual_certificate.npz',z=z)
        assert report['infeasibility_certificate_verified']
    elif meta['status']=='Solved':
        point=torch.load(root/'feasibility_point.pt',map_location='cpu',weights_only=False)
        v=data['prefix_X'].numpy().astype(float)@point['P_edit'].numpy().astype(float)
        s=v/np.linalg.norm(v,axis=1,keepdims=True)@H.T;delta=s-teacher
        r=(delta[np.arange(len(y)),y,None]-delta)/np.where(gap>0,gap,1);r[np.arange(len(y)),y]=0
        np.testing.assert_allclose(s,point['source_scores'].numpy(),atol=1e-8)
        report.update(actual_source_R=float(abs(r).max()),actual_P_distance=float(np.linalg.norm(point['P_edit'].numpy()-P)/np.linalg.norm(P)),
            scope='512 source feasibility only;4096 source/task-SVD/fresh-confirmation/full-task/TA/AGA gates remain. Not a validated defense.')
        assert report['actual_source_R']<=cfg['task_budget']
        assert (actual_D**2+report['actual_P_distance']**2)**.5<=cfg['parameter_budget']
    else:
        report['scope']='Solver did not return a fully solved point or a full infeasibility certificate; retain numerical/limit status without a scientific feasibility claim.'
    report['wall_seconds']=time.monotonic()-start;save_json(root/'independent_validation.json',report);print(json.dumps(report),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('outputs/phase3/finite_compensation_source.pt'))
    parser.add_argument('--report', type=Path, default=Path('outputs/finite_compensation_bound.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/finite_compensation_bound_validation.json'))
    parser.add_argument('--projection-cone',type=Path)
    args = parser.parse_args()
    if args.projection_cone:verify_projection_cone(args.projection_cone)
    else:verify(args.source, args.report, args.output)
