"""Independent NumPy chord, directional curvature and readout certificates."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import expit

from clip_benchmark import digest
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_cross_module import ln
from verify_levelset_audits import convert, raw_scores


def verify(output):
    out=Path(output);report=json.loads((out/'results.json').read_text());path=out/'diagnostic.pt'
    assert digest(path)==report['diagnostic_sha256'];raw=torch.load(path,map_location='cpu');p=convert(raw['parameters']);checks=[]
    labels=official_labels(report['target'],True)
    for rec in report['records']:
        data=raw['cases'][rec['case']];cache=convert(data['cache']);d=convert(data['delta']);h0=cache['h']
        x=ln(cache['r'],p['gamma2'],p['beta2'],p['eps2']);pre=x@p['A'].T+p['a'];z=pre*expit(1.702*pre)
        def hidden(t):
            if 'A' in d:
                a=x@(p['A']+t*d['A']).T+p['a'];zz=a*expit(1.702*a)
                return h0-z@p['B'].T+zz@(p['B']+t*d['B']).T
            r=cache['r']+t*cache['o_input']@d['O'].T
            a=ln(r,p['gamma2'],p['beta2'],p['eps2'])@p['A'].T+p['a'];zz=a*expit(1.702*a)
            return h0-cache['r']-z@p['B'].T+r+zz@(p['B']+t*d['B']).T
        hm,h1=hidden(.5),hidden(1.)
        for name,value in [('hidden0',h0),('hidden_midpoint',hm),('hidden1',h1)]:
            np.testing.assert_allclose(value,data[name].numpy(),atol=1e-9,rtol=1e-9)
        before=raw_scores(h0,p);mid=raw_scores(hm,p);after=raw_scores(h1,p);y=before.argmax(1);rows=np.arange(len(y))
        np.testing.assert_array_equal(y,after.argmax(1));assert int((mid.argmax(1)!=y).sum())==rec['source_midpoint_changed']
        np.testing.assert_array_equal(labels[data['indices']],data['labels'].numpy())
        # A small directional finite difference validates exact autograd curvature;
        # it is not a fit of an endpoint quadratic model.
        h=1e-3;first=(hidden(h)-hidden(-h))/(2*h);second=(hidden(h)-2*h0+hidden(-h))/(h*h)
        first_error=float(np.linalg.norm(first-data['first_derivative'].numpy())/np.linalg.norm(first))
        second_error=float(np.linalg.norm(second-data['second_derivative'].numpy())/np.linalg.norm(second))
        assert first_error<1e-3
        if 'O' in d:
            centered=cache['r']-cache['r'].mean(1,keepdims=True)
            dr=cache['o_input']@d['O'].T;dc=dr-dr.mean(1,keepdims=True)
            sd=np.sqrt(np.mean(centered**2,1,keepdims=True)+p['eps2'])
            a=np.mean(centered*dc,1,keepdims=True);b=np.mean(dc**2,1,keepdims=True)
            xp=(dc/sd-centered*a/sd**3)*p['gamma2']
            xpp=(-2*dc*a/sd**3-centered*b/sd**3+3*centered*a*a/sd**5)*p['gamma2']
            dp=xp@p['A'].T;ddp=xpp@p['A'].T
        else:
            dp=x@d['A'].T;ddp=np.zeros_like(dp)
        sig=expit(1.702*pre);gp=sig+1.702*pre*sig*(1-sig)
        gpp=3.404*sig*(1-sig)+1.702**2*pre*sig*(1-sig)*(1-2*sig)
        exact_second=(gpp*dp*dp+gp*ddp)@p['B'].T+2*(gp*dp)@d['B'].T
        analytic_error=float(np.linalg.norm(second-exact_second)/np.linalg.norm(second));assert analytic_error<1e-3
        chord=hm-.5*(h0+h1)
        corrected_chord_error=float(np.linalg.norm(chord+exact_second/8)/np.linalg.norm(chord))
        m=p['gamma'][:,None]*(p['projection']@p['head'].T);m-=m.mean(0,keepdims=True)
        bias=p['beta']@p['projection']@p['head'].T;dim=h0.shape[1]
        centered=h0-h0.mean(1,keepdims=True);centered_m=hm-hm.mean(1,keepdims=True)
        sd0=np.sqrt(np.mean(centered**2,1)+p['eps']);sdm=np.sqrt(np.mean(centered_m**2,1)+p['eps'])
        v0=h0@m+sd0[:,None]*bias;vm=hm@m+sdm[:,None]*bias
        np.testing.assert_array_equal(v0.argmax(1),y);np.testing.assert_array_equal(vm.argmax(1),mid.argmax(1))
        gaps0=v0[rows,y,None]-v0;gapsm=vm[rows,y,None]-vm
        contrasts=m[:,y].T[:,None,:]-m.T[None,:,:];norm=np.linalg.norm(contrasts,axis=2);bd=bias[y,None]-bias
        lip=norm+np.abs(bd)/np.sqrt(dim);scale=np.linalg.norm(centered,axis=1);cm=np.linalg.norm(centered_m,axis=1)
        mask=np.eye(len(bias),dtype=bool)[y]
        clean=np.divide(gaps0,lip,out=np.zeros_like(lip),where=lip>1e-20);clean[mask]=np.inf
        corrupt=np.divide(np.maximum(-gapsm,0),lip,out=np.zeros_like(lip),where=lip>1e-20);corrupt[mask]=0
        fixed=np.minimum(clean.min(1),corrupt.max(1))/scale
        q=2*norm*scale[:,None]
        l0=scale[:,None]*lip+norm*scale[:,None]+np.sqrt(dim)*norm*sd0[:,None]
        lm=scale[:,None]*lip+norm*cm[:,None]+np.sqrt(dim)*norm*sdm[:,None]
        def root(value,l):
            v=np.maximum(value,0);return 2*v/np.maximum(l+np.sqrt(l*l+4*q*v),1e-30)
        clean=root(gaps0,l0);clean[mask]=np.inf;corrupt=root(-gapsm,lm);corrupt[mask]=0
        joint=np.minimum(clean.min(1),corrupt.max(1))
        np.testing.assert_allclose(fixed,data['certificates']['fixed_readout_radius'].numpy(),atol=1e-9,rtol=1e-8)
        np.testing.assert_allclose(joint,data['certificates']['joint_radius'].numpy(),atol=1e-9,rtol=1e-8)
        # The maximal root is a supremum. Strict radius inequalities are needed
        # for a guaranteed strict class order, including ties at the boundary.
        nonzero=np.flatnonzero(joint>0);witness_error=0.
        for i in nonzero:
            candidate=np.flatnonzero(gapsm[i]<0)
            k=max(candidate,key=lambda c:root(-gapsm,lm)[i,c])
            rad=joint[i]*(1-1e-8)
            assert gapsm[i,k]+lm[i,k]*rad+q[i,k]*rad*rad<1e-9
            others=np.arange(len(bias))!=y[i]
            assert np.min(gaps0[i,others]-l0[i,others]*rad-q[i,others]*rad*rad)>-1e-8
        checks.append({'case':rec['case'],'midpoint_changed':rec['source_midpoint_changed'],
            'first_derivative_relative_fd_error':first_error,'second_derivative_relative_fd_error':second_error,
            'original_autograd_second_derivative_valid':second_error<1e-3,
            'analytic_second_derivative_relative_fd_error':analytic_error,
            'corrected_quadratic_chord_relative_error':corrected_chord_error,
            'exact_readout_sorting':'passed','conditional_radius_roots':'passed','nonzero_joint_radius_count':len(nonzero),
            'strict_radius_inequality_required':True})
    result={'status':'passed_forward_and_certificates_with_corrected_analytic_curvature','checks':checks,'source_sha256':digest(__file__),'diagnostic_sha256':digest(path),
        'initial_verifier_failure_preserved':'outputs/phase2_source_geometry_verify.log; source commit613a805',
        'correction':'Original O/B LayerNorm automatic second derivative disagrees with independent finite difference by2.75%; exact analytic derivative replaces it for curvature interpretation. Frozen weights, forward scores and original diagnostic file are unchanged. No error tolerance was relaxed.',
        'new_full_model_forward_instances':0,'new_release_weights':0,'publication_updates':0,'test_used':False,'partner_used':False,
        'scope':'Independent NumPy nonlinear source tails, official train labels, directional Hessian finite differences and analytic neighborhood certificate roots. No assertion that unknown peer fusions lie inside the certified neighborhoods.'}
    save_json(out/'independent_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True);args=parser.parse_args();torch.set_num_threads(4);verify(args.output)
