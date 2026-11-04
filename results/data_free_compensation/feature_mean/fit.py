"""Fit one visual projection on frozen target/base synthetic features for MNIST."""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

from _impl.io import digest, save_json
from _impl.models import apply, checked_models, initialize


def fit(X, logits, target, basis):
    """Dual of a minimum-Frobenius-norm multiclass margin problem."""
    n, k = len(target), logits.shape[1]
    competitors = np.tile(np.arange(k), (n,1))[np.arange(k) != target[:,None]].reshape(n,k-1)
    rows = np.arange(n)[:,None]
    native_gaps = logits[rows,target[:,None]] - logits[rows,competitors]
    assert native_gaps.min() > 0
    rival = competitors[rows[:,0], logits[n+rows,competitors].argmax(1)]
    old_gap = logits[n+rows[:,0],target]-logits[n+rows[:,0],rival]
    image = np.r_[np.repeat(rows[:,0],k-1),n+rows[:,0]]
    win = np.r_[np.repeat(target,k-1),rival]; lose = np.r_[competitors.ravel(),target]
    # Retain each image's original smallest native gap. Reverse the signed gap at the mean.
    rhs = np.r_[(native_gaps.min(1,keepdims=True)-native_gaps).ravel(),abs(old_gap)+old_gap]
    norm = np.linalg.norm(X,axis=1)[image]*np.linalg.norm(basis[:,win]-basis[:,lose],axis=0)
    rhs = rhs/norm
    def primal(alpha):
        weights = np.zeros_like(logits); np.add.at(weights,(image,win),alpha/norm); np.add.at(weights,(image,lose),-alpha/norm)
        return X.T @ (weights @ basis.T)
    def dual(alpha):
        V = primal(alpha); score = (X@V)@basis
        return .5*np.sum(V*V)-rhs@alpha, (score[image,win]-score[image,lose])/norm-rhs
    opt = minimize(dual,np.zeros_like(rhs),jac=True,method='L-BFGS-B',bounds=[(0,None)]*len(rhs),
                   options={'maxiter':1000,'ftol':1e-13,'gtol':1e-8})
    return primal(opt.x),dict(iterations=opt.nit,message=str(opt.message),solver_success=bool(opt.success),
                              max_normalized_violation=float(max(0,-dual(opt.x)[1].min())))


@torch.no_grad()
def main():
    initialize();out=Path('outputs/feature_mean_fit_20261104');out.mkdir(exist_ok=False)
    spec=json.loads(Path('results/data_free_compensation/synthetic_recovery/protocol.json').read_text())['cases'][0]
    ref=spec['synthetic_source'];assert digest(ref['path'])==ref['sha256']
    data=torch.load(ref['path'],map_location='cpu',weights_only=False)
    base, original, head, _=checked_models('MNIST'); pub=apply(original,spec['releases']['Defense2'])
    targets=data['teacher_class'].numpy(); features=[]; norm=original.val_preprocess.transforms[-1]
    for model in (pub,base):
        cls=[]; model.cuda(); hook=model.model.visual.ln_post.register_forward_pre_hook(lambda m,x:cls.append(x[0].cpu()))
        for pixels in data['final'].split(8):
            x=F.interpolate(pixels.cuda(),(224,224),mode='bilinear',align_corners=False)
            views=torch.cat([torch.roll(x,d,(2,3)) for d in [(0,0),(7,0),(-7,0),(0,7),(0,-7)]])
            model((views-x.new_tensor(norm.mean)[None,:,None,None])/x.new_tensor(norm.std)[None,:,None,None])
        hook.remove();model.cpu();features.append(torch.cat(cls).double())
    v=pub.model.visual;b=base.model.visual;mix=lambda p,q:(q+.3*(p-q)).double()
    Xn=v.ln_post.double()(features[0]); base_ln=b.ln_post
    Xm=F.layer_norm(sum(features)/2,(768,),mix(v.ln_post.weight.float(),base_ln.weight),mix(v.ln_post.bias.float(),base_ln.bias),v.ln_post.eps)
    P=v.proj.double();Pm=mix(v.proj,b.proj);W=head.weight.double().T;assert head.bias.abs().max()==0
    initial=torch.cat([Xn@P@W,Xm@Pm@W]).numpy()
    U,S,_=np.linalg.svd((W-W[:,[0]]).numpy(),full_matrices=False);U=U[:,S>S[0]*1e-12]
    V,info=fit(torch.cat([Xn,.3*Xm]).numpy(),initial,targets,U.T@W.numpy())
    proposal=(P+torch.from_numpy(V@U.T)).float();assert proposal.isfinite().all()
    edits=torch.load(spec['releases']['Defense2']['path'],map_location='cpu',weights_only=False)
    edits['model.visual.proj']=proposal;point=out/'point.pt';torch.save(edits,point)
    final=torch.stack([Xn@proposal.double()@W,Xm@mix(proposal,b.proj)@W]);t=torch.from_numpy(targets)
    torch.save(dict(target=t,initial=initial,final=final),out/'source_scores.pt')
    save_json(out/'freeze.json',dict(**info,path=str(point),sha256=digest(point),script_sha256=digest(__file__),source_sha256=ref['sha256'],
        source_views=len(t),source_native_matches=int((final[0].argmax(1)==t).sum()),source_mean_matches=int((final[1].argmax(1)==t).sum()),
        source_initial_native=int((initial[:len(t)].argmax(1)==targets).sum()),source_initial_mean=int((initial[len(t):].argmax(1)==targets).sum()),
        projection_relative_change=float((proposal.double()-P).norm()/P.norm()),real_construction_images=0,changed_from_current_release=['model.visual.proj']))
    print((out/'freeze.json').read_text())


if __name__=='__main__':
    main()
