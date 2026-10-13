"""Independent algebra, official labels, saved-state replay, and finite response check."""

import copy
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.io import loadmat
from contextlib import nullcontext
from torchvision.datasets import MNIST, SVHN

ROOT = Path('/root/autodl-tmp/merge_security_20261009')
OUT = ROOT / 'outputs/feature_noise'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
meta = json.loads((OUT/'execution.json').read_text())
cfg = json.loads((OUT/'protocol.json').read_text())
assert meta['completed'] and meta['new_candidates']==1
assert not (OUT/'independent_validation.json').exists()
assert sha(OUT/'artifacts/release.pt')==meta['release_sha256']
assert sha(OUT/'artifacts/predictions.pt')==meta['predictions_sha256']
assert sha(ROOT/'configs/feature_noise.json')==meta['protocol_sha256']
assert cfg==json.loads((ROOT/'configs/feature_noise.json').read_text())
assert sha(ROOT/'experiments/feature_noise.py')==meta['runner_sha256']
for name,pin in cfg['source_sha256'].items():assert sha(ROOT/name)==pin
indices=json.loads((OUT/'training_indices.json').read_text())
assert {k:len(v) for k,v in indices.items()}==cfg['calibration_sizes']
assert len(set(sum(indices.values(),[])))==sum(map(len,indices.values()))
a=torch.load(OUT/'artifacts/release.pt',map_location='cpu')
z=a['fit_hidden'].numpy();mean=z.mean(0);centered=z-mean
np.testing.assert_allclose(mean,a['mean'].numpy(),atol=1e-12,rtol=0)
with nullcontext():
    gram=centered@centered.T
    values,vectors=np.linalg.eigh(gram)
    values=np.maximum(values[::-1],0);vectors=vectors[:,::-1]
    rank=int(np.searchsorted(np.cumsum(values)/values.sum(),cfg['retained_energy']))+1
    expected_basis=centered.T@vectors[:,:rank]/np.sqrt(values[:rank])
    saved_basis=a['basis'].numpy()
    np.testing.assert_allclose(saved_basis.T@saved_basis,np.eye(rank),atol=1e-10,rtol=0)
    basis_error=float(np.linalg.norm(saved_basis-expected_basis@(expected_basis.T@saved_basis)))
    assert basis_error<1e-5,basis_error
    shift=a['proxy_hidden'].numpy().mean(0)-mean
    v=shift-expected_basis@(expected_basis.T@shift);v/=np.linalg.norm(v)
    np.testing.assert_allclose(v,a['direction'].numpy(),atol=1e-7,rtol=0)
    native_rms=max(float(np.sqrt(np.mean(((a[k].numpy()-mean)@v)**2)))
                   for k in ['fit_hidden','scale_hidden'])
    h_rms=min(float(np.sqrt(np.mean(np.sum(a[k].numpy()**2,axis=1))))
              for k in ['fit_residual','scale_residual'])

torch.set_grad_enabled(False);torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
sys.path.insert(0,str(ROOT/'experiments'))
from clip_benchmark import checkpoint_loader
from feature_noise import capture
from subspace_probe import make_loader
load=checkpoint_loader(ROOT/'external/secure-merge-attack')
cp=ROOT/'checkpoints/ViT-B-32'
for name,pin in meta['checkpoint_sha256'].items():assert sha(cp/name)==pin
anchor=load(str(cp/'zeroshot.pt')).cpu().eval()
original=load(str(cp/'MNIST/finetuned.pt')).cpu().eval()
head=load(str(cp/'head_MNIST.pt')).cpu().eval()
layer=original.model.visual.transformer.resblocks[11].mlp.c_proj
weight=layer.weight.double().numpy()
alpha=min(cfg['native_residual_fraction']*h_rms/native_rms,
          cfg['weight_relative_cap']*float(np.linalg.norm(weight)))
assert abs(alpha-a['amplitude'])<1e-5
visual=original.model.visual
contrast=head.weight.double().numpy()-head.weight.double().numpy().mean(0)
matrix=visual.ln_post.weight.double().numpy()[:,None]*(visual.proj.double().numpy()@contrast.T)
matrix-=matrix.mean(0)
g=torch.randn(len(contrast),generator=torch.Generator().manual_seed(cfg['seed']),dtype=torch.float64).numpy()
u=matrix@g;u/=np.linalg.norm(u)
np.testing.assert_allclose(u,a['output'].numpy(),atol=1e-10,rtol=0)
delta=alpha*np.outer(u,v);bias_delta=-delta@mean
np.testing.assert_allclose(a['released_weight'].numpy(),(weight+delta).astype('float32'),atol=1e-7,rtol=0)
np.testing.assert_allclose(a['released_bias'].numpy(),(layer.bias.double().numpy()+bias_delta).astype('float32'),atol=2e-6,rtol=0)
released=copy.deepcopy(original)
new_layer=released.model.visual.transformer.resblocks[11].mlp.c_proj
new_layer.weight.copy_(a['released_weight']);new_layer.bias.copy_(a['released_bias'])
changed={k for k,t in released.state_dict().items() if not torch.equal(t,original.state_dict()[k])}
assert changed=={'model.visual.transformer.resblocks.11.mlp.c_proj.'+part for part in ['weight','bias']}

def idx_labels(path,n):
    b=path.read_bytes();assert struct.unpack('>II',b[:8])==(2049,n)
    return np.frombuffer(b[8:],dtype=np.uint8).astype('int64')
labels_m=idx_labels(ROOT/'data/MNIST/raw/t10k-labels-idx1-ubyte',10000)
labels_train=idx_labels(ROOT/'data/MNIST/raw/train-labels-idx1-ubyte',60000)
labels_s=loadmat(ROOT/'data/test_32x32.mat')['y'].reshape(-1).astype('int64')%10
raw=torch.load(OUT/'artifacts/predictions.pt',map_location='cpu')
rows=json.loads((OUT/'results.json').read_text());counts=[]
for r in rows:
    key='/'.join(r[k] for k in ['preset','stage','dataset'])
    pred,y=[raw[key][k].numpy() for k in ['prediction','target']]
    expected=(labels_train[indices['confirmation']] if r['stage']=='confirmation'
              else (labels_m if r['dataset']=='MNIST' else labels_s))
    np.testing.assert_array_equal(y,expected)
    correct=int(np.sum(pred==y));assert correct==r['correct'] and len(y)==r['n']
    assert r['top1']==correct/len(y)
    if r['preset']=='released':
        p0=raw['original/'+r['stage']+'/'+r['dataset']]['prediction'].numpy()
        old,new=p0==y,pred==y
        pair={'both_correct':int(np.sum(old&new)),'lost':int(np.sum(old&~new)),
              'gained':int(np.sum(~old&new)),'both_wrong':int(np.sum(~old&~new))}
        assert all(r['paired'][k]==v0 for k,v0 in pair.items())
        assert int(np.sum(p0!=pred))==r['changed_predictions']
    counts.append(dict(key=key,correct=correct,n=len(y)))

actual=new_layer.weight.double()-layer.weight.double()
models={'original':original,'released':released}
if meta['fusion_run']:
    partner=load(str(cp/'SVHN/finetuned.pt')).cpu().eval()
    models['partner']=partner
    for name in ['original','released','partner']:
        m=copy.deepcopy(original if name!='partner' else partner)
        m.load_state_dict(torch.load(OUT/f'artifacts/{name}_AGA.pt',map_location='cpu'),strict=True)
        models['attacked/'+name]=m
    base=anchor.state_dict()
    for stage in ['direct','AGA']:
        for name in ['original','released']:
            left=models[name if stage=='direct' else 'attacked/'+name].state_dict()
            right=models['partner' if stage=='direct' else 'attacked/partner'].state_dict()
            state={k:t+cfg['coefficient']*((left[k]-t)+(right[k]-t))
                   if t.is_floating_point() else t.clone() for k,t in base.items()}
            m=copy.deepcopy(anchor);m.load_state_dict(state,strict=True)
            models[stage+'/'+name]=m
    # Author MLP alignment only reorders this noise; no approximate inference needed.
    fc='model.visual.transformer.resblocks.11.mlp.c_fc.weight'
    down='model.visual.transformer.resblocks.11.mlp.c_proj.'
    src=original.state_dict()[fc];aligned=models['attacked/original'].state_dict()[fc]
    assert torch.equal(aligned,models['attacked/released'].state_dict()[fc])
    lookup={hashlib.sha256(row.numpy().tobytes()).hexdigest():i for i,row in enumerate(src)}
    assert len(lookup)==len(src)
    perm=[lookup[hashlib.sha256(row.numpy().tobytes()).hexdigest()] for row in aligned]
    actual=new_layer.weight.double()-layer.weight.double()
    after=models['attacked/released'].state_dict()[down+'weight'].double()-models['attacked/original'].state_dict()[down+'weight'].double()
    assert torch.equal(after,actual[:,perm])

datasets={'MNIST':MNIST(ROOT/'data',train=False,transform=original.val_preprocess)}
if meta['fusion_run']:datasets['SVHN']=SVHN(ROOT/'data',split='test',transform=original.val_preprocess)
train=MNIST(ROOT/'data',train=True,transform=original.val_preprocess)
heads={task:load(str(cp/f'head_{task}.pt')).cpu().eval() for task in datasets}
replays=[]
for r in rows:
    preset,stage,task=[r[k] for k in ['preset','stage','dataset']]
    name=('partner' if stage=='partner_standalone' else
          ('attacked/'+preset if stage=='AGA_standalone' else
           (stage+'/'+preset if stage in ['direct','AGA'] else preset)))
    dataset=train if stage=='confirmation' else datasets[task]
    ix=indices['confirmation'][:128] if stage=='confirmation' else list(range(128))
    images=torch.stack([dataset[i][0] for i in ix]).cuda()
    model=models[name].cuda().eval();h0=heads[task].cuda().eval()
    features=model(images)
    if h0.normalize:features=features/features.norm(dim=-1,keepdim=True)
    pred=torch.nn.functional.linear(features,h0.weight,h0.bias).argmax(-1).cpu()
    key='/'.join([preset,stage,task]);assert torch.equal(pred,raw[key]['prediction'][:128]),key
    model.cpu();h0.cpu();replays.append(dict(key=key,n=128,mismatches=0))

finite=[]
for context in (['native','direct','AGA'] if meta['fusion_run'] else ['native']):
    before=models['original' if context=='native' else context+'/original']
    after=models['released' if context=='native' else context+'/released']
    loader=make_loader(train,indices['confirmation'][:128],64)
    zb,hb=capture(before,loader);zr,hr=capture(after,loader)
    assert torch.equal(zb,zr)
    factor=1 if context=='native' else cfg['coefficient']
    dw=actual if context!='AGA' else actual[:,perm]
    direction=a['direction'] if context!='AGA' else a['direction'][perm]
    offset=a['mean'] if context!='AGA' else a['mean'][perm]
    expected=factor*((zb@dw.T)+(new_layer.bias.double()-layer.bias.double()))
    observed=hr-hb
    error=float((expected-observed).norm()/expected.norm())
    assert error<2e-4,(context,error)
    finite.append(dict(context=context,n=128,relative_error=error,
                       projected_input_rms=float(((zb-offset)@direction).square().mean().sqrt()),
                       observed_residual_rms=float(observed.square().sum(-1).mean().sqrt())))
result=dict(status='passed',verifier_sha256=sha(__file__),independent_algebra='NumPy Gram eigendecomposition, class direction and deployed weight reconstruction',
            input_rank=rank,basis_error=basis_error,amplitude=alpha,
            official_label_recounts=counts,fresh_replays=replays,replayed_examples=128*len(replays),
            finite_response=finite,AGA_noise_columns_only_permuted=bool(meta['fusion_run']),
            raw_prediction_sha256=meta['predictions_sha256'],release_sha256=meta['release_sha256'],
            new_candidates=0,optimizer_steps=0,public_cleaning_run=False)
(OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':'passed','replayed_examples':result['replayed_examples'],'finite_response':finite}),flush=True)
