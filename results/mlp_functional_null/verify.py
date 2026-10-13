"""Independent CPU spectrum/formula checks and fresh native calibration replay."""
import hashlib
import inspect
import json
import sys
import time
from pathlib import Path

import numpy as np
import scipy.linalg as la
from scipy.special import expit
import torch
from torchvision.datasets import MNIST

ROOT = Path('/root/autodl-tmp/merge_security_20261009')
OUT = ROOT / 'outputs/mlp_functional_null'
sys.path.insert(0, str(ROOT / 'experiments'))
from clip_benchmark import checkpoint_loader, digest
from subspace_probe import make_loader


@torch.no_grad()
def main():
    start = time.time()
    assert not (OUT / 'independent_validation.json').exists()
    p = json.loads((OUT / 'protocol.json').read_text())
    result = json.loads((OUT / 'results.json').read_text())
    assert digest(OUT / 'protocol.json') == result['protocol_sha256']
    assert digest(OUT / 'audit.py') == p['runner_sha256'] == result['runner_sha256']
    assert digest(OUT / 'native_capsule.pt') == result['capsule_sha256']
    for name, expected in p['source_sha256'].items():
        assert digest(ROOT / name) == expected
    raw = torch.load(OUT / 'native_capsule.pt', map_location='cpu', weights_only=True)
    ids = raw['ids']
    ordered = torch.randperm(60000, generator=torch.Generator().manual_seed(p['seed'])).tolist()
    assert ids == {'defender': ordered[:256], 'confirmation': ordered[256:512]}
    assert not set(ids['defender']) & set(ids['confirmation'])
    w = raw['w'].numpy()
    a = raw['a'].numpy()
    mean = raw['summaries']['defender']['input_sum'].numpy()/12800
    np.testing.assert_allclose(a, mean/la.norm(mean), rtol=1e-12, atol=1e-14)
    v = raw['weakest_direction'].numpy()
    np.testing.assert_allclose(la.norm(v), 1, atol=1e-12)
    native = raw['selected']['defender']
    z = native['z'].numpy()
    t = native['x'].numpy() @ a
    sig = expit(1.702*z)
    d = sig + 1.702*z*sig*(1-sig)
    jacobian = np.concatenate([w*(t[i]*d[i])[None, :] for i in range(6)])
    spectrum = la.svdvals(jacobian, check_finite=True)
    np.testing.assert_allclose(spectrum, raw['singular_values'].numpy(), rtol=1e-7, atol=1e-10)
    tolerance = np.finfo(np.float64).eps*max(jacobian.shape)*spectrum[0]
    rank = int(np.count_nonzero(spectrum > tolerance))
    assert rank == result['spectrum']['numerical_rank']
    min_norm = float(la.norm(jacobian@v))
    np.testing.assert_allclose(min_norm, spectrum[-1], rtol=1e-7, atol=1e-10)
    del jacobian
    print('INDEPENDENT_SPECTRUM', rank, float(spectrum[-1]), flush=True)
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3090'
    checkpoint = ROOT/'checkpoints/ViT-B-32/MNIST/finetuned.pt'
    assert digest(checkpoint) == p['checkpoint_sha256']
    for file, expected in p['dataset_sha256'].items():
        assert digest(ROOT/'data/MNIST/raw'/file) == expected
    model = checkpoint_loader(ROOT/'external/secure-merge-attack')(checkpoint, map_location='cpu').eval()
    block = model.model.visual.transformer.resblocks[0]
    assert digest(inspect.getfile(type(block))) == p['open_clip_model_sha256']
    np.testing.assert_array_equal(w, block.mlp.c_proj.weight.double().numpy())
    # A central difference through the installed activation, not the audit derivative.
    finite_checks = {}
    for name, sample in raw['selected'].items():
        zz = sample['z'].numpy()
        factor = sample['x'].numpy()@a
        ss = expit(1.702*zz)
        dd = ss+1.702*zz*ss*(1-ss)
        local = ((dd*v)@w.T)*factor[:,None]
        dz = torch.from_numpy(factor[:,None]*v)
        zz_t = torch.from_numpy(zz)
        h = 1e-5
        central = ((block.mlp.gelu(zz_t+h*dz)-block.mlp.gelu(zz_t-h*dz))/(2*h)).numpy()@w.T
        error = float(np.max(np.abs(central-local)))
        assert error < 3e-9, (name, error)
        rms = float(np.sqrt(np.mean(np.sum(local**2,axis=1))))
        np.testing.assert_allclose(rms, result['local_response'][name]['response_rms'], rtol=1e-8,atol=1e-12)
        finite_checks[name] = dict(max_abs_error=error, response_rms=rms)
    dataset = MNIST(ROOT/'data',train=True,transform=model.val_preprocess,download=False)
    fresh = {}
    current = {}

    def capture(_module, _inputs, value):
        # Different capture point: recompute c_fc affine output from LN2 output.
        x = value.detach()
        zz = torch.nn.functional.linear(x, block.mlp.c_fc.weight, block.mlp.c_fc.bias).double()
        flat = zz.reshape(-1,3072)
        current['minimum'] = torch.minimum(current['minimum'],flat.min(0).values)
        current['maximum'] = torch.maximum(current['maximum'],flat.max(0).values)
        current['input_sum'].add_(x.double().reshape(-1,768).sum(0))
        if current['images']==0:
            current['x']=torch.stack([x[t,i].double() for i in range(3) for t in p['tokens']]).cpu()
            current['z']=torch.stack([zz[t,i] for i in range(3) for t in p['tokens']]).cpu()
        current['images']+=x.shape[1]

    model.cuda()
    handle=block.ln_2.register_forward_hook(capture)
    try:
        for name, indices in ids.items():
            current=dict(minimum=torch.full((3072,),float('inf'),device='cuda',dtype=torch.float64),
                maximum=torch.full((3072,),-float('inf'),device='cuda',dtype=torch.float64),
                input_sum=torch.zeros(768,device='cuda',dtype=torch.float64),images=0)
            for images,_ in make_loader(dataset,indices,p['batch_size']):
                model(images.cuda(non_blocking=True))
            err={}
            for key in ('minimum','maximum','input_sum'):
                new=current[key].cpu().numpy(); old=raw['summaries'][name][key].numpy()
                np.testing.assert_allclose(new,old,rtol=1e-7,atol=1e-6)
                err[key]=float(np.max(np.abs(new-old)))
            for key in ('x','z'):
                new=current[key].numpy();old=raw['selected'][name][key].numpy()
                np.testing.assert_allclose(new,old,rtol=1e-7,atol=1e-6)
                err[key]=float(np.max(np.abs(new-old)))
            fresh[name]=dict(images=current['images'],max_abs_errors=err)
    finally:
        handle.remove();model.cpu()
    joint_min=torch.minimum(raw['summaries']['defender']['minimum'],raw['summaries']['confirmation']['minimum']).numpy()
    joint_max=torch.maximum(raw['summaries']['defender']['maximum'],raw['summaries']['confirmation']['maximum']).numpy()
    positive=joint_min>0; negative=joint_max<0
    assert np.count_nonzero(positive)==result['neuron_regime']['stable_positive']
    assert np.count_nonzero(negative)==result['neuron_regime']['stable_negative']
    np.testing.assert_allclose(float(np.sum(v[negative]**2)),result['neuron_regime']['weakest_direction_stable_negative_energy'],atol=1e-12)
    verification=dict(passed=True,verifier_sha256=digest(__file__),
        protocol_sha256=result['protocol_sha256'],capsule_sha256=result['capsule_sha256'],
        independent_method='SciPy CPU SVD and expit derivative; central difference through installed QuickGELU; fresh native LN2-hook affine replay',
        numerical_rank=rank,smallest_singular_value=float(spectrum[-1]),smallest_direction_response_norm=min_norm,
        finite_difference=finite_checks,fresh_replay=fresh,replay_examples=sum(v['images'] for v in fresh.values()),
        new_protected_weights=0,classification_rows=0,partner_access=False,seconds=time.time()-start)
    (OUT/'independent_validation.json').write_text(json.dumps(verification,indent=2)+'\n')
    print(json.dumps(verification),flush=True)


if __name__=='__main__':
    main()
