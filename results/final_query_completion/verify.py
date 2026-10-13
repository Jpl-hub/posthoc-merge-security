"""Independent NumPy completion, official labels, and fresh prediction replay."""

import copy
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from torchvision.datasets import DTD, MNIST

ROOT = Path('/root/autodl-tmp/merge_security_20261009')
OUT = ROOT / 'outputs/final_query_completion'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (OUT / 'independent_validation.json').exists()
execution = json.loads((OUT / 'execution.json').read_text())
freeze = json.loads((OUT / 'freeze.json').read_text())
protocol = json.loads((OUT / 'protocol.json').read_text())
assert execution['completed'] and execution['new_rows'] == 6 and execution['cached_rows'] == 4
assert sha(ROOT / 'results/final_query_completion/audit.py') == execution['runner_sha256']
assert sha(OUT / 'protocol.json') == execution['protocol_sha256']
assert sha(OUT / 'artifacts/capsule.pt') == execution['capsule_sha256'] == freeze['capsule_sha256']
assert sha(OUT / 'artifacts/predictions.pt') == execution['predictions_sha256']
capsule = torch.load(OUT / 'artifacts/capsule.pt', map_location='cpu')
raw = torch.load(OUT / 'artifacts/predictions.pt', map_location='cpu')
rows = json.loads((OUT / 'results.json').read_text())
assert len(rows) == 10
indices = json.loads((OUT / 'training_indices.json').read_text())
assert {k:len(v) for k,v in indices.items()} == protocol['calibration_sizes']
assert len(set(sum(indices.values(), []))) == sum(map(len, indices.values()))
np_checks = []
for name, artifact in capsule['spaces'].items():
    u = artifact['basis']['u'].numpy()
    covariance = artifact['moment']['covariance'].numpy()
    values, vectors = np.linalg.eigh(covariance)
    values, vectors = np.maximum(values[::-1], 0), vectors[:, ::-1]
    rank = int(np.searchsorted(np.cumsum(values)/values.sum(), protocol['retained_energy']))+1
    expected_u = vectors[:, :rank]
    normal = capsule['ln1_normal'].numpy()
    extra = normal - expected_u @ (expected_u.T @ normal)
    if np.linalg.norm(extra)>1e-8:
        expected_u = np.column_stack((expected_u, extra/np.linalg.norm(extra)))
    projector_gap = float(np.linalg.norm(u@u.T-expected_u@expected_u.T))
    assert projector_gap < 1e-6, (name,projector_gap)
    w, w0 = [capsule[k].numpy().astype('float64') for k in ['query_weight','anchor_query_weight']]
    b = capsule['query_bias'].numpy().astype('float64')
    mean = artifact['basis']['mean'].numpy()
    delta = (w-w0) @ (np.eye(768)-u@u.T)
    expected_w, expected_b = w-delta, b+delta@mean
    saved_w = artifact['completed_weight'].numpy()
    saved_b = artifact['completed_bias'].numpy()
    np.testing.assert_allclose(saved_w,expected_w.astype('float32'),rtol=0,atol=1e-7)
    np.testing.assert_allclose(saved_b,expected_b.astype('float32'),rtol=0,atol=1e-7)
    np_checks.append(dict(basis=name,independent_rank=expected_u.shape[1],
        projector_gap=projector_gap,weight_error=float(np.max(np.abs(saved_w-expected_w))),
        bias_error=float(np.max(np.abs(saved_b-expected_b)))))

label_path = ROOT/'data/MNIST/raw/t10k-labels-idx1-ubyte'
blob = label_path.read_bytes()
assert struct.unpack('>II',blob[:8]) == (2049,10000)
mnist_labels = np.frombuffer(blob[8:],dtype=np.uint8).astype('int64')
dtd = ROOT/'data/dtd/dtd'
classes = sorted(p.name for p in (dtd/'images').iterdir() if p.is_dir())
assert len(classes)==47
dtd_list = dtd/'labels/test1.txt'
dtd_labels = np.array([classes.index(t.split('/')[0]) for t in dtd_list.read_text().splitlines()])
expected = dict(MNIST=mnist_labels,DTD=dtd_labels)
counts, changes = [], []
for r in rows:
    key='/'.join(r[k] for k in ['preset','stage','dataset'])
    p,y = [raw[key][k].numpy() for k in ['prediction','target']]
    np.testing.assert_array_equal(y,expected[r['dataset']])
    assert p.shape==y.shape and np.isfinite(p).all()
    correct=int(np.sum(p==y))
    assert correct==r['correct'] and len(y)==r['n'] and r['top1']==correct/len(y)
    counts.append(dict(key=key,correct=correct,n=len(y),fresh=r['fresh_evaluation']))
    if r['preset']!='original':
        old=raw['original/'+r['stage']+'/'+r['dataset']]['prediction'].numpy()
        old_ok,new_ok=old==y,p==y
        paired=dict(both_correct=int(np.sum(old_ok&new_ok)),lost=int(np.sum(old_ok&~new_ok)),
                    gained=int(np.sum(~old_ok&new_ok)),both_wrong=int(np.sum(~old_ok&~new_ok)))
        assert all(r['paired'][k]==v for k,v in paired.items())
        mismatches=int(np.sum(old!=p));assert mismatches==r['changed_predictions']
        changes.append(dict(key=key,changed_predictions=mismatches,**paired))

torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32=False
torch.backends.cudnn.allow_tf32=False
sys.path.insert(0,str(ROOT/'experiments'))
from clip_benchmark import checkpoint_loader
load=checkpoint_loader(ROOT/'external/secure-merge-attack')
cp=ROOT/'checkpoints/ViT-B-32'
for name,pin in execution['checkpoint_sha256'].items():assert sha(cp/name)==pin
anchor,original,partner=[load(cp/name).cpu().eval() for name in ['zeroshot.pt','MNIST/finetuned.pt','DTD/finetuned.pt']]
models={'original':original,'partner':partner}
for name,a in capsule['spaces'].items():
    model=copy.deepcopy(original)
    q=model.model.visual.transformer.resblocks[11].attn
    q.in_proj_weight[:768].copy_(a['completed_weight'])
    q.in_proj_bias[:768].copy_(a['completed_bias'])
    models[name]=model
def fingerprint(model):
    h=hashlib.sha256()
    for name,tensor in sorted(model.state_dict().items()):
        for value in [name,str(tensor.dtype),json.dumps(list(tensor.shape))]:
            h.update(value.encode()+b'\0')
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
for name in ['original','defender','attacker']:
    assert fingerprint(models[name])==freeze['state_fingerprints'][name]
assert fingerprint(anchor)==freeze['state_fingerprints']['anchor']
base,sp=anchor.state_dict(),partner.state_dict()
for name in ['original','defender','attacker']:
    target=models[name].state_dict()
    state={k:a.clone() if a.dtype in [torch.int64,torch.uint8]
           else a+0.5*((target[k]-a)+(sp[k]-a)) for k,a in base.items()}
    model=copy.deepcopy(anchor);model.load_state_dict(state,strict=True)
    models['direct/'+name]=model
datasets=dict(MNIST=MNIST(ROOT/'data',train=False,transform=original.val_preprocess),
              DTD=DTD(ROOT/'data',split='test',partition=1,transform=original.val_preprocess))
heads={task:load(cp/f'head_{task}.pt') for task in datasets}
batches={task:torch.stack([dataset[i][0] for i in range(128)]) for task,dataset in datasets.items()}
replays=[]
for r in rows:
    task,preset,stage=[r[k] for k in ['dataset','preset','stage']]
    model=models['partner' if stage=='partner_standalone' else
                 ('direct/'+preset if stage=='direct' else preset)].cuda().eval()
    head=heads[task].cuda().eval()
    features=model(batches[task].cuda())
    if head.normalize:features=features/features.norm(dim=-1,keepdim=True)
    p=torch.nn.functional.linear(features,head.weight,head.bias).argmax(-1).cpu()
    key='/'.join([preset,stage,task]);assert torch.equal(p,raw[key]['prediction'][:128]),key
    replays.append(dict(key=key,n=128,mismatches=0))
    model.cpu();head.cpu()
result=dict(status='passed',verifier_sha256=sha(__file__),
    independent_algebra='NumPy eigendecomposition and explicit projector completion',numpy_checks=np_checks,
    official_label_checks='Raw MNIST IDX and official DTD test1/category directory order',
    MNIST_label_sha256=sha(label_path),DTD_test_list_sha256=sha(dtd_list),
    recounts=counts,paired_changes=changes,independent_merge='Explicit tensor formula, not TaskVector',
    state_fingerprints_verified=True,
    fresh_replays=replays,replayed_examples=1280,
    raw_prediction_sha256=execution['predictions_sha256'],new_protected_weights=0,optimizer_steps=0)
(OUT/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'count_rows':len(counts),
                  'numpy_checks':np_checks,'replayed_examples':1280}),flush=True)
