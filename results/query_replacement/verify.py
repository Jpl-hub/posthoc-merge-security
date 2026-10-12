"""Recount raw predictions, parse official labels, and independently replay 128."""
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
OUT = ROOT / 'outputs/query_replacement'
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (OUT / 'independent_validation.json').exists()
execution = json.loads((OUT / 'execution.json').read_text())
freeze = json.loads((OUT / 'freeze.json').read_text())
rows = json.loads((OUT / 'results.json').read_text())
assert execution['completed'] and len(rows) == execution['rows'] == 7
assert sha(OUT / 'audit.py') == execution['runner_sha256'] == freeze['runner_sha256']
assert sha(OUT / 'protocol.json') == execution['protocol_sha256'] == freeze['protocol_sha256']
assert sha(OUT / 'artifacts/predictions.pt') == execution['predictions_sha256']
raw = torch.load(OUT / 'artifacts/predictions.pt', map_location='cpu')
mnist_path = ROOT / 'data/MNIST/raw/t10k-labels-idx1-ubyte'
blob = mnist_path.read_bytes()
assert struct.unpack('>II', blob[:8]) == (2049, 10000)
mnist_labels = np.frombuffer(blob[8:], dtype=np.uint8).astype(np.int64)
dtd_root = ROOT / 'data/dtd/dtd'
dtd_list = dtd_root / 'labels/test1.txt'
classes = sorted(p.name for p in (dtd_root / 'images').iterdir() if p.is_dir())
assert len(classes) == 47
dtd_labels = np.array([classes.index(s.split('/')[0]) for s in dtd_list.read_text().splitlines()])
assert len(dtd_labels) == 1880
expected = {'MNIST': mnist_labels, 'DTD': dtd_labels}
counts, changes = [], []
for row in rows:
    key = '/'.join(row[k] for k in ['preset', 'stage', 'dataset'])
    p, y = raw[key]['prediction'].numpy(), raw[key]['target'].numpy()
    np.testing.assert_array_equal(y, expected[row['dataset']])
    assert p.shape == y.shape and np.isfinite(p).all()
    assert p.min() >= 0 and p.max() < {'MNIST': 10, 'DTD': 47}[row['dataset']]
    correct = int((p == y).sum())
    assert correct == row['correct'] and len(y) == row['n']
    assert row['top1'] == correct / len(y)
    counts.append(dict(key=key, correct=correct, n=len(y)))
    if row['preset'] == 'public_Q_replacement':
        ref = raw['original/' + row['stage'] + '/' + row['dataset']]['prediction'].numpy()
        ok, old_ok = p == y, ref == y
        paired = dict(both_correct=int((old_ok & ok).sum()), lost=int((old_ok & ~ok).sum()),
                      gained=int((~old_ok & ok).sum()), both_wrong=int((~old_ok & ~ok).sum()))
        assert all(row['paired'][k] == v for k, v in paired.items())
        changes.append(dict(key=key, prediction_mismatches=int((p != ref).sum()), **paired))

torch.set_num_threads(4)
torch.set_grad_enabled(False)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
sys.path.insert(0, str(ROOT / 'experiments'))
from clip_benchmark import checkpoint_loader
load = checkpoint_loader(ROOT / 'external/secure-merge-attack')
cp = ROOT / 'checkpoints/ViT-B-32'
anchor = load(cp / 'zeroshot.pt').cpu().eval()
original = load(cp / 'MNIST/finetuned.pt').cpu().eval()
partner = load(cp / 'DTD/finetuned.pt').cpu().eval()
for name, pin in execution['checkpoint_sha256'].items():
    assert sha(cp / name) == pin
cleaned = copy.deepcopy(original)
state = {k: v.clone() for k, v in original.state_dict().items()}
base = anchor.state_dict()
for key in ['model.visual.transformer.resblocks.0.attn.in_proj_weight',
            'model.visual.transformer.resblocks.0.attn.in_proj_bias']:
    state[key][:64] = base[key][:64]
cleaned.load_state_dict(state, strict=True)

def fingerprint(m):
    h = hashlib.sha256()
    for name, tensor in sorted(m.state_dict().items()):
        for value in [name, str(tensor.dtype), json.dumps(list(tensor.shape))]:
            h.update(value.encode() + b'\0')
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

models = {'original': original, 'public_Q_replacement': cleaned, 'partner': partner}
for key, model in [('anchor', anchor), *models.items()]:
    assert fingerprint(model) == freeze['state_fingerprints'][key]
# Independent dictionary formula, not TaskVector or benchmark.merge.
for preset in ['original', 'public_Q_replacement']:
    target_state, partner_state = models[preset].state_dict(), partner.state_dict()
    fused_state = {k: a.clone() if a.dtype in [torch.int64, torch.uint8]
                   else a + 0.5 * ((target_state[k] - a) + (partner_state[k] - a))
                   for k, a in base.items()}
    fused = copy.deepcopy(anchor)
    fused.load_state_dict(fused_state, strict=True)
    models['direct/' + preset] = fused
datasets = {
    'MNIST': MNIST(ROOT / 'data', train=False, transform=original.val_preprocess),
    'DTD': DTD(ROOT / 'data', split='test', partition=1, transform=original.val_preprocess),
}
heads = {task: load(cp / f'head_{task}.pt') for task in datasets}
batches = {task: torch.stack([dataset[i][0] for i in range(128)]) for task, dataset in datasets.items()}
replays = []
for row in rows:
    task, preset, stage = row['dataset'], row['preset'], row['stage']
    model = models['partner'] if stage == 'partner_standalone' else models[
        'direct/' + preset if stage == 'direct' else preset]
    model.to('cuda').eval()
    head = heads[task].to('cuda').eval()
    features = model(batches[task].cuda())
    if head.normalize:
        features = features / features.norm(dim=-1, keepdim=True)
    pred = torch.nn.functional.linear(features, head.weight, head.bias).argmax(-1).cpu()
    key = '/'.join([preset, stage, task])
    assert torch.equal(pred, raw[key]['prediction'][:128]), key
    replays.append(dict(key=key, n=128, mismatches=0))
    model.cpu()
    head.cpu()
result = dict(status='passed', raw_prediction_sha256=execution['predictions_sha256'],
    official_label_checks='Raw MNIST IDX and official DTD test1/category directory order',
    MNIST_label_file_sha256=sha(mnist_path), DTD_test_list_sha256=sha(dtd_list),
    recounts=counts, paired_changes=changes, state_fingerprints_verified=True,
    independent_merge='Explicit checkpoint tensor arithmetic, not TaskVector',
    fresh_first_batch_replays=replays, replayed_examples=7 * 128,
    verifier_sha256=sha(__file__), new_protected_weights=0, optimizer_steps=0)
(OUT / 'independent_validation.json').write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result), flush=True)
