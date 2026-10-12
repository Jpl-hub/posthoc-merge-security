"""One public Query replacement control, not a protection construction."""
import copy
import hashlib
import json
import sys
from pathlib import Path

import torch
from torchvision.datasets import DTD, MNIST

ROOT = Path('/root/autodl-tmp/merge_security_20261009')
OUT = ROOT / 'outputs/query_replacement'
protocol = json.loads((OUT / 'protocol.json').read_text())
assert not (OUT / 'execution.json').exists(), 'Do not overwrite a completed run'
(OUT / 'artifacts').mkdir(exist_ok=False)
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(__file__) == protocol['runner_sha256']
for path, pin in protocol['source_sha256'].items():
    assert sha(ROOT / path) == pin, path
torch.set_grad_enabled(False)
torch.set_num_threads(4)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
sys.path.insert(0, str(ROOT / 'experiments'))
from clip_benchmark import checkpoint_loader, digest, evaluate, merge, paired_counts
from subspace_probe import make_loader, save_json

load = checkpoint_loader(ROOT / 'external/secure-merge-attack')
cp = ROOT / 'checkpoints/ViT-B-32'
files = json.loads((ROOT / 'configs/checkpoints.json').read_text())['files']
names = ['zeroshot.pt', 'MNIST/finetuned.pt', 'DTD/finetuned.pt', 'head_MNIST.pt', 'head_DTD.pt']
hashes = {name: digest(cp / name) for name in names}
for name, value in hashes.items():
    assert value == next(e['sha256'] for e in files if e['path'] == 'ViT-B-32/' + name)
anchor, original, partner = [load(cp / name).cpu().eval() for name in names[:3]]
heads = {task: load(cp / f'head_{task}.pt') for task in ('MNIST', 'DTD')}
cleaned = copy.deepcopy(original).cpu().eval()
a = cleaned.model.visual.transformer.resblocks[0].attn
b = anchor.model.visual.transformer.resblocks[0].attn
a.in_proj_weight[:64].copy_(b.in_proj_weight[:64])
a.in_proj_bias[:64].copy_(b.in_proj_bias[:64])
state, reset_state = original.state_dict(), cleaned.state_dict()
prefix = 'model.visual.transformer.resblocks.0.attn.'
expected = {prefix + 'in_proj_weight', prefix + 'in_proj_bias'}
assert {k for k in state if not torch.equal(state[k], reset_state[k])} == expected
assert torch.equal(state[prefix + 'in_proj_weight'][64:], reset_state[prefix + 'in_proj_weight'][64:])
assert torch.equal(state[prefix + 'in_proj_bias'][64:], reset_state[prefix + 'in_proj_bias'][64:])

def fingerprint(model):
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode() + b'\0')
        h.update(str(tensor.dtype).encode() + b'\0')
        h.update(json.dumps(list(tensor.shape)).encode() + b'\0')
        h.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

fingerprints = {name: fingerprint(m) for name, m in [('anchor', anchor), ('original', original), ('public_Q_replacement', cleaned), ('partner', partner)]}
datasets = {
    'MNIST': MNIST(ROOT / 'data', train=False, transform=original.val_preprocess),
    'DTD': DTD(ROOT / 'data', split='test', partition=1, transform=original.val_preprocess),
}
assert {k: len(v) for k, v in datasets.items()} == {'MNIST': 10000, 'DTD': 1880}
loaders = {k: make_loader(v, batch_size=128) for k, v in datasets.items()}
save_json(OUT / 'freeze.json', dict(protocol_sha256=digest(OUT / 'protocol.json'),
    runner_sha256=sha(__file__), checkpoint_sha256=hashes, state_fingerprints=fingerprints,
    changed_fields=sorted(expected), freeze_before_evaluation=True))
rows, predictions = [], {}

def record(model, preset, stage, task):
    batches = []
    metrics = evaluate(model, heads[task], loaders[task], 'cuda',
                       prediction_sink=lambda p, y: batches.append((p, y)))
    pred, labels = [torch.cat([b[i] for b in batches]) for i in (0, 1)]
    key = preset + '/' + stage + '/' + task
    predictions[key] = dict(prediction=pred, target=labels)
    row = dict(preset=preset, stage=stage, dataset=task, **metrics)
    if preset == 'public_Q_replacement':
        control = predictions['original/' + stage + '/' + task]
        assert torch.equal(control['target'], labels)
        row['paired'] = paired_counts(control['prediction'].eq(labels), pred.eq(labels))
    if stage == 'partner_standalone':
        row['applies_to_both_presets'] = True
    rows.append(row)
    save_json(OUT / 'results.json', rows)
    torch.save(predictions, OUT / 'artifacts/predictions.pt')
    print('QUERY_REPLACEMENT_RESULT', json.dumps(row), flush=True)

record(original, 'original', 'standalone', 'MNIST')
record(cleaned, 'public_Q_replacement', 'standalone', 'MNIST')
record(partner, 'original', 'partner_standalone', 'DTD')
for preset, model in [('original', original), ('public_Q_replacement', cleaned)]:
    fused = merge(anchor, model, partner, protocol['coefficient'])
    record(fused, preset, 'direct', 'MNIST')
    record(fused, preset, 'direct', 'DTD')
    del fused
assert fingerprint(original) == fingerprints['original']
assert fingerprint(partner) == fingerprints['partner']
save_json(OUT / 'execution.json', dict(completed=True, rows=len(rows),
    predictions_sha256=digest(OUT / 'artifacts/predictions.pt'),
    checkpoint_sha256=hashes, runner_sha256=sha(__file__),
    protocol_sha256=digest(OUT / 'protocol.json'), gpu=torch.cuda.get_device_name(),
    optimizer_steps=0, new_protected_weights=0, saved_model_weights=0,
    source_parent=protocol['source_parent'], AGA_run=False,
    scope='Matched public replacement cost only; no protected-model security experiment'))
