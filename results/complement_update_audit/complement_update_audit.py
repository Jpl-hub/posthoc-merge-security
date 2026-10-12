"""CPU reanalysis of frozen block-5 matrices and existing activation moments.
No forward evaluation, candidate, protection weights, calibration or search.
"""
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path('experiments').resolve()))
from clip_benchmark import checkpoint_loader, digest

torch.set_num_threads(4)
torch.set_grad_enabled(False)
source = Path('outputs/middle_completion')
output = Path('outputs/complement_update_audit')
output.mkdir(exist_ok=False)
frozen = json.loads((source / 'freeze.json').read_text())
load = checkpoint_loader(Path('external/secure-merge-attack'))
root = Path('checkpoints/ViT-B-32')
for name, sha in frozen['checkpoint_sha256'].items():
    assert digest(root / name) == sha
for name, sha in frozen['artifacts_sha256'].items():
    assert digest(source / 'artifacts' / name) == sha
protocol = dict(source_parent='4f6f257',question='Is the original expert-specific complement response small compared with generic anchor response despite fusion activation?',basis='Two frozen MNIST PCAs from fda0a57; no new location, rank or direction',input='Only previously saved matrices and four activation moment records',new_CLIP_experiments=0,new_protected_weights=0,new_defense_candidates=0,device='CPU only; no model forward or CUDA call',interpretation='Preactivation RMS is descriptive, not a classification or security guarantee.')
(output / 'protocol.json').write_text(json.dumps(protocol,indent=2))
anchor = load(str(root / 'zeroshot.pt')).cpu().eval()
expert = load(str(root / 'MNIST/finetuned.pt')).cpu().eval()
base = anchor.model.visual.transformer.resblocks[5].mlp.c_fc
target = expert.model.visual.transformer.resblocks[5].mlp.c_fc
w0, w = base.weight.double(), target.weight.double()
tau = w - w0
moments = torch.load(source / 'artifacts/diagnostic_moments.pt',map_location='cpu')
rows, geometry = [], {}
for name in ['defender','attacker']:
    saved = torch.load(source / f'artifacts/{name}.pt',map_location='cpu')
    u, mu = saved['basis']['u'], saved['basis']['mean']
    c = torch.eye(768,dtype=torch.float64)-u@u.T
    original_c, anchor_c, task_c = w@c, w0@c, tau@c
    actual_dw = saved['completed']['weight'].double()-w
    actual_db = saved['completed']['bias'].double()-target.bias.double()
    geometry[name] = dict(task_update_norm=float(tau.norm()),complement_task_update_norm=float(task_c.norm()),anchor_complement_norm=float(anchor_c.norm()),complement_update_over_anchor_norm=float(task_c.norm()/anchor_c.norm()))
    grams = {key:matrix.T@matrix for key,matrix in [('expert',original_c),('anchor',anchor_c),('task',task_c),('delta',actual_dw)]}
    for context,m in moments.items():
        shift = m['mean']-mu
        second = m['covariance']+torch.outer(shift,shift)
        values = {key:float((gram*second.T).sum().clamp_min(0).sqrt()) for key,gram in grams.items() if key!='delta'}
        # This separately uses the actual frozen float32 edit and its bias,
        # rather than assuming a mathematically exact mean compensation.
        delta_mean = actual_dw@m['mean']+actual_db
        actual_square = (grams['delta']*m['covariance'].T).sum()+delta_mean.square().sum()
        scale = 1.0 if context.endswith('/native') else 0.5
        actual_rms = scale*float(actual_square.clamp_min(0).sqrt())
        expected_rms = scale*values['task']
        assert abs(actual_rms-expected_rms)<1e-5*max(1,expected_rms)
        rows.append(dict(basis=name,context=context,generic_anchor_complement_rms=values['anchor'],original_expert_complement_rms=values['expert'],task_update_complement_rms=values['task'],task_update_over_anchor_response=values['task']/values['anchor'],actual_completion_preactivation_rms=actual_rms,mathematical_completion_preactivation_rms=expected_rms,coefficient=scale))
result=dict(protocol=protocol,geometry=geometry,responses=rows,provenance=dict(script_sha256=digest(__file__),source_commit='fda0a57',input_sha256={str(p):digest(p) for p in [source/'artifacts/defender.pt',source/'artifacts/attacker.pt',source/'artifacts/diagnostic_moments.pt',source/'results.json',source/'independent_validation.json']},checkpoint_sha256=frozen['checkpoint_sha256']),completed=True)
(output/'results.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result),flush=True)
