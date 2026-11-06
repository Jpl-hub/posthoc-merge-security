"""Inspect the fixed attack trajectory; never update or select a defense."""
import copy
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from _impl import prodistill
from _impl.data import dataset
from _impl.io import check_files, digest, save_json
from _impl.models import apply, checked_models, initialize, load_checkpoint


cfg = json.loads(Path('results/data_free_compensation/recovery_trace_protocol.json').read_text())
check_files(cfg['sources'])
initialize()
start = time.monotonic()
out = Path(cfg['output']) / 'results'
out.mkdir(exist_ok=False)
save_json(out / 'protocol.json', cfg)
base, original, head, _ = checked_models('MNIST')
peer = load_checkpoint('SVHN/finetuned.pt')
peer_head = load_checkpoint('head_SVHN.pt')
training = json.loads(Path(cfg['shared_prefix_training']).read_text())
assert digest(cfg['shared_prefix_model']) == training['model_sha256']
trained = apply(base, dict(path=cfg['shared_prefix_model'], sha256=training['model_sha256']), full=True)
env = prodistill.author(cfg)
data = {t: dataset(t, 'data', original.val_preprocess, True) for t in cfg['calibration_indices']}
images = torch.stack([data[t][i][0]
                      for t, ids in cfg['calibration_indices'].items() for i in ids]).cuda()
del data
assert images.shape[0] == 128
assert hashlib.sha256(images.cpu().numpy().tobytes()).hexdigest() == training['image_tensor_sha256']


@torch.no_grad()
def prefix(model):
    model.cuda()
    z = env['create_preresblock_activation_extractor'](model)(images)
    for block in model.model.visual.transformer.resblocks[:11]:
        z = block(z)
    model.cpu()
    return z


xm = prefix(trained)
own_x, peer_x = prefix(original), prefix(peer)
y = torch.cat((own_x[:, :64], peer_x[:, 64:]), 1)
initial_model = prodistill.initial(base, [original, peer])
readout = initial_model.model.visual
gain, bias, projection = [p.detach().cuda() for p in
                         (readout.ln_post.weight, readout.ln_post.bias, readout.proj)]
heads = [head.weight.detach().cuda(), peer_head.weight.detach().cuda()]
source = torch.tensor([0] * 64 + [1] * 64, device='cuda')
logits_raw, records = {}, []


def values(model, names):
    result = {}
    for name in names:
        value = model
        for part in name.split('.'):
            value = getattr(value, part)
        result[name] = value
    return result


@torch.no_grad()
def classify(hidden):
    z = F.layer_norm(hidden[0], (768,), gain, bias, readout.ln_post.eps) @ projection
    return [F.normalize(z[:64], dim=-1) @ heads[0].T,
            F.normalize(z[64:], dim=-1) @ heads[1].T]


for name, release in [('Replay', cfg['reference_release']), ('MainAB', cfg['release']['path'])]:
    sha = digest(release)
    if name == 'MainAB':
        assert sha == cfg['release']['sha256']
    own = apply(original, dict(path=release, sha256=sha))
    changed = torch.load(release, map_location='cpu', weights_only=True)
    assert all(k.startswith('model.visual.transformer.resblocks.11.mlp.') for k in changed)
    teachers = [copy.deepcopy(m.model.visual.transformer.resblocks[11]).cuda().eval()
                for m in (own, peer)]
    layers = [copy.deepcopy(m) for m in teachers]
    anchor = copy.deepcopy(base.model.visual.transformer.resblocks[11]).cuda()
    torch.manual_seed(cfg['seed'])
    for _ in range(1100):
        torch.randperm(128, device='cuda')
    wrapper = env['MergedModel'](anchor, layers, 'elementwise')
    wrapper.train()
    optimizer = torch.optim.Adam([p for p in wrapper.parameters() if p.requires_grad], lr=cfg['learning_rate'])
    with torch.no_grad():
        teacher_outputs = [m(y) for m in teachers]
        initial_state = {k: v.detach().clone() for k, v in
                         values(wrapper.get_merged_model(), wrapper.names).items()}
        # Own native readout supplies teacher decisions; dataset labels are unused.
        own.cuda(); peer.cuda(); head.cuda(); peer_head.cuda()
        labels = [head(own(images[:64])).argmax(1), peer_head(peer(images[64:])).argmax(1)]
        own.cpu(); peer.cpu(); head.cpu(); peer_head.cpu()
    for epoch in range(101):
        if epoch in cfg['record_epochs']:
            with torch.no_grad():
                model = wrapper.get_merged_model()
                h = model(xm)
                scores = classify(h)
                loss = sum(((h - target).square().sum((0, 2)) * source.eq(j)).sum()
                           for j, target in enumerate(teacher_outputs))
                geometry = {}
                for key in ('mlp.c_fc.weight', 'mlp.c_proj.weight'):
                    value = values(model, [key])[key]
                    delta = value - initial_state[key]
                    geometry[key] = dict(relative_update=float(delta.norm() / initial_state[key].norm()))
                row = dict(group=name, epoch=epoch, loss=float(loss),
                           teacher_matches=[int(s.argmax(1).eq(t).sum()) for s, t in zip(scores, labels)],
                           geometry=geometry)
                records.append(row)
                logits_raw[f'{name}/{epoch}'] = [s.cpu() for s in scores]
                print(row, flush=True)
        if epoch == 100:
            break
        order = torch.randperm(128, device='cuda')
        optimizer.zero_grad()
        student = wrapper.get_merged_model()(xm[:, order]).permute(1, 0, 2).reshape(128, -1)
        loss = 0
        for j, teacher in enumerate(teachers):
            with torch.no_grad():
                target = teacher(y[:, order]).permute(1, 0, 2).reshape(128, -1)
            loss += (F.mse_loss(student, target, reduction='none').sum(1) * source[order].eq(j)).sum()
        assert torch.isfinite(loss)
        loss.backward()
        optimizer.step()
    state = values(wrapper.get_merged_model(), wrapper.names)
    if name == 'Replay':
        reference = trained.model.visual.transformer.resblocks[11].state_dict()
        replay_error = max(float((v.cpu() - reference[k]).abs().max()) for k, v in state.items())
        save_json(out / 'replay.json', dict(maximum_weight_error=replay_error, tolerance=.0005))
        assert replay_error <= .0005, replay_error
    for key, value in state.items():
        i = wrapper.names.index(key)
        expected = anchor.state_dict()[key].cpu().numpy().copy()
        for j in range(2):
            expected += wrapper.alphas[j][i].detach().cpu().numpy() * (
                layers[j].state_dict()[key].cpu().numpy() - anchor.state_dict()[key].cpu().numpy())
        np.testing.assert_allclose(value.detach().cpu().numpy(), expected, atol=2e-5, rtol=2e-5)
    torch.save({k: v.detach().cpu() for k, v in state.items()}, out / (name + '_block.pt'))
    del own, teachers, layers, anchor, wrapper, optimizer, model, state, student
    torch.cuda.empty_cache()
torch.save(logits_raw, out / 'scores.pt')
save_json(out / 'result.json', dict(status='completed', rows=records, replay_error=replay_error,
          seconds=time.monotonic()-start, coefficient_steps=200, defense_updates=0,
          benchmark_images=0, attack_images=128, real_defense_construction_images=0,
          source_sha256=digest(__file__), scores_sha256=digest(out / 'scores.pt')))
