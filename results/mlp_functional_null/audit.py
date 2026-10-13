"""One native-calibration Jacobian check; never edit or evaluate model weights."""
from __future__ import annotations

import inspect
import json
import sys
import time
from pathlib import Path

import torch
from torchvision.datasets import MNIST

ROOT = Path('/root/autodl-tmp/merge_security_20261009')
OUT = ROOT / 'outputs/mlp_functional_null'
sys.path.insert(0, str(ROOT / 'experiments'))
from clip_benchmark import checkpoint_loader, digest
from subspace_probe import make_loader, split_training


def derivative(z):
    s = torch.sigmoid(1.702 * z)
    return s + 1.702 * z * s * (1 - s)


@torch.no_grad()
def main():
    started = time.time()
    p = json.loads((OUT / 'protocol.json').read_text())
    assert not (OUT / 'results.json').exists(), 'Never overwrite results'
    assert digest(__file__) == p['runner_sha256']
    for name, expected in p['source_sha256'].items():
        assert digest(ROOT / name) == expected, name
    checkpoint = ROOT / 'checkpoints/ViT-B-32/MNIST/finetuned.pt'
    assert digest(checkpoint) == p['checkpoint_sha256']
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert torch.cuda.get_device_name() == 'NVIDIA GeForce RTX 3090'
    model = checkpoint_loader(ROOT / 'external/secure-merge-attack')(checkpoint, map_location='cpu').eval()
    block = model.model.visual.transformer.resblocks[0]
    assert type(block.mlp.gelu).__name__ == 'QuickGELU'
    assert digest(inspect.getfile(type(block))) == p['open_clip_model_sha256']
    dataset = MNIST(ROOT / 'data', train=True, transform=model.val_preprocess, download=False)
    ids = split_training(dataset, p['calibration_sizes'], p['seed'])
    for file, expected in p['dataset_sha256'].items():
        assert digest(ROOT / 'data/MNIST/raw' / file) == expected, file
    selected = {}
    summaries = {}
    current = {}

    def capture(_module, inputs, output):
        x = inputs[0].detach().double()  # sequence, batch, channels
        z = output.detach().double()
        assert tuple(z.shape[::2]) == (50, 3072)
        flat = z.reshape(-1, 3072)
        current['minimum'] = torch.minimum(current['minimum'], flat.min(0).values)
        current['maximum'] = torch.maximum(current['maximum'], flat.max(0).values)
        current['derivative_squares'].add_(derivative(flat).square().sum(0))
        current['input_sum'].add_(x.reshape(-1, 768).sum(0))
        current['tokens'] += len(flat)
        if current['images'] == 0:
            current['x'] = torch.stack([x[t, i] for i in range(3) for t in p['tokens']]).cpu()
            current['z'] = torch.stack([z[t, i] for i in range(3) for t in p['tokens']]).cpu()
        current['images'] += z.shape[1]

    handle = block.mlp.c_fc.register_forward_hook(capture)
    model.cuda()
    try:
        for name, indices in ids.items():
            current = dict(minimum=torch.full((3072,), float('inf'), device='cuda', dtype=torch.float64),
                maximum=torch.full((3072,), -float('inf'), device='cuda', dtype=torch.float64),
                derivative_squares=torch.zeros(3072, device='cuda', dtype=torch.float64),
                input_sum=torch.zeros(768, device='cuda', dtype=torch.float64), tokens=0, images=0)
            for images, _ in make_loader(dataset, indices, p['batch_size']):
                model(images.cuda(non_blocking=True))
            selected[name] = {k: current[k] for k in ('x', 'z')}
            summaries[name] = {k: v.cpu() if isinstance(v, torch.Tensor) else v
                for k, v in current.items() if k not in ('x', 'z')}
            print('CALIBRATION', name, current['images'], current['tokens'], flush=True)
    finally:
        handle.remove()
        model.cpu()
    w = block.mlp.c_proj.weight.detach().double()
    mean = summaries['defender']['input_sum'] / summaries['defender']['tokens']
    a = mean / mean.norm()
    native = selected['defender']
    t = native['x'] @ a
    d = derivative(native['z'])
    jacobian = torch.cat([w * (t[i] * d[i])[None, :] for i in range(len(t))]).cuda()
    # SVD supplies a necessity diagnostic, not a rank chosen for protection.
    _, s, vh = torch.linalg.svd(jacobian, full_matrices=False)
    v = vh[-1].cpu()
    s = s.cpu()
    del jacobian, vh
    combined_min = torch.minimum(summaries['defender']['minimum'], summaries['confirmation']['minimum'])
    combined_max = torch.maximum(summaries['defender']['maximum'], summaries['confirmation']['maximum'])
    active = combined_min > 0
    active_columns = w[:, active]
    active_s = torch.linalg.svdvals(active_columns.cuda()).cpu() if active.any() else torch.empty(0)
    metrics = {}
    column_power = w.square().sum(0)
    for name, values in selected.items():
        factor = values['x'] @ a
        deriv = derivative(values['z'])
        vectors = deriv * v[None, :]
        response = (vectors @ w.T) * factor[:, None]
        incoherent = (vectors.square() @ column_power).sqrt() * factor.abs()
        metrics[name] = dict(input_factor_min_abs=float(factor.abs().min()),
            response_rms=float(response.square().sum(1).mean().sqrt()),
            incoherent_rms=float(incoherent.square().mean().sqrt()),
            cancellation_ratio=float(response.norm() / incoherent.norm()),
            unweighted_derivative_rms=float(vectors.square().sum(1).mean().sqrt()))
    tol = torch.finfo(torch.float64).eps * 4608 * s[0]
    active_tol = torch.finfo(torch.float64).eps * max(active_columns.shape) * active_s[0] if len(active_s) else 0
    raw = dict(selected=selected, summaries=summaries, ids=ids, a=a, w=w, singular_values=s,
        weakest_direction=v, stable_active_singular_values=active_s)
    torch.save(raw, OUT / 'native_capsule.pt')
    result = dict(protocol_sha256=digest(OUT / 'protocol.json'), runner_sha256=digest(__file__),
        checkpoint_sha256=digest(checkpoint), capsule_sha256=digest(OUT / 'native_capsule.pt'),
        source_parent=p['source_parent'], site=p['site'], calibration_images={k: len(i) for k, i in ids.items()},
        forward_examples=sum(len(i) for i in ids.values()), classification_rows=0,
        new_protected_weights=0, partner_accessed=False, optimizer_used=False, AGA_run=False,
        spectrum=dict(rows=4608, columns=3072, largest=float(s[0]), smallest=float(s[-1]),
            smallest_over_largest=float(s[-1] / s[0]), numerical_rank=int((s > tol).sum()),
            machine_rank_tolerance=float(tol), relative_1e8_rank=int((s > s[0]*1e-8).sum())),
        neuron_regime=dict(stable_positive=int(active.sum()), stable_negative=int((combined_max < 0).sum()),
            sign_changing=int(((combined_min <= 0) & (combined_max >= 0)).sum()),
            stable_positive_column_rank=int((active_s > active_tol).sum()),
            weakest_direction_stable_positive_energy=float(v[active].square().sum()),
            weakest_direction_stable_negative_energy=float(v[combined_max < 0].square().sum())),
        local_response=metrics, seconds=time.time()-started,
        interpretation='Local derivative/rank evidence only; not finite-edit preservation, a classifier result, or an anti-recovery defense.')
    (OUT / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
