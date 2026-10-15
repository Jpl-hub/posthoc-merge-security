"""Frozen final-O spectrum and the author's ridge-recovery bound; no release transform."""

import argparse
import json
from pathlib import Path
import time

import torch

from clip_benchmark import checkpoint_loader, digest
from mechanisms import load_author


@torch.no_grad()
def run(output):
    start = time.monotonic(); out = Path(output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(4)
    assert '3090' in torch.cuda.get_device_name()
    torch.backends.cuda.matmul.allow_tf32 = False
    pins = {r['path']:r['sha256'] for r in json.loads(Path('configs/checkpoints.json').read_text())['files']}
    load = checkpoint_loader(Path('external/secure-merge-attack'))
    matrices, hashes = [], {}
    key = 'model.visual.transformer.resblocks.11.attn.out_proj.weight'
    for name in ['ViT-B-32/MNIST/finetuned.pt','ViT-B-32/zeroshot.pt']:
        path = Path('checkpoints')/name
        hashes[name] = digest(path); assert hashes[name] == pins[name]
        model = load(str(path)).cpu().eval(); model.requires_grad_(False)
        matrices.append(model.state_dict()[key].detach().clone())
        del model
    author, pin = load_author(Path('external/secure-merge-attack/utils_attack_B32.py'))
    assert pin == '4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778'
    lam = 1e-8; rows, raw = [], []
    for h in range(12):
        x, base = [w[:,h*64:(h+1)*64].cuda().double() for w in matrices]
        u, sigma, _ = torch.linalg.svd(x, full_matrices=False)
        filt = sigma.square()/(sigma.square()+lam)
        canonical = (u*filt)@u.T@base
        implemented = (x.float()@author['solve_linear_transformation'](x.float(),base.float())).double()
        task = float((x-base).norm()); error = float((canonical-x).norm())
        ridge = float(lam/(sigma[-1].square()+lam)*base.norm())
        assert error <= task+ridge+1e-10
        difference = float((implemented-canonical).abs().max())
        assert difference < 1e-5
        row = dict(head=h,shape=list(x.shape),sigma_min=float(sigma[-1]),sigma_max=float(sigma[0]),
            condition_number=float(sigma[0]/sigma[-1]),ridge_filter_defect_max=float(1-filt[-1]),
            task_vector_relative_norm=task/float(x.norm()),
            canonical_recovery_relative_error=error/float(x.norm()),
            theoretical_relative_bound=(task+ridge)/float(x.norm()),
            author_FP32_vs_formula_max_abs=difference,
            minimum_relative_Frobenius_distance_to_rank_deficiency=float(sigma[-1]/x.norm()))
        rows.append(row); raw.append(dict(X=x.cpu(),anchor=base.cpu(),canonical=canonical.cpu(),author=implemented.cpu()))
    torch.save(raw,out/'matrices.pt')
    result = dict(status='frozen_original_final_O_ridge_bound_verified',preregistered_commit='a3b2b69',
        matrix_key=key,heads=rows,ridge_lambda=lam,checkpoints_sha256=hashes,author_source_sha256=pin,
        source_sha256=digest(__file__),raw_sha256=digest(out/'matrices.pt'),
        gradients=0,images_loaded=0,peer_checkpoints_loaded=0,new_release_weights=0,
        head_or_rank_selected=False,official_column_solver_calls=12,
        wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        scope='Original final block O only. No task-function or class-margin preservation follows from these matrix bounds. No rank-deleted matrix was released or evaluated.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',required=True)
    run(ap.parse_args().output)
