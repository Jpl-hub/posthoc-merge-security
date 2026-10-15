"""Exact and author-solver checks of one preregistered synthetic alignment example."""

import argparse
from fractions import Fraction as Q
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from mechanisms import load_author


def run(output):
    start = time.monotonic()
    out = Path(output); out.mkdir(parents=True, exist_ok=False)
    author, pin = load_author(Path('external/secure-merge-attack/utils_attack_B32.py'))
    assert pin == '4600479e44ad8c99fd1bff616664567c047c1940fddcd02b9213ed139fe7a778'
    lam = Q(1, 100000000); a = 1/(1+lam); b = Q(1,100)/(Q(1,100)+lam)
    exact = dict(native=[Q(1), Q(-1)], canonical_original=[-a+2*b,a-2*b],
                 canonical_release=[-a,a], direct_merge=[Q(1),Q(-1)],
                 canonical_merge_original=[(1-a+2*b)/2,(-1+a-2*b)/2],
                 canonical_merge_release=[(1-a)/2,(-1+a)/2])
    for name, gaps in exact.items():
        wanted = [-1,1] if name == 'canonical_release' else [1,-1]
        assert [1 if g>0 else -1 for g in gaps] == wanted
    x = np.diag([1.,1.,.1]); release = np.diag([1.,1.,0.])
    base = np.array([[-1.,0.,0.],[0.,-1.,0.],[1.,-1.,.1]])
    z = np.eye(3)[:,:2]; g = np.array([1.,-1.,2.])
    _, sv, vh = np.linalg.svd(x-base, full_matrices=False)
    assert np.linalg.norm((release-x)@vh[:2].T) < 1e-14
    assert sv[2] == 0 and np.linalg.norm(release@z-x@z) == 0
    distance = np.linalg.norm(release-x)/np.linalg.norm(x)
    assert distance < .5
    def aligned(value):
        src = torch.tensor(value, dtype=torch.float32)
        ref = torch.tensor(base, dtype=torch.float32)
        return (src@author['solve_linear_transformation'](src,ref)).numpy()
    old, new = aligned(x), aligned(release)
    formula_old = np.diag([float(a),float(a),float(b)])@base
    formula_new = np.diag([float(a),float(a),0.])@base
    error = max(np.abs(old-formula_old).max(), np.abs(new-formula_new).max())
    assert error < 1e-6
    rows = {}
    for name, value in dict(native=x, canonical_original=old, canonical_release=new,
                            direct_merge=(release+base)/2,
                            canonical_merge_original=(old+base)/2,
                            canonical_merge_release=(new+base)/2).items():
        # Include final feature normalization and preserve exact FP32 ties.
        f = value.astype(np.float32)@z.astype(np.float32)
        f = f/np.linalg.norm(f,axis=0,keepdims=True)
        gaps = g.astype(np.float32)@f
        pred = np.argmax(np.stack([gaps/2,-gaps/2]),axis=0)
        rows[name] = dict(normalized_gaps=gaps.tolist(),predictions=pred.tolist(),
                          correct=int((pred==np.array([0,1])).sum()),n=2,ties=int((gaps==0).sum()))
    result = dict(status='synthetic_rank_condition_verified_no_CLIP_candidate',
        preregistered_commit='2e6d01e',parameter_distance=distance,source_task_drift=0,
        source_task_SVD_probe_error=0,exact_gap_numerators={k:list(map(str,v)) for k,v in exact.items()},
        author_FP32_rows=rows,author_vs_float64_formula_max_abs=float(error),
        ridge_lambda=float(lam),author_source_sha256=pin,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        official_column_solver_calls=2,CLIP_checkpoints_loaded=0,images_loaded=0,gradients=0,
        new_defense_weights=0,wall_seconds=time.monotonic()-start,
        scope='Synthetic column solver only, not full Attention/AGA or CLIP. Exact positive fused margins near1e-8 can tie after author FP32 conversion; ties are reported without changing coefficients or tolerance. No unknown-peer or defense-success claim.')
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output',required=True)
    run(ap.parse_args().output)
