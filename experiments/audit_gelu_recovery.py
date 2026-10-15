"""Post-freeze author-AGA permutation/scaling audit; no new attack or selection."""

import argparse
import hashlib
import json
from pathlib import Path

import torch

from clip_benchmark import digest
from subspace_probe import save_json
from task_probe_distance import PREFIX


def row_permutation(reference, changed):
    def key(row): return hashlib.sha256(row.contiguous().numpy().tobytes()).hexdigest()
    index = {key(row): i for i, row in enumerate(reference)}; assert len(index) == len(reference)
    result = torch.tensor([index[key(row)] for row in changed]); assert len(result.unique()) == len(reference)
    return result


def audit(output):
    out = Path(output); cfg = json.loads((out/'protocol.json').read_text()); meta = json.loads((out/'execution.json').read_text())
    assert meta['completed'] and meta['fusion_run']
    root = Path(cfg['publication_point']); original = torch.load(root/'diagnostic.pt', map_location='cpu')['parameters']
    assert digest(root/'candidate.pt') == cfg['candidate_sha256']
    point = torch.load(root/'candidate.pt', map_location='cpu'); names = dict(A=PREFIX+'c_fc.weight', a=PREFIX+'c_fc.bias', B=PREFIX+'c_proj.weight')
    reference = cfg['AGA_references']['MNIST']; assert digest(reference['path']) == reference['sha256']
    anchor = torch.load(reference['path'], map_location='cpu'); delta = torch.load(out/'artifacts/released_AGA_edits.pt', map_location='cpu')
    recovered = {site: delta.get(key, anchor[key]) for site, key in names.items()}
    release = {site: (original[site]+point['release_deltas'][site]).float() for site in names}
    permutation0 = row_permutation(original['A'].float(), anchor[names['A']])
    permutation1 = row_permutation(release['A'], recovered['A'])
    errors = {}
    for site in names:
        expected = release[site][:, permutation1] if site == 'B' else release[site][permutation1]
        errors[site] = float((expected.double()-recovered[site].double()).abs().max())
        assert torch.equal(expected, recovered[site]), site
    result = dict(status='passed', unchanged_author_AGA=True, no_new_recovery_attack=True, new_model_forwards=0,
                  source_sha256=digest(__file__), candidate_sha256=cfg['candidate_sha256'],
                  AGA_edits_sha256=meta['AGA_edits_sha256'], actual_changed_keys=list(delta),
                  original_and_protected_MLP_permutations_equal=torch.equal(permutation0, permutation1),
                  scaling_survives_author_AGA_exactly=True, transformed_A_a_B_maximum_error=errors,
                  minimum_selected_gain=float(point['gain'][point['gain'] > 1].min()), maximum_selected_gain=float(point['gain'].max()),
                  limitation='Only checks unchanged authorAGA on this frozen point. Public gain estimation/canonicalization was not tested; no anti-recovery claim. Retained scale does not imply classification failure.')
    save_json(out/'recovery_scope_audit.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', required=True); args = ap.parse_args()
    torch.set_num_threads(4); audit(args.output)
