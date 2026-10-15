"""Recount existing author-AGA matrix retention; no attack or model forward."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import digest
from subspace_probe import save_json


def audit(output):
    start = time.monotonic(); root = Path(output); cfg = json.loads((root/'protocol.json').read_text())
    meta = json.loads((root/'execution.json').read_text()); assert json.loads((root/'independent_validation.json').read_text())['status'] == 'passed'
    source = Path(cfg['publication_point']); construction = json.loads((source/'results.json').read_text())
    assert digest(source/'diagnostic.pt') == construction['diagnostic_sha256']
    original_a = torch.load(source/'diagnostic.pt', map_location='cpu')['parameters']['A'].float().numpy()
    reference = cfg['AGA_references']['MNIST']; assert digest(reference['path']) == reference['sha256']
    original = torch.load(reference['path'], map_location='cpu'); released = torch.load(root/'artifacts/release.pt', map_location='cpu')
    attacked = torch.load(root/'artifacts/released_AGA_edits.pt', map_location='cpu')
    assert digest(root/'artifacts/release.pt') == meta['release_sha256']
    assert digest(root/'artifacts/released_AGA_edits.pt') == meta['AGA_edits_sha256']
    ka = 'model.visual.transformer.resblocks.11.mlp.c_fc.weight'; kb = ka.replace('c_fc', 'c_proj')
    key = lambda a: hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
    lookup = {key(row): i for i, row in enumerate(original_a)}
    perm = np.array([lookup[key(row)] for row in original[ka].numpy()]); assert len(np.unique(perm)) == len(perm)
    assert set(attacked) == {ka, kb}
    np.testing.assert_array_equal(attacked[ka].numpy(), released[ka].numpy()[perm])
    np.testing.assert_array_equal(attacked[kb].numpy(), released[kb].numpy()[:, perm])
    result = dict(status='passed', author_AGA_pair_retained_exactly=True, AGA_MLP_permutation_matches_original=True,
                  A_array_equal=True, B_array_equal=True, new_attack_calls=0, new_model_forwards=0,
                  release_sha256=meta['release_sha256'], AGA_edits_sha256=meta['AGA_edits_sha256'],
                  source_sha256=digest(__file__), wall_seconds=time.monotonic()-start,
                  interpretation='The fixed pair survives unchanged author AGA up to its ORIGINAL neuron permutation. Direct TA was already ineffective; source compensation plus latent source B damage is not sufficient for real fusion failure. No new candidate, attack, or parameter selection.')
    save_json(root/'AGA_retention_validation.json', result); print(json.dumps(result), flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__); ap.add_argument('--output', required=True); args = ap.parse_args()
    torch.set_num_threads(4); audit(args.output)
