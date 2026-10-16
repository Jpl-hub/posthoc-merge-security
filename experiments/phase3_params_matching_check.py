"""CPU permutation-orientation check of pinned PaRaMS; no CLIP attack evaluation."""

import argparse
import hashlib
import importlib.util
import itertools
import json
import os
import time
from pathlib import Path

os.environ.setdefault('JAX_PLATFORMS', 'cpu')
if hasattr(os, 'sched_getaffinity'):
    os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:4])

import jax
import numpy as np
import scipy
import torch


def check(source, output):
    started = time.monotonic()
    raw = source.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    assert sha == '40c114a121907d5633d29a49984efa9be2ac03e29e649e801fed2dccad89fb60'
    spec = importlib.util.spec_from_file_location('official_params_matching', source)
    author = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(author)
    torch.set_num_threads(1)
    prefix = 'model.visual.transformer.resblocks.0.mlp.'
    base = {prefix+'c_fc.weight': 8*torch.eye(3),
            prefix+'c_proj.weight': torch.tensor([[1., 0., 2.], [0., 1., 1.]]),
            prefix+'c_fc.bias': torch.tensor([.25, .5, .75]),
            prefix+'c_proj.bias': torch.tensor([.125, -.25])}
    ps = author.vit_permutation_spec_MLP(num_layers=1)
    cycle = torch.tensor([1, 2, 0])  # An involution would hide an inverse-orientation error.
    protected = {k: torch.tensor(np.array(v)) for k, v in
                 author.apply_permutation(ps, {'P_0': cycle}, base).items()}

    def matching(reference, moving):
        perm, _, _, _ = author.parameter_rearrangement(
            jax.random.PRNGKey(0), ps, reference, moving, obj='matching')
        restored = author.apply_permutation(ps, perm, moving)
        return perm['P_0'].tolist(), {k: torch.tensor(np.array(v)) for k, v in restored.items()}

    def distance(left, right):
        return sum(float((left[k].double()-right[k].double()).square().sum()) for k in left)**.5

    forward, aligned_base = matching(protected, base)
    inverse, restored = matching(base, protected)
    kwargs = dict(num_layers=1, enable_permutation=True, enable_scaling=False,
                  perm_config={'max_iter': 200, 'obj': 'matching', 'tolerance': 1}, rng_seed=0)
    wrapper = author.params(protected, base, **kwargs)
    identity = author.params(base, base, **kwargs)
    assert forward == [1, 2, 0] and inverse == [2, 0, 1]
    assert distance(aligned_base, protected) == distance(restored, base) == distance(identity, base) == 0
    wrapper_distance = distance(wrapper, base)
    assert wrapper_distance > 0

    # Independently enumerate all six assignments using the source's three terms.
    left = [protected[prefix+'c_fc.weight'].numpy(), protected[prefix+'c_proj.weight'].numpy().T,
            protected[prefix+'c_fc.bias'].numpy()[:, None]]
    right = [base[prefix+'c_fc.weight'].numpy(), base[prefix+'c_proj.weight'].numpy().T,
             base[prefix+'c_fc.bias'].numpy()[:, None]]
    cost = sum(a.astype('float64')@b.astype('float64').T for a, b in zip(left, right))
    assignments = [(float(cost[np.arange(3), p].sum()), list(p))
                   for p in itertools.permutations(range(3))]
    assignments.sort(reverse=True)
    assert assignments[0][1] == forward and assignments[0][0] > assignments[1][0]
    inputs = torch.tensor([[0., 0., 0.], [1., -1., .5], [-.25, .75, 1.], [-1., -.5, -.25]])

    def response(state):
        z = inputs@state[prefix+'c_fc.weight'].T+state[prefix+'c_fc.bias']
        return (z*torch.sigmoid(1.702*z))@state[prefix+'c_proj.weight'].T+state[prefix+'c_proj.bias']

    errors = {name: float((response(state)-response(base)).abs().max())
              for name, state in [('protected', protected), ('kernel_restored', restored), ('wrapper', wrapper)]}
    assert max(errors.values()) < 5e-6
    result = dict(status='official_MLP_matching_kernel_checked_wrapper_direction_counterexample',
                  repository='ISCT-W/PaRaMS', commit='4159a9a8bf253088694154d944f4583db8eca60e',
                  official_source_sha256=sha, check_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  runtime=dict(torch=torch.__version__, jax=jax.__version__, numpy=np.__version__, scipy=scipy.__version__),
                  fixture='One synthetic MLP, hidden width3, fixed three-cycle; no expert checkpoint or dataset.',
                  protection_permutation=cycle.tolist(), kernel_protected_reference_base_moving=forward,
                  kernel_base_reference_protected_moving=inverse,
                  kernel_correct_operand_reconstruction_error=distance(restored, base),
                  wrapper_matching_reconstruction_error=wrapper_distance,
                  identity_control_reconstruction_error=distance(identity, base),
                  NumPy_exhaustive_assignment_agrees=True, assignment_optimum_margin=assignments[0][0]-assignments[1][0],
                  finite_QuickGELU_response_max_abs=errors,
                  conclusion='Kernel permutes params_b. params wrapper estimates that permutation from (victim, base), then applies it to victim; matching mode need not restore coordinates. Kernel (base, protected) plus apply to protected is verified on this fixture.',
                  scope='MLP source check only. No Attention recovery, real CLIP accuracy, unknown-peer guarantee or new attack result.',
                  images_used=0, gradients=0, GPU_used=False, protected_checkpoint_modified=False,
                  wall_seconds=time.monotonic()-started)
    assert not output.exists(), 'Preserve the previous check result; choose a new output for any independent rerun.'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('external/PaRaMS/src/PaRaMS/PaRaMS_algorithm.py'))
    parser.add_argument('--output', type=Path, default=Path('results/phase3/params_matching_check.json'))
    args = parser.parse_args()
    check(args.source, args.output)
