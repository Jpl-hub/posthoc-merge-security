"""One fixed input-coverage experiment using existing A/B and evaluation code."""
import argparse
import gc
import subprocess
import sys
from pathlib import Path

from _impl.io import check_files, digest, read_json, save_json

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--protocol', required=True)
cfg = read_json(parser.parse_args().protocol)
check_files(cfg['source_sha256'])
root = Path(cfg['output'])
root.mkdir(parents=True, exist_ok=True)
assert not (root / 'status.json').exists(), 'Existing job: inspect it instead of restarting.'


def status(stage, **extra):
    save_json(root / 'status.json', dict(stage=stage, **extra))


def run(*args):
    print('RUN', *args, flush=True)
    subprocess.run([sys.executable, '-u', *map(str, args)], check=True, timeout=5400)


import torch
fit = cfg['fit']
if 'recovered_model' not in fit:
    from _impl.models import apply, checked_models, initialize
    from _impl.prodistill import author, train
    status('synthetic_recovery_prefix')
    initialize()
    base, original, head, _ = checked_models('MNIST')
    own = apply(original, fit['parent'])
    prefix = root / 'recovery_prefix'
    prefix.mkdir(exist_ok=False)
    recovery = cfg['synthetic_recovery']
    train(recovery, dict(target='MNIST', partner='public_base',
                        synthetic_source=fit['synthetic_source'],
                        synthetic_partner_source=fit.get('synthetic_partner_source')), prefix, base,
          {'MNIST': own, 'public_base': base}, own, base, author(recovery))
    fit['recovered_model'] = dict(path=str(prefix / 'model.pt'), sha256=digest(prefix / 'model.pt'))
    del base, original, head, own
save_json(root / 'construction.json', fit)
gc.collect()
torch.cuda.empty_cache()

status('A_B_construction', recovered_model=fit['recovered_model'])
run('outputs/task_inputs_20261106/construct.py', '--protocol', root / 'construction.json')
release = read_json(Path(fit['output']) / 'completed.json')['releases']['JointRecovery']
status('frozen_before_real_images', publication=release)
# The first 800 cached views are the unchanged original source. This count is
# replayed by the existing full-author AGA evaluator before real-image scoring.
scores = torch.load(Path(fit['output']) / 'JointRecovery/source_scores.pt',
                    map_location='cpu', weights_only=False)
matches = int(scores['scores']['recovered'][:800].argmax(1).eq(
              scores['reference']['native'][:800].argmax(1)).sum())
adapter = dict(eligible=True, publication=release, counts={'recovered/TA': matches},
               eligibility_scope='Evaluate this finite final point regardless of source accuracy; no selection.')
save_json(root / 'evaluation_point.json', adapter)
save_json(root / 'evaluation.json', dict(output=str(root / 'comparison')))
status('complete_fixed_evaluation', publication=release)
run(cfg['evaluation_driver']['path'], '--protocol', root / 'evaluation.json',
    '--source-result', root / 'evaluation_point.json')
status('completed_no_automatic_expansion', publication=release)
