"""Run the four registered standalone tests when the active GPU job finishes."""
import datetime
import os
import subprocess
import sys
import time
from pathlib import Path

from _impl.io import read_json, save_json

protocol = 'results/phase3/bias_weight_only/eight_expert/remaining_native.json'
cfg = read_json(protocol)
budget = cfg['queue_budget']
deadline = datetime.datetime.fromisoformat(budget['deadline_utc']).timestamp()
status_file = Path(cfg['output']).parent / 'native_queue_status.json'


def status(stage, **extra):
    save_json(status_file, dict(stage=stage, **extra))


status('waiting_for_existing_GPU_job')
while Path('/proc/' + str(budget['after_timeout_pid'])).exists():
    if time.time() >= deadline:
        status('deferred_deadline')
        sys.exit(0)
    time.sleep(20)
if deadline - time.time() < budget['start_only_if_seconds_remaining_at_least']:
    status('deferred_insufficient_time')
    sys.exit(0)
stop = min(deadline, time.time() + budget['maximum_gpu_seconds'])
for index, case in enumerate(cfg['cases']):
    status('evaluating', case=index, target=case['target'])
    prefix = [sys.executable, '-u', '-m', 'experiment.main_table']
    folder = Path(cfg['output']) / 'full' / f"{index:02d}_{case['target']}_{case['partner']}"
    commands = [prefix + ['--protocol', protocol, '--case', str(index), '--pilot'],
                prefix + ['--protocol', protocol, '--case', str(index)],
                prefix + ['--verify', str(folder)]]
    for command in commands:
        print('RUN', *command, flush=True)
        subprocess.run(command, check=True, timeout=max(1, stop-time.time()))
status('completed_four_standalone_tests', cases=4)
