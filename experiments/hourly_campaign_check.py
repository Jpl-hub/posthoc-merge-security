"""Deadline-bounded server checks; research decisions remain in the scheduled chat."""

import argparse
import datetime as dt
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time


def write_json(path, value):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    tmp.replace(path)


def run(campaign):
    path = Path(campaign); cfg = json.loads(path.read_text())
    output = Path(cfg['output']); output.mkdir(parents=True, exist_ok=True)
    lock = (output/'monitor.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    start = dt.datetime.fromisoformat(cfg['start_utc']).timestamp()
    end = dt.datetime.fromisoformat(cfg['end_utc']).timestamp()
    write_json(output/'monitor_process.json', dict(pid=os.getpid(), start_utc=cfg['start_utc'], end_utc=cfg['end_utc']))
    previous = start; next_check = time.time()
    while True:
        time.sleep(max(0, next_check-time.time()))
        now = time.time(); expired = now >= end
        state = json.loads(Path('research_state.json').read_text())
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used', '--format=csv,noheader'], capture_output=True, text=True, timeout=20)
        # Limit process output to project Python jobs; do not collect shells or credentials.
        processes = subprocess.run(['ps', '-eo', 'pid,etimes,comm,args'], capture_output=True, text=True, timeout=20).stdout.splitlines()
        jobs = [line for line in processes if 'python' in line and ('experiments/' in line or 'hourly_campaign_check.py' in line)]
        recent = []
        for p in Path('outputs').glob('*/results.json'):
            if previous <= p.stat().st_mtime <= now:
                data = json.loads(p.read_text())
                recent.append(dict(path=str(p), status=data.get('status'), wall_seconds=data.get('wall_seconds')))
        logs = {}
        for pattern in cfg['log_patterns']:
            for p in Path('/tmp').glob(pattern):
                lines = p.read_text(errors='replace').splitlines()
                logs[str(p)] = lines[-5:]
        row = dict(time_utc=dt.datetime.now(dt.timezone.utc).isoformat(), pid=os.getpid(),
                   expired=expired, GPU=gpu.stdout.strip(), gpu_error=gpu.stderr.strip(), jobs=jobs,
                   recent_results=recent, log_tails=logs, research_status=state.get('status'),
                   next_action=state.get('next_action'), decision='deadline_closeout' if expired else 'inspect_in_scheduled_chat',
                   research_progress_claimed=False)
        with (output/'hourly_checks.jsonl').open('a') as f:
            f.write(json.dumps(row, ensure_ascii=False)+'\n')
        write_json(output/'latest_check.json', row)
        state['hourly_monitor'] = dict(last_check_utc=row['time_utc'], pid=os.getpid(), expired=expired,
                                       receipt=str(output/'latest_check.json'), is_research_executor=False)
        write_json(Path('research_state.json'), state)
        cfg = json.loads(path.read_text()); cfg['last_server_check_utc'] = row['time_utc']
        cfg['server_monitor_status'] = 'finished' if expired else 'running'; write_json(path, cfg)
        print(row['time_utc'], row['decision'], flush=True)
        if expired:
            break
        previous = now
        next_check = min(end, start + (int((now-start)//3600)+1)*3600)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('--campaign', required=True)
    run(p.parse_args().campaign)
