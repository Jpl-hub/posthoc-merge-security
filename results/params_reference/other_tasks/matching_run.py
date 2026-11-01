"""Run the existing MLP matcher on fixed releases; reuse clean scores."""
import argparse
import copy
import sys
import json
import subprocess
import time
from pathlib import Path

import torch

from _impl.attacks import public_mlp_matching
from _impl.io import check_files, digest, save_json
from _impl.merge import merge, parameter_hash
from _impl.models import apply, initialize, load_checkpoint

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--stage", choices=["all", "prepare", "evaluate"], default="all")
parser.add_argument("--protocol", default="results/params_reference/other_tasks/matching_protocol.json")
args = parser.parse_args()
stage = args.stage
initialize()
protocol_path = Path(args.protocol)
definition = json.loads(protocol_path.read_text())
defense_name = definition.get('defense_name', 'PaRaMS')
check_files(definition['source_sha256'])
out = Path(definition['output'])
if stage != "evaluate":
    out.mkdir(parents=True, exist_ok=False)
    save_json(out / 'definition.json', definition)
    base = load_checkpoint('zeroshot.pt')
    records = []
    for case in definition['cases']:
        target, partner = case['target'], case['partner']
        expert, peer = load_checkpoint(target + '/finetuned.pt'), load_checkpoint(partner + '/finetuned.pt')
        cfg = copy.deepcopy(definition['evaluation'])
        cfg.update(cases=[dict(target=target, partner=partner, defenses=[defense_name])],
                   output=str(out / target), defenses={}, legacy={})
        for name, published in [('Unprotected', expert), (defense_name, apply(expert, case['release']))]:
            start = time.perf_counter()
            state, detail = public_mlp_matching(published.state_dict(), base.state_dict())
            result = copy.deepcopy(published)
            result.load_state_dict(state, strict=True)
            edits = {k: v for k, v in state.items() if not torch.equal(v, expert.state_dict()[k])}
            path = out / (target + '_' + name + '.pt')
            torch.save(edits, path)
            cfg['defenses'][name] = dict(target=target, release=dict(path=str(path), sha256=digest(path)))
            records.append(dict(target=target, name=name, seconds=time.perf_counter()-start, layers=detail,
                                release=cfg['defenses'][name]['release']))
            if name == 'Unprotected':
                old = case['matched_control']
                assert digest(old['scores']) == old['scores_sha256']
                assert digest(old['execution']) == old['execution_sha256']
                hashes = json.loads(Path(old['execution']).read_text())['evaluated_parameter_sha256']
                assert parameter_hash(state) == hashes['Unprotected/Standalone/None']
                assert parameter_hash(merge(base, result, peer, .5).state_dict()) == hashes['Unprotected/TA/None']
                cfg['legacy'][target + '_' + partner + '/Unprotected'] = {
                    phase: dict(path=old['scores'], sha256=old['scores_sha256'], key_template=key)
                    for phase, key in [('standalone', 'Unprotected/Standalone/None/{task}'),
                                       ('direct', 'Unprotected/TA/None/{task}')]
                }
        save_json(out / (target + '_protocol.json'), cfg)
    save_json(out / 'matching.json', dict(records=records, input_images=0, new_defense_updates=0,
        source_sha256=digest(__file__), definition_sha256=digest(protocol_path)))
    print('ALL_MATCHED_WEIGHTS_FROZEN', len(records), flush=True)
else:
    frozen = json.loads((out / "matching.json").read_text())
    assert frozen["source_sha256"] == digest(__file__)
    assert frozen["definition_sha256"] == digest(protocol_path)
if stage == "prepare":
    sys.exit(0)
# All mappings are fixed before any benchmark images are read.
for case in definition['cases']:
    task, peer = case['target'], case['partner']
    protocol = str(out / (task + '_protocol.json'))
    for pilot in [True, False]:
        command = [sys.executable, '-m', 'experiment.main_table', '--protocol', protocol, '--case', '0']
        subprocess.run(command + (['--pilot'] if pilot else []), check=True)
        directory = out / task / ('pilot' if pilot else 'full') / ('00_' + task + '_' + peer)
        subprocess.run([sys.executable, '-m', 'experiment.main_table', '--verify', str(directory)], check=True)
print('COMPLETE', flush=True)
