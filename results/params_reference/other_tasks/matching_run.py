"""Run the existing MLP matcher on fixed PaRaMS releases; reuse clean scores."""
import copy
import json
import subprocess
import time
from pathlib import Path

import torch

from _impl.attacks import public_mlp_matching
from _impl.io import check_files, digest, save_json
from _impl.merge import merge, parameter_hash
from _impl.models import apply, initialize, load_checkpoint

initialize()
root = Path('results/params_reference/other_tasks')
definition = json.loads((root / 'matching_protocol.json').read_text())
check_files(definition['source_sha256'])
out = Path(definition['output'])
out.mkdir(parents=True, exist_ok=False)
save_json(out / 'definition.json', definition)
base = load_checkpoint('zeroshot.pt')
records = []
for case in definition['cases']:
    target, partner = case['target'], case['partner']
    expert, peer = load_checkpoint(target + '/finetuned.pt'), load_checkpoint(partner + '/finetuned.pt')
    cfg = copy.deepcopy(definition['evaluation'])
    cfg.update(cases=[dict(target=target, partner=partner, defenses=['PaRaMS'])],
               output=str(out / target), defenses={}, legacy={})
    for name, published in [('Unprotected', expert), ('PaRaMS', apply(expert, case['release']))]:
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
                stage: dict(path=old['scores'], sha256=old['scores_sha256'], key_template=key)
                for stage, key in [('standalone', 'Unprotected/Standalone/None/{task}'),
                                   ('direct', 'Unprotected/TA/None/{task}')]
            }
    save_json(out / (target + '_protocol.json'), cfg)
save_json(out / 'matching.json', dict(records=records, input_images=0, new_defense_updates=0,
    source_sha256=digest(__file__), definition_sha256=digest(root / 'matching_protocol.json')))
print('ALL_MATCHED_WEIGHTS_FROZEN', len(records), flush=True)
# All mappings are fixed before any benchmark images are read.
for case in definition['cases']:
    task, peer = case['target'], case['partner']
    protocol = str(out / (task + '_protocol.json'))
    for pilot in [True, False]:
        command = [__import__('sys').executable, '-m', 'experiment.main_table', '--protocol', protocol, '--case', '0']
        subprocess.run(command + (['--pilot'] if pilot else []), check=True)
        directory = out / task / ('pilot' if pilot else 'full') / ('00_' + task + '_' + peer)
        subprocess.run([__import__('sys').executable, '-m', 'experiment.main_table', '--verify', str(directory)], check=True)
print('COMPLETE', flush=True)
