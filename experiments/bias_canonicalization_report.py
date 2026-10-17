"""A compact recovery table from completed and independently verified raw results."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


def run(root, output):
    meta=json.loads((root/'execution.json').read_text());valid=json.loads((root/'validation.json').read_text())
    assert meta['completed'] and valid['status']=='passed' and meta['predictions_sha256']==valid['predictions_sha256']
    rows=json.loads((root/'results.json').read_text());old=json.loads(Path('results/phase3/results.json').read_text())
    assert len(rows)==20
    recovered=json.loads((root/'recovery_pairs.json').read_text())
    assert recovered['status']=='NumPy_paired_recovery_recount_passed' and len(recovered['records'])==8
    assert all(r['changed_predictions']==0 for r in rows if r['defense']=='Phase1_MNIST')
    native=json.loads((root/'native_function_audit.json').read_text())
    assert native['old_raw_hashes_checked_against_frozen_execution'] and native['after_raw_sha256']==meta['predictions_sha256']
    def one(items, **keys):
        found=[r for r in items if all(r.get(k)==v for k,v in keys.items())];assert len(found)==1,keys
        return found[0]['accuracy_percent']
    def pair(items,target,partner,defense,scope):
        return ' / '.join(f'{one(items,target=target,partner=partner,defense=defense,merger="TA",attack_scope=scope,dataset=t):.2f}' for t in [target,partner])
    lines=['# Targeted LN-bias recovery', '', 'Each fused cell reports protected-task / partner-task accuracy (%), TA(0.5).', '',
        '| Protected task | Partner | Recovery scope | Defense 1: no recovery | Defense 1: bias recovery | Unprotected: same recovery |',
        '|---|---|---|---:|---:|---:|']
    for partner in ['SVHN','DTD']:
        for scope in ['Protected expert','All experts']:
            lines.append('| '+' | '.join(['MNIST',partner,scope,pair(old,'MNIST',partner,'Phase1_MNIST','None'),pair(rows,'MNIST',partner,'Phase1_MNIST',scope),pair(rows,'MNIST',partner,'Unprotected',scope)])+' |')
    lines+=['', 'The recovery uses only the published visual projection and final LN bias. It projects the bias onto the projection column space; no clean expert, public base, images or partner enters recovery construction. The base and partner are used only for subsequent TA evaluation.', '',
        'This is an architecture-aware supplementary adaptive diagnostic, not original AGA or a claimed author attack reproduction. Defense weights are unchanged. Both recovery scopes and their matching unprotected controls are retained; tests were previously observed.', '',
        'Scope: the two MNIST cases passed the unchanged 0.0005 function-control threshold on RTX3090. The four-case pilot stopped at DTD (0.00082493); SVHN as protected task was not reached. That failure remains recorded. Earlier RTX4090 pilot verification is separate. No threshold or recovery coefficient was changed.', '',
        'Native full-test comparison is separate from replay verification: all 10000 MNIST predictions remain unchanged, but the maximum protected-model score difference before/after cleanup is 0.00069427, above the pilot/replay tolerance0.0005. Worst relative margin drift is0.00112328. The ideal real-arithmetic identity does not imply exact FP32 score agreement. See native_function_audit.json; the original tolerance is unchanged.', '',
        f'Full evaluation: {meta["wall_seconds"]:.2f}s. Independent full-model selected-score replays: {valid["selected_score_replays"]}, maximum error {valid["maximum_score_error"]:.6g}. Replay cost is additional; concurrent workloads mean these wall times are not an isolated speed benchmark.', '',
        'Every recovered Defense 1 prediction equals its matching recovered unprotected control, including standalone MNIST. Against the frozen no-recovery predictions, MNIST gains 8962 correct predictions and loses 5 with SVHN; it gains 8966 and loses 1 with DTD. Both recovery scopes give the same counts. Partner net gains are 22335/26032 (SVHN) and 1262/1880 (DTD).', '',
        'Per-class counts and comparisons to matching recovered unprotected controls are in results.json. Recovery gains against the frozen no-recovery models, exact paired p values and both raw-score fingerprints are in recovery_pairs.json. Raw scores and six frozen recovery edits stay on the server. No all-attack or cross-protected-task robustness claim.']
    output.mkdir(parents=True,exist_ok=True);(output/'tables.md').write_text('\n'.join(lines)+'\n')
    fields=['target','partner','defense','merger','attack_scope','dataset','role','n','correct','accuracy_percent','changed_predictions','mcnemar_log10_p']
    with (output/'results.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)
    sources={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [root/'results.json',root/'execution.json',root/'validation.json',root/'recovery_pairs.json',root/'native_function_audit.json',Path('results/phase3/results.json'),Path(__file__)]}
    (output/'table_sources.json').write_text(json.dumps(sources,indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--results',type=Path,default=Path('results/phase3/bias_canonicalization'))
    ap.add_argument('--output',type=Path,default=Path('results/phase3/bias_canonicalization'));args=ap.parse_args();run(args.results,args.output)
