"""One comparison table from verified frozen recovery results; no manual scores."""

import csv
import argparse
import hashlib
import json
from pathlib import Path


def verify_margins(raw_path):
    import numpy as np
    import torch
    out = Path('results/phase3/params_adaptive')
    meta = json.loads((out/'execution.json').read_text())
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest() == meta['predictions_sha256']
    raw = torch.load(raw_path, map_location='cpu', weights_only=False)
    rows = json.loads((out/'results.json').read_text())
    for row in rows:
        key = '/'.join(('Unprotected', row['merger'], row['attack_scope'], row['dataset']))
        before = raw[key]['logits'].double().numpy(); after = raw[row['key']]['logits'].double().numpy()
        y = before.argmax(1); ids = np.arange(len(y)); gaps = before[ids, y, None]-before
        ds = after-before
        ratios = np.divide(ds[ids, y, None]-ds, gaps, out=np.zeros_like(gaps), where=gaps>0)
        np.testing.assert_allclose(np.abs(ratios).max(), row['worst_relative_margin_drift_to_matched_control'], rtol=1e-8, atol=1e-8)
        np.testing.assert_allclose((-ratios).max(), row['maximum_signed_margin_harm_to_matched_control'], rtol=1e-8, atol=1e-8)
    (out/'margin_validation.json').write_text(json.dumps(dict(status='passed', rows=len(rows),
        independent_NumPy_full_signed_margin_recount=True, predictions_sha256=meta['predictions_sha256'],
        verifier_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()), indent=2)+'\n')


def generate():
    out = Path('results/phase3/params_adaptive')
    validation = json.loads((out/'independent_validation.json').read_text())
    execution = json.loads((out/'execution.json').read_text())
    assert validation['status'] == 'passed' and execution['completed']
    prediction_hash = validation.get('raw_sha256', validation.get('predictions_sha256'))
    assert prediction_hash == execution['predictions_sha256']
    rows = json.loads((out/'results.json').read_text())
    old = json.loads(Path('results/phase3/results.json').read_text())
    assert len(rows) == 30
    def one(items, **keys):
        found = [r for r in items if all(r.get(k) == v for k, v in keys.items())]
        assert len(found) == 1, keys
        return found[0]['accuracy_percent']
    def pair(name, scope, historical=False):
        extra = dict(target='MNIST', partner='SVHN') if historical else {}
        return ' / '.join(f'{one(old if historical else rows, defense=name, merger="TA", attack_scope=scope, dataset=t, **extra):.2f}' for t in ['MNIST','SVHN'])
    labels = [('Unprotected','Unprotected'), ('Phase1_MNIST','Defense 1'), ('O/B','Defense 2 O/B†'),
              ('A/B','Defense 2 A/B†'), ('PaRaMS','PaRaMS full‡'), ('PaRaMS_Attention','PaRaMS Attention ablation')]
    lines = ['# PaRaMS diagonal adaptive recovery', '',
        'TA(0.5), protected MNIST / partner SVHN. Each fused cell reports both accuracies (%).', '',
        '| Defense | MNIST standalone: original / recovered | No recovery | AGA: protected expert | Diagonal recovery: protected expert | AGA: all experts | Diagonal recovery: all experts |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name, label in labels:
        before = one(old, defense=name, merger='Standalone', attack_scope='None', dataset='MNIST', target='MNIST', partner='SVHN')
        after = one(rows, defense=name, merger='Standalone', attack_scope='Protected expert', dataset='MNIST')
        lines.append(f'| {label} | {before:.2f} / {after:.2f} | {pair(name,"None",True)} | {pair(name,"Protected expert",True)} | {pair(name,"Protected expert")} | {pair(name,"All experts",True)} | {pair(name,"All experts")} |')
    lines += ['', '† Both frozen Defense 2 variants have known violations of the original 0.5 task budget; effect references only. ‡ Full official PaRaMS is over the 0.5 parameter budget; the Attention ablation is not full PaRaMS.', '',
        'Diagonal recovery uses the unchanged official MLP matching kernel plus an explicitly specified row-wise least-squares Attention adaptation of the paper objective. It is not the complete author attack implementation or paper table reproduction. No defense weights were updated.', '',
        'Historical no-recovery/AGA columns were measured on RTX 3090; new diagonal-recovery columns use RTX 4090 D with the same hashed checkpoints and evaluation functions. They are references across hardware and different recovery operators; do not interpret tiny differences as a controlled hardware-independent gain. All new paired statistics compare against the matching unprotected diagonal-recovery control on RTX 4090 D.', '',
        'Each recovery scope is retained. Tests were previously observed. Only MNIST/SVHN is covered by this new attack evaluation; no cross-protected-task or arbitrary-attack robustness claim.', '',
        f'Frozen attack edits were reused after pilot verification. Full evaluation: {execution["wall_seconds"]:.2f}s; selected-score full-model replays: {validation["complete_model_selected_scores"]}, max score error {validation["maximum_score_error"]:.3g}. Construction and replay costs are additional.', '',
        'Raw predictions and recovered parameters remain on the servers. JSON contains per-class counts, matched paired statistics and signed margin changes.']
    (out/'tables.md').write_text('\n'.join(lines)+'\n')
    fields = ['defense','recovery','merger','attack_scope','dataset','role','n','correct','accuracy_percent','changed_predictions',
              'mcnemar_log10_p','worst_relative_margin_drift_to_matched_control','maximum_signed_margin_harm_to_matched_control']
    with (out/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore'); writer.writeheader(); writer.writerows(rows)
    provenance = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [out/'results.json', out/'execution.json', out/'independent_validation.json', Path('results/phase3/results.json'), Path(__file__)]}
    (out/'table_sources.json').write_text(json.dumps(provenance, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-raw', type=Path)
    args = parser.parse_args()
    if args.verify_raw: verify_margins(args.verify_raw)
    generate()
