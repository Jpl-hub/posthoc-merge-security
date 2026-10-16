"""Generate paper tables and figures exclusively from verified Phase 3 records."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def build(source, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    source = Path(source); rows, pins, executions = [], {}, []
    for case in sorted((source/'full').glob('*')):
        validation = case/'independent_validation.json'
        if not validation.exists(): continue
        assert json.loads(validation.read_text())['status'] == 'passed'
        for name in ['results.json', 'execution.json', 'protocol.json', 'independent_validation.json']:
            p = case/name; pins[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.extend(json.loads((case/'results.json').read_text()))
        executions.append(json.loads((case/'execution.json').read_text()))
    assert rows, 'No independently verified full results.'
    comparisons = [r for r in rows if r['defense'] != 'Unprotected' and r['merger'] != 'Standalone']
    for r in comparisons:
        r['mcnemar_bonferroni_log10_p'] = min(0., r['mcnemar_log10_p']+math.log10(len(comparisons)))
    (out/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    columns = ['target', 'partner', 'defense', 'merger', 'attack_scope', 'dataset', 'role',
               'correct', 'n', 'accuracy_percent', 'changed_predictions', 'lost', 'gained', 'net_loss_pp',
               'mcnemar_log10_p', 'mcnemar_bonferroni_log10_p', 'evaluation_seconds', 'imported_from']
    with (out/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for r in rows:
            entry = {k: r.get(k, '') for k in columns}
            entry.update(lost=r['paired']['lost'], gained=r['paired']['gained'], net_loss_pp=r['paired']['loss_pp'],
                         imported_from=r['provenance'].get('imported_from', ''))
            writer.writerow(entry)
    scopes = ['None', 'Protected expert', 'All experts']
    titles = ['No attack', 'AGA (protected expert)', 'AGA (all experts)']
    names = ['Unprotected', 'Phase1_MNIST', 'O/B', 'A/B', 'PaRaMS', 'PaRaMS_Attention']
    labels = ['Unprotected', 'Phase 1 LN bias*', 'O/B (task violation)', 'A/B', 'PaRaMS (full)*', 'PaRaMS (Attn.)']
    primary = [r for r in rows if (r['target'], r['partner']) == ('MNIST', 'SVHN')]
    def get(table, name, method, scope, task):
        hits = [r for r in table if (r['defense'], r['merger'], r['attack_scope'], r['dataset']) == (name, method, scope, task)]
        assert len(hits) == 1, (name, method, scope, task)
        return hits[0]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 240})
    fig, axes = plt.subplots(2, 2, figsize=(10.6, 7.0), layout='constrained')
    for j, method in enumerate(['TA', 'TIES']):
        for i, task in enumerate(['MNIST', 'SVHN']):
            values = np.array([[get(primary, n, method, s, task)['accuracy_percent'] for s in scopes] for n in names])
            ax = axes[i, j]; image = ax.imshow(values, vmin=0, vmax=100, cmap='cividis', aspect='auto')
            ax.set_xticks(range(3), ['No attack', 'AGA\nprotected expert', 'AGA\nall experts'])
            ax.set_yticks(range(6), labels if j == 0 else [])
            ax.set_title(f'{method}: {task} ({"protected task" if i == 0 else "partner task"})', loc='left')
            for a in range(6):
                for b in range(3):
                    ax.text(b, a, f'{values[a,b]:.2f}', ha='center', va='center',
                            color='white' if values[a,b] < 48 else 'black', fontsize=9)
    fig.colorbar(image, ax=axes, label='Accuracy (%)', shrink=.78)
    fig.suptitle('Frozen MNIST expert merged with SVHN', fontsize=12)
    for ext in ['pdf', 'svg', 'png']: fig.savefig(out/f'attack_comparison.{ext}')
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.5), layout='constrained')
    methods = ['Phase1_MNIST', 'O/B', 'A/B', 'PaRaMS']
    for ax, method in zip(axes, ['TA', 'TIES']):
        losses = []
        for name in methods:
            row = get(primary, name, method, 'Protected expert', 'MNIST')
            losses.append([100*(c['original_correct']-c['correct'])/c['n'] for c in row['per_class']])
        values = np.array(losses)
        image = ax.imshow(values, vmin=-100, vmax=100, cmap='RdBu_r', aspect='auto')
        ax.set_title(f'{method} + AGA (protected expert)', loc='left')
        ax.set_xticks(range(10)); ax.set_xlabel('MNIST class')
        ax.set_yticks(range(4), ['Phase 1 LN bias*', 'O/B (task violation)', 'A/B', 'PaRaMS (full)*'])
    fig.colorbar(image, ax=axes, label='Net accuracy loss vs. matched unprotected model (pp)', shrink=.9)
    for ext in ['pdf', 'svg', 'png']: fig.savefig(out/f'class_losses.{ext}')
    plt.close(fig)
    lines = ['# Phase 3 evaluation results', '',
             'Accuracy (%). Frozen defenses; historically exposed test sets. No defense was selected or updated from these results.', '',
             '*Phase 1 and full PaRaMS are different-budget references. O/B has a known task-preservation violation. A/B has finite confirmation only.*', '']
    latex = []
    for target, partner in sorted({(r['target'], r['partner']) for r in rows}):
        table = [r for r in rows if (r['target'], r['partner']) == (target, partner)]
        ordered = list(dict.fromkeys(r['defense'] for r in table))
        for method in ['TA', 'TIES']:
            lines += [f'## {target} → {partner}: {method}', '',
                      'Each entry is **protected-task / partner-task** accuracy.', '',
                      '| Defense | Standalone target | No attack | AGA (protected expert) | AGA (all experts) |',
                      '|---|---:|---:|---:|---:|']
            latex += [f'% {target} / {partner}: {method}', r'\begin{tabular}{lrrrr}', r'\toprule',
                      r'Defense & Standalone & No attack & AGA (protected) & AGA (all) \\', r'\midrule']
            for name in ordered:
                native = get(table, name, 'Standalone', 'None', target)['accuracy_percent']
                pairs = [f'{get(table,name,method,s,target)["accuracy_percent"]:.2f} / {get(table,name,method,s,partner)["accuracy_percent"]:.2f}' for s in scopes]
                label = labels[names.index(name)] if name in names else 'Phase 1 LN bias*'
                lines.append(f'| {label} | {native:.2f} | '+ ' | '.join(pairs)+' |')
                latex.append(label.replace('*', r'$^{*}$')+f' & {native:.2f} & '+' & '.join(pairs)+r' \\')
            lines.append(''); latex += [r'\bottomrule', r'\end{tabular}', '']
    lines += ['## Evidence and limits', '',
              f'- Independently verified cases: {len(executions)}/4; result rows: {len(rows)}.',
              '- TA coefficient 0.5; TIES top 20%, mass signs, disjoint mean, coefficient 1.0. Each has a separate unprotected control.',
              '- Original AGA leaves final LN bias unchanged. Phase 1 accuracy collapse does not establish coverage against arbitrary recovery.',
              '- Per-class counts, paired losses/gains, approximate paired 95% intervals and exact McNemar log-p are in results.json/CSV. Bonferroni correction covers all non-native protected comparisons in this generated batch.',
              '- Figures show fixed deterministic configurations, not seed means. Significance is not a defense-success definition.',
              '- No DARE, full TSV-Merge, data-assisted adaptation or distillation result is claimed in this batch.', '']
    (out/'tables.md').write_text('\n'.join(lines))
    (out/'tables.tex').write_text('\n'.join(latex))
    (out/'provenance.json').write_text(json.dumps(dict(status='verified_batch', cases=len(executions),
        source_files_sha256=pins, plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        evaluation_wall_seconds=sum(x['wall_seconds'] for x in executions)), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='outputs/phase3'); p.add_argument('--output', default='results/phase3')
    args = p.parse_args(); build(args.source, args.output)
