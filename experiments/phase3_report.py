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


def save_figure(fig, output, name):
    for ext in ['pdf', 'svg', 'png']:
        metadata = {'CreationDate': None, 'ModDate': None} if ext == 'pdf' else {'Date': None} if ext == 'svg' else None
        fig.savefig(output/f'{name}.{ext}', metadata=metadata)


def dare(source, output):
    """Three prespecified seeds; never pool repeated images as independent draws."""
    source, output = Path(source), Path(output); output.mkdir(parents=True, exist_ok=True)
    evidence = json.loads((source/'execution_and_validation.json').read_text()); rows = []
    assert [e['seed'] for e in evidence] == [42,43,44]
    for e in evidence:
        assert e['execution']['completed'] and e['validation']['status'] == 'passed'
        assert e['execution']['predictions_sha256'] == e['validation']['raw_sha256']
        table = json.loads((source/f'seed{e["seed"]}.json').read_text()); assert len(table) == 42
        rows.extend(dict(r, seed=e['seed']) for r in table)
    names = ['Unprotected','Phase1_MNIST','O/B','A/B','PaRaMS','PaRaMS_Attention']
    labels = ['Unprotected','Defense 1','O/B†','A/B†','PaRaMS (full)*','PaRaMS (attention)']
    scopes = ['None','Protected expert','All experts']; tasks = ['MNIST','SVHN']
    def values(name, scope, task):
        return np.array([next(r['accuracy_percent'] for r in rows if (r['seed'],r['defense'],r['merger'],r['attack_scope'],r['dataset']) ==
            (seed,name,'DARE+TA',scope,task)) for seed in [42,43,44]])
    lines = ['# Fixed DARE+TA and original AGA','','MNIST protected / SVHN partner. Drop0.2, TA0.5; all seeds42/43/44 retained. Cells are mean ± sample SD (%) across three masks, not a confidence interval.','',
        '| Defense | No attack: target / partner | AGA (protected expert): target / partner | AGA (all experts): target / partner |','|---|---:|---:|---:|']
    summary = []
    for name,label in zip(names,labels):
        cells = [label]
        for scope in scopes:
            shown = []
            for task in tasks:
                v = values(name,scope,task); control = values('Unprotected',scope,task); before = values(name,'None',task)
                shown.append(f'{v.mean():.2f} ± {v.std(ddof=1):.2f}')
                summary.append(dict(defense=name,attack_scope=scope,dataset=task,mean_percent=float(v.mean()),sample_SD_pp=float(v.std(ddof=1)),
                    minimum_percent=float(v.min()),maximum_percent=float(v.max()),seed_values=v.tolist(),remaining_gap_mean_pp=float((control-v).mean()),
                    recovery_from_no_attack_mean_pp=float((v-before).mean())))
            cells.append(' / '.join(shown))
        lines.append('| '+' | '.join(cells)+' |')
    lines += ['', '† O/B and A/B violate the task0.5 budget; effect references only. * Full official PaRaMS is over-budget. Defense1 has different constraints. Frozen native accuracies and per-class/paired counts remain in each seed record.', '',
        'Each seed/scope has its own unprotected control. Images repeat across seeds: per-seed paired tests are not pooled. DARE is a merger component, not an additional recovery attack. Official mask function is unchanged; native CLIP adaptation and FP32 reconstruction arithmetic are explicit. This does not reproduce the original LM/GLUE hyperparameter search.', '',
        f'Full evaluation wall time: {sum(e["execution"]["wall_seconds"] for e in evidence):.2f}s; independent selected-score replays: {sum(e["validation"]["full_image_prediction_replays"] for e in evidence)}, maximum error {max(e["validation"]["maximum_score_error"] for e in evidence):.6g}. Replay wall time is additional and was not separately timed.']
    (output/'tables.md').write_text('\n'.join(lines)+'\n'); (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    columns = ['seed','defense','merger','attack_scope','dataset','correct','n','accuracy_percent','changed_predictions','lost','gained','net_loss_pp','mcnemar_log10_p','bonferroni_log10_p']
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in rows:
            if r['merger']=='Standalone': continue
            record={k:r.get(k,'') for k in columns};record.update(lost=r['paired']['lost'],gained=r['paired']['gained'],net_loss_pp=r['paired']['loss_pp'],
                bonferroni_log10_p=min(0.,r['mcnemar_log10_p']+math.log10(90)))
            writer.writerow(record)
    old=json.loads(Path('results/phase3/results.json').read_text()); fig,axes=plt.subplots(1,2,figsize=(9,3.5),sharey=True,layout='constrained')
    fig.suptitle('AGA on the protected expert',fontsize=10)
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False}):
        for ax,task in zip(axes,tasks):
            for i,name in enumerate(names):
                v=values(name,'Protected expert',task); baseline=next(r['accuracy_percent'] for r in old if
                    (r['target'],r['partner'],r['defense'],r['merger'],r['attack_scope'],r['dataset']) == ('MNIST','SVHN',name,'TA','Protected expert',task))
                ax.scatter(i+np.array([-.08,0,.08]),v,color='#2878a5',s=17,label='DARE+TA: three seeds' if i==0 else None)
                ax.errorbar(i,v.mean(),yerr=v.std(ddof=1),color='#2878a5',capsize=3,lw=1)
                ax.scatter(i,baseline,marker='D',facecolors='none',edgecolors='#555555',s=30,label='TA: frozen reference' if i==0 else None)
            ax.set(title=task,ylim=(0,103),xticks=range(len(names)),xticklabels=labels);ax.tick_params(axis='x',rotation=30,labelsize=8);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
        axes[0].set_ylabel('Accuracy (%)');axes[1].legend(loc='lower right',fontsize=8,frameon=False)
        save_figure(fig,output,'dare_aga_comparison');plt.close(fig)
    pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/f'seed{s}.json' for s in [42,43,44]]+[source/'execution_and_validation.json',Path('configs/phase3_dare.json'),Path('results/phase3/results.json'),Path(__file__)]}
    (output/'provenance.json').write_text(json.dumps(dict(source_sha256=pins,scopes=scopes,seeds=[42,43,44],source_test_sets_historically_exposed=True),indent=2)+'\n')


def build(source, output):
    out = Path(output); out.mkdir(parents=True, exist_ok=True)
    source = Path(source); rows, pins, executions, validations = [], {}, [], []
    for case in sorted((source/'full').glob('*')):
        validation = case/'independent_validation.json'
        if not validation.exists(): continue
        checked = json.loads(validation.read_text())
        assert checked['status'] == 'passed'
        validations.append(checked)
        for name in ['results.json', 'execution.json', 'protocol.json', 'independent_validation.json']:
            p = case/name; pins[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
        rows.extend(json.loads((case/'results.json').read_text()))
        executions.append(json.loads((case/'execution.json').read_text()))
    nufilt = source/'nufilt/full'
    nufilt_validation = None
    if (nufilt/'validation.json').exists():
        nufilt_validation = json.loads((nufilt/'validation.json').read_text())
        assert nufilt_validation['status'] == 'passed'
        rows.extend(json.loads((nufilt/'results.json').read_text()))
        for p in [nufilt/'results.json', nufilt/'execution.json', nufilt/'validation.json', nufilt.parent/'protocol.json', nufilt.parent/'pilot/execution.json']:
            pins[str(p)] = hashlib.sha256(p.read_bytes()).hexdigest()
    assert rows, 'No independently verified full results.'
    utility_path = source/'utility/results.json'
    utility_validation_path = source/'utility/independent_validation.json'
    utility_validation = json.loads(utility_validation_path.read_text()) if utility_validation_path.exists() else None
    if utility_validation: assert utility_validation['status'] == 'passed_numpy_score_recount'
    utilities = {r['defense']: r for r in json.loads(utility_path.read_text())} if utility_validation else {}
    if utilities:
        (out/'utility.json').write_text(utility_path.read_text())
        pins[str(utility_validation_path)] = hashlib.sha256(utility_validation_path.read_bytes()).hexdigest()
    witness_path = source/'utility/witnesses.json'
    if witness_path.exists():
        (out/'utility_witnesses.json').write_text(witness_path.read_text())
        pins[str(witness_path)] = hashlib.sha256(witness_path.read_bytes()).hexdigest()
    ab_violation = utilities.get('A/B', {}).get('n_inputs_over_task_budget', 0) > 0
    for r in rows:
        group = [v for v in rows if (v['target'], v['partner'], v['merger'], v['dataset'], v.get('order')) ==
                 (r['target'], r['partner'], r['merger'], r['dataset'], r.get('order'))]
        clean = next(v for v in group if v['defense'] == 'Unprotected' and v['attack_scope'] == r['attack_scope'])
        before = next(v for v in group if v['defense'] == r['defense'] and v['attack_scope'] == 'None')
        clean_before = next(v for v in group if v['defense'] == 'Unprotected' and v['attack_scope'] == 'None')
        r.update(matched_unprotected_accuracy_percent=clean['accuracy_percent'],
                 defense_drop_pp=clean_before['accuracy_percent']-before['accuracy_percent'],
                 attack_recovery_pp=r['accuracy_percent']-before['accuracy_percent'],
                 remaining_gap_pp=clean['accuracy_percent']-r['accuracy_percent'])
        native = next(v for v in rows if (v['target'],v['partner'],v['defense'],v['merger']) ==
                      (r['target'],r['partner'],r['defense'],'Standalone'))
        r['native_target_accuracy_percent'] = native['accuracy_percent']
        peer_native = [v for v in rows if v['target'] == r['partner'] and v['defense'] == 'Unprotected' and v['merger'] == 'Standalone']
        r['native_partner_accuracy_percent'] = peer_native[0]['accuracy_percent'] if peer_native else None
        u = utilities.get(r['defense'])
        r['parameter_distance_changed_tensors'] = u['parameter_distance_changed_tensors'] if u else (0. if r['defense'] == 'Unprotected' else None)
        r['finite512_worst_task_drift'] = u['worst_relative_task_margin_drift'] if u else (0. if r['defense'] == 'Unprotected' else None)
        r['budget_status'] = ('Finite audit exceeds 0.5; ' if u and u['n_inputs_over_task_budget'] else '') + (u['prior_budget_status'] if u else 'Unprotected' if r['defense']=='Unprotected' else 'See frozen protocol')
    comparisons = [r for r in rows if r['defense'] != 'Unprotected' and r['merger'] != 'Standalone']
    for r in comparisons:
        r['mcnemar_bonferroni_log10_p'] = min(0., r['mcnemar_log10_p']+math.log10(len(comparisons)))
    (out/'results.json').write_text(json.dumps(rows, indent=2)+'\n')
    columns = ['target', 'partner', 'defense', 'merger', 'order', 'attack_scope', 'dataset', 'role',
               'native_target_accuracy_percent', 'native_partner_accuracy_percent',
               'parameter_distance_changed_tensors', 'finite512_worst_task_drift', 'budget_status',
               'correct', 'n', 'accuracy_percent', 'matched_unprotected_accuracy_percent',
               'defense_drop_pp', 'attack_recovery_pp', 'remaining_gap_pp',
               'changed_predictions', 'lost', 'gained', 'net_loss_pp',
               'mcnemar_log10_p', 'mcnemar_bonferroni_log10_p', 'evaluation_seconds', 'imported_from']
    with (out/'results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader()
        for r in rows:
            entry = {k: r.get(k, '') for k in columns}
            entry['order'] = '/'.join(r.get('order', []))
            entry.update(lost=r['paired']['lost'], gained=r['paired']['gained'], net_loss_pp=r['paired']['loss_pp'],
                         imported_from=r.get('provenance', {}).get('imported_from', ''))
            writer.writerow(entry)
    scopes = ['None', 'Protected expert', 'All experts']
    titles = ['No attack', 'AGA (protected expert)', 'AGA (all experts)']
    names = ['Unprotected', 'Phase1_MNIST', 'O/B', 'A/B', 'PaRaMS', 'PaRaMS_Attention']
    labels = ['Unprotected', 'Phase 1 (LN bias)*', 'O/B†', 'A/B†' if ab_violation else 'A/B', 'PaRaMS (full)*', 'PaRaMS (attention)']
    primary = [r for r in rows if (r['target'], r['partner']) == ('MNIST', 'SVHN')]
    def get(table, name, method, scope, task):
        hits = [r for r in table if (r['defense'], r['merger'], r['attack_scope'], r['dataset']) == (name, method, scope, task)]
        assert len(hits) == 1, (name, method, scope, task)
        return hits[0]
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9,
                         'pdf.fonttype': 42, 'svg.fonttype': 'none', 'axes.spines.top': False,
                         'svg.hashsalt': 'posthoc-merge-security-phase3',
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
    fig.suptitle('MNIST–SVHN model merging', fontsize=12)
    fig.supxlabel('* Different-budget references. † Task-preservation bound exceeded (see finite audit).', fontsize=8)
    save_figure(fig, out, 'attack_comparison')
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
        ax.set_title(f'AGA → {method} (protected expert)', loc='left')
        ax.set_xticks(range(10)); ax.set_xlabel('MNIST class')
        ax.set_yticks(range(4), [labels[i] for i in [1, 2, 3, 4]])
    fig.colorbar(image, ax=axes, label='Net class accuracy loss (pp)', shrink=.9)
    fig.supxlabel('Compared with matched unprotected controls. * Different-budget references. † Task budget exceeded.', fontsize=8)
    save_figure(fig, out, 'class_losses')
    plt.close(fig)
    lines = ['# Phase 3 evaluation results', '',
             'Accuracy (%). Frozen defenses; historically exposed test sets. No defense was selected or updated from these results.', '',
             '*Phase 1 and full PaRaMS are different-budget references. † Task-preservation bound exceeded. Earlier finite confirmations do not prove global feasibility.*', '']
    if ab_violation:
        lines += ['The read-only 512-image audit also finds an A/B violation. Its historical confirmation and merge results remain valid observations; it cannot be presented as globally satisfying the 0.5 task bound.', '']
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
                latex.append(label.replace('*', r'$^{*}$').replace('†', r'$^{\dagger}$')+f' & {native:.2f} & '+' & '.join(pairs)+r' \\')
            lines.append(''); latex += [r'\bottomrule', r'\end{tabular}', '']
    if nufilt_validation:
        lines += ['## MNIST → SVHN: NUFILT', '',
                  'Official kernels with a native weight adapter; fixed author defaults. Both task orders are reported. This is not the full FusionBench application/table.', '',
                  '| Task order | Defense | No attack | AGA (protected expert) | AGA (all experts) |',
                  '|---|---|---:|---:|---:|']
        nr = [r for r in rows if r['merger'] == 'NUFILT']
        for order in [('MNIST', 'SVHN'), ('SVHN', 'MNIST')]:
            table = [r for r in nr if tuple(r['order']) == order]
            for name in ['Unprotected', 'Phase1_MNIST', 'A/B']:
                pairs = [f'{get(table,name,"NUFILT",s,"MNIST")["accuracy_percent"]:.2f} / {get(table,name,"NUFILT",s,"SVHN")["accuracy_percent"]:.2f}' for s in scopes]
                lines.append(f'| {" → ".join(order)} | {labels[names.index(name)]} | '+' | '.join(pairs)+' |')
        lines += ['', 'Phase 1 scores reuse the exact same merged models after checking all selected inputs are identical. Unselected biases, LN, fc2 and visual projection remain at the public anchor. No defense was changed. Merger gates were optimized on parameter projection losses only: 12 distinct two-task merges × 50 global backward steps, with zero expert/defense updates. Full evaluation uses frozen pilot merge weights.', '']
        latex += ['% NUFILT: same native adapter and fixed defaults; both task orders, one fixed seed.',
                  r'\begin{tabular}{llrrr}', r'\toprule', r'Order & Defense & No attack & AGA (protected) & AGA (all) \\', r'\midrule']
        for order in [('MNIST', 'SVHN'), ('SVHN', 'MNIST')]:
            table = [r for r in nr if tuple(r['order']) == order]
            for name in ['Unprotected', 'Phase1_MNIST', 'A/B']:
                pairs = [f'{get(table,name,"NUFILT",s,"MNIST")["accuracy_percent"]:.2f} / {get(table,name,"NUFILT",s,"SVHN")["accuracy_percent"]:.2f}' for s in scopes]
                label = labels[names.index(name)].replace('*', r'$^{*}$').replace('†', r'$^{\dagger}$')
                latex.append('--'.join(order)+' & '+label+' & '+' & '.join(pairs)+r' \\')
        latex += [r'\bottomrule', r'\end{tabular}', '']
    lines += ['## Evidence and limits', '',
              f'- Independently verified cases: {len(executions)}/4; result rows: {len(rows)}.',
              '- TA coefficient 0.5; TIES top 20%, mass signs, disjoint mean, coefficient 1.0. Each has a separate unprotected control.',
              '- Original AGA leaves final LN bias unchanged. Phase 1 accuracy collapse does not establish coverage against arbitrary recovery.',
              '- Per-class counts, paired losses/gains, approximate paired 95% intervals and exact McNemar log-p are in results.json/CSV. Bonferroni correction covers all non-native protected comparisons in this generated batch.',
              '- Figures show fixed deterministic configurations, not seed means. Significance is not a defense-success definition.',
              '- No DARE, full TSV-Merge, data-assisted adaptation or distillation result is claimed in this batch.', '']
    if utilities:
        lines += ['## Finite utility and parameter audit', '',
                  'Fixed 512 training images per task; read-only evaluation, historical overlap possible. This is not a global or fresh-confirmation certificate.', '',
                  '| Defense | Task | Changed-tensor distance | Worst margin drift | Inputs above 0.5 |',
                  '|---|---|---:|---:|---:|']
        for name, u in utilities.items():
            distance = u['parameter_distance_changed_tensors']
            shown = f'{distance:.6g}' if distance is not None else 'NA (zero reference norm)'
            lines.append(f'| {name} | {u["target"]} | {shown} | {u["worst_relative_task_margin_drift"]:.6g} | {u["n_inputs_over_task_budget"]}/512 |')
        lines.append('')
    (out/'tables.md').write_text('\n'.join(lines))
    (out/'tables.tex').write_text('\n'.join(latex))
    (out/'provenance.json').write_text(json.dumps(dict(status='verified_batch', cases=len(executions),
        source_files_sha256=pins, independent_validations=validations, plotting_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        nufilt_validation=nufilt_validation,
        nufilt_pilot_execution=json.loads((nufilt.parent/'pilot/execution.json').read_text()) if nufilt_validation else None,
        nufilt_full_execution=json.loads((nufilt/'execution.json').read_text()) if nufilt_validation else None,
        utility_independent_validation=utility_validation,
        utility_source_sha256=hashlib.sha256(utility_path.read_bytes()).hexdigest() if utilities else None,
        evaluation_wall_seconds=sum(x['wall_seconds'] for x in executions)), indent=2)+'\n')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='outputs/phase3'); p.add_argument('--output', default='results/phase3')
    p.add_argument('--dare', action='store_true'); args = p.parse_args(); (dare if args.dare else build)(args.source, args.output)
