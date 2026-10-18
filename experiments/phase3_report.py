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


def eight_task(source, output):
    """Recount the fixed eight-expert experiment; only three tasks were scored."""
    import time
    import torch
    from clip_benchmark import digest
    from verify_covariance_noise import official_labels
    from verify_prodistill import verify_metrics
    start=time.monotonic();source,output=Path(source),Path(output)
    cfg=json.loads((source/'full/protocol.json').read_text())
    assert cfg==json.loads((source/'pilot/protocol.json').read_text())
    assert digest('experiments/phase3_eight_task.py')==cfg['source_sha256']
    assert digest(cfg['author_TaskVector']['path'])==cfg['author_TaskVector']['sha256']
    for ref in cfg['inputs'].values():assert digest(ref['path'])==ref['sha256']
    active=json.loads(Path(cfg['inputs']['active']['path']).read_text())
    for p,sha in active['source_sha256'].items():assert digest(p)==sha,p
    frozen=json.loads((source/'pilot/pre_evaluation.json').read_text())
    assert frozen==json.loads((source/'full/pre_evaluation.json').read_text())
    expected=sum(2*(1+len(c['defenses']))*len(cfg['evaluate']) for c in cfg['cases'])
    maximum=0.;count=0;validations=[];pins={}
    for stage in ['pilot','full']:
        root=source/stage;v=json.loads((root/'validation.json').read_text());assert v['status']=='passed'
        assert digest(root/'scores.pt')==v['raw_sha256'] and v['rows']==expected
        raw=torch.load(root/'scores.pt',map_location='cpu',weights_only=False)
        rows=json.loads((root/'results.json').read_text());assert len(rows)==expected
        for row in rows:
            key='/'.join([row['target'],row['defense'],row['attack_scope']]);task=row['dataset'];x=raw[key+'/'+task]
            labels=official_labels(task);ids=list(range(len(labels))) if stage=='full' else np.linspace(0,len(labels)-1,64,dtype=int).tolist()
            assert x['indices']==ids;np.testing.assert_array_equal(x['target'].numpy(),labels[ids])
            assert row['model_parameter_sha256']==frozen[key] and row['NumPy_weight_error']==0.
            before=raw['/'.join([row['target'],'Unprotected',row['attack_scope'],task])]['logits']
            maximum=max(maximum,verify_metrics(x['logits'],x['target'],before,row));count+=1
            if stage=='full':
                prediction,reference=x['logits'].numpy().argmax(1),before.numpy().argmax(1)
                y=x['target'].numpy();correct,was=prediction==y,reference==y
                for c in row['per_class']:
                    mask=y==c['class_id'];lost=int((mask&was&~correct).sum());gained=int((mask&~was&correct).sum())
                    c.update(lost=lost,gained=gained,net_loss_count=lost-gained,net_loss_pp=100*(lost-gained)/c['n'])
                assert sum(c['lost'] for c in row['per_class'])==row['paired']['lost']
                assert sum(c['gained'] for c in row['per_class'])==row['paired']['gained']
        validations.append(v);pins[str(root/'scores.pt')]=v['raw_sha256']
    output.mkdir(parents=True,exist_ok=True)
    def get(target,name,scope,task):
        return next(r for r in rows if (r['target'],r['defense'],r['attack_scope'],r['dataset'])==(target,name,scope,task))
    for r in rows:
        t,n,s,k=r['target'],r['defense'],r['attack_scope'],r['dataset']
        r['defense_drop_without_attack_pp']=get(t,'Unprotected','None',k)['accuracy_percent']-get(t,n,'None',k)['accuracy_percent']
        r['attack_recovery_pp']=r['accuracy_percent']-get(t,n,'None',k)['accuracy_percent']
        r['remaining_gap_pp']=get(t,'Unprotected',s,k)['accuracy_percent']-r['accuracy_percent']
    comparisons=sum(r['defense']!='Unprotected' for r in rows)
    for r in rows:r['bonferroni_log10_p']=min(0.,r['mcnemar_log10_p']+math.log10(comparisons))
    labels={'Unprotected':'Unprotected','O/B':'O/B†','A/B':'A/B†','PaRaMS':'PaRaMS (full)*','PaRaMS_Attention':'PaRaMS (attention)'}
    lines=['# Eight participating experts; three evaluated tasks','','Original author task order, TA0.2, and AGA applied to the protected expert. Accuracy (%); no eight-task average. The other five partner datasets were not evaluated. This setting changes both coefficient and participant count relative to the two-expert TA0.5 table.','']
    latex=[]
    for case in cfg['cases']:
        target=case['target'];tasks=[target]+[t for t in cfg['evaluate'] if t!=target]
        lines += ['## Protected task: '+target,'','Cells: '+' / '.join(tasks)+' (target / evaluated partners).','',
            '| Defense | No attack | AGA (protected expert) |','|---|---:|---:|']
        latex += ['% Protected task: '+target+'; cells: '+' / '.join(tasks),r'\begin{tabular}{lrr}',r'\toprule',r'Defense & No attack & AGA (protected expert) \\',r'\midrule']
        for name in ['Unprotected']+case['defenses']:
            label=labels.get(name,'Defense 1*')
            cells=[' / '.join(f'{get(target,name,s,t)["accuracy_percent"]:.2f}' for t in tasks) for s in ['None','Protected expert']]
            lines.append('| '+label+' | '+' | '.join(cells)+' |')
            latex.append(label.replace('†',r'$^{\dagger}$').replace('*',r'$^{*}$')+' & '+' & '.join(cells)+r' \\')
        lines.append('');latex += [r'\bottomrule',r'\end{tabular}','']
    lines += ['† O/B and A/B violate the task-preservation budget; effect references only. * Defense 1 has different constraints; full PaRaMS is over-budget. The attention ablation is not the complete PaRaMS method. Historically exposed tests; no tuning.','',
        f'Per-class and paired statistics compare each row with its own unprotected control at the same attack scope. Bonferroni correction covers all {comparisons} protected comparisons. Recovery is attacked minus unattacked protected accuracy; remaining gap is matched unprotected minus protected accuracy. Neither distance nor high accuracy proves unique-knowledge extraction. Independent NumPy metric recount is separate from the producer\'s bitwise NumPy reconstruction of all merged parameter tensors; this is not a second implementation of the complete author attack.']
    (output/'tables.md').write_text('\n'.join(lines)+'\n');(output/'tables.tex').write_text('\n'.join(latex)+'\n')
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(output/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    columns=['target','defense','attack_scope','dataset','role','correct','n','accuracy_percent','defense_drop_without_attack_pp',
        'attack_recovery_pp','remaining_gap_pp','changed_predictions','mcnemar_log10_p','bonferroni_log10_p','model_parameter_sha256']
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns,extrasaction='ignore');writer.writeheader();writer.writerows(rows)
    result=dict(status='passed_independent_numpy_metrics_and_frozen_artifact_hashes',rows_checked=count,maximum_logp_error=maximum,
        pilot_and_full_validations=validations,raw_files_sha256=pins,model_parameter_hashes=frozen,
        report_source_sha256=digest(__file__),audit_and_table_seconds=time.monotonic()-start,new_GPU_forwards=0)
    (output/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')


def controls(source, output):
    """Independent raw-score recount and one knowledge-source comparison table."""
    import time
    import torch
    from clip_benchmark import digest
    from verify_covariance_noise import official_labels
    from verify_prodistill import verify_metrics
    start=time.monotonic();source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((source/'full/protocol.json').read_text());baseline={};pins={};maximum=0.;count=0
    assert cfg==json.loads((source/'pilot/protocol.json').read_text())
    for path,sha in cfg['source_sha256'].items():assert digest(path)==sha,path
    if source.name=='single':
        for t,ref in cfg['native_scores'].items():
            assert digest(ref['path'])==ref['sha256'];pins[ref['path']]=ref['sha256']
            baseline[t]=torch.load(ref['path'],map_location='cpu',weights_only=False)[ref['key']]['logits']
    else:
        eight=json.loads(Path(cfg['eight_task_protocol']['path']).read_text());p=Path(eight['output'])/'full'
        v=json.loads((p/'validation.json').read_text());assert v['status']=='passed' and digest(p/'scores.pt')==v['raw_sha256']
        old=torch.load(p/'scores.pt',map_location='cpu',weights_only=False);pins[str(p/'scores.pt')]=v['raw_sha256']
        baseline={t:old['MNIST/Unprotected/None/'+t]['logits'] for t in cfg['tasks']}
    validations=[];frozen=json.loads((source/'pilot/pre_evaluation.json').read_text())
    assert frozen==json.loads((source/'full/pre_evaluation.json').read_text())
    for stage in ['pilot','full']:
        p=source/stage;v=json.loads((p/'validation.json').read_text());assert v['status']=='passed'
        assert digest(p/'scores.pt')==v['raw_sha256'];pins[str(p/'scores.pt')]=v['raw_sha256'];validations.append(v)
        raw=torch.load(p/'scores.pt',map_location='cpu',weights_only=False);rows=json.loads((p/'results.json').read_text())
        assert len(rows)==len(frozen)*len(cfg['tasks'])
        for row in rows:
            x=raw[row['model']+'/'+row['dataset']];ids=x['indices']
            labels=official_labels(row['dataset']);expected=list(range(len(labels))) if stage=='full' else np.linspace(0,len(labels)-1,64,dtype=int).tolist()
            assert ids==expected;np.testing.assert_array_equal(x['target'].numpy(),labels[ids])
            assert row['model_parameter_sha256']==frozen[row['model']]
            maximum=max(maximum,verify_metrics(x['logits'],x['target'],baseline[row['dataset']][ids],row));count+=1
    lines=['# Frozen knowledge-source controls','','Accuracy (%) with the same three fixed classification heads. Historically exposed tests; no tuning.','',
        '| Model | MNIST | DTD | SVHN |','|---|---:|---:|---:|']
    latex=[r'\begin{tabular}{lrrr}',r'\toprule',r'Model & MNIST & DTD & SVHN \\',r'\midrule']
    names=list(frozen);values=[]
    for name in names:
        v=[next(r['accuracy_percent'] for r in rows if (r['model'],r['dataset'])==(name,t)) for t in cfg['tasks']];values.append(v)
        label=name if source.name=='single' else 'Without '+name
        lines.append('| '+label+' | '+' | '.join(f'{a:.2f}' for a in v)+' |');latex.append(label+' & '+' & '.join(f'{a:.2f}' for a in v)+r' \\')
    lines+=['','Single-expert diagonal scores reuse hash-verified native records. Leave-target-out retains the other seven TA coefficients at0.2 without renormalizing. Neither source accuracy nor a leave-out difference proves unique-knowledge extraction. Per-class and paired statistics use the original dataset expert (single models) or the eight-expert unprotected TA control (leave-out).']
    (output/'tables.md').write_text('\n'.join(lines)+'\n');(output/'tables.tex').write_text('\n'.join(latex+[r'\bottomrule',r'\end{tabular}'])+'\n')
    (output/'results.json').write_text(json.dumps(rows,indent=2)+'\n');(output/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
    result=dict(status='passed_independent_numpy_recount_and_hashes',rows_checked=count,maximum_logp_error=maximum,
        pilot_and_full_validations=validations,raw_files_sha256=pins,model_parameters_sha256=frozen,report_source_sha256=digest(__file__),
        audit_and_table_seconds=time.monotonic()-start,new_GPU_forwards=0)
    (output/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')
    if source.name=='single':
        with plt.rc_context({'font.size':9}):
            fig,ax=plt.subplots(figsize=(4.5,3),layout='constrained');image=ax.imshow(values,vmin=0,vmax=100,cmap='Blues',aspect='auto')
            ax.set(xticks=range(3),xticklabels=cfg['tasks'],yticks=range(len(names)),yticklabels=names,xlabel='Evaluation task',ylabel='Frozen model')
            for i,v in enumerate(values):
                for j,a in enumerate(v):ax.text(j,i,f'{a:.2f}',ha='center',va='center',color='white' if a>65 else '#222222')
            fig.colorbar(image,ax=ax,label='Accuracy (%)',shrink=.85);save_figure(fig,output,'knowledge_sources');plt.close(fig)


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


def nufilt_ob(source, output):
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    rows=json.loads((source/'results.json').read_text());valid=json.loads((source/'validation.json').read_text());cost=json.loads((source/'cost_audit.json').read_text())
    assert valid['status']=='passed' and len(rows)==36 and cost['new_merger_backward_steps']==300
    lines=['# Frozen O/B under NUFILT and original AGA','','O/B is an effect reference with known task0.5 violations. Every accuracy cell is MNIST protected / SVHN partner (%). Two orders are reported separately, not as random seeds.','',
        '| Merge order | Recovery scope | Matched unprotected | O/B | Remaining gap (pp) |','|---|---|---:|---:|---:|']
    for order in [['MNIST','SVHN'],['SVHN','MNIST']]:
        for scope in ['None','Protected expert','All experts']:
            def cells(name):return [next(r for r in rows if (r['order'],r['attack_scope'],r['defense'],r['dataset'])==(order,scope,name,t)) for t in ['MNIST','SVHN']]
            a,b=cells('Unprotected'),cells('O/B');values=[' → '.join(order),scope]
            values+=[' / '.join(f'{r["accuracy_percent"]:.2f}' for r in records) for records in [a,b]]
            values.append(' / '.join(f'{u["accuracy_percent"]-r["accuracy_percent"]:.2f}' for u,r in zip(a,b)));lines.append('| '+' | '.join(values)+' |')
    lines+=['','Only12 O/B dataset results and six merged models are new. The24 matched unprotected/Phase1 records reuse hash-verified earlier scores; unchanged selected weights establish the Phase1 reuse. This is an official NUFILT kernel with native CLIP adaptation, not the complete FusionBench paper experiment.','',
        f'All per-class and paired counts passed NumPy recount; {valid["full_replay_count"]} selected-score replays in original contexts, maximum error {valid["maximum_score_error"]:.6g}. Full raw scores and merged weights remain on3090.','',
        f'The original aggregate wall timers are invalid because a replay index overwrote the timer. They are retained unchanged. Valid sub-costs: merger construction {cost["measured_merger_seconds"]:.2f}s, full evaluation {cost["measured_full_evaluation_seconds"]:.2f}s,300 merger-gate backwards,0 expert/defense updates. The filesystem campaign interval was {cost["timestamp_campaign_seconds"]:.2f}s including pilot, full evaluation, replays and gaps; it is not isolated GPU timing.']
    (output/'tables.md').write_text('\n'.join(lines)+'\n');pins={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [source/'results.json',source/'validation.json',source/'cost_audit.json',Path(__file__)]}
    (output/'provenance.json').write_text(json.dumps(pins,indent=2)+'\n')


def prodistill(source, output):
    """Report fixed data-assisted pairs; no model or checkpoint selection."""
    source, output = Path(source), Path(output); output.mkdir(parents=True, exist_ok=True)
    protocol=json.loads((source/'00/protocol.json').read_text())
    rows, evidence, calibrations = [], [], {}
    for i,case in enumerate(protocol['cases']):
        folder = source/f'{i:02d}'
        read = lambda name: json.loads((folder/name).read_text())
        training, freeze = read('training.json'), read('pre_evaluation.json')
        pilot, full = read('pilot/validation.json'), read('full/validation.json')
        assert training['completed'] and pilot['status'] == full['status'] == 'passed'
        assert training['model_sha256'] == freeze['model_sha256'] == full['model_sha256']
        assert training['backward_steps'] == 1200 and len(training['layers']) == 12
        assert training['expert_updates'] == training['defense_updates'] == 0
        pair=case.get('target','MNIST')+'/'+case.get('partner','SVHN');ids=read('calibration_indices.json')
        if pair in calibrations:assert ids==calibrations[pair]
        calibrations[pair]=ids
        rows.extend(read('full/results.json'))
        evidence.append(dict(case=i,case_metadata=case,training=training,freeze=freeze,pilot=pilot,full=full))
    assert len(rows)==sum(4 if c['defense']=='Unprotected' else 2 for c in protocol['cases'])
    scopes = ['None', 'Protected expert', 'All experts']
    get = lambda target,name,scope,task: next(r for r in rows if (r['target'],r['defense'],r['attack_scope'],r['dataset']) == (target,name,scope,task))
    comparisons = sum(r['defense'] != 'Unprotected' for r in rows)
    for row in rows:
        target,name,scope,task=row['target'],row['defense'],row['attack_scope'],row['dataset']
        row.update(matched_unprotected_accuracy_percent=get(target,'Unprotected',scope,task)['accuracy_percent'],
            defense_drop_pp=get(target,'Unprotected','None',task)['accuracy_percent']-get(target,name,'None',task)['accuracy_percent'],
            attack_recovery_pp=row['accuracy_percent']-get(target,name,'None',task)['accuracy_percent'],
            remaining_gap_pp=get(target,'Unprotected',scope,task)['accuracy_percent']-row['accuracy_percent'],
            mcnemar_bonferroni_log10_p=min(0.,row['mcnemar_log10_p']+math.log10(comparisons)))
    lines = ['# ProDistill: fixed two-expert evaluation', '',
        'Cells are protected task / partner accuracy (%). Attacker uses64 unlabeled train images per task and task membership; only merger coefficients receive gradients. This is a native two-expert adapter of the official kernel, not the original eight-task table.']
    latex=[]
    for target,partner in sorted({(r['target'],r['partner']) for r in rows}):
        names=['Unprotected','Phase1_'+target]+[n for n in ['O/B','A/B'] if any(r['target']==target and r['defense']==n for r in rows)]
        lines+=['',f'## {target} / {partner}','','| Defense | No alignment | AGA (protected expert) | AGA (all experts) |','|---|---:|---:|---:|']
        latex += [f'% {target} / {partner}',r'\begin{tabular}{lrrr}',r'\toprule',r'Defense & No alignment & AGA (protected) & AGA (all) \\',r'\midrule']
        for name in names:
            label='Defense 1' if name.startswith('Phase1_') else name+'†' if name in ['O/B','A/B'] else name
            cells=[' / '.join(f'{get(target,name,scope,t)["accuracy_percent"]:.2f}' for t in [target,partner]) for scope in scopes]
            lines.append('| '+' | '.join([label]+cells)+' |');latex.append(label.replace('†',r'$^{\dagger}$')+' & '+' & '.join(cells)+r' \\')
        latex += [r'\bottomrule',r'\end{tabular}','']
    train_seconds = sum(e['training']['wall_seconds'] for e in evidence)
    eval_seconds = sum(r['evaluation_seconds'] for r in rows)
    checked = sum(e['full']['score_replays'] for e in evidence)
    maximum = max(e['full']['maximum_context_score_error'] for e in evidence)
    lines += ['', '† O/B and A/B have independently confirmed task0.5 violations: effect references only. Defense1 uses a different parameter constraint. Frozen standalone utility, SVD probes and budgets remain in ../utility.json; no defense is retrained or selected from these results.', '',
        f'Every scope has a matched unprotected control. Equal accuracy does not imply equal predictions: raw per-class counts and lost/gained pairs are retained. Bonferroni correction covers{comparisons} protected comparisons in this batch. Historically exposed tests are not new blind evaluation.', '',
        'Defense1 shares all observed prefixes/blocks with the unprotected control. Its identical coefficient training is reused only after exact input-weight equality; its own frozen final LN bias is retained. This attack does not optimize final LN/P. Failure against that bias does not establish resistance to attacks that cover it; the separately reported bias canonicalization already restores it.', '',
        f'{len(evidence)} coefficient fits,{sum(e["training"]["backward_steps"] for e in evidence)} backwards,0 expert/defense updates. Training wall time {train_seconds:.2f}s; full evaluation sub-cost {eval_seconds:.2f}s. {checked} original-context score replays passed with maximum error {maximum:.6g}. Assessment wall times including their replays are retained per case; the sub-costs are not whole-project GPU time.']
    for name,value in [('results.json',rows),('execution_and_validation.json',evidence)]:
        (output/name).write_text(json.dumps(value,indent=2)+'\n')
    (output/'tables.md').write_text('\n'.join(lines)+'\n');(output/'tables.tex').write_text('\n'.join(latex)+'\n')
    (output/'calibration_indices.json').write_text(json.dumps(calibrations,indent=2)+'\n')
    (output/'protocol.json').write_bytes((source/'00/protocol.json').read_bytes())
    (output/'kernel_validation.json').write_bytes((source/'kernel_validation.json').read_bytes())
    columns = ['target','partner','defense','attack_scope','dataset','n','correct','accuracy_percent','matched_unprotected_accuracy_percent',
        'defense_drop_pp','attack_recovery_pp','remaining_gap_pp','changed_predictions','lost','gained','mcnemar_log10_p','mcnemar_bonferroni_log10_p']
    with (output/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=columns);writer.writeheader()
        for r in rows:
            item={k:r.get(k,'') for k in columns};item.update(lost=r['paired']['lost'],gained=r['paired']['gained']);writer.writerow(item)
    pins = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.json')}
    pins[str(Path(__file__))] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (output/'provenance.json').write_text(json.dumps(dict(source_sha256=pins,training_seconds=train_seconds,
        full_evaluation_seconds=eval_seconds,full_score_replays=checked,maximum_score_error=maximum,
        whole_author_training_independently_reimplemented=False,labels_used_for_optimization=False,task_membership_used=True),indent=2)+'\n')


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
    modes=dict(dare=dare,nufilt_ob=nufilt_ob,prodistill=prodistill,controls=controls,eight_task=eight_task)
    g=p.add_mutually_exclusive_group()
    for name in modes:g.add_argument('--'+name.replace('_','-'),action='store_true')
    args=p.parse_args();next((fn for name,fn in modes.items() if getattr(args,name)),build)(args.source,args.output)
