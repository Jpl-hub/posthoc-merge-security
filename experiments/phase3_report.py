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


def paper_tables(_source, output):
    """Collect frozen comparisons into matching TeX/Markdown paper tables."""
    out=Path(output);out.mkdir(parents=True,exist_ok=True);sources={};used=[];tables=[];loaded={};seen=set()
    def read(path):
        if path in loaded:return loaded[path]
        p=Path(path);sources[path]=hashlib.sha256(p.read_bytes()).hexdigest()
        loaded[path]=json.loads(p.read_text());return loaded[path]
    def pick(path, **where):
        data=read(path);hits=[(i,r) for i,r in enumerate(data) if all(r.get(k)==v for k,v in where.items())]
        assert len(hits)==1,(path,where,len(hits));i,r=hits[0]
        if 'correct' in r:
            value=100*r['correct']/r['n'];reported=r.get('accuracy_percent',100*r.get('top1',value/100))
            assert abs(value-reported)<1e-9,(path,i)
            if 'per_class' in r:assert sum(x['correct'] for x in r['per_class'])==r['correct'] and sum(x['n'] for x in r['per_class'])==r['n']
        if (path,i) not in seen:used.append(dict(path=path,row=i,selector=where));seen.add((path,i))
        return r
    def acc(r):return r.get('accuracy_percent',100*r.get('top1',0))
    def pair(path, tasks=('MNIST','SVHN'), **where):return ' / '.join(f'{acc(pick(path,dataset=t,**where)):.2f}' for t in tasks)
    def table(key,title,headers,rows,note,colspec=None):
        assert all(len(r)==len(headers) for r in rows),key
        tables.append(dict(key=key,title=title,headers=headers,rows=rows,note=note,colspec=colspec or 'l'+'r'*(len(headers)-1)))
    main='results/phase3/results.json';scopes=['None','Protected expert','All experts']
    # These are previously completed independent audits, not new raw-score replays.
    for path in ['results/phase3/provenance.json','results/phase3/params_adaptive/independent_validation.json',
        'results/phase3/bias_canonicalization/validation.json','results/phase3/bias_tail_precision/validation.json',
        'results/phase3/distillation/independent_validation.json','results/phase3/eight_task/independent_validation.json',
        'results/phase3/unmask/independent_validation.json','results/phase3/recovery_overview/provenance.json',
        'results/phase3/nufilt_other_tasks/independent_validation.json','results/phase3/prodistill_other_tasks/independent_validation.json',
        'results/mean_compensation/evaluation/independent_validation.json',
        'results/residual_coordinate_control/evaluation/independent_validation.json',
        'results/mean_compensation/components_independent_validation.json']:
        read(path)
    pairing_root=Path('results/task_directed_pairing')
    pairing_path=pairing_root/'evaluation/full/00_MNIST_SVHN/results.json'
    pairing_complete=pairing_path.with_name('independent_validation.json').exists()
    pairing_other_complete=all((pairing_root/'other_tasks/evaluation/full'/c/'independent_validation.json').exists()
        for c in ['00_DTD_MNIST','01_SVHN_MNIST'])
    distance_root=pairing_root/'distance_ablation';distance_path=distance_root/'evaluation/full/00_MNIST_SVHN/results.json'
    distance_complete=distance_path.with_name('independent_validation.json').exists()
    if pairing_complete:
        check=read(str(pairing_path.with_name('independent_validation.json')))
        execution=read(str(pairing_path.with_name('execution.json')))
        assert check['status']=='passed' and check['raw_sha256']==execution['predictions_sha256']
        assert read(str(pairing_root/'native/MNIST/independent_validation.json'))['passed']
    rows=[]
    names=[('Unprotected','Unprotected'),('PaRaMS','PaRaMS (full)'),('Phase1_MNIST','Defense 1'),('O/B','O/B [H]'),('A/B','A/B [H]')]
    for name,label in names:
        where=dict(target='MNIST',partner='SVHN',defense=name)
        native=acc(pick(main,merger='Standalone',attack_scope='None',dataset='MNIST',**where))
        rows.append([label,f'{native:.2f}']+[pair(main,merger='TA',attack_scope=s,**where) for s in scopes])
    for folder,label in [('residual_coordinate_control','Defense 2 (structure) [K]'),('mean_compensation','Defense 2 candidate (failed) [F]')]:
        path=f'results/{folder}/evaluation/results.json'
        # Matching untreated controls must agree before adding these rows.
        for s,stage in zip(scopes,['direct','AGA_target_only','AGA_both']):
            for task in ['MNIST','SVHN']:
                a=pick(path,preset='original',stage=stage,dataset=task)
                b=pick(main,target='MNIST',partner='SVHN',defense='Unprotected',merger='TA',attack_scope=s,dataset=task)
                assert (a['correct'],a['n'])==(b['correct'],b['n'])
        native=acc(pick(path,preset='released',stage='standalone',dataset='MNIST'))
        rows.append([label,f'{native:.2f}']+[pair(path,preset='released',stage=s) for s in ['direct','AGA_target_only','AGA_both']])
    if pairing_complete:
        path=str(pairing_path);where=dict(defense='TaskDirected_MNIST')
        for scope in scopes:
            for task in ['MNIST','SVHN']:
                a=pick(path,defense='Unprotected',merger='TA',attack_scope=scope,dataset=task)
                b=pick(main,target='MNIST',partner='SVHN',defense='Unprotected',merger='TA',attack_scope=scope,dataset=task)
                assert (a['correct'],a['n'])==(b['correct'],b['n'])
        native=acc(pick(path,merger='Standalone',attack_scope='None',dataset='MNIST',**where))
        rows.append(['Defense 2 (task-directed)',f'{native:.2f}']+[pair(path,merger='TA',attack_scope=s,**where) for s in scopes])
    table('main','Main comparison: two experts, TA coefficient 0.5',
        ['Protection','Standalone M','No recovery M / S','AGA target M / S','AGA both M / S'],rows,
        'M: protected MNIST; S: partner SVHN. AGA target/both denotes recovery of only the protected expert/both experts. [H] Historical O/B and A/B fail the original complete-task bound; effect references only. [K] Known global symmetry control, not an original method; family-aware public recovery is reported separately. [F] Previous candidate preserves the complete task check but does not provide useful protection. Full PaRaMS is the full-default baseline; parameter scales differ and are not a universal exclusion rule.')
    rows=[]
    for target,partner,case in [('MNIST','SVHN',None),('DTD','MNIST','00_DTD_MNIST'),('SVHN','MNIST','01_SVHN_MNIST')]:
        task_names=[('Unprotected','Unprotected'),('Phase1_'+target,'Defense 1'),('PaRaMS','PaRaMS (full)'),('Residual','Defense 2 (structure)')]
        if pairing_complete and (case is None or pairing_other_complete):task_names.append(('TaskDirected','Defense 2 (task-directed)'))
        for name,label in task_names:
            if case and name in ['PaRaMS','Residual','TaskDirected']:
                root='results/'+{'PaRaMS':'params_reference','Residual':'residual_coordinate_control','TaskDirected':'task_directed_pairing'}[name]+'/other_tasks'
                check=read(root+'/'+target+'/independent_validation.json')
                assert check['passed'] or (name=='PaRaMS' and check['complete_native_observation'])
                check=read(root+'/evaluation/full/'+case+'/independent_validation.json');assert check['status']=='passed'
                path=root+'/evaluation/full/'+case+'/results.json';where=dict(defense=name+'_'+target)
                native=acc(pick(path,merger='Standalone',attack_scope='None',dataset=target,**where))
                values=[pair(path,(target,partner),merger='TA',attack_scope=s,**where) for s in scopes]
                for s in scopes:
                    for t in [target,partner]:
                        a=pick(path,defense='Unprotected',merger='TA',attack_scope=s,dataset=t)
                        b=pick(main,target=target,partner=partner,defense='Unprotected',merger='TA',attack_scope=s,dataset=t)
                        assert (a['correct'],a['n'])==(b['correct'],b['n'])
            elif name=='TaskDirected':
                path=str(pairing_path);where=dict(defense='TaskDirected_MNIST')
                native=acc(pick(path,merger='Standalone',attack_scope='None',dataset=target,**where))
                values=[pair(path,(target,partner),merger='TA',attack_scope=s,**where) for s in scopes]
            elif name=='Residual':
                path='results/residual_coordinate_control/evaluation/results.json'
                native=acc(pick(path,preset='released',stage='standalone',dataset=target))
                values=[pair(path,(target,partner),preset='released',stage=s) for s in ['direct','AGA_target_only','AGA_both']]
            else:
                where=dict(target=target,partner=partner,defense=name)
                native=acc(pick(main,merger='Standalone',attack_scope='None',dataset=target,**where))
                values=[pair(main,(target,partner),merger='TA',attack_scope=s,**where) for s in scopes]
            rows.append([target+' / '+partner,label,f'{native:.2f}']+values)
    table('three_tasks','Matched defenses across three protected tasks',
        ['Target / partner','Protection','Standalone','No recovery T / P','AGA target T / P','AGA both T / P'],rows,
        'TA0.5 and unchanged original AGA; every merged cell is protected / partner accuracy. Full PaRaMS uses official defaults without clipping. Structure uses a fixed global rule; task-directed uses the frozen finite objective with each target task SVD. Three protected tasks correspond to two unordered task pairs, not a complete eight-task table. Novelty and known-family/location-aware recovery are separate. Complete original-task predictions, parameter scales and numerical protocols are reported separately.')
    recovery_root=pairing_root/'recovery'
    if (recovery_root/'validation.json').exists():
        check=read(str(recovery_root/'validation.json'));assert check['status']=='passed_independent_public_matching_and_all_tensors'
        recovered=read(str(recovery_root/'results.json'));rows=[]
        for case in recovered['cases']:
            assert case['status']=='exact_original_state_recovered' and not any(case['tensor_differences'])
            path=case['verified_evaluation']['results']['path'];target,peer=case['target'],case['partner']
            for defense,label in [('TaskDirected_'+target,'Without family matching'),('Unprotected','Known-family matching')]:
                rows.append([target+' / '+peer,label]+[pair(path,(target,peer),defense=defense,merger='TA',attack_scope=s) for s in scopes])
        table('task_directed_recovery','Task-directed protection and known-family public recovery',
            ['Target / partner','Recovery rule','TA T / P','AGA target T / P','AGA both T / P'],rows,
            'All percentages. Same published/public-base weight access; targeted matching knows the global residual-permutation family and native CLIP layout. No clean expert, true task vector, secret order or images enter the attack. Each target and matched untreated control equals all158 original tensors exactly; original independently audited scores are therefore reused,not rerun. Original AGA remains unchanged and is reported separately. This does not establish recovery of arbitrary defenses.',colspec='llrrr')
    rows=[]
    for target,partner in [('MNIST','SVHN'),('MNIST','DTD'),('DTD','MNIST'),('SVHN','MNIST')]:
        for name,label in [('Unprotected','Unprotected'),('Phase1_'+target,'Defense 1')]:
            where=dict(target=target,partner=partner,defense=name)
            native=acc(pick(main,merger='Standalone',attack_scope='None',dataset=target,**where))
            rows.append([target+' / '+partner,label,f'{native:.2f}']+[pair(main,(target,partner),merger='TA',attack_scope=s,**where) for s in scopes])
    table('defense1','Defense 1 across three protected tasks',
        ['Target / partner','Protection','Standalone','No recovery T / P','AGA target T / P','AGA both T / P'],rows,
        'TA0.5. Every merged cell is target / partner accuracy. The task role is fixed by the first column, not by a global dataset order. The original AGA does not modify the final LayerNorm bias; these results do not imply resistance to location-aware or data-assisted recovery.')
    rows=[]
    for s,label,access in [('None','No recovery','Published expert and partner'),('Protected expert','AGA target','Public base; defense-agnostic'),('All experts','AGA both','Public base; defense-agnostic')]:
        rows.append([label,access]+[pair(main,target='MNIST',partner='SVHN',defense=n,merger='TA',attack_scope=s) for n in ['Unprotected','Phase1_MNIST']])
    path='results/phase3/params_adaptive/results.json'
    rows.append(['Diagonal adaptation','Specified diagonal family; no images']+[pair(path,defense=n,merger='TA',attack_scope='Protected expert') for n in ['Unprotected','Phase1_MNIST']])
    path='results/phase3/bias_canonicalization/results.json'
    rows.append(['LN-bias cleanup','Known location; published projection']+[pair(path,target='MNIST',partner='SVHN',defense=n,merger='TA',attack_scope='Protected expert') for n in ['Unprotected','Phase1_MNIST']])
    path='results/phase3/distillation/results.json'
    for mode,label in [('TA','Student + TA'),('AGA-target-only','Student + AGA target'),('AGA-both','Student + AGA both')]:
        value=pair(path,mode=mode);rows.append([label,'2048 target-domain queries; student training',value,value])
    table('recovery','Recovery permissions: protected MNIST and partner SVHN',
        ['Recovery pipeline','Additional access','Matched control M / S','Defense 1 M / S'],rows,
        'All merges use TA0.5. Diagonal adaptation is the official MLP kernel plus an explicitly specified native Attention adaptation, not a complete official attack release. Cleanup is a known-location diagnostic. Native FP32 cleanup changes some scores beyond the historical numerical-control tolerance; no exact FP32 function identity is claimed. Student rows share one training problem because all2048 queried hard labels coincide; they are not independent seed replications or the complete PaRaMS KD implementation.',
        r'p{39mm}p{49mm}rr')
    overview='results/phase3/recovery_overview/results.json';rows=[]
    for method in ['TA','TIES','DARE+TA','NUFILT M→S','NUFILT S→M','ProDistill']:
        cells=[' / '.join(f'{pick(overview,merger=method,defense=n,dataset=t)["accuracy_percent"]:.2f}' for t in ['MNIST','SVHN']) for n in ['Unprotected','Phase1_MNIST','O/B','A/B']]
        rows.append([method]+cells)
    table('mergers','Supplementary mergers after original AGA on the protected expert',
        ['Merger','Unprotected M / S','Defense 1 M / S','O/B [H] M / S','A/B [H] M / S'],rows,
        'MNIST target / SVHN partner. Each merger has its own matching control. DARE+TA is the mean of fixed mask seeds42/43/44; other entries are fixed runs. NUFILT reports both task orders and uses native adapters of official kernels. ProDistill uses64+64 unlabeled training calibration images and attacker-side elementwise coefficient training; it is not a data-free attack. Mergers are not counted as separate recovery attacks. [H] Task-bound failure retained.')
    rows=[]
    for name,label in names[:1]+names[2:]:
        cells=[]
        for t in ['MNIST','SVHN']:
            values=pick(overview,merger='DARE+TA',defense=name,dataset=t)['accuracy_values']
            assert len(values)==3;cells.append(f'{np.mean(values):.2f} +/- {np.std(values,ddof=1):.2f}')
        rows.append([label]+cells)
    table('dare','DARE mask variability under AGA target',
        ['Protection','MNIST mean +/- SD','SVHN mean +/- SD'],rows,
        'Sample standard deviation across the same three fixed masks42/43/44. This is mask variability, not three independent expert or defense trainings. All three mask values are retained in the cited source rows.')
    rows=[]
    for target in ['DTD','SVHN']:
        path='results/phase3/nufilt_other_tasks/results.json'
        for order in [[target,'MNIST'],['MNIST',target]]:
            for scope in scopes:
                if target=='SVHN' and scope!='Protected expert':
                    value=pair(main,('SVHN','MNIST'),target='MNIST',partner='SVHN',defense='Unprotected',merger='NUFILT',order=order,attack_scope=scope)
                    cells=[value,value]
                else:cells=[pair(path,(target,'MNIST'),target=target,partner='MNIST',defense=name,order=order,attack_scope=scope) for name in ['Unprotected','Phase1_'+target]]
                rows.append([target+' / MNIST','NUFILT '+' / '.join(order),{'None':'None','Protected expert':'Target','All experts':'Both'}[scope]]+cells)
        path='results/phase3/prodistill_other_tasks/results.json'
        for scope in scopes:
            rows.append([target+' / MNIST','ProDistill',{'None':'None','Protected expert':'Target','All experts':'Both'}[scope]]+
                [pair(path,(target,'MNIST'),target=target,partner='MNIST',defense=name,attack_scope=scope) for name in ['Unprotected','Phase1_'+target]])
    table('other_tasks','Supplementary mergers: DTD and SVHN protected experts',
        ['Target / partner','Merger / order','AGA scope','Control T / P','Defense 1 T / P'],rows,
        'NUFILT uses both task orders. SVHN no-attack/both-AGA cells reuse the identical earlier MNIST/SVHN merger models with dataset roles exchanged; no new fits. The selected-parameter identity for Defense1 is retained. ProDistill uses64+64 unlabeled calibration images and official coefficient kernels. These are merger evaluations, not additional recovery attacks.')
    for folder,key,title,scope in [('eight_task','eight','Eight participating experts: three evaluated tasks','Protected expert'),('unmask','unmask','Paper-formula Unmask: a separate eight-expert recovery setting','Unmask')]:
        path=f'results/phase3/{folder}/results.json';rows=[]
        if folder=='unmask':
            for name in ['rows','validation']:
                ref=read('results/phase3/unmask/protocol.json')['no_attack_reference'][name]
                read(ref['path']);assert sources[ref['path']]==ref['sha256']
        for target in ['MNIST','DTD','SVHN']:
            methods=['Unprotected','Phase1_'+target]+(['O/B','A/B','PaRaMS','PaRaMS_Attention'] if target=='MNIST' else [])
            for name in methods:
                label='Defense 1' if name.startswith('Phase1') else {'O/B':'O/B [H]','A/B':'A/B [H]','PaRaMS':'PaRaMS (full)','PaRaMS_Attention':'PaRaMS attention ablation'}.get(name,name)
                rows.append([target,label]+[pair('results/phase3/eight_task/results.json' if folder=='unmask' and s=='None' else path,
                    ('MNIST','DTD','SVHN'),target=target,defense=name,attack_scope=s) for s in ['None',scope]])
        note='TA0.2 with eight experts; cells always MNIST / DTD / SVHN, including partner tasks. Other five task datasets were not evaluated. No eight-task average. [H] Original task-bound failures retained. Complete PaRaMS and its Attention ablation remain distinct.'
        if folder=='unmask':note+=' Fixed merged parameters minus0.3 times the whole published expert. Both protected and unprotected controls receive the operation. Paper-formula native adaptation, not official code or a complete paper table. No AGA composition. The unprotected model also collapses; low absolute scores do not demonstrate successful expert recovery.'
        table(key,title,['Protected task','Protection','No recovery M / D / S',('AGA target' if folder=='eight_task' else 'Unmask0.3')+' M / D / S'],rows,note)
    rows=[]
    for target,partner,path,mode in [('MNIST','SVHN','results/phase3/bias_canonicalization/results.json','Native FP32'),('MNIST','DTD','results/phase3/bias_canonicalization/results.json','Native FP32'),('DTD','MNIST','results/phase3/bias_tail_precision/results.json','FP32 prefix; FP64 readout'),('SVHN','MNIST','results/phase3/bias_tail_precision/results.json','FP32 prefix; FP64 readout')]:
        for scope,label in [('Protected expert','Target'),('All experts','Both')]:
            rows.append([target+' / '+partner,mode,label]+[pair(path,(target,partner),target=target,partner=partner,merger='TA',defense=name,attack_scope=scope) for name in ['Unprotected','Phase1_'+target]])
    table('cleanup','Targeted bias cleanup: numerical protocols kept separate',
        ['Target / partner','Numerics','Scope','Control T / P','Defense 1 T / P'],rows,
        'TA0.5; no images, original expert or secret perturbation enters recovery construction. The FP64-readout rows are a separately frozen protocol and must not be pooled with native FP32. Original native DTD control failure0.00082493 and MNIST before/after score difference0.00069427 remain recorded. Equal recovered predictions do not establish exact FP32 function identity.')
    geometry=read('results/mean_compensation/allocation_geometry.json');svd=read('results/task_subspace_comparison/results.json')
    old=next(g for g in svd['groups'] if g['method']=='global_permutation' and g['site']=='all')
    geometry_rows=[]
    for folder,label,di,do in [('residual_coordinate_control','Global permutation [K]',old['distances']['right_inside_distance'],old['distances']['right_outside_distance']),
        ('mean_compensation','Previous compensation [F]',geometry['task_inside_distance'],geometry['task_outside_distance'])]:
        path=f'results/{folder}/evaluation/results.json'
        geometry_rows.append([label,f'{di:.5f}',f'{do:.5f}']+[f'{acc(pick(path,preset="released",stage=s,dataset="MNIST")):.2f}' for s in ['standalone','direct','AGA_both']])
    if pairing_complete:
        measured=read(str(pairing_root/'construction/matrices.json'))
        for variant,label in [('distance_only','Distance-only pairing'),('task_aware','Task-directed pairing')]:
            rows=[r for r in measured if r['variant']==variant];assert len(rows)==74
            di,do=[math.sqrt(sum(r[k] for r in rows)) for k in ['inside_squared','outside_squared']]
            values=[f'{acc(pick(str(pairing_path),defense="TaskDirected_MNIST",merger=m,attack_scope=s,dataset="MNIST")):.2f}' for m,s in [('Standalone','None'),('TA','None'),('TA','All experts')]] if variant=='task_aware' else ['--']*3
            if variant=='distance_only' and distance_complete:
                values=[f'{acc(pick(str(distance_path),defense="DistanceOnly_MNIST",merger=m,attack_scope=s,dataset="MNIST")):.2f}' for m,s in [('Standalone','None'),('TA','None'),('TA','All experts')]]
            geometry_rows.append([label,f'{di:.5f}',f'{do:.5f}']+values)
    table('geometry','Task-direction changes and actual classification effects',
        ['Fixed model','Task-inside distance','Task-outside distance','Standalone M','TA M','AGA both M'],
        geometry_rows,
        'Same 74 matrices and original expert-minus-base rank 32 right singular directions. Squared per-matrix relative norms are summed. These matrix-only distances exclude biases/LN/embeddings and are not the full-model distance. Comparable task-outside distances do not imply comparable protection. [K] Known symmetry; [F] failed efficacy. Distance-only pairing is a fixed control,not a replacement selected by accuracy; -- means no classification run. The task-directed release is selected before images; no novelty claim for standard symmetry/SVD/greedy alone.')
    if distance_complete:
        check=read(str(distance_path.with_name('independent_validation.json')));assert check['status']=='passed'
        assert read(str(distance_root/'native/independent_validation.json'))['passed']
        rows=[]
        for path,name,label in [(str(pairing_path),'TaskDirected_MNIST','Task-directed'),(str(distance_path),'DistanceOnly_MNIST','Distance-only')]:
            for scope in scopes:
                for task in ['MNIST','SVHN']:
                    a=pick(path,defense='Unprotected',merger='TA',attack_scope=scope,dataset=task)
                    b=pick(main,target='MNIST',partner='SVHN',defense='Unprotected',merger='TA',attack_scope=scope,dataset=task)
                    assert (a['correct'],a['n'])==(b['correct'],b['n'])
            rows.append([label,f'{acc(pick(path,defense=name,merger="Standalone",attack_scope="None",dataset="MNIST")):.2f}']+
                [pair(path,defense=name,merger='TA',attack_scope=scope) for scope in scopes])
        table('distance_ablation','Fixed ablation of the task-direction objective',
            ['Construction','Standalone M','TA M / S','AGA target M / S','AGA both M / S'],rows,
            'Same target, base, rank, 74 matrices, greedy rule, TA0.5 and original AGA. The distance-only order was fixed before task-directed images; its classification run was registered after the main MNIST result. All scores are retained without selecting a replacement. This isolates the objective in this fixed family; it is not a seed study or universal security claim.')
    comp=read('results/mean_compensation/components.json');rows=[]
    labels={'task_inside':'Task-inside matrices only','task_outside':'Task-outside matrices only','all_matrices':'All changed matrices','other_parameters':'Other parameters only','complete':'Complete joint change'}
    for r in comp['rows']:rows.append([labels[r['name']],f'{r["correct"]}/{r["n"]}',str(r['metric']['changed']),f'{r["metric"]["worst_task_drift"]:.6g}'])
    table('ablation','Previous Defense 2 candidate: source-only ablations',
        ['Counterfactual','Correct','Prediction changes','Max. relative gap change'],rows,
        'The fixed512 target-training source images, not a test table. Matrix inside/outside components use the same original rank32 task basis; non-matrix changes are separate. Each counterfactual is quantized separately. Complete-release replay and independent NumPy recount both have zero error. No candidate was repaired or selected from these counterfactual outcomes.')
    coverage=[['TA + original AGA','Completed: two-expert pairs; target/both recovery scopes; protected and partner accuracy.'],
        ['Defense 1','MNIST, DTD, SVHN completed; targeted cleanup and fixed-budget hard-label recovery completed with separate permissions/numerics.'],
        ['Defense 2','Effective fixed data-free structural scheme verified on MNIST/DTD/SVHN. Task-SVD-guided construction and new method contribution remain under study. Later failed candidates and historical O/B/A/B task-bound failures remain distinct.'],
        ['PaRaMS','Full-default protection completed for MNIST/DTD/SVHN, including matched TA and original AGA. Attention-only ablation completed for MNIST; diagonal recovery is a native adaptation.'],
        ['MergeLock','Not run: official repository currently contains only README/Coming Soon. No invented implementation or paper scores.'],
        ['MergeGuard / ARF','Full defenses not run; training-stage conditions differ. Unmask formula was tested separately; GradErase not reproduced.'],
        ['MergeBarrier','Full defense not run; architecture adaptation and verified implementation missing.'],
        ['Eight-task benchmark','Eight experts participated; protected/evaluated targets cover MNIST/DTD/SVHN. Complete protected cases and evaluation for SUN397/Cars/RESISC45/EuroSAT/GTSRB are missing.'],
        ['Other mergers','TIES, three DARE masks, NUFILT and ProDistill completed in documented scopes. Full TSV-Merge, CAT and LOT not run.'],
        ['Reliability','Per-class/pair counts and source audits retained. Only DARE has the stated three-mask comparison; do not present deterministic rows as multi-seed results.']]
    if pairing_complete:coverage[2][1]='Fixed structural scheme verified on three protected tasks. A task-SVD-directed construction now has a complete MNIST native/TA/originalAGA check and a parameter-only distance ablation. New construction cross-task evidence,novelty and known-family recovery remain separate.'
    if pairing_other_complete:coverage[2][1]='Fixed structure and the frozen task-SVD-directed construction verified on MNIST/DTD/SVHN. Parameter-only distance ablation retained. Method contribution and known-family recovery remain separate; not a complete eight-task benchmark.'
    table('coverage','Completed work and remaining comparisons',['Item','Status'],coverage,
        'This is a completion inventory, not an accuracy table. Missing methods are not assigned zero accuracy. More merger names do not increase the number of recovery attacks.',r'>{\raggedright\arraybackslash}p{37mm}>{\raggedright\arraybackslash}p{132mm}')
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(9.5,4.7),sharey=True,layout='constrained')
    shown=['Unprotected','PaRaMS','Defense 1','O/B (historical)','A/B (historical)','Defense 2 (structure)','Previous candidate']
    if pairing_complete:shown.append('Defense 2 (task-directed)')
    for task,ax,title in zip(range(2),axes,['Protected task: MNIST','Partner task: SVHN']):
        for j,(label,color) in enumerate(zip(['Direct TA','AGA target','AGA both'],['#bfd3e6','#6baed6','#2171b5'])):
            values=[float(r[2+j].split(' / ')[task]) for r in tables[0]['rows']]
            ax.barh(np.arange(len(shown))+(j-1)*.23,values,height=.21,label=label,color=color,edgecolor='white',linewidth=.3)
        ax.set_yticks(range(len(shown)),shown);ax.set_xlim(0,100);ax.set_xticks([0,25,50,75,100])
        ax.set_xlabel('Accuracy (%)');ax.set_title(title,loc='left',fontsize=10);ax.grid(axis='x',color='.9',linewidth=.5);ax.set_axisbelow(True)
    axes[0].invert_yaxis();handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,1.07),ncol=3,frameon=False)
    fig.savefig(out/'comparison.pdf',metadata={'CreationDate':None,'ModDate':None},bbox_inches='tight')
    fig.savefig(out/'comparison.png',dpi=220,bbox_inches='tight');plt.close(fig)
    figure=r'\begin{figure}[!htbp]\centering\includegraphics[width=\linewidth]{comparison.pdf}\caption{Two-expert TA0.5 before and after original AGA. Both protected and partner tasks are shown. Every bar is one fixed configuration; historical O/B and A/B task-bound failures remain labelled. Exact numbers appear in Table~\ref{tab:main}.}\label{fig:comparison}\end{figure}'
    (out/'figure.tex').write_text(figure+'\n')
    md=['# 论文实验表汇总','','主实验采用TA融合与原版AGA。以下均为本项目实际结果，不是原论文的数值。**对比尚未全部完成，缺项见最后一表。**',
        '','表格形式参考AGA原文v1第9页Table1：明确保护方法与攻击状态、数据集列、独立未保护控制；使用booktabs横线与简短表注。此处内容、评价范围与原论文不同。',
        '','[AGA原文](https://arxiv.org/pdf/2606.30360v1)。MergeLock仓库于2026-10-21核对仍仅README；未把其论文结果填入我们的表。',
        '','TeX正文为英文，Markdown解释使用中文。SA为单独使用准确率；M/D/S分别为MNIST/DTD/SVHN；T/P为该行目标/伙伴。所有准确率单位为百分比。','']
    tex=['% Generated from frozen, previously audited project records. Requires booktabs,array,graphicx.','% Accuracy in percent. No new model evaluation or raw-score audit.']
    chinese={
        'task_directed_recovery':('任务方向防御二：已知修改方式的针对性恢复','每格为目标 / 伙伴。两种设置都只给发布权重与公开基座；针对性匹配另知道全网残差坐标换序和CLIP结构，不给原专家、真实任务向量、秘密排列或图片。三个恢复模型及无保护控制均与原专家全部158张量精确相同，因此复用已核原分数，不称新图片前向。原版AGA与该针对性恢复分别报告，不外推所有防御。'),
        'three_tasks':('主对比：三个保护任务、相同TA与原版AGA','每格为该行受保护任务 / 伙伴准确率。PaRaMS保留官方完整默认设置；固定结构与任务方向构造分列，后者使用同一有限目标和各自原任务SVD。三保护任务对应两个无序任务组合，不冒充完整八任务表。方法贡献、已知结构和已知位置等针对性恢复单独报告。参数改动尺度和数值协议分列，不通过裁剪基线追求同参数距离。'),
        'main':('主实验：双专家TA融合与原版AGA','每格为受保护MNIST / 伙伴SVHN。H表示旧O/B、A/B不符合原任务间隔限制；K为已知全网置换对照；F为先前失败候选。完整PaRaMS作为完整基线保留，不因参数改动较大而排除。'),
        'defense1':('Defense 1：三个保护任务','每格为该行目标 / 伙伴，均为TA0.5。原版AGA不修改最终LayerNorm偏置；这些低分不等于能抵抗所有恢复。'),
        'recovery':('恢复权限对比','不同恢复拥有的信息与数据不同，不能混称原版AGA。蒸馏使用2048张同分布训练图片和学生训练；两个教师的查询标签相同，因此只训练了一个问题。偏置清理知道修改位置；原生FP32前后分数差超限的旧记录保留。'),
        'mergers':('补充：更换融合方法','均先执行原版AGA仅处理保护专家，每格MNIST / SVHN，各融合器各用自己的无保护控制。DARE为三个掩码种子的均值；NUFILT两种顺序分开；ProDistill另需每任务64张无标签训练校准图片和攻击方融合系数训练。融合方法不算新增恢复攻击。'),
        'dare':('DARE：三个固定掩码的波动','均值及样本标准差来自42、43、44三个掩码，不能称三个独立专家训练或三次防御训练。'),
        'other_tasks':('补充：DTD、SVHN保护专家的融合比较','每格为该行目标 / MNIST伙伴。NUFILT保留两种顺序；SVHN无恢复、双方AGA的单元复用此前相同融合模型并交换任务列，不重复训练。ProDistill需要每任务64张无标签校准图和融合系数训练，不能算新增无数据恢复攻击。'),
        'eight':('八专家参与、三个数据集评价','TA0.2，每格固定MNIST / DTD / SVHN顺序。其余五个数据集未评价，不能写完整八任务表或八任务均值；与双专家TA0.5分开。'),
        'unmask':('Unmask：独立的八专家恢复设置','按论文公式，从融合参数减去0.3倍整份发布专家；未保护控制也执行同样操作，不与AGA组合。这里是公式适配而非官方完整复现。未保护模型也会崩溃，不能只凭绝对低分宣布防御成功或知识恢复。'),
        'cleanup':('针对偏置的恢复：数值协议分列','每格目标 / 伙伴。MNIST采用原生FP32；DTD、SVHN采用预先固定的FP32前缀、FP64最终读出。两类不可混表。原生DTD数值控制失败与MNIST分数差超限都保留，不声称完整FP32函数严格相同。'),
        'geometry':('任务方向内外变化与真实融合效果','相同74个矩阵、原专家减基座的32个任务右奇异方向；矩阵距离不含偏置、LayerNorm与embedding。任务外距离相近，但融合效果差别很大，不能把距离本身当作保护效果。Distance-only pairing为固定消融，不按分类成绩择优；未实测项留空。'),
        'distance_ablation':('固定消融：去除任务方向项','同一目标、基座、74矩阵、rank32及贪心规则，仅比较原构造目标。Distance-only顺序在主方法读图前已固定，分类消融在MNIST主结果后追加登记；不是所有消融均在主实验前登记。全部成绩保留，不按结果换发布；不把一个固定点当作所有排序的结论。'),
        'ablation':('最新Defense 2候选：任务内外与联合补偿消融','固定512张构造训练图，不是测试结果。仅矩阵任务内、仅任务外、全部矩阵、其他参数与完整联合共五项。完整联合保持原能力，但真实AGA后融合仍接近99%；没有根据消融改参数。'),
        'coverage':('已完成与尚缺的实验','这是一份完成情况清单，不能说所有对比已做完；缺项不填零分，也不拿别人的论文成绩冒充我们的实测。')}
    translations={'Unprotected':'无保护','Defense 1':'防御一','Global permutation [K]':'全网置换 [K]','Defense 2 candidate (failed) [F]':'先前失败候选 [F]',
        'Defense 2 (structure) [K]':'防御二（固定结构）[K]','Defense 2 (task-directed)':'防御二（任务方向构造）',
        'Without family matching':'不作针对性匹配','Known-family matching':'已知方式匹配'}
    method_sections=[]
    for i,t in enumerate(tables):
        md_start,tex_start=len(md),len(tex)
        title,note=chinese[t['key']]
        md+=['## '+str(i+1)+'. '+title,'','| '+' | '.join(t['headers'])+' |','|'+'---|'*len(t['headers'])]
        md+=['| '+' | '.join(translations.get(x,x) for x in row)+' |' for row in t['rows']];md+=['',note,'']
        def escape(s):return s.replace('→',r'$\to$').replace('&',r'\&').replace('%',r'\%').replace('_',r'\_').replace('#',r'\#').replace(' +/- ',r' $\pm$ ')
        tex+=[r'\begin{table}[!htbp]',r'\centering',r'\small',r'\caption{'+escape(t['title'])+'}',r'\label{tab:'+t['key']+'}',r'\begin{tabular}{'+t['colspec']+'}',r'\toprule',' & '.join(escape(x) for x in t['headers'])+r' \\',r'\midrule']
        tex+=[' & '.join(escape(x) for x in row)+r' \\' for row in t['rows']]
        tex += [r'\bottomrule',r'\end{tabular}',r'\par\vspace{3pt}\begin{minipage}{0.98\linewidth}\footnotesize '+escape(t['note'])+r'\end{minipage}',r'\end{table}','']
        if t['key'] in ['main','geometry','distance_ablation']:method_sections.append(dict(key=t['key'],md=md[md_start:],tex=tex[tex_start:],table=t))
    md+=['## 图1. 同一组模型的直接融合与AGA后融合','','![主实验准确率](comparison.png)','','目标任务和伙伴任务分别显示；每个柱来自固定配置，未添加不存在的随机重复误差条。详细数值见表1。','']
    (out/'tables.md').write_text('\n'.join(md)+'\n');(out/'tables.tex').write_text('\n'.join(tex)+'\n')
    (out/'table_data.json').write_text(json.dumps(tables,ensure_ascii=False,indent=2)+'\n')
    if pairing_complete:
        intro=['# 任务方向参与构造的Defense 2：MNIST实测','',
            '任务SVD直接进入有限目标F=任务外平方距离−任务内平方距离；从恒等开始逐次选择最大正增量的互不重叠坐标对，全网LN、输入、矩阵及读出配套修改。只读原专家与公开基座，不用图片训练专家。',
            '原固定反转模型保持冻结。新发布只有一份；距离单独最大化是参数消融，不按分类成绩选择模型。此表新构造只验证MNIST，不能冒充新方法的三任务结果。已知全网排列及SVD、贪心工具本身不是新的贡献，针对已知家族的公开恢复边界仍保留。','']
        if pairing_other_complete:intro[3]='原固定反转模型保持冻结。每个目标仅一份任务方向发布；距离单独最大化作为固定消融，不按分类成绩选择。MNIST结果见本表；同一构造的DTD／SVHN及三任务表见[跨任务结果](other_tasks/summary.md)。已知全网排列、SVD及贪心工具本身不构成新贡献，已知家族下公开恢复的边界单列。'
        (pairing_root/'summary.md').write_text('\n'.join(intro+sum([s['md'] for s in method_sections],[]))+'\n')
        (pairing_root/'tables.tex').write_text('\n'.join(sum([s['tex'] for s in method_sections],[]))+'\n')
        (pairing_root/'table_provenance.json').write_text(json.dumps(dict(source_files_sha256=sources,
            report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),tables=[s['table'] for s in method_sections],new_image_forwards=0),indent=2)+'\n')
        if distance_complete:
            section=next(s for s in method_sections if s['key']=='distance_ablation')
            (distance_root/'summary.md').write_text('\n'.join(['# 固定距离目标的分类消融','','原distance-only顺序在主方法读图前已保存；分类运行在主结果后追加登记。它是消融控制，不是按成绩替换的新防御。','']+section['md'])+'\n')
            (distance_root/'tables.tex').write_text('\n'.join(section['tex'])+'\n')
    (out/'README.md').write_text('''# 实验表与完成情况

**没有全部做完。** 已完成的真实结果已经生成同一份TeX和Markdown；缺项明确保留。

- [Markdown总表](tables.md)：主实验、三个保护任务、不同恢复权限、融合方法、八专家三数据集、任务内外与补偿消融、缺项清单。
- [TeX表格](tables.tex)：可取对应table段落放入论文，需要booktabs、array；宽表放在两栏论文时改为table*。
- [完整TeX预览入口](main.tex)与[PDF预览](main.pdf)。图的[TeX片段](figure.tex)、[PDF](comparison.pdf)和[PNG](comparison.png)同数据生成。
- [数值与来源](table_data.json)、[输入哈希与原记录行号](table_sources.json)。新制表没有重跑GPU，也没有冒充新一轮原始分数复核。

已完成：三个保护任务的相同TA／原版AGA下无保护、Defense1、完整默认PaRaMS和有效Defense2结构方案对比；对角适配、位置已知偏置清理、MNIST固定数据蒸馏、TIES/DARE/NUFILT/ProDistill、八专家参与的三任务评价及Unmask公式适配。

尚缺：Defense2任务方向参与构造及新方法贡献的充分验证；独立DTD/SVHN组合；完整八个保护任务和八数据集成绩；MergeLock、完整MergeGuard/ARF/MergeBarrier防御基线；GradErase；更多独立防御重复。MergeLock官方库此前核对仅README，其他未运行方法不填论文成绩。不同训练阶段/架构/权限须另列。完整TSV-Merge、CAT、LOT也未运行，不把融合器数量当攻击数量。

历史完整分表保留，不用本次摘要替代：

- [TA/TIES完整双任务表](../phase3/tables.md)
- [PaRaMS对角恢复](../phase3/params_adaptive/tables.md)
- [DARE三个掩码](../phase3/dare/tables.md)
- [NUFILT O/B补充](../phase3/nufilt_OB/tables.md)
- [NUFILT DTD/SVHN](../phase3/nufilt_other_tasks/tables.md)
- [ProDistill MNIST](../phase3/prodistill/tables.md)及[DTD/SVHN](../phase3/prodistill_other_tasks/tables.md)
- [公开基座/单专家知识来源对照](../phase3/controls/single/tables.md)及[固定七专家去目标对照](../phase3/controls/leave_out/tables.md)
- [位置已知清理](../phase3/bias_canonicalization/tables.md)、[单列读出精度协议](../phase3/bias_tail_precision/tables.md)、[固定数据蒸馏](../phase3/distillation/tables.md)
- [八专家三数据集](../phase3/eight_task/tables.md)、[Unmask](../phase3/unmask/tables.md)
- [最新Defense2失败闭环](../mean_compensation/summary.md)、[已知全网置换对照](../residual_coordinate_control/summary.md)

重建表和图：`python experiments/phase3_report.py --paper-tables --output results/paper_tables`。本机系统Matplotlib与用户NumPy版本不同，当前使用`PYTHONNOUSERSITE=1`调用匹配的系统环境；没有修改3090科研环境。main.tex可由桌面LaTeX编辑器编译。
''')
    if pairing_complete:
        p=out/'README.md';p.write_text(p.read_text().replace('尚缺：Defense2任务方向参与构造及新方法贡献的充分验证；','已完成任务SVD参与构造的MNIST闭环；尚缺新构造的跨任务及新方法贡献验证；'))
    if pairing_other_complete:
        p=out/'README.md';p.write_text(p.read_text().replace('已完成任务SVD参与构造的MNIST闭环；尚缺新构造的跨任务及新方法贡献验证；','已完成同一任务SVD构造的MNIST／DTD／SVHN验证；尚缺新方法贡献的充分验证；'))
    manifest=dict(status='completed_table_collection',report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=sources,selected_rows=used,tables=len(tables),new_GPU_forwards=0,new_raw_score_revalidation=False,
        scope='Checks imported integer/class totals and joins; reuses prior independent raw-score audits. Not a new attack/defense experiment.',
        style_reference=dict(paper='https://arxiv.org/pdf/2606.30360v1',page=9,table=1,visually_checked=True),
        output_sha256={p:hashlib.sha256((out/p).read_bytes()).hexdigest() for p in ['README.md','tables.md','tables.tex','table_data.json','figure.tex','comparison.pdf','comparison.png']})
    (out/'table_sources.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print('PAPER_TABLES_WRITTEN',len(tables),'source selections',len(used))


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
    scopes=cfg.get('scopes',['None','Protected expert']);reference_rows=[]
    if 'no_attack_reference' in cfg:
        for ref in cfg['no_attack_reference'].values():assert digest(ref['path'])==ref['sha256']
        reference_rows=json.loads(Path(cfg['no_attack_reference']['rows']['path']).read_text())
        audit=json.loads(Path(cfg['no_attack_reference']['validation']['path']).read_text())
        assert audit['status']=='passed_independent_numpy_metrics_and_frozen_artifact_hashes'
        assert audit['model_parameter_hashes']==json.loads(Path(cfg['no_attack_reference']['parameters']['path']).read_text())
    expected=sum(len(scopes)*(1+len(c['defenses']))*len(cfg['evaluate']) for c in cfg['cases'])
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
        return next(r for r in rows+reference_rows if (r['target'],r['defense'],r['attack_scope'],r['dataset'])==(target,name,scope,task))
    for r in rows:
        t,n,s,k=r['target'],r['defense'],r['attack_scope'],r['dataset']
        r['defense_drop_without_attack_pp']=get(t,'Unprotected','None',k)['accuracy_percent']-get(t,n,'None',k)['accuracy_percent']
        r['attack_recovery_pp']=r['accuracy_percent']-get(t,n,'None',k)['accuracy_percent']
        r['remaining_gap_pp']=get(t,'Unprotected',s,k)['accuracy_percent']-r['accuracy_percent']
    comparisons=sum(r['defense']!='Unprotected' for r in rows)
    for r in rows:r['bonferroni_log10_p']=min(0.,r['mcnemar_log10_p']+math.log10(comparisons))
    labels={'Unprotected':'Unprotected','O/B':'O/B†','A/B':'A/B†','PaRaMS':'PaRaMS (full)*','PaRaMS_Attention':'PaRaMS (attention)'}
    lines=['# Eight participating experts; three evaluated tasks','','Original author task order, TA0.2, and AGA applied to the protected expert. Accuracy (%); no eight-task average. The other five partner datasets were not evaluated. This setting changes both coefficient and participant count relative to the two-expert TA0.5 table.','']
    if 'Unmask' in scopes:
        lines=['# Paper-defined Unmask recovery: eight experts, three evaluated tasks','',
            'Fixed MergeGuard section7 operator: merged parameters minus0.3 times the whole published expert. TA0.2, one known protected expert; no AGA composition. Native CLIP paper-formula adaptation, not official execution or complete Table5 replication. Accuracy (%); no eight-task average.','']
    shown_scopes=['None']+[s for s in scopes if s!='None']
    scope_labels={'None':'No attack','Protected expert':'AGA (protected expert)','Unmask':'Unmask (0.3)'}
    headers=[scope_labels[v] for v in shown_scopes];latex=[]
    for case in cfg['cases']:
        target=case['target'];tasks=[target]+[t for t in cfg['evaluate'] if t!=target]
        lines += ['## Protected task: '+target,'','Cells: '+' / '.join(tasks)+' (target / evaluated partners).','',
            '| Defense | '+' | '.join(headers)+' |','|---|'+'---:|'*len(headers)]
        latex += ['% Protected task: '+target+'; cells: '+' / '.join(tasks),r'\begin{tabular}{lrr}',r'\toprule','Defense & '+' & '.join(headers)+r' \\',r'\midrule']
        for name in ['Unprotected']+case['defenses']:
            label=labels.get(name,'Defense 1*')
            cells=[' / '.join(f'{get(target,name,s,t)["accuracy_percent"]:.2f}' for t in tasks) for s in shown_scopes]
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
        report_source_sha256=digest(__file__),reference_files=cfg.get('no_attack_reference'),audit_and_table_seconds=time.monotonic()-start,new_GPU_forwards=0)
    (output/'independent_validation.json').write_text(json.dumps(result,indent=2)+'\n')


def nufilt_tasks(source, output):
    """Recount missing task cells; finish only an interrupted score replay, never refit."""
    import copy
    import time
    import torch
    from clip_benchmark import digest, evaluate
    from covariance_noise import dataset
    from phase3_evaluate import setup
    from phase3_nufilt_check import load_official
    from subspace_probe import make_loader, save_json
    from verify_covariance_noise import official_labels
    from verify_prodistill import verify_metrics
    source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=True)
    spec=json.loads(Path('configs/phase3_nufilt_other_tasks.json').read_text())
    parents=[]
    for ref in spec['frozen_inputs'].values():
        assert digest(ref['path'])==ref['sha256'];parents.append(json.loads(Path(ref['path']).read_text()))
    cfg=parents[0]
    for key in ['tasks','AGA_references','defenses']:cfg[key].update(parents[1][key])
    _,author=load_official();pins={};rows_out=[];checks=[];maximum=0.;count=0;new_replays=0
    for i,case in enumerate(spec['cases']):
        root=source/f'{i:02d}';protocol=json.loads((root/'protocol.json').read_text())
        assert protocol['parent_protocol_sha256']==digest('configs/phase3_nufilt_other_tasks.json')
        assert protocol['author_source_sha256']==author
        for path,sha in protocol['source_sha256'].items():assert digest(path)==sha
        pilot=json.loads((root/'pilot/execution.json').read_text())
        for c in pilot['costs']:
            key=c['key'].replace('Phase1_'+case['target'],'Unprotected');p=root/(key+'.pt')
            assert digest(p)==c['artifact_sha256'];pins[str(p)]=c['artifact_sha256']
        for stage in ['pilot','full']:
            p=root/stage;raw=torch.load(p/'scores.pt',map_location='cpu',weights_only=False)
            rows=json.loads((p/'results.json').read_text());assert len(rows)==len(protocol['orders'])*len(protocol['scopes'])*4
            raw_sha=digest(p/'scores.pt');pins[str(p/'scores.pt')]=raw_sha
            missing=not (p/'validation.json').exists();replays=0;error=0.;start=time.monotonic()
            if missing:
                assert stage=='full' and json.loads((root/'pilot/validation.json').read_text())['status']=='passed'
                anchor,experts,heads,_,_=setup(cfg)
                data={t:dataset(t,'data',experts[t].val_preprocess) for t in [case['target'],case['partner']]}
            for row in rows:
                order=protocol['orders'].index(row['order']);scope=protocol['scopes'].index(row['attack_scope'])
                prefix=f'{order}_{scope}_';task=row['dataset'];x=raw[prefix+row['defense']+'/'+task]
                before=raw[prefix+'Unprotected/'+task]['logits'];labels=official_labels(task)
                ids=list(range(len(labels))) if stage=='full' else np.linspace(0,len(labels)-1,64,dtype=int).tolist()
                assert x['indices']==ids;np.testing.assert_array_equal(x['target'].numpy(),labels[ids])
                maximum=max(maximum,verify_metrics(x['logits'],x['target'],before,row));count+=1
                if row['defense']!='Unprotected':torch.testing.assert_close(x['logits'],before,rtol=0,atol=0)
                if missing:
                    name=prefix+'Unprotected';record=torch.load(root/(name+'.pt'),map_location='cpu',weights_only=False)
                    model=copy.deepcopy(anchor);model.load_state_dict(anchor.state_dict()|record['weights'],strict=True)
                    selected=np.linspace(0,len(ids)-1,16,dtype=int)
                    for block in sorted(set(selected//128)):
                        lo,hi=block*128,min((block+1)*128,len(ids));wanted=selected[(selected>=lo)&(selected<hi)];sink=[]
                        evaluate(model,heads[task],make_loader(data[task],ids[lo:hi]),'cuda',logit_sink=sink.append)
                        error=max(error,float((torch.cat(sink)[wanted-lo]-x['logits'][wanted]).abs().max()));replays+=len(wanted)
                    del model
                    assert error<=protocol['score_tolerance']
                if stage=='full':
                    y=x['target'].numpy();ok=x['logits'].numpy().argmax(1)==y;was=before.numpy().argmax(1)==y
                    for c in row['per_class']:
                        mask=y==c['class_id'];c.update(lost=int((mask&was&~ok).sum()),gained=int((mask&~was&ok).sum()))
                    rows_out.append(row)
            if missing:
                v=dict(status='passed',rows=len(rows),full_replay_count=replays,maximum_score_error=error,
                    raw_sha256=raw_sha,resumed_after_interruption=True,replay_seconds=time.monotonic()-start,
                    original_complete_wall_seconds=None,verification_source_sha256=digest(__file__),new_merger_updates=0)
                save_json(p/'validation.json',v);new_replays+=replays
            v=json.loads((p/'validation.json').read_text());assert v['status']=='passed'
            if (p/'execution.json').exists():assert json.loads((p/'execution.json').read_text())['scores_sha256']==raw_sha
            else:assert v['resumed_after_interruption'] and v['raw_sha256']==raw_sha
            checks.append(dict(case=i,stage=stage,validation=v,pilot_execution=pilot))
    lines=['# Missing NUFILT protected-task cells','',
        'Accuracy (%), protected task / partner. Every Phase1 result has the exact same predictions and scores as its matched unprotected merger; selected-input identity was checked before reuse. Official kernels, native adapter; no calibration images. These are merger evaluations, not new recovery attacks.','',
        '| Protected task / partner | AGA scope | Order | Unprotected = Defense1 |','|---|---|---|---:|']
    for row in rows_out:
        if row['defense']!='Unprotected' or row['dataset']!=row['target']:continue
        peer=next(r for r in rows_out if r['defense']=='Unprotected' and r['order']==row['order'] and r['target']==row['target'] and r['attack_scope']==row['attack_scope'] and r['dataset']==row['partner'])
        lines.append(f'| {row["target"]} / {row["partner"]} | {row["attack_scope"]} | {" → ".join(row["order"])} | {row["accuracy_percent"]:.2f} / {peer["accuracy_percent"]:.2f} |')
    lines+=['','SVHN/MNIST without AGA and with all-expert AGA reuse earlier identical-model results after target/partner relabeling; they were not retrained here. DTD full scores survived interruption: only the unfinished original-context replay was completed; its original total wall time is unavailable. Frozen weights, thresholds and raw scores were unchanged.']
    save_json(output/'results.json',rows_out);(output/'tables.md').write_text('\n'.join(lines)+'\n')
    save_json(output/'independent_validation.json',dict(status='passed_independent_numpy_metrics_and_artifact_hashes',
        rows_checked=count,maximum_logp_error=maximum,raw_and_weights_sha256=pins,stages=checks,
        resumed_score_replays=new_replays,expert_updates=0,defense_updates=0,report_source_sha256=digest(__file__)))


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


def distillation(source, output):
    """Audit the single frozen hard-label attack; no student or defense fitting."""
    import time
    import torch
    from scipy.special import logsumexp
    from clip_benchmark import digest, evaluate
    from covariance_noise import dataset
    from phase3_eight_task import parameter_hash
    from phase3_evaluate import apply, setup
    from subspace_probe import make_loader, save_json
    from verify_covariance_noise import official_labels
    from verify_prodistill import verify_metrics
    start=time.monotonic();source,output=Path(source),Path(output);output.mkdir(parents=True,exist_ok=False)
    read=lambda name:json.loads((source/name).read_text())
    cfg,freeze,validation,queries=map(read,['protocol.json','freeze.json','validation.json','queries.json'])
    assert validation['status']=='passed' and freeze['all_students_frozen_before_test']
    assert digest(source/'scores.pt')==validation['raw_sha256'] and digest(source/'teacher_queries.pt')==queries['raw_sha256']
    assert digest(source/'training_indices.json')==queries['indices_sha256']
    for p,sha in cfg['source_sha256'].items():assert digest(p)==sha,p
    ids=read('training_indices.json');expected=torch.randperm(60000,generator=torch.Generator().manual_seed(cfg['data_seed']))[:cfg['unlabeled_n']].tolist()
    assert ids==expected and len(set(ids))==2048
    queried=torch.load(source/'teacher_queries.pt',map_location='cpu',weights_only=False)
    assert queried['indices']==ids
    for name,scores in queried['scores'].items():assert torch.equal(scores.argmax(1),queried['labels'][name]) and torch.isfinite(scores).all()
    assert int((queried['labels']['Unprotected']!=queried['labels'][cfg['defense']]).sum())==queries['teacher_prediction_disagreements']
    for group,names in freeze['groups'].items():
        t=freeze['training'][group];state=torch.load(source/(group+'_student.pt'),map_location='cpu',weights_only=False)
        assert digest(source/(group+'_student.pt'))==t['model_sha256'] and parameter_hash(state)==t['parameter_sha256']
        assert t['completed'] and t['backward_steps']==800 and len(t['history'])==50
        assert t['expert_updates']==t['defense_updates']==t['head_updates']==0
        assert hashlib.sha256(queried['labels'][group].numpy().tobytes()).hexdigest()==t['pseudo_labels_sha256']
        assert all(torch.equal(queried['labels'][group],queried['labels'][n]) for n in names)
    raw=torch.load(source/'scores.pt',map_location='cpu',weights_only=False);rows=read('results.json');maximum=0.
    for row in rows:
        key='/'.join([row['teacher_group'],row['mode'],row['dataset']]);x=raw[key];baseline=raw['Unprotected/'+row['mode']+'/'+row['dataset']]['logits']
        np.testing.assert_array_equal(x['target'].numpy(),official_labels(row['dataset']));assert x['indices']==list(range(len(x['target'])))
        maximum=max(maximum,verify_metrics(x['logits'],x['target'],baseline,row))
        s=x['logits'].double().numpy();y=x['target'].numpy();ce=logsumexp(s,axis=1)-s[np.arange(len(y)),y]
        before=baseline.numpy().argmax(1);after=s.argmax(1)
        for c in row['per_class']:
            mask=y==c['class_id'];assert abs(c['mean_cross_entropy']-ce[mask].mean())<1e-10
            assert c['lost']==int((mask&(before==y)&(after!=y)).sum()) and c['gained']==int((mask&(before!=y)&(after==y)).sum())
    # Independent teacher-query contexts through the existing evaluation entry point.
    parent=json.loads(Path(cfg['frozen_inputs']['path']).read_text());assert digest(cfg['frozen_inputs']['path'])==cfg['frozen_inputs']['sha256']
    parent['source_sha256']=cfg['source_sha256'];_,originals,heads,_,_=setup(parent)
    teachers={'Unprotected':originals[cfg['target']],cfg['defense']:apply(originals[cfg['target']],parent['defenses'][cfg['defense']]['release'])}
    data=dataset(cfg['target'],'data',originals[cfg['target']].val_preprocess,training=True);error=0.;query_replays=0
    for name,model in teachers.items():
        assert parameter_hash(model.state_dict())==queries['teacher_parameter_sha256'][name]
        for block in [0,5,10,15]:
            lo=128*block;hi=lo+128;sink=[]
            evaluate(model,heads[cfg['target']],make_loader(data,ids[lo:hi]),'cuda',logit_sink=sink.append)
            e=float((torch.cat(sink)-queried['scores'][name][lo:hi]).abs().max());error=max(error,e);assert e<=cfg['score_tolerance'];query_replays+=128
    ref={'path':'outputs/phase3/full/00_MNIST_SVHN/artifacts/predictions.pt',
        'sha256':'b7afe0067a24273e6da9d759221bc863bf429435c15842b4a03407220af55dc9'}
    assert digest(ref['path'])==ref['sha256'];native=torch.load(ref['path'],map_location='cpu',weights_only=False)
    original_rows=[]
    for name in ['Unprotected',cfg['defense']]:
        cells=[]
        for method,scope,tasks in [('Standalone','None',[cfg['target']]),('TA','None',[cfg['target'],cfg['partner']]),
                ('TA','Protected expert',[cfg['target'],cfg['partner']]),('TA','All experts',[cfg['target'],cfg['partner']])]:
            values=[]
            for task in tasks:
                x=native['/'.join([name,method,scope,task])];np.testing.assert_array_equal(x['target'].numpy(),official_labels(task))
                values.append(100*float((x['logits'].argmax(1)==x['target']).double().mean()))
            cells.append(' / '.join(f'{v:.2f}' for v in values))
        original_rows.append('| Native '+name+' | '+' | '.join(cells)+' |')
    columns=['Standalone','TA','AGA-target-only','AGA-both'];lines=['# Fixed-budget hard-label recovery','',
        'Native adaptation, not full PaRaMS KD reproduction. MNIST/SVHN; fixed2048 unlabeled training images,50epochs,batch128. '
        'Only an attacker-owned public-base visual encoder is trained. Historical tests; no defense changes or test selection.','',
        '| Student group / teacher | Standalone MNIST | TA MNIST / SVHN | AGA-target MNIST / SVHN | AGA-both MNIST / SVHN |',
        '|---|---:|---:|---:|---:|']+original_rows
    for group,names in freeze['groups'].items():
        cells=[]
        for mode in columns:
            selected=[r for r in rows if r['teacher_group']==group and r['mode']==mode]
            cells.append(' / '.join(f'{next(r for r in selected if r["dataset"]==t)["accuracy_percent"]:.2f}' for t in ([cfg['target']] if mode=='Standalone' else [cfg['target'],cfg['partner']])))
        lines.append('| '+' + '.join(names)+' | '+' | '.join(cells)+' |')
    lines+=['',f'- Teacher disagreements on the fixed query set: {queries["teacher_prediction_disagreements"]}. Exact matching labels share one training problem; this is not independent seed replication.',
        '- The matched unprotected-teacher student is the attack control. Native unprotected TA/AGA and public-base controls remain separate; high accuracy is not proof of unique-knowledge theft.',
        f'- Student training: {sum(t["training_seconds"] for t in freeze["training"].values()):.2f}s, {sum(t["backward_steps"] for t in freeze["training"].values())} full steps; discarded one-step pilot separately recorded.',
        f'- Full evaluation context score replays: {validation["score_replays"]}; maximum error {validation["maximum_context_score_error"]:.6g}. Teacher-query replay inputs: {query_replays}; maximum error {error:.6g}.','']
    (output/'tables.md').write_text('\n'.join(lines))
    for name in ['protocol.json','queries.json','training.json','training_indices.json','freeze.json','pilot.json','results.json','validation.json']:
        (output/name).write_bytes((source/name).read_bytes())
    save_json(output/'independent_validation.json',dict(status='passed_independent_numpy_and_query_replay',rows_checked=len(rows),
        maximum_independent_logp_error=maximum,teacher_query_replayed_inputs=query_replays,teacher_query_score_error=error,
        native_reference=ref,native_reference_scores_recomputed=False,
        CPU_and_replay_wall_seconds=time.monotonic()-start,source_sha256=digest(__file__),raw_sha256=validation['raw_sha256'],
        new_student_training=0,new_defense_updates=0,independent_full_training_reproduction=False))


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


def joint_task_gate(source, output):
    """A stopped Defense2 task gate; do not manufacture merger/attack rows."""
    import torch
    source, output = Path(source), Path(output)
    cfg = json.loads((source/'protocol.json').read_text())
    root = Path(cfg['release']['path']).parent
    calibration = json.loads((root/'results.json').read_text())
    confirmation = json.loads((root/'confirmation_validation.json').read_text())
    native = json.loads((source/'full_native_independent_validation.json').read_text())
    execution = json.loads((source/'execution.json').read_text())
    assert native['status'] == 'verified_full_native_budget_failure_fixed_candidate_closed'
    assert execution['author_AGA_calls'] == 0 and not execution['completed']
    for filename in ['independent_validation.json', 'confirmation_independent_validation.json']:
        assert json.loads((root/filename).read_text())['status'] == 'passed'
    raw_path = source/'artifacts/predictions.pt'
    pin = hashlib.sha256(raw_path.read_bytes()).hexdigest()
    assert pin == execution['predictions_sha256'] == native['raw_sha256']
    raw = torch.load(raw_path, map_location='cpu')
    before, after = [raw[k]['logits'].double().numpy() for k in ['original/standalone/MNIST', 'released/standalone/MNIST']]
    y = before.argmax(1); gap = before[np.arange(len(y)), y, None]-before
    delta = after-before; gap[np.arange(len(y)), y] = 1.
    rho = (delta[np.arange(len(y)), y, None]-delta)/gap
    rho[np.arange(len(y)), y] = 0.
    violating = (abs(rho)>cfg['task_budget']).any(1)
    assert abs(float(abs(rho).max())-native['worst_task_drift']) < 1e-12
    stages = [
        ('Construction + finite probes', calibration['task_inputs'], calibration['metrics']['joint']['worst_task_drift'], calibration['metrics']['joint']['changed']),
        ('Fresh train + finite probes', confirmation['source']['n'], confirmation['source']['worst_task_drift'], confirmation['source']['changed']),
        ('Complete MNIST test', native['n'], native['worst_task_drift'], native['changed_predictions']),
    ]
    lines = ['# Fixed joint-response direction: task gate', '',
             'One frozen B10/B11 direction, parameter distance '+f'{calibration["parameter_distance"]:.7f}'+'. Original task budget 0.5; no change after confirmation or tests.', '',
             '| Evaluation set | Inputs | Worst relative gap drift | Prediction changes | Task gate |',
             '|---|---:|---:|---:|---|']
    for label, n, drift, changed in stages:
        lines.append(f'| {label} | {n} | {drift:.6g} | {changed} | '+('Pass' if drift<=cfg['task_budget'] else 'Fail')+' |')
    w = native['witness']
    lines += ['', f'Complete-test accuracy is {100*native["original_correct"]/native["n"]:.2f}% before and after protection; all predictions match. {int(violating.sum())} images exceed the functional budget.',
              f'The independently replayed witness (test index {w["index"]}, class {w["teacher_class"]} versus {w["competitor"]}) increases the gap from {w["original_gap"]:.8f} to {w["released_gap"]:.8f}. Increased confidence still violates the original absolute relative-gap constraint.',
              '', '**TA and original AGA were not run:** the preregistered native-task gate stopped this fixed candidate. This is neither a feasible-defense result nor evidence about its attack resistance. No radius, rank, layer, output axis or weights were changed to repair it.', '',
              '## Coupling and limits', '',
              f'On the construction inputs, B10 alone has worst drift {calibration["metrics"]["B10_only"]["worst_task_drift"]:.6g}, B11 alone {calibration["metrics"]["B11_only"]["worst_task_drift"]:.6g}, and their fixed combination {calibration["metrics"]["joint"]["worst_task_drift"]:.6g}. Each has zero construction prediction changes; this does not establish harmful isolated branches.',
              'The minimum empirical joint response and finite task-SVD probes support source compensation, not unseen-input preservation. Standard SVD, Rayleigh optimization and Gauss-Newton geometry are not novelty claims. Historically exposed tests are not a first blind evaluation.', '',
              f'Construction: {calibration["wall_seconds"]:.2f}s; confirmation: {confirmation["wall_seconds"]:.2f}s; stopped full-task evaluation: {execution["wall_seconds"]:.2f}s. Independent audits and their costs remain separate. No optimizer, expert updates or new attack fits.', '']
    output.mkdir(parents=True, exist_ok=True)
    (output/'summary.md').write_text('\n'.join(lines))
    provenance = dict(status='verified_closed_task_budget_failure', raw_sha256=pin, source_release_sha256=cfg['release']['sha256'],
                      n_functional_budget_violations=int(violating.sum()),
                      violations_by_teacher_class=np.bincount(y[violating], minlength=10).tolist(),
                      report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      source_and_confirmation_passed=True, full_task_passed=False, new_AGA_calls=0, new_fusions=0,
                      matched_original_score_error=native['matched_original_score_error'], matched_release_score_error=native['matched_release_score_error'])
    activation_path = root/'activation_response.json'
    if activation_path.exists():
        activation = json.loads(activation_path.read_text())
        assert activation['input_artifact_sha256']['release.pt'] == cfg['release']['sha256']
        assert activation['new_images'] == activation['new_weights'] == activation['new_directions'] == 0
        raw_activation = root/'activation_response.npz'
        assert hashlib.sha256(raw_activation.read_bytes()).hexdigest() == activation['raw_sha256']
        archive = np.load(raw_activation)
        recount_error = 0.
        for row in activation['rows']:
            values = archive[row['name']]
            assert values.shape == (row['input_rows'], 3) and np.isfinite(values).all()
            measured = [np.sqrt(values[:, 0].mean()), np.sqrt(values[:, 1].mean()),
                        values[:, 2].mean(), np.sqrt(values[:, 0].sum()/values[:, 1].sum())]
            fields = ['actual_MLP_delta_rms', 'original_MLP_output_rms',
                      'activation_scalar_mean_square', 'actual_delta_over_original_MLP_rms']
            error = max(abs(x-row[k])/max(abs(row[k]), 1e-30) for k, x in zip(fields, measured))
            assert error < 1e-12
            recount_error = max(recount_error, error)
        lines += ['', '## Parameter distance versus source activation', '',
                  '| Branch / inputs | Relative parameter distance | MLP output change (RMS %) | Activation energy / uniform-complement mean |',
                  '|---|---:|---:|---:|']
        for row in activation['rows']:
            layer = 0 if row['name'].startswith('B10') else 1
            lines.append(f'| {row["name"].replace("_", " ")} | {calibration["per_layer_distance"][layer]:.6f} | {100*row["actual_delta_over_original_MLP_rms"]:.6f} | {row["excitation_over_uniform_reference"]:.6g} |')
        lines += ['', 'All rows use the 4,096 original construction images. B10 includes separate all-token and CLS statistics; B11 uses CLS. MLP output here means the linear Bz contribution before the unchanged bias. The uniform-complement column is an analytic second-moment reference, with no sampled or optimized replacement direction.',
                  f'The joint linear response energy is {activation["linear_response"]["joint_over_separate_sum"]:.6g} times the sum of isolated-branch energies: coupling exists, while the dominant B11 change is weakly excited on these inputs.',
                  'This diagnoses the frozen source solution only. It does not prove the cause of every unseen-input violation or establish attack efficacy. No rewhitening, new weights, threshold changes or additional TA/AGA evaluations follow from this table.', '']
        provenance['activation_audit_sha256'] = hashlib.sha256(activation_path.read_bytes()).hexdigest()
        provenance['activation_raw_sha256'] = activation['raw_sha256']
        provenance['activation_archive_recount_maximum_relative_error'] = recount_error
        (output/'summary.md').write_text('\n'.join(lines))
    base_path = root/'activation_public_base.json'
    if base_path.exists():
        base = json.loads(base_path.read_text())
        assert hashlib.sha256(activation_path.read_bytes()).hexdigest() == base['native_result_sha256']
        assert base['input_artifact_sha256'] == activation['input_artifact_sha256']
        assert base['new_weights'] == base['new_directions'] == base['new_AGA_calls'] == base['new_fusions'] == 0
        raw_base = root/'activation_public_base.npz'
        assert hashlib.sha256(raw_base.read_bytes()).hexdigest() == base['raw_sha256']
        native_rows = {r['name']: r for r in activation['rows']}
        base_error = 0.
        with np.load(raw_base) as base_archive:
            lines += ['', '## Fixed edit under public-base inputs', '',
                      '| Branch / inputs | Native delta RMS | Public-base delta RMS | Base / native delta RMS | Delta / base Bz (RMS %) |',
                      '|---|---:|---:|---:|---:|']
            for row in base['rows']:
                values = base_archive[row['name']]
                assert values.shape == (row['input_rows'], 3) and np.isfinite(values).all()
                measured = [np.sqrt(values[:, 0].mean()), np.sqrt(values[:, 1].mean()),
                            values[:, 2].mean(), np.sqrt(values[:, 0].sum()/values[:, 1].sum())]
                fields = ['actual_MLP_delta_rms', 'original_MLP_output_rms',
                          'activation_scalar_mean_square', 'actual_delta_over_original_MLP_rms']
                ratio = measured[0]/native_rows[row['name']]['actual_MLP_delta_rms']
                errors = [abs(x-row[k])/max(abs(row[k]), 1e-30) for k, x in zip(fields, measured)]
                errors.append(abs(ratio-row['delta_RMS_ratio_public_base_over_native'])/ratio)
                assert max(errors) < 1e-12
                base_error = max(base_error, *errors)
                lines.append(f'| {row["name"].replace("_", " ")} | {native_rows[row["name"]]["actual_MLP_delta_rms"]:.7g} | {measured[0]:.7g} | {ratio:.6g} | {100*measured[3]:.6f} |')
        lines += ['', 'The same frozen edits are multiplied by activations from the unchanged public base on the same 4,096 construction images. Native statistics are reused by hash. Base Bz is a separate denominator; these are matrix responses, not complete protected-model features or class scores.',
                  'Native silence alone therefore does not establish lack of meaningful response in other environments. This comparison demonstrates input-conditional amplification, not unknown-partner transfer, actual fusion damage or AGA resistance. The complete-task failure remains decisive: this point stays closed, with no new defense or repair.',
                  f'Public-base forward pass: {base["wall_seconds"]:.2f}s, {base["full_model_forward_inputs"]} image instances, zero new training indices, protected-model evaluations or merges.', '']
        provenance.update(public_base_activation_audit_sha256=hashlib.sha256(base_path.read_bytes()).hexdigest(),
                          public_base_raw_sha256=base['raw_sha256'], public_base_archive_recount_maximum_relative_error=base_error)
        (output/'summary.md').write_text('\n'.join(lines))
    class_path = root/'class_response.json'
    if class_path.exists():
        from scipy.special import logsumexp
        from verify_covariance_noise import official_labels
        intervention = json.loads(class_path.read_text())
        assert intervention['candidate_still_closed'] and intervention['input_release_sha256'] == cfg['release']['sha256']
        class_raw = root/'class_response.npz'
        assert hashlib.sha256(class_raw.read_bytes()).hexdigest() == intervention['raw_sha256']
        audit_rows = []
        assert {(r['environment'], r['branch']) for r in intervention['rows']} == {(e, b) for e in ['native', 'public_base'] for b in ['original', 'B10_only', 'B11_only', 'joint']}
        with np.load(class_raw) as archive:
            ids, labels = archive['indices'], archive['labels']
            assert ids.tolist() == intervention['indices'] == json.loads((root/'training_indices.json').read_text())['fit'][:64]
            np.testing.assert_array_equal(labels, official_labels('MNIST', True)[ids])
            lines += ['', '## Finite source classification interventions', '',
                      '| Input environment | Edit | Correct / 64 | Prediction changes | Worst relative gap drift |',
                      '|---|---|---:|---:|---:|']
            for row in intervention['rows']:
                before = archive[row['environment']+'_original']; after = archive[row['environment']+'_'+row['branch']]
                teacher, pred = before.argmax(1), after.argmax(1)
                n = len(labels); idx = np.arange(n)
                gap = before[idx, teacher, None]-before; gap[idx, teacher] = 1.
                ds = after-before; ratio = (ds[idx, teacher, None]-ds)/gap; ratio[idx, teacher] = 0.
                assert int((pred!=teacher).sum()) == row['changed']
                assert int((teacher==labels).sum()) == row['original_correct'] and int((pred==labels).sum()) == row['after_correct']
                assert abs(float(abs(ratio).max())-row['worst_relative_gap_drift']) < 1e-8
                assert abs(float((-ratio).max())-row['maximum_signed_gap_harm']) < 1e-8
                np.testing.assert_array_equal(np.bincount(teacher[pred!=teacher], minlength=10), row['changed_by_teacher_class'])
                ce0 = logsumexp(before, axis=1)-before[idx, labels]; ce1 = logsumexp(after, axis=1)-after[idx, labels]
                classes = []
                for c, expected in enumerate(row['per_true_class']):
                    keep = labels==c
                    actual = dict(class_id=c, n=int(keep.sum()), original_correct=int(((teacher==labels)&keep).sum()), after_correct=int(((pred==labels)&keep).sum()),
                                  lost=int(((teacher==labels)&(pred!=labels)&keep).sum()), gained=int(((teacher!=labels)&(pred==labels)&keep).sum()))
                    assert actual == expected
                    classes.append(actual | dict(original_mean_nll=float(ce0[keep].mean()), after_mean_nll=float(ce1[keep].mean())))
                audit_rows.append(dict(environment=row['environment'], branch=row['branch'], per_true_class=classes, prediction_histogram=np.bincount(pred, minlength=10).tolist()))
                lines.append(f'| {row["environment"].replace("_", " ")} | {row["branch"].replace("_", " ")} | {row["after_correct"]}/64 | {row["changed"]} | {row["worst_relative_gap_drift"]:.6g} |')
        lines += ['', 'Fixed first64 original construction indices, never selected by response or class. Public-base tail interventions are not merged-model or AGA results. Native64/64 preservation on these source inputs does not remove the previously verified complete-task violation. No candidate is reopened, repaired or promoted by this table.',
                  f'The first run saved64 public-base inputs before missing historical feature metadata stopped it. Its source/log remain archived; the corrected run reused that cache and added64 original-model feature controls. The fixed thresholds and inputs were retained. Successful run{intervention["wall_seconds"]:.2f}s; failed-run wall time was not separately measured.', '']
        audit = dict(status='independent_raw_score_label_index_class_and_gap_recount_passed', rows=audit_rows,
                     raw_sha256=intervention['raw_sha256'], production_result_sha256=hashlib.sha256(class_path.read_bytes()).hexdigest(),
                     report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), new_images=0, new_candidates=0, new_AGA_calls=0)
        (output/'class_response_independent_validation.json').write_text(json.dumps(audit, indent=2)+'\n')
        provenance['class_intervention_independent_rows'] = len(audit_rows)
        provenance['class_intervention_raw_sha256'] = intervention['raw_sha256']
        (output/'summary.md').write_text('\n'.join(lines))
    (output/'table_provenance.json').write_text(json.dumps(provenance, indent=2)+'\n')


def coordinate_tasks(source, output, family='Residual'):
    """Three protected tasks under a frozen rule or full-default baseline."""
    root=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True);sources={}
    def read(path):
        path=Path(path);sources[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    native={};merged={};partners={'MNIST':'SVHN','DTD':'MNIST','SVHN':'MNIST'};paired_audits=[]
    if family=='PaRaMS':
        read('results/phase3/provenance.json');rows=read('results/phase3/results.json')
        def old_acc(p,s,t):
            merger='Standalone' if s=='standalone' else 'TA'
            scope={'standalone':'None','direct':'None','AGA_target_only':'Protected expert','AGA_both':'All experts'}[s]
            r=next(r for r in rows if (r['target'],r['partner'],r['defense'],r['merger'],r['attack_scope'],r['dataset'])==
                ('MNIST','SVHN','Unprotected' if p=='original' else 'PaRaMS',merger,scope,t))
            assert abs(r['accuracy_percent']-100*r['correct']/r['n'])<1e-10
            return r['accuracy_percent']
    elif family=='TaskDirected':
        old=Path('results/task_directed_pairing/evaluation/full/00_MNIST_SVHN')
        check=read(old/'independent_validation.json');assert check['status']=='passed'
        rows=read(old/'results.json')
        def old_acc(p,s,t):
            merger='Standalone' if s=='standalone' else 'TA';scope={'standalone':'None','direct':'None','AGA_target_only':'Protected expert','AGA_both':'All experts'}[s]
            return next(r['accuracy_percent'] for r in rows if (r['defense'],r['merger'],r['attack_scope'],r['dataset'])==('Unprotected' if p=='original' else 'TaskDirected_MNIST',merger,scope,t))
    else:
        old=Path('results/residual_coordinate_control/evaluation')
        check=read(old/'independent_validation.json');assert check['status']=='passed'
        rows=read(old/'results.json')
        def old_acc(p,s,t):return next(r['top1']*100 for r in rows if (r['preset'],r['stage'],r['dataset'])==(p,s,t))
    native['MNIST']=[old_acc(p,'standalone','MNIST') for p in ['original','released']]
    for stage in ['direct','AGA_target_only','AGA_both']:
        merged['MNIST',stage]=[old_acc(p,stage,t) for t in ['MNIST','SVHN'] for p in ['original','released']]
    for case,target in [('00_DTD_MNIST','DTD'),('01_SVHN_MNIST','SVHN')]:
        check=read(root/target/'independent_validation.json')
        assert check['passed'] or (family=='PaRaMS' and check['complete_native_observation'])
        nat=read(root/target/'native.json');assert check['native_raw_sha256']==nat['raw_sha256']
        full=next(r for r in nat['rows'] if r['stage']=='full')
        if family!='PaRaMS':assert full['passed'] and full['changed']==0
        native[target]=[100*full[k]/full['n'] for k in ['original_correct','released_correct']]
        folder=root/'evaluation/full'/case;check=read(folder/'independent_validation.json');meta=read(folder/'execution.json')
        assert check['status']=='passed' and check['raw_sha256']==meta['predictions_sha256'] and check['original_batch_context_replays']
        data=read(folder/'results.json')
        for r in data:
            n=r['n'];c=r['paired'];lost,gained=c['lost'],c['gained'];m=lost+gained
            assert sum(c[k] for k in ['both_correct','both_wrong','lost','gained'])==n
            assert c['both_correct']+gained==r['correct']==sum(k['correct'] for k in r['per_class'])
            assert sum(k['n'] for k in r['per_class'])==n
            assert sum(k['original_correct'] for k in r['per_class'])==c['both_correct']+lost
            terms=[math.lgamma(m+1)-math.lgamma(k+1)-math.lgamma(m-k+1)-m*math.log(2) for k in range(min(lost,gained)+1)]
            top=max(terms);logp=min(0.,math.log(2)+top+math.log(math.fsum(math.exp(x-top) for x in terms)))/math.log(10) if m else 0.
            mean=(gained-lost)/n;half=1.96*math.sqrt(max(0.,(m-n*mean**2)/(n-1))/n)
            log_error=abs(logp-r['mcnemar_log10_p']);ci_error=max(abs(a-b) for a,b in zip([100*(mean-half),100*(mean+half)],r['paired_accuracy_delta_95ci_pp']))
            assert log_error<1e-8 and ci_error<1e-10
            paired_audits.append(dict(case=case,defense=r['defense'],scope=r['attack_scope'],dataset=r['dataset'],log10_p_error=log_error,interval_error=ci_error))
        for stage,scope in [('direct','None'),('AGA_target_only','Protected expert'),('AGA_both','All experts')]:
            hit=[next(r for r in data if (r['defense'],r['merger'],r['attack_scope'],r['dataset'])==(p,'TA',scope,t))
                 for t in [target,'MNIST'] for p in ['Unprotected',family+'_'+target]]
            assert all(abs(r['accuracy_percent']-100*r['correct']/r['n'])<1e-10 for r in hit)
            merged[target,stage]=[r['accuracy_percent'] for r in hit]
    headers=['Protected task','Native original','Native protected','TA original','TA protected','AGA target original','AGA target protected','AGA both original','AGA both protected']
    target_rows=[[t]+[f'{v:.2f}' for v in native[t]+sum([merged[t,s][:2] for s in ['direct','AGA_target_only','AGA_both']],[])] for t in partners]
    peer_rows=[[t+' / '+partners[t]]+[f'{v:.2f}' for v in sum([merged[t,s][2:] for s in ['direct','AGA_target_only','AGA_both']],[])] for t in partners]
    note='同一固定全网换序规则；TA系数0.5，未修改原版AGA，目标专家与双方作用范围分别报告。构造不使用图片，无优化或按测试选参。三个保护任务对应两种无序任务组合；MNIST/SVHN交换保护方后的无攻击模型在实数下为全网坐标等价，不能当三个独立伙伴。已知坐标对称结构，不主张新算法原创性或抗所有恢复攻击；原MNIST针对性公开匹配恢复的边界保留。原参数D0.5对照不变，本批D20事前单列，任务R0.5保持。'
    if family=='PaRaMS':
        note='PaRaMS官方完整默认调用：MLP重排及Attention对角缩放，seed0，未裁剪到项目原参数限制；TA系数0.5，原版AGA目标／双方分列。MNIST复用已核完整结果；DTD及SVHN为本次新批，伙伴固定MNIST。独立能力、原R0.5及改判数均保留，不因项目功能阈值筛去完整基线。构造不使用图片，评价数据已有历史暴露；不主张完整原论文表复现。'
    if family=='TaskDirected':
        note='同一任务SVD有限构造：74矩阵、rank32、F=T−2I及相同贪心规则，目标／公开基座参数决定一次发布；无图片构造或专家训练。MNIST复用本批已核结果；DTD、SVHN新实测，伙伴固定MNIST。DTD沿用历史训练划分，不称新确认；SVHN排除9306个旧训练索引后一次抽1024。完整测试均历史暴露，原R0.5及零改判要求不变。TA0.5、原AGA仅目标／双方分列；三保护任务对应两个无序组合，不能当三个独立伙伴。已知结构、SVD和贪心不单独主张原创，已知家族恢复边界保留。'
    title='PaRaMS完整默认基线的三任务结果' if family=='PaRaMS' else '固定全网规则的三任务结果'
    caption='Full-default PaRaMS.' if family=='PaRaMS' else 'Fixed global coordinate permutation.'
    if family=='TaskDirected':title='同一任务方向构造的三任务结果';caption='Frozen task-SVD-directed construction.'
    md=['# '+title,'',note,'']
    tex=[]
    for label,title,h,table in [('target','保护任务准确率（%）',headers,target_rows),('partner','伙伴任务准确率（%）',['Protected / partner']+headers[3:],peer_rows)]:
        md += ['## '+title,'','| '+' | '.join(h)+' |','|'+'|'.join(['---']+['---:']*(len(h)-1))+'|']
        md += ['| '+' | '.join(row)+' |' for row in table];md.append('')
        groups=(['Native'] if label=='target' else [])+['TA','AGA (target)','AGA (both)']
        tex += ['\\begin{table*}[t]','\\centering','\\footnotesize','\\setlength{\\tabcolsep}{4pt}','\\begin{tabular}{l'+'r'*(len(h)-1)+'}','\\toprule',
            ' & '+' & '.join('\\multicolumn{2}{c}{'+g+'}' for g in groups)+' \\\\',
            ('Task' if label=='target' else 'Target / partner')+' & '+' & '.join(['Before & After']*len(groups))+' \\\\','\\midrule']+[' & '.join(row)+' \\\\' for row in table]+['\\bottomrule','\\end{tabular}',
            '\\caption{'+caption+' Accuracy (\\%). Before/after protection. TA coefficient 0.5; original AGA. '+('Protected task.' if label=='target' else 'Partner task.')+'}',
            '\\label{tab:'+family.lower()+'-'+label+'}','\\end{table*}','']
    (out/'summary.md').write_text('\n'.join(md));(out/'tables.tex').write_text('\n'.join(tex))
    fig,axes=plt.subplots(1,3,figsize=(9.2,2.8),sharey=True)
    for ax,t in zip(axes,partners):
        values=np.array(native[t]+sum([merged[t,s][:2] for s in ['direct','AGA_target_only','AGA_both']],[])).reshape(4,2)
        for j,(label,color) in enumerate([('No protection','#4c78a8'),('Protected','#e07b39')]):
            ax.bar(np.arange(4)+(j-.5)*.34,values[:,j],width=.34,label=label,color=color)
        ax.set_title(t);ax.set_ylim(0,105);ax.set_xticks(range(4),['Native','TA','AGA\n(target)','AGA\n(both)'])
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Protected-task accuracy (%)')
    fig.legend(*axes[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.08),ncol=2,frameon=False)
    fig.tight_layout();fig.savefig(out/'accuracy.pdf',bbox_inches='tight');fig.savefig(out/'accuracy.png',dpi=180,bbox_inches='tight');plt.close(fig)
    (out/'table_provenance.json').write_text(json.dumps(dict(input_sha256=sources,report_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        target_rows=target_rows,partner_rows=peer_rows,MD_and_TeX_same_values=True,new_image_forwards=0),indent=2)+'\n')
    (out/'paired_statistics_validation.json').write_text(json.dumps(dict(status='passed_independent_scalar_statistics',rows=len(paired_audits),
        exact_two_sided_McNemar_log10_recomputed=True,paired_interval_from_integer_counts=True,per_class_totals_checked=True,
        maximum_log10_p_error=max(r['log10_p_error'] for r in paired_audits),maximum_interval_error=max(r['interval_error'] for r in paired_audits),
        existing_NumPy_raw_score_audits_reused=True,new_image_forwards=0),indent=2)+'\n')


def task_pairing_recovery(source, output):
    """Independent NumPy row maxima and native-array inverse, without images."""
    import torch
    from clip_benchmark import checkpoint_loader
    root=Path(source);out=Path(output);out.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((root/'protocol.json').read_text());result=json.loads((root/'results.json').read_text())
    sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
    frozen=Path('configs/task_directed_recovery.json')
    assert cfg==json.loads(frozen.read_text()) and result['protocol_sha256']==sha(frozen)
    load=checkpoint_loader(Path('external/secure-merge-attack'))
    def state(ref):
        assert sha(ref['path'])==ref['sha256']
        return load(ref['path']).cpu().eval().state_dict()
    public=state(cfg['public_base']);key='model.visual.conv1.weight';b=public[key].double().flatten(1).numpy()
    b=b/np.linalg.norm(b,axis=1,keepdims=True);audits=[];rows=[];pins={}
    for case,record in zip(cfg['cases'],result['cases']):
        assert case['target']==record['target'] and record['status']=='exact_original_state_recovered'
        original=state(case['original_for_validation']);ref=case['release'];assert sha(ref['path'])==ref['sha256']
        edits=torch.load(ref['path'],map_location='cpu');published=original|edits
        orders=[];margins=[]
        for s,label in [(published,'protected'),(original,'matched_unprotected')]:
            x=s[key].double().flatten(1).numpy();cost=(x/np.linalg.norm(x,axis=1,keepdims=True))@b.T
            forward=cost.argmax(axis=1);assert len(set(forward.tolist()))==len(forward)
            order=np.argsort(forward);assert order.tolist()==record[label]['public_coordinate_to_published_row']
            other=cost.copy();other[np.arange(len(forward)),forward]=-np.inf
            margin=float((cost[np.arange(len(forward)),forward]-other.max(1)).min());assert margin>0
            assert abs(margin-record[label]['strict_row_margin'])<1e-12
            differences=[]
            for k,v in s.items():
                if not k.startswith('model.visual.') or k.endswith(('attn.in_proj_bias','mlp.c_fc.bias')):restored=v.numpy()
                else:
                    axis=1 if k.endswith(('positional_embedding','attn.in_proj_weight','mlp.c_fc.weight')) else 0
                    restored=np.take(v.numpy(),order,axis=axis)
                if not np.array_equal(restored,original[k].numpy()):differences.append(k)
            assert not differences
            orders.append(order.tolist());margins.append(margin)
        assert orders[1]==list(range(len(orders[1])))
        for pin in case['verified_evaluation'].values():assert sha(pin['path'])==pin['sha256']
        data=json.loads(Path(case['verified_evaluation']['results']['path']).read_text())
        assert record['reuse_original_rows']==[r for r in data if r['defense']=='Unprotected']
        for r in data:
            assert r['correct']==sum(c['correct'] for c in r['per_class'])
            assert r['n']==sum(c['n'] for c in r['per_class'])
            assert abs(r['accuracy_percent']-100*r['correct']/r['n'])<1e-10
        audits.append(dict(target=case['target'],tensor_count=len(original),protected_and_control_NumPy_differences=[[],[]],
            Hungarian_order_equals_independent_row_maximum=True,strict_margins=margins,verified_evaluation=case['verified_evaluation']))
        pins[case['target']]=ref['sha256']
    assert len(audits)==len(cfg['cases'])==3
    validation=dict(status='passed_independent_public_matching_and_all_tensors',cases=audits,release_sha256=pins,
        original_expert_used_only_for_publication_assembly_and_validation=True,new_image_forwards=0,new_AGA_calls=0,
        results_sha256=sha(root/'results.json'),protocol_sha256=sha(root/'protocol.json'),report_source_sha256=sha(__file__))
    (out/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
    print('PUBLIC_RECOVERY_INDEPENDENT_CHECK',len(audits),'cases;all158tensors per protected and control exact',flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source', default='outputs/phase3'); p.add_argument('--output', default='results/phase3')
    modes=dict(dare=dare,nufilt_ob=nufilt_ob,nufilt_tasks=nufilt_tasks,prodistill=prodistill,controls=controls,eight_task=eight_task,distillation=distillation,joint_task_gate=joint_task_gate,paper_tables=paper_tables,coordinate_tasks=coordinate_tasks,params_tasks=lambda s,o:coordinate_tasks(s,o,'PaRaMS'),task_directed_tasks=lambda s,o:coordinate_tasks(s,o,'TaskDirected'),task_pairing_recovery=task_pairing_recovery)
    g=p.add_mutually_exclusive_group()
    for name in modes:g.add_argument('--'+name.replace('_','-'),action='store_true')
    args=p.parse_args();next((fn for name,fn in modes.items() if getattr(args,name)),build)(args.source,args.output)
