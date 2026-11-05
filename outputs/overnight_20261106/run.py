"""Run one registered construction and its existing evaluation tools serially."""
import copy
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from _impl.io import digest, save_json

spec_path=Path('results/data_free_compensation/conditional_compensation/protocol.json')
spec=json.loads(spec_path.read_text())
root=Path(spec['output'])
root.mkdir(exist_ok=True,parents=True)
py=sys.executable
def read(p): return json.loads(Path(p).read_text())
def run(*args):
    print('RUN',*args,flush=True)
    subprocess.run([py,'-u',*map(str,args)],check=True,timeout=spec['command_timeout_seconds'])
def pin(cfg):
    if isinstance(cfg.get('source_sha256'),dict):
        cfg['source_sha256']={p:digest(p) for p in cfg['source_sha256']}
    return cfg
def status(stage,**extra):
    save_json(root/'status.json',dict(stage=stage,utc=datetime.now(timezone.utc).isoformat(),**extra))
def location(task):
    return Path('outputs/centered_readout_transfer_20261106') if task=='MNIST' else Path('outputs/centered_readout_other_20261106')/task
def verify_rows(folder):
    for p in sorted(folder.glob('full/*/execution.json')):
        run('-m','experiment.main_table','--verify',p.parent)
def eight(label,defenses):
    parent=pin(read('results/data_free_compensation/finite_eight/evaluation.json'))
    parent.update(defenses=defenses,cases=[dict(target=t,partner=('SVHN' if t=='MNIST' else 'MNIST'),defenses=[label+'_'+t]) for t in spec['tasks']])
    p=root/(label+'_eight_inputs.json')
    save_json(p,parent)
    cfg=read('results/data_free_compensation/finite_eight/protocol.json')
    cfg.pop('score_reference',None)
    cfg.pop('model_reference',None)
    cfg.pop('reuse_defenses',None)
    cfg['source_sha256']=digest('experiment/eight_task.py')
    cfg['inputs']['evaluation']=dict(path=str(p),sha256=digest(p))
    cfg.update(output=str(root/(label+'_eight')),experiment='Frozen '+label+' in eight-expert TA; seven verified test datasets',
        cases=[dict(target=t,defenses=[label+'_'+t]) for t in spec['tasks']])
    p=root/(label+'_eight.json')
    save_json(p,cfg)
    run('-m','experiment.eight_task','--protocol',p)
    run('-m','experiment.eight_task','--protocol',p,'--full')

status('preflight')
for path,sha in spec['sources'].items():
    assert digest(path)==sha,path
# Fail before optimization if required data or authors' kernels are absent.
for task,details in spec['tasks'].items():
    assert digest(details['parent']['path'])==details['parent']['sha256']
    src=read('configs/defense2_inputs.json')[task]
    assert digest(Path(src['root'])/'source/source.pt')==src['source_sha256']
for task in spec['tasks']:
    cfg=read(location(task)/'evaluation.json')
    for v in cfg['legacy'].values():
        for ref in v.values(): assert digest(ref['path'])==ref['sha256']

failure=None
try:
    status('conditional_derivative_check')
    run(root/'construct.py','--protocol',spec_path,'--task','MNIST','--check-only')
    for task in spec['tasks']:
        if datetime.now(timezone.utc).timestamp()>=spec['new_construction_deadline_unix']:
            raise RuntimeError('Registered deadline reached; no new construction started')
        status('construct_'+task)
        run(root/'construct.py','--protocol',spec_path,'--task',task)
    # All three publications are frozen before any real image is loaded.
    status('all_releases_frozen')
    defenses={}
    for task in spec['tasks']:
        dest=root/task
        result=read(dest/'point/result.json')
        ref=result['publication']
        r=location(task)
        recovery=pin(read('outputs/centered_readout_transfer_20261106/recovery.json'))
        rp=root/'recovery.json'
        save_json(rp,recovery)
        run('-m','experiment.recover','--target',task,'--method','aga','--aligned-reference',
            '--protocol',rp,'--publication',ref['path'],'--sha256',ref['sha256'],'--output',dest/'AGA')
        a=dest/'AGA/edits.pt'
        entry=dict(target=task,release=ref,AGA_edits=dict(path=str(a),sha256=digest(a)))
        defenses['Conditional_'+task]=entry
        for mode in ['evaluation','b_reset']:
            cfg=pin(read(r/(mode+'.json')))
            cfg.update(output=str(dest/mode),defenses={'Conditional':entry},
                       cases=[dict(target=task,partner=('SVHN' if task=='MNIST' else 'MNIST'),defenses=['Conditional'])])
            if mode=='b_reset': cfg['cases'][0]['public_replacement']=dict(layer=11,side='B+b')
            path=dest/(mode+'.json')
            save_json(path,cfg)
            status(task+'_'+mode)
            run('-m','experiment.main_table','--protocol',path,'--pilot')
            run('-m','experiment.main_table','--protocol',path)
            verify_rows(dest/mode)
    status('prodistill')
    cfg=pin(read('outputs/centered_readout_transfer_20261106/prodistill.json'))
    other=read('outputs/centered_readout_other_20261106/prodistill.json')
    cfg['cases']+=other['cases']
    for case in cfg['cases']:
        case['releases']={'Conditional':defenses['Conditional_'+case['target']]['release']}
        # Exact matched control models and support data are reused.
    cfg.update(output=str(root/'prodistill'),experiment='Conditional B compensation: fixed author ProDistill recovery')
    path=root/'prodistill.json'
    save_json(path,cfg)
    run('-m','experiment.distill','--prodistill','--protocol',path)
    status('candidate_eight_expert')
    eight('Conditional',defenses)
except Exception as exc:
    failure=repr(exc)
    save_json(root/'failure.json',dict(failure=failure,utc=datetime.now(timezone.utc).isoformat(),
              note='Original files and settings preserved; no retry or revised model.'))
    print('CANDIDATE_FAILURE',failure,flush=True)

# This independently registered benchmark evaluates existing frozen weights;
# it does not repair or replace a failed candidate.
status('existing_readout_eight_expert',candidate_failure=failure)
centered={}
for task in spec['tasks']:
    r=location(task)
    centered['Centered_'+task]=dict(target=task,release=read(r/'freeze.json')['publication'],
        AGA_edits=dict(path=str(r/'AGA/edits.pt'),sha256=digest(r/'AGA/edits.pt')))
eight('Centered',centered)

status('tables',candidate_failure=failure)
rows=[]
for task in spec['tasks']:
    partner='SVHN' if task=='MNIST' else 'MNIST'
    for mode in ['evaluation','b_reset']:
        path=root/task/mode/'full'/('00_'+task+'_'+partner)/'results.json'
        if not path.exists(): continue
        for row in read(path):
            rows.append([task,mode,row['defense'],row['merger'],row['attack_scope'],row['dataset'],f"{row['accuracy_percent']:.2f}"])
if (root/'prodistill').exists():
    for path in (root/'prodistill').glob('*/results.json'):
        for row in read(path):
            if row.get('stage')=='full':
                rows.append([row['target'],'ProDistill',row['defense'],'ProDistill','None',row['dataset'],f"{row['accuracy_percent']:.2f}"])
headers=['Target','Test','Defense','Merger','Scope','Dataset','Accuracy']
md=['# Overnight evaluation','', 'One fixed construction; all finite results are retained. Accuracy in percent.','',
    '| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
md+=['| '+' | '.join(row)+' |' for row in rows]
(root/'tables.md').write_text('\n'.join(md)+'\n')
tex=['\\begin{longtable}{llllllr}','\\toprule',' & '.join(headers)+r' \\',r'\midrule']
tex+=[' & '.join(row).replace('_',r'\_')+r' \\' for row in rows]
tex += [r'\bottomrule',r'\end{longtable}']
(root/'tables.tex').write_text('\n'.join(tex)+'\n')
run('-m','experiment.report','--defense-comparison','--output',root/'main_tables')
status('completed',candidate_failure=failure,rows=len(rows),
       note='Completed numerical evidence is not an assertion that all defenses succeeded.')
print('OVERNIGHT_COMPLETED',failure,flush=True)
