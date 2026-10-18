"""One matched-control view of verified two-expert recovery results; no new inference."""

import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def build():
    root=Path('results/phase3');out=root/'recovery_overview';out.mkdir(exist_ok=True)
    pins={}
    def read(path):
        p=root/path;pins[str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
        return json.loads(p.read_text())
    base=read('results.json');nufilt=read('nufilt_OB/results.json');pro=read('prodistill/results.json')
    assert read('provenance.json')['status']=='verified_batch'
    assert read('nufilt_OB/validation.json')['status']=='passed'
    assert read('prodistill/independent_audit.json')['status'].startswith('passed')
    dare=[read(f'dare/seed{s}.json') for s in [42,43,44]]
    assert all(x['validation']['status']=='passed' for x in read('dare/execution_and_validation.json'))
    names=['Unprotected','Phase1_MNIST','O/B','A/B'];tasks=['MNIST','SVHN'];rows=[]
    methods=['TA','TIES','DARE+TA','NUFILT M→S','NUFILT S→M','ProDistill']
    def select(source,name,task,method,order=None):
        values=[r for r in source if r['defense']==name and r['dataset']==task and
            r['attack_scope']=='Protected expert' and r['target']=='MNIST' and r['partner']=='SVHN'
            and (method=='ProDistill' or r['merger']==method) and (order is None or r.get('order')==order)]
        assert len(values)==1,(name,task,method,order)
        r=values[0];assert abs(r['accuracy_percent']-100*r['correct']/r['n'])<1e-10
        return r['accuracy_percent']
    for method in methods:
        for name in names:
            for task in tasks:
                if method=='DARE+TA':
                    actual=np.array([select(a,name,task,method) for a in dare])
                    control=np.array([select(a,'Unprotected',task,method) for a in dare])
                else:
                    order=['MNIST','SVHN'] if method=='NUFILT M→S' else ['SVHN','MNIST'] if method=='NUFILT S→M' else None
                    source=(nufilt if name in ['Unprotected','Phase1_MNIST','O/B'] else base) if order else pro if method=='ProDistill' else base
                    key='NUFILT' if order else method
                    actual=np.array([select(source,name,task,key,order)])
                    control=np.array([select(source,'Unprotected',task,key,order)])
                gap=control-actual
                rows.append(dict(merger=method,defense=name,dataset=task,accuracy_percent=float(actual.mean()),
                    matched_control_percent=float(control.mean()),remaining_gap_pp=float(gap.mean()),
                    minimum_gap_pp=float(gap.min()),maximum_gap_pp=float(gap.max()),
                    accuracy_values=actual.tolist(),matched_control_values=control.tolist(),
                    permission='64 unlabeled training images/task, task membership and merger-coefficient gradients' if method=='ProDistill' else 'No calibration images'))
    (out/'results.json').write_text(json.dumps(rows,indent=2)+'\n')
    with (out/'results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    labels={'Unprotected':'Unprotected','Phase1_MNIST':'Defense 1*','O/B':'O/B†','A/B':'A/B†'}
    lines=['# Recovery depends on the merger and its permissions','',
        'MNIST protected / SVHN partner; original AGA on the protected expert only. Accuracy (%). Each row uses its own unprotected control. DARE entries are means over all three fixed masks; NUFILT orders are separate. ProDistill uses calibration data and is a stronger-permission comparison.','',
        '| Merger | Defense | MNIST | SVHN | Remaining gap: MNIST / SVHN (pp) |','|---|---|---:|---:|---:|']
    tex=[r'\begin{tabular}{llrrr}',r'\toprule',r'Merger & Defense & MNIST & SVHN & Gap (M / S) \\',r'\midrule']
    for method in methods:
        for name in names:
            r=[next(x for x in rows if (x['merger'],x['defense'],x['dataset'])==(method,name,t)) for t in tasks]
            cells=[method,labels[name],*[f'{x["accuracy_percent"]:.2f}' for x in r],' / '.join(f'{x["remaining_gap_pp"]:.2f}' for x in r)]
            lines.append('| '+' | '.join(cells)+' |')
            tex.append(' & '.join(c.replace('→',r'$\to$').replace('†',r'$^{\dagger}$').replace('*',r'$^{*}$') for c in cells)+r' \\')
    note='† O/B and A/B violate task0.5 preservation; effect references only. * Defense1 has different constraints and known adaptive bias-cleanup failure. These frozen tests are historically exposed. Similar accuracy does not prove identical predictions or unique-knowledge recovery. This summary reuses prior independent audits; it does not rerun raw-score verification.'
    (out/'tables.md').write_text('\n'.join(lines+['',note])+'\n')
    (out/'tables.tex').write_text('\n'.join(tex+[r'\bottomrule',r'\end{tabular}'])+'\n')
    colors=['#2878A5','#D18532','#777777']
    with plt.rc_context({'font.size':9,'axes.spines.top':False,'axes.spines.right':False}):
        fig,axes=plt.subplots(1,2,figsize=(8.7,3.7),sharey=True,layout='constrained')
        for ax,task in zip(axes,tasks):
            for i,(name,color) in enumerate(zip(names[1:],colors)):
                r=[next(x for x in rows if (x['merger'],x['defense'],x['dataset'])==(m,name,task)) for m in methods]
                x=np.array([a['remaining_gap_pp'] for a in r]);y=np.arange(len(methods))+(i-1)*.18
                errors=np.array([[a['remaining_gap_pp']-a['minimum_gap_pp'] for a in r],[a['maximum_gap_pp']-a['remaining_gap_pp'] for a in r]])
                ax.errorbar(x,y,xerr=errors,fmt='o',ms=4,color=color,capsize=2,lw=1,label=labels[name])
            ax.axvline(0,color='#555555',lw=.7);ax.axhline(4.5,color='#BBBBBB',lw=.7,ls='--')
            ax.set(title=task,xlim=(-4,100),xlabel='Unprotected − protected accuracy (pp)',yticks=range(len(methods)),yticklabels=methods)
            ax.grid(axis='x',alpha=.18);ax.set_axisbelow(True)
        axes[0].invert_yaxis()
        fig.legend(*axes[1].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,-.02),ncol=3,fontsize=8,frameon=False)
        fig.suptitle('After AGA: matched-control gaps across mergers',fontsize=11)
        for ext in ['pdf','svg','png']:fig.savefig(out/f'recovery_gaps.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    pins[__file__]=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (out/'provenance.json').write_text(json.dumps(dict(source_files_sha256=pins,new_model_forwards=0,new_gradients=0,
        scope='Same two experts and protected-expert AGA; merger rules/permissions differ. No universal robustness or new-defense claim.',
        error_bars='DARE mask min/max around mean, not image confidence intervals; no pooling repeated images.'),indent=2)+'\n')


if __name__=='__main__':build()
