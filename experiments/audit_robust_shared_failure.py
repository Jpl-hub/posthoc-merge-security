"""Read-only independent attribution of the frozen confirmation failure."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from clip_benchmark import digest
from subspace_probe import save_json
from verify_covariance_noise import official_labels
from verify_downstream_first import convert
from verify_levelset_audits import residual


def run(cfg):
    start=time.monotonic();root=Path(cfg['output']);report=json.loads((root/'results.json').read_text());confirm=json.loads((root/'confirmation_validation.json').read_text())
    assert not confirm['confirmation_passed'] and digest(root/'confirmation.pt')==confirm['confirmation_sha256']
    c=convert(torch.load(root/'confirmation.pt',map_location='cpu'));d=convert(torch.load(root/'diagnostic.pt',map_location='cpu'))
    point=convert(torch.load(root/'training_point.pt',map_location='cpu'));old=convert(torch.load(cfg['initial_point'],map_location='cpu'))
    pars=convert(torch.load(cfg['cache'],map_location='cpu')['parameters']);theta=point['theta'];r=cfg['probe_rank'];q=old['output_basis']
    delta=dict(O=np.linalg.norm(pars['O'])*old['basis_O']['u']@theta[:r*r].reshape(r,r)@old['basis_O']['vh'],
               B=np.linalg.norm(pars['B'])*q@theta[r*r:].reshape(len(q.T),pars['B'].shape[1]))
    delta_errors={k:float(np.max(np.abs(v-point['release_deltas'][k]))) for k,v in delta.items()}
    assert max(delta_errors.values())<1e-9
    fp32={k:(pars[k]+point['release_deltas'][k]).astype(np.float32).astype(np.float64)-pars[k] for k in delta}
    assert all(np.array_equal(fp32[k],c['release_deltas'][k]) for k in delta)
    before,after=c['teacher'],c['released_scores'];err=residual(before,after);risk=np.abs(err).max(1)
    n=len(c['indices']);ctx=cfg['contexts_per_class']*before.shape[1];count=2*r;probe_n=ctx*count
    assert len(before)==n+2*probe_n
    labels=official_labels(cfg['target'],True)[c['indices']];np.testing.assert_array_equal(labels,c['labels'])
    def segment(name,lo,hi):
        b,a=before[lo:hi],after[lo:hi];values=risk[lo:hi];index=lo+int(values.argmax());y=int(before[index].argmax())
        k=int(np.abs(err[index]).argmax());gap=float(before[index,y]-before[index,k]);delta_gap=float((after[index,y]-after[index,k])-gap)
        return dict(group=name,n=hi-lo,worst_margin_drift=float(values.max()),over_budget_rows=int((values>.5).sum()),
            prediction_changes=int((a.argmax(1)!=b.argmax(1)).sum()),worst_row=index,teacher_class=y,competitor=k,
            original_class_gap=gap,finite_gap_change=delta_gap,
            changed_by_teacher_class=np.bincount(b.argmax(1)[a.argmax(1)!=b.argmax(1)],minlength=before.shape[1]).tolist())
    groups=[segment('natural',0,n),segment('O_task_probe',n,n+probe_n),segment('B_task_probe',n+probe_n,len(before))]
    changed=before[:n].argmax(1)!=after[:n].argmax(1);items=[]
    for i in np.flatnonzero(changed):
        items.append(dict(position=int(i),training_index=int(c['indices'][i]),label=int(labels[i]),before=int(before[i].argmax()),after=int(after[i].argmax()),relative_drift=float(risk[i])))
    result=dict(status='confirmed_frozen_candidate_task_constraint_failure',groups=groups,natural_prediction_changes=items,
        original_correct=int((before[:n].argmax(1)==labels).sum()),released_correct=int((after[:n].argmax(1)==labels).sum()),
        maximum_relative_drift=float(risk.max()),budget=.5,threshold_changed=False,point_changed=False,confirmation_redrawn=False,
        independently_reconstructed_delta_errors=delta_errors,FP32_edits_array_equal=True,
        new_model_forwards=0,new_attack_calls=0,new_weights=0,used_for_parameter_selection=False,
        point_sha256=report['training_point_sha256'],confirmation_sha256=confirm['confirmation_sha256'],
        source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        interpretation='Source finite constraints do not generalize to the sole fresh split under the frozen metric. A gain in raw accuracy is not a waiver of the predeclared functional constraint. This validation-only analysis must not feed parameter repairs or retrospective threshold changes.')
    save_json(root/'failure_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();torch.set_num_threads(4)
    run(json.loads(Path(args.protocol).read_text()))
