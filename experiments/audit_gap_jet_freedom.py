"""Post-freeze source-only matched-coordinate full-hidden/gap-jet comparison.

Reporting only. Does not select parameters or read confirmation/partner/test
responses; the old full-hidden chart is evaluated at the SAME final C and L.
"""
import json,os,time,argparse
from pathlib import Path
os.environ['OPENBLAS_NUM_THREADS']='4'
import numpy as np
import torch
from clip_benchmark import digest
from subspace_probe import save_json
from verify_downstream_first import convert
from verify_gap_jet_publication import decode,forward
from verify_hermite_publication import decode as full_decode
from verify_levelset_audits import residual


def direction(before,after):
    y=before.argmax(1);row=np.arange(len(y));bs=before-before.mean(1,keepdims=True);aa=after-after.mean(1,keepdims=True)
    scale=np.sum(bs*aa,1)/np.sum(bs*bs,1);normal=aa-scale[:,None]*bs
    gap=before[row,y,None]-before;mask=np.ones_like(before,dtype=bool);mask[row,y]=False
    normalized=np.where(mask,(after[row,y,None]-after)/np.where(mask,gap,1),np.inf)
    crossed=normalized.min(1)<0
    assert int(crossed.sum())==int((after.argmax(1)!=y).sum())
    changed=(after-before);identity=np.abs(changed[row,y,None]-changed-((scale-1)[:,None]*gap+normal[row,y,None]-normal)).max()
    return dict(n=len(y),changed=int(crossed.sum()),per_class=[int((crossed&(y==i)).sum()) for i in range(before.shape[1])],
        positive_score_scales=int((scale>0).sum()),median_score_scale=float(np.median(scale)),
        non_scaling_score_residual_rms=float(np.sqrt(np.mean(normal**2))),minimum_teacher_gap_ratio=float(normalized.min()),
        decomposition_identity_error=float(identity))


def run(cfg):
    start=time.monotonic();out=Path(cfg['output']);report=json.loads((out/'results.json').read_text())
    assert digest(out/'diagnostic.pt')==report['diagnostic_sha256']
    cal=convert(torch.load(cfg['calibration_reference'],map_location='cpu'));d=convert(torch.load(out/'diagnostic.pt',map_location='cpu'))
    old=convert(torch.load(cfg['environment_cache'],map_location='cpu'));p=cal['parameters'];records=[]
    for name,fn in [('decision_normal_gap_jet',decode),('full_hidden_at_same_C_L',full_decode)]:
        c,k,info=fn(d['theta'],cal,cfg);ss=forward(d['data'],p,c,k,cal)
        source=direction(d['teacher'],ss);source['worst_task_drift']=float(np.abs(residual(d['teacher'],ss)).max())
        env=[dict(name=e['name'],**direction(e['teacher'],forward(e['data'],old['proxy_parameters'],c,k,cal,.5))) for e in d['environments']]
        records.append(dict(chart=name,parameter_distance=float(np.sqrt(np.sum(c*c)+np.sum(k*k))),source=source,environments=env))
    result=dict(status='passed_frozen_coordinate_source_ablation',records=records,training_point_sha256=report['training_point_sha256'],
        diagnostic_sha256=report['diagnostic_sha256'],source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        same_optimization_coordinates=True,new_parameter_updates=0,new_full_model_forwards=0,release_selected=False,
        confirmation_responses_loaded=False,test_loaded=False,peer_loaded=False,
        limitation='Post-freeze source-only functional comparison; alternate chart is not a new candidate. Same C/L does not imply equal matrix distance; no causal claim about all possible optimizers or unknown partners.')
    save_json(out/'matched_chart_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);a=ap.parse_args();torch.set_num_threads(4);run(json.loads(Path(a.protocol).read_text()))
