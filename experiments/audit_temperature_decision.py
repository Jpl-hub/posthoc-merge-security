"""Finite source decision decomposition of a frozen temperature point."""

import argparse
import json
import time
from pathlib import Path

import torch

from angular_distance import angular_scores, signed_residual
from clip_benchmark import digest
from probe_margin_distance import scores
from subspace_probe import save_json


@torch.no_grad()
def audit(root):
    start=time.monotonic();root=Path(root);meta=json.loads((root/'results.json').read_text())
    assert digest(root/'diagnostic.pt')==meta['diagnostic_sha256'];assert digest(root/'predictions.pt')==meta['predictions_sha256']
    raw=torch.load(root/'diagnostic.pt',map_location='cpu');pred=torch.load(root/'predictions.pt',map_location='cpu');data=raw['native'];p=raw['parameters'];n=data['natural_n']
    before=pred['source_scores'][:n];cent=before-before.mean(1,keepdim=True);y=before.argmax(1);rows=torch.arange(n)
    groups={'temperature_only':scores(data['base'][:n],p),'B_only':pred['B_only_scores'],'joint':pred['released_scores'][:n]};records={}
    for name,after in groups.items():
        delta=after-after.mean(1,keepdim=True);alpha=(cent*delta).sum(1)/cent.square().sum(1)
        angular=(angular_scores(after)-angular_scores(before)).norm(dim=1);transverse=(delta-alpha[:,None]*cent).norm(dim=1)/cent.norm(dim=1)
        competing=after.clone();competing[rows,y]=-torch.inf;k=competing.argmax(1)
        gap0=before[rows,y]-before[rows,k];gap1=after[rows,y]-after[rows,k];harm=1-gap1/gap0
        per_class=[]
        for cls in range(before.shape[1]):
            take=y==cls;per_class.append(dict(class_id=cls,n=int(take.sum()),changed=int((after.argmax(1)[take]!=cls).sum()),mean_finite_margin_harm=float(harm[take].mean()),mean_direction_change=float(angular[take].mean())))
        records[name]=dict(changed=int((after.argmax(1)!=y).sum()),radial_scale_quantiles=torch.quantile(alpha,torch.tensor([0.,.5,1.],dtype=torch.float64)).tolist(),
            transverse_relative_quantiles=torch.quantile(transverse,torch.tensor([0.,.5,1.],dtype=torch.float64)).tolist(),angular_change_quantiles=torch.quantile(angular,torch.tensor([0.,.5,1.],dtype=torch.float64)).tolist(),
            worst_raw_margin_drift=float(signed_residual(before,after).abs().max()),finite_margin_crossings=int((harm>1.).sum()),per_class=per_class)
    result=dict(target=meta['target'],natural_n=n,records=records,point_sha256=meta['training_point_sha256'],source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
                partner_used=False,test_used=False,new_release_weights=0,selection_used=False,scope='Post-freeze finite source score radial/transverse decomposition and actual competitor margins. It does not predict unknown-partner accuracy or gate real evaluations.')
    save_json(root/'decision_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);args=ap.parse_args();torch.set_num_threads(4);audit(args.output)
