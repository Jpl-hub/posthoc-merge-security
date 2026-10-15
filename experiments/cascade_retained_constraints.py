"""Retain source adversarial witnesses during finite margin-slab completion."""

import argparse
import json
from pathlib import Path

import torch

from angular_distance import signed_residual
from attention_cascade import forward,select
from cascade_game import Cascade
from cascade_slab_completion import construct
from clip_benchmark import digest


class RetainedCascade(Cascade):
    def __init__(self,cfg):
        super().__init__(cfg);self.base_core=list(self.core);self.witnesses=[]
        if cfg.get('resume_point'):
            old=torch.load(cfg['resume_point'],map_location='cpu')['theta'].cuda()
            violation=self.residuals(old).abs().max(1).values
            self.witnesses=[row for row in violation.argsort(descending=True,stable=True).cpu().tolist() if row not in self.base_core][:cfg['retained_witness_cap']]

    @torch.no_grad()
    def critic(self,theta):
        residual=self.residuals(theta);violation=residual.abs().max(1).values
        ordered=violation.argsort(descending=True,stable=True);new=0
        for row in ordered.cpu().tolist():
            if float(violation[row])<=self.cfg['task_budget']:break
            if row in self.base_core or row in self.witnesses:continue
            if len(self.witnesses)>=self.cfg['retained_witness_cap'] or new>=self.cfg['witness_batch']:break
            self.witnesses.append(row);new+=1
        active=self.base_core+self.witnesses;worst=int(violation.argmax())
        # The historical memory cap must never exclude the current separation
        # oracle. It is a fresh adversarial constraint, not added memory.
        if worst not in active:active.append(worst)
        Path(self.cfg['completion_output'],'adversary_witnesses.json').write_text(json.dumps(dict(retained=self.witnesses,current_worst=worst,current_always_included=True)))
        rows=torch.tensor(active,device='cuda');packed=select(self.native,rows)
        h,j=forward(packed,self.p,theta,self.basis,self.q,want_jacobian=True)
        from probe_margin_distance import scores
        after=scores(h,self.p);teacher=packed['teacher'];y=teacher.argmax(1);ri=torch.arange(len(y),device='cuda')
        keep=torch.ones_like(teacher,dtype=torch.bool);keep[ri,y]=False;gaps=(teacher[ri,y,None]-teacher).masked_fill(~keep,1.)
        matrix=((j[ri,y,None]-j)/gaps[:,:,None])[keep];r=signed_residual(teacher,after)[keep]
        # This downstream-only solver needs the matrix itself, not an SVD of
        # the larger upstream/downstream matrix that it would immediately discard.
        return None,None,None,r,packed,matrix


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;construct(cfg,RetainedCascade)
