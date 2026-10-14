"""Task-SVD distance optimization with classification sufficient-statistic completion."""

import argparse
import json
from pathlib import Path

import torch

from classification_completion import amplitude_target
from clip_benchmark import digest
from conditional_task_distance import diagnose, optimize
from cross_module_distance import CrossCoupled


class ClassCompletion(CrossCoupled):
    def coefficients(self,co):
        self.solve_calls+=1;p=self.problem;z,base=self.state(co)
        _,target,root=amplitude_target(base,self.teacher,p.class_space,p.p,p.cfg['margin_fraction'],p.cfg['amplitude_bisection_steps'])
        self.last_root=root
        if not hasattr(self,'full_eye'):self.full_eye=torch.eye(len(z),dtype=z.dtype,device=z.device)
        k=torch.linalg.solve(z@z.T+self.ridge*self.full_eye,target).T@z
        return torch.cat([co.flatten(),(k/p.nb).flatten()])

    def publication_deltas(self,co,theta):
        p=self.problem
        return {'O':p.na*p.basis['A']['u']@co.reshape(p.r,p.r)@p.basis['A']['vh'],
                'B':p.nb*p.output_basis@theta[p.a_size:].reshape(p.b_rows,p.b_width)}

    @torch.no_grad()
    def fit_residual(self,co):
        p=self.problem;z,base=self.state(co);theta=self.coefficients(co)
        k=p.nb*theta[p.a_size:].reshape(p.b_rows,p.b_width)
        return float((base+z@k.T@p.output_basis.T-self.teacher).norm()/self.teacher.norm())


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--target',required=True);parser.add_argument('--protocol',required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.protocol).read_text());cfg['protocol_path']=args.protocol
    assert cfg['full_B_completion'] and cfg['class_output_completion']
    for path,pin in cfg['source_sha256'].items():assert digest(path)==pin,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if cfg.get('mode')=='optimize':optimize(args.target,cfg,problem_class=ClassCompletion)
    else:diagnose(args.target,cfg,problem_class=ClassCompletion)
