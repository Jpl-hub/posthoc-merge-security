"""Minimum-Frobenius full-B distillation at the same task-probe and distance budget."""

import argparse
import json
from pathlib import Path

import torch

from clip_benchmark import digest
from conditional_task_distance import diagnose, optimize
from cross_module_distance import CrossCoupled


class FullCompletion(CrossCoupled):
    def coefficients(self,co):
        self.solve_calls+=1;p=self.problem;z,base=self.state(co)
        if not hasattr(self,"full_eye"):
            self.full_eye=torch.eye(len(z),dtype=torch.float64,device=z.device)
        # Dual ridge solution uses the same absolute lambda as the restricted B solve.
        solution=torch.linalg.solve(z@z.T+self.ridge*self.full_eye,self.teacher-base)
        eb=solution.T@z
        return torch.cat([co.flatten(),(eb/p.nb).flatten()])

    def publication_deltas(self,co,theta):
        p=self.problem
        return {"O":p.na*p.basis["A"]["u"]@co.reshape(p.r,p.r)@p.basis["A"]["vh"],
                "B":p.nb*theta[p.a_size:].reshape_as(p.p["B"])}

    @torch.no_grad()
    def fit_residual(self,co):
        p=self.problem;z,base=self.state(co);theta=self.coefficients(co)
        eb=p.nb*theta[p.a_size:].reshape_as(p.p["B"])
        return float((base+z@eb.T-self.teacher).norm()/self.teacher.norm())


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument("--target",required=True);parser.add_argument("--protocol",required=True)
    args=parser.parse_args();cfg=json.loads(Path(args.protocol).read_text());cfg["protocol_path"]=args.protocol
    assert cfg["full_B_completion"]
    for path,pin in cfg["source_sha256"].items():assert digest(path)==pin,path
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    if cfg.get("mode")=="optimize":optimize(args.target,cfg,problem_class=FullCompletion)
    else:diagnose(args.target,cfg,problem_class=FullCompletion)
