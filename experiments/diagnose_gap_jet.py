"""One predeclared non-optimized gap-jet solvability/finite-response check."""
import argparse,json,time
from pathlib import Path
import numpy as np
import torch
from clip_benchmark import digest
from gap_jet_publication import GapPublication
from subspace_probe import save_json
from task_probe_distance import initialize
from verify_downstream_first import convert
from verify_gap_jet_publication import decode


def run(cfg):
    out=Path(cfg['mechanism_output']);out.mkdir(parents=True,exist_ok=False);start=time.monotonic();p=GapPublication(cfg)
    cal=convert(torch.load(cfg['calibration_reference'],map_location='cpu'))
    theta=p.zero.clone();theta[:p.size]=(-cfg['mechanism_O_distance']*torch.diag(p.chart.basis['s'])/p.chart.basis['s'].norm()).flatten()
    with torch.no_grad():
        state=p.evaluate(theta);c,k,info=p.decode(theta)
        cn,kn,ni=decode(theta.cpu().numpy(),cal,cfg)
        error=float(np.abs(kn-k.cpu().numpy()).max());assert error<1e-8
    with torch.enable_grad():
        leaf=theta.detach().requires_grad_(True);v=p.objective(leaf);g=torch.autograd.grad(v,leaf)[0].detach()
    direction=g/g.norm();eps=cfg['derivative_step']
    with torch.no_grad():fd=float((p.objective(theta+eps*direction)-p.objective(theta-eps*direction))/(2*eps))
    fd_error=abs(fd-float(g@direction));assert fd_error<cfg['derivative_tolerance']
    passed=bool(state['metrics']['gap_error']<cfg['score_tolerance'] and state['metrics']['slope_error']<cfg['slope_tolerance']
        and state['metrics']['constraint_condition']<cfg['maximum_condition'] and state['metrics']['hidden_change']>1e-6)
    torch.save(dict(theta=theta.cpu(),K=k.cpu(),gradient=g.cpu(),scores=state['scores'].cpu(),metrics=state['metrics']),out/'mechanism.pt')
    result=dict(status='passed_mechanism' if passed else 'closed_gap_jet_mechanism',metrics=state['metrics'],
        nonlinear_equation_unique=True,proof_condition='b-|d|/sqrt(width)>0, sigma includes eps; strictly monotone surjective scalar numerator',
        independent_solver='SciPy brentq with analytic bracket + NumPy SVD least-squares, no Torch quadratic or normal equations',
        numpy_coefficient_error=error,numpy_gap_error=ni['gap_error'],numpy_slope_error=ni['slope_error'],
        objective_directional_derivative_error=fd_error,source_sha256=digest(__file__),mechanism_sha256=digest(out/'mechanism.pt'),
        new_release_weights=0,optimizer_updates=0,explicit_reverse_gradient_calls=1,full_model_forwards=0,
        source_training_reused=True,confirmation_data_used=False,partner_used=False,test_used=False,wall_seconds=time.monotonic()-start,
        limitation='One fixed non-releasable diagnostic seed, not scaled into candidate; root-jet preservation is not independent utility or unknown-peer protection. Candidate, if any, starts from zero under separate frozen optimization.')
    save_json(out/'results.json',result);save_json(out/'protocol.json',cfg);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);a=ap.parse_args();initialize();cfg=json.loads(Path(a.protocol).read_text())
    for file,pin in cfg['source_sha256'].items():assert digest(file)==pin,file
    run(cfg)
