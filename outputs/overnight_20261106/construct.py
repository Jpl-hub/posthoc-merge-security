"""One fixed A update with B recomputed by conditional ridge regression."""
import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _impl.io import digest, save_json
from _impl.models import checked_models, initialize


def fit(a, bias, populations, original, rho):
    g = rho * torch.eye(a.shape[0] + 1, device=a.device, dtype=a.dtype)
    t = rho * original
    for x, y, scale in populations:
        for xx, yy in zip(x.split(256), y.split(256)):
            h = F.linear(xx, a, bias)
            h = h * torch.sigmoid(1.702 * h)
            h = torch.cat((h, h.new_ones(len(h), 1)), 1)
            g = g + scale * (h.T @ h)
            t = t + scale * (yy.T @ h)
    return torch.cholesky_solve(t.T, torch.linalg.cholesky(g)).T


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--protocol', required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--check-only', action='store_true')
    args = p.parse_args()
    cfg = json.loads(Path(args.protocol).read_text())
    source_key=str(Path(__file__).resolve().relative_to(Path.cwd()))
    assert digest(__file__) == cfg['sources'][source_key]
    initialize()
    task = args.task
    spec = cfg['tasks'][task]
    out = Path(cfg['output']) / task / ('source_check' if args.check_only else 'point')
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    base, expert, head, pins = checked_models(task)
    raw_spec = json.loads(Path('configs/defense2_inputs.json').read_text())[task]
    source = Path(raw_spec['root']) / 'source/source.pt'
    assert digest(source) == raw_spec['source_sha256']
    raw = torch.load(source, map_location='cpu', weights_only=False)
    assert digest(spec['parent']['path']) == spec['parent']['sha256']
    parent = torch.load(spec['parent']['path'], map_location='cpu', weights_only=True)
    state, public = expert.state_dict(), base.state_dict()
    prefix = 'model.visual.transformer.resblocks.11.mlp.'
    keys = [prefix + x for x in ('c_fc.weight','c_fc.bias','c_proj.weight','c_proj.bias')]
    to = lambda x: x.detach().cuda().double()
    A, a, B, b = [to(state[k]) for k in keys]
    A0, a0, B0, b0 = [to(public[k]) for k in keys]
    Ap = to(parent[keys[0]])
    beta = to(state['model.visual.transformer.resblocks.11.ln_2.bias'])
    gamma = to(state['model.visual.transformer.resblocks.11.ln_2.weight'])
    assert digest(raw_spec['layer_inputs']['path']) == raw_spec['layer_inputs']['sha256']
    u = torch.load(raw_spec['layer_inputs']['path'], map_location='cpu', weights_only=False)
    xu = beta + gamma * to(u)
    act = lambda x: x * torch.sigmoid(1.702 * x)
    with torch.no_grad():
        yu = torch.cat([F.linear(act(F.linear(x, A, a)), B, b) for x in xu.split(256)])
    cache = {name:{k:to(v) for k,v in row.items()} for name,row in raw['cache'].items()}
    native = cache['native']
    xn, yn = native['x'], native['final'] - native['prefix']
    populations = []
    for x,y in ((xu,yu),(xn,yn)):
        variance = y.square().sum()/len(y) - y.mean(0).square().sum()
        populations.append((x,y,1/(len(y)*variance)))
    R0 = torch.cat((B,b[:,None]),1)
    ridge = raw_spec['original_ridge']
    tau = raw_spec['original_native_limit']
    U,_,Vh = torch.linalg.svd(A-A0, full_matrices=False)
    U,V = U[:,:32],Vh[:32].T
    W = to(head.weight)
    eps = expert.model.visual.ln_post.eps
    names=['model.visual.ln_post.weight','model.visual.ln_post.bias','model.visual.proj']
    rn = [to(state[k]) for k in names]
    rp = [to((state[k]+public[k])/2) for k in names]
    classify=lambda h,r: F.normalize(F.layer_norm(h,(768,),r[0],r[1],eps)@r[2],dim=-1)@W.T
    target=native['scores'].softmax(1).detach()
    labels=native['scores'].argmax(1)
    scale=float(A.norm())
    variable=torch.zeros_like(A,requires_grad=True)

    def parameters(v):
        d=Ap-A+scale*v
        d=d/(torch.linalg.vector_norm(d)/ (cfg['A_limit']*scale)).clamp_min(1)
        aa=A+d
        bb=a-(aa-A)@beta
        return aa,bb

    def compute(v,details=False):
        aa,bb=parameters(v)
        r=fit(aa,bb,populations,R0,ridge)
        h=act(F.linear(xn,aa,bb))
        score=classify(native['prefix']+F.linear(h,r[:,:-1],r[:,-1]),rn)
        kl=F.kl_div(score.log_softmax(1),target,reduction='batchmean')
        error=ridge*(r-R0).square().sum()
        for x,y,s in populations:
            for xx,yy in zip(x.split(256),y.split(256)):
                error=error+s*(F.linear(act(F.linear(xx,aa,bb)),r[:,:-1],r[:,-1])-yy).square().sum()
        responses=[]
        public_counts={}
        for name in ('direct','recovered'):
            row=cache[name]
            # Parent order is fixed in this source proxy. Actual author AGA
            # is rerun on the frozen final model during evaluation.
            if name=='recovered':
                ax,bx=aa[order],bb[order]
                rr=r[:,:-1][:,order]
            else: ax,bx,rr=aa,bb,r[:,:-1]
            hh=act(F.linear(row['x'],(ax+A0)/2,(bx+a0)/2))
            for mode,ro,bo in [('B_reset',B0,b0),('TA',(rr+B0)/2,(r[:,-1]+b0)/2)]:
                z=classify(row['prefix']+F.linear(hh,ro,bo),rp)
                logprior=torch.logsumexp(z.log_softmax(1),0)-torch.log(z.new_tensor(len(z)))
                for processing,zz in [('raw',z),('calibrated',z-logprior)]:
                    own=zz.gather(1,labels[:,None]).flatten()
                    other=zz.scatter(1,labels[:,None],-torch.inf).amax(1)
                    responses.append((own-other).clamp_min(0).mean())
                    public_counts[name+'/'+mode+'/'+processing]=int(zz.argmax(1).eq(labels).sum())
        # Parameter probes are a soft penalty. Native finite outputs, rather
        # than a zero parameter projection, determine task preservation.
        delta=aa-A
        probe=((U.T@delta).square().sum()+(delta@V).square().sum())/(scale*scale)
        radius=(r[:,:-1]-B).norm()/B.norm()
        limits=(radius-cfg['B_limit']).clamp_min(0).square()
        loss=kl+error/tau+torch.stack(responses).mean()+probe+100*limits
        if details:
            return r,dict(loss=float(loss),native_KL=float(kl),native_error=float(error),native_limit=tau,
                          task_probe=float(probe),B_relative=float(radius),
                          native_matches=int(score.argmax(1).eq(labels).sum()),native_n=len(labels),
                          source_counts=public_counts)
        return loss

    from _impl.attacks import load_author
    author,_=load_author(Path('external/secure-merge-attack/utils_attack_B32.py'))
    order=author['solve_permutation'](Ap.float().cpu(),A0.float().cpu()).cuda()
    calls=0
    trace=[]
    solver=torch.optim.LBFGS([variable],lr=1,max_iter=cfg['iterations'],max_eval=cfg['evaluations'],
                            history_size=5,line_search_fn='strong_wolfe')
    def closure():
        nonlocal calls
        solver.zero_grad(set_to_none=True)
        value=compute(variable)
        assert torch.isfinite(value),float(value)
        value.backward()
        assert variable.grad is not None and torch.isfinite(variable.grad).all()
        calls+=1
        trace.append(dict(call=calls,loss=float(value.detach()),gradient_norm=float(variable.grad.norm())))
        print(task,trace[-1],flush=True)
        return value
    with torch.no_grad():
        _,initial=compute(variable,True)
    save_json(out/'initial.json',initial)
    if args.check_only:
        initial_loss=compute(variable)
        gradient,=torch.autograd.grad(initial_loss,variable)
        direction=gradient/gradient.norm()
        step=1e-5
        with torch.no_grad():
            difference=float((compute(variable+step*direction)-compute(variable-step*direction))/(2*step))
        analytic=float((gradient*direction).sum())
        relative_error=abs(difference-analytic)/(1+abs(analytic))
        save_json(out/'check.json',dict(analytic=analytic,central_difference=difference,
                  relative_error=relative_error,passed=relative_error<.02,
                  seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated()))
        assert relative_error<.02,(analytic,difference,relative_error)
        print('CHECK_PASSED',task,relative_error,flush=True)
        return
    solver.step(closure)
    with torch.no_grad():
        final_r,report=compute(variable,True)
        aa,bb=parameters(variable)
        edits={k:v.float().cpu() for k,v in zip(keys,(aa,bb,final_r[:,:-1],final_r[:,-1]))}
        report['A_relative']=float((aa-A).norm()/A.norm())
        report['relative_edits']={k:float((v.double()-state[k].double()).norm()/state[k].double().norm()) for k,v in edits.items()}
    torch.save(edits,out/'point.pt')
    save_json(out/'result.json',dict(task=task,publication=dict(path=str(out/'point.pt'),sha256=digest(out/'point.pt')),
        source=report,initial=initial,trace=trace,source_checks_passed=(report['native_error']<=tau and
        report['A_relative']<=cfg['A_limit']+1e-6 and report['B_relative']<=cfg['B_limit']+1e-6),
        source_sha256=digest(__file__),protocol_sha256=digest(args.protocol),checkpoints=pins,
        real_construction_images=0,expert_updates=0,evaluations=calls,seconds=time.monotonic()-start))
    print('FROZEN',task,report,flush=True)


if __name__=='__main__':
    main()
