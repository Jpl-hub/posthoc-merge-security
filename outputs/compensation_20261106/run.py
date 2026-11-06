"""One finite A/B compensation test, retaining the main B displacement."""
import ast
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from _impl.io import check_files, digest, save_json
from _impl.models import initialize


cfg = json.loads(Path('results/data_free_compensation/compensation_protocol.json').read_text())
check_files(cfg['sources'])
assert digest(cfg['parent_source']) == cfg['parent_source_sha256']
out = Path(cfg['output']) / 'source'
out.mkdir(exist_ok=False)
save_json(out / 'protocol.json', cfg)
initialize()
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)
start = time.monotonic()
tree = ast.parse(Path(cfg['parent_source']).read_text())
main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
def assigns(node, name):
    return isinstance(node, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id == name or isinstance(t, ast.Tuple)
        and any(isinstance(v, ast.Name) and v.id == name for v in t.elts) for t in node.targets)
lo = next(i for i, n in enumerate(main.body) if assigns(n, 'pins'))
hi = next(i for i, n in enumerate(main.body) if assigns(n, 'releases'))
imports = [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]
env = dict(cfg=json.loads(Path(cfg['parent_protocol']).read_text()), __name__='frozen_preparation')
exec(compile(ast.Module(body=imports + main.body[lo:hi], type_ignores=[]), cfg['parent_source'], 'exec'), env)
theta, clean, anchor = [env[k] for k in ('initial', 'clean', 'anchor')]
iA, ia, iB, ib = [env[k] for k in ('iA', 'ia', 'iB', 'ib')]
U, V, beta, native = [env[k] for k in ('U', 'V', 'beta', 'native')]
act = lambda z: z * torch.sigmoid(1.702 * z)
project = lambda d: d-U@(U.T@d)-(d@V)@V.T+U@((U.T@d)@V)@V.T
scale = float(clean[iA].norm())
task = json.loads(Path('configs/defense2_inputs.json').read_text())['MNIST']
assert digest(task['layer_inputs']['path']) == task['layer_inputs']['sha256']
u = torch.load(task['layer_inputs']['path'], map_location='cpu', weights_only=False).cuda().double()
gamma = clean[env['names'].index('ln_2.weight')]
xu = beta + gamma*u
with torch.no_grad():
    yu = torch.cat([F.linear(act(F.linear(x, clean[iA], clean[ia])), clean[iB], clean[ib]) for x in xu.split(256)])
source = torch.load(env['cfg']['synthetic_source']['path'], map_location='cpu', weights_only=False)
yn = source['cache']['native']['final'].cuda().double()-native['prefix']
populations = [(x, y, 1/(len(y)*(y.square().sum()/len(y)-y.mean(0).square().sum())))
               for x, y in [(xu, yu), (native['x'], yn)]]
Rorig = torch.cat((clean[iB], clean[ib][:, None]), 1)
Rparent = torch.cat((theta[iB], theta[ib][:, None]), 1)
rho, tau = task['original_ridge'], task['original_native_limit']
constant = rho*Rorig.square().sum()+sum(s*y.square().sum() for _,y,s in populations)


def conditional(a, b):
    g = rho*torch.eye(3073, device='cuda', dtype=torch.float64)
    t = rho*Rorig
    for x,y,s in populations:
        for xx,yy in zip(x.split(256),y.split(256)):
            h = act(F.linear(xx,a,b))
            h = torch.cat((h,h.new_ones(len(h),1)),1)
            g = g+s*(h.T@h)
            t = t+s*(yy.T@h)
    l = torch.linalg.cholesky(g)
    r = torch.cholesky_solve(t.T,l).T
    return r,l,constant-(r*t).sum()


with torch.no_grad():
    center, l0, _ = conditional(theta[iA],theta[ia])
    excess = (Rparent-center)@l0
    excess_square = float(excess.square().sum())


def parameters(v):
    da = scale*project(v)
    a, b = theta[iA]+da,theta[ia]-da@beta
    center,l,minimum = conditional(a,b)
    r = center+torch.linalg.solve_triangular(l.T,excess.T,upper=True).T
    p = list(theta)
    p[iA],p[ia],p[iB],p[ib] = a,b,r[:,:-1],r[:,-1]
    return tuple(p),minimum+excess_square


def actual_error(p):
    r = torch.cat((p[iB],p[ib][:,None]),1)
    e = rho*(r-Rorig).square().sum()
    for x,y,s in populations:
        for xx,yy in zip(x.split(256),y.split(256)):
            e = e+s*(F.linear(act(F.linear(xx,p[iA],p[ia])),p[iB],p[ib])-yy).square().sum()
    return e


contexts = dict(env['contexts'])
own = torch.load(cfg['task_contexts'],map_location='cpu',weights_only=False)
contexts.update({n:{k:v.cuda().double() for k,v in row.items()} for n,row in own.items()})
labels = source['cache']['native']['scores'].argmax(1).cuda()


def source_scores(p):
    h = native['prefix']+F.linear(act(F.linear(native['x'],p[iA],p[ia])),p[iB],p[ib])
    scores = dict(native=env['head'](env['native_ln'](h)@env['native_proj']))
    order = env['attack']['solve_permutation'](p[iA].detach().float().cpu(),anchor[iA].float().cpu()).cuda()
    for name,row in contexts.items():
        own_context = name.startswith('task_')
        reference = clean if own_context else anchor
        ln,proj = (env['native_ln'],env['native_proj']) if own_context else (env['public_ln'],env['public_proj'])
        a,b,r = p[iA],p[ia],p[iB]
        if name.endswith('recovered'):
            a,b,r = a[order],b[order],r[:,order]
        z = act(F.linear(row['x'],(a+reference[iA])/2,(b+reference[ia])/2))
        for mode,B,bias in [('TA',(r+reference[iB])/2,(p[ib]+reference[ib])/2),
                            ('B_reset',(anchor[iB]+reference[iB])/2,(anchor[ib]+reference[ib])/2)]:
            h = row['prefix']+F.linear(z,B,bias)
            scores[name+'/'+mode] = env['head'](ln(h)@proj)
    return scores


def margin(scores, reference):
    true = scores.gather(1,reference[:,None]).flatten()
    other = scores.scatter(1,reference[:,None],-torch.inf).amax(1)
    return (other-true).clamp_max(0).mean()


def objectives(p):
    w = env['recovered_forward'](p)
    return (float(margin(env['recovery_scores'](w),env['labels'])),
            float(margin(source_scores(p)['recovered/B_reset'],labels)))


v = torch.zeros_like(theta[iA])
with torch.no_grad():
    p,e = parameters(v)
    reconstruction = max(float((p[i]-theta[i]).abs().max()) for i in (iA,ia,iB,ib))
    identity_error = abs(float(actual_error(p))-float(e))
    initial_scores = source_scores(p)
    initial_counts = {k:int(s.argmax(1).eq(labels).sum()) for k,s in initial_scores.items()}
assert reconstruction<1e-8 and identity_error<1e-7,(reconstruction,identity_error)
current = objectives(p)
initial = dict(error=float(e),counts=initial_counts,objectives=current,
               reconstruction_error=reconstruction,quadratic_identity_error=identity_error)
save_json(out/'initial.json',initial)
trace = []
for step in range(cfg['steps']):
    if time.monotonic()-start>cfg['maximum_construction_seconds']:
        break
    variable = v.detach().requires_grad_(True)
    p,e = parameters(variable)
    if current[0]<=current[1]:
        measure = lambda w:(margin(env['recovery_scores'](w),env['labels']),)
        value, gp, _ = env['recover_gradient'](env['inner_loss'],measure,anchor,
                                               tuple(t.detach() for t in p),lambda q:env['call'](q,env['xt']))
        gradient, = torch.autograd.grad([p[i] for i in (iA,ia,iB,ib)],variable,
                                        grad_outputs=[gp[i] for i in (iA,ia,iB,ib)])
        active = 'distillation'
        assert abs(value-current[0])<1e-5*(1+abs(value)),(value,current)
    else:
        value = margin(source_scores(p)['recovered/B_reset'],labels)
        gradient, = torch.autograd.grad(value,variable)
        active = 'B_reset'
    assert torch.isfinite(gradient).all() and float(gradient.norm())>0
    direction = gradient/gradient.norm()
    if step==0:
        eps = cfg['gradient_step']
        pairs = [objectives(parameters(v+sign*eps*direction)[0]) for sign in [-1,1]]
        j = 0 if active=='distillation' else 1
        finite = (pairs[1][j]-pairs[0][j])/(2*eps)
        analytic = float((gradient*direction).sum())
        error = abs(finite-analytic)/(1+abs(analytic))
        save_json(out/'gradient_check.json',dict(analytic=analytic,finite=finite,relative_error=error,
                                               passed=error<cfg['gradient_tolerance']))
        assert error<cfg['gradient_tolerance'],(analytic,finite,error)
    accepted = False
    for backtrack in range(cfg['backtracking']):
        trial = v+cfg['step']/2**backtrack*direction
        with torch.no_grad():
            q,error = parameters(trial)
            legal = float(error)<=tau and float((q[iA]-clean[iA]).norm()/clean[iA].norm())<=cfg['A_limit']
            legal = legal and float((q[iB]-clean[iB]).norm()/clean[iB].norm())<=cfg['B_limit']
            scores = source_scores(q) if legal else {}
            counts = {k:int(s.argmax(1).eq(labels).sum()) for k,s in scores.items()}
            legal = legal and counts['native']>=initial_counts['native'] and all(
                n<=initial_counts[k] for k,n in counts.items() if k!='native')
        if legal:
            pair = objectives(q)
            if min(pair)>min(current)+1e-10:
                v,current,accepted = trial.detach(),pair,True
                break
    trace.append(dict(step=step+1,active=active,accepted=accepted,backtracks=backtrack,
                      objectives=current,seconds=time.monotonic()-start))
    save_json(out/'progress.json',trace)
    print(trace[-1],flush=True)
    if not accepted:
        break
with torch.no_grad():
    p,_ = parameters(v)
    p = tuple(t.float().double() if i in (iA,ia,iB,ib) else t for i,t in enumerate(p))
    scores = source_scores(p)
    with torch.enable_grad():
        w = env['recovered_forward'](p)
    scores['distilled'] = env['recovery_scores'](w)
    counts = {k:int(s.argmax(1).eq(env['labels'] if k=='distilled' else labels).sum()) for k,s in scores.items()}
    error = float(actual_error(p))
    eligible = any(r['accepted'] for r in trace) and counts['native']>=initial_counts['native'] and error<=tau
    eligible = eligible and counts['distilled']<len(env['labels']) and all(
        n<=initial_counts[k] for k,n in counts.items() if k not in ('native','distilled'))
    publication = None
    if any(r['accepted'] for r in trace):
        edits={f'model.visual.transformer.resblocks.11.{env["names"][i]}':p[i].float().cpu() for i in (iA,ia,iB,ib)}
        torch.save(edits,out/'point.pt')
        publication=dict(path=str(out/'point.pt'),sha256=digest(out/'point.pt'))
    torch.save(dict(scores={k:s.cpu() for k,s in scores.items()},labels=labels.cpu(),
                    recovery_labels=env['labels'].cpu()),out/'scores.pt')
save_json(out/'result.json',dict(status='source_completed',eligible=eligible,publication=publication,
          initial=initial,trace=trace,counts=counts,error=error,limit=tau,seconds=time.monotonic()-start,
          real_construction_images=0,expert_updates=0,source_sha256=digest(__file__)))
print('SOURCE_COMPLETED',eligible,counts,flush=True)
