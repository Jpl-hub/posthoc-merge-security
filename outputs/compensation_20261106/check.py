"""Read-only diagnosis of the failed finite-difference check, same point/step."""
import ast
import json
import time
from pathlib import Path

import torch
from torch.func import grad, jvp

from _impl.io import digest, save_json


protocol = json.loads(Path('results/data_free_compensation/compensation_derivative_diagnosis.json').read_text())
assert digest(__file__) == protocol['source_sha256']
out = Path('outputs/compensation_20261106/diagnosis')
out.mkdir(exist_ok=False)
start = time.monotonic()
source = Path('outputs/compensation_20261106/run.py')
tree = ast.parse(source.read_text())
nodes = []
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'v' for t in node.targets):
        break
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'out' for t in node.targets):
        continue
    if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
        f = node.value.func
        if isinstance(f, ast.Name) and f.id == 'save_json' or isinstance(f, ast.Attribute) and f.attr == 'mkdir':
            continue
    nodes.append(node)
env = dict(out=out, __name__='read_only_compensation_check')
exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), env)
variable = torch.zeros_like(env['theta'][env['iA']], requires_grad=True)
p, _ = env['parameters'](variable)
measure = lambda w: torch.stack((env['margin'](env['env']['recovery_scores'](w), env['env']['labels']),))
value, gp, _ = env['env']['recover_gradient'](env['env']['inner_loss'], measure, env['anchor'],
    tuple(t.detach() for t in p), lambda q: env['env']['call'](q, env['env']['xt']))
indices = [env[k] for k in ('iA', 'ia', 'iB', 'ib')]
gradient, = torch.autograd.grad([p[i] for i in indices], variable, grad_outputs=[gp[i] for i in indices])
direction = gradient / gradient.norm()
theta, tangent = jvp(lambda v: env['parameters'](v)[0], (variable.detach(),), (direction,))
teacher, dteacher = jvp(lambda p: env['env']['call'](p, env['env']['xt']), (theta,), (tangent,))
reverse = float((gradient * direction).sum())
chain = float(sum((g*d).sum() for g,d in zip(gp,tangent)))
print('REVERSE', reverse, chain, flush=True)

# Run the existing independently implemented forward tangent of the100 Adam steps.
path = Path('results/data_free_compensation/synthetic_recovery/differentiate.py')
old = ast.parse(path.read_text())
main = next(n for n in old.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
lo = next(i for i,n in enumerate(main.body) if isinstance(n, ast.Assign) and any(
    isinstance(t, ast.Name) and t.id == 'alphas' for t in n.targets))
hi = next(i for i,n in enumerate(main.body) if isinstance(n, ast.FunctionDef) and n.name == 'reference')
forward = dict(torch=torch, grad=grad, jvp=jvp, time=time, start=start, out=out,
    save_json=save_json, loss=env['env']['inner_loss'], anchor=env['anchor'], theta=theta,
    delta=tuple(p-b for p,b in zip(theta,env['anchor'])), direction=tangent,
    teacher=teacher, dteacher=dteacher, measure=measure)
exec(compile(ast.Module(body=main.body[lo:hi], type_ignores=[]), str(path), 'exec'), forward)
exact = float(forward['exact'][0])
scores, dscores = jvp(env['env']['recovery_scores'], (forward['weights'],), (forward['total'],))
labels = env['env']['labels']
eps = env['cfg']['gradient_step']
linear = [scores + sign*eps*dscores for sign in [-1,1]]
actual, parameter_points = [], []
for sign in [-1,1]:
    with torch.no_grad():
        q, _ = env['parameters'](variable.detach()+sign*eps*direction)
    w = env['env']['recovered_forward'](q)
    actual.append(env['env']['recovery_scores'](w).detach())
    parameter_points.append(q)
margin = env['margin']
finite = float((margin(actual[1],labels)-margin(actual[0],labels))/(2*eps))
linear_finite = float((margin(linear[1],labels)-margin(linear[0],labels))/(2*eps))
competitor = lambda s:s.scatter(1,labels[:,None],-torch.inf).argmax(1)
curvature = {env['env']['names'][i]:float(((parameter_points[1][i]-parameter_points[0][i])/(2*eps)-tangent[i]).norm() /
             tangent[i].norm().clamp_min(1e-30)) for i in indices}
torch.save(dict(scores=scores.cpu(),score_tangent=dscores.cpu(),labels=labels.cpu(),
           actual=[s.cpu() for s in actual],linearized=[s.cpu() for s in linear]),out/'scores.pt')
save_json(out/'result.json',dict(reverse=reverse,parameter_chain=chain,forward=exact,
    forward_reverse_relative_error=abs(reverse-exact)/(1+abs(reverse)),finite=finite,
    linearized_score_finite=linear_finite,competitor_changes=[int(competitor(s).ne(competitor(scores)).sum()) for s in actual],
    parameter_finite_errors=curvature,seconds=time.monotonic()-start,defense_updates=0,new_defense_models=0,
    source_sha256=digest(__file__),raw_sha256=digest(out/'scores.pt')))
