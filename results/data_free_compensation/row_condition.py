"""One weight-only equality condition; it does not construct a defense model."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn.functional as F

from _impl.attacks import load_author
from _impl.io import digest
from _impl.models import initialize, load_checkpoint

initialize()
start = time.perf_counter()
path = Path('results/data_free_compensation/row_condition_protocol.json')
cfg = json.loads(path.read_text())
out = Path('outputs/row_condition_20261025')
out.mkdir(exist_ok=False)
(out/'protocol.json').write_text(json.dumps(cfg,indent=2)+'\n')
base, expert = load_checkpoint('zeroshot.pt'), load_checkpoint('MNIST/finetuned.pt')
key = 'model.visual.transformer.resblocks.11.mlp.'
A_cpu, A0_cpu = [m.state_dict()[key+'c_fc.weight'] for m in [expert,base]]
A, A0 = A_cpu.double().cuda(), A0_cpu.double().cuda()
assert digest(cfg['basis']['path']) == cfg['basis']['sha256']
basis = torch.load(cfg['basis']['path'],map_location='cpu',weights_only=False)
basis = next(x for x in basis.values() if x['source_key'] == key+'c_fc.weight')
u0, v0 = basis['u'].double(), basis['v'].double()
assert u0.shape[1] == v0.shape[1] == 32
U = torch.linalg.qr(u0.cuda(),mode='reduced').Q
V = torch.linalg.qr(v0.cuda(),mode='reduced').Q
PL, PR = torch.eye(len(A),device='cuda',dtype=torch.double)-U@U.T, torch.eye(A.shape[1],device='cuda',dtype=torch.double)-V@V.T
G = PL*(A0@PR@A0.T)
b = -(A*A0).sum(1)
L, info = torch.linalg.cholesky_ex(G)
assert int(info) == 0, int(info)
lam = torch.cholesky_solve(b[:,None],L)[:,0]
delta = PL@(lam[:,None]*(A0@PR))
residual = float((G@lam-b).norm()/b.norm())
norm_identity = float(abs(delta.square().sum()-lam@b)/delta.square().sum())
assert residual < 1e-8 and norm_identity < 1e-8
un, _ = np.linalg.qr(u0.numpy());vn, _ = np.linalg.qr(v0.numpy())
pn, qn = np.eye(len(A))-un@un.T, np.eye(A.shape[1])-vn@vn.T
an, a0n = A_cpu.double().numpy(), A0_cpu.double().numpy()
gn = pn*(a0n@qn@a0n.T);bn = -(an*a0n).sum(1)
ln = scipy.linalg.cho_solve(scipy.linalg.cho_factor(gn,lower=True),bn)
dn = pn@(ln[:,None]*(a0n@qn))
independent = float(np.linalg.norm(delta.cpu().numpy()-dn)/np.linalg.norm(dn))
assert independent < 1e-8
Ap = (A+delta).float().cpu()
actual = Ap.double().cuda()-A
inside_left = float((U.T@actual).norm()/A.norm())
inside_right = float((actual@V).norm()/A.norm())
row_cos = (Ap.double()*A0_cpu.double()).sum(1)/(Ap.double().norm(dim=1)*A0_cpu.double().norm(dim=1))
assert torch.isfinite(row_cos).all() and float(Ap.norm(dim=1).min()) > 0
assert max(inside_left,inside_right)<5e-7 and float(row_cos.abs().max())<1e-5
author, source_hash = load_author(Path(cfg['author_matching']['path']))
assert source_hash == cfg['author_matching']['sha256_text']
order = author['solve_permutation'](Ap,A0_cpu).cpu()
assert sorted(order.tolist()) == list(range(len(A)))
ref = cfg['author_matching']['original_clean_AGA']
assert digest(ref) == cfg['author_matching']['original_clean_AGA_sha256']
clean = torch.load(ref,map_location='cpu',weights_only=False)
row_map = {x.numpy().tobytes():i for i,x in enumerate(A_cpu)}
old = torch.tensor([row_map[x.numpy().tobytes()] for x in clean[key+'c_fc.weight']])
aa, bb = [expert.state_dict()[key+k] for k in ['c_fc.bias','c_proj.weight']]
assert torch.equal(aa[old],clean[key+'c_fc.bias']) and torch.equal(bb[:,old],clean[key+'c_proj.weight'])
aligned = {name:float((new.double()-clean[key+k].double()).norm()/original.double().norm()) for name,k,new,original in [('A','c_fc.weight',Ap[order],A_cpu),('a','c_fc.bias',aa[order],aa),('B','c_proj.weight',bb[:,order],bb)]}
probes = torch.cat([V.T,-V.T])
a = aa.double().cuda();B = bb.double().cuda()
output_bias = expert.state_dict()[key+'c_proj.bias'].double().cuda()
activation = lambda z:z*torch.sigmoid(1.702*z)
before = F.linear(activation(F.linear(probes,A,a)),B,output_bias)
after = F.linear(activation(F.linear(probes,Ap.double().cuda(),a)),B,output_bias)
probe_error = float((before-after).abs().max())
torch.save(dict(delta_A_FP64=delta.cpu(),A_prime_FP32=Ap,lambda_FP64=lam.cpu(),author_assignment=order,original_assignment=old,probe_before=before.cpu(),probe_after=after.cpu()),out/'geometry.pt')
source = Path(__file__).read_text()
result = dict(status='finite_weight_condition_passed_not_a_defense',relative_A_distance=float(actual.norm()/A.norm()),task_left_relative=inside_left,task_right_relative=inside_right,max_corresponding_anchor_cosine=float(row_cos.abs().max()),minimum_row_norm=float(Ap.norm(dim=1).min()),author_reassignments_vs_original=int((order!=old).sum()),rows=len(order),aligned_lastMLP_relative_changes=aligned,finite64probe_max_output_error=probe_error,relative_linear_residual=residual,minimum_norm_KKT_identity_error=norm_identity,independent_numpy_relative_error=independent,geometry_sha256=digest(out/'geometry.pt'),protocol_sha256=digest(path),source=source,source_sha256=hashlib.sha256(source.encode()).hexdigest(),seconds=time.perf_counter()-start,new_image_forwards=0,real_images=0,classhead_used=False,author_lastMLP_matching_calls=1,new_full_AGA_calls=0,new_defense_models=0,limits=cfg['limits'])
(out/'results.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='source'},indent=2))
