import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

start=time.monotonic()
path=Path('outputs/phase2_postactivation_subspace_mnist/diagnostic.pt')
pin='39a3fc2946b89d8b0fab6bd0c45d649abfac6fe4b436e85e28956d257d413827'
assert hashlib.sha256(path.read_bytes()).hexdigest()==pin
raw=torch.load(path,map_location='cpu',weights_only=False)
p={k:v.numpy() if torch.is_tensor(v) else v for k,v in raw['parameters'].items()}
h,z=raw['h'].numpy(),raw['z'].numpy()
linear=p['gamma'][:,None]*p['projection'];linear-=linear.mean(0)
bias=p['beta']@p['projection'];head=p['head']
sd=lambda a:np.sqrt(np.mean((a-a.mean(1,keepdims=True))**2,axis=1)+p['eps'])
sigma=sd(h);u=h@linear+sigma[:,None]*bias
norm=np.linalg.norm(u,axis=1);before=(u/norm[:,None])@head.T
y=before.argmax(1);i=np.arange(len(y));mask=np.ones(before.shape,bool);mask[i,y]=False
gap=before[i,y,None]-before;gap[i,y]=1
numerator=gap*norm[:,None];numerator[i,y]=1
b=bias@head.T;bdiff=b[y,None]-b[None,:]
records=[]
for key,point in raw['points'].items():
    delta=point['delta'].numpy();dh=z@delta.T;after_h=h+dh
    sigma_after=sd(after_h);ua=after_h@linear+sigma_after[:,None]*bias
    na=np.linalg.norm(ua,axis=1);after=(ua/na[:,None])@head.T
    actual=(after[i,y,None]-after)/gap
    q=norm/na;eta=(sigma_after-sigma)[:,None]*bdiff/numerator
    leakage=dh@linear@head.T;linear_relative=(leakage[i,y,None]-leakage)/numerator
    kernel_only=q[:,None]*(1+eta)
    exact=q[:,None]*(1+eta+linear_relative)
    error=float(np.max(abs(exact[mask]-actual[mask])))
    assert error<1e-8
    changed=after.argmax(1)!=y
    assert int(changed.sum())==int((point['source'].numpy().argmax(1)!=y).sum())
    records.append(dict(point=key,n=len(h),complete_identity_error=error,
        kernel_only_error=float(np.max(abs(kernel_only[mask]-actual[mask]))),
        maximum_relative_linear_term=float(np.max(abs(linear_relative[mask]))),
        q_range=[float(q.min()),float(q.max())],
        worst_task_drift=float(np.max(abs(actual[mask]-1))),changed=int(changed.sum()),
        function_constraint_violations=int(((abs(actual-1)>.5)&mask).any(1).sum()),
        finite_variance_flip_inputs=int(((eta < -1)&mask).any(1).sum())))
report=dict(status='all_six_source_point_identities_passed',source_raw_sha256=pin,
    scope='Only1024 frozen source inputs; no confirmation, test, peer, new direction or repair.',
    records=records,wall_seconds=time.monotonic()-start,new_images=0,gradients=0,expert_updates=0,
    GPU_used=False,checker_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
Path('results/postactivation_subspace/margin_identity.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
