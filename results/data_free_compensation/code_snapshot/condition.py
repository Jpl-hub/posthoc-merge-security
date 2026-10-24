"""Read-only finite-sample lower bound for the already closed quadratic subproblem."""
import hashlib,json,sys,time
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/root/autodl-tmp/merge_security_20261009');sys.path.insert(0,str(ROOT))
from _impl.io import digest,save_json
start=time.monotonic();torch.set_num_threads(4)
out=ROOT/'outputs/signed_function_20261024';assert not (out/'condition.json').exists()
assert digest(out/'statistics.pt')=='d07245c70b67e8e49b97a500194daf0a5dfb12ac1bf2e2322fe65f43eccf368a'
a=torch.load(out/'statistics.pt',map_location='cpu',weights_only=False)
b=torch.load(ROOT/'outputs/phase2_task_subspace_comparison_20261020/basis.pt',map_location='cpu',weights_only=False)['73']
C=a['source_native_covariance'].numpy();C=(C+C.T)/2;c=a['source_task_cross'].numpy();V=b['v'].numpy();q=.5;energy=a['source_task_energy']
# C is the finite source covariance,not the unknown task-data covariance.
np.linalg.cholesky(C)
X=np.linalg.solve(C,np.column_stack([c,V]));x,T=X[:,0],X[:,1:]
d=x-T@np.linalg.solve(V.T@T,V.T@x)
capacity=float(c@d);norm=float(np.sqrt(d@C@d));res=C@d-c;res-=V@(V.T@res)
error=float(np.linalg.norm(res)/np.linalg.norm(c));constraint=float(np.linalg.norm(V.T@d)/np.linalg.norm(d))
assert capacity>0 and error<1e-6 and constraint<1e-8
np.testing.assert_allclose(capacity,d@C@d,rtol=1e-6,atol=1e-12)
minimum_zero_cross=energy/(q*np.sqrt(capacity));minimum_reverse_cross=2*minimum_zero_cross
w=a['w'].numpy();current_native=float(np.sqrt(w@C@w));current_cross=1+q*float(c@w)/energy
regularity=float(np.linalg.norm(C,ord=np.inf)*np.linalg.norm(X[:,0],ord=np.inf)/max(np.linalg.norm(c,ord=np.inf),1e-300))
result=dict(status='finite_source_lower_bound_checked_not_a_new_point',formula='min sqrt(w^T C_native w) subject V_B^T w=0 and E[tau*(tau+q*s_m)]<=0 equals E[tau^2]/(q*sqrt(c^T D c)),D=C^-1-C^-1 V(V^T C^-1 V)^-1 V^T C^-1',source_statistic_sha256=digest(out/'statistics.pt'),source=Path(__file__).read_text(),source_sha256=digest(__file__),numpy_projected_stationarity_error=error,task_action_error=constraint,source_task_increment_rms=float(np.sqrt(energy)),minimum_native_RMS_for_zero_average_cross=minimum_zero_cross,minimum_native_RMS_for_minus_one_average_cross=minimum_reverse_cross,current_fixed_solution_source_native_RMS=current_native,current_fixed_solution_source_signed_cross=current_cross,capacity=capacity,positive_definite_finite_source_covariance=True,solve_residual_amplification_indicator=regularity,new_parameter_points=0,new_releases=0,real_images=0,new_model_forwards=0,seconds=time.monotonic()-start,scope='All3072Bcolumns with fixed A and32task equalities,on this finite source only;unrestricted norm in the lower-bound problem. No unknown-task/no-partner impossibility,no new coefficient fit or retest of the closed point.')
save_json(out/'condition.json',result)
print(json.dumps({k:v for k,v in result.items() if k!='source'}))
