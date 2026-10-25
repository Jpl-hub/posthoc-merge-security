"""Read the fixed target classifier on already frozen generated features."""
import hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from _impl.models import initialize,load_checkpoint
from _impl.io import digest
initialize();start=time.perf_counter()
root=Path('outputs/frozen_classifier_20261025');root.mkdir(exist_ok=False)
refs={'outputs/signed_function_control_20261024/full_readout.pt':'9187d189c376a0cb3390334df1baaa5bf097154820c1f2b92b0cd8be900312f1','outputs/2026-10-24_research/task_function_probe.pt':'129fb2c7dde6b530b9af05b8c62555b47e59859cd34efaa676530ee0cfeab57a'}
for p,sha in refs.items():assert digest(p)==sha
raw=torch.load(list(refs)[0],map_location='cpu',weights_only=False);probe=torch.load(list(refs)[1],map_location='cpu',weights_only=False)
head=load_checkpoint('head_MNIST.pt');W=head.weight.double().numpy();bias=head.bias.double().numpy();C=W-W.mean(0,keepdims=True)
_,s,Vh=np.linalg.svd(C,full_matrices=False);rank=int((s>s[0]*1e-12).sum());Q=Vh[:rank].T;V=probe['basis'].double().numpy()
assert np.max(abs(Q.T@Q-np.eye(rank)))<1e-12
records={};replay_error=0.
for name,data in raw['captured'].items():
 f=data['features'];f=f/np.linalg.norm(f,axis=1,keepdims=True);scores=f@W.T+bias
 with torch.no_grad():actual=head(torch.from_numpy(data['features']).float()).numpy()
 replay_error=max(replay_error,float(np.max(abs(scores-actual))))
 rows={}
 for tag,sl in [('initial',slice(0,64)),('task_generated',slice(64,128))]:
  z=scores[sl];y=z.argmax(1);sorted_z=np.sort(z,axis=1);gap=sorted_z[:,-1]-sorted_z[:,-2];prob=np.exp(z-z.max(1,keepdims=True));prob/=prob.sum(1,keepdims=True)
  rows[tag]={'prediction_histogram':np.bincount(y,minlength=len(W)).tolist(),'represented_predictions':int(np.unique(y).size),'top_two_gap_min_mean_max':[float(gap.min()),float(gap.mean()),float(gap.max())],'mean_maximum_probability':float(prob.max(1).mean())}
 records[name]=rows
pairs={}
for name in ['native','mid']:
 a=raw['captured'][name]['features'];b=raw['captured'][name+'_changed']['features'];a=a/np.linalg.norm(a,axis=1,keepdims=True);b=b/np.linalg.norm(b,axis=1,keepdims=True);delta=b-a
 before=a@W.T+bias;after=b@W.T+bias;rows={}
 for tag,sl in [('initial',slice(0,64)),('task_generated',slice(64,128))]:
  dd=delta[sl];energy=np.sum(dd*dd);z0=before[sl];z1=after[sl];y=z0.argmax(1);gap=z0[np.arange(64),y,None]-z0;dg=(z1-z0)[np.arange(64),y,None]-(z1-z0);keep=np.arange(len(W))[None,:]!=y[:,None]
  rows[tag]={'changed_predictions':int(np.sum(z1.argmax(1)!=y)),'changed_by_teacher_prediction':np.bincount(y[z1.argmax(1)!=y],minlength=len(W)).tolist(),'classifier_contrast_subspace_delta_energy_fraction':float(np.sum((dd@Q)**2)/energy),'class_centered_score_delta_RMS':float(np.sqrt(np.mean((dd@C.T)**2))),'relative_all_competitor_gap_change_max':float(np.max(abs(dg[keep]/gap[keep])))}
 pairs[name]=rows
assert replay_error<5e-4
source=Path(__file__).read_text();result={'status':'read_only_existing_generated_feature_classification_diagnosis','inputs':refs,'head_sha256':digest('checkpoints/ViT-B-32/head_MNIST.pt'),'head_shape':list(W.shape),'head_bias_maximum':float(abs(bias).max()),'head_contrast_rank':rank,'head_contrast_energy_in_fixed_task_output_probes':float(np.sum((C@V)**2)/np.sum(C*C)),'principal_cosines_head_contrast_to_task_probes':np.linalg.svd(Q.T@V,compute_uv=False).tolist(),'records':records,'changes':pairs,'NumPy_Torch_score_max_error':replay_error,'seconds':time.perf_counter()-start,'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'real_images':0,'new_model_forwards':0,'new_AGA_calls':0,'new_publications':0,'parameter_updates':0,'head_used_for':'Evaluation only of the existing fixed point and fixed generated inputs;no construction,selection or repair','scope':'Generated images have no natural ground truth. Prediction coverage and confidence are not test accuracy or natural-task coverage.'}
(root/'class_readout.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k!='source'},indent=2))
