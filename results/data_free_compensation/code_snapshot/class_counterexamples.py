"""Read-only common-class counterexamples for the frozen069e point;no model repair."""
import copy,hashlib,json,time
from _impl.merge import merge
from pathlib import Path
import torch
import torch.nn.functional as F
from _impl.models import initialize,load_checkpoint
from _impl.io import digest
initialize();torch.manual_seed(2026102505);start=time.perf_counter()
out=Path('outputs/frozen_classifier_20261025/frozen_counterexamples');out.mkdir(parents=True,exist_ok=False)
expert=load_checkpoint('MNIST/finetuned.pt').cuda().eval();head=load_checkpoint('head_MNIST.pt').cuda().eval()
base=load_checkpoint('zeroshot.pt');original_cpu=copy.deepcopy(expert).cpu();point=Path('outputs/frozen_classifier_20261025/class_coupling/point.pt');aga=Path('outputs/frozen_classifier_20261025/class_coupling/AGA/edits.pt');assert digest(point)=='069e6091c34fe6f88f0c9b6c1044204228b096e68d9dae33dd6f8bf03e8236de';assert digest(aga)=='da94e988182b684b4daa5b89eb6a24cfab4ee1d921eed531915734cddeb4ad54'
native=copy.deepcopy(original_cpu);native.load_state_dict(native.state_dict()|torch.load(point,map_location='cpu',weights_only=False),strict=True);aligned=copy.deepcopy(original_cpu);aligned.load_state_dict(torch.load('outputs/feature_noise/artifacts/original_AGA.pt',map_location='cpu',weights_only=False),strict=True);aligned.load_state_dict(aligned.state_dict()|torch.load(aga,map_location='cpu',weights_only=False),strict=True)
models={'teacher':expert,'native':native.cuda(),'direct':merge(base,native.cpu(),base,.5).cuda(),'recovered':merge(base,aligned,base,.5).cuda()};native.cuda()
normalizer=expert.val_preprocess.transforms[-1];mean=torch.tensor(normalizer.mean,device='cuda')[None,:,None,None];std=torch.tensor(normalizer.std,device='cuda')[None,:,None,None]
labels=torch.arange(1,10).repeat_interleave(8);initial=torch.rand(72,3,32,32);final=initial.clone();shifts=[(0,0),(7,0),(-7,0),(0,7),(0,-7)];trace=[]
def views(z):
 x=F.interpolate(z,size=(224,224),mode='bilinear',align_corners=False)
 return torch.cat([torch.roll(x,shifts=s,dims=(2,3)) for s in shifts])
for offset in range(0,72,8):
 z=initial[offset:offset+8].cuda().clone().requires_grad_(True);y=labels[offset:offset+8].cuda();opt=torch.optim.Adam([z],lr=.05)
 for step in range(128):
  opt.zero_grad(set_to_none=True);x=(views(z)-mean)/std;losses=torch.cat([F.cross_entropy(head(m(x)),y.repeat(5),reduction='none').reshape(5,len(y)) for m in models.values()],dim=0);loss=losses.max(0).values.mean()
  assert torch.isfinite(loss);loss.backward();assert torch.isfinite(z.grad).all();opt.step()
  with torch.no_grad():z.clamp_(0,1)
 final[offset:offset+8]=z.detach().cpu();trace.append({'offset':offset,'last_preupdate_worst_CE':float(loss.detach())});print('CLASS_SOURCE',offset,trace[-1],flush=True)
records={};raw={'initial_latent':initial,'final_latent':final,'requested_classes':labels,'shifts':shifts};score_map={}
with torch.no_grad():
 for name,m in models.items():
  scores=[]
  for z in final.split(8):scores.append(head(m((views(z.cuda())-mean)/std)).reshape(5,len(z),10).cpu())
  scores=torch.cat(scores,dim=1);score_map[name]=scores;pred=scores.argmax(-1);records[name]={'requested_view_matches':int(pred.eq(labels[None]).sum()),'all_five_views_requested_inputs':int(pred.eq(labels[None]).all(0).sum()),'prediction_histogram':torch.bincount(pred.flatten(),minlength=10).tolist()}
 joint=torch.stack([v.argmax(-1).eq(labels[None]) for v in score_map.values()]).all(0);raw['scores']=score_map;torch.save(raw,out/'counterexamples.pt');source=Path(__file__).read_text()
result={'status':'frozen_point_counterexample_search_complete','seed':2026102505,'point_sha256':digest(point),'actual_AGA_edits_sha256':digest(aga),'inputs':72,'requested_classes':'1through9,8each;allfinalinputskept','views':5,'fixed_pixel_updates':128,'models':list(models),'loss':'Minimize per-input maximum requested-class CE over all4frozenmodels and5views;no trained/real data','records':records,'all_four_models_and_five_views_requested_inputs':int(joint.all(0).sum()),'all_four_models_requested_view_matches':int(joint.sum()),'source_trace':trace,'real_images':0,'new_model_points':0,'parameter_updates':0,'new_AGA_calls':0,'generated_image_model_forwards':185760,'pixel_optimizer_steps':1152,'seconds':time.perf_counter()-start,'raw_path':str(out/'counterexamples.pt'),'raw_sha256':digest(out/'counterexamples.pt'),'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'limits':'Read-only diagnosis of closed069e;not new defense,natural examples,attack count or inputs for repairing thispoint. All critic results retained;no steps/seed/class selection.'}
assert all(not p.requires_grad and p.grad is None for m in models.values() for p in m.parameters())
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source','source_trace']},indent=2))
