"""One fixed class-conditioned generated source; no expert/model optimization."""
import hashlib,json,time
from pathlib import Path
import torch
import torch.nn.functional as F
from _impl.models import initialize,load_checkpoint
from _impl.io import digest
initialize();torch.manual_seed(2026102501);start=time.perf_counter()
out=Path('outputs/frozen_classifier_20261025/generated_source');out.mkdir(parents=True,exist_ok=False)
expert=load_checkpoint('MNIST/finetuned.pt').cuda().eval();head=load_checkpoint('head_MNIST.pt').cuda().eval()
normalizer=expert.val_preprocess.transforms[-1];mean=torch.tensor(normalizer.mean,device='cuda')[None,:,None,None];std=torch.tensor(normalizer.std,device='cuda')[None,:,None,None]
labels=torch.arange(10).repeat_interleave(16);initial=torch.rand(160,3,32,32);final=initial.clone();shifts=[(0,0),(7,0),(-7,0),(0,7),(0,-7)];trace=[]
def views(z):
 x=F.interpolate(z,size=(224,224),mode='bilinear',align_corners=False)
 return torch.cat([torch.roll(x,shifts=s,dims=(2,3)) for s in shifts])
for offset in range(0,160,8):
 z=initial[offset:offset+8].cuda().clone().requires_grad_(True);y=labels[offset:offset+8].cuda();opt=torch.optim.Adam([z],lr=.05)
 for step in range(128):
  opt.zero_grad(set_to_none=True);logits=head(expert((views(z)-mean)/std));losses=F.cross_entropy(logits,y.repeat(5),reduction='none').reshape(5,len(y));loss=losses.max(0).values.mean()
  assert torch.isfinite(loss);loss.backward();assert torch.isfinite(z.grad).all();opt.step()
  with torch.no_grad():z.clamp_(0,1)
 final[offset:offset+8]=z.detach().cpu();trace.append({'offset':offset,'last_preupdate_worst_CE':float(loss.detach())});print('CLASS_SOURCE',offset,trace[-1],flush=True)
scores=[];features=[]
with torch.no_grad():
 for z in final.split(8):
  f=expert((views(z.cuda())-mean)/std);features.append(F.normalize(f,dim=-1).reshape(5,len(z),-1).cpu());scores.append(head(f).reshape(5,len(z),-1).cpu())
 features=torch.cat(features,dim=1);scores=torch.cat(scores,dim=1);pred=scores.argmax(-1);robust=pred.eq(labels[None]).all(0)
assert all(not p.requires_grad and p.grad is None for m in [expert,head] for p in m.parameters())
raw={'initial_latent':initial,'final_latent':final,'requested_classes':labels,'shifts':shifts,'features':features,'scores':scores};torch.save(raw,out/'source.pt');source=Path(__file__).read_text()
result={'status':'fixed_source_generation_complete','information_access':'Frozen target encoder and existing target classification head;uses class IDs,zero real images;not head-free','seed':2026102501,'source_n':160,'requested_per_class':16,'latent_shape':[3,32,32],'upsampling':'bilinear224align_cornersFalse','fixed_shifts':shifts,'pixel_optimizer':'Adam128stepslr.05,project[0,1],only pixels','selection':'Keep every final sample;no best-step/model/test selection,no retries','all_requested_views_correct':int(pred.eq(labels[None]).sum()),'all_views':800,'all_five_views_correct_inputs':int(robust.sum()),'robust_requested_by_class':[int(robust[labels.eq(k)].sum()) for k in range(10)],'unshifted_prediction_histogram':torch.bincount(pred[0],minlength=10).tolist(),'per_shift_requested_correct':pred.eq(labels[None]).sum(1).tolist(),'source_trace':trace,'raw_path':str(out/'source.pt'),'raw_sha256':digest(out/'source.pt'),'checkpoint_sha256':{'MNIST':digest('checkpoints/ViT-B-32/MNIST/finetuned.pt'),'head_MNIST':digest('checkpoints/ViT-B-32/head_MNIST.pt')},'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'seconds':time.perf_counter()-start,'real_images':0,'classification_heads':1,'expert_parameter_updates':0,'new_defense_publications':0,'new_AGA_calls':0,'generated_image_model_forwards':103200,'pixel_optimizer_steps':2560,'limits':'Teacher self-predictions on generated pixels,not natural truth or accuracy. Generation/robustness/ordinaryCE are not new defense mechanisms or novelty. Future A/B construction and complete native/TA/originalAGA checks still required.'}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({k:v for k,v in result.items() if k not in ['source','source_trace']},indent=2))
