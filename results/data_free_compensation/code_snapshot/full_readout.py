"""Read-only complete-model check of the closed rank-one B reference."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from _impl.io import digest
from _impl.models import initialize,load_checkpoint
from _impl.merge import merge
initialize(); start=time.perf_counter()
out=Path('outputs/signed_function_control_20261024/full_readout.json')
assert not out.exists()
refs={
 'outputs/2026-10-24_research/task_function_probe.pt':'129fb2c7dde6b530b9af05b8c62555b47e59859cd34efaa676530ee0cfeab57a',
 'outputs/signed_function_control_20261024/release.pt':'787c235df28214ea18b659ead85c9233b9075d7e55fe40440bbd70a8661e08fc',
 'outputs/feature_noise/artifacts/original_AGA.pt':'07195e779ea2a8203aaf2ec7bccc651d00460c3a966810bc04342af5e5088607',
 'outputs/signed_function_control_20261024/AGA/edits.pt':'56aa72f0ff8ff2bfe4c2f10a333b24d0e329637a734b28aa70ac2b1b0eb3ae91',
 'outputs/signed_function_20261024/statistics.pt':'d07245c70b67e8e49b97a500194daf0a5dfb12ac1bf2e2322fe65f43eccf368a'}
for p,sha in refs.items(): assert digest(p)==sha,p
probe=torch.load(next(iter(refs)),map_location='cpu',weights_only=False)
source=torch.load('outputs/signed_function_20261024/statistics.pt',map_location='cpu',weights_only=False)
v=source['v'].double().numpy(); assert v.shape==(768,)
base=load_checkpoint('zeroshot.pt'); original=load_checkpoint('MNIST/finetuned.pt')
released=copy.deepcopy(original);released.load_state_dict(original.state_dict()|torch.load(list(refs)[1],weights_only=False),strict=True)
aligned=copy.deepcopy(original);aligned.load_state_dict(torch.load(list(refs)[2],weights_only=False),strict=True)
changed=copy.deepcopy(aligned);changed.load_state_dict(aligned.state_dict()|torch.load(list(refs)[3],weights_only=False),strict=True)
mid=merge(base,aligned,base,.5);mid_changed=merge(base,changed,base,.5)
latent=torch.cat([probe['initial_latent'],probe['final_latent']])
fbase=torch.cat([probe['initial_features'][1],probe['final_features'][1]]).double().numpy()
normalizer=original.val_preprocess.transforms[-1]
mean=torch.tensor(normalizer.mean)[None,:,None,None].cuda();std=torch.tensor(normalizer.std)[None,:,None,None].cuda()
@torch.no_grad()
def capture(model):
 rows={k:[] for k in ['h','r','features']}; visual=model.model.visual
 handles=[visual.transformer.resblocks[11].mlp.c_proj.register_forward_pre_hook(lambda m,args:rows['h'].append(args[0][0].detach().cpu())),visual.ln_post.register_forward_pre_hook(lambda m,args:rows['r'].append(args[0].detach().cpu()))]
 try:
  model.cuda().eval()
  for z in latent.split(4):
   pixels=F.interpolate(z.cuda(),size=(224,224),mode='bilinear',align_corners=False)
   rows['features'].append(F.normalize(model((pixels-mean)/std),dim=-1).cpu())
 finally:
  for h in handles:h.remove()
  model.cpu()
 return {k:torch.cat(a).double().numpy() for k,a in rows.items()}
raw={}
for name,m in [('native',original),('native_changed',released),('mid',mid),('mid_changed',mid_changed)]:
 raw[name]=capture(m);print('CAPTURE',name,flush=True)
def unit(x):return x/np.linalg.norm(x,axis=1,keepdims=True)
def rms(x):return float(np.sqrt(np.mean(np.sum(x*x,axis=-1))))
def tail(r,m):
 z=r-r.mean(-1,keepdims=True);sigma=np.sqrt(np.mean(z*z,axis=-1,keepdims=True)+m.ln_post.eps)
 g=m.ln_post.weight.double().numpy();b=m.ln_post.bias.double().numpy();p=m.proj.double().numpy()
 y=(z/sigma*g+b)@p
 return unit(y),y,z,sigma,g,b,p
def stats(a):return dict(min=float(np.min(a)),median=float(np.median(a)),mean=float(np.mean(a)),max=float(np.max(a)))
summary={}; all_arrays={}
for name,m,mc in [('native',original,released),('mid',mid,mid_changed)]:
 a,b=raw[name],raw[name+'_changed'];f,y,z,s,g,beta,p=tail(a['r'],m.model.visual);fc,yc,zc,sc,*_=tail(b['r'],mc.model.visual)
 h_error=float(np.max(np.abs(a['h']-b['h'])));assert h_error==0
 key='model.visual.transformer.resblocks.11.mlp.c_proj.'
 dB=(mc.state_dict()[key+'weight']-m.state_dict()[key+'weight']).double().numpy()
 db=(mc.state_dict()[key+'bias']-m.state_dict()[key+'bias']).double().numpy()
 dr=a['h']@dB.T+db
 scalar=dr@v/(v@v);rank_error=rms(dr-scalar[:,None]*v)/max(rms(dr),1e-30)
 k=((v-v.mean())*g)@p;c=beta@p;Q,svals,_=np.linalg.svd(np.column_stack([k,c]),full_matrices=False)
 Q=Q[:,svals>svals[0]*1e-12]
 alpha=(s/sc).reshape(-1);pred_y=alpha[:,None]*y+scalar[:,None]/sc*k+(1-alpha[:,None])*c
 pred_f=unit(pred_y)
 outside=f-(f@Q)@Q.T;outside_new=fc-(fc@Q)@Q.T
 ratio=alpha*np.linalg.norm(y,axis=1)/np.linalg.norm(yc,axis=1)
 span_error=rms(outside_new-ratio[:,None]*outside)
 projection=np.linalg.norm(f@Q,axis=1);lower=np.sqrt(np.maximum(0,2-2*projection))
 readout_error=max(float(np.max(np.abs(f-a['features']))),float(np.max(np.abs(fc-b['features']))))
 residual_error=float(np.max(np.abs(b['r']-a['r']-dr)))
 assert readout_error<5e-4 and residual_error<5e-4,(readout_error,residual_error)
 task=f-fbase;d=fc-f
 per={}
 for label,sl in [('initial',slice(0,64)),('task_generated',slice(64,128))]:
  t,dd=task[sl],d[sl];tc=t-t.mean(0);dc=dd-dd.mean(0)
  per[label]=dict(complete_feature_delta_RMS=rms(dd),task_increment_RMS=rms(t),complete_task_cross_ratio=float(1+np.sum(t*dd)/np.sum(t*t)),centered_task_cross_ratio=float(1+np.sum(tc*dc)/np.sum(tc*tc)),task_relative_delta_RMS=rms(dd)/rms(t),MLP_residual_delta_RMS=rms(dr[sl]),antipodal_error_RMS=rms(fc[sl]+f[sl]),antipodal_lower_bound=stats(lower[sl]),outside_plane_energy_fraction=stats(np.sum(outside[sl]**2,axis=1)),positive_outside_ratio=stats(ratio[sl]))
 summary[name]=dict(synthetic_sets=per,unchanged_upstream_hidden_max_error=h_error,FP32_rank_one_relative_error=rank_error,complete_FP32_vs_NumPy_readout_max_error=readout_error,FP32_MLP_residual_identity_max_error=residual_error,finite_rank_one_formula_max_error=float(np.max(np.abs(pred_f-fc))),positive_outside_relation_RMS_error=span_error,readout_plane_rank=Q.shape[1],readout_plane_singular_values=svals.tolist())
 all_arrays[name]=dict(task=task,delta_features=d,raw_residual_change=dr,scalar=scalar,readout_plane=Q,lower_bound=lower,predicted_features=pred_f,readout_features=f,changed_readout_features=fc,outside_scale=ratio)
source_text=Path(__file__).read_text();raw_path=out.with_suffix('.pt')
torch.save(dict(captured=raw,derived=all_arrays),raw_path)
result=dict(status='completed_read_only_full_prefix_and_final_readout_check',source_sha256=hashlib.sha256(source_text.encode()).hexdigest(),source=source_text,inputs=refs,results=summary,raw_path=str(raw_path),raw_sha256=digest(raw_path),seconds=time.perf_counter()-start,real_images=0,classification_heads=0,partner_models=0,new_AGA_calls=0,new_publications=0,model_parameter_updates=0,generated_image_model_forwards=512,upstream_models_frozen=True,scope='Only the existing rank-one finalMLP reference and public-base midpoint;not actual unknown-partner fusion or a classification result. No point or generated input refit.')
assert all(not q.requires_grad and q.grad is None for m in [base,original,released,aligned,changed,mid,mid_changed] for q in m.parameters())
out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:w for k,w in result.items() if k!='source'},indent=2),flush=True)
