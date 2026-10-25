"""Independent generated-input evaluation of one frozen point and actual authorAGA."""
import copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from _impl.io import digest
from _impl.models import initialize,load_checkpoint
from _impl.merge import merge
initialize();start=time.perf_counter();root=Path('outputs/frozen_classifier_20261025');out=root/'generated_confirmation'/'point_evaluation.json';assert not out.exists()
meta=json.loads((root/'generated_confirmation/results.json').read_text());assert meta['source_sha256']==json.loads(Path('results/data_free_compensation/joint.json').read_text())['classifier_coupling_plan']['generated_confirmation_source_sha256']
gen=torch.load(meta['raw_path'],map_location='cpu',weights_only=False);assert digest(meta['raw_path'])==meta['raw_sha256']
point=root/'class_coupling/point.pt';aga=root/'class_coupling/AGA/edits.pt';assert digest(point)=='069e6091c34fe6f88f0c9b6c1044204228b096e68d9dae33dd6f8bf03e8236de';assert digest(aga)=='da94e988182b684b4daa5b89eb6a24cfab4ee1d921eed531915734cddeb4ad54'
base=load_checkpoint('zeroshot.pt');e=load_checkpoint('MNIST/finetuned.pt');head=load_checkpoint('head_MNIST.pt').cuda();native=copy.deepcopy(e);native.load_state_dict(e.state_dict()|torch.load(point,map_location='cpu',weights_only=False),strict=True)
aligned=copy.deepcopy(e);aligned.load_state_dict(torch.load('outputs/feature_noise/artifacts/original_AGA.pt',map_location='cpu',weights_only=False),strict=True);aligned.load_state_dict(aligned.state_dict()|torch.load(aga,map_location='cpu',weights_only=False),strict=True)
models={'native':native,'direct':merge(base,native,base,.5),'recovered':merge(base,aligned,base,.5)};norm=e.val_preprocess.transforms[-1];mean=torch.tensor(norm.mean,device='cuda')[None,:,None,None];std=torch.tensor(norm.std,device='cuda')[None,:,None,None]
raw={'teacher':gen['scores'],'requested_classes':gen['requested_classes'],'input_sha256':meta['raw_sha256']};records={}
with torch.no_grad():
 for name,m in models.items():
  m.cuda().eval();items=[]
  for z in gen['final_latent'].split(8):
   p=F.interpolate(z.cuda(),size=(224,224),mode='bilinear',align_corners=False);x=torch.cat([torch.roll(p,shifts=s,dims=(2,3)) for s in gen['shifts']]);items.append(head(m((x-mean)/std)).reshape(5,len(z),10).cpu())
  m.cpu();scores=torch.cat(items,dim=1);raw[name]=scores;pred=scores.argmax(-1);before=gen['scores'].argmax(-1);changed=pred.ne(before);hist=torch.bincount(pred.flatten(),minlength=10).tolist();byclass=[int((changed&before.eq(k)).sum()) for k in range(10)]
  np.testing.assert_array_equal(scores.numpy().argmax(-1),pred.numpy());records[name]={'n':800,'changed_from_teacher':int(changed.sum()),'class0':int(pred.eq(0).sum()),'prediction_histogram':hist,'changed_by_teacher_prediction':byclass,'changed_each_view':changed.sum(1).tolist(),'all_views_same_as_teacher_inputs':int((~changed).all(0).sum()),'NumPy_counts_match':int(np.sum(scores.numpy().argmax(-1)!=gen['scores'].numpy().argmax(-1)))==int(changed.sum())};assert records[name]['NumPy_counts_match'];print(name,records[name],flush=True)
 torch.save(raw,out.with_suffix('.pt'));source=Path(__file__).read_text();result={'status':'independent_generated_confirmation_complete','point_sha256':digest(point),'actual_AGA_edits_sha256':digest(aga),'input_sha256':meta['raw_sha256'],'generator_seed':meta['seed'],'generator_requested_allviews_correct':meta['all_requested_views_correct'],'records':records,'real_images':0,'new_full_AGA_calls':0,'parameter_updates':0,'generated_image_model_forwards':2400,'seconds':time.perf_counter()-start,'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'raw_path':str(out.with_suffix('.pt')),'raw_sha256':digest(out.with_suffix('.pt')),'scope':'Newgeneratedteacherprototypes afterfreeze,allkept. Notnaturaltruth/accuracy or actualpartnerfusion. No constructionfeedback or point selection.'};out.write_text(json.dumps(result,indent=2)+'\n')
