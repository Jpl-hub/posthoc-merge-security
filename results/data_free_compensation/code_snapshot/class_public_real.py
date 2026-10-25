"""Read-only source-proxy versus natural-input diagnosis of the same frozen point."""
import copy,hashlib,importlib.util,json,time
from pathlib import Path
import numpy as np
import torch
from _impl.models import initialize,load_checkpoint
from _impl.io import digest,save_json
from _impl.merge import merge
from _impl.data import dataset,make_loader,official_labels
from _impl.evaluate import evaluate
from _impl.metrics import row_metrics
initialize();start=time.perf_counter();root=Path('outputs/frozen_classifier_20261025/class_coupling');out=root/'public_real';out.mkdir(exist_ok=False)
assert digest(root/'point.pt')=='069e6091c34fe6f88f0c9b6c1044204228b096e68d9dae33dd6f8bf03e8236de';assert digest(root/'AGA/edits.pt')=='da94e988182b684b4daa5b89eb6a24cfab4ee1d921eed531915734cddeb4ad54'
spec=importlib.util.spec_from_file_location('frozen_native_check',root/'class_native.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
base=load_checkpoint('zeroshot.pt');e=load_checkpoint('MNIST/finetuned.pt');head=load_checkpoint('head_MNIST.pt');p=copy.deepcopy(e);p.load_state_dict(e.state_dict()|torch.load(root/'point.pt',weights_only=False,map_location='cpu'),strict=True)
a=copy.deepcopy(e);a.load_state_dict(torch.load('outputs/feature_noise/artifacts/original_AGA.pt',weights_only=False,map_location='cpu'),strict=True);ap=copy.deepcopy(a);ap.load_state_dict(a.state_dict()|torch.load(root/'AGA/edits.pt',weights_only=False,map_location='cpu'),strict=True)
models={'clean_direct':merge(base,e,base,.5),'protected_direct':merge(base,p,base,.5),'clean_recovered':merge(base,a,base,.5),'protected_recovered':merge(base,ap,base,.5)};data=dataset('MNIST','data',e.val_preprocess);ids=list(range(10000));labels=official_labels('MNIST');raw={};rows={};maximum=0.;forwards=0;replays=0
for name,m in models.items():
 logits=[];runtime=evaluate(m,head,make_loader(data,ids),'cuda',logit_sink=logits.append);scores=torch.cat(logits).double();before=scores if name.startswith('clean') else raw[name.replace('protected','clean')]['logits'];row=row_metrics(scores,torch.as_tensor(labels),before,10);row['prediction_histogram']=torch.bincount(scores.argmax(1),minlength=10).tolist();row['NumPy']=module.numpy_check(before,scores,labels,row);row['runtime']=runtime;raw[name]={'logits':scores,'labels':labels,'indices':ids};forwards+=10000
 wanted=np.linspace(0,9999,16,dtype=int);blocks=sorted(set(wanted//128));context=[i for k in blocks for i in range(k*128,min((k+1)*128,10000))];chosen=[context.index(int(i)) for i in wanted];logits=[];evaluate(m,head,make_loader(data,context),'cuda',logit_sink=logits.append);actual=torch.cat(logits).double()[chosen];err=float((actual-scores[wanted]).abs().max());assert err<=5e-4;np.testing.assert_array_equal(actual.argmax(1),scores[wanted].argmax(1));maximum=max(maximum,err);forwards+=len(context);replays+=16;rows[name]=row;print(name,row['accuracy_percent'],row['prediction_histogram'],flush=True)
torch.save(raw,out/'scores.pt');source=Path(__file__).read_text();result={'status':'completed_read_only_same_point_public_proxy_diagnosis','point_sha256':digest(root/'point.pt'),'rows':rows,'source':source,'source_sha256':hashlib.sha256(source.encode()).hexdigest(),'seconds':time.perf_counter()-start,'new_real_image_forwards':forwards,'original_context_score_replays':replays,'maximum_context_score_error':maximum,'raw_scores_sha256':digest(out/'scores.pt'),'new_AGA_calls':0,'new_publications':0,'parameter_updates':0,'scope':'Public-base.5sourceproxies only,MNISTnaturaltest labels for read-onlydiagnosis after realpartnerfailure. Not additionalfusionmethods,independent attacks or constructionfeedback.'};save_json(out/'results.json',result)
