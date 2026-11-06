"""One source-gated MNIST reference-context control, with fixed evaluation."""
import json
import argparse
import subprocess
import sys
from pathlib import Path
from _impl.io import digest,save_json

parser=argparse.ArgumentParser()
parser.add_argument('--protocol',default='results/data_free_compensation/conditional_compensation/self_reference_protocol.json')
parser.add_argument('--source-result',help='Evaluate one already completed, source-eligible point')
args=parser.parse_args()
path=Path(args.protocol)
spec=json.loads(path.read_text());root=Path(spec['output']);py=sys.executable
root.mkdir(parents=True,exist_ok=True)
def read(p):return json.loads(Path(p).read_text())
def run(*args):
    print('RUN',*args,flush=True)
    subprocess.run([py,'-u',*map(str,args)],check=True,timeout=3000)
def pin(cfg):
    cfg['source_sha256']={p:digest(p) for p in cfg['source_sha256']}
    return cfg
def status(stage,**kw):save_json(root/'status.json',dict(stage=stage,**kw))
if args.source_result:
    result=read(args.source_result)
    eligible=result['eligible']
    s=dict(source_counts={'recovered/TA/raw':result['counts']['recovered/TA']})
else:
    status('source_derivative_check')
    run('outputs/overnight_20261106/construct.py','--protocol',path,'--task','MNIST','--check-only')
    status('source_construction')
    run('outputs/overnight_20261106/construct.py','--protocol',path,'--task','MNIST')
    result=read(root/'MNIST/point/result.json');s=result['source'];initial=result['initial']
    eligible=result['source_checks_passed'] and s['native_matches']>=initial['native_matches'] and all(
        value<initial['source_counts'][key] for key,value in s['source_counts'].items())
if not eligible:
    status('closed_at_source',eligible=False,source=s)
    print('SOURCE_NOT_ELIGIBLE',s,flush=True)
    sys.exit(0)
status('frozen_before_real_images',publication=result['publication'])
ref=result['publication'];previous=Path('outputs/centered_readout_transfer_20261106')
recovery=pin(read(previous/'recovery.json'));rp=root/'recovery.json';save_json(rp,recovery)
run('-m','experiment.recover','--target','MNIST','--method','aga','--aligned-reference',
    '--protocol',rp,'--publication',ref['path'],'--sha256',ref['sha256'],'--output',root/'MNIST/AGA')
a=root/'MNIST/AGA/edits.pt'
entry=dict(target='MNIST',release=ref,AGA_edits=dict(path=str(a),sha256=digest(a)))
# Check the actual attack on the original synthetic views before any real data.
import torch
import torch.nn.functional as F
from _impl.models import checked_models,apply
from _impl.merge import merge
source_spec=read('configs/defense2_inputs.json')['MNIST']
raw=torch.load(Path(source_spec['root'])/'source/source.pt',map_location='cpu',weights_only=False)
base,expert,head,_=checked_models('MNIST')
clean_aligned=apply(expert,source_spec['clean_AGA'],full=True)
attacked=apply(clean_aligned,entry['AGA_edits'])
model=merge(base,attacked,base,.5).cuda().eval();head.cuda()
scores=[];norm=expert.val_preprocess.transforms[-1]
with torch.no_grad():
    for pixels in raw['final'].split(8):
        x=F.interpolate(pixels,(224,224),mode='bilinear',align_corners=False)
        x=torch.cat([torch.roll(x,s,(2,3)) for s in [(0,0),(7,0),(-7,0),(0,7),(0,-7)]])
        x=(x-x.new_tensor(norm.mean)[None,:,None,None])/x.new_tensor(norm.std)[None,:,None,None]
        scores.append(head(model(x.cuda())).cpu())
logits=torch.cat(scores);labels=raw['cache']['native']['scores'].argmax(1)
actual_matches=int(logits.argmax(1).eq(labels).sum())
expected_matches=s['source_counts']['recovered/TA/raw']
torch.save(dict(logits=logits,teacher_class=labels),root/'actual_AGA_source_scores.pt')
save_json(root/'actual_AGA_source_check.json',dict(actual_matches=actual_matches,construction_matches=expected_matches,n=len(labels),
    raw_sha256=digest(root/'actual_AGA_source_scores.pt'),passed=actual_matches==expected_matches))
del model,head,base,expert,attacked,clean_aligned,raw,scores
torch.cuda.empty_cache()
if actual_matches!=expected_matches:
    status('stopped_actual_AGA_source_mismatch',actual=actual_matches,expected=expected_matches)
    sys.exit(2)
for mode in ['evaluation','b_reset']:
    cfg=pin(read(previous/(mode+'.json')))
    cfg.update(output=str(root/mode),defenses={'Conditional':entry},
               cases=[dict(target='MNIST',partner='SVHN',defenses=['Conditional'])])
    if mode=='b_reset':cfg['cases'][0]['public_replacement']=dict(layer=11,side='B+b')
    p=root/(mode+'.json');save_json(p,cfg);status(mode)
    run('-m','experiment.main_table','--protocol',p,'--pilot')
    run('-m','experiment.main_table','--protocol',p)
    run('-m','experiment.main_table','--verify',root/mode/'full/00_MNIST_SVHN')
status('prodistill')
cfg=pin(read(previous/'prodistill.json'));cfg['cases'][0]['releases']={'Conditional':ref}
cfg.update(output=str(root/'prodistill'),experiment='One frozen self-reference conditional-compensation control')
p=root/'prodistill.json';save_json(p,cfg)
run('-m','experiment.distill','--prodistill','--protocol',p)
status('completed_no_automatic_expansion',publication=ref)
