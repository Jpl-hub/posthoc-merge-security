"""One source-gated MNIST reference-context control, with fixed evaluation."""
import json
import subprocess
import sys
from pathlib import Path
from _impl.io import digest,save_json

path=Path('results/data_free_compensation/conditional_compensation/self_reference_protocol.json')
spec=json.loads(path.read_text());root=Path(spec['output']);py=sys.executable
def read(p):return json.loads(Path(p).read_text())
def run(*args):
    print('RUN',*args,flush=True)
    subprocess.run([py,'-u',*map(str,args)],check=True,timeout=3000)
def pin(cfg):
    cfg['source_sha256']={p:digest(p) for p in cfg['source_sha256']}
    return cfg
def status(stage,**kw):save_json(root/'status.json',dict(stage=stage,**kw))
status('source_derivative_check')
run('outputs/overnight_20261106/construct.py','--protocol',path,'--task','MNIST','--check-only')
status('source_construction')
run('outputs/overnight_20261106/construct.py','--protocol',path,'--task','MNIST')
result=read(root/'MNIST/point/result.json');s=result['source'];initial=result['initial']
eligible=result['source_checks_passed'] and s['native_matches']>=initial['native_matches'] and all(
    value<initial['source_counts'][key] for key,value in s['source_counts'].items())
if not eligible:
    status('closed_at_source',eligible=False,source=s,initial=initial)
    print('SOURCE_NOT_ELIGIBLE',s,flush=True)
    sys.exit(0)
status('frozen_before_real_images',publication=result['publication'])
ref=result['publication'];previous=Path('outputs/centered_readout_transfer_20261106')
recovery=pin(read(previous/'recovery.json'));rp=root/'recovery.json';save_json(rp,recovery)
run('-m','experiment.recover','--target','MNIST','--method','aga','--aligned-reference',
    '--protocol',rp,'--publication',ref['path'],'--sha256',ref['sha256'],'--output',root/'MNIST/AGA')
a=root/'MNIST/AGA/edits.pt'
entry=dict(target='MNIST',release=ref,AGA_edits=dict(path=str(a),sha256=digest(a)))
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
