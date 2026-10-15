"""Full-model validation that the shared inner environment is realizable."""

import argparse
import copy
import json
import time
from pathlib import Path

import torch

from attention_cascade import parameters
from clip_benchmark import digest,merge
from covariance_noise import dataset
from subspace_probe import make_loader,save_json
from task_probe_distance import checked_models,initialize


@torch.no_grad()
def run(cfg):
    start=time.monotonic();root=Path(cfg['output']);report=json.loads((root/'results.json').read_text());valid=json.loads((root/'independent_validation.json').read_text())
    assert valid['status']=='passed' and valid['finite_multiclass_signal']
    assert digest(root/'diagnostic.pt')==report['diagnostic_sha256']==valid['diagnostic_sha256']
    d=torch.load(root/'diagnostic.pt',map_location='cpu');cal=torch.load(Path(cfg['cache_output'])/'calibration.pt',map_location='cpu')
    anchor,expert,head,pins=checked_models(cfg['target']);p=parameters(expert,head)
    s=d['theta'].reshape(cfg['probe_rank'],cfg['probe_rank']);basis=cal['basis']
    e=p['D'].norm()*basis['u']@s@basis['vh'];point=torch.load(cfg['protected_point'],map_location='cpu')
    r=cfg['probe_rank'];th=point['theta'];q=point['output_basis']
    delta=dict(O=p['O'].norm()*point['basis_O']['u']@th[:r*r].reshape(r,r)@point['basis_O']['vh'],
               B=p['B'].norm()*q@th[r*r:].reshape(len(q.T),p['B'].shape[1]))
    prefix='model.visual.transformer.resblocks.';key=prefix+'10.mlp.c_proj.weight'
    source=copy.deepcopy(expert);reference=merge(anchor,expert,anchor,cfg['coefficient'])
    for model in [source,reference]:model.state_dict()[key].copy_((model.state_dict()[key].double()+e).float())
    protected=copy.deepcopy(reference)
    for name,k in [('O',prefix+'11.attn.out_proj.weight'),('B',prefix+'11.mlp.c_proj.weight')]:
        protected.state_dict()[k].copy_((protected.state_dict()[k].double()+.5*delta[name]).float())
    ids=d['indices']['proxy'][:32];lookup={idx:i for i,idx in enumerate(d['indices']['fit'])};pos=[lookup[i] for i in ids]
    expected=[d['final_source'][pos],d['final_baseline'][:32],d['final_protected'][:32]]
    training=dataset(cfg['target'],'data',expert.val_preprocess,True);records=[];raw={}
    tolerance=json.loads(Path('configs/attention_cascade.json').read_text())['full_cache_tolerance']
    for name,model,want in zip(['source_plus_shared','reference_plus_shared','protected_reference_plus_shared'],[source,reference,protected],expected):
        vals=[];model.cuda().eval();head.cuda().eval()
        for images,_ in make_loader(training,ids,32):vals.append(head(model(images.cuda())).cpu().double())
        model.cpu();head.cpu();actual=torch.cat(vals);error=float((actual-want).abs().max());match=torch.equal(actual.argmax(1),want.argmax(1))
        assert error<tolerance and match,(name,error,match)
        records.append(dict(model=name,n=len(ids),maximum_FP32_score_error=error,all_predictions_match=True));raw[name]=actual
    torch.save(dict(predictions=raw,indices=ids,checkpoint_sha256=pins),root/'full_model_replay.pt')
    result=dict(status='passed',records=records,new_full_model_forwards=3*len(ids),cache_tolerance=tolerance,
        actual_shared_parameter_distance=float(e.norm()/p['D'].norm()),
        diagnostic_sha256=report['diagnostic_sha256'],predictions_sha256=digest(root/'full_model_replay.pt'),
        source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,peer_used=False,test_used=False,new_release_weights=0,
        interpretation='The SAME B10 edit is instantiated in three complete models on fixed32 target-training images. Source, public-reference and original-O/B protected-reference predictions match finite cached calculations. This validates realizability, not coverage of trained unknown peers or new protection efficacy.')
    save_json(root/'full_model_replay_validation.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol',required=True);args=ap.parse_args();initialize()
    run(json.loads(Path(args.protocol).read_text()))
