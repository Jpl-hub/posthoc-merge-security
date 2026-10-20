"""One frozen publication, matched cached baselines and unchanged original AGA."""
import argparse, contextlib, copy, io, json, time
from pathlib import Path
import numpy as np
import torch
from author_protocol_audit import setup
from clip_benchmark import author_recover, digest, evaluate, merge, paired_counts
from covariance_noise import dataset
from frozen_AB_target_only import paired_logp
from mechanisms import load_author
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels
from verify_probe_candidate import explicit_merge

def inputs(cfg):
    anchor,own,peer,released,heads,distance=setup(cfg);aligned={}
    for task,model in [('MNIST',own),('SVHN',peer)]:
        ref=cfg['AGA_references'][task];assert digest(ref['path'])==ref['sha256']
        aligned[task]=copy.deepcopy(model);aligned[task].load_state_dict(torch.load(ref['path'],map_location='cpu'),strict=True)
    return anchor,own,peer,released,heads,distance,aligned

def counts(pred,y):
    return [dict(class_id=c,n=int(y.eq(c).sum()),correct=int((pred.eq(y)&y.eq(c)).sum())) for c in range(10)]

@torch.no_grad()
def run(protocol):
    cfg=json.loads(Path(protocol).read_text());out=Path(cfg['output']);out.mkdir(parents=True,exist_ok=False);(out/'artifacts').mkdir()
    save_json(out/'protocol.json',cfg);start=time.monotonic();anchor,own,peer,released,heads,distance,aligned=inputs(cfg);torch.cuda.reset_peak_memory_stats()
    ref=cfg['reference_predictions'];assert digest(ref['path'])==ref['sha256'];old=torch.load(ref['path'],map_location='cpu')
    raw={k:v for k,v in old.items() if k.startswith('original/') and '/confirmation/' not in k};rows=[]
    for key,r in raw.items():
        preset,stage,task=key.split('/');pred,y=r['prediction'],r['target']
        rows.append(dict(preset=preset,stage=stage,dataset=task,n=len(y),correct=int(pred.eq(y).sum()),top1=float(pred.eq(y).double().mean()),per_class=counts(pred,y),cached_reference=True))
    save_json(out/'pre_evaluation.json',dict(release=cfg['release'],confirmation=cfg['prior_confirmation'],parameter_distance=distance,
        source_point_reselected=False,protocol_amendment=cfg['protocol_amendment'],future_peer_used_for_construction=False,historical_tests_not_fresh_blind=True))
    data={t:dataset(t,'data',own.val_preprocess) for t in heads};assert {t:len(v) for t,v in data.items()}==cfg['test_counts']
    def record(model,stage,task):
        sink,logits=[],[];metric=evaluate(model,heads[task],make_loader(data[task]),'cuda',prediction_sink=lambda p,y:sink.append((p,y)),logit_sink=logits.append)
        pred,y=[torch.cat([v[i] for v in sink]) for i in [0,1]];before=raw[f'original/{stage}/{task}'];assert torch.equal(y,before['target'])
        pair=paired_counts(before['prediction'].eq(y),pred.eq(y));raw[f'released/{stage}/{task}']=dict(prediction=pred,target=y,logits=torch.cat(logits))
        row=dict(preset='released',stage=stage,dataset=task,**metric,original_correct=int(before['prediction'].eq(y).sum()),
            changed_predictions=int(before['prediction'].ne(pred).sum()),paired=pair,mcnemar_log10_p=paired_logp(pair['lost'],pair['gained']),per_class=counts(pred,y))
        rows.append(row);save_json(out/'results.json',rows);torch.save(raw,out/'artifacts/predictions.pt');print('FROZEN_LOCAL_EVAL',json.dumps(row),flush=True)
        assert time.monotonic()-start<cfg['wall_limit_seconds']
    diagnostic=cfg.get('utility_failed_diagnostic')
    if diagnostic:
        # Explicitly separate a requested fixed-model diagnostic from the main
        # protocol. Its failed utility gate remains failed; no weights change.
        for key in ['predictions','results','independent_receipt']:
            ref=diagnostic[key];assert digest(ref['path'])==ref['sha256'],key
        check=json.loads(Path(diagnostic['independent_receipt']['path']).read_text())
        assert check['status']=='passed_failure_receipt' and not check['original_task_gate_passed']
        assert diagnostic['release_sha256']==cfg['release']['sha256']
        previous=torch.load(diagnostic['predictions']['path'],map_location='cpu')
        for key in ['prediction','target','logits']:
            assert torch.equal(previous['original/standalone/MNIST'][key],raw['original/standalone/MNIST'][key])
        raw['released/standalone/MNIST']=previous['released/standalone/MNIST']
        prior=json.loads(Path(diagnostic['results']['path']).read_text())
        native=next(r for r in prior if r['preset']=='released' and r['stage']=='standalone')
        rows.append(native|dict(cached_reference=True))
        save_json(out/'results.json',rows);torch.save(raw,out/'artifacts/predictions.pt')
    else:record(released,'standalone','MNIST')
    # Main protocol stops here on utility failure. Only an explicitly pinned,
    # separately reported fixed-model diagnostic may proceed past that failure.
    original=raw['original/standalone/MNIST']['logits'].double();after=raw['released/standalone/MNIST']['logits'].double()
    winner=original.argmax(1);ix=torch.arange(len(winner));gaps=original[ix,winner,None]-original;mask=torch.ones_like(gaps,dtype=torch.bool);mask[ix,winner]=False;assert (gaps[mask]>0).all()
    change=after-original;ratio=(change[ix,winner,None]-change)/gaps.masked_fill(~mask,1.);ratio[ix,winner]=0
    worst=float(ratio.abs().max());pos=int(ratio.abs().argmax());image,competitor=divmod(pos,ratio.shape[1])
    gate=dict(passed=worst<=cfg['task_budget'],n=len(winner),task_budget=cfg['task_budget'],worst_relative_margin_drift=worst,witness_index=image,teacher_class=int(winner[image]),competitor=competitor,source_point_repaired=False,historical_test_not_blind=True)
    save_json(out/'full_native_gate.json',gate)
    if diagnostic:assert not gate['passed'] and abs(worst-check['worst_relative_margin_drift'])<1e-10
    if not gate['passed'] and not diagnostic:
        save_json(out/'execution.json',dict(completed=False,stopped='full_native_task_budget_violation',author_AGA_calls=0,parameter_distance=distance,utility_gate=gate,predictions_sha256=digest(out/'artifacts/predictions.pt'),source_sha256=digest(__file__),wall_seconds=time.monotonic()-start));return
    model=merge(anchor,released,peer,.5)
    for task in heads:record(model,'direct',task)
    del model;author,pin=load_author(Path('external/secure-merge-attack/utils_attack_B32.py'),True);assert pin==cfg['AGA_text_sha256']
    log=io.StringIO();attack_start=time.monotonic()
    with contextlib.redirect_stdout(log):attacked=author_recover(released,anchor,author)
    attack_seconds=time.monotonic()-attack_start;text=log.getvalue();(out/'AGA.log').write_text(text)
    assert not any(x in text for x in ['Skipping','Warning:','Error']);assert all(f'Attacking Layer {i}...' in text for i in range(12))
    assert '>>> MLP-only attack completed.' in text
    edits={k:v.clone() for k,v in attacked.state_dict().items() if not torch.equal(v,aligned['MNIST'].state_dict()[k])}
    torch.save(edits,out/'artifacts/AGA_release_edits.pt');record(attacked,'AGA_standalone','MNIST')
    for stage,other in [('AGA_target_only',peer),('AGA_both',aligned['SVHN'])]:
        model=merge(anchor,attacked,other,.5)
        for task in heads:record(model,stage,task)
        del model
    matrix_cost=[]
    for k,v in own.state_dict().items():
        if v.ndim==2 and v.is_floating_point() and v.norm()>0:
            matrix_cost.append(float(((attacked.state_dict()[k].double()-aligned['MNIST'].state_dict()[k].double()).norm()/v.double().norm()).square()))
    save_json(out/'execution.json',dict(completed=True,parameter_distance=distance,AGA_aligned_matrix_distance=sum(matrix_cost)**.5,
        full_model_forward_instances=sum(r['n'] for r in rows if r['preset']=='released' and not r.get('cached_reference')),
        evaluation_scope='utility_failed_fixed_model_diagnostic' if diagnostic else 'main_protocol',utility_gate=gate,
        source_point_repaired=False,new_defense_optimizer_steps=0,author_AGA_calls=1,
        cached_original_and_peer_AGA=True,attack_seconds=attack_seconds,wall_seconds=time.monotonic()-start,peak_gpu_bytes=torch.cuda.max_memory_allocated(),
        release_sha256=cfg['release']['sha256'],predictions_sha256=digest(out/'artifacts/predictions.pt'),
        AGA_edits_sha256=digest(out/'artifacts/AGA_release_edits.pt'),source_sha256=digest(__file__),protocol_sha256=digest(protocol),
        protocol_amendment=cfg['protocol_amendment'],GPU=torch.cuda.get_device_name(),first_phase_changed=False))

@torch.no_grad()
def verify(output):
    out=Path(output);cfg=json.loads((out/'protocol.json').read_text());meta=json.loads((out/'execution.json').read_text());assert meta['completed']
    if cfg.get('utility_failed_diagnostic'):
        assert meta['evaluation_scope']=='utility_failed_fixed_model_diagnostic' and not meta['utility_gate']['passed']
    assert digest(out/'artifacts/predictions.pt')==meta['predictions_sha256'] and digest(out/'artifacts/AGA_release_edits.pt')==meta['AGA_edits_sha256']
    anchor,own,peer,released,heads,distance,aligned=inputs(cfg);attacked=copy.deepcopy(aligned['MNIST'])
    attacked.load_state_dict(attacked.state_dict()|torch.load(out/'artifacts/AGA_release_edits.pt',map_location='cpu'),strict=True)
    raw=torch.load(out/'artifacts/predictions.pt',map_location='cpu');rows=json.loads((out/'results.json').read_text());data={t:dataset(t,'data',own.val_preprocess) for t in heads};replays=0
    for row in rows:
        preset,stage,task=[row[k] for k in ['preset','stage','dataset']];r=raw[f'{preset}/{stage}/{task}'];y=official_labels(task);pred=r['prediction'].numpy()
        np.testing.assert_array_equal(y,r['target']);np.testing.assert_array_equal(pred,r['logits'].numpy().argmax(1));assert np.isfinite(r['logits'].numpy()).all()
        assert int((pred==y).sum())==row['correct']
        for c in row['per_class']:
            mask=y==c['class_id'];assert int(mask.sum())==c['n'] and int(((pred==y)&mask).sum())==c['correct']
        if preset=='released':
            old=raw[f'original/{stage}/{task}']['prediction'].numpy();lost=int(((old==y)&(pred!=y)).sum());gained=int(((old!=y)&(pred==y)).sum())
            assert (lost,gained)==(row['paired']['lost'],row['paired']['gained']) and int((old!=pred).sum())==row['changed_predictions']
            assert abs(paired_logp(lost,gained)-row['mcnemar_log10_p'])<1e-9
        if stage=='partner_standalone':model=peer
        elif stage=='standalone':model=own if preset=='original' else released
        elif stage=='AGA_standalone':model=aligned['MNIST'] if preset=='original' else attacked
        else:
            left=(own if preset=='original' else released) if stage=='direct' else (aligned['MNIST'] if preset=='original' else attacked)
            other=aligned['SVHN'] if stage=='AGA_both' else peer;model=explicit_merge(anchor,left,other,.5)
        ids=np.linspace(0,len(y)-1,16,dtype=int);actual=[];model.cuda();heads[task].cuda()
        for i in ids:
            lo=int(i)//128*128;hi=min(lo+128,len(y));images=torch.stack([data[task][j][0] for j in range(lo,hi)]).cuda();actual.append(heads[task](model(images))[int(i)-lo].cpu())
        actual=torch.stack(actual);model.cpu();heads[task].cpu();np.testing.assert_array_equal(actual.argmax(1),pred[ids])
        assert float((actual.double()-r['logits'][ids].double()).abs().max())<=cfg['score_tolerance'];replays+=len(ids)
    save_json(out/'independent_validation.json',dict(status='passed',rows=len(rows),explicit_TA_full_model_replays=replays,
        official_label_perclass_and_pair_counts=True,predictions_sha256=meta['predictions_sha256'],parameter_distance=distance,
        original_AGA_not_reimplemented=True,source_point_reselected=False,source_sha256=digest(__file__)))
    print('INDEPENDENT_LOCAL_EVAL_PASSED',len(rows),replays,flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--protocol');ap.add_argument('--verify');a=ap.parse_args()
    verify(a.verify) if a.verify else run(a.protocol)
