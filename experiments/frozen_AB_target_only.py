"""One missing historical A/B evaluation; reuse frozen author AGA weights, never optimize."""

import argparse
import copy
import json
from pathlib import Path
import time

import numpy as np
from scipy.special import gammaln, logsumexp
import torch

from author_protocol_audit import setup
from clip_benchmark import digest, evaluate, merge, paired_counts
from covariance_noise import dataset
from subspace_probe import make_loader, save_json
from verify_covariance_noise import official_labels
from verify_probe_candidate import explicit_merge


def inputs(cfg):
    anchor, own, peer, release, heads, distance = setup(cfg)
    for name in ['aligned_release','unprotected_predictions','AGA_original_reference']:
        ref = cfg[name]; assert digest(ref['path']) == ref['sha256']
    attacked = copy.deepcopy(own)
    attacked.load_state_dict(torch.load(cfg['AGA_original_reference']['path'],map_location='cpu'),strict=True)
    edited = copy.deepcopy(attacked); edits = torch.load(cfg['aligned_release']['path'],map_location='cpu')
    assert sorted(edits) == cfg['allowed_keys']; edited.load_state_dict(attacked.state_dict()|edits,strict=True)
    reference = torch.load(cfg['unprotected_predictions']['path'],map_location='cpu')
    return anchor, attacked, peer, edited, heads, reference, distance


def paired_logp(lost, gained):
    n = lost+gained
    if n == 0: return 0.
    k = np.arange(min(lost,gained)+1)
    return float(min(0., np.log(2)+logsumexp(gammaln(n+1)-gammaln(k+1)-gammaln(n-k+1))-n*np.log(2))/np.log(10))


@torch.no_grad()
def run(output, verify=False):
    start = time.monotonic(); out = Path(output)
    cfg = json.loads(Path('configs/frozen_AB_target_only.json').read_text())
    if not verify: out.mkdir(parents=True,exist_ok=False); save_json(out/'protocol.json',cfg)
    else: assert cfg == json.loads((out/'protocol.json').read_text())
    anchor, attacked, peer, edited, heads, ref, distance = inputs(cfg)
    model = explicit_merge(anchor,edited,peer,.5) if verify else merge(anchor,edited,peer,.5)
    if verify:
        meta = json.loads((out/'execution.json').read_text())
        assert digest(out/'predictions.pt') == meta['predictions_sha256']
        raw = torch.load(out/'predictions.pt',map_location='cpu'); rows = json.loads((out/'results.json').read_text())
    else: raw, rows = {}, []
    replays = 0
    for task in ['MNIST','SVHN']:
        data = dataset(task,'data',model.val_preprocess); before = ref['original/AGA_target_only/'+task]
        assert len(data) == cfg['test_counts'][task]
        if not verify:
            sink, logits = [], []
            metric = evaluate(model,heads[task],make_loader(data),'cuda',
                              prediction_sink=lambda p,y:sink.append((p,y)),logit_sink=logits.append)
            pred, y = [torch.cat([r[i] for r in sink]) for i in [0,1]]
            assert torch.equal(y,before['target'])
            raw[task] = dict(prediction=pred,target=y,logits=torch.cat(logits))
            old = before['prediction']; pair = paired_counts(old.eq(y),pred.eq(y))
            row = dict(task=task,**metric,original_correct=int(old.eq(y).sum()),changed_predictions=int(pred.ne(old).sum()),
                paired=pair,mcnemar_log10_p=paired_logp(pair['lost'],pair['gained']),
                per_class=[dict(class_id=c,n=int(y.eq(c).sum()),original_correct=int((old.eq(y)&y.eq(c)).sum()),
                                correct=int((pred.eq(y)&y.eq(c)).sum())) for c in range(10)])
            rows.append(row); save_json(out/'results.json',rows); torch.save(raw,out/'predictions.pt')
            print('FROZEN_AB_TARGET_ONLY',json.dumps(row),flush=True)
        else:
            r = raw[task]; y = official_labels(task); pred = r['prediction'].numpy(); old = before['prediction'].numpy()
            np.testing.assert_array_equal(y,r['target'].numpy()); np.testing.assert_array_equal(pred,r['logits'].numpy().argmax(1))
            row = next(v for v in rows if v['task']==task)
            assert int((pred==y).sum()) == row['correct'] and int((old!=pred).sum()) == row['changed_predictions']
            assert int(((old==y)&(pred!=y)).sum()) == row['paired']['lost']
            assert int(((old!=y)&(pred==y)).sum()) == row['paired']['gained']
            for c in row['per_class']:
                mask = y==c['class_id']; assert int(mask.sum())==c['n'] and int(((pred==y)&mask).sum())==c['correct']
            chosen = np.linspace(0,len(data)-1,cfg['replay_n_per_task'],dtype=int)
            images = torch.stack([data[int(i)][0] for i in chosen]).cuda(); heads[task].cuda()
            for m, expected in [(model,r),(explicit_merge(anchor,attacked,peer,.5),before)]:
                m.cuda(); actual = heads[task](m(images)).cpu(); m.cpu()
                np.testing.assert_array_equal(actual.argmax(1).numpy(),expected['prediction'][chosen].numpy())
                assert float((actual-expected['logits'][chosen]).abs().max()) < cfg['replay_score_atol']
                replays += len(chosen)
            heads[task].cpu()
        assert time.monotonic()-start < cfg['wall_limit_seconds']
    result = dict(status='passed' if verify else 'completed_frozen_reference_evaluation',new_candidates=0,
        optimizer_steps=0,new_attack_calls=0,parameter_distance=distance,peer_attacked=False,
        attack_scope='Original author AGA applied only to protected expert; reuse saved attack weights.',
        predictions_sha256=digest(out/'predictions.pt'),source_sha256=digest(__file__),wall_seconds=time.monotonic()-start,
        replay_forwards=replays,source_cfg_sha256=digest('configs/frozen_AB_target_only.json'),
        no_test_parameter_selection=True,historical_tests_not_fresh_blind=True,first_phase_changed=False)
    save_json(out/('independent_validation.json' if verify else 'execution.json'),result);print(json.dumps(result),flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',required=True);ap.add_argument('--verify',action='store_true')
    args = ap.parse_args();run(args.output,args.verify)
