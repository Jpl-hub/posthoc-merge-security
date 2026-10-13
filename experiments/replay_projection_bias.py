"""Rebuild two frozen releases from fresh target inputs before extending the task."""

import argparse
import json
from pathlib import Path

import torch

from clip_benchmark import checkpoint_loader, digest
from covariance_noise import dataset
from projection_bias_noise import construct
from subspace_probe import save_json, split_training


@torch.no_grad()
def replay(protocol, output):
    cfg=json.loads(Path(protocol).read_text())
    assert not cfg.get("native_cache"), "This check must capture fresh proxy features"
    for path,pin in cfg["source_sha256"].items():
        assert digest(path)==pin,path
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    load=checkpoint_loader(Path("external/secure-merge-attack"))
    root=Path("checkpoints/ViT-B-32")
    pins={p["path"].removeprefix("ViT-B-32/"):p["sha256"]
          for p in json.loads(Path("configs/checkpoints.json").read_text())["files"]}

    def checked(name):
        assert digest(root/name)==pins[name],name
        return load(str(root/name)).cpu().eval()

    out=Path(output)
    out.mkdir(parents=True,exist_ok=False)
    anchor=checked("zeroshot.pt")
    records=[]
    for target in ["MNIST","DTD"]:
        ref=Path(cfg["reproduction_references"][target])
        meta=json.loads((ref/"execution.json").read_text())
        assert json.loads((ref/"independent_validation.json").read_text())["status"]=="passed"
        assert digest(ref/"artifacts/release.pt")==meta["release_sha256"]
        assert digest(ref/"artifacts/calibration.pt")==meta["calibration_sha256"]
        original=checked(f"{target}/finetuned.pt")
        head=checked(f"head_{target}.pt")
        training=dataset(target,"data",original.val_preprocess,True)
        n=min(cfg["maximum_calibration_group"],len(training)//3)
        indices=split_training(training,{k:n for k in ["fit","scale","confirmation"]},cfg["seed"])
        assert indices==json.loads((ref/"training_indices.json").read_text())
        released,edits,cal,geo=construct(original,anchor,head,training,indices,cfg,target)
        expected=torch.load(ref/"artifacts/release.pt",map_location="cpu")
        assert set(edits)==set(expected)
        assert all(torch.equal(edits[k],expected[k]) for k in edits),"Fresh release differs; do not extend"
        prior=torch.load(ref/"artifacts/calibration.pt",map_location="cpu")
        assert digest(prior["cache_path"])==prior["cache_sha256"]
        cached=torch.load(prior["cache_path"],map_location="cpu")
        assert torch.equal(cal["proxy_residual"],cached["proxy_residual"])
        assert torch.equal(cal["direction"],prior["direction"])
        assert cal["amplitude"]==prior["amplitude"]
        records.append(dict(target=target,release_sha256=meta["release_sha256"],
                            exact_release_tensors=True,exact_proxy_features=True,
                            exact_direction=True,exact_amplitude=True,
                            fresh_inputs=n,amplitude=geo["amplitude"]))
        del released,original,head,training,edits,cal,prior,cached,expected
    receipt=dict(status="passed",cases=records,protocol_sha256=digest(protocol),
                 source_sha256=cfg["source_sha256"],partner_loaded=False,
                 test_loaded=False,full_test_rerun=False,new_algorithm=False)
    save_json(out/"reproduction.json",receipt)
    print(json.dumps(receipt),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol",default="configs/projection_bias_noise_svhn.json")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    replay(args.protocol,args.output)
