"""Official Attention-only ablation; interval determined analytically by budget."""

import importlib.util
import json
import sys
import time

from params_reference import Path, checkpoint_loader, digest, save_json, torch


@torch.no_grad()
def run():
    protocol=Path("configs/params_scaling_reference.json");cfg=json.loads(protocol.read_text())
    out=Path(cfg["output"]);out.mkdir(parents=True,exist_ok=False);(out/"artifacts").mkdir()
    save_json(out/"protocol.json",cfg);torch.set_num_threads(4)
    assert "3090" in torch.cuda.get_device_name()
    for p,sha in cfg["source_sha256"].items():assert digest(p)==sha,p
    source=Path(cfg["official_source"]);assert digest(source)==cfg["official_source_sha256"]
    root=Path("checkpoints/ViT-B-32");load=checkpoint_loader(Path("external/secure-merge-attack"))
    pins={x["path"].removeprefix("ViT-B-32/"):x["sha256"] for x in json.loads(Path("configs/checkpoints.json").read_text())["files"]}
    models=[];hashes={}
    for name in ["zeroshot.pt","MNIST/finetuned.pt"]:
        hashes[name]=digest(root/name);assert hashes[name]==pins[name]
        m=load(str(root/name)).cpu().eval();m.requires_grad_(False);models.append(m)
    anchor,own=models
    spec=importlib.util.spec_from_file_location("official_scaling_ablation",source)
    author=importlib.util.module_from_spec(spec);sys.modules[spec.name]=author;spec.loader.exec_module(author)
    epsilon=cfg["parameter_budget"]/(36**.5+cfg["parameter_budget"])
    assert abs(epsilon-cfg["epsilon"])<1e-15
    start=time.monotonic()
    state=author.params(own.state_dict(),anchor.state_dict(),**cfg["author_call"])
    wall=time.monotonic()-start;rows=[];all_sq=[];matrix_sq=[];num=den=0.
    for k,w in own.state_dict().items():
        if not w.is_floating_point():continue
        d=state[k].double()-w.double();n=float(w.double().norm());dn=float(d.norm());num+=dn**2;den+=n**2
        if dn:
            assert ".attn." in k and n>0
            r=dn/n;all_sq.append(r*r)
            if w.ndim==2:matrix_sq.append(r*r)
            rows.append(dict(key=k,shape=list(w.shape),original_norm=n,delta_norm=dn,relative=r))
    distance=sum(all_sq)**.5;assert distance<=.5
    torch.save(state,out/"artifacts/PaRaMS.pt")
    save_json(out/"case.json",dict(target="MNIST",partner="SVHN",role="official scaling-only ablation, not full PaRaMS"))
    save_json(out/"release_freeze.json",dict(status="frozen_before_any_data_or_peer",reference_variant="scaling_only",
              parameter_distance_all_changed_tensors=distance,parameter_distance_matrices=sum(matrix_sq)**.5,
              MLP_matrices_distance_lower_bound=0.,global_model_relative_distance=(num/den)**.5,
              parameter_budget=.5,within_parameter_budget=True,is_different_budget_reference=False,
              original_expert_updated=False,gradients=0,labels_used=False,images_used=False,
              future_peer_loaded=False,test_loaded=False,source_sha256=cfg["source_sha256"],
              original_checkpoint_sha256=hashes,official_source_sha256=digest(source),
              release_sha256=digest(out/"artifacts/PaRaMS.pt"),construction_wall_seconds=wall,
              actual_default_call=False,edited_tensors=rows,analytic_bound=(36**.5)*epsilon/(1-epsilon),
              definition="Published module ablation using official enable_permutation=False; interval from budget, seed0. No clipping, interpolation or test selection. Not full PaRaMS."))
    print("OFFICIAL_SCALING_ABLATION_FROZEN",distance,flush=True)


if __name__=="__main__":run()
