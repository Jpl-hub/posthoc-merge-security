"""One source-only test of a readout conditional on the published MLP input."""
import copy, json
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from _impl.io import digest, save_json
from _impl.models import checked_models, initialize
from _impl.metrics import logical_parameter_distance

out = Path("outputs/readout_replacement_20261103")
protocol = json.loads(Path("results/data_free_compensation/readout_replacement/protocol.json").read_text())
assert digest(__file__) == protocol["source_sha256"]
assert not (out / "evaluation.npz").exists()
initialize()
base, expert, head, pins = checked_models("MNIST")
source, point = [protocol[k] for k in ("source_images", "frozen_release")]
for item in (source, point):
    assert digest(item["path"]) == item["sha256"]
pixels = torch.load(source["path"], map_location="cpu", weights_only=False)["final"]
edits = torch.load(point["path"], map_location="cpu", weights_only=False)
incoming = {k:v for k,v in edits.items() if ".mlp.c_fc." in k}
assert len(incoming) == 2 and len(pixels) == 160
student = copy.deepcopy(base)
student.load_state_dict(base.state_dict() | incoming)
norm = expert.val_preprocess.transforms[-1]
mean, std = [torch.as_tensor(v).view(1,-1,1,1).cuda() for v in (norm.mean, norm.std)]

@torch.no_grad()
def forward(model):
    activations, embeddings = [], []
    handle = model.model.visual.ln_post.register_forward_hook(
        lambda _, inputs, outputs: activations.append(outputs.detach().cpu()))
    model.cuda()
    for images in pixels.split(16):
        x = F.interpolate(images.cuda(), (224,224), mode="bilinear", align_corners=False)
        embeddings.append(model((x-mean)/std).cpu())
    handle.remove()
    model.cpu()
    return torch.cat(activations).numpy().astype('float64'), torch.cat(embeddings).numpy()

x0, z0 = forward(base)
xa, za = forward(student)
_, zt = forward(expert)
P0 = base.model.visual.proj.detach().numpy().astype('float64')
np.testing.assert_allclose(x0@P0,z0,atol=3e-4,rtol=1e-4)
np.testing.assert_allclose(xa@P0,za,atol=3e-4,rtol=1e-4)
fit = np.arange(len(xa)) % 2 == 0
cutoff = max(xa.shape) * np.finfo('float64').eps
X = x0[fit]
Q = np.eye(X.shape[1]) - np.linalg.pinv(X,rcond=cutoff) @ X
Z = xa[fit] @ Q
delta = Q @ np.linalg.pinv(Z,rcond=cutoff) @ (zt[fit] - xa[fit]@P0)
P = torch.tensor(P0+delta, dtype=base.model.visual.proj.dtype)
assert torch.isfinite(P).all()
projection = {"model.visual.proj":P}
published = incoming | projection
torch.save(published,out/"fixed_edits_from_base.pt")
frozen_sha = digest(out/"fixed_edits_from_base.pt")
teacher_labels = head(torch.tensor(zt)).argmax(1).numpy()
scores = {"expert":head(torch.tensor(zt)).detach().numpy()}
for name, state in (("base",{}),("A_only",incoming),("P_only",projection),("joint",published)):
    student.load_state_dict(base.state_dict() | state)
    _, z = forward(student)
    scores[name] = head(torch.tensor(z)).detach().numpy()
    direct = (xa if name in ("A_only","joint") else x0) @ (P.double().numpy() if name in ("P_only","joint") else P0)
    np.testing.assert_allclose(direct,z,atol=3e-4,rtol=1e-4)
counts = {k:{s:int(np.sum(v[m].argmax(1)==teacher_labels[m])) for s,m in (("fit",fit),("held",~fit))} for k,v in scores.items()}
held = {k:v["held"] for k,v in counts.items()}
passed = held["joint"]==held["expert"] and min(held["joint"]-held[n] for n in ("A_only","P_only"))>0 and max(held["A_only"],held["P_only"])<=held["base"]
np.savez_compressed(out/"evaluation.npz",fit=fit,**scores,x0=x0,xa=xa,zt=zt,P=P.numpy(),P0=P0)
save_json(out/"result.json",dict(source_gate_passed=passed,agreement=counts,held_per_class=np.bincount(teacher_labels[~fit],minlength=10).tolist(),images_per_partition=80,fit_residual_max=float(np.abs(xa[fit]@(P0+delta)-zt[fit]).max()),base_fit_invariance_error=float(np.abs(X@delta).max()),relative_projection_change=float(np.linalg.norm(delta)/np.linalg.norm(P0)),D_relative_to_expert=logical_parameter_distance(base.state_dict()|published,expert.state_dict()),checkpoint_pins=pins,release_sha256=frozen_sha,arrays_sha256=digest(out/"evaluation.npz"),source_sha256=digest(__file__),natural_images=0,AGA_calls=0,partner_models=0))
print(json.dumps(dict(passed=passed,agreement=counts)),flush=True)
