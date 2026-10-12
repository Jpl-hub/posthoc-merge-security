"""One frozen CPU source/matrix audit; no model forward or new release."""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.linalg
import torch

torch.set_num_threads(4)
torch.set_grad_enabled(False)
out = Path("outputs/aga_canonical_audit")
protocol = json.loads((out / "protocol.json").read_text())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert not (out / "results.json").exists(), "Do not overwrite a completed audit"
capsule = Path(protocol["input_capsule"])
edit_path = Path(protocol["archived_edit"])
author_path = Path(protocol["author_file"])
for p, pin in ((capsule, "input_capsule_sha256"), (edit_path, "archived_edit_sha256"),
               (author_path, "author_raw_sha256")):
    assert sha(p) == protocol[pin], str(p)
tree = ast.parse(author_path.read_text())
tree.body = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == "solve_linear_transformation"]
assert len(tree.body) == 1
env = {"torch": torch}
exec(compile(tree, str(author_path), "exec"), env)
solve = env["solve_linear_transformation"]
inputs = torch.load(capsule, map_location="cpu")
edit = torch.load(edit_path, map_location="cpu")
delta = edit["delta"][:64].float()
rows = []
checks = []
for operator, sign in (("q", 1), ("k", -1)):
    original = inputs[operator].float()
    anchor = inputs[operator + "0"].float()
    archived = original + sign * delta
    for name, w in (("original", original), ("archived_release", archived)):
        x, y = w.T.double().numpy(), anchor.T.double().numpy()
        u, singular, _ = scipy.linalg.svd(x, full_matrices=False, lapack_driver="gesvd")
        tolerance = max(x.shape) * np.finfo(float).eps * singular[0]
        rank = int((singular > tolerance).sum())
        assert rank == 64, "Ideal full-rank formula needs review if rank changes"
        ideal = u @ (u.T @ y)
        lam = protocol["ridge"]
        augmented_x = np.vstack((x, np.sqrt(lam) * np.eye(64)))
        augmented_y = np.vstack((y, np.zeros((64, 64))))
        ridge_t = scipy.linalg.lstsq(augmented_x, augmented_y, lapack_driver="gelsd")[0]
        ridge = x @ ridge_t
        # Reproduce source's FP32 return and FP32 matrix product, on CPU only.
        author_t = solve(w.T, anchor.T, lambda_reg=lam)
        recovered = (author_t.T @ w).T.double().numpy()
        norm = lambda z: float(np.linalg.norm(z))
        update_norm = norm(x-y)
        ideal_error = norm(ideal-x)
        np.testing.assert_allclose(x.T @ (y-ideal), 0, atol=1e-7, rtol=0)
        assert ideal_error <= update_norm * (1+1e-10)
        direct_t = scipy.linalg.lstsq(x, y, lapack_driver="gelsd")[0]
        direct = x @ direct_t
        independent_error = norm(direct-ideal)/max(norm(ideal), 1e-30)
        assert independent_error < 1e-8
        rows.append(dict(operator=operator, state=name, numerical_rank=rank,
            singular_min=float(singular[-1]), singular_max=float(singular[0]),
            condition_number=float(singular[0]/singular[-1]),
            input_update_frobenius=update_norm,
            ideal_error_to_input_frobenius=ideal_error,
            ideal_vs_original_expert_frobenius=norm(ideal-original.T.double().numpy()),
            author_vs_ideal_relative_frobenius=norm(recovered-ideal)/max(norm(ideal),1e-30),
            ridge_vs_ideal_relative_frobenius=norm(ridge-ideal)/max(norm(ideal),1e-30),
            author_vs_augmented_ridge_relative_frobenius=norm(recovered-ridge)/max(norm(ridge),1e-30),
            independent_svd_lstsq_relative_error=independent_error))
        checks.append(dict(operator=operator,state=name,
            projection_orthogonality_passed=True,input_relative_update_bound_passed=True,
            independent_SVD_lstsq_passed=True))
result = dict(protocol=protocol, rows=rows, checks=checks,
    independent_algebra_validation="passed",script_sha256=sha(__file__),
    protocol_sha256=sha(out/"protocol.json"),new_model_forwards=0,
    new_protected_weights=0,new_classification_results=0,completed=True)
(out / "results.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result),flush=True)
