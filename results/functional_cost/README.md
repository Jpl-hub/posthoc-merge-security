# Functional compensation cost: unresolved

Read-only analysis of the already closed fixedpoint O, on its same4096 original fitting inputs. Original experts, O and the failed fitted B remain unchanged. No replacement B, candidate, fresh confirmation or fusion evaluation was produced.

A matrix-free complete normalized-margin Jacobian gives36,864 linear constraints. After the preregistered100 diagonally preconditioned CG steps:

- Maximum linear residual **0.505623443**, above0.5: **not feasible**.
- Computed linear vector norm0.180282198; this is **not** a feasible cost upper bound.
- Valid dual cost lower bound0.0; no nontrivial B-cost exclusion.
- Therefore the cost interval is unresolved. No extra iterations, rescaling or weaker threshold were used to pass it.

This problem constrains linearized full class margins, whereas the earlier3.245843 B cost reconstructed complete finite hidden outputs. The different objectives cannot be compared as two successful defenses or used to claim finite functional feasibility. Nor does this result establish that large B cost is intrinsically necessary.

Independent NumPy kernel/adjoint relative error1.66e-15; the complete normalized-score finite-difference relative error is1.47e-7. Wall time17.09s plus independent recount;4,096 repeated original forwards,0 new images,0 reverse gradients or releases. The diagnostic Gram operations and numerical CG solve are not expert training.

Protocol: `configs/functional_cost.json`; source at Git6e9ee71 and archived server executed_source.py. Raw Jacobians and vectors remain `/root/posthoc_evaluation_cache/phase2_functional_cost_20261018/` on3090. Only these light reports are in Git.
