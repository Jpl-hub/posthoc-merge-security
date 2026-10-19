# Finite prefix/readout compensation

Source-only feasibility study. No valid defense release, confirmation, test, real merge or AGA result.

| Quantity | Observed value |
|---|---:|
| Joint parameter budget | 0.5 |
| Fixed B0 parameter distance | 0.353553022 |
| Remaining P distance | 0.353553406 |
| Original task bound | 0.5 |
| Sufficient source-cone tolerance | 0.25 |
| Shared P coefficients | 288 |
| Evaluated source inputs | 512 |
| Source indices drawn (not all evaluated) | 4096 |
| Original / prefix-only correct | 511 / 454 |
| Prefix-only R | 90.692108 |
| Prefix right-task32 response fraction | 9.204e-09 |
| Solver status | PrimalInfeasible |
| Solver iterations / seconds | 11 / 49.98 |
| Independent bounded-dual margin | 0.998989 |
| Constraint-assembly max error | 1.876e-12 |

The fixed upstream edit has a real classification effect despite negligible direct response on its task-SVD input directions. The shared readout cannot satisfy this particular sufficient cone within the remaining budget. The cone is conservative (epsilon0.25) and not equivalent to the original0.5 functional band; this does not prove that all finite compensation is impossible. No radius/rank/layer/data/solver retries follow.

The raw dual has nonzero stationarity residual. Verification projects it into the self-dual cones and explicitly allows r*||A.T z||=0.000790504; b.T z=-0.999779777 leaves a positive infeasibility margin. This is a numerical certificate, not interval arithmetic.

Initial finite-difference and mixed-kernel derivative-control failures remain in failures.json. The same source indices, original gradient norm and original finite-difference discrepancy were reused exactly. Matched-kernel forward/reverse AD passed1e-5; the original5% finite-difference threshold was never raised.

Completed production64.79s (solver49.98s), independent NumPy audit6.66s. Failed-invocation runtime was not recorded and is not included. Derived forward counts are not independent sample counts. Heavy source scores, cones, projected certificate and prefix diagnostics remain on the RTX3090.
