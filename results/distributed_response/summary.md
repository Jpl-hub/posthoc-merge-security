# Multilayer B-only source feasibility

Frozen source-only diagnostic. No release, confirmation, fusion or attack performance. All rows fail the original task0.5 bound as well as the predeclared source reserve0.25.

| Coefficient radius | R: core20 | R: other492 | Changed /512 | Correct /512 |
|---:|---:|---:|---:|---:|
| 0.49999950 | 22.155267 | 2.851945 | 467 | 45 |
| 0.24999975 | 38.114900 | 2.757836 | 408 | 104 |
| 0.12499987 | 42.349667 | 2.428779 | 180 | 332 |
| 0.06249994 | 35.567622 | 2.422764 | 49 | 463 |
| 0.03124997 | 15.407004 | 2.373381 | 18 | 494 |
| 0.01562498 | 3.103148 | 2.202835 | 2 | 510 |
| 0.00781249 | 0.660606 | 1.667653 | 1 | 511 |
| 0.00390625 | 0.152437 | 0.650801 | 0 | 512 |

Radii refer to the low-rank forward prototype; unquantized logical distances are in the JSON. No FP32 publication was materialized.
Per-class lost/gained counts, source labels/indices, directional geometry and provenance are in independent_validation.json.
The initial FP32 finite-difference failure is retained separately. The later FP64 reference changes numerical derivative validation only, not the primary FP32 function budget.

## Source tangent decomposition

Fixed original512 source inputs and all eight closed points; no new direction or correction. Full-model forward AD is checked against the original180 reverse-mode rows and two independent reverse checks.

| Coefficient radius | Finite R | First-order R | Nonlinear remainder R |
|---:|---:|---:|---:|
| 0.49999950 | 22.155267 | 46.824177 | 46.655909 |
| 0.24999975 | 38.114900 | 23.412089 | 38.114926 |
| 0.12499987 | 42.349667 | 11.706044 | 42.349680 |
| 0.06249994 | 35.567622 | 5.853022 | 35.567628 |
| 0.03124997 | 15.407004 | 2.926511 | 15.407007 |
| 0.01562498 | 3.103148 | 1.463256 | 3.103149 |
| 0.00781249 | 1.667653 | 0.731628 | 1.399021 |
| 0.00390625 | 0.650801 | 0.365814 | 0.436976 |

These are separate maxima and must not be added. At the smallest point, the actual worst source gap has signed first-order/remainder changes of -0.365814/-0.284987, summing to -0.650801. Core20 first-order maximum is only3.96e-7 while its finite remainder reaches0.152437. Source coverage and finite nonlinear response both matter. This is a source diagnostic, not a defense or attack result.
