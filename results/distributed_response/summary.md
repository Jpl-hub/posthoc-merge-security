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
