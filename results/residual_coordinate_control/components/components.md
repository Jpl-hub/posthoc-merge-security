# Frozen-release component comparison

Same512 original construction images. Separate counterfactuals, no new defense selection, training or attack. The original full-model TA/AGA results remain unchanged.

| Components applied | Correct | Changed predictions | Relative class-gap change |
|---|---:|---:|---:|
| input only | 49/512 | 463 | 1.19124 |
| transformer only | 77/512 | 434 | 1.16101 |
| readout only | 45/512 | 466 | 1.09682 |
| complete | 511/512 | 0 | 1.44827e-06 |
