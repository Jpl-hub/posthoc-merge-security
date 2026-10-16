# PaRaMS diagonal adaptive recovery

TA(0.5), protected MNIST / partner SVHN. Each fused cell reports both accuracies (%).

| Defense | MNIST standalone: original / recovered | No recovery | AGA: protected expert | Diagonal recovery: protected expert | AGA: all experts | Diagonal recovery: all experts |
|---|---:|---:|---:|---:|---:|---:|
| Unprotected | 99.69 / 99.69 | 99.37 / 92.50 | 99.30 / 92.78 | 99.37 / 92.50 | 99.32 / 91.77 | 99.37 / 92.50 |
| Defense 1 | 99.69 / 99.69 | 9.80 / 6.70 | 9.80 / 6.70 | 9.80 / 6.70 | 9.80 / 6.70 | 9.80 / 6.70 |
| Defense 2 O/B† | 99.71 / 99.71 | 78.77 / 62.94 | 79.53 / 65.82 | 78.77 / 62.94 | 77.85 / 61.42 | 78.77 / 62.94 |
| Defense 2 A/B† | 99.65 / 99.65 | 91.07 / 77.46 | 85.42 / 75.50 | 91.07 / 77.46 | 79.20 / 69.84 | 91.07 / 77.46 |
| PaRaMS full‡ | 99.69 / 99.69 | 8.92 / 9.15 | 99.30 / 92.79 | 99.37 / 92.50 | 99.32 / 91.77 | 99.37 / 92.50 |
| PaRaMS Attention ablation | 99.69 / 99.69 | 99.37 / 92.51 | 99.30 / 92.78 | 99.37 / 92.50 | 99.32 / 91.77 | 99.37 / 92.50 |

† Both frozen Defense 2 variants have known violations of the original 0.5 task budget; effect references only. ‡ Full official PaRaMS is over the 0.5 parameter budget; the Attention ablation is not full PaRaMS.

Diagonal recovery uses the unchanged official MLP matching kernel plus an explicitly specified row-wise least-squares Attention adaptation of the paper objective. It is not the complete author attack implementation or paper table reproduction. No defense weights were updated.

Historical no-recovery/AGA columns were measured on RTX 3090; new diagonal-recovery columns use RTX 4090 D with the same hashed checkpoints and evaluation functions. They are references across hardware and different recovery operators; do not interpret tiny differences as a controlled hardware-independent gain. All new paired statistics compare against the matching unprotected diagonal-recovery control on RTX 4090 D.

Each recovery scope is retained. Tests were previously observed. Only MNIST/SVHN is covered by this new attack evaluation; no cross-protected-task or arbitrary-attack robustness claim.

Frozen attack edits were reused after pilot verification. Full evaluation: 480.77s; selected-score full-model replays: 480, max score error 0. Construction and replay costs are additional.

Raw predictions and recovered parameters remain on the servers. JSON contains per-class counts, matched paired statistics and signed margin changes.
