# Fixed-budget hard-label recovery

Native adaptation, not full PaRaMS KD reproduction. MNIST/SVHN; fixed2048 unlabeled training images,50epochs,batch128. Only an attacker-owned public-base visual encoder is trained. Historical tests; no defense changes or test selection.

| Student group / teacher | Standalone MNIST | TA MNIST / SVHN | AGA-target MNIST / SVHN | AGA-both MNIST / SVHN |
|---|---:|---:|---:|---:|
| Native Unprotected | 99.69 | 99.37 / 92.50 | 99.30 / 92.78 | 99.32 / 91.77 |
| Native Phase1_MNIST | 99.69 | 9.80 / 6.70 | 9.80 / 6.70 | 9.80 / 6.70 |
| Unprotected + Phase1_MNIST | 98.50 | 95.53 / 95.95 | 95.35 / 95.94 | 95.25 / 95.29 |

- Teacher disagreements on the fixed query set: 0. Exact matching labels share one training problem; this is not independent seed replication.
- The matched unprotected-teacher student is the attack control. Native unprotected TA/AGA and public-base controls remain separate; high accuracy is not proof of unique-knowledge theft.
- Student training: 226.01s, 800 full steps; discarded one-step pilot separately recorded.
- Full evaluation context score replays: 112; maximum error 0. Teacher-query replay inputs: 1024; maximum error 0.
