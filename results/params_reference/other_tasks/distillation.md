# Matched data-assisted distillation

Accuracy (%). The attacker uses 2048 unlabeled training images for MNIST/SVHN and 1880 DTD train1 images, teacher hard labels and the fixed public-base student training rule: 50 epochs, batch128, AdamW1e-5, seed2026101912. Defense construction uses no real images. Additional AGA processing of the student is shown separately.

All six Defense1/PaRaMS teacher-label vectors exactly equal their unprotected reference on the same frozen image tensors and indices. Consequently each task reuses its already evaluated unprotected student. These shared rows are not new training repetitions. Defense2 results also reuse the previously completed students.

| Task | Student teacher | Standalone | TA target | AGA target + TA | AGA both + TA |
|---|---|---:|---:|---:|---:|
| MNIST | Unprotected / Defense 1 / PaRaMS | 98.50 | 95.53 | 95.35 | 95.25 |
| MNIST | Defense 2 | 98.50 | 95.53 | 95.35 | 95.25 |
| SVHN | Unprotected / Defense 1 / PaRaMS | 93.09 | 86.22 | 85.28 | 85.58 |
| SVHN | Defense 2 | 92.39 | 85.84 | 85.00 | 85.21 |
| DTD | Unprotected / Defense 1 / PaRaMS | 73.30 | 58.19 | 57.82 | 58.30 |
| DTD | Defense 2 | 73.24 | 58.56 | 57.71 | 58.19 |

Independent NumPy argmax and index checks passed for17928 new teacher queries; no student or defender training and no new AGA solve. Partner results and exact provenance are in [distillation.json](distillation.json). This attack requires additional images and training; these results do not describe weight-only recovery.
