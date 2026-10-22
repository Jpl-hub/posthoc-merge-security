# Fixed-model TA and original AGA evaluation

Standalone MNIST: 99.69% / 99.69%. Historical relative-gap result: 1.06848; original task check passed: False. TA coefficient 0.5; unchanged original AGA. This separately registered fixed-model diagnostic does not repair the original task-check failure or establish a new defense. The original construction used target training images, so this model is not data-free.

| Evaluation | MNIST unprotected | MNIST edited | SVHN unprotected | SVHN edited |
|---|---:|---:|---:|---:|
| TA | 99.37 | 99.38 | 92.50 | 92.48 |
| AGA target only | 99.30 | 99.29 | 92.78 | 92.79 |
| AGA both | 99.32 | 99.31 | 91.77 | 91.74 |

Independent checks: 17 rows and 272 scores replayed in their original batch contexts.
Evaluation time: 132.62s; peak GPU memory: 819.1 MiB. Verification cost is separate.
