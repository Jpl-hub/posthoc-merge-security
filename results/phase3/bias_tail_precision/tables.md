# Targeted bias recovery: separate readout precision control

FP32 prefix; FP64 final LayerNorm, visual projection and normalized classifier. Every cell is protected-task / partner-task accuracy (%). Do not pool with the native FP32 main table.

| Protected task | Partner | Recovery scope | Defense 1: no recovery | Defense 1: recovered | Unprotected: same recovery |
|---|---|---|---:|---:|---:|
| DTD | MNIST | Protected expert | 2.13 / 21.09 | 69.26 / 99.45 | 69.26 / 99.45 |
| DTD | MNIST | All experts | 2.13 / 21.09 | 69.26 / 99.45 | 69.26 / 99.45 |
| SVHN | MNIST | Protected expert | 6.70 / 9.80 | 92.50 / 99.37 | 92.50 / 99.37 |
| SVHN | MNIST | All experts | 6.70 / 9.80 | 92.50 / 99.37 | 92.50 / 99.37 |

Standalone original / protected / recovered protected accuracy (%):

- DTD: 79.41 / 79.41 / 79.41
- SVHN: 97.46 / 97.46 / 97.46

Full evaluation 126.38s; 512 selected-score replays with maximum error 2.49356e-13. All observed standalone before/after cleanup score differences are below unchanged0.0005. NumPy complete readout, official labels, per-class and paired counts passed.

The six recovery edits were frozen before earlier experiments and are reused verbatim. No image/label/base/peer enters recovery construction; public base and peer are only used in TA(0.5). Both scopes are retained, and recovered protected predictions equal matching recovered unprotected controls.

This is the same supplementary architecture-aware diagnosis under a new predeclared numerical protocol, not another attack algorithm or an official paper reproduction. The earlier nativeFP32 DTD control failure0.00082493 remains failed. No threshold, defense weight or recovery edit was changed. These tests were observed historically. Raw hidden states/scores remain on3090.
