Full-test standalone accuracy (%). Parameters remain FP32; FP16/BF16 denote AMP operations. R32 keeps final LayerNorm, projection and classifier in FP32. Invalid denotes nonfinite scores.

| Task | Precision | None | NBP |
| --- | --- | ---: | ---: |
| Cars | FP32 | 77.66 | 77.63 |
| Cars | FP16 | 77.71 | Invalid |
| Cars | BF16 | 77.52 | 0.55 |
| Cars | FP16 + R32 | 77.69 | 77.65 |
| Cars | BF16 + R32 | 77.64 | 77.64 |
| SVHN | FP32 | 97.46 | 97.46 |
| SVHN | FP16 | 97.46 | 97.46 |
| SVHN | BF16 | 97.47 | 97.33 |
| SVHN | FP16 + R32 | 97.46 | 97.46 |
| SVHN | BF16 + R32 | 97.47 | 97.47 |
| MNIST | FP32 | 99.69 | 99.69 |
| MNIST | FP16 | 99.69 | 99.69 |
| MNIST | BF16 | 99.68 | 99.68 |
| MNIST | FP16 + R32 | 99.69 | 99.69 |
| MNIST | BF16 + R32 | 99.68 | 99.68 |
