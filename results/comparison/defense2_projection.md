# Effect of task-projection constraints

Fixed component comparison of the current Defense 2. Only the two task-SVD constraints on A are removed; synthetic inputs, bias compensation, functional bound, optimizer and stopping rules are unchanged. Task-shaped synthetic probes are retained. Two experts, TA coefficient 0.5; AGA-t processes the protected expert, AGA-b both experts. Calibration uses the same 64 random inputs. All three controls are frozen before real-image evaluation. MNIST and SVHN controls stop with nonzero synthetic classification violations; DTD reaches zero. All controls are reported, including these failures. No task-wise replacement of the main method or proof of improved performance on other datasets.

| Task | Projection | Standalone | TA | AGA-t | AGA-b | Calib. | AGA-t+calib. | AGA-b+calib. |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| MNIST | Retained | 99.57 | 25.15 | 2.99 | 1.33 | 29.78 | 10.76 | 9.18 |
| MNIST | Removed | 99.68 | 8.91 | 8.90 | 8.90 | 9.80 | 10.23 | 10.21 |
| SVHN | Retained | 96.66 | 2.26 | 1.86 | 1.87 | 7.41 | 6.99 | 6.95 |
| SVHN | Removed | 97.41 | 2.54 | 4.16 | 3.80 | 9.15 | 12.32 | 11.26 |
| DTD | Retained | 76.33 | 2.13 | 2.07 | 2.07 | 6.60 | 5.00 | 5.00 |
| DTD | Removed | 77.87 | 1.60 | 2.13 | 2.18 | 2.18 | 0.90 | 0.90 |
