# PaRaMS under MLP weight matching

Accuracy (%). TA coefficient 0.5. MNIST pairs with SVHN; SVHN and DTD pair with MNIST. The attack uses only public-base weights and published weights. It matches the incoming and outgoing weights of each of the 12 MLPs; attention parameters remain unchanged.

| Protected task | Model | Standalone | Merged target | Merged partner |
|---|---|---:|---:|---:|
| MNIST | Unprotected | 99.69 | 99.37 | 92.50 |
| MNIST | PaRaMS | 99.69 | 10.72 | 9.10 |
| SVHN | Unprotected | 97.46 | 92.50 | 99.37 |
| SVHN | PaRaMS | 97.46 | 9.55 | 9.82 |
| DTD | Unprotected | 79.41 | 69.31 | 99.45 |
| DTD | PaRaMS | 79.41 | 2.93 | 10.35 |

The fixed MLP matching attack does not recover PaRaMS. All standalone predictions remain equal to the clean expert. This is a CLIP MLP adaptation of the Git Re-Basin weight objective, not full-architecture recovery or an official CLIP reproduction. AGA also processes attention and is reported separately.

18 full rows and 18 pilot rows passed independent counts, official labels, full indices, per-class losses/gains, paired intervals and exact McNemar checks. All 576 selected scores were replayed in their original batches; maximum difference 1.34e-5. Nine full clean rows were reused after exact model and fused-parameter hash checks. New work: 72 CPU assignments, 121856 full evaluation forwards, 1152 pilot forwards; no defense update, attack fitting images or new AGA solve.

Sources: [protocol](matching_protocol.json), [results and validation](matching.json). Raw scores and executed source remain in `outputs/params_matching_20261031` on the evaluation host and in the local raw archive.
