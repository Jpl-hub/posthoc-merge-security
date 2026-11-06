# Conditional compensation: closed experiment

Original experts remain frozen. Construction uses no real images. The three complete-test results below did not retain the main method's protection; this experiment is closed. Eight-expert, seven-dataset evaluation also completed. The unrelated readout supplemental evaluation was stopped after 68 rows and is partial.

| Target | Standalone | TA | AGA-T | AGA-B | AGA-T + B replacement | ProDistill |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| MNIST | 99.70 | 99.25 | 99.15 | 99.16 | 98.72 | 99.44 |
| SVHN | 97.42 | 80.47 | 84.48 | 83.62 | 74.24 | 96.74 |
| DTD | 78.99 | 35.21 | 39.47 | 39.89 | 17.39 | 72.66 |

A frozen same-input check gives 18/800 teacher matches with the public base, but 792/800 with the actual partner. Cached and full public-candidate logits agree to 2.861e-5. The construction omitted the previous method's self-expert reference. This supports a controlled reference-context check; it does not prove that restoring it will solve recovery.

Raw scores, model/source fingerprints, per-class results and paired intervals remain on21516 under outputs/overnight_20261106. Compact counts and validations are in result.json.
