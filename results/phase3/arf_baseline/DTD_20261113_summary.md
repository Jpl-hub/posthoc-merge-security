# ARF DTD matched comparison

Fixed protocol: ViT-B/32, seed 420, public base initialization, official DTD partition 1, 76 epochs, AdamW with learning rate 1e-5, batch 128, warm-up 500, no weight decay, and the declared max-hinge interpretation of the ARF distance term. The CE run uses the identical split, schedule and seed with the protection term set to zero. No test image is used for training or selection.

| Method | Standalone | Direct TA | AGA + TA (target) | AGA + TA (both) |
| --- | ---: | ---: | ---: | ---: |
| CE | 73.46 | 58.72 | 57.77 | 58.14 |
| ARF | 75.32 | 2.34 | 70.90 | 71.01 |

All values are percentages on the complete 1,880-image DTD test split. The fixed MNIST companion rows are in the server-side evaluation record. ARF satisfies all 48 recorded attention-matrix margins; the final distance/anchor-norm ratios range from 0.05385 to 0.05744 (required minimum 0.05). The independent replay checked 224 score contexts with zero score error.

This is a paper-prose ARF adaptation, not an official-code reproduction. The original equation and prose differ on the hinge direction; this run follows the explicitly recorded prose interpretation. It should be reported as a matched adaptation and kept separate from the seven-task released-expert ranking.
