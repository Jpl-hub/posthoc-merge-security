# Fixed joint-response direction: task gate

One frozen B10/B11 direction, parameter distance 0.4999995. Original task budget 0.5; no change after confirmation or tests.

| Evaluation set | Inputs | Worst relative gap drift | Prediction changes | Task gate |
|---|---:|---:|---:|---|
| Construction + finite probes | 6656 | 7.93762e-05 | 0 | Pass |
| Fresh train + finite probes | 3072 | 0.0313674 | 0 | Pass |
| Complete MNIST test | 10000 | 1.06848 | 0 | Fail |

Complete-test accuracy is 99.69% before and after protection; all predictions match. 2 images exceed the functional budget.
The independently replayed witness (test index 8527, class 4 versus 9) increases the gap from 0.04598331 to 0.09511566. Increased confidence still violates the original absolute relative-gap constraint.

**TA and original AGA were not run:** the preregistered native-task gate stopped this fixed candidate. This is neither a feasible-defense result nor evidence about its attack resistance. No radius, rank, layer, output axis or weights were changed to repair it.

## Coupling and limits

On the construction inputs, B10 alone has worst drift 0.00568254, B11 alone 0.00569164, and their fixed combination 7.93762e-05. Each has zero construction prediction changes; this does not establish harmful isolated branches.
The minimum empirical joint response and finite task-SVD probes support source compensation, not unseen-input preservation. Standard SVD, Rayleigh optimization and Gauss-Newton geometry are not novelty claims. Historically exposed tests are not a first blind evaluation.

Construction: 152.82s; confirmation: 53.50s; stopped full-task evaluation: 21.46s. Independent audits and their costs remain separate. No optimizer, expert updates or new attack fits.


## Parameter distance versus source activation

| Branch / inputs | Relative parameter distance | MLP output change (RMS %) | Activation energy / uniform-complement mean |
|---|---:|---:|---:|
| B10 all tokens | 0.039248 | 0.303353 | 0.00709173 |
| B10 CLS | 0.039248 | 0.084877 | 0.00119874 |
| B11 CLS | 0.498457 | 0.043667 | 8.0648e-06 |

All rows use the 4,096 original construction images. B10 includes separate all-token and CLS statistics; B11 uses CLS. MLP output here means the linear Bz contribution before the unchanged bias. The uniform-complement column is an analytic second-moment reference, with no sampled or optimized replacement direction.
The joint linear response energy is 0.000666904 times the sum of isolated-branch energies: coupling exists, while the dominant B11 change is weakly excited on these inputs.
This diagnoses the frozen source solution only. It does not prove the cause of every unseen-input violation or establish attack efficacy. No rewhitening, new weights, threshold changes or additional TA/AGA evaluations follow from this table.


## Fixed edit under public-base inputs

| Branch / inputs | Native delta RMS | Public-base delta RMS | Base / native delta RMS | Delta / base Bz (RMS %) |
|---|---:|---:|---:|---:|
| B10 all tokens | 0.017952 | 0.03586686 | 1.99793 | 0.947390 |
| B10 CLS | 0.002665004 | 0.04216527 | 15.8218 | 1.094427 |
| B11 CLS | 0.002738089 | 12.37516 | 4519.63 | 294.484439 |

The same frozen edits are multiplied by activations from the unchanged public base on the same 4,096 construction images. Native statistics are reused by hash. Base Bz is a separate denominator; these are matrix responses, not complete protected-model features or class scores.
Native silence alone therefore does not establish lack of meaningful response in other environments. This comparison demonstrates input-conditional amplification, not unknown-partner transfer, actual fusion damage or AGA resistance. The complete-task failure remains decisive: this point stays closed, with no new defense or repair.
Public-base forward pass: 49.10s, 4096 image instances, zero new training indices, protected-model evaluations or merges.


## Finite source classification interventions

| Input environment | Edit | Correct / 64 | Prediction changes | Worst relative gap drift |
|---|---|---:|---:|---:|
| native | original | 64/64 | 0 | 0 |
| native | B10 only | 64/64 | 0 | 0.00178905 |
| native | B11 only | 64/64 | 0 | 0.00174697 |
| native | joint | 64/64 | 0 | 6.54434e-05 |
| public base | original | 33/64 | 0 | 0 |
| public base | B10 only | 34/64 | 1 | 4.2616 |
| public base | B11 only | 6/64 | 61 | 899.1 |
| public base | joint | 6/64 | 61 | 899.878 |

Fixed first64 original construction indices, never selected by response or class. Public-base tail interventions are not merged-model or AGA results. Native64/64 preservation on these source inputs does not remove the previously verified complete-task violation. No candidate is reopened, repaired or promoted by this table.
The first run saved64 public-base inputs before missing historical feature metadata stopped it. Its source/log remain archived; the corrected run reused that cache and added64 original-model feature controls. The fixed thresholds and inputs were retained. Successful run18.90s; failed-run wall time was not separately measured.
