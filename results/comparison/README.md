# NBP results

Construction uses frozen weights and a fixed classifier, without images or expert training. The same seven NBP releases are reused across partners; no test-based direction or magnitude selection is performed.

| Question | Recorded evidence |
| --- | --- |
| Standalone utility and seven protected directions | [Original three tasks](../phase3/bias_weight_only/other_tasks/summary.md), [remaining directions](../phase3/bias_weight_only/eight_expert/remaining_native_result.json) |
| All 42 directed task pairs | [Full matrix and counts](../phase3/bias_weight_only/partner_analysis/README.md) |
| Eight participating experts, seven evaluated datasets | [Main results](../phase3/bias_weight_only/eight_expert/summary.json), [additional directions](../phase3/bias_weight_only/eight_expert/extension/summary.md) |
| Other fusion rules | [TIES and DARE+TA](../phase3/bias_weight_only/fusion_methods/extension/summary.json) |
| Direction and magnitude controls | [Original controls](../phase3/bias_weight_only/ablation/summary.md), [five fixed random draws and class projection](../phase3/bias_weight_only/ablation/direction_controls/README.md) |
| Labelled coefficient selection | [Three tasks, fixed calibration grid](../phase3/bias_weight_only/coefficient_selection/README.md) |
| Weight geometry and rounding | [All class directions and three storage types](../phase3/bias_weight_only/ablation/direction_controls/weight_analysis/README.md) |
| Public final-bias projection | [Matched recovery results](../phase3/bias_weight_only/attacks/projection.json) |
| ViT-B/16 | [Three complete task evaluations](../backbone_b16/summary.json) |
| Matched PaRaMS reference | [Official defaults and results](../params_reference/) |
| Reproducible MD/TeX exports | [Tables and input fingerprints](../phase3/bias_weight_only/tables/) |

The task averages exclude EuroSAT: its checkpoint's original held-out split remains unverified. Eight participating models are not an eight-task accuracy table. AGA and public recovery operations are distinct from TA/TIES/DARE fusion rules. Fixed ProDistill block-coefficient fitting is not unrestricted student training. Public final-bias projection removes the NBP increment and is reported as a successful recovery operation.

Rebuild without model loading:

```bash
python -m experiment.report --output outputs/nbp_tables --figure
```

`sources.json` retains the original aggregate input record and its fingerprints. Its historical internal labels are preserved; the exporter selects only the relevant NBP and matched-control rows. Source cleanup does not change the measured counts.
