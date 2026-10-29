#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/posthoc-merge-security
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
PY=/root/miniconda3/bin/python
BASE=outputs/attack_generalization_20261029
trap 'rc=$?; printf "%s\n" "$rc" > "$BASE/distillation_other_exit_code.txt"' EXIT
for TASK in SVHN DTD; do
  OUT="$BASE/distillation_$TASK"
  "$PY" -m experiment.distill --protocol "results/attack_generalization_20261028/distillation/$TASK/protocol.json"
  CFG="$OUT/evaluation_protocol.json"
  "$PY" -m experiment.main_table --protocol "$CFG" --pilot
  "$PY" -m experiment.main_table --verify "$OUT/evaluation/pilot/00_${TASK}_MNIST"
  "$PY" -m experiment.main_table --protocol "$CFG"
  "$PY" -m experiment.main_table --verify "$OUT/evaluation/full/00_${TASK}_MNIST"
done
