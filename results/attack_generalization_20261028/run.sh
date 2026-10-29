#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/posthoc-attack-20261028
PY=/root/autodl-tmp/posthoc-env/bin/python
ROOT=outputs/attack_generalization_20261028
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/exit_code.txt"' EXIT
for CASE in 0 1 2 3 4 5 6 7 8; do
  if [ "$CASE" -lt 3 ]; then PAIR=MNIST_SVHN; elif [ "$CASE" -lt 6 ]; then PAIR=SVHN_MNIST; else PAIR=DTD_MNIST; fi
  ID=$(printf "%02d_%s" "$CASE" "$PAIR")
  if [ "$CASE" -ne 0 ]; then
    "$PY" -m experiment.main_table --protocol results/attack_generalization_20261028/protocol.json --case "$CASE" --pilot
    "$PY" -m experiment.main_table --verify "$ROOT/public_replacement/pilot/$ID"
  fi
  "$PY" -m experiment.main_table --protocol results/attack_generalization_20261028/protocol.json --case "$CASE"
  "$PY" -m experiment.main_table --verify "$ROOT/public_replacement/full/$ID"
done
PYTHONPATH=. "$PY" results/attack_generalization_20261028/calibrate.py results/attack_generalization_20261028/calibration_protocol.json
PYTHONPATH=. "$PY" results/attack_generalization_20261028/prepare_controls.py
for CASE in 0 1 2; do
  if [ "$CASE" -eq 0 ]; then PAIR=MNIST_SVHN; elif [ "$CASE" -eq 1 ]; then PAIR=SVHN_MNIST; else PAIR=DTD_MNIST; fi
  ID=$(printf "%02d_%s" "$CASE" "$PAIR")
  "$PY" -m experiment.main_table --protocol "$ROOT/baselines/evaluation_protocol.json" --case "$CASE" --pilot
  "$PY" -m experiment.main_table --verify "$ROOT/baselines/pilot/$ID"
  "$PY" -m experiment.main_table --protocol "$ROOT/baselines/evaluation_protocol.json" --case "$CASE"
  "$PY" -m experiment.main_table --verify "$ROOT/baselines/full/$ID"
done
