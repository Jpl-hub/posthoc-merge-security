#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/posthoc-match-20261028
ROOT=outputs/attack_generalization_20261028
PY=/root/autodl-tmp/posthoc-env/bin/python
while [ ! -f "$ROOT/matching_exit_code.txt" ]; do sleep 30; done
[ "$(cat "$ROOT/matching_exit_code.txt")" = 0 ]
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/mergers_exit_code.txt"' EXIT
for CASE in 0 1 2; do
  if [ "$CASE" -eq 0 ]; then PAIR=MNIST_SVHN; elif [ "$CASE" -eq 1 ]; then PAIR=SVHN_MNIST; else PAIR=DTD_MNIST; fi
  ID=$(printf "%02d_%s" "$CASE" "$PAIR")
  "$PY" -m experiment.main_table --protocol results/attack_generalization_20261028/mergers_protocol.json --case "$CASE" --pilot
  "$PY" -m experiment.main_table --verify "$ROOT/mergers/pilot/$ID"
  "$PY" -m experiment.main_table --protocol results/attack_generalization_20261028/mergers_protocol.json --case "$CASE"
  "$PY" -m experiment.main_table --verify "$ROOT/mergers/full/$ID"
done
