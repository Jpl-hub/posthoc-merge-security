#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/posthoc-match-20261028
ROOT=outputs/attack_generalization_20261028
PY=/root/autodl-tmp/posthoc-env/bin/python
while [ ! -f "$ROOT/exit_code.txt" ]; do sleep 30; done
[ "$(cat "$ROOT/exit_code.txt")" = 0 ]
trap 'rc=$?; printf "%s\n" "$rc" > "$ROOT/matching_exit_code.txt"' EXIT
PYTHONPATH=. "$PY" results/attack_generalization_20261028/prepare_matching.py
for TASK in MNIST SVHN DTD; do
  if [ "$TASK" = MNIST ]; then PAIR=MNIST_SVHN; elif [ "$TASK" = SVHN ]; then PAIR=SVHN_MNIST; else PAIR=DTD_MNIST; fi
  CFG="$ROOT/matching/${TASK}_protocol.json"
  "$PY" -m experiment.main_table --protocol "$CFG" --case 0 --pilot
  "$PY" -m experiment.main_table --verify "$ROOT/matching/$TASK/pilot/00_$PAIR"
  "$PY" -m experiment.main_table --protocol "$CFG" --case 0
  "$PY" -m experiment.main_table --verify "$ROOT/matching/$TASK/full/00_$PAIR"
done
