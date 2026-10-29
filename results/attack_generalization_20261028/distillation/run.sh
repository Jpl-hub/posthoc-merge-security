#!/usr/bin/env bash
set -euo pipefail
cd /root/autodl-tmp/posthoc-merge-security
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONNOUSERSITE=1
PY=/root/miniconda3/bin/python
OUT=outputs/attack_generalization_20261029/distillation_MNIST
trap 'rc=$?; printf "%s\n" "$rc" > "${OUT}_exit_code.txt"' EXIT
"$PY" -m experiment.distill --protocol results/attack_generalization_20261028/distillation/protocol.json
CFG="$OUT/evaluation_protocol.json"
"$PY" -m experiment.main_table --protocol "$CFG" --pilot
"$PY" -m experiment.main_table --verify "$OUT/evaluation/pilot/00_MNIST_SVHN"
"$PY" -m experiment.main_table --protocol "$CFG"
"$PY" -m experiment.main_table --verify "$OUT/evaluation/full/00_MNIST_SVHN"
