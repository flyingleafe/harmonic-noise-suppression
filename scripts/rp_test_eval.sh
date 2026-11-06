#!/usr/bin/env bash
# Score every checkpoint of the wrap-up paper's Table VI (tab:pretrain) on the
# HELD-OUT test panel (conf/validation/rp_real_test.yaml, rp-real-test-v1),
# and the same checkpoints again on the validation panel (rp_real) so each
# test row can be matched to its Table VI row. One GPU job:
#
#   omnirun submit --backend uni --gpus 1 --time 3h --name rp-test-eval \
#       --outputs 'results/rp_test_eval/**' -- bash scripts/rp_test_eval.sh
#
# Checkpoint per row = the one Table VI reads (its source comments):
# real-only and fine-tuned/mixed rows at the lowest five-round median of
# real_macro (Conv+BiGRU real-only and hard-tuned at best_real_macro_raw,
# HarmoF0-L2 tuned rows at best_real_median); synthetic-only rows at the
# synthetic selector (easy: near_macro, hard: uniform) and, for the
# "real-sel." column, at real_macro.
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD/scripts"
OUT=results/rp_test_eval
mkdir -p "$OUT"

TARGETS=()
declare -A REAL=(
  [pyrseplstm_k64]=rp_pyrseplstm_k64_alllab:best_real_macro_median
  [harmof0_l2]=rp_harmof0_l2_all:best_real_macro_median
  [scv2]=rp_scv2_all:best_real_macro_raw
  [hppnet_l2]=rp_hppnet_l2_alllab:best_real_macro_median
)
for m in pyrseplstm_k64 harmof0_l2 scv2 hppnet_l2; do
  tuned=best_real_macro_median
  hard_tuned=best_real_macro_median
  [[ $m == harmof0_l2 ]] && tuned=best_real_median && hard_tuned=best_real_median
  [[ $m == scv2 ]] && hard_tuned=best_real_macro_raw
  TARGETS+=(
    "${REAL[$m]}"
    "rp_${m}_easyv4:best_near_macro_median"
    "rp_${m}_easyv4:best_real_macro_median"
    "rp_${m}_priorv4:best_uniform_median"
    "rp_${m}_priorv4:best_real_macro_median"
    "rp_${m}_easyv4_ft_lr5e4:${tuned}"
    "rp_${m}_priorv4_ft_lr5e4:${hard_tuned}"
    "rp_${m}_easyv4_mixlab:best_real_macro_median"
    "rp_${m}_priorv4_mixlab:best_real_macro_median"
  )
done

nvidia-smi -L || true
python scripts/rp_panel_eval.py "${TARGETS[@]}" --validation rp_real_test --out "$OUT/test.json"
python scripts/rp_panel_eval.py "${TARGETS[@]}" --validation rp_real --out "$OUT/valid.json"
