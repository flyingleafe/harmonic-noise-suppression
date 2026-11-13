#!/usr/bin/env bash
# Score the checkpoints of the wrap-up paper's tables on the HELD-OUT test
# panel (conf/validation/rp_real_test.yaml, rp-real-test-v1), and the same
# checkpoints again on the validation panel (rp_real) so each test row can be
# matched to its validation row. One GPU job per call; one group per table:
#
#   omnirun submit --backend uni --gpus 1 --time 1h --name rp-test-eval \
#       --outputs 'results/rp_test_eval/**' -- bash scripts/rp_test_eval.sh arch ablation real
#
# Groups (output: results/rp_test_eval/<group>/{test,valid}.json):
#   pretrain  Table VI (tab:pretrain). Real-only and fine-tuned/mixed rows at the
#             lowest five-round median of real_macro (Conv+BiGRU real-only and
#             hard-tuned at best_real_macro_raw, HarmoF0-L2 tuned rows at
#             best_real_median); synthetic-only rows at the synthetic selector
#             (easy: near_macro, hard: uniform) and, for "real-sel.", at real_macro.
#   arch      Table VIII (tab:arch): HPPNet-L2 to HRSPNet, both real streams, at
#             the lowest five-round median of real_macro; plus HRSPNet k <= 84.
#   ablation  Table IX (tab:ablation): the two comb generators synthetic-only (at
#             their synthetic selector and at real_macro), and the fine-tuning
#             seeds 0-2 of all four generators at their lowest raw real_macro
#             (best_real_macro_raw). Seeds 1-2 ran as experiment_name=<exp>_s<n>
#             of the seed-0 experiment, so they load the seed-0 config with an
#             explicit checkpoint URI.
#   real      Table IV (tab:real): HarmoF0-L2 and Conv+BiGRU on both real streams
#             (HRSPNet and HPPNet-L2 are in `arch`).
set -euo pipefail
export PYTHONPATH="$PWD/src:$PWD/scripts"
(($#)) || { echo "usage: $0 pretrain|arch|ablation|real ..." >&2; exit 2; }
SEEDS=r2://ml-data/artifacts

targets_pretrain() {
  declare -A real=(
    [pyrseplstm_k64]=rp_pyrseplstm_k64_alllab:best_real_macro_median
    [harmof0_l2]=rp_harmof0_l2_all:best_real_macro_median
    [scv2]=rp_scv2_all:best_real_macro_raw
    [hppnet_l2]=rp_hppnet_l2_alllab:best_real_macro_median
  )
  for m in pyrseplstm_k64 harmof0_l2 scv2 hppnet_l2; do
    local tuned=best_real_macro_median hard_tuned=best_real_macro_median
    [[ $m == harmof0_l2 ]] && tuned=best_real_median && hard_tuned=best_real_median
    [[ $m == scv2 ]] && hard_tuned=best_real_macro_raw
    TARGETS+=(
      "${real[$m]}"
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
}

targets_arch() {
  local e
  for e in rp_hppnet_l2 rp_hppnet_l2k64 rp_hppnet_l2k64_convlstm rp_pyrseplstm_k64 rp_pyrseplstm; do
    TARGETS+=("${e}_all:best_real_macro_median" "${e}_alllab:best_real_macro_median")
  done
}

targets_ablation() {
  TARGETS+=(
    rp_pyrseplstm_k64_combdiv:best_static_comb_median
    rp_pyrseplstm_k64_combdiv:best_real_macro_median
    rp_pyrseplstm_k64_stochcomb:best_stoch_comb_median
    rp_pyrseplstm_k64_stochcomb:best_real_macro_median
  )
  local a e
  for a in combdiv stochcomb easyv4 priorv4; do
    e=rp_pyrseplstm_k64_${a}_ft_lr5e4
    TARGETS+=(
      "${e}:best_real_macro_raw"
      "${e}:${SEEDS}/${e}_s1/checkpoints/best_real_macro_raw.ckpt"
      "${e}:${SEEDS}/${e}_s2/checkpoints/best_real_macro_raw.ckpt"
    )
  done
}

targets_real() {
  TARGETS+=(
    rp_harmof0_l2_all:best_real_macro_median rp_harmof0_l2_alllab:best_real_macro_median
    rp_scv2_all:best_real_macro_raw rp_scv2_alllab:best_real_macro_median
  )
}

nvidia-smi -L || true
for group in "$@"; do
  TARGETS=()
  "targets_${group}"
  out=results/rp_test_eval/${group}
  mkdir -p "$out"
  python scripts/rp_panel_eval.py "${TARGETS[@]}" --validation rp_real_test --out "$out/test.json"
  python scripts/rp_panel_eval.py "${TARGETS[@]}" --validation rp_real --out "$out/valid.json"
done
