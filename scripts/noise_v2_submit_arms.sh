#!/usr/bin/env bash
# Submit the noise-model-v2 transfer arms (docs/experiments/noise-v2-transfer.md).
#
# Usage: scripts/noise_v2_submit_arms.sh [OPTIONS] <arm|all-synth|all-ft|mixed>
#
#   all-synth   nv2_{easy,hard}_{scv2,hppnet_l2}   — the four synthetic-only arms
#   mixed       nv2_mixed_{scv2,hppnet_l2}         — the two joint real+v2 arms
#   all-ft      nv2_{easy,hard}_ft_{scv2,hppnet_l2} — the four curriculum arms
#   <arm>       any one of the ten experiment names
#
# Options:
#   -n, --dry-run     print the submit commands and exit (nothing is submitted,
#                     no worktree is created, the daemon is not contacted)
#       --time <t>    job time estimate (default 24h)
#       --ref <ref>   git ref the worktree is pinned to (default origin/main)
#       --after <id>  omnirun job id this submission waits for (repeatable);
#                     this is how a curriculum arm is chained behind its stage 1
#
# EACH ARM GETS ITS OWN DETACHED WORKTREE at a pinned SHA, so a later edit to
# the main checkout cannot change what a queued job runs, and several arms can
# be prepared without fighting over one working tree.
#
# THE BANK IS BUILT IN THE JOB. data/rig_banks/noise_v2_{easy,hard}_n2048.json
# are gitignored build products and `omnirun` ships a clean pushed checkout, so
# every synthetic or mixed arm runs `scripts/noise_v2_build_bank.py` first. The
# build is bit-reproducible (seed 20260921) and idempotent — it skips itself
# when the provenance digest already matches — so a restarted job pays once.
# The curriculum arms train on REAL audio and build no bank.
#
# ORDERING. A curriculum arm resolves `best:real_overall@<stage 1>` to an R2
# artifact, so its stage-1 arm must have FINISHED and uploaded
# `checkpoints/best_real_overall.ckpt` before it is submitted. Use `--after`
# to let the scheduler enforce that.
set -euo pipefail

DRY=0
TIME=24h
REF=origin/main
AFTER=()

die() { printf 'noise_v2_submit_arms: %s\n' "$*" >&2; exit 1; }

SYNTH=(nv2_easy_scv2 nv2_hard_scv2 nv2_easy_hppnet_l2 nv2_hard_hppnet_l2)
MIXED=(nv2_mixed_scv2 nv2_mixed_hppnet_l2)
FT=(nv2_easy_ft_scv2 nv2_hard_ft_scv2 nv2_easy_ft_hppnet_l2 nv2_hard_ft_hppnet_l2)

TARGET=""
while [ $# -gt 0 ]; do
  case "$1" in
    -n|--dry-run) DRY=1; shift ;;
    --time) TIME="${2:?--time needs a value}"; shift 2 ;;
    --ref) REF="${2:?--ref needs a value}"; shift 2 ;;
    --after) AFTER+=(--after "${2:?--after needs a job id}"); shift 2 ;;
    -h|--help) sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) die "unknown option: $1" ;;
    *) [ -z "$TARGET" ] || die "one target at a time, got '$TARGET' and '$1'"; TARGET="$1"; shift ;;
  esac
done
[ -n "$TARGET" ] || die "no target; try --help"

case "$TARGET" in
  all-synth) ARMS=("${SYNTH[@]}") ;;
  mixed)     ARMS=("${MIXED[@]}") ;;
  all-ft)    ARMS=("${FT[@]}") ;;
  *)
    ARMS=("$TARGET")
    found=0
    for a in "${SYNTH[@]}" "${MIXED[@]}" "${FT[@]}"; do [ "$a" = "$TARGET" ] && found=1; done
    [ "$found" = 1 ] || die "unknown arm '$TARGET'; expected one of ${SYNTH[*]} ${MIXED[*]} ${FT[*]} or all-synth|all-ft|mixed"
    ;;
esac

# The bank preset each arm needs, or empty for an arm that trains on real audio.
preset_of() {
  case "$1" in
    nv2_easy_*ft*|nv2_hard_*ft*) printf '' ;;
    *_easy_*) printf 'easy' ;;
    *_hard_*|*_mixed_*) printf 'hard' ;;
  esac
}

# The environment wrapper every job runs before anything else: the repo's own
# .env, then the job-local one omnirun drops next to its outputs directory, and
# src/ on PYTHONPATH.
ENV_WRAPPER='set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src'

SHA="$(git rev-parse --verify "${REF}^{commit}")"
printf 'pinned %s = %s\n' "$REF" "$(git rev-parse --short "$SHA")" >&2

for arm in "${ARMS[@]}"; do
  preset="$(preset_of "$arm")"
  job="${arm//_/-}"
  payload="$ENV_WRAPPER"
  [ -n "$preset" ] && payload="$payload; python scripts/noise_v2_build_bank.py --preset $preset"
  payload="$payload; python train.py experiment=$arm"

  cmd=(omnirun --daemon localhost:18787 submit --backend uni --gpus 1
       --time "$TIME" --name "$job" "${AFTER[@]}" -- bash -lc "$payload")

  printf '\n# %s\n' "$arm"
  printf 'WT=$(scripts/mk-worktree.sh submit-%s %s) && git -C "$WT" checkout --detach %s\n' \
    "$job" "$REF" "$SHA"
  after_str=""
  [ ${#AFTER[@]} -gt 0 ] && after_str="$(printf ' %s' "${AFTER[@]}")"
  printf 'cd "$WT" && omnirun --daemon localhost:18787 submit --backend uni --gpus 1 --time %s --name %s%s -- bash -lc '\''%s'\''\n' \
    "$TIME" "$job" "$after_str" "$payload"

  [ "$DRY" = 1 ] && continue

  WT="$(scripts/mk-worktree.sh "submit-$job" "$REF")"
  git -C "$WT" checkout --detach "$SHA"
  ( cd "$WT" && "${cmd[@]}" )
done
