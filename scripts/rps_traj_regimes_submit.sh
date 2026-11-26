#!/usr/bin/env bash
# Submit the flight-regime campaign (docs/experiments/rps-flight-regimes.md).
#
# Usage: scripts/rps_traj_regimes_submit.sh [OPTIONS] <rig>...|all|bench
#
#   <rig>   one uni-cpu job for that rig: the held-out comparison of the cruise
#           models on every fold, then the fits on every flight (the rig's
#           parameter set), then both passes again for whole flights (a no-op
#           on a rig whose recordings start in the air)
#   all     every rig of the campaign
#   bench   one uni-gpushort job: the Gaussian stage timed on CPU and on a GPU
#
# Options:
#   -n, --dry-run     print the submit commands and exit (nothing is submitted,
#                     no worktree is created, the daemon is not contacted)
#       --ref <ref>   git ref the worktree is pinned to (default origin/main)
#       --time <t>    time estimate of a rig job (default 12h)
#       --cpus <n>    CPU cores of a rig job (default 16); units run two
#                     threads each, n/2 at a time
#       --mem <gb>    RAM of a rig job in GB (default 48)
#
# ALL JOBS ARE SUBMITTED FROM ONE DETACHED WORKTREE at the pinned SHA
# (.worktrees/submit-rps-regimes): omnirun refuses a dirty tree, and a later
# edit to the main checkout must not change what a queued job runs.  Each job
# pushes its own files to R2; collect them in the main checkout with
# `python scripts/r2sync.py pull rps_traj/regimes` (lands in results/).
#
# A unit that fails leaves <uid>.err beside the JSONs and the rest go on; the
# job prints the number of .err files as its last line and fails if any pass
# did (its files are pushed either way).  Re-submitting a rig at the same
# SHA resumes it: the cluster worktree keeps the finished units.
set -euo pipefail

die() { printf 'rps_traj_regimes_submit: %s\n' "$*" >&2; exit 1; }

RIGS=(michaels vid_m100 dregon blackbird_quad nanobench_cf21b neurobem_quad pitcn_quad)
OUT=results/rps_traj/regimes

DRY=0
REF=origin/main
TIME=12h
CPUS=16
MEM=48
TARGETS=()
while [ $# -gt 0 ]; do
  case "$1" in
    -n|--dry-run) DRY=1; shift ;;
    --ref) REF="${2:?--ref needs a value}"; shift 2 ;;
    --time) TIME="${2:?--time needs a value}"; shift 2 ;;
    --cpus) CPUS="${2:?--cpus needs a value}"; shift 2 ;;
    --mem) MEM="${2:?--mem needs a value}"; shift 2 ;;
    -h|--help) sed -n '2,31p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    -*) die "unknown option: $1" ;;
    all) TARGETS+=("${RIGS[@]}"); shift ;;
    *) TARGETS+=("$1"); shift ;;
  esac
done
[ "${#TARGETS[@]}" -gt 0 ] || die "no target; try --help"
for t in "${TARGETS[@]}"; do
  [ "$t" = bench ] && continue
  found=0
  for r in "${RIGS[@]}"; do [ "$r" = "$t" ] && found=1; done
  [ "$found" = 1 ] || die "unknown target '$t'; expected one of ${RIGS[*]}, all or bench"
done

# The environment every job runs in: the repo's own .env, then the job-local
# one omnirun drops next to its outputs directory, src/ on PYTHONPATH, and a
# numba cache of its own (same-SHA jobs share one cluster worktree).
ENV_WRAPPER='set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH="$PWD/src"; export NUMBA_CACHE_DIR="${TMPDIR:-/tmp}/numba-${SLURM_JOB_ID:-$$}"'

SHA="$(git rev-parse --verify "${REF}^{commit}")"
printf 'pinned %s = %s\n' "$REF" "$(git rev-parse --short "$SHA")" >&2

submit() {  # submit <printable name> <omnirun argv...>
  local name="$1"; shift
  printf '\n# %s\ncd .worktrees/submit-rps-regimes &&' "$name"
  printf ' %q' "$@"
  printf '\n'
  [ "$DRY" = 1 ] && return 0
  ( cd "$WT" && "$@" )
}

if [ "$DRY" = 0 ]; then
  WT="$(scripts/mk-worktree.sh submit-rps-regimes "$REF")"
  git -C "$WT" checkout --detach "$SHA"
  # results/rps_traj holds a tracked file, so mk-worktree leaves it a real
  # directory: link the campaign's own (ignored) subdirectory instead
  MAIN="$(git rev-parse --path-format=absolute --git-common-dir)"
  MAIN="${MAIN%/.git}"
  mkdir -p "$MAIN/$OUT"
  [ -e "$WT/$OUT" ] || ln -s "$MAIN/$OUT" "$WT/$OUT"
fi

for t in "${TARGETS[@]}"; do
  if [ "$t" = bench ]; then
    payload="$ENV_WRAPPER; python scripts/rps_traj_regimes.py bench --rigs michaels pitcn_quad"
    payload="$payload && python scripts/r2sync.py push $OUT/bench_device.json"
    submit bench omnirun submit --backend uni-gpushort --gpus 1 --cpus 8 --mem 32 --time 20m \
      --name rps-regimes-bench -- bash -lc "$payload"
    continue
  fi
  run="python scripts/rps_traj_regimes.py run --rigs $t --jobs $((CPUS / 2)) --threads 2"
  payload="$ENV_WRAPPER; st=0; $run --stage cruise || st=1; $run --stage cruise --final || st=1"
  payload="$payload; $run --stage flight || st=1"
  payload="$payload; $run --stage flight --final --arms ours msvar hsmm || st=1"
  payload="$payload; python scripts/r2sync.py push $OUT/raw/*__${t}__* >/dev/null || st=1"
  payload="$payload; echo \"failed units: \$(ls $OUT/raw/*__${t}__*.err 2>/dev/null | wc -l)\""
  payload="$payload; exit \$st"
  submit "$t" omnirun submit --backend uni-cpu --gpus 0 --cpus "$CPUS" --mem "$MEM" --time "$TIME" \
    --name "rps-regimes-${t//_/-}" -- bash -lc "$payload"
done
