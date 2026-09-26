"""Throwaway: the round-3 noise-v3 fit jobs, one per pool with seeds 0-3.

Each job builds the pool's supports once, then runs
`scripts/noise_v2_fit.py flight --mode flight_v3` per seed with the round-1
campaign CLI (`--init-from` the same v2 fits, `--channel-gains`, `--wind` on
DREGON, `--max-frames 0`, `--restart-tag sN`) plus the given schedule, under
the wander `results/noise_v3/wander/<rig><suffix>.json`. The wrapper is the v3
campaign's: helpers `_nv3_upload.py`/`_nv3_fit_summary.py` shipped base64, the
synced out dir tees the job log, R2 sync every 10 min and after every seed, a
failed seed's grid `.err`/`.log` echoed. With `--parallel` the seeds run as
concurrent processes on the one GPU (each logs to `<out>/logs/<name>_sN.log`).

Writes `<out>/jobs/r3_job_<pool>.sh`; submit with
`omnirun submit ... -- bash -lc "$(cat <job>.sh)"` from a detached worktree on
the pushed HEAD.

Usage: `_nv3_mkjobs_r3.py --sched '--ridge-step --lbfgs-rtol 0 --rounds 20
--lbfgs-frames 64 --lbfgs-iters 150' --out results/noise_v3/fits_r3a [--wander-suffix _ujk]
[--seeds '0 1 2 3'] [--parallel]`.
"""

import argparse
import base64
from pathlib import Path

G = Path(__file__).resolve().parent

POOLS = {
    "dregon": (
        "dregon-floor",
        "dregon_room2_floor",
        "dregon",
        "--wind",
        "results/noise_v2/rounds/round5/fits/dregon_room2_floor__flight_profile.json",
    ),
    "cruise": (
        "michaels-cruise",
        "michaels_fly125_cruise",
        "michaels",
        "",
        "results/noise_v2/rounds/round3/fits/michaels_fly125_cruise__flight.json",
    ),
    "standby": (
        "michaels-standby",
        "michaels_fly125_standby",
        "michaels",
        "",
        "results/noise_v2/rounds/round3/fits/michaels_fly125_standby__flight.json",
    ),
}


def b64(name):
    return base64.b64encode((G / name).read_bytes()).decode()


def head(out: str, name: str) -> str:
    return f"""set -a; [ -f ./.env ] && . ./.env; [ -f "${{OMNIRUN_OUTPUT%/outputs}}/.env" ] && . "${{OMNIRUN_OUTPUT%/outputs}}/.env"; set +a; export PYTHONPATH=src
PY=python; [ -n "${{VIRTUAL_ENV:-}}" ] && PY="$VIRTUAL_ENV/bin/python"
F={out}; C=results/noise_v2/rounds/round1/supports; mkdir -p "$F/logs" "$C"
exec > >(tee -a "$F/{name}_job.log") 2>&1
T0=$(date +%s)
echo {b64("_nv3_upload.py")} | base64 -d > /tmp/upload.py
echo {b64("_nv3_fit_summary.py")} | base64 -d > /tmp/summ.py
export AWS_DEFAULT_REGION=auto AWS_ENDPOINT_URL="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
JOB="${{OMNIRUN_JOB:-$(basename "${{OMNIRUN_OUTPUT%/outputs}}")}}"
DEST="s3://omnirun-artifacts/$JOB/outputs/{out}"
if command -v aws >/dev/null 2>&1; then AWS=aws
elif command -v uvx >/dev/null 2>&1; then AWS="uvx --quiet --from awscli aws"
else python3 -m pip install -q --user awscli >/dev/null 2>&1; AWS="python3 -m awscli"; fi
sync_out() {{ $AWS s3 sync --only-show-errors --endpoint-url "$AWS_ENDPOINT_URL" "$F" "$DEST" \\
  || $PY /tmp/upload.py "$JOB/outputs" "$F"; }}
dump_err() {{ for f in "$1"/raw/*.err "$1"/raw/*.log; do [ -f "$f" ] && {{ echo "=== $f"; cat "$f"; }}; done; }}
stamp() {{ while IFS= read -r l; do printf '%s %s\\n' "$(date -u +%H:%M:%S)" "$l"; done; }}
echo "V3R3 job=$JOB dest=$DEST sha=$(git rev-parse --short=12 HEAD 2>/dev/null) host=$(hostname) nproc=$(nproc)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
( while sleep 600; do sync_out; done ) >/dev/null 2>&1 & SYNC_PID=$!
$PY -c "import torch; print('V3R3 torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
"""


def job(pool: str, args: argparse.Namespace) -> str:
    set_, name, rig, extra, init = POOLS[pool]
    wander = f"results/noise_v3/wander/{rig}{args.wander_suffix}.json"
    fit = f"""  $PY -u scripts/noise_v2_fit.py flight --mode flight_v3 --set {set_} --name {name} \\
    --wander {wander} \\
    --channel-gains results/noise_v2/mic_gains/mic_gains.json --channel-gains-rig {rig} {extra} \\
    --init-from {init} {args.sched} --max-frames 0 --seed $N --restart-tag s$N --jobs 1 --threads 4 --device cuda \\
    --progress 50 --out "$F" --grid-dir "$F/grid/{name}_s$N" 2>&1 | grep -v Warn | stamp"""
    seed = f"""run_seed() {{
  N=$1; T1=$(date +%s)
  echo "V3R3 seed=$N start"
{fit} > "$F/logs/{name}_s$N.log"; EF=${{PIPESTATUS[0]}}
  echo "V3R3 seed=$N fit exit=$EF fit_wall_s=$(( $(date +%s) - T1 )) total_wall_s=$(( $(date +%s) - T0 ))"
  tail -3 "$F/logs/{name}_s$N.log"
  [ $EF -ne 0 ] && dump_err "$F/grid/{name}_s$N"
  $PY /tmp/summ.py "$F/restarts/{name}__flight_v3__s$N.json" || true
  sync_out
}}
"""
    loop = (
        # wait on the seed PIDs only: a bare `wait` also waits for the sync loop,
        # which never exits (every round-3 job hung after its last seed)
        f'P=""; for N in {args.seeds}; do run_seed $N & P="$P $!"; sleep 20; done; wait $P\n'
        if args.parallel
        else f"for N in {args.seeds}; do run_seed $N; done\n"
    )
    return (
        head(args.out, name)
        + f"""S=/tmp/supports_{name}; mkdir -p "$S"
echo "V3R3 spec: set={set_} name={name} wander={wander} extra='{extra}' init={init} sched='{args.sched}' seeds='{args.seeds}' parallel={int(args.parallel)}"
$PY scripts/noise_v2_supports.py build --set {set_} --out "$S" > "$F/{name}_build.log" 2>&1; EB=$?
tail -2 "$F/{name}_build.log"
for f in "$S"/*.npz; do b=$(basename "$f"); [ -f "$C/$b" ] || {{ cp "$f" "$C/.tmp.$$.$b" && mv -f "$C/.tmp.$$.$b" "$C/$b"; }}; done
echo "V3R3 build exit=$EB wall_s=$(( $(date +%s) - T0 ))"
"""
        + seed
        + loop
        + """kill $SYNC_PID 2>/dev/null; sync_out
echo "V3R3_DONE total_wall_s=$(( $(date +%s) - T0 ))"
"""
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--sched", required=True, help="schedule flags passed to the fit CLI")
    ap.add_argument("--out", required=True, help="fit out dir, e.g. results/noise_v3/fits_r3a")
    ap.add_argument("--wander-suffix", default="", help="wander file <rig><suffix>.json")
    ap.add_argument("--seeds", default="0 1 2 3")
    ap.add_argument("--pools", default="dregon,cruise,standby")
    ap.add_argument("--parallel", action="store_true", help="run the seeds concurrently")
    args = ap.parse_args()
    d = Path(args.out) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    for pool in args.pools.split(","):
        (d / f"r3_job_{pool}.sh").write_text(job(pool, args))
    print(sorted(str(p) for p in d.glob("r3_job_*.sh")))


if __name__ == "__main__":
    main()
