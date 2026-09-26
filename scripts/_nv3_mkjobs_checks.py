"""Throwaway: one uni-cpu job running the noise-v3 check set on one round's fits.

The check set of rounds 1-2 (`docs/experiments/noise-model-v3.md` § Results and
§ Round 2 against round 1), which ran on the laptop, as one CPU job:

* `noise_v3_checks.py latents|heldout|rendered` (checks (e), (b), (c) renders)
  on the reduced fits `<fits>/<pool>__flight_v3.json`, then `summary`;
* `noise_v2_tonality_audit.py --fit` (check (c) expectation);
* `noise_v2_round_score.py --round 6 ... --fit` per rig (check (d) parity, the
  frozen HPPNet probe) into `<out>/parity/arm_<rig>_v3.json`;
* the LTAS proxy of every restart (`--no-probe`) into
  `<out>/proxy_seeds/arm_<rig>_<tag>_s<N>.json`.

The independent steps run as concurrent groups of 4 threads each; `summary`
runs last. The fits must be in the pushed HEAD the job checks out. Writes
`<out>/jobs/checks_job.sh`; submit with `omnirun submit --backend uni-cpu
--gpus 0 ... -- bash -lc "$(cat <job>.sh)"` from a detached worktree.

Usage: `_nv3_mkjobs_checks.py --tag r3a --fits results/noise_v3/fits_r3a
--out results/noise_v3/checks_r3a`.
"""

import argparse
import base64
from pathlib import Path

G = Path(__file__).resolve().parent


def b64(name):
    return base64.b64encode((G / name).read_bytes()).decode()


def job(tag: str, fits: str, out: str, seeds: str) -> str:
    r = f"{fits}/restarts"
    d = f"{fits}/dregon_room2_floor__flight_v3.json"
    c = f"{fits}/michaels_fly125_cruise__flight_v3.json"
    s = f"{fits}/michaels_fly125_standby__flight_v3.json"
    score = "scripts/noise_v2_round_score.py --round 6 --fits results/noise_v2/rounds/round3/fits"
    return f"""set -a; [ -f ./.env ] && . ./.env; [ -f "${{OMNIRUN_OUTPUT%/outputs}}/.env" ] && . "${{OMNIRUN_OUTPUT%/outputs}}/.env"; set +a
export PYTHONPATH=src:scripts OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=python; [ -n "${{VIRTUAL_ENV:-}}" ] && PY="$VIRTUAL_ENV/bin/python"
O={out}; L=$O/logs; mkdir -p "$L" "$O/parity" "$O/proxy_seeds"
exec > >(tee -a "$L/checks_job.log") 2>&1
T0=$(date +%s)
echo {b64("_nv3_upload.py")} | base64 -d > /tmp/upload_$$.py
export AWS_DEFAULT_REGION=auto AWS_ENDPOINT_URL="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
JOB="${{OMNIRUN_JOB:-$(basename "${{OMNIRUN_OUTPUT%/outputs}}")}}"
DEST="s3://omnirun-artifacts/$JOB/outputs/$O"
if command -v aws >/dev/null 2>&1; then AWS=aws
elif command -v uvx >/dev/null 2>&1; then AWS="uvx --quiet --from awscli aws"
else python3 -m pip install -q --user awscli >/dev/null 2>&1; AWS="python3 -m awscli"; fi
sync_out() {{ $AWS s3 sync --only-show-errors --endpoint-url "$AWS_ENDPOINT_URL" "$O" "$DEST" \\
  || $PY /tmp/upload_$$.py "$JOB/outputs" "$O"; }}
echo "V3CHK job=$JOB tag={tag} fits={fits} out=$O sha=$(git rev-parse --short=12 HEAD 2>/dev/null) host=$(hostname) nproc=$(nproc)"
( while sleep 600; do sync_out; done ) >/dev/null 2>&1 & SYNC_PID=$!
P=""
step() {{ local name=$1; shift; local t0=$(date +%s); "$@" > "$L/$name.log" 2>&1; local e=$?
  echo "STEP $name exit=$e wall_s=$(( $(date +%s) - t0 ))"; grep -v Warn "$L/$name.log" | tail -2 | cut -c1-220; }}
chk() {{ $PY scripts/noise_v3_checks.py "$1" --fits {fits} --out $O --keys "$2"; }}
( step latents chk latents dregon,michaels_cruise,michaels_standby ) & P="$P $!"
( step heldout_dregon chk heldout dregon; step heldout_michaels chk heldout michaels_cruise,michaels_standby ) & P="$P $!"
( step rendered chk rendered dregon,michaels_cruise,michaels_standby ) & P="$P $!"
( step tonality $PY scripts/noise_v2_tonality_audit.py --out $O/tonality --fit dregon={d} --fit michaels={c} --fit michaels_standby={s} ) & P="$P $!"
( step parity_dregon $PY {score} --rigs dregon --fit dregon={d} --candidate dregon_v3 --arm-out $O/parity/arm_dregon_v3.json ) & P="$P $!"
( step parity_michaels $PY {score} --rigs michaels --fit michaels={c} --fit michaels_standby={s} --candidate michaels_v3 --arm-out $O/parity/arm_michaels_v3.json ) & P="$P $!"
( for n in {seeds}; do step proxy_dregon_s$n $PY {score} --rigs dregon --no-probe --fit dregon={r}/dregon_room2_floor__flight_v3__s$n.json --candidate dregon_{tag}_s$n --arm-out $O/proxy_seeds/arm_dregon_{tag}_s$n.json; done ) & P="$P $!"
( for n in {seeds}; do step proxy_michaels_s$n $PY {score} --rigs michaels --no-probe --fit michaels={r}/michaels_fly125_cruise__flight_v3__s$n.json --fit michaels_standby={r}/michaels_fly125_standby__flight_v3__s$n.json --candidate michaels_{tag}_s$n --arm-out $O/proxy_seeds/arm_michaels_{tag}_s$n.json; done ) & P="$P $!"
wait $P
step summary chk summary dregon,michaels_cruise,michaels_standby
kill $SYNC_PID 2>/dev/null; pkill -P $SYNC_PID 2>/dev/null; sync_out
echo "V3CHK_DONE total_wall_s=$(( $(date +%s) - T0 ))"
"""


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--tag", required=True, help="round tag, e.g. r3a")
    ap.add_argument("--fits", required=True, help="dir of the reduced fits (and restarts/)")
    ap.add_argument("--out", required=True, help="checks out dir, e.g. results/noise_v3/checks_r3a")
    ap.add_argument("--seeds", default="0 1 2 3")
    args = ap.parse_args()
    d = Path(args.out) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "checks_job.sh"
    p.write_text(job(args.tag, args.fits, args.out, args.seeds))
    print(p)


if __name__ == "__main__":
    main()
