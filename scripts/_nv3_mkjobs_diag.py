"""Throwaway: the GPU job scripts of the noise-v3 diagnosis (steps 2-3), one per pool.

Writes `results/noise_v3/diag/jobs/job_{dregon,cruise}_<TAG>.sh`, each running
`scripts/noise_v3_diag_rig.py` on the round-2 selected fit of its pool with the
extra arguments given, out dir `results/noise_v3/diag/rig/<TAG>`. The wrapper is
the v3 campaign's: `_nv3_upload.py` shipped base64, the synced results dir tees
the job log, R2 sync every 10 min and at the end, the support build into the
cache first. Submit with `omnirun submit ... -- bash -lc "$(cat <job>.sh)"`
from a detached worktree on the pushed HEAD.

Usage: `_nv3_mkjobs_diag.py <TAG> [noise_v3_diag_rig.py args...]`, e.g.
`_nv3_mkjobs_diag.py A --steps fit,rig,ridge,alt --rounds 2 --rig-wall-s 540`.
"""

import base64
import sys
from pathlib import Path

G = Path(__file__).resolve().parent
D = Path("results/noise_v3/diag/jobs")
OUT = "results/noise_v3/diag/rig/" + sys.argv[1]


def b64(name):
    return base64.b64encode((G / name).read_bytes()).decode()


HEAD = f"""set -a; [ -f ./.env ] && . ./.env; [ -f "${{OMNIRUN_OUTPUT%/outputs}}/.env" ] && . "${{OMNIRUN_OUTPUT%/outputs}}/.env"; set +a; export PYTHONPATH=src
PY=python; [ -n "${{VIRTUAL_ENV:-}}" ] && PY="$VIRTUAL_ENV/bin/python"
F={OUT}; C=results/noise_v2/rounds/round1/supports; mkdir -p "$F" "$C"
exec > >(tee -a "$F/@KEY@_job.log") 2>&1
T0=$(date +%s)
echo {b64("_nv3_upload.py")} | base64 -d > /tmp/upload.py
export AWS_DEFAULT_REGION=auto AWS_ENDPOINT_URL="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
JOB="${{OMNIRUN_JOB:-$(basename "${{OMNIRUN_OUTPUT%/outputs}}")}}"
DEST="s3://omnirun-artifacts/$JOB/outputs/{OUT}"
if command -v aws >/dev/null 2>&1; then AWS=aws
elif command -v uvx >/dev/null 2>&1; then AWS="uvx --quiet --from awscli aws"
else python3 -m pip install -q --user awscli >/dev/null 2>&1; AWS="python3 -m awscli"; fi
sync_out() {{ $AWS s3 sync --only-show-errors --endpoint-url "$AWS_ENDPOINT_URL" "$F" "$DEST" \\
  || $PY /tmp/upload.py "$JOB/outputs" "$F"; }}
echo "V3DIAG job=$JOB dest=$DEST sha=$(git rev-parse --short=12 HEAD 2>/dev/null) host=$(hostname) nproc=$(nproc)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
( while sleep 600; do sync_out; done ) >/dev/null 2>&1 & SYNC_PID=$!
$PY -c "import torch; print('V3DIAG torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
"""

POOLS = {
    "dregon": ("dregon-floor", "dregon_room2_floor", "--wind"),
    "cruise": ("michaels-cruise", "michaels_fly125_cruise", ""),
}
TAG = sys.argv[1]
EXTRA = " ".join(sys.argv[2:])
D.mkdir(parents=True, exist_ok=True)
for key, (set_, name, extra) in POOLS.items():
    job = (
        HEAD.replace("@KEY@", name)
        + f"""S=/tmp/supports_{name}; mkdir -p "$S"
$PY scripts/noise_v2_supports.py build --set {set_} --out "$S" > "$F/{name}_build.log" 2>&1; EB=$?
tail -2 "$F/{name}_build.log"
for f in "$S"/*.npz; do b=$(basename "$f"); [ -f "$C/$b" ] || {{ cp "$f" "$C/.tmp.$$.$b" && mv -f "$C/.tmp.$$.$b" "$C/$b"; }}; done
echo "V3DIAG build exit=$EB wall_s=$(( $(date +%s) - T0 ))"
$PY -u scripts/noise_v3_diag_rig.py --fit results/noise_v3/fits_r2/{name}__flight_v3.json --set {set_} {extra} \\
  --out "$F" {EXTRA} 2>&1 | grep -v Warn; EF=${{PIPESTATUS[0]}}
echo "V3DIAG exit=$EF total_wall_s=$(( $(date +%s) - T0 ))"
kill $SYNC_PID 2>/dev/null; sync_out
echo "V3DIAG_DONE"
"""
    )
    (D / f"job_{key}_{TAG}.sh").write_text(job)
print(sorted(p.name for p in D.glob("job_*.sh")))
