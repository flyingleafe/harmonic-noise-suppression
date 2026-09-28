set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
D=results/noise_v3r4plain/rig_sampler; mkdir -p "$D/banks"
exec > >(tee -a "$D/build_hard_job.log") 2>&1
T0=$(date +%s)
echo "NV3R4PLAINBANK preset=hard sha=$(git rev-parse --short=12 HEAD 2>/dev/null) host=$(hostname) nproc=$(nproc)"
$PY scripts/noise_v2_build_bank.py --preset hard --generation v3r4plain --workers 14 --out "$D/banks/noise_v3r4plain_hard_n2048.json"; E=$?
echo "NV3R4PLAINBANK_DONE exit=$E wall_s=$(( $(date +%s) - T0 ))"
export AWS_DEFAULT_REGION=auto AWS_ENDPOINT_URL="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
JOB="${OMNIRUN_JOB:-$(basename "${OMNIRUN_OUTPUT%/outputs}")}"
if command -v aws >/dev/null 2>&1; then AWS=aws; elif command -v uvx >/dev/null 2>&1; then AWS="uvx --quiet --from awscli aws"; else python3 -m pip install -q --user awscli >/dev/null 2>&1; AWS="python3 -m awscli"; fi
$AWS s3 sync --only-show-errors "$D" "s3://omnirun-artifacts/$JOB/outputs/$D" && echo "NV3R4PLAINBANK synced to R2"
exit $E
