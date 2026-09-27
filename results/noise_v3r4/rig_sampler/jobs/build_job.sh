set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a
export PYTHONPATH=src OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
D=results/noise_v3r4/rig_sampler; mkdir -p "$D/banks"
exec > >(tee -a "$D/build_PRESET_job.log") 2>&1
T0=$(date +%s)
echo "NV3R4BANK preset=PRESET sha=$(git rev-parse --short=12 HEAD 2>/dev/null) host=$(hostname) nproc=$(nproc)"
$PY scripts/noise_v2_build_bank.py --preset PRESET --generation v3r4 --workers 14 --out "$D/banks/noise_v3r4_PRESET_n2048.json"; E=$?
echo "NV3R4BANK_DONE exit=$E wall_s=$(( $(date +%s) - T0 ))"
exit $E
