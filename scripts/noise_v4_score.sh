#!/usr/bin/env bash
# Noise model v4: score a bench (DREGON) and a flight (MD2) payload with the
# nv3r4 protocol (docs/experiments/noise-model-v4-bench.md § Scoring): HPPNet
# parity + LTAS proxy through scripts/noise_v2_round_score.py (frozen supports,
# render seeds 2001-2004, all 8 mics) and the Listen LTAS through
# scripts/noise_v3_latent_runaway_figs.py --listen-v4 (seed 0, 10 s, mic 0).
#
#   bash scripts/noise_v4_score.sh <bench payload> <flight payload> <out dir>
#
# CPU only (uni-cpu, 4 threads): the four steps run concurrently.
set -uo pipefail
BENCH=$1 FLIGHT=$2 O=$3
set -a; [ -f ./.env ] && . ./.env; set +a
export PYTHONPATH=src:scripts OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
L=$O/logs; mkdir -p "$L" "$O/parity" "$O/proxy"
FITS=results/noise_v2/rounds/round3/fits
step() { local name=$1; shift; local t0; t0=$(date +%s); "$@" > "$L/$name.log" 2>&1; local e=$?
  echo "STEP $name exit=$e wall_s=$(( $(date +%s) - t0 ))"; grep -v Warn "$L/$name.log" | tail -3 | cut -c1-240; }
echo "NV4SCORE bench=$BENCH flight=$FLIGHT out=$O sha=$(git rev-parse --short=12 HEAD) host=$(hostname) nproc=$(nproc)"
P=""
( step parity_dregon python scripts/noise_v2_round_score.py --round 6 --fits $FITS --rigs dregon \
    --fit dregon="$BENCH" --candidate dregon_v4 --arm-out "$O/parity/arm_dregon_v4.json" ) & P="$P $!"
( step parity_michaels python scripts/noise_v2_round_score.py --round 6 --fits $FITS --rigs michaels \
    --fit michaels="$FLIGHT" --candidate michaels_v4 --arm-out "$O/parity/arm_michaels_v4.json" ) & P="$P $!"
( step listen python scripts/noise_v3_latent_runaway_figs.py --listen-v4 dregon="$BENCH" michaels="$FLIGHT" ) & P="$P $!"
wait $P
cp docs/explainers/noise-model-v3-latent-runaway/figdata_v4.json "$O/" 2>/dev/null
echo NV4SCORE_DONE
