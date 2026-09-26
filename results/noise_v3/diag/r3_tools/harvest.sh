#!/usr/bin/env bash
# Throwaway: pull one job's R2 outputs; copy a fit dir's restarts + logs into the repo.
# Usage: harvest.sh <job-id> <results subdir, e.g. results/noise_v3/fits_r3a>
set -uo pipefail
cd /home/flyingleafe/Research/PhD/projects/harmonic-noise-suppression
set -a; . ./.env; set +a
export AWS_ENDPOINT_URL="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com" AWS_DEFAULT_REGION=auto
job=$1; sub=$2
d=/tmp/r3/pull/$job; mkdir -p "$d"
timeout 900 aws s3 sync --only-show-errors --exclude "*/grid/*" "s3://omnirun-artifacts/$job/outputs/$sub" "$d/"
find "$d" -maxdepth 2 -type f -printf '%s %P\n' | sort -k2
mkdir -p "$sub/restarts" "$sub/logs"
[ -d "$d/restarts" ] && cp "$d"/restarts/*.json "$sub/restarts/" 2>/dev/null
[ -d "$d/logs" ] && cp "$d"/logs/* "$sub/logs/" 2>/dev/null
cp "$d"/*_job.log "$d"/*_build.log "$sub/" 2>/dev/null
grep -hE "V3R3 (seed|build|_DONE)|V3R3_DONE|A100|Tesla|H100" "$d"/*_job.log 2>/dev/null | cut -c1-160
true
