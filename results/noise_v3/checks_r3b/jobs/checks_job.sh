set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a
export PYTHONPATH=src:scripts OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
O=results/noise_v3/checks_r3b; L=$O/logs; mkdir -p "$L" "$O/parity" "$O/proxy_seeds"
exec > >(tee -a "$L/checks_job.log") 2>&1
T0=$(date +%s)
echo IiIiVGhyb3dhd2F5OiB1cGxvYWQgYSByZXN1bHRzIGRpciB0byBSMiAodGhlIG9tbmlydW4tYXJ0aWZhY3RzIGJ1Y2tldCkuCgpUaGUgbm9pc2UtdjMgam9iIHdyYXBwZXJzIChgc2NyaXB0cy9fbnYzX21ram9ic18qLnB5YCkgc2hpcCBpdCBiYXNlNjQgYW5kIGNhbGwKaXQgd2hlbiBgYXdzIHMzIHN5bmNgIGlzIHVuYXZhaWxhYmxlLiBVc2FnZTogYF9udjNfdXBsb2FkLnB5IDxwcmVmaXg+IDxkaXI+YAp1cGxvYWRzIGV2ZXJ5IGZpbGUgdW5kZXIgYDxkaXI+YCB0byBgczM6Ly9vbW5pcnVuLWFydGlmYWN0cy88cHJlZml4Pi88ZGlyPi8uLi5gCndpdGggdGhlIFIyIGNyZWRlbnRpYWxzIG9mIGAuZW52YC4iIiIKCmltcG9ydCBvcwppbXBvcnQgcGF0aGxpYgppbXBvcnQgc3lzCgppbXBvcnQgYm90bzMKCnByZWZpeCwgcm9vdCA9IHN5cy5hcmd2WzFdLCBwYXRobGliLlBhdGgoc3lzLmFyZ3ZbMl0pCnMzID0gYm90bzMuY2xpZW50KAogICAgInMzIiwKICAgIGVuZHBvaW50X3VybD1mImh0dHBzOi8ve29zLmVudmlyb25bJ1IyX0FDQ09VTlRfSUQnXX0ucjIuY2xvdWRmbGFyZXN0b3JhZ2UuY29tIiwKICAgIGF3c19hY2Nlc3Nfa2V5X2lkPW9zLmVudmlyb25bIkFXU19BQ0NFU1NfS0VZX0lEIl0sCiAgICBhd3Nfc2VjcmV0X2FjY2Vzc19rZXk9b3MuZW52aXJvblsiQVdTX1NFQ1JFVF9BQ0NFU1NfS0VZIl0sCiAgICByZWdpb25fbmFtZT0iYXV0byIsCikKbiA9IDAKZm9yIHAgaW4gc29ydGVkKHJvb3Qucmdsb2IoIioiKSk6CiAgICBpZiBwLmlzX2ZpbGUoKToKICAgICAgICBzMy51cGxvYWRfZmlsZShzdHIocCksICJvbW5pcnVuLWFydGlmYWN0cyIsIGYie3ByZWZpeH0ve3B9IikKICAgICAgICBuICs9IDEKcHJpbnQoZiJVUExPQURFRCB7bn0gZmlsZXMgdG8gczM6Ly9vbW5pcnVuLWFydGlmYWN0cy97cHJlZml4fS97cm9vdH0iLCBmbHVzaD1UcnVlKQo= | base64 -d > /tmp/upload_$$.py
export AWS_DEFAULT_REGION=auto AWS_ENDPOINT_URL="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
JOB="${OMNIRUN_JOB:-$(basename "${OMNIRUN_OUTPUT%/outputs}")}"
DEST="s3://omnirun-artifacts/$JOB/outputs/$O"
if command -v aws >/dev/null 2>&1; then AWS=aws
elif command -v uvx >/dev/null 2>&1; then AWS="uvx --quiet --from awscli aws"
else python3 -m pip install -q --user awscli >/dev/null 2>&1; AWS="python3 -m awscli"; fi
sync_out() { $AWS s3 sync --only-show-errors --endpoint-url "$AWS_ENDPOINT_URL" "$O" "$DEST" \
  || $PY /tmp/upload_$$.py "$JOB/outputs" "$O"; }
echo "V3CHK job=$JOB tag=r3b fits=results/noise_v3/fits_r3b out=$O sha=$(git rev-parse --short=12 HEAD 2>/dev/null) host=$(hostname) nproc=$(nproc)"
( while sleep 600; do sync_out; done ) >/dev/null 2>&1 & SYNC_PID=$!
P=""
step() { local name=$1; shift; local t0=$(date +%s); "$@" > "$L/$name.log" 2>&1; local e=$?
  echo "STEP $name exit=$e wall_s=$(( $(date +%s) - t0 ))"; grep -v Warn "$L/$name.log" | tail -2 | cut -c1-220; }
chk() { $PY scripts/noise_v3_checks.py "$1" --fits results/noise_v3/fits_r3b --out $O --keys "$2"; }
( step latents chk latents dregon,michaels_cruise,michaels_standby ) & P="$P $!"
( step heldout_dregon chk heldout dregon; step heldout_michaels chk heldout michaels_cruise,michaels_standby ) & P="$P $!"
( step rendered chk rendered dregon,michaels_cruise,michaels_standby ) & P="$P $!"
( step tonality $PY scripts/noise_v2_tonality_audit.py --out $O/tonality --fit dregon=results/noise_v3/fits_r3b/dregon_room2_floor__flight_v3.json --fit michaels=results/noise_v3/fits_r3b/michaels_fly125_cruise__flight_v3.json --fit michaels_standby=results/noise_v3/fits_r3b/michaels_fly125_standby__flight_v3.json ) & P="$P $!"
( step parity_dregon $PY scripts/noise_v2_round_score.py --round 6 --fits results/noise_v2/rounds/round3/fits --rigs dregon --fit dregon=results/noise_v3/fits_r3b/dregon_room2_floor__flight_v3.json --candidate dregon_v3 --arm-out $O/parity/arm_dregon_v3.json ) & P="$P $!"
( step parity_michaels $PY scripts/noise_v2_round_score.py --round 6 --fits results/noise_v2/rounds/round3/fits --rigs michaels --fit michaels=results/noise_v3/fits_r3b/michaels_fly125_cruise__flight_v3.json --fit michaels_standby=results/noise_v3/fits_r3b/michaels_fly125_standby__flight_v3.json --candidate michaels_v3 --arm-out $O/parity/arm_michaels_v3.json ) & P="$P $!"
( for n in 0 1 2 3; do step proxy_dregon_s$n $PY scripts/noise_v2_round_score.py --round 6 --fits results/noise_v2/rounds/round3/fits --rigs dregon --no-probe --fit dregon=results/noise_v3/fits_r3b/restarts/dregon_room2_floor__flight_v3__s$n.json --candidate dregon_r3b_s$n --arm-out $O/proxy_seeds/arm_dregon_r3b_s$n.json; done ) & P="$P $!"
( for n in 0 1 2 3; do step proxy_michaels_s$n $PY scripts/noise_v2_round_score.py --round 6 --fits results/noise_v2/rounds/round3/fits --rigs michaels --no-probe --fit michaels=results/noise_v3/fits_r3b/restarts/michaels_fly125_cruise__flight_v3__s$n.json --fit michaels_standby=results/noise_v3/fits_r3b/restarts/michaels_fly125_standby__flight_v3__s$n.json --candidate michaels_r3b_s$n --arm-out $O/proxy_seeds/arm_michaels_r3b_s$n.json; done ) & P="$P $!"
wait $P
step summary chk summary dregon,michaels_cruise,michaels_standby
kill $SYNC_PID 2>/dev/null; pkill -P $SYNC_PID 2>/dev/null; sync_out
echo "V3CHK_DONE total_wall_s=$(( $(date +%s) - T0 ))"
