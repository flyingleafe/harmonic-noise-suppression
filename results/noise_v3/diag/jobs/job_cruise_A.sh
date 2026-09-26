set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
F=results/noise_v3/diag/rig/A; C=results/noise_v2/rounds/round1/supports; mkdir -p "$F" "$C"
exec > >(tee -a "$F/michaels_fly125_cruise_job.log") 2>&1
T0=$(date +%s)
echo IiIiVXBsb2FkIGEgcmVzdWx0cyBkaXIgdG8gUjIgKHRocm93YXdheSwgc2hpcHBlZCBiYXNlNjQgaW4gdGhlIGthZ2dsZSBqb2IpLiIiIgppbXBvcnQgb3MKaW1wb3J0IHBhdGhsaWIKaW1wb3J0IHN5cwoKaW1wb3J0IGJvdG8zCgpwcmVmaXgsIHJvb3QgPSBzeXMuYXJndlsxXSwgcGF0aGxpYi5QYXRoKHN5cy5hcmd2WzJdKQpzMyA9IGJvdG8zLmNsaWVudCgKICAgICJzMyIsCiAgICBlbmRwb2ludF91cmw9ZiJodHRwczovL3tvcy5lbnZpcm9uWydSMl9BQ0NPVU5UX0lEJ119LnIyLmNsb3VkZmxhcmVzdG9yYWdlLmNvbSIsCiAgICBhd3NfYWNjZXNzX2tleV9pZD1vcy5lbnZpcm9uWyJBV1NfQUNDRVNTX0tFWV9JRCJdLAogICAgYXdzX3NlY3JldF9hY2Nlc3Nfa2V5PW9zLmVudmlyb25bIkFXU19TRUNSRVRfQUNDRVNTX0tFWSJdLAogICAgcmVnaW9uX25hbWU9ImF1dG8iLAopCm4gPSAwCmZvciBwIGluIHNvcnRlZChyb290LnJnbG9iKCIqIikpOgogICAgaWYgcC5pc19maWxlKCk6CiAgICAgICAgczMudXBsb2FkX2ZpbGUoc3RyKHApLCAib21uaXJ1bi1hcnRpZmFjdHMiLCBmIntwcmVmaXh9L3twfSIpCiAgICAgICAgbiArPSAxCnByaW50KGYiVVBMT0FERUQge259IGZpbGVzIHRvIHMzOi8vb21uaXJ1bi1hcnRpZmFjdHMve3ByZWZpeH0ve3Jvb3R9IiwgZmx1c2g9VHJ1ZSkK | base64 -d > /tmp/upload.py
export AWS_DEFAULT_REGION=auto AWS_ENDPOINT_URL="https://$R2_ACCOUNT_ID.r2.cloudflarestorage.com"
JOB="${OMNIRUN_JOB:-$(basename "${OMNIRUN_OUTPUT%/outputs}")}"
DEST="s3://omnirun-artifacts/$JOB/outputs/results/noise_v3/diag/rig/A"
if command -v aws >/dev/null 2>&1; then AWS=aws
elif command -v uvx >/dev/null 2>&1; then AWS="uvx --quiet --from awscli aws"
else python3 -m pip install -q --user awscli >/dev/null 2>&1; AWS="python3 -m awscli"; fi
sync_out() { $AWS s3 sync --only-show-errors --endpoint-url "$AWS_ENDPOINT_URL" "$F" "$DEST" \
  || $PY /tmp/upload.py "$JOB/outputs" "$F"; }
echo "V3DIAG job=$JOB dest=$DEST sha=$(git rev-parse --short=12 HEAD 2>/dev/null) host=$(hostname) nproc=$(nproc)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
( while sleep 600; do sync_out; done ) >/dev/null 2>&1 & SYNC_PID=$!
$PY -c "import torch; print('V3DIAG torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
S=/tmp/supports_michaels_fly125_cruise; mkdir -p "$S"
$PY scripts/noise_v2_supports.py build --set michaels-cruise --out "$S" > "$F/michaels_fly125_cruise_build.log" 2>&1; EB=$?
tail -2 "$F/michaels_fly125_cruise_build.log"
for f in "$S"/*.npz; do b=$(basename "$f"); [ -f "$C/$b" ] || { cp "$f" "$C/.tmp.$$.$b" && mv -f "$C/.tmp.$$.$b" "$C/$b"; }; done
echo "V3DIAG build exit=$EB wall_s=$(( $(date +%s) - T0 ))"
$PY -u scripts/noise_v3_diag_rig.py --fit results/noise_v3/fits_r2/michaels_fly125_cruise__flight_v3.json --set michaels-cruise  \
  --out "$F" --steps fit,rig,ridge,alt --rounds 2 --rig-wall-s 540 2>&1 | grep -v Warn; EF=${PIPESTATUS[0]}
echo "V3DIAG exit=$EF total_wall_s=$(( $(date +%s) - T0 ))"
kill $SYNC_PID 2>/dev/null; sync_out
echo "V3DIAG_DONE"
