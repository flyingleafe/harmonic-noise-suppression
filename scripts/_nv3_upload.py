"""Throwaway: upload a results dir to R2 (the omnirun-artifacts bucket).

The noise-v3 job wrappers (`scripts/_nv3_mkjobs_*.py`) ship it base64 and call
it when `aws s3 sync` is unavailable. Usage: `_nv3_upload.py <prefix> <dir>`
uploads every file under `<dir>` to `s3://omnirun-artifacts/<prefix>/<dir>/...`
with the R2 credentials of `.env`."""

import os
import pathlib
import sys

import boto3

prefix, root = sys.argv[1], pathlib.Path(sys.argv[2])
s3 = boto3.client(
    "s3",
    endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
    aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
    region_name="auto",
)
n = 0
for p in sorted(root.rglob("*")):
    if p.is_file():
        s3.upload_file(str(p), "omnirun-artifacts", f"{prefix}/{p}")
        n += 1
print(f"UPLOADED {n} files to s3://omnirun-artifacts/{prefix}/{root}", flush=True)
