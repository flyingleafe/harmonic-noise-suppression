# R2 score arms on `kaggle` — FAILED on missing R2 credentials

Written by `R2Close`. Both kaggle re-pins died the same way; no arm was scored.

## The jobs

| field | michaels | dregon v2b |
| --- | --- | --- |
| job | `nv2-r2-score-michaels-kg-9740a8` | `nv2-r2-score-dregon-v2b--62f1c9` |
| backend | `kaggle`, `--gpus 1 --time 30m` | `kaggle`, `--gpus 1 --time 30m` |
| code SHA | `643b3058` (`.worktrees/submit-R2Close`, detached) | same |
| submitted | 2026-09-18T19:05Z | 2026-09-18T19:06Z |
| started | 19:06:30Z | 19:08:44Z |
| outcome | **failed, exit 1, 2 min in** | same failure (see below) |

These replaced the two `uni-gpushort` arms (`nv2-r2-score-michaels2-78d01c`,
`nv2-r2-score-dregon-v2b-41d97e`), which sat `queued` from 10:59/11:13Z to
19:05Z without ever being placed and were cancelled once the kaggle jobs were
accepted. The daemon itself was unreachable 16:39Z-19:04Z (wg tunnel down);
since 19:04Z every call goes through `omnirun --daemon localhost:18787`
(ssh tunnel). History: `round2/render/submit_note_kaggle.md`.

## The cause — a credential gap on this backend, not a code or data problem

The scorer needs the frozen HPPNet probe checkpoint
`r2://ml-data/artifacts/hppnet_l2_r2_s0/checkpoints/best.ckpt`.
`src/utils/checkpoints.py:resolve_checkpoint_uri` resolves `r2://` URIs through
`load_r2_env()`, which needs `R2_ACCOUNT_ID`, `AWS_ACCESS_KEY_ID` and
`AWS_SECRET_ACCESS_KEY`. The job wrapper obtains them with
`set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a`
— which works on the `uni*` backends, where a project `.env` exists on the
shared filesystem. On kaggle NEITHER path exists: the run tree is
`/kaggle/tmp/omnirun/projects/harmonic-noise-suppression/.trees/643b3058c81c`
and `.env` is gitignored, so it is not in the pushed tree and not on the node.

So the run dies before any scoring work, 35 s after `running`:

```
RuntimeError: r2:// checkpoint 'r2://ml-data/artifacts/hppnet_l2_r2_s0/checkpoints/best.ckpt' requested but R2 creds missing in .env
```

Note what this also means: the checkpoint is never fetched, so the sha256
verification the user asked to see in the first log lines never runs either —
there is nothing to confirm, the failure is upstream of it. The environment
itself was fine: `dload-ml==0.3.1`, torch 2.7.0 + torchaudio 2.7.0 all
installed, no missing dependency.

## The fix available without touching code

`omnirun submit --env KEY=VALUE` forwards environment into the job (the repo
already uses it for `PYTHONPATH`/`RESULTS_ROOT`, e.g. `scripts/chain_train.sh`).
Re-pinning with

```
--env R2_ACCOUNT_ID=... --env AWS_ACCESS_KEY_ID=... --env AWS_SECRET_ACCESS_KEY=...
```

supplies exactly what `load_r2_env()` wants; nothing in the scorer changes and
the arm stays comparable. The caveat is that the values then live in the job
spec on the daemon, which is a secrets-handling decision for the user, not for
an agent — so it is reported, not done unilaterally.

## Log tail (last 40 lines, michaels job)

```
+ widgetsnbextension==4.0.15
+ wrapt==2.4.0
+ xdg==6.0.0
+ xxhash==3.6.0
+ xyzservices==2025.11.0
+ yarl==1.22.0
+ zipp==4.1.0
+ zstandard==0.25.0
OMNIRUN: [2026-09-18T19:08:01Z] running 
/kaggle/tmp/omnirun/projects/harmonic-noise-suppression/.venv/lib/python3.12/site-packages/pyro/ops/stats.py:527: SyntaxWarning: invalid escape sequence '\g'
we have :math:`ES^{*}(P,Q) \ge ES^{*}(Q,Q)` with equality holding if and only if :math:`P=Q`, i.e.
Traceback (most recent call last):
File "/kaggle/tmp/omnirun/projects/harmonic-noise-suppression/.trees/643b3058c81c/scripts/noise_v2_round_score.py", line 2169, in <module>
sys.exit(main())
^^^^^^
File "/kaggle/tmp/omnirun/projects/harmonic-noise-suppression/.trees/643b3058c81c/scripts/noise_v2_round_score.py", line 2133, in main
payload = run(
^^^^
File "/kaggle/tmp/omnirun/projects/harmonic-noise-suppression/.trees/643b3058c81c/scripts/noise_v2_round_score.py", line 864, in run
probe = Probe.load() if with_probe else None
^^^^^^^^^^^^
File "/kaggle/tmp/omnirun/projects/harmonic-noise-suppression/.trees/643b3058c81c/scripts/noise_v2_round_score.py", line 518, in load
record = ev.checkpoint_record(GT.SCORER_EXPERIMENT, GT.SCORER_CKPT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/kaggle/tmp/omnirun/projects/harmonic-noise-suppression/.trees/643b3058c81c/scripts/stochastic_fit_revised_eval.py", line 115, in checkpoint_record
local = Path(resolve_checkpoint_uri(ref, REPO_ROOT / ".cache" / "r2_checkpoints"))
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
File "/kaggle/tmp/omnirun/projects/harmonic-noise-suppression/.trees/643b3058c81c/src/utils/checkpoints.py", line 62, in resolve_checkpoint_uri
raise RuntimeError(f"r2:// checkpoint {uri!r} requested but R2 creds missing in .env")
RuntimeError: r2:// checkpoint 'r2://ml-data/artifacts/hppnet_l2_r2_s0/checkpoints/best.ckpt' requested but R2 creds missing in .env
OMNIRUN: [2026-09-18T19:08:36Z] collecting 
OMNIRUN: [2026-09-18T19:08:36Z] done 
OMNIRUN: [2026-09-18T19:08:36Z] finished with exit code 1
OMNIRUN: harness finished, bootstrap exit code 1
/usr/local/lib/python3.12/dist-packages/mistune.py:435: SyntaxWarning: invalid escape sequence '\|'
cells[i][c] = re.sub('\\\\\|', '|', cell)
/usr/local/lib/python3.12/dist-packages/nbconvert/filters/filter_links.py:36: SyntaxWarning: invalid escape sequence '\_'
text = re.sub(r'_', '\_', text) # Escape underscores in display text
[NbConvertApp] Converting notebook __script__.ipynb to html
[NbConvertApp] Writing 299873 bytes to __results__.html
```
