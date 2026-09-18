# R2 DREGON render diagnosis — the frozen-scorer job

Written by `DregonRender`. The numbers in `findings.md` and `render_dregon.json`
already exist: the frozen HPPNet ran on this LAPTOP's CPU (~5 s per arm, 20 arms
in 183 s wall), with `gates.SCORER_SHA256` verified by `Probe.load()` before
anything was scored — the digest it printed is recorded in
`render_dregon.json:protocol.scorer.sha256` and equals
`6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`.

This job is the `uni-gpushort` re-run of the SAME command at the same code SHA
and the same render seed, kept because the campaign's HPPNet numbers are
supposed to come off the cluster. Its record is merged back with
`--merge-probe`, which refuses the merge unless every shared arm's band level
agrees to within `LEVEL_TOL_DB` = 0.01 dB — i.e. unless the cluster rendered
bit-identical audio.

## The job

| field | value |
| --- | --- |
| job | `nv2-r2-dregon-render-0aa14f` |
| backend | `uni-gpushort`, `--gpus 1 --time 30m` |
| code SHA | `78638126bf3dc3565b30585e825e820bb0e74b6c` (`78638126`, pushed on `main`) |
| submitted | 2026-09-18T08:57:14Z; `starting` / scheduler `running` at 08:57:43Z (`omnirun status nv2-r2-dregon-render-0aa14f`) |
| submitted from | detached worktree `.worktrees/submit-DregonRender` |
| output the job writes | `results/noise_v2/rounds/round2/render_dregon/render_dregon.json`, `findings.md` |

## The command

```
cd .worktrees/submit-DregonRender && COLUMNS=200 omnirun submit \
  --backend uni-gpushort --gpus 1 --time 30m --name nv2-r2-dregon-render \
  --outputs 'results/noise_v2/rounds/round2/render_dregon/*.json' \
  --outputs 'results/noise_v2/rounds/round2/render_dregon/*.md' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src; python scripts/noise_v2_render_dregon.py --probe --out results/noise_v2/rounds/round2/render_dregon'
```

No `--figures`: the four committed PNGs are the laptop pass's and are identical
by construction (same seed, same renders).

## What the harvest step must do

1. `set -a; . ./.env; set +a; aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" s3://omnirun-artifacts/<job>/outputs/results/ results/` into a scratch tree, NOT over the committed record.
2. Compare the cluster `render_dregon.json` arm-by-arm against the committed one
   (`pit_mae`, `band_level_db_mic0`). They must agree; the renders are
   deterministic in `--seed 2001` and neither the renderer nor the tracker has
   a nondeterministic path on this input.
3. If they agree, the committed record stands as is and the only edit is this
   note (job id + agreement). If they do NOT agree, the CLUSTER number wins and
   the committed record is replaced by the cluster's.
