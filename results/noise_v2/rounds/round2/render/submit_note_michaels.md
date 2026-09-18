# R2 Michael's score arm — attempt 1 FAILED (code gate), attempt 2 submitted and STILL QUEUED

Written by `R2MichaelsHarvest`. Nothing in `arm_michaels_v2.json` exists yet;
this note is the handover (job ids + exact commands) in case the harness dies
before the arm lands.

## Attempt 1 — FAILED, and why

`nv2-r2-score-michaels-5099b4` (`uni-gpushort`, SHA `c089d978`, submitted
2026-09-18T09:30:21Z, queued 63 min behind another project's job burst,
started 10:33:40Z) died 23 s into the run with **exit code 1** and one line:

```
error: results/noise_v2/rounds/round2/fits/michaels_fly125_all__flight.json:
carries support 'michaels_fly125_all', but the michaels arm is defined on
'michaels_fly125_cruise'
```

That was a CODE gate, not a data problem: `scripts/noise_v2_round_score.py`
held a single hard-coded `MICHAELS_FIT_SUPPORT = "michaels_fly125_cruise"`
(R1's cruise-only pool) and `v2_arm` refused any other support name, so R2's
whole point — a pooled standby + ramp + cruise fit under a NEW support name —
could not be scored. Fixed in `e6c06769`: the constant is now
`MICHAELS_FIT_SUPPORTS = ("michaels_fly125_all", "michaels_fly125_cruise")`,
`v2_arm` accepts either (directory selection prefers the first present), the
arm label states which pool it rendered from, and the DREGON side is
untouched. Verified before resubmitting: the pooled fit resolves with the
pooled label, R1's cruise fit still resolves with its own label, directory
selection picks the pooled fit in `round2/fits` and the cruise fit in
`round1/fits`, and a DREGON fit passed as Michael's (or the reverse) is still
rejected.

## Attempt 2 — the job

| field | value |
| --- | --- |
| job | `nv2-r2-score-michaels2-78d01c` |
| backend | `uni-gpushort`, `--gpus 1 --time 45m` |
| code SHA | `e6c06769d1d4542311cf7a75cda60b16c1403f50` (`e6c06769`, `origin/main` — carries the harvested flight fit AND the arm-selection fix) |
| submitted | 2026-09-18T11:04Z |
| status at submit | `queued` ("no slot free right now"; any later `omnirun ps`/`status`/`tick` retries placement — expect a long queue, attempt 1 waited 63 min) |
| submitted from | detached worktree `.worktrees/submit-R2MichaelsHarvest` at `e6c06769` |
| output the job writes | `results/noise_v2/rounds/round2/render/arm_michaels_v2.json` (plus the runner's own `round2/render/score/findings.md`) |

## The command (identical in both attempts apart from the name and the SHA)

```
cd .worktrees/submit-R2MichaelsHarvest && COLUMNS=200 omnirun submit \
  --backend uni-gpushort --gpus 1 --time 45m --name nv2-r2-score-michaels2 \
  --outputs 'results/noise_v2/rounds/round2/render/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 2 --fits results/noise_v2/rounds/round2/fits --rigs michaels --fit michaels=results/noise_v2/rounds/round2/fits/michaels_fly125_all__flight.json --candidate michaels_v2_r2_all --arm-out results/noise_v2/rounds/round2/render/arm_michaels_v2.json'
```

This is the DREGON arm's recipe (`round2/render/submit_note.md`) with the rig,
the `--fit`, the candidate name and the arm path moved to Michael's. No
`--dump-audio`: R1's first Michael's attempt was cut off mid-upload by the
~100 MB audio dump, and the arm record is what the round needs.

The score path loads each frozen FLY124 scored window with `RE.load_window`,
so no supports `.npz` and no in-job supports build is needed (same reasoning as
the DREGON arm's note).

## What the harvest step must do

1. `omnirun status nv2-r2-score-michaels2-78d01c`; on `succeeded`:

   ```
   set -a; . ./.env; set +a
   aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
     s3://omnirun-artifacts/nv2-r2-score-michaels2-78d01c/outputs/results/ results/
   ```

   The sync overwrites `round2/fits/findings.md` and `round2/render/score/findings.md`
   with the runner's regenerated versions — check `git diff` and restore any
   hand-written file the job did not mean to replace (`git checkout --`).

2. Verify `round2/render/arm_michaels_v2.json` carries `gates.hppnet`,
   `gates.proxy.ltas_abs_db`, `bars` and
   `protocol.scorer.sha256 == 6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`,
   then commit it with `git add -f`.

3. Re-compose the round record with BOTH arms (R2FloorFix confirmed at 09:2xZ
   that `arm_dregon_v2b.json` cannot exist before ~10:30Z, so the existing
   `arm_dregon_v2.json` is the DREGON arm; if v2b lands later it becomes the
   PRIMARY dregon arm and the record is re-composed):

   ```
   PYTHONPATH=src:scripts python scripts/noise_v2_round_score.py --round 2 \
     --compose results/noise_v2/rounds/round2/render/arm_michaels_v2.json \
               results/noise_v2/rounds/round2/render/arm_dregon_v2.json \
     --supports-index results/noise_v2/rounds/round2/supports/index.json \
     --arm-job michaels_v2_r2_all=nv2-r2-score-michaels2-78d01c \
     --arm-job dregon_v2_r2_floor_benchcomb=nv2-r2-score-dregon-3f8397 \
     --previous results/noise_v2/rounds/round1.json \
     --compare results/noise_v2/rounds/round1.json
   ```

Bars: Michael's PARITY equal-regime ≤ 3.177994 rev/s (legacy per-regime
0.317 standby / 8.007 ramp / 0.756 cruise), DREGON PARITY ≤ 2.187786, STRETCH
≤ 1.897063. Proxy `ltas_abs_db` ≤ 1.219668 dB Michael's, 1.978609 dB DREGON.

## Status at handover: STILL UNPLACED after 3 h 34 min

`R2MichaelsHarvest` polled `omnirun status nv2-r2-score-michaels2-78d01c`
every ~20 s from 11:04Z to **14:39Z** — it never left `queued`. This is slot
contention, not a job problem: `uni-gpushort` is held by another project
(`kla-loglinear`: `ff-kl-256-k16-s2-l32e3-b46ca7` and
`ff-kl-256-k64-s2-l32e3-2432c8`, both 5.1 h in) with **162 jobs queued**
cluster-wide at 14:40Z. Each `omnirun status`/`ps` call retries placement, so
polling IS the placement mechanism; attempt 1 got a slot after 63 min, this
one has not.

The work itself is 6 minutes: R1's equivalent arm
(`nv2-r1-score-michaels-re-eafb5e`) ran 22:23:23Z → 22:29:30Z, exit 0, on the
same backend with the same rig. `--time 45m` is therefore ample once placed —
do NOT resubmit with a longer wall, and do NOT move it to `uni` (Main's
standing instruction at 12:40Z: hold on `uni-gpushort`, user constraint).

**So: nothing is lost by waiting.** The job is submitted at a SHA
(`e6c06769`) that carries both the harvested fit and the arm-selection fix, it
writes its arm to `round2/render/arm_michaels_v2.json`, and the harvest +
compose recipe above is complete. Whoever picks this up: poll the job id, then
run steps 1-3. If the job is eventually `failed` or `timeout`, read
`omnirun logs nv2-r2-score-michaels2-78d01c` — the failure mode to expect is
NOT the support-name gate (fixed and verified) but an upload/wall issue.

If `arm_dregon_v2b.json` exists by compose time it is the PRIMARY DREGON arm
and `arm_dregon_v2.json` follows it in the `--compose` list (R2FloorFix owns
it; job `nv2-r2-score-dregon-v2b-41d97e`, note
`round2/render/submit_note_v2b.md`, also unplaced as of 14:40Z).

What IS already committed and needs nothing further: the fit
(`michaels_fly125_all__flight.json`, `c089d978`), its findings
(`round2/fits/findings_flight.md`, `855d923b`), the render-regime k = 2
sanity (`round2/render_regime_flight/`, same commit), the scorer fix
(`e6c06769`).
