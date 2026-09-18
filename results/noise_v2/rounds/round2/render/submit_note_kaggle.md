# R2 score arms — daemon outage, kaggle re-pin, colab finish (RESOLVED)

Written by `R2Close`. Both arms are now scored and committed; the resolution is
at the bottom. The body below is the handover written while they were still in
flight, kept because the failure modes it records are the reusable part.

## Timeline (all UTC, 2026-09-18)

| time | event |
| --- | --- |
| 10:59 | `nv2-r2-score-michaels2-78d01c` submitted (`uni-gpushort`, SHA `e6c06769`) |
| 11:13 | `nv2-r2-score-dregon-v2b-41d97e` submitted (`uni-gpushort`, SHA `dec46b62`) |
| 14:42 → 16:33 | `R2Close` polls both every 5 min: `status: queued` on every single call, never `starting` |
| 16:10 | `omnirun explain` shows all three `uni-gpushort` slots "provider at capacity", estimated wait down to 720 s (from 2640 s at 13:17) — contention easing, but no placement |
| ~16:39 | **the omnirun daemon becomes unreachable**: every CLI call returns `error: cannot reach the omnirun daemon at http://10.100.0.1:8787 (timed out)` |
| 18:08 | raw `curl http://10.100.0.1:8787/` also times out (exit 124) — the tunnel/host is down, not the CLI |
| 18:08 | R2 bucket checked directly: `s3://omnirun-artifacts/nv2-r2-score-michaels2-78d01c/` and `.../nv2-r2-score-dregon-v2b-41d97e/` are both **empty** — neither job ever ran or uploaded anything |
| 18:12 | Main relays a user override: re-pin BOTH arms to the `kaggle` backend, retry the submit every 2 min until the daemon answers, cancel the two `uni-gpushort` jobs once the kaggle ones are accepted |
| 18:14 → 19:02+ | kaggle submit retried every ~3 min from `.worktrees/submit-R2Close` @ `643b3058`; every attempt dies at `· resolving repo state…` with the same daemon-unreachable error |

Nothing has been lost. The two `uni-gpushort` jobs are still registered
daemon-side (they cannot be cancelled while the daemon is unreachable either),
and no artifact of either job exists anywhere yet.

## The kaggle commands being retried

Both are the notes' invocations (`submit_note_michaels.md`,
`submit_note_v2b.md`) with only the backend, the wall and the job name changed;
the wrapper, the scorer invocation, the fit paths, the candidate names, the arm
paths and the `--outputs` patterns are untouched, so the arms stay comparable
with `arm_dregon_v2.json` and with R1.

```
cd .worktrees/submit-R2Close && COLUMNS=200 omnirun submit \
  --backend kaggle --gpus 1 --time 30m --name nv2-r2-score-michaels-kg \
  --outputs 'results/noise_v2/rounds/round2/render/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 2 --fits results/noise_v2/rounds/round2/fits --rigs michaels --fit michaels=results/noise_v2/rounds/round2/fits/michaels_fly125_all__flight.json --candidate michaels_v2_r2_all --arm-out results/noise_v2/rounds/round2/render/arm_michaels_v2.json'
```

```
cd .worktrees/submit-R2Close && COLUMNS=200 omnirun submit \
  --backend kaggle --gpus 1 --time 30m --name nv2-r2-score-dregon-v2b-kg \
  --outputs 'results/noise_v2/rounds/round2/render/*.json' \
  --outputs 'results/noise_v2/rounds/round2/render/*.md' \
  --outputs 'results/noise_v2/rounds/round2/render/audio/**' \
  -- bash -lc 'set -a; [ -f ./.env ] && . ./.env; [ -f "${OMNIRUN_OUTPUT%/outputs}/.env" ] && . "${OMNIRUN_OUTPUT%/outputs}/.env"; set +a; export PYTHONPATH=src:scripts; python scripts/noise_v2_round_score.py --round 2 --fits results/noise_v2/rounds/round2/fits --rigs dregon --fit dregon=results/noise_v2/rounds/round2/fits/dregon_room2_floor__flight_floor_only_v2.json --candidate dregon_v2_r2_floor_combgain --arm-out results/noise_v2/rounds/round2/render/arm_dregon_v2b.json --dump-audio results/noise_v2/rounds/round2/render/audio/dregon_v2b'
```

`643b3058` is `origin/main` and is a descendant of BOTH `e6c06769` (the
`MICHAELS_FIT_SUPPORTS` arm-selection fix) and `dec46b62` (the
`comb_gain_db` refit JSON), so one SHA carries everything both arms need.

### Kaggle-specific checks the harvest step must make

The user's instruction: in the FIRST log lines of each kaggle run, verify that
(a) the HPPNet checkpoint is fetched and its sha256 check passes and (b) the
`dload`/R2 credentials resolve on that backend. If kaggle fails on a missing
dependency or credential, the failure log is committed and reported rather than
retried blindly.

## Harvest + compose, unchanged

```
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/<job>/outputs/results/ results/
```

then `git add -f` the arm JSON and compose:

```
PYTHONPATH=src:scripts python scripts/noise_v2_round_score.py --round 2 \
  --compose results/noise_v2/rounds/round2/render/arm_michaels_v2.json \
            results/noise_v2/rounds/round2/render/arm_dregon_v2b.json \
            results/noise_v2/rounds/round2/render/arm_dregon_v2.json \
  --supports-index results/noise_v2/rounds/round2/supports/index.json \
  --arm-job michaels_v2_r2_all=<michaels job> \
  --arm-job dregon_v2_r2_floor_combgain=<dregon job> \
  --arm-job dregon_v2_r2_floor_benchcomb=nv2-r2-score-dregon-3f8397 \
  --previous results/noise_v2/rounds/round1.json \
  --compare results/noise_v2/rounds/round1.json
```

`arm_dregon_v2b.json` leads the DREGON list (candidate
`dregon_v2_r2_floor_combgain` is the primary; `arm_dregon_v2.json` is the
superseded one). If a rig never lands, add
`--not-run <rig>=<reason>` for it instead of its arm.

Bars: DREGON PARITY ≤ 2.187786 rev/s, STRETCH ≤ 1.897063, proxy ≤ 1.978609 dB.
Michael's PARITY equal-regime ≤ 3.177994 rev/s (legacy per-regime 0.317103
standby / 8.007098 ramp / 0.755783 cruise), proxy ≤ 1.219668 dB.
`protocol.scorer.sha256` must be
`6e50e025ba40df055412ae5d59c2f7a54a23acc0d61c788ab3871fb9fd2877b1`.

## Resolution (what actually scored the arms)

| arm | job | backend | wall | outcome |
| --- | --- | --- | --- | --- |
| michaels | `nv2-r2-score-michaels-kg-86343e` | kaggle | 19:17→19:32Z | **succeeded**, `arm_michaels_v2.json` (`dee0e77a`) |
| dregon v2b | `nv2-r2-score-dregon-v2b--bce8f6` | **colab** | 21:13→21:22Z | **succeeded**, `arm_dregon_v2b.json` (`9579ca94`) |

Three things had to be fixed on the way, in this order.

1. **No `.env` on a kaggle node.** The first kaggle pair died on
   `RuntimeError: r2:// checkpoint ... requested but R2 creds missing in .env`
   (note `nv2-r2-score-michaels-kg-9740a8_failure.md`). The wrapper's two
   sources, `./.env` and `${OMNIRUN_OUTPUT%/outputs}/.env`, do not exist there.
   Settled empirically rather than assumed: the resubmitted wrapper printed
   `find /kaggle -maxdepth 4 -name .env` and it returned **nothing**, with
   `OMNIRUN_OUTPUT=/kaggle/tmp/omnirun/jobs/<job>/outputs` and cwd
   `/kaggle/tmp/omnirun/projects/harmonic-noise-suppression/.trees/643b3058c81c`.
   There is no path to add to the wrapper; `omnirun submit --env` is the only
   mechanism. The three vars `R2_ACCOUNT_ID`, `AWS_ACCESS_KEY_ID`,
   `AWS_SECRET_ACCESS_KEY` are sufficient and necessary —
   `utils.checkpoints.load_r2_env` and
   `stochastic_fit_revised_eval._download_s3_uri` derive bucket and endpoint
   from the account id, and nothing else in the score path reads `.env`. They
   were expanded from the local `.env` at call time, so no value appears in any
   command line recorded here, in the job notes, or in the logs.

2. **Daemon reachability.** From ~16:39Z the wireguard route to the daemon was
   dead. Every `omnirun` call in the resolution used
   `omnirun --daemon localhost:18787` over an ssh tunnel; `config.toml` was not
   touched.

3. **kaggle would not start the DREGON arm.** Two attempts,
   `nv2-r2-score-dregon-v2b--f1b547` (19:32→20:48Z) and
   `nv2-r2-score-dregon-v2b--9934d1` (20:48→21:22Z), sat in `starting` with
   **zero** log lines, while the michaels arm on the identical backend, wrapper
   and SHA reached `running` 90 s after start. That is kaggle-side scheduling,
   not the job. `uni-gpushort` was not an option either — `kla-loglinear` still
   had ~60 jobs queued 11.5 h deep. The same arm was fired on `colab` as a twin
   (`omnirun backends check`: colab ok, 0 active sessions), it reached `running`
   in 2 min and finished in 9; the kaggle twin was cancelled. The DREGON audio
   dump (10 npz, ~100 MB) lives in that colab job's artifacts, not in git.

`round2/render/findings.md` in git is the per-arm findings the MICHAEL'S job
regenerated; the DREGON v2b job's own copy of that same generated file stays in
its artifact tree (`s3://omnirun-artifacts/nv2-r2-score-dregon-v2b--bce8f6/`).
The round-level document that covers both arms is
`round2/score/findings.md`, composed from the three arm records.

### Numbers (all three frozen gates FAIL; the round record is `round2.json`)

| gate | number | bar | verdict |
| --- | ---: | ---: | --- |
| HPPNet DREGON cruise (primary arm `dregon_v2_r2_floor_combgain`) | 72.340606 rev/s (95 % upper 74.830651) | 2.187786 PARITY / 1.897063 STRETCH | FAIL (margin -70.152820 / -70.443543) |
| HPPNet Michael's equal-regime | 2.456399 rev/s (ratio 0.811587) | 3.177994 | **PASS** (margin +0.721596) |
| proxy `ltas_abs_db` dregon_cruise | 3.5024 dB | 1.9786 | FAIL (-1.5238) |
| proxy `ltas_abs_db` michaels_cruise | 2.1491 dB | 1.2197 | FAIL (-0.9294) |
| likelihood comb (decisive) | -372,886.6588 vs oracle -371,916.5041 nats/s | below oracle | PASS (margin -970.1548) |

The comb-gain freedom is measured, and it is not the missing piece: it moves
DREGON cruise from the bench-comb arm's 75.093968 to 72.340606 rev/s, 3.7 %, on
a gate that needs roughly a 38x reduction. Michael's pooled fit passes its bar
only because the equal-regime mean absorbs a standby regime that got 9.48x
worse (3.006774 vs frozen 0.317103) while ramp improved 2.4x (3.393256 vs
8.007098) and cruise degraded 1.28x (0.969167 vs 0.755783).
