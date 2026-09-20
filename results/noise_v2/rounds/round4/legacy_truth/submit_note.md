# R4 legacy-truth B: fitting v2 to the LEGACY render — submitted, awaiting harvest

Written by `LegacyTruth`. Both fits are RUNNING on the cluster; nothing in this
note is a result. Job ids, the exact commands, and what the harvest must do.

## The data being fitted

`scripts/noise_v2_legacy_truth.py supports` renders the LEGACY stage-2 model
(identity-matched per recording, `results/S2/dregon_room2_cruise_refined.json`,
seed 2001, 8 mics, `normalize_rms=None`) on the frozen `motors_command` label
track of **exactly the five 8 s windows the R3 v2 DREGON flight fit pooled**
(`dregon_room2_floor__flight_floor_lowk.json`'s own `supports` list: free-flight
@1512727403.205, hovering @1511903911.394, updown @1511903584.348, rectangle
@1511905731.953, spinning @1511905206.978, all `+8_motors_command`), and builds
a v2 support from that audio through `supports.synthetic_support` — the same
periodogram route and the same `.npz` cache as a real support, spec
`synthetic:legacy_<real support name>`, provenance in `meta["synthetic"]`.
Measured band power of the render against the real clip's, per window: -18.94
vs -24.77, -22.01 vs -24.59, -21.83 vs -25.44, -20.65 vs -23.25, -19.49 vs
-24.01 dB (`supports.json`).

## The jobs

| field | value |
| --- | --- |
| job A (same mode as R3) | `nv2-r4-lt-lowk-410220` |
| job B (frees the most) | `nv2-r4-lt-free-a8eb07` |
| backend | `uni-cpu`, `--gpus 0 --cpus 32 --mem 64 --time 5h` |
| code SHA | `f94c0392` (`noise-v2 R4 legacy-truth: A anatomy`, pushed on `main`) |
| submitted | 2026-09-20, via `omnirun --daemon localhost:18787` (ssh tunnel to the wg-bound daemon, `hub` process `omnirun-wg`) |
| submitted from | detached worktree `.worktrees/submit-LegacyTruth` at `f94c0392` |
| pool | the five `synthetic:legacy_flight_dregon_*@*+8_motors_command` supports |
| seeds | 4 per pool (`--seeds 4`, reduced to the best polished objective plus a `restarts` block) |
| output | `results/noise_v2/rounds/round4/legacy_truth/fits/dregon_legacy_render__flight_floor_lowk.json` and `..__flight.json` |

Job A is the R3 DREGON mode exactly: `--floor-low-k --low-orders 8` with the
comb transplanted frozen from the four R3 `bench_dregon_Motor{1..4}_70` fits, so
only the floor, the mic gains, one `comb_gain_db` and eight per-order low-order
gains move. Job B is mode `flight`, which frees the WHOLE per-order profile
(k <= k_max, no transplant), the per-line `gamma_hz` and the dynamics — the
mode that can express an arbitrary comb shape, so if v2 cannot reproduce the
legacy render here it cannot reproduce it at all.

`--seeds` on a flight pool is new in this round (`scripts/noise_v2_fit.py`:
the bench restart loop now also builds flight units and `reduce_restarts` is
called with the pool's mode). The four restarts differ in the optimiser seed
(Adam frame minibatches) and, where the dynamics are free, in the init jitter;
in `flight_floor_lowk` the dynamics arrive frozen, so there the spread is the
minibatch path alone.

## The commands

```
cd .worktrees/submit-LegacyTruth && COLUMNS=200 omnirun --daemon localhost:18787 submit \
  --backend uni-cpu --gpus 0 --cpus 32 --mem 64 --time 5h \
  --name nv2-r4-lt-lowk \
  --outputs 'results/noise_v2/rounds/round4/legacy_truth/fits/**' \
  --outputs 'results/noise_v2/rounds/round4/legacy_truth/supports.json' \
  -- bash -lc "$(cat /tmp/r4lt_lowk.sh)"
```

In-job body (`/tmp/r4lt_lowk.sh`; `/tmp/r4lt_free.sh` is the same with the
`--floor-low-k --low-orders 8 --frozen-mean $FM` block removed and a separate
grid dir):

```
set -a; [ -f ./.env ] && . ./.env
JOBENV="${OMNIRUN_OUTPUT%/outputs}/.env"; [ -f "$JOBENV" ] && . "$JOBENV"; set +a
export PYTHONPATH=src
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
O=results/noise_v2/rounds/round4/legacy_truth; F=$O/fits
C=results/noise_v2/rounds/round1/supports          # SU.CACHE_DIR
R3=results/noise_v2/rounds/round3/fits
$PY scripts/noise_v2_legacy_truth.py supports --out "$O" --cache "$O/supports"
cp -f "$O"/supports/*.npz "$C"/                    # REQUIRED: load_support reads CACHE_DIR
SPECS=<the five synthetic: specs out of supports.json>
FM=<the four $R3/bench_dregon_Motor{1..4}_70__bench.json>
$PY scripts/noise_v2_fit.py flight --support $SPECS --name dregon_legacy_render \
  --floor-low-k --low-orders 8 --frozen-mean $FM --seeds 4 \
  --jobs 2 --threads 16 --progress 200 --out "$F" --grid-dir "$F/grid_lowk"
# then a read-back print of the convergence label, nats/cell, comb_gain_db,
# low_order_gain_db, amp_exp, the dynamics, the floor, the span pins and the
# restart losses
```

Each job prints `R4LT specs: ...` then `R4LT_DONE exit_supports=.. exit_fit=..`.

## Harvest

```
omnirun --daemon localhost:18787 logs nv2-r4-lt-lowk-410220 | tail -40
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/nv2-r4-lt-lowk-410220/outputs/results/ results/
git add -f results/noise_v2/rounds/round4/legacy_truth/fits/*.json
```

Then, on the laptop (HPPNet is a laptop job, three 4 s windows):

```
PYTHONPATH=src python scripts/noise_v2_legacy_truth.py score --probe --with-real \
  --stem score_legacy_fit --ladder v2_legacy_lowk,v2_legacy_free \
  --fit v2_real=results/noise_v2/rounds/round3/fits/dregon_room2_floor__flight_floor_lowk.json \
  --fit v2_legacy_lowk=<job A fit> --fit v2_legacy_free=<job B fit> \
  --out results/noise_v2/rounds/round4/legacy_truth
```

PASS for B is HPPNet mean over the three windows <= 2.19 rev/s (the legacy
render's own bar); the comparanda are real 1.074, legacy 1.963, v2-fitted-to-
real 69.555.
