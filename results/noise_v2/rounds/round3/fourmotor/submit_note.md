# Stage-3 four-motor rig fits — submitted, awaiting harvest

Written by `FourMotor`. Two arms are RUNNING on the cluster; `F0` is the already
committed R3 fit and is not re-run.

| field | value |
| --- | --- |
| job | `nv2-r3-fourmotor-3d587f` |
| backend | `uni-cpu`, `--gpus 0 --cpus 16 --mem 64 --time 2h` |
| code SHA | `d27f58be` (`noise-v2 R3 fourmotor: estimators on the real support`, pushed on `main`) |
| submitted | 2026-09-19, via `omnirun --daemon localhost:18787` |
| submitted from | detached worktree `.worktrees/submit-FourMotor` at `d27f58be` |
| support | `bench_dregon_motor:allMotors:70`, rebuilt in-job (see below) |
| starts | 4 per arm (`--seeds 4`, `--init-jitter` default 0.8 on starts 1-3) |
| parallelism | the two arms run concurrently, `--jobs 4 --threads 1` each (8 units) |
| outputs | `results/noise_v2/rounds/round3/fourmotor/**` |

## The three arms

| arm | what it is | fit JSON |
| --- | --- | --- |
| **F0** | the committed R3 rig fit, as-is, NOT re-run | `round3/fits/bench_dregon_allMotors_70__bench.json` |
| **F1** | `profile_db` initialised from `estimate_e2(multi=True)` on the real support, with the synthetic study's per-order prior widths (1 dB for k ≤ 16, 3 dB for 17 ≤ k ≤ 48, the model's own two-regime prior above) | `round3/fourmotor/fits/F1/bench_dregon_allMotors_70__bench.json` |
| **F2** | F1 plus the estimator's per-rotor carrier offsets added to the frozen bench carriers | `round3/fourmotor/fits/F2/bench_dregon_allMotors_70__bench.json` |

F2 is run because rotor 3's refined offset is **0.59 bins** (0.01725 rev/s
against the 0.02907 Hz bin), over the 0.5-bin threshold. The offsets fed to both
the estimator and F2 are `+0.00545 / +0.00090 / +0.01725 / +0.01090` rev/s, from
`refine_offsets` at the tightened `bound_rev_s = 0.05` (see `findings.md` §2 for
why the committed default 0.5 is wrong here).

## Why the support is rebuilt in-job

The committed `results/noise_v2/rounds/round1/supports/bench_dregon_allMotors_70.npz`
is the **R1** window: carriers 64.63956 / 67.66182 / 68.73794 / 69.57179 rev/s,
0.002-0.007 rev/s off the R2/R3 ones the committed R3 fit and the stage-2
estimators use. `supports.load_support` reads that path (`SU.CACHE_DIR`), so the
job rebuilds the support with `use_cache=False` and overwrites the cache entry
first. Checked in the main checkout before submit: a fresh build reproduces the
committed `round2/supports` copy with `power` **bit-identical** (max relative
difference 0.0) and the same segment `[3.85, 38.25] s`, 550400 samples and
carriers 64.636277 / 67.659628 / 68.736409 / 69.565239. So F1/F2 are comparable
to F0 cell for cell.

## The recipe (in-job body, `/tmp/fm_submit.sh` at submit time)

```
set -a; [ -f ./.env ] && . ./.env
JOBENV="${OMNIRUN_OUTPUT%/outputs}/.env"; [ -f "$JOBENV" ] && . "$JOBENV"; set +a
export PYTHONPATH=src
PY=python; [ -n "${VIRTUAL_ENV:-}" ] && PY="$VIRTUAL_ENV/bin/python"
F=results/noise_v2/rounds/round3/fourmotor/fits
C=results/noise_v2/rounds/round1/supports
INIT=results/noise_v2/rounds/round3/fourmotor/profile_init_e2multi.npz
mkdir -p "$F/F1" "$F/F2" "$C"
$PY - <<'PYEOF'                      # rebuild the support into SU.CACHE_DIR
from pathlib import Path
import numpy as np
from experiments.noise_model import supports as SU
s = SU.load_support("bench_dregon_motor:allMotors:70", use_cache=False)
p = SU.save_support(s, out_dir=Path("results/noise_v2/rounds/round1/supports"))
print("FM_SUPPORT", p, s.n_samples, s.segment, np.round(s.carrier_rev_s[:, 0], 6).tolist())
PYEOF
$PY scripts/noise_v2_fit.py bench --support bench_dregon_motor:allMotors:70 \
  --out "$F/F1" --jobs 4 --threads 1 --seeds 4 --progress 500 \
  --profile-init "$INIT" --profile-prior-sigma '16:1,48:3' > "$F/F1/log.txt" 2>&1 &
$PY scripts/noise_v2_fit.py bench --support bench_dregon_motor:allMotors:70 \
  --out "$F/F2" --jobs 4 --threads 1 --seeds 4 --progress 500 \
  --profile-init "$INIT" --profile-prior-sigma '16:1,48:3' --apply-carrier-offset > "$F/F2/log.txt" 2>&1 &
# wait on both, then `echo FM_DONE exit_f1=.. exit_f2=..` and tail both logs
```

Wall estimate: the committed R3 `allMotors_70` fit took 3576 s for its reported
start on one thread; 4 starts per arm on 4 cores each, both arms concurrent on
16 cpus, so ~1-1.3 h inside the 2 h wall.

## Harvest

```
omnirun --daemon localhost:18787 logs nv2-r3-fourmotor-3d587f | tail -60
set -a; . ./.env; set +a
aws s3 sync --endpoint-url "https://${R2_ACCOUNT_ID}.r2.cloudflarestorage.com" \
  s3://omnirun-artifacts/nv2-r3-fourmotor-3d587f/outputs/results/ results/
git add -f results/noise_v2/rounds/round3/fourmotor/fits/**/*.json
python scripts/noise_v2_fourmotor.py compare \
  --arm F0=results/noise_v2/rounds/round3/fits/bench_dregon_allMotors_70__bench.json \
  --arm F1=results/noise_v2/rounds/round3/fourmotor/fits/F1/bench_dregon_allMotors_70__bench.json \
  --arm F2=results/noise_v2/rounds/round3/fourmotor/fits/F2/bench_dregon_allMotors_70__bench.json
```

The stage-3 criterion `compare` applies, per rotor: median
`|profile_db(fit) − profile_db(Motor r)|` over in-band orders `k ≤ 48` at most
3 dB AND at least 70 % of those orders inside 3 dB. "Harmonic profiles largely
restored" means both, for all four rotors.
