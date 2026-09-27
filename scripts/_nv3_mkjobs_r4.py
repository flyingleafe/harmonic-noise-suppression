"""Throwaway: the round-4 noise-v3 fit jobs (rig-only, round-4 priors).

The round-3 job (`_nv3_mkjobs_r3.py`: support build, per-seed fit CLI, R2
sync, seed summaries) with round 4's differences:

* `--priors v4` (`model.PRIORS_V4`: shaft rate at the label's band edge,
  log-normal `sigma_nu`, parity width floor, floor-relative profile centre,
  aeroacoustic speed exponents) and per-pool `--priors-set` overrides —
  DREGON's own measured label residual (`results/noise_v2/shaft/findings.md`,
  `motors_measured` 3.48 rad/s) narrows its `sigma_nu` prior;
* `--wander results/noise_v3/wander/<rig>_off.json` (every sigma 0) with
  `--rounds 0` and no `--ridge-step`: the rig MAP at zero latents, then the
  all-frames polish — no block wander is fitted or rendered ("r4a");
* Michael's channels normalised by the FULL-band gain (`--channel-gains-band
  full`, r4a/r4b) or by each channel's per-band TRANSFER (`transfer`, r4c: the
  fit records it as `params.array_response` and the renderer applies it
  back); DREGON keeps the >= 500 Hz gain and its per-mic wind term.

Writes `<out>/jobs/r4_job_<pool>[_<seeds>].sh`; submit with
`omnirun submit --backend uni-gpushort --gpus 1 --time 60m --yes -- bash -lc
"$(cat <job>.sh)"` from a detached worktree on the pushed HEAD (2 seeds per
job keep a pool inside the hour; `--seeds '0 1' --parallel`).

Usage: `_nv3_mkjobs_r4.py --out results/noise_v3/fits_r4a [--seeds '0 1']
[--pools dregon,cruise,standby] [--parallel] [--wander-suffix _off]
[--sched '--lbfgs-rtol 0 --rounds 0 --lbfgs-frames 64 --lbfgs-iters 150']`.
"""

import argparse
import math
from pathlib import Path

from _nv3_mkjobs_r3 import POOLS, head

#: Per-pool round-4 flags: the prior overrides and the channel-gain band.
R4_EXTRA = {
    "dregon": (
        # LN(ln 3.48, 0.5): centred on DREGON room 1's measured label residual
        # (motors_measured); the pool's carrier is motors_command, whose gap
        # to the shaft the fit must also carry, so wider than a pure
        # measurement (0.3) and narrower than the population (0.66)
        f"--priors v4 --priors-set sigma_nu_lognormal=[{math.log(3.48):.4f},0.5] "
        "--channel-gains-band above_500"
    ),
    "cruise": "--priors v4 --channel-gains-band {band}",
    "standby": "--priors v4 --channel-gains-band {band}",
}

SCHED_DEFAULT = "--lbfgs-rtol 0 --rounds 0 --lbfgs-frames 64 --lbfgs-iters 150"


def job(pool: str, args: argparse.Namespace) -> str:
    set_, name, rig, extra, init = POOLS[pool]
    wander = f"results/noise_v3/wander/{rig}{args.wander_suffix}.json"
    r4_extra = R4_EXTRA[pool].format(band=args.michaels_band)
    fit = f"""  $PY -u scripts/noise_v2_fit.py flight --mode flight_v3 --set {set_} --name {name} \\
    --wander {wander} \\
    --channel-gains results/noise_v2/mic_gains/mic_gains.json --channel-gains-rig {rig} {extra} \\
    {r4_extra} \\
    --init-from {init} {args.sched} --max-frames 0 --seed $N --restart-tag s$N --jobs 1 --threads 4 --device cuda \\
    --progress 50 --out "$F" --grid-dir "$F/grid/{name}_s$N" 2>&1 | grep -v Warn | stamp"""
    seed = f"""run_seed() {{
  N=$1; T1=$(date +%s)
  echo "V3R4 seed=$N start"
{fit} > "$F/logs/{name}_s$N.log"; EF=${{PIPESTATUS[0]}}
  echo "V3R4 seed=$N fit exit=$EF fit_wall_s=$(( $(date +%s) - T1 )) total_wall_s=$(( $(date +%s) - T0 ))"
  tail -3 "$F/logs/{name}_s$N.log"
  [ $EF -ne 0 ] && dump_err "$F/grid/{name}_s$N"
  $PY /tmp/summ.py "$F/restarts/{name}__flight_v3__s$N.json" || true
  sync_out
}}
"""
    loop = (
        f'P=""; for N in {args.seeds}; do run_seed $N & P="$P $!"; sleep 20; done; wait $P\n'
        if args.parallel
        else f"for N in {args.seeds}; do run_seed $N; done\n"
    )
    return (
        head(args.out, name)
        + f"""S=/tmp/supports_{name}; mkdir -p "$S"
echo "V3R4 spec: set={set_} name={name} wander={wander} extra='{extra} {r4_extra}' init={init} sched='{args.sched}' seeds='{args.seeds}' parallel={int(args.parallel)}"
$PY scripts/noise_v2_supports.py build --set {set_} --out "$S" > "$F/{name}_build.log" 2>&1; EB=$?
tail -2 "$F/{name}_build.log"
for f in "$S"/*.npz; do b=$(basename "$f"); [ -f "$C/$b" ] || {{ cp "$f" "$C/.tmp.$$.$b" && mv -f "$C/.tmp.$$.$b" "$C/$b"; }}; done
echo "V3R4 build exit=$EB wall_s=$(( $(date +%s) - T0 ))"
"""
        + seed
        + loop
        + """kill $SYNC_PID 2>/dev/null; sync_out
echo "V3R4_DONE total_wall_s=$(( $(date +%s) - T0 ))"
"""
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--sched", default=SCHED_DEFAULT, help="schedule flags passed to the fit CLI")
    ap.add_argument("--out", required=True, help="fit out dir, e.g. results/noise_v3/fits_r4a")
    ap.add_argument("--wander-suffix", default="_off", help="wander file <rig><suffix>.json")
    ap.add_argument("--seeds", default="0 1")
    ap.add_argument("--pools", default="dregon,cruise,standby")
    ap.add_argument("--parallel", action="store_true", help="run the seeds concurrently")
    ap.add_argument(
        "--michaels-band",
        default="transfer",
        choices=("full", "transfer"),
        help="Michael's channel normalisation: the full-band flat gain (r4a/r4b) or the "
        "per-band transfer recorded as params.array_response (r4c)",
    )
    args = ap.parse_args()
    d = Path(args.out) / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    tag = "_s" + args.seeds.replace(" ", "")
    for pool in args.pools.split(","):
        (d / f"r4_job_{pool}{tag}.sh").write_text(job(pool, args))
    print(sorted(str(p) for p in d.glob("r4_job_*.sh")))


if __name__ == "__main__":
    main()
