# Round-4 fit set (`fits_r4`)

Assembled from two rounds, each 4 restarts (seeds 0–3, two per `uni-gpushort`
job), reduced by `scripts/noise_v2_fit.py reduce --mode flight_v3` (the best
`lbfgs_loss_after`). Restart files were harvested from each job's OWN R2 prefix
and checked by md5 against it.

| pool | source dir | jobs | selected | HEAD | normalisation |
|---|---|---|---|---|---|
| `dregon_room2_floor` | `fits_r4b` | `nv3r4b-dregon-s01-906ef7`, `nv3r4b-dregon-s23-8a8e04` | s0 | `9fc5ca62` | ≥ 500 Hz flat gain + wind |
| `michaels_fly125_cruise` | `fits_r4c` | `nv3r4c-cruise-s01-004366`, `nv3r4c-cruise-s23-7c0a86` | s3 | `ab30eafa` | per-band transfer (`params.array_response`) |
| `michaels_fly125_standby` | `fits_r4c` | `nv3r4c-standby-s01-2c6a8c`, `nv3r4c-standby-s23-d8c1de` | s0 | `ab30eafa` | per-band transfer (`params.array_response`) |

Every restart: `--priors v4` (`model.PRIORS_V4`; DREGON adds
`--priors-set sigma_nu_lognormal=[1.247,0.5]`), `--wander
results/noise_v3/wander/<rig>_off.json --rounds 0` (rig only: no latent is
fitted, the record's wander block is all zero, so a render draws no track),
`--lbfgs-rtol 0 --lbfgs-frames 64 --lbfgs-iters 600`, `--init-from` the v2
fit of the pool, `--max-frames 0`. Job scripts: `results/noise_v3/fits_r4{b,c}/jobs/`.

Convergence (`optimiser.polish` of the selected restarts): the move from
the 64-frame subset optimum to the all-frames polish is 0.00721 (DREGON) /
0.01992 (cruise) / 0.00218 (standby) nats per cell; `rig_converged` is false
on all three at the 600-iteration cap (the terminal L-BFGS restart still
gains; see `optimiser.polish` in each file).

Checks: `results/noise_v3/checks_r4/` (job `nv3r4-checks-1d0d6c`). The
`latents` step is N/A on this set (no latent track by construction): it
reports `status: disabled` and exits successfully.
