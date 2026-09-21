---
experiment: nv2_hard_hppnet_l2
training_config: conf/experiment/nv2_hard_hppnet_l2.yaml
batch: docs/experiments/noise-v2-transfer.md
---

## Motivation

The salience trunk (HPPNet port at L2: per-rotor Gaussian layers on the linear
0-150 rev/s / 300-bin grid, CRF readout, scored through the model's own
`decode_logits`) on the noise-model-v2 **HARD** stream — the hard arm of the
fitted-family transfer batch for this trunk.

Does a distribution in which NEITHER evaluation rig is privileged — neither in
the comb (path rigs) nor in the flight (hyperprior trajectories) — still
transfer? The legacy pair's answer was yes and slightly better than easy (5.37
against 5.72); if that holds on the fitted family, a new rig needs no fit of
its own.

Reference rows: `hppnet_l2_r2_s0`, real noise in training, **2.27**; the legacy
stochastic pair `rig_easy_scv2_unified` **5.72** and `rig_hard_scv2_unified`
**5.37**; and the same-stream arm of the other trunk (`nv2_hard_scv2`), which
is what separates "this noise family transfers" from "this trunk transfers".

## Setup

**Exact diff against its parent `rig_easy_hppnet_l2_unified`: two fields,
`experiment_name` and `data.train.params.path`. Nothing else.** The parent's
defaults chain, `override /validation: rps_unified`, batch 128, 6 workers, 2 s
clips, `samples_per_validation: null`, validation batch 64, lr 1e-3, patience
20 all stand.

Stream `conf/online_mix/noise_v2_hard_5050.yaml`, bank
`data/rig_banks/noise_v2_hard_n2048.json` (2048 entries, `python
scripts/noise_v2_build_bank.py --preset hard`, seed 20260921, gitignored build
product rebuilt in-job). The path interpolates CRUISE <-> CRUISE only on the
common order range K = 1..81 (DREGON's orders 82-88 are dropped and the drop is
in the provenance); the regime policy is carried, not blended — at mixing
coordinate *t* the standby slot is Michael's standby fit with probability *t*
and null otherwise. Every entry has `traj_rig: null`, so every flight comes
from `rigs: {posterior: 1}`, the rig HYPERPRIOR: a drone that has never been
built, on a comb that is neither rig. `rps_scale_range` is [1.0, 1.0] here and
not the easy arm's [0.45, 1.2] — a posterior draw already carries its own hover
level and ESC clamp, and a multiplier on top would smear the rig-balanced speed
distribution this arm exists to fly.

**The hyperprior is TRUNCATED and the arm owns that.** `rps_max: 150` (the
salience trunks' grid top) rejects and redraws any flight whose peak rotor
speed exceeds it; acceptance is **0.296** (512 accepted / 1217 rejected),
because uncapped flight peaks are median 190, p90 396, max 615 rev/s and 66 %
exceed 150. The cost falls on the hover level and NOT on rig diversity: of the
32 scale-free coordinates only coordinate 0 (log hover scale) moves (z = -33.2,
std 0.497 -> 0.275) while all 31 shape/dynamics coordinates are statistically
unmoved (max |z| = 1.71, accepted/all std ratio 0.951-1.031). Surviving hover:
median 86.1, p90 111.9, max 182.0 rev/s.

**The level draw differs from the legacy pair deliberately.**
`rig_easy_5050.yaml` named no level key and ran the stochastic pool's defaults
(`normalize_rms: 0.1`, `level_mode: window`), which normalises every window to
the same root-mean-square and so removes level as a speed cue. This stream uses
`normalize_rms_range: [0.02, 0.25]` with `level_mode: flight`
(`hb_stochmixed_dload.yaml`'s values): the drawn level is the level at the
REFERENCE speed and the window's own speed envelope is kept. `render_noise` is
absolute and its speed-level law is a fitted quantity — normalising it away
would discard the thing the v2 fit measures and the stochastic family guessed.
Every bank entry also carries the pinned short-span contract (`amp_exp` 2.0,
`floor_exp` 2.0, `floor_static_rel` 2.5e-3): the R5 fitted exponents (-0.87 /
15.3) are unidentified on a cruise-only pool and are not flown.

**`render_reuse: 192`.** A v2 render costs 1973 ms per 2 s x 8 mics against the
legacy family's 416 ms (`results/noise_v2/stream_bench/`), and at
`render_reuse: 48` the loader is RENDER-BOUND: 12 workers deliver a batch of
128 about every 1.2 s. 192 = 4 x 48 is the value at which this arm produces
fresh renders at the SAME WALL-CLOCK RATE the legacy pair had at 48. It is a
throughput decision with a distributional cost, stated here: each rendered
window is reused four times as often as in the legacy arms.

## What will be measured

Selection and reporting on **`real_overall`** (not `overall_macro`: on the
point-preset run the synthetic half improved monotonically while every real
view degraded, so the macro neither stopped the run nor cut the LR). Reported:
best `real_overall` with the epoch it occurred at, the per-view `real_r1` /
`real_r2` / `real_r3` rev/s MAE at that epoch, the `r1`/`r2` ratio against the
point-preset run's 1.93, and the four-regime decomposition of the best
checkpoint — `python scripts/_regime_decomp.py --exp nv2_hard_hppnet_l2 --ckpt
best` (ramp / zero / standby / cruise). The RAMP cell is the one this batch
turns on: both legacy synthetic arms were close to real at cruise and collapsed
the four rotors onto nearly one speed in the transitions (output spread 0.28
easy / 0.19 hard against 4.71 for the real-trained model).

## Train

```bash
scripts/noise_v2_submit_arms.sh nv2_hard_hppnet_l2
```

The bank is a gitignored build product and `omnirun` ships a clean pushed
checkout, so the job rebuilds it before `train.py`; the build is
bit-reproducible and skips itself when the provenance digest already matches.

## Conclusion

**PENDING** — not submitted as of 2026-09-22.
