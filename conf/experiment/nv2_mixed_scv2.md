---
experiment: nv2_mixed_scv2
training_config: conf/experiment/nv2_mixed_scv2.yaml
batch: docs/experiments/noise-v2-transfer.md
---

## Motivation

The direct regressor (`real_r1_scv2` architecture, `hb_scv2_mag_nogate` head)
on the noise-model-v2 **MIXED** stream — the mixed arm of the fitted-family
transfer batch for this trunk.

Does the fitted family ADD to real data? The real half already carries both
evaluation rigs, so this arm reads directly against the family's real
reference: anything below it is a contribution from rigs and flights that were
never built.

Reference rows: `real_r4_scv2_unified`, real noise in training, `val/real_r3`
**2.99**; the legacy stochastic pair `rig_easy_scv2_unified` **5.72** and
`rig_hard_scv2_unified` **5.37**; and the same-stream arm of the other trunk
(`nv2_mixed_hppnet_l2`), which is what separates "this noise family transfers"
from "this trunk transfers".

## Setup

**Exact diff against its parent `rig_easy_scv2_unified`: two fields,
`experiment_name` and `data.train.params.path`. Nothing else.** The parent's
defaults chain, `override /validation: rps_unified`, bfloat16, batch 128, 12
workers, 2 s clips, `samples_per_validation: null` all stand.

Stream `conf/online_mix/noise_v2_mixed_dload.yaml`, bank
`dload:noise-v2-banks@7f6a3242b353…/noise_v2_hard_n2048.json` (the pinned
hard bank, pulled by the job). Structure is
`hb_stochmixed_dload.yaml`'s with its stochastic + static-comb pair replaced by
ONE `kind: noise_v2` hard source: real 2.0 (the DREGON `in_flight_noise` pool
minus `free-flight_nosource_room1` plus FLY125, whole envelope) : silence 0.4 :
v2-hard 2.0 — the same 45.5 / 9.1 / 45.5 split, with the fitted family standing
exactly where the stochastic pair stood, and the same one-stage augmentation
block (gain/polarity, `freq_scale`, noise time warp). The synthetic half is the
HARD bank rather than the easy one because the real half already contains both
evaluation rigs as recordings: what the synthetic half can add is rigs and
flights that are in neither. `rps_scale_range` [1.0, 1.0] and `rps_max: 150` as
on the hard arm, with the same measured truncation (acceptance 0.296; hover
level truncated, rig diversity not).

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
checkpoint — `python scripts/_regime_decomp.py --exp nv2_mixed_scv2 --ckpt
best` (ramp / zero / standby / cruise). The RAMP cell is the one this batch
turns on: both legacy synthetic arms were close to real at cruise and collapsed
the four rotors onto nearly one speed in the transitions (output spread 0.28
easy / 0.19 hard against 4.71 for the real-trained model).

## Train

```bash
scripts/noise_v2_submit_arms.sh nv2_mixed_scv2
```

The bank does not travel with the checkout: `omnirun` ships a clean pushed
tree and the job pulls the pinned `noise-v2-banks` dataset before
`train.py`, so the fetch happens once and outside the DataLoader workers.

## Conclusion

**PENDING** — not submitted as of 2026-09-22.
