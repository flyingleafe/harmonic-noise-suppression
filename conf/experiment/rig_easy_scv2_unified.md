---
experiment: rig_easy_scv2_unified
training_config: conf/experiment/rig_easy_scv2_unified.yaml
batch: docs/experiments/stochastic-fit.md
---

> **The bank this run trains on has TWO KNOWN DEFECTS and a pending user
> decision (2026-09-15).** Do not re-run this experiment, and do not rebuild
> `data/rig_banks/`, before reading
> `docs/experiments/rig-sampler-transfer-pair.md` § "Open decision: rebuild
> the banks": the fitted comb's last order is unconstrained and loud, and the
> two rigs' anchors disagree on label provenance (Michael raw, DREGON
> refined).

## Motivation

The easy arm of the rig-sampler transfer pair. Same model and same regime as
`rig_fitted_scv2_unified`, with the two rigs' cruise fits replaced by a bank of
2048 draws from their measured NEIGHBOURHOODS — the same rigs, seen as a
distribution instead of as two points.

The question it answers is narrow on purpose. `rig_fitted_scv2_unified` trained
on two point fits and transferred unevenly (Michael's arm 5.4 rev/s at the
plateau against real-trained 2.65; DREGON 18.9 against 3.78), and its best real
score was at epoch 22, after which every real view degraded while the synthetic
half kept improving — synthetic overfitting to two fixed parameter vectors. If
that is what limits transfer, then drawing a fresh rig from each fit's own
neighbourhood should help, because nothing in the stream is a fixed vector any
more. If instead the limit is the fit's FIDELITY, a neighbourhood of a wrong
point is still wrong and the arm will not move.

Paired with `rig_hard_scv2_unified`, which smears the same cloud width along
the path between the two rigs. The two arms differ only in where the cloud
sits, so the pair separates "more rigs" from "rigs between the rigs".

## Setup

Identical to `rig_fitted_scv2_unified` except `experiment_name` and the stream
path: same `real_r1_scv2` architecture, `override /validation: rps_unified`,
bfloat16, batch 128, 12 workers, 2 s clips, `samples_per_validation: null`.

Stream: `conf/online_mix/rig_easy_5050.yaml` — the fitted policy with each
stochastic source's `ranges:` replaced by
`preset_bank: data/rig_banks/rig_easy_n2048.json`. Weights, the full-flight
trajectory, `rps_scale_range`, `render_reuse: 48`, the level draw, the silence
source, the speech source and the mixing policy are the base policy's.

Bank (`scripts/_build_rig_bank.py --mode neighbourhood`, strength 2.0, seed
20260914, 1024 draws per anchor):

- anchors `results/S2/cruise_8clip.json:fly125_cruise_00` (Michael's) and
  `results/S2/dregon_room2_cruise_refined.json:free-flight_nosource_room2_cruise_00`,
  both on the physical level scale (`power_scale` folded);
- strength 2.0 is the measured coverage setting: at 2.0 the cloud brackets the
  real clip in 100% of 1/3-octave bands above 300 Hz on BOTH rigs (95.2% /
  90.5% of all bands including the low ones), and 2.5 and 3.0 add nothing above
  300 Hz while costing acceptance;
- the effective distribution is the TRUNCATED one: the sampler's guards reject
  23.8% of Michael's draws and 5.9% of DREGON's at this strength (this build:
  1.29 attempts per accepted draw, 77.9% accepted first try; the guards that
  fired are `ltas` 383, `speed_law` 190, `trend_falls` 39, `parity_sign` 24);
- each entry carries the sampled rig's IDENTITY (timbre, floor shape, line
  widths and the shaft jitter derived from them, the per-(mic, rotor) pattern,
  the floor's static share, the fitted speed law) over the BASE POLICY's own
  per-clip dynamics draw (the measured amplitude OU process, floor drift,
  per-mic modulation, jitter time constant and spread, phase diffusion, label
  error), so the arm keeps the base policy's dynamics regime;
- **the speed exponents are BOUNDED to the fitted range**: `amp_exp` to
  [4.398, 14.111] and `floor_exp` to [0, 6.792], the span the six measured fits
  occupy (`results/rig_sampler/structure.json:between_rig.per_fit`;
  `floor_exp`'s measured minimum is -3.711 and is raised to 0 because a
  negative floor exponent diverges at zero rotor speed). The sampler's measured
  between-refit width (sigma 2.733) is a legitimate width but its tail is not
  supported by any fit — unbounded, the bank reached 24.7 dB/dB, at which an
  idle window's comb sits ~20 dB under the same rig at cruise. The bound
  clipped 706 `amp_exp` and 510 `floor_exp` draws of 2050, of which 2 were then
  refused by the guards and redrawn, so the realised bank is `amp_exp`
  8.85 +- 3.57 and `floor_exp` 2.79 +- 2.94, both filling their bounds;
- realised spread over the bank: rotor gain -23.2 +- 8.4 dB, trend slope
  -17.5 +- 6.7 dB/decade, gamma0 6.6 +- 6.6 Hz, gamma_slope 0.39 +- 0.19
  Hz/order, label error 1.32 +- 0.93 rev/s;
- the build is bit-reproducible: an independent rebuild with the same
  arguments gives file SHA-256
  `58b24d2da024a39910f4014c905ab4d11bee05dc06a8264812c30a9bd3dd307d`, and the
  provenance carries an `inputs_digest` over the anchors, the donor ranges,
  every width and the source of the builder, the sampler and the renderer, so a
  rebuild with matching inputs is skipped and a code change is not.

One caveat that belongs to the arm rather than to the plumbing: **the
per-microphone pattern is still index-locked.** The base policy pins
`fixed_mic_gain_db` / `fixed_mic_floor_db` at fixed channel indices, which is
the mechanism behind the fitted run's r1/r2 ratio of 1.93. The bank perturbs
those vectors per entry (sigma 1.12 dB gain, 0.20 dB floor), so the arm sees a
population rather than one pinned vector, but the perturbation is small against
DREGON's -13 to +11 dB span and the indices are not permuted.

## Comparison rows

| run | stream | what it tests |
|---|---|---|
| `rig_fitted_scv2_unified` | two fitted rigs as POINT presets | the fit itself; best real_r3 9.97 at ep 22, plateau 13.42, r1/r2 = 1.93 |
| `ctrl_diverse_scv2_unified` | wide measured ranges | diversity without the fits |
| `rig_easy_scv2_unified` (this) | 2048 draws from the two fits' neighbourhoods | does turning each fit into a distribution transfer better |
| `rig_hard_scv2_unified` | 2048 draws along the path between the fits | does covering the gap between rigs transfer better |
| `real_r4_scv2_unified` | REAL noise in training | the reference: val/real_r3 2.99 |

Select on `real_overall`, not `overall_macro`: on the fitted-preset run the
synthetic half of the macro improved monotonically while every real view
degraded, so the macro neither stopped the run nor reduced the LR.

## Train

The bank is a gitignored BUILD PRODUCT (24.6 MB) and `omnirun` ships a clean
pushed checkout, so the bank does not travel — the job rebuilds it first. The
build is bit-reproducible from the pinned anchors and the seed (56 s on one
core), and it skips itself if the file is already current, so a restarted job
pays for it once:

```bash
omnirun submit --backend uni --gpus 1 --time 8h --yes -- bash -lc '
  python scripts/_build_rig_bank.py --mode neighbourhood \
      --name rig_easy_n2048 --n 2048 --strength 2.0 --seed 20260914 \
      --anchor results/S2/cruise_8clip.json:fly125_cruise_00 \
      --dynamics conf/online_mix/rig_fitted_5050.yaml:1 \
      --anchor results/S2/dregon_room2_cruise_refined.json:free-flight_nosource_room2_cruise_00 \
      --dynamics conf/online_mix/rig_fitted_5050.yaml:0 \
  && python train.py experiment=rig_easy_scv2_unified'
```

The anchors live in `results/S2/`, which is also gitignored — `omnirun pull` /
`scripts` R2 transport must put those two summaries on the backend before the
build step, exactly as the fitting jobs do. A missing bank fails at pool
construction with the rebuild command in the error message, not mid-run.

## Conclusion
