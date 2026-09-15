---
experiment: rig_hard_scv2_unified
training_config: conf/experiment/rig_hard_scv2_unified.yaml
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

The hard arm of the rig-sampler transfer pair. Same model and same regime as
`rig_fitted_scv2_unified`, with the two rigs' cruise fits replaced by a bank of
2048 draws from a cloud along the PATH between them: the mixing coordinate is
uniform on [0, 1], so rigs resembling either real rig are present and neither
is especially likely.

Hypothesis: a model that must handle every rig between the two it will be
evaluated on cannot memorise either one, and has to learn the operation instead
of the fingerprint. The between-rig span is where the parameters actually
differ — the two fits' line widths differ by 14.7x, against a between-clip
spread of zero on the same quantity — so this arm asks whether covering that
span is what transfer needs, while the easy arm asks whether a tighter cloud
around each fit is enough.

The failure mode to avoid is the one the original stochastic-comb stream fell
into: diversity in the wrong direction, occupying regions no real rig does and
posing a harder, different task. Here the cloud is bounded by the same guards
the easy arm uses (falling power-law trend, positive parity envelope, bounded
gamma excursion, non-negative speed exponents, an LTAS plausibility envelope
calibrated on the measured anchor-to-real deviation), and the interpolation
convention per coordinate is the sampler's declared one, evaluated at the
INTERPOLATED point rather than at either anchor.

## Setup

Identical to `rig_fitted_scv2_unified` except `experiment_name` and the stream
path: same `real_r1_scv2` architecture, `override /validation: rps_unified`,
bfloat16, batch 128, 12 workers, 2 s clips, `samples_per_validation: null`.

Stream: `conf/online_mix/rig_hard_5050.yaml` — the fitted policy with each
stochastic source's `ranges:` replaced by
`preset_bank: data/rig_banks/rig_hard_n2048.json`; everything else is the base
policy's.

Bank (`scripts/_build_rig_bank.py --mode path`, spread 2.0, seed 20260914):

- same two anchors as the easy arm, on the physical level scale;
- the cloud WIDTH is the easy arm's strength, 2.0, so the two arms differ only
  in where the cloud sits — two tight blobs at the anchors against the same
  blob smeared along the whole path — and not in how wide it is;
- the realised mixing coordinate is uniform after guard rejection: ten-bin
  counts [208, 194, 212, 216, 187, 199, 207, 199, 214, 212], mean t 0.503,
  Kolmogorov-Smirnov distance to U(0, 1) 0.0127 against a 95% threshold of
  0.0301. So "neither rig especially likely" survived the truncation; 1017
  entries took Michael's per-clip dynamics and 1031 DREGON's (the nearer
  anchor's, since the per-clip dynamics have no path convention);
- acceptance 1.21 attempts per draw, 82.7% first try; guards that fired:
  `ltas` 293, `speed_law` 162, `trend_falls` 2, `parity_sign` 1;
- **the speed exponents are BOUNDED to the fitted range**, as in the easy arm:
  `amp_exp` to [4.398, 14.111] and `floor_exp` to [0, 6.792], the span the six
  measured fits occupy (`results/rig_sampler/structure.json:between_rig.per_fit`;
  `floor_exp`'s measured minimum is -3.711, raised to 0 because a negative
  floor exponent diverges at zero rotor speed). The bound clipped 672 `amp_exp`
  and 449 `floor_exp` draws of 2049, of which 1 was then refused by the guards
  and redrawn; the realised bank is `amp_exp` 8.76 +- 3.50 and `floor_exp`
  2.93 +- 2.82, both filling their bounds. Unbounded, this arm had reached
  31.9 dB/dB, at which an idle window's comb sits ~20 dB under the same rig at
  cruise;
- realised spread: rotor gain -23.0 +- 7.6 dB, trend slope -17.6 +- 4.8
  dB/decade, gamma0 6.6 +- 3.8 Hz, gamma_slope 0.36 +- 0.11 Hz/order, label
  error 1.31 +- 0.91 rev/s, `k_use` 109 for every entry (the path truncates to
  the shorter anchor's ladder);
- the build is bit-reproducible: an independent rebuild with the same arguments
  gives file SHA-256
  `b0a977a0d5eb9832b2ddabcbe77fa6b14d5772876eefe79a3d5e0870111209a6`, and the
  provenance's `inputs_digest` covers the anchors, the donor ranges, every
  width and the source of the builder, the sampler and the renderer.

The easy arm's remaining caveat applies unchanged: the per-microphone pattern
is perturbed per entry (sigma 1.12 dB gain, 0.20 dB floor) but not permuted
across channel indices, so the index-locked mechanism behind the fitted run's
r1/r2 ratio of 1.93 is softened, not removed.

## Comparison rows

| run | stream | what it tests |
|---|---|---|
| `rig_fitted_scv2_unified` | two fitted rigs as POINT presets | the fit itself; best real_r3 9.97 at ep 22, plateau 13.42, r1/r2 = 1.93 |
| `ctrl_diverse_scv2_unified` | wide measured ranges | diversity without the fits |
| `rig_easy_scv2_unified` | 2048 draws from the two fits' neighbourhoods | does turning each fit into a distribution transfer better |
| `rig_hard_scv2_unified` (this) | 2048 draws along the path between the fits | does covering the gap between rigs transfer better |
| `real_r4_scv2_unified` | REAL noise in training | the reference: val/real_r3 2.99 |

Select on `real_overall`, not `overall_macro`, for the reason the fitted-preset
run demonstrated: its synthetic half improved monotonically while every real
view degraded.

## Train

The bank is a gitignored BUILD PRODUCT (24.4 MB) and `omnirun` ships a clean
pushed checkout, so the bank does not travel — the job rebuilds it first. The
build is bit-reproducible from the pinned anchors and the seed (65 s on one
core), and it skips itself if the file is already current, so a restarted job
pays for it once:

```bash
omnirun submit --backend uni --gpus 1 --time 8h --yes -- bash -lc '
  python scripts/_build_rig_bank.py --mode path \
      --name rig_hard_n2048 --n 2048 --spread 2.0 --seed 20260914 \
      --anchor results/S2/cruise_8clip.json:fly125_cruise_00 \
      --dynamics conf/online_mix/rig_fitted_5050.yaml:1 \
      --anchor results/S2/dregon_room2_cruise_refined.json:free-flight_nosource_room2_cruise_00 \
      --dynamics conf/online_mix/rig_fitted_5050.yaml:0 \
  && python train.py experiment=rig_hard_scv2_unified'
```

The anchors live in `results/S2/`, which is also gitignored — those two
summaries must reach the backend before the build step, exactly as the fitting
jobs arrange. A missing bank fails at pool construction with the rebuild
command in the error message, not mid-run.

## Conclusion
