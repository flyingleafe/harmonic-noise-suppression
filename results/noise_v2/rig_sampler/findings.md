# The v2 rig sampler: widths, guards, and the two preset banks

**What this is.** `src/experiments/noise_model/rig_sampler.py` draws perturbed
`noise-v2-fit/2` payloads around the campaign's two fitted rigs, and
`scripts/noise_v2_build_bank.py` writes them as `noise-v2-bank/1` preset banks
for the `nv2_*` transfer arms. This file is the measurement record: every width,
every guard rate, the coverage that chose the strength, and the two banks'
identities. Figures: `widths_ladder.png`, `coverage.png`, `profiles.png`,
`path_cloud.png`. Machine-readable: `structure.json` (phase A + the coverage
ladder), `build_easy.json`, `build_hard.json`.

Anchors, both speed-law **pinned** to the v2 short-span contract (`amp_exp` 2.0,
`floor_exp` 2.0, `floor_static_rel` 2.5e-3, the model's prior medians, with the
pre-pin values in each entry's `params.span_pin_record`):

| rig | cruise fit | k_max | standby | traj_rig |
|---|---|---|---|---|
| dregon | `round5/fits/dregon_room2_floor__flight_profile.json` | 88 | none | `dregon` |
| michaels | `round3/fits/michaels_fly125_cruise__flight.json` | 81 | `michaels_fly125_standby__flight.json` | `michaels` |

The pin is applied at the 80 rev/s reference, where every speed factor is
exactly 1, so it leaves a cruise spectrum untouched (verified to 1e-9 dB in
`tests/experiments/test_noise_model_rig_sampler.py`). It DOES move a standby
payload at its own 20–40 rev/s: Michael's standby fit's own `floor_exp` 4.93 and
`amp_exp` 4.13 become 2.0, which raises its 30 rev/s floor by about 12 dB and its
comb by about 9 dB. That is a consequence of the campaign decision, it is
recorded per entry, and the self-check renders standby windows so the result is
measured rather than assumed.

## 1. The widths, and the three tiers

`strength = 1` is the **between-restart** spread — the same fit, the same data,
a different optimiser seed. R5's restarts `s0..s3` carry it for everything the
flight-profile fit leaves free; R5 FROZE `gamma_hz`, `sigma_nu` and `lam` from
the bench (`frozen_from.gamma_rotor_matched`), so their between-restart spread
in that family is identically zero and the four DREGON bench `Motor{1..4}_70`
restart families carry it instead.

| coordinate | strength-1 sigma | between-rotor | between-rig | -> rotor | -> rig |
|---|---:|---:|---:|---:|---:|
| rotor_gain_db (dB) | 0.3662 | 1.7039 | 3.3314 | 4.7 | 9.1 |
| slope_db_dec (dB/decade) | 0.3700 | 1.5279 | 1.7941 | 4.1 | 4.8 |
| resid_redraw_frac | 0.0903 | 0.6402 | 0.9628 | 7.1 | 10.7 |
| gamma_level_ln | 0.0618 | 0.2452 | 0.0102 | 4.0 | 0.2 |
| gamma_order_ln | 0.3060 | 1.5348 | 2.4208 | 5.0 | 7.9 |
| sigma_nu_ln | 0.0121 | 0.1586 | 0.9700 | 13.1 | 80.3 |
| lam_ln | 0.2495 | 2.0549 | 1.4899 | 8.2 | 6.0 |
| floor_mean_db (dB) | 0.0377 | 1.5364 | 1.6726 | 40.7 | 44.3 |
| floor_tilt_db_oct | 0.1470 | 2.9946 | 3.8973 | 20.4 | 26.5 |
| floor_shape_z | 0.1824 | 3.0236 | 4.0243 | 16.6 | 22.1 |
| mic_line_gain_db (dB) | 0.1459 | 6.3724 | 3.7842 | 43.7 | 25.9 |
| mic_floor_db (dB) | 0.0767 | 7.1335 | 2.8619 | 93.0 | 37.3 |
| mic_gains_db (dB) | 0.1266 | 7.0861 | 4.1350 | 56.0 | 32.7 |
| **level_db (dB)** | **1.3609** | — | — | — | — |

between-rotor = the standard deviation over the four bench motors at throttle
70; between-rig = `|rig-mean difference| / sqrt(2)`, the per-draw sigma two
independent draws would need in order to differ that much. The last two columns
are the strength at which a strength-1 draw reaches those tiers.

**The ladder's headline is that one scalar cannot walk every coordinate up
together.** A rotor's profile gain reaches the between-rotor spread at strength
4.7 and the trend slope at 4.1, but the floor LEVEL needs 40.7 and the
microphone pattern 44–93, because a 255-frame flight fit reproduces its floor
level to 0.038 dB across restarts while four real rotors differ by 1.54 dB. A
between-restart spread is a convergence tolerance for those coordinates, not a
width.

**The one coordinate on a different tier.** `level_db` moves the whole rig —
every rotor's `profile_db` and `floor_mean_db` by the same dB — and its
strength-1 width is **between-clip**: the standard deviation of the 300–7900 Hz
band level over each rig's own real windows, 2.006 dB over DREGON's ten frozen
room-2 windows, 1.157 dB over Michael's eight cruise windows, 0.438 dB over
Michael's three standby windows, 1.361 dB pooled as an RMS. Used at the
between-restart tier (0.038 dB) it would clone all 2048 entries at one level,
which is exactly the point-ness the transfer pair exists to break. This is the
only coordinate whose tier differs, it is labelled `between_clip` in
`structure.json`, and it is the coordinate the coverage criterion turns out to
depend on.

**Residual structure.** The measured squared-exponential correlation length of
the profile residual in log10 order is 0.024 decades (variogram, median over the
16 measured rotor profiles; twelve of them sit at 0.005–0.045), i.e. the v2
residual is essentially WHITE in order — a positive kernel cannot represent the
even/odd blade-passing alternation at all, which is why the variogram collapses
to the grid edge. The v2 decomposition therefore carries NO separate parity
envelope (the legacy sampler's fourth term): the split lives inside the
residual, and the redraw is a MIXTURE
(`resid' = rho resid + sqrt(1 - rho^2) sigma z`, `rho = 1 - r^2/2`,
`r = strength * 0.0903`), so at strength 3 `rho = 0.963` and an entry keeps its
anchor's parity structure to within a few percent while its fine structure still
moves. `profiles.png` shows that directly: Michael's zig-zag survives in every
one of 120 plotted draws.

**Trend guard calibration.** The smallest total trend drop over the 16 measured
v2 rotor profiles is 4.47 dB, so the 3 dB guard admits every measured profile.

## 2. The guards

Reject and redraw, never clip; `check_sample` returns every number.

| guard | rule | calibration |
|---|---|---|
| `finite` | the v2 EXPECTED periodogram is finite and strictly positive at the 80 rev/s cruise probe AND the 30 rev/s idle probe | the idle probe reads the entry's STANDBY payload when it has one, because that is what the regime composition renders near idle |
| `trend_falls` | each rotor's trend falls >= 3 dB from k=1 to k=K, cruise and standby payloads alike | smallest measured drop 4.47 dB |
| `gamma_excursion` | per-rotor mean log `gamma_hz` within 1.5x of the reference | stated, not measured |
| `ltas` | the draw's expected periodogram in 1/3-octave bands over 300–7900 Hz within 2x the measured REAL clip-to-clip band RMS of the anchor's own windows | DREGON 2.660 dB -> 5.319 dB tolerance; Michael's 2.326 dB -> 4.653 dB. On the path the tolerance is interpolated in t |
| `speed_law` | `amp_exp >= 0` and `floor_exp >= 0` | pinned entries cannot fail it; checked because a caller may hand in an unpinned anchor |

The expected periodogram is `experiments.noise_model.render`'s forward model on
ONE frame of the declared 2 s window: with a constant carrier every frame of the
window has the same expectation, verified to 0.0 dB against the full 59-frame
window, so the guard costs 0.2 s instead of 8.5 s.

## 3. Coverage: how strength 3.0 was chosen

Per rig, the fraction of 1/3-octave bands in which the CLOUD (min-to-max over
accepted draws) brackets each real window's measured band level — the legacy
transfer pair's criterion (`docs/experiments/rig-sampler-transfer-pair.md`).
Target 90 % above 300 Hz on both rigs and both presets. Ladder at 128 draws per
cell:

| strength | easy dregon | easy michaels | hard dregon | hard michaels | acceptance (easy/hard) | exhausted |
|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 0.723 | 0.913 | 0.792 | 0.808 | 1.000 / 0.992 | 0 / 0 |
| 2.0 | 0.962 | 0.952 | **0.892** | 0.981 | 0.810 / 0.865 | 0 / 0 |
| **3.0** | 0.915 | 0.962 | 0.938 | 1.000 | 0.557 / 0.566 | 0 / 0 |
| 4.0 | 0.969 | 0.971 | 0.946 | 1.000 | 0.239 / 0.282 | **3** / 0 |
| 6.0 | 0.985 | 0.962 | 0.900 | 0.962 | 0.055 / 0.063 | **54** / **46** |

**The default 2.0 misses, on one cell: the hard cloud brackets 89.2 % of
DREGON's bands above 300 Hz against the 90 % target.** Strength 3.0 is the first
ladder setting that clears the target everywhere with no draw exhausting its
16-attempt budget, so it is what both banks are built at. Beyond 4.0 the guards
stop being a filter and become a wall — at 6.0 more than a third of the draws
cannot be realised at all — which is the honest upper bound on this family's
width, set by the gamma cap and the LTAS envelope, not by taste.

At the built banks' full size the clouds are wider than the ladder's 128-draw
sample and coverage is correspondingly higher:

| bank | dregon > 300 Hz | dregon all bands | michaels > 300 Hz | michaels all bands |
|---|---:|---:|---:|---:|
| easy | 100.0 % | 100.0 % | 99.0 % | 99.4 % |
| hard | 98.5 % | 96.2 % | 100.0 % | 100.0 % |

## 4. The two banks

| | easy | hard |
|---|---|---|
| file | `noise_v2_easy_n2048.json` | `noise_v2_hard_n2048.json` |
| sha256 | `cea6c5ad56b1337d…` | `b5ff345ecd2b8289…` |
| content digest | `16ad7d6797aec704…` | `b43eacd34b993d70…` |
| size | 29.8 MB | 29.3 MB |
| entries | 1024 dregon + 1024 michaels | 2048 on the cruise-to-cruise path |
| mode | neighbourhood, strength 3.0 | path, spread 3.0, K = 1..81 |
| `traj_rig` | `dregon` / `michaels` | `null` (the policy's rig hyperprior flies them) |
| standby slot | 1024 (every Michael's entry) | 1028 (50.2 %, carried with probability t) |
| attempts per entry | 2.017 (53.6 % first try) | 1.778 (57.0 % first try) |
| draws rejected | 2082 (50.4 %) | 1593 (43.8 %) |
| guards fired | `ltas` 1041, `gamma_excursion` 859, `trend_falls` 697 | `ltas` 913, `gamma_excursion` 728, `trend_falls` 232 |
| orders dropped | none | dregon 7 (orders 82–88), michaels 0 |
| wall | 915 s | 980 s |

`ltas` is the most active guard in both banks and `trend_falls` is three times
more active in the easy bank than in the hard one, which follows from where the
draws sit: a path point at intermediate t already has a shallower trend than
either anchor, so a perturbation is less likely to tip it past the 3 dB margin
than the same perturbation on DREGON's steep rotor 3.

**The mixing coordinate survives the truncation.** Realised t over the 2048
accepted hard entries: mean 0.5009, KS distance to U(0,1) **0.0166** against the
95 % threshold **0.0300** (passes), decile counts
`[233, 194, 186, 214, 196, 197, 191, 213, 192, 232]`. The standby slot is a
regime POLICY carried with probability t, and `path_cloud.png` shows the
realised fraction per decile tracking the diagonal.

**Wall time.** 915 s (easy) and 980 s (hard) on 11 workers of a 14-thread
laptop, i.e. **~16 min per bank, not under 15**: the guard's expected
periodogram is 0.2 s per probe, an accepted entry costs about 1.8 attempts
times two probes, and 2048 of them do not compress further without giving up
the idle probe. The banks are therefore PUBLISHED as a pinned dload dataset
rather than rebuilt in-job — `noise-v2-banks`, version
`7f6a3242b353b1d0806f2add1ee0826dbc8032e904d9ae9825cf4c1d064d2925`, a flat tree
holding the two banks, `build_easy.json`, `build_hard.json`, `structure.json`,
`manifest.json` and `README.md`. The arms name
`dload:noise-v2-banks/noise_v2_{easy,hard}_n2048.json` and take the version
from the committed `dload.lock`; `--preset easy|hard` stays the reproduction
path.

## 5. Reproducibility

The bank payload carries **no build timestamp, no git HEAD, no worker count and
no wall time** — all of that lives in the sidecar `build_<preset>.json` — so a
bank's bytes are a function of its inputs alone. Each draw reads
`default_rng([seed, index])`, so the bank is identical at any worker count and
any completion order.

The skip digest covers the CODE as well as the inputs: sha256 of
`scripts/noise_v2_build_bank.py`, `src/experiments/noise_model/rig_sampler.py`,
`src/data_processing/noise_model/{render,params,spectrum,lag}.py` and
`results/noise_v2/rig_sampler/structure.json`, plus both anchors' sha256s, the
widths, the guard constants, the seed, the strength and the count. Verified:

* appending a no-op comment to `rig_sampler.py` moves the easy digest, and
  reverting the comment restores it exactly (measured on the pre-publication
  source: `a936e4d8cb78792f…` -> `0ed37a9dba39e1ce…` -> `a936e4d8cb78792f…`), so
  a code change rebuilds and an unchanged rerun skips;
* the PUBLISHED banks' digests equal what the committed source computes, checked
  by rerunning the builder WITHOUT `--force`: both presets print "already the
  bank this spec describes" and skip (easy `16ad7d6797aec704…`, hard
  `b43eacd34b993d70…`). The banks were rebuilt from the final committed source
  for exactly this reason — the pre-commit typing and formatting fixes that
  landed after the first build are non-functional, which the rebuild
  demonstrates: every draw statistic is unchanged (2.017 / 1.778 attempts per
  entry, the same guard counts 1041/859/697 and 913/728/232, the same realised
  t and KS 0.0166), and only the provenance's source hashes and hence the file
  bytes differ;
* determinism against the worker count and the completion order is pinned by
  `tests/experiments/test_noise_model_rig_sampler.py::test_bank_bytes_are_reproducible_and_the_digest_covers_the_code`,
  which builds the same spec at 1 and 2 workers and compares the serialised
  payloads byte for byte.

A skip is not blind: when the digest matches, the builder still loads the
existing bank and renders from it before exiting 0.

## 6. The self-check

Every build re-reads the bank from disk through `NoiseV2Pool` and renders CHOSEN
entries, each through its OWN one-entry pool, so the entry that was checked is
the entry that was named. An easy bank checks 4 DREGON and 4 Michael's entries
spread across the index range; a hard bank checks 8 spread over the mixing
coordinate with at least two carrying the standby slot. Every per-regime entry
is rendered a SECOND time on a standby-speed window (`rps_scale_range` 0.35,
windows redrawn until every rotor is at or under 40 rev/s), which is the only
way the two-regime smoothstep composition is exercised at all.

Easy bank: indices 0 / 341 / 682 / 1023 (DREGON, cruise-only) and 1024 / 1365 /
1706 / 2047 (Michael's, per-regime). Cruise windows 67.3–86.9 rev/s, RMS
0.0352–0.1126; the four Michael's standby windows 23.6–30.4 rev/s with a cruise
blend weight of 0.00 — the pure standby payload — RMS 0.0128–0.0329.

Hard bank: indices 455 (t=0.000), 1271 (0.130), 1974 (0.284), 1541 (0.420),
1072 (0.577), 1727 (0.730), 463 (0.867) and 1825 (0.023), the last four
per-regime. Cruise RMS 0.0233–0.0548, standby windows 23.6–30.4 rev/s, blend
0.00, RMS 0.0203–0.0280.

All renders finite, all 8 microphones, 2 s each, none silent. The levels are
absolute fitted ones, before an arm's own `normalize_rms_range`.
