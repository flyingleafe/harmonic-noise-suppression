# Noise model v2 — secondary checks: what the data supports

Author: `CriteriaStudy`. Every number below is produced by
`scripts/noise_v2_likelihood_window.py` (→ `likelihood.json`) or
`scripts/noise_v2_spectrogram_proxy.py` (→ `proxy.json`) in this directory.
Figures are in `docs/explainers/noise-model-v2-plan/` and copied here.

This note RANKS and QUANTIFIES. It does not choose. Every threshold is
proposed as a GAP-CLOSURE FRACTION against an oracle floor, in the same form
as the HPPNet gate, with the implied absolute value on each rig.

## 0. What was measured, on what

Held-out supports only. Two frozen cohorts:

* DREGON: the five room-2 cruise scoring windows of
  `docs/revised-phase-baseline-manifest-v2.json` (4 s, all mics), carriers
  `motors_command` (raw) and `rps_refined` (posterior label);
* Michael's: the five FLY124 windows the frozen evaluator resolved —
  `FLY124@8+8` and `FLY124@16+8` (standby), `FLY124@27.68+8` (ramp, 1.0 s in
  regime), `FLY124@40+8` and `FLY124@56+8` (cruise) — carrier `rps`.

Every guarded training support is excluded from every measurement: the legacy
stage-2 16 s windows AND the C3 fit windows, each widened by the hashed
1.024 s training guard.

For the line-shape study the scoring window is extended FORWARD by one stated
rule (inside one legal in-regime span, clear of every guarded training
support, cap 16 s). Achieved line-shape supports:
`free-flight 16.00 s`, `hovering 6.98 s`, `updown 4.00 s`,
`rectangle 4.70 s`, `spinning 4.76 s`; Michael's `16.00 / 15.06 / 16.00 /
10.50 / 16.00 s`. Three DREGON recordings cannot supply more than 4.0-4.8 s of
legal held-out cruise, which is itself a constraint on any window choice.

## 1. Check A — the held-out spectral line: what window the data supports

### 1.1 The comb is only visible on Michael's, and only at low order

Median line SNR over the local baseband floor, at `T = 4.096 s`
(figure `criteria_lineshape_summary.png`, bottom row; the dashed line is the
preregistered 10 dB moment gate):

| rig / carrier | k=1 | k=2 | k=4 | k=8 | k=16 | k=32 |
|---|---|---|---|---|---|---|
| dregon / motors_command | 6.4 | 4.3 | 3.5 | 2.7 | 2.7 | 1.8 |
| dregon / rps_refined | 7.1 | 4.2 | 3.6 | 1.9 | 2.7 | 1.8 |
| michaels / rps | 4.7 | **24.4** | **13.6** | 7.6 | 3.3 | 2.3 |

Fraction of (support, rotor, mic) cells passing the 10 dB gate at
`T = 4.096 s`: DREGON `motors_command` 0.31 at k=1 and 0.00-0.06 at every
k >= 2; Michael's 0.95 at k=2, 0.85 at k=4, 0.23 at k=8, 0.00 at k >= 16.

**The two DREGON carriers are indistinguishable for this purpose.** At k=1,
`motors_command` gives 6.4 dB and `rps_refined` 7.1 dB; the half-power widths
agree to 0.13 Hz at every window. A line-shape criterion cannot be used to
argue for or against the refined label.

Consequence for a v2 criterion: **a per-harmonic line-shape check is
measurable on Michael's cruise at k = 2 and 4 only.** On the five DREGON
room-2 cruise supports the comb above the shaft line is not above the floor
at any window between 0.128 s and 4.096 s.

### 1.2 The line core is UNRESOLVED at every window up to 4.096 s

Median half-power width against the periodic Hann's own resolution 1.44/T
(figure `criteria_lineshape_summary.png`, top row):

| T (s) | 1.44/T (Hz) | michaels k=2 (Hz) | michaels k=4 (Hz) | dregon k=1 (Hz) |
|---|---|---|---|---|
| 0.128 | 11.25 | 10.12 | 11.60 | 12.32 |
| 0.256 | 5.63 | 5.27 | 6.67 | 7.00 |
| 0.512 | 2.81 | 2.77 | 4.18 | 3.77 |
| 1.024 | 1.41 | 1.46 | 2.02 | 2.36 |
| 2.048 | 0.70 | 0.83 | 1.17 | 1.21 |
| 4.096 | 0.35 | 0.46 | 0.69 | 0.50 |

The measured width tracks 1.44/T over a factor of 32 in window length. The
line's own half-power width is therefore **below 0.46 Hz on Michael's k=2**,
and the measurement is window-limited, not line-limited, everywhere.

Read against the revised model: a per-harmonic Wiener phase with diffusion
`D` gives a Lorentzian of full half-power width `D / pi` Hz. `< 0.46 Hz`
implies `D < 1.45 rad^2/s` on Michael's cruise, which brackets the C3
Michael's fit (`D = 1.6308`, predicted width 0.52 Hz) and REFUTES the C3
DREGON fit (`D = 844.69`, predicted width 268.9 Hz) wherever a line is
visible at all — 269 Hz is 7.0 times the whole isolation bandwidth
`B = f_r/2 = 38.5 Hz` of the DREGON cruise comb.

### 1.3 There is a narrow core on a broad pedestal

Power share of the baseline-subtracted line inside +-2 bins, median over
cells that pass the 10 dB gate (`criteria_lineshape_summary.png`, second row):
Michael's k=2 gives 1.00 / 0.73 / 0.39 / 0.38 / 0.34 / 0.28 at
T = 0.128 ... 4.096 s. Above T = 0.512 s the share is flat at 0.28-0.39: the
remaining 61-72 % of the excess power is a genuine broad pedestal inside
`|f| <= B`, not window leakage (window leakage would keep shrinking with T).

The same quantity at k=4 falls 1.00 -> 0.23 and at k=8 1.00 -> 0.20. The
pedestal share GROWS with order, which is the qualitative signature the
revised model's `k theta_r` term predicts (a `k`-scaled broadening on top of a
`k`-independent core).

### 1.4 Window invariance, and why it cannot pick the window by itself

`window_invariance_l1` is the L1 distance between the normalised line shape at
`T` and at `2T`, rebinned onto the coarser grid (in `[0, 2]`). Median over
gated Michael's k=2 cells: 0.43 / 0.56 / 0.45 / 0.57 / 0.53 / 0.65 at
T = 0.128 ... 4.096 s. The split-half null — the same statistic between two
disjoint halves of the same material at the same `T` — is 0.16 / 0.28 / 0.28 /
0.40 / 0.62 / 0.95.

The excess `L1 - null` is +0.27 / +0.28 / +0.17 / +0.17 / -0.10 / -0.29: the
shape still changes when the window is doubled up to about 1 s, and above
about 2 s the change is no longer larger than estimator noise. The null is
computed on half the material and so is biased UPWARD, which makes the excess
a conservative lower bound; the statistic is dominated by Welch variance, not
by resolution, and should NOT be used as a gate on its own.

### 1.5 The non-stationarity time scales, and the window they justify

Amplitude-envelope autocorrelation 1/e time of the band-limited complex line
(`likelihood.json:lineshape.envelope`), medians over supports, rotors, mics:

| band | dregon (s) | michaels (s) | resolution limit (s) |
|---|---|---|---|
| comb isolation `+-B` | 0.011-0.026 | 0.017-0.034 | 0.013 (DREGON), 0.013-0.032 (Michael's) |
| narrow `+-2 Hz` | see `likelihood.json` | see `likelihood.json` | 0.25 |

At the comb's own isolation bandwidth the medians sit AT the resolution limit
`1/(2B)`, so the honest statement is: within `|f| <= B` the line amplitude is
not resolved as slow by that band — it is not that the envelope is 11 ms fast.
The p75/max reach 0.13-2.56 s on Michael's, i.e. some cells do carry a slow
envelope. **The envelope therefore does not bind the window.**

What does bind it is the telemetry carrier itself
(`likelihood.json:lineshape.supports[*].carrier_drift`). With
`|d f_r / dt|` measured on a 0.1 s-smoothed track, harmonic `k` drifts past
the window's own resolution above
`T_max(k) = sqrt(1.44 / (k |d f_r/dt|))`. On the Michael's standby support the
median slope is 0.230 rev/s per s (p95 0.801, max 2.14), giving
`T_max(k=2) = 1.77 s` at the median and 0.95 s at the p95.

**Window I would justify from the data: `T = 1.024 s` (periodic Hann, 50 %
overlap).** The four reasons, each a number above:

1. it is the longest window whose carrier drift stays inside the window's own
   resolution at the p95 slope (`T_max(k=2) = 0.95 s` at p95, 1.77 s at
   median);
2. it fits every held-out support: 6 Welch segments on the shortest DREGON
   support (4.0 s) against 2 at T = 2.048 s and 1 at T = 4.096 s;
3. it is past the point where doubling the window still changes the shape
   beyond noise (excess +0.17 at 1.024 s, negative at 2.048 s);
4. it already collects a gated line on Michael's k=2 (median SNR 17.5 dB,
   80 % of cells above 10 dB).
   `T = 2.048 s` buys 3.2 dB more line SNR (20.7 dB) and would be the choice
   if only Michael's mattered; it costs two thirds of the DREGON segment count
   and sits above the p95 drift limit.

## 2. Check A — held-out composite risk per arm per NFFT

Arms, all through the evaluator's own primitives
(`revised_eval.marginal_frame_nll` / `FrameScore` / `composite_score`), band
30-7900 Hz, all 8 mics, `hop = NFFT/16`:

* `oracle_np` — `M` is the Welch MEAN PERIODOGRAM of a speed-matched DISJOINT
  real segment of the same recording at the same NFFT. No model, no carrier;
* `current_best` — the legacy stage-2 export the frozen evaluator selects
  (`read_export` + `aggregate_nuisance` + `predicted_m`);
* `c3` — `revised_phase.predict_spectrum(mode='prior')` of the round-3 C3
  export (DREGON `D = 844.69`, `sigma = 3.1012`; Michael's `D = 1.6308`,
  `sigma = 4.2072`).

**NOT COMPUTED in this run.** Reason: cost of the local `predict_spectrum`
path. Measured on this workstation, a 4 s DREGON clip costs 97.7 s at
NFFT 16384 / hop 1024 (47 frames, 8193 bins), 89.1 s at 4096/256 and 79 s at
2048/128 — about 20 s of model time per audio-second per setting. The declared
grid is 5 DREGON supports x 4 s plus 5 Michael's supports x 8 s = 60
audio-seconds x 3 settings, i.e. ~75 min of `predict_spectrum` plus ~15 min of
`predicted_m`; the run reached 2 h 31 min under peer CPU contention and was
stopped before it wrote. It could not be sent to `uni-cpu`: `omnirun` ships
the git revision only, and the C3 exports (under `.worktrees/**`) and
`results/S2/cruise_8clip_refined.json` / `standby.json` are gitignored, so
both submitted jobs (`nv2-criteria-like-e0991b`,
`nv2-criteria-proxy-40fc74`) failed with `FileNotFoundError` on the C3 export.

The stage is implemented, smoke-tested end to end and idempotent. To produce
the table, either track those four artefacts and submit

    omnirun submit --backend uni-cpu --gpus 0 --time 4h --cpus 8 --mem 48 \
      --outputs 'results/noise_v2/**' -- \
      python scripts/noise_v2_likelihood_window.py --stage composite --threads 8

or run the same command locally (`--stage composite` keeps the existing
line-shape section of `likelihood.json` and only adds the composite one). A
reduced grid that lands in about 10 minutes is
`--stage composite --supports-per-rig 1 --nffts 2048`.

What the stage emits, so the reader knows what is missing: per (rig, regime,
support, arm, NFFT) the score in nats per unique second and per band cell, the
pooled score per (rig, regime, arm, NFFT), and the gap-closure fraction
`(c3 - x) / (c3 - oracle_np)` on both conventions
(`likelihood.json:composite.gap_closure`). The frozen reference points it
would be read against are already known: DREGON `E_real = 1.2187`,
`E_best_synth = 2.1878` rev/s, and the C3 DREGON export's
`sigma = 3.1012`, `D = 844.69`.

## 3. Check B — the spectrogram-similarity proxy

### 3.1 The oracle has to be speed-matched, or it is not a floor

First attempt took the earliest legal disjoint segment. On
`free-flight_nosource_room2` that gave a real-vs-real `mr_ltas` of 3.17 dB
against 2.45 dB for the current-best synthetic arm: **the oracle read WORSE
than the model**, because the synthetic arm is rendered on the scored
window's own carrier while an arbitrary later segment of a moving flight sits
at other rotor speeds and other levels.

The rule was therefore fixed and stated: among every candidate start on a
0.25 s grid inside the legal disjoint spans, take the one minimising the
per-rotor mean-speed mismatch. Achieved mismatches: DREGON 0.397 / 1.725 /
1.488 / 0.984 / 1.846 rev/s; Michael's 0.020 / 0.047 / 8.286 (ramp) / 0.877 /
0.389 rev/s. The DREGON `free-flight` floor fell from 3.17 to 1.83 dB and the
oracle became a floor.

**This is a design constraint for model v2's evaluation, not a detail: any
real-vs-real floor must be carrier-matched, and the achieved mismatch must be
reported with it.** The Michael's RAMP support has no speed-matched partner
(8.29 rev/s mismatch) and its oracle is not usable.

The split-half oracle — the scored window's own two halves — is NOT tighter:
4.05 dB on DREGON cruise and 1.63 dB on Michael's cruise against 1.83 and
1.03 dB for the speed-matched disjoint segment, because each half is only
half as long. Use the speed-matched disjoint segment.

### 3.2 Whitening in the log domain is a proven no-op

Dividing both clips by one whitener is an additive per-bin constant in the log
domain and cancels exactly in a difference of logs. Measured:
`max |unwhitened - unfloored-whitened| = 6.7e-15 dB` over all 142 rows. The
floored variant differs only where the floor bites: identical to 0.001 dB on
DREGON cruise (1.827 vs 1.827), and 0.33-1.63 dB on Michael's standby/ramp
where levels are low. **A whitened multi-resolution LTAS is not a new
candidate; it is the unwhitened one plus a floor.** Figure
`criteria_proxy_spectrogram.png` shows the whitened and unwhitened
log-spectrograms of real vs current-best on
`free-flight_nosource_room2@1512727397.205+4`.

### 3.3 The candidates, on CRUISE (the comparable regime)

Figure `criteria_proxy_bars.png` (bars) and `criteria_proxy_ladder_d.png` /
`criteria_proxy_ladder_sigma.png` (ladders). `sep` is
`(current_best - oracle) / (oracle spread across supports)`; `span` is the
full range of the pooled ladder; `rho` is the Spearman correlation on the
DEGRADATION branch (scale >= 1).

DREGON cruise (5 supports):

| candidate | oracle | spread | best | C3 | sep | span(D) | span(sigma) | rho(D>=1) | argmin D |
|---|---|---|---|---|---|---|---|---|---|
| `mr_ltas` | 1.827 | 0.766 | 2.302 | 2.179 | 0.62 | **5.481** | 0.063 | **+1.00** | x1 |
| `ltas_abs_db` | 1.785 | 1.142 | 2.126 | 2.431 | 0.30 | **6.174** | 0.030 | **+1.00** | x1 |
| `texture` | 0.786 | 0.196 | 1.046 | 1.061 | 1.33 | 0.053 | 0.027 | -0.40 | x1/30 |
| `modulation` | 0.647 | 0.048 | 0.646 | 0.650 | -0.03 | 0.010 | 0.003 | +0.40 | x1/30 |
| `texture_lines` | 0.771 | 0.189 | 1.037 | 1.058 | 1.40 | 0.047 | 0.039 | +0.20 | x1/30 |
| `modulation_lines` | 0.649 | 0.054 | 0.644 | 0.650 | -0.10 | 0.012 | 0.005 | +0.60 | x1/30 |

Michael's cruise (2 supports):

| candidate | oracle | spread | cross-rec | best | C3 | sep | span(D) | span(sigma) | rho(D>=1) | argmin D |
|---|---|---|---|---|---|---|---|---|---|---|
| `ltas_abs_db` | 1.085 | 0.053 | 1.670 | 1.332 | 1.533 | **4.62** | **1.522** | **1.809** | **+0.90** | x1/3 |
| `mr_ltas` | 1.030 | 0.259 | 1.254 | 1.554 | 1.956 | 2.02 | 0.135 | 0.370 | 0.00 | x3 |
| `texture` | 0.428 | 0.035 | 0.434 | 0.404 | 0.735 | -0.67 | 0.397 | 0.398 | -1.00 | x30 |
| `modulation` | 0.445 | 0.001 | 0.449 | 0.449 | 0.472 | 3.68 | 0.024 | 0.028 | -0.90 | x30 |
| `texture_lines` | 0.426 | 0.036 | 0.428 | 0.402 | 0.717 | -0.66 | 0.367 | 0.377 | -1.00 | x30 |
| `modulation_lines` | 0.446 | 0.004 | 0.445 | 0.451 | 0.473 | 1.23 | 0.018 | 0.031 | -0.40 | x30 |

(`mr_ltas_whitened` is omitted: it equals `mr_ltas` to 1e-14 dB unfloored, and
to 0.001 dB floored, on cruise.)

The pooled D ladder, DREGON cruise, `ltas_abs_db` in dB at
`D x {1/30, 1/10, 1/3, 1, 3, 10, 30}`:
`8.604, 7.197, 4.899, 2.431, 2.772, 5.661, 7.487` — a clean V with its
minimum exactly at the C3 value and a 6.17 dB dynamic range.
The same ladder for `texture`: `1.008, 1.038, 1.040, 1.061, 1.055, 1.051,
1.061` — a 0.05 dB range, i.e. no response at all.

### 3.4 Ranking

1. **`ltas_abs_db`** — the FROZEN absolute-level band LTAS error that the
   campaign already gates on. Smallest relative oracle floor on Michael's
   cruise (1.085 dB, spread 0.053), the largest separation ratio (4.62), the
   largest ladder dynamic range in BOTH parameters (6.17 dB in D on DREGON,
   1.81 dB in sigma on Michael's), a V-shaped D ladder whose minimum sits at
   the C3 value (DREGON) or one ladder step below it (Michael's), and a
   monotone degradation branch (rho = +1.00 / +0.90). It is the only candidate
   that responds to `sigma` at all.
2. **`mr_ltas`** — the multi-resolution variant. Best D response on DREGON
   (5.48 dB span, rho = +1.00, argmin exactly x1) but nearly dead on Michael's
   cruise (0.135 dB span, rho = 0.00) and its oracle spread on DREGON is large
   (0.766 dB, 42 % of the floor). Keep as a secondary, not as the gate.
3. **`texture` / `texture_lines`** — REJECT. No D or sigma response on DREGON
   (0.05 dB span), and on Michael's it falls monotonically as D and sigma grow
   with its minimum pinned at the x30 boundary: it prefers ever more smearing
   and cannot locate either parameter. Its apparent separation (1.33-1.40 on
   DREGON) is a level effect, not a phase effect.
4. **`modulation` / `modulation_lines`** — REJECT. Ranges of 0.010-0.031 on a
   `[0, 2]` scale, i.e. inside the numerical noise of the statistic; the
   DREGON separation ratio is -0.03 (the current-best arm sits BELOW the
   oracle floor).

Restricting texture and modulation to the comb bins (346 of 1008 in-band bins
at NFFT 2048) changes nothing: `texture_lines` 0.771 vs `texture` 0.786 on the
DREGON oracle, spans 0.047 vs 0.053. The problem is the statistic, not the
band.

## 4. Proposed thresholds, as gap-closure fractions

The form, for a quantity where lower is better:

    closure = (x_bad - x_candidate) / (x_bad - x_oracle)

`closure = 1` reaches the oracle, `closure = 0` is no better than the
known-bad arm. `x_bad` is the C3 arm on the same supports with the same front
end; `x_oracle` is the speed-matched real-vs-real floor.

Where the current best already sits, measured:

| check | rig / regime | oracle | C3 (bad) | current best | closure of current best |
|---|---|---|---|---|---|
| proxy `ltas_abs_db` | dregon / cruise | 1.785 | 2.431 | 2.126 | 0.47 |
| proxy `ltas_abs_db` | michaels / cruise | 1.085 | 1.533 | 1.332 | 0.45 |
| proxy `mr_ltas` | dregon / cruise | 1.827 | 2.179 | 2.302 | **-0.35** |
| proxy `mr_ltas` | michaels / cruise | 1.030 | 1.956 | 1.554 | 0.43 |
| likelihood | not computed (section 2) | | | | |

### Proposal B — spectrogram proxy

**Quantity: `ltas_abs_db` (the frozen absolute-level band LTAS error, mic 0),
per rig on the cruise supports. Threshold: closure >= 0.70 of the
C3-to-oracle gap.** Implied absolute values:

* DREGON cruise: `2.431 - 0.70 x (2.431 - 1.785) = 1.979 dB` (current best
  2.126 dB, closure 0.47 — the current best would FAIL this gate);
* Michael's cruise: `1.533 - 0.70 x (1.533 - 1.085) = 1.220 dB` (current best
  1.332 dB, closure 0.45 — also FAIL).

0.70 is the fraction the DREGON HPPNet gate already uses
(`gates.dregon_gap_fraction = 0.7` in the frozen manifest), so adopting it
here introduces no new number. It is a demanding but reachable target: the
ladder shows 0.95 dB of headroom between the C3 D and the DREGON minimum, and
1.52 dB of D-range plus 1.81 dB of sigma-range on Michael's, so a correct
`(D, sigma)` is worth more than the 0.45 dB the gate asks for.

A weaker alternative, if Main wants the current best to pass:
**closure >= 0.40** gives 2.173 dB (DREGON) and 1.354 dB (Michael's), which
both arms clear by 0.02-0.05 dB — a gate that certifies no improvement. I do
not recommend it.

Both numbers must be read with the oracle spread: DREGON 1.142 dB across five
supports (larger than the whole C3-to-oracle gap of 0.646 dB) against
Michael's 0.053 dB across two. **On DREGON this gate is support-noise limited
and should be stated as a five-support mean with the spread quoted; on
Michael's cruise it is tight.**

### Proposal A — held-out spectral likelihood

Quantity: the frozen composite risk on the held-out supports, same band, same
mic set, against the non-parametric oracle (`oracle_np`) and the C3 arm.
Threshold in the same gap-closure form. **The absolute values cannot be stated
here: the composite table was not computed (section 2).** What the data DOES
settle about this check:

* the resolution to use. The line's own half-power width is below 0.46 Hz
  (section 1.2), i.e. narrower than one bin at every NFFT in
  `{2048, 4096, 16384}` (bin widths 7.81 / 3.91 / 0.977 Hz). Only NFFT 16384
  puts the core inside a couple of bins, so the frozen 16384/1024 front end is
  the one a line-sensitive likelihood needs; 2048 and 4096 smear the core into
  the floor and should be read as robustness checks, not as the primary;
* the window for the line-shape companion measurement: `T = 1.024 s`
  (section 1.5);
* where the check has any signal: `k = 2` and `k = 4` on Michael's cruise
  (section 1.1). On DREGON room-2 cruise a per-harmonic line criterion has no
  measurable line to score at any window between 0.128 s and 4.096 s, so the
  DREGON arm of this check can only be a broadband/floor statement;
* the threshold FRACTION to reuse: 0.70, the frozen
  `gates.dregon_gap_fraction` of the baseline manifest, so no new number
  enters. Its implied absolute value follows directly from section 2's table
  once that table exists.

## 5. Caveats

* The degradation ladder is rendered from the C3 REVISED export, because the
  legacy current-best arm has no `D` and no `sigma` to scale (its phase model
  is a fixed coherent/Lorentzian mixture). The ladder therefore measures each
  candidate's sensitivity to the two parameters v2 must identify, on the real
  carrier, at one render seed (2001, one of the frozen
  `null_variation.seeds`). It is not a ladder of the current-best arm.
* Michael's cruise has only two held-out supports, so its oracle spread
  (0.053 dB) is a two-sample standard deviation.
* The composite-risk stage and the proxy stage ran on the laptop, not on
  `uni-cpu`: `omnirun` ships the git revision only, and both the C3 exports
  (under `.worktrees/**`) and `results/S2/cruise_8clip_refined.json` /
  `standby.json` are gitignored, so the submitted jobs died with
  `FileNotFoundError` on the C3 export. Anything that has to score these arms
  remotely needs those artefacts tracked first.
* `mr_ltas` uses the time-average of the log magnitude (not the log of the
  time-average), 30-7900 Hz, equal weight per bin, no per-arm normalisation.
