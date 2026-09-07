# v4: the unified Gaussian model — design

Date: 2026-08-17. Status: design frozen for implementation; v3 paths must
stay bitwise intact behind default-off flags.

## The model (one sentence)

A window of drone audio is a zero-mean Gaussian process whose power
spectral density is a smooth broadband floor plus a comb of Lorentzian
lines riding the rotor trajectories:

    M_c(f, t) = S_c(f, t) + sum_{i,k} H_{c,i,k}(t) * L_{gamma_k}(f - k r_i(t))

with L_gamma a unit-peak Lorentzian of half-width gamma_k = max(0.6 k, gamma_min) Hz
(the measured linewidth law), S_c smooth in f (and piecewise/interpolated
in t), and H >= 0 the per-line powers. The phase-noise structure of v3
(theta shaft corrections, psi per-track corrections, both integrated OU
-> D2 penalties) is unchanged and enters through the line positions and
the coherent posterior.

Two deliberate consequences:

- The objective is the MARGINAL Whittle likelihood — the line processes
  are integrated out, so J has no envelope term at all:

      J_v4 = sum_{c,f,t} [ P_c(f,t)/M_c(f,t) + log M_c(f,t) ]
             + lambda_theta ||D2 theta||^2 + sum lambda_psi(k) ||D2 psi||^2
             (+ R(log S) as an explicit penalty, below)

  This is what the H-aware `total_h` bolt-on approximated; in v4 it IS
  the objective. `rent` and `data` are no longer separate stories.

- The decomposition channels are POSTERIOR estimates under the fitted
  model, not products of a separate split stage:
  comb channel = posterior mean of the line processes (block A with a
  ridge — see below); broadband = x − comb (exact identity preserved).
  Per-line powers H are first-class outputs — the generator's amplitude
  targets, by construction.

## The two fixes, concretely

### F1 — the floor inside the likelihood

Replace `masked_smooth_psd`'s projection with a penalized-Whittle fit of
log S jointly with H, per (channel, time-block), NO mask:

    minimize over (g = log S, H >= 0):
      sum_f [ P~(f)/(e^g + sum H L) + log(e^g + sum H L) ]
      + lambda_f || D2_f g ||^2

alternating two exactly-solvable-ish steps, 2–3 rounds:
  (a) H-step: given S, nonnegative fit of the Lorentzian amplitudes to
      the excess — the existing `_lorentzian_design` + NNLS machinery,
      per block (frames pooled by the block's median periodogram P~).
  (b) S-step: given H, damped Newton/IRLS on g with the pentadiagonal
      D2 penalty — banded Hessian (diag + D2^T D2), `solveh_banded`.
      Init g from the CURRENT masked fit (a good warm start and the
      fallback if Newton diverges: keep the iterate with lower objective).

lambda_f is a new hyperparameter and it must be a length scale in hertz
on the frequency axis (the "floor may not vary faster than B_f" knob),
calibrated once on the synthetic fixture to reproduce the current
cepstral-family smoothness (~400 Hz at 32 kHz / n_fft 4096). Time
smoothness stays structural (blocks + the existing time interpolation).

### F2 — proper (amplitude) priors on the envelopes

Block A keeps the banded solver; each track's prior precision gains a
diagonal ridge:

    prior_track = rho_k^2 D2^T D2 + beta_k I,   beta_k = c0 / H_k

with H_k the fitted line power of that (c, i, k) (block-piecewise; use
the block value at each envelope frame). Bandwidths open to the physical
law: b_A(k) = max(b0, 0.6 k) Hz with NO spacing cap — the ridge is what
makes the overlapping system positive-definite (the v3 spacing cap and
the coherent/stochastic split existed only because the prior was
improper). c0 is calibrated ONCE on synthetic: an OU line of known
(H, gamma) in known noise must come back with posterior power within
±20% of the Wiener target across k in {5, 20, 60}; record the
calibration in the tests.

Shrinkage behaves correctly by itself: strong coherent low-k lines have
H >> S·b so the ridge is negligible (no bias); floor-level tracks are
shrunk toward zero (no absorption). The WOLA stochastic split stage is
NOT run in the v4 arm — the comb channel already carries the line flanks.

Phase reading (block B) is unchanged, but reads only tracks whose
posterior is phase-informative: keep the existing concentration gate —
it is exactly the right filter and needs no new machinery.

## What v4 deletes (the simplification ledger)

- the stochastic WOLA split stage (v4 arm) — regions, per-bin gain,
  union bookkeeping;
- the mask in the floor fit (the whole "how wide is the mask" axis and
  its measured pathologies);
- the H-aware / marginal / adaptive-floor / h-lorentzian MEASURE bolt-ons
  (their content becomes the objective itself; keep the flags for v3
  comparison runs, freeze development on them);
- the coherent/stochastic boundary as an estimation concept (survives
  only as a presentation split by coherence time).

## Alternation (per window)

  0. init: carrier from labels/hypothesis; S from the current masked fit
     (warm start); H = 0.
  1. (S, H) step — F1 fit on the ORIGINAL periodogram (not the residual:
     the model explains lines through H, so no subtraction is needed and
     hard-EM bias through the residual is avoided).
  2. A step — banded solve with whitening 1/S and ridge c0/H (F2).
  3. B step — phase corrections from gated tracks (unchanged), under the
     existing k-trust ladder.
  4. repeat 1–3 (3 iterations as today); J_v4 read at convergence.

Note step 1 fits powers on the original: the coherent reconstruction is
NOT subtracted before the fit; H at low k then includes the coherent
line's power, which is correct (H is the line's total power; the
posterior split between "waveform captured" and "power only" is block
A's business, not the fit's).

## Seams and drivers

- `tracking.joint_decompose`: new `fit_floor_powers()` (F1), ridge
  support threaded into the block-A path (F2; `vk_envelopes` gains an
  optional per-track diagonal `ridge` argument), `map_objective_v4()`
  (or `map_objective(v4=True)`) computing J_v4; `JointConfig.v4: bool
  = False` master switch selecting the alternation above.
- `scripts/vk_decompose.py`: `--v4` arm (implies no stochastic stage;
  writes the same envelopes/residual/report products plus an `h_powers`
  table in the npz and a `v4` block in the report).
- `scripts/joint_rescore.py`: `--v4` (ranks by J_v4/cell).
- Everything default-off; the pinned v3b regression and all current
  tests must pass untouched.

## Acceptance gates (in order)

1. Synthetic (the regime-3 fixture + a blanket-band variant): fitted S
   within ±0.5 dB of truth INSIDE dense bands (the v3 failure); comb
   channel takes the lines, broadband PSD ≈ S (no carve, no dents);
   ridge calibration test passes.
2. w01 probe (local): the rotor-0-style carve numbers — comb take in a
   sole-owner band vs the track-free control band — must be comparable
   (no >1 dB extra stripping).
3. w01 rescore, 4 hypotheses at k=65: J_v4 ranking; the fan's
   floor-artifact channel is closed by construction — measure whether
   refined <= telemetry < fan finally holds.
4. Full recordings (cluster): gates >= v3e on all three recordings;
   FLY rotor-0 band carve gone; visual pass.

## Success criterion for the paper

If gates 1–4 pass, v4 is the centerpiece: one Gaussian model, one
marginal-likelihood objective, decomposition = posterior inference,
measure = the objective, amplitude targets = model parameters. The v3
machinery becomes the ablation story (what each simplification breaks).


## Implementation reference — as built (moved from `src/tracking/AGENTS.md`, 2026-09)

`JointConfig.v4`, default off. Everything in v3 (marginal and H-aware readouts included) is a
correction bolted onto a measure whose noise model does not contain the comb; v4 puts the comb
IN the model — one Gaussian process whose power spectral density is a smooth floor plus a comb of
Lorentzians riding the trajectories:

```
M_c(f, t) = S_c(f, t) + sum_{i,k} H_{c,i,k}(t) L_{gamma_k}(f - k r_i(t)),  gamma_k = max(0.6 k, one bin)
```

The v3 primitive inventory is `docs/vk-decompose-v3-design.md` §10; these are the v4 additions in
`tracking/joint_decompose.py`. One switch selects them, because the four parts only work together,
and off is the v3 arm call for call.

| Primitive | Purpose |
|---|---|
| `fit_floor_powers(audio, sr, r_audio, k_hi, ...) -> (SmoothPSD, HPowers)` | **F1**: `S` and the line powers `H`, fitted JOINTLY with NO mask on the ORIGINAL signal, per (microphone, time block) |
| `HPowers` | the fitted powers: `(C, B, L)` peak power spectral density on `masked_smooth_psd`'s own units, plus the line table. `.pooled()` (arithmetic mean over microphones), `.block_of(t)` |
| `floor_lambda(b_f_hz, sr, n_fft)` / `floor_penalty(psd, b_f_hz)` | the floor's smoothness weight from a LENGTH SCALE in hertz, and the penalty term itself |
| `whittle_floor_objective(p, hump, g, lam)` | the F1 cost of one cell — what GUARDS every step and chooses between the starts |
| `v4_rho2_gain(r_audio, k_hi, cfg, ...)` | **F2a**: the band law `max(b0, 0.6 k)` Hz with NO spacing cap, in the solver's own currency |
| `v4_ridge(psd, hp, k, rotor, r_env, t_env, weight, c0=)` | **F2b**: the amplitude prior `beta = c0 S / H`, fed to `vk_envelopes(ridge=)` |
| `map_objective(..., v4_powers=, v4_carrier=)` | **J_v4**: `sum [P/M + log M]` + the two phase priors + the floor penalty. No envelope term, no separate rent |
| `joint_objective(state, audio)` | the same off a state — and under v4 the `audio` is REQUIRED |

Five things a caller must know, and the first two are the ones that will bite:

- **`J_v4` scores the ORIGINAL signal, not the residual.** The line processes are integrated out
  rather than conditioned on, so the thing the model describes is the audio; scoring the residual
  would count the comb twice, once by subtracting it and once by modelling it. `joint_objective`
  raises rather than guess. Its column is therefore NOT comparable with `total` — different model,
  different signal — which is why `scripts/joint_rescore.py --v4` ranks on it alone.
- **The `(S, H)` alternation is BISTABLE, and the two starts are a guard and not a knob.** Where
  the comb blankets a band, "floor on the blanket with `H = 0`" and "lines claim the blanket" are
  both honest stationary points; from the masked warm start the first H-step finds no excess and
  nothing ever moves, so the v3 failure survives INSIDE the v4 fit. The objective ranks them
  correctly, so the fit screens `FLOOR_START_DB = (0, -12)` dB on `FLOOR_SCREEN_ROUNDS` and
  refines the winner. Measured on the dense fixture the two differ by 13 dB of fitted floor.
- **The band law changes `rho^2` and NOTHING else**, so the coupling partition, `group_plan` and
  the banded memory are identical to v3's — measured on the smoke window, 0.149 GB either way.
  What moves is wall time: 1.42 s to 2.53 s there, the difference being the three `(S, H)` fits.
- **Regime 3 does not run.** The comb channel already carries the line flanks, so the
  decomposition is two channels and a subtraction, and `joint_solve_window(stochastic=True)` is
  REFUSED under v4 rather than silently ignored.
- **`H` is a first-class product**, not a diagnostic: it is the generator's amplitude targets by
  construction, and `scripts/vk_decompose.py --v4` writes it into the unit `.npz`
  (`h_rotor` / `h_k` / `h_t` / `h_lines` / `h_half` / `h_power`).
- **A v4 group that will not factorize FAILS — it never falls into `splu`.** The automatic
  fallback is right for a v2-sized group and is the OOM bomb for a v4 one (SuperLU's fill-in on
  300+ tracks does not fit in memory; the measured field failure was a spin-up window that took
  its worker and then the pool with it). Under `ridge` the banded path keeps its two `diag_scale`
  repair retries and then raises a `MemoryError` naming the group, the harmonics and the knob, so
  `gridrun` writes one `.err` and the other units keep running. An EXPLICIT `solver="splu"` is a
  different statement and is still honoured.

**Conditioning, and the limit of the fix.** With the bands uncapped, two rotors whose lines nearly
coincide have passbands that nearly coincide, so the difference direction of that pair has almost
no data curvature; `rho^2` is small because the band is wide, and `beta = c0 S / H` is small for
exactly the STRONG lines — so the loudest comb in a window is the one that breaks the
factorization. `RIDGE_FLOOR_FRAC = 0.03` holds the prior above 3 % of the group's own mean data
curvature, and the constant is squeezed from both sides: below it nothing factorizes on the
four-rotor spin-up fixture (120 tracks, rotors fanning 2 rev/s and crossing), and above it the
estimator starts to move — at 0.3 the Wiener calibration breaks (0.72-0.76, outside ±20 %) and a
strong line keeps 0.61 of its power. At the chosen value the calibration is unmoved to three
decimals and a strong line keeps 0.945 instead of 0.980.

The deficiency is NOT float rounding — it is the decimated cross term's own approximation error,
which is percent-relative — so the floor has a reach and past it there is nothing to do. Four
rotors within ~1 rev/s would need 0.1, costing a strong line 17 %; those windows are genuinely
unidentifiable at these bands (four combs 0.3 Hz apart at `k` 1 are not four combs to a 3-second
window). `tests/tracking/test_v4_conditioning.py` pins both halves: the floor factorizes a group
that fails without it, and past its reach the failure is clean and names its own mechanism.

**Past that reach, the band law steps aside — `JointConfig.v4_band_law`.** DREGON is a twin rig
(its pairs sit 0.43 and 0.81 rev/s apart), so at `k_hi` 83 six of its seven windows are in the
unidentifiable regime and fail cleanly. `scripts/vk_decompose.py` catches that one exception per
window and retries ONCE with `v4_band_law=False`: the envelope bands come from the `--bw-schedule`
(the v3 spacing-capped law) and **everything else stays v4** — the joint `(S, H)` fit, the
amplitude prior with its floor, `J_v4`. That is not a degradation of what the model estimates.
The amplitude targets ARE `(S, H)`, they come from the F1 fit, and that fit never looks at a band;
the uncapped bands refine the WAVEFORM channel, which is only identifiable where the rotor spreads
allow it. The row carries `v4_band_fallback: true` and the report's `v4` block carries
`n_band_fallback` / `band_law_mixed`.

Mixed sets stitch: the stitch's only compatibility check is the HARMONIC SET, and `k_hi` comes
from the recording's reference trajectory rather than from any window's bands, so windows solved
under different band laws stitch exactly as windows that were not. `bw_track` is carried from the
first window and never enters the arithmetic — which is why `band_law_mixed` has to be reported,
because `bw_track_hz_by_band` is then one window's law and not the recording's.

**The fallback derives its own bands — `v4_fallback_rho2_gain`, not `track_rho2_gain`.** This is
the one place where reusing the v2 seam is wrong, and it cost a second field failure (FLY124,
`k_hi` 83: 13 of 21 units died in the objective READOUT with `bw_hz=100.0 exceeds fs_env=100.0`).
The chain, all four links measured:

1. `bw_rps` is 1.0, so the solver's own band for track `k` is `min(k, 0.9 fs_env)` capped by the
   group's minimum line separation. On DREGON everything is in one dense group and the cap holds
   every track near 1 Hz; on a rig whose rotors are tens of hertz apart an isolated high-`k` track
   is never capped, and its band is 60 to 90 Hz on a **100 Hz** envelope grid.
2. `schedule_bandwidths` FLOORS at that band — "a schedule never NARROWS a track below v1" is
   right for v2 and wrong here — so a schedule asking for 3 Hz achieves 30 to 90.
3. `bandwidth_neutral` multiplies `rho^2` by the track's own `mean(u^2)`, which is BELOW one for a
   loud track (the weight is normalized over all cells, not per track, and the clamp bottoms at
   `10^-1.5`), so it WIDENS those bands further.
4. `_tuma_bw` then saturates at `fs_env` exactly, and `_tuma_rho` refuses that value.

So the fallback asks for the same schedule with the floor at the smallest numerically usable band
instead of the solver's, and `JointState` keeps `bw_schedule` / `rho_scale` so the retry can
re-derive rather than reuse a compiled gain. `solve_block` additionally holds the v4 arm's
achieved band inside `0.9 fs_env` after the neutral gain, and `map_objective` reads a saturated
band at the widest CONVERTIBLE one — an observer must not be the thing that raises. Measured on
the FLY fixture at a schedule absmax of 3 Hz: the solver's clamp and the v2 gain both give 30.0 Hz,
the fallback law gives 3.0, and with no schedule at all it gives 1.0.

The two calibrated constants, both frozen with their measurement in the test that made it:

- `FLOOR_LENGTH_HZ = 600` — the floor's smoothness length in hertz. On the dense fixture (one
  comb whose density `gamma / Delta` runs 0.06 to 0.48) the fitted floor lands at
  0.36 / 0.53 / 0.70 / 0.46 dB rms in the four bands, against 0.37 / 0.63 / 1.03 / 0.73 at 400 Hz
  (the v3 cepstral lift's own scale) and 0.33 / 0.61 / 0.49 / 0.31 at 800 Hz.
- `V4_RIDGE_C0 = 1` — and it is 1 for a derivable reason, not a fitted one: the band is `0.6 k` Hz
  and the line's own half width is the same `0.6 k`, so line and noise contribute in proportion to
  the SAME noise-equivalent bandwidth and the ratio of the two powers the band admits is exactly
  the ratio of the two densities, `S / H`. Measured against the Wiener target over three seeds:
  0.97 / 1.02 / 1.01 at `k` 5 / 20 / 60 for a line 10 dB over the floor.

Acceptance (gate 1 of the design, plus the carve):
`tests/tracking/test_v4_floor.py` reads the fitted floor at 0.33 / 0.45 / 0.58 / 0.40 dB rms
against the masked fit's 2.00 / 4.83 / 8.12 / 10.46 on the same fixture — and against 0.33 / 2.05
/ 6.69 / 5.75 for the v4 fit given only the warm start, which is the bistability measured.
`tests/tracking/test_v4_carve.py` gives one rotor a band it owns with lines on the low harmonics
only: the line-free tracks take 3.9 % of what the same wide bands take with the prior switched
off, while the owned harmonics keep 84 %. `tests/tracking/test_v4_ridge.py` holds the `c0`
calibration; `tests/tracking/test_v4_objective.py` holds `J_v4`'s discrimination.

The bar on the floor gate is 0.9 dB rms and not the design's 0.5, for a measured reason that is
in the fit's favour: the Hann main lobe smears each line's skirts into the bins the floor is read
from, so a CORRECT fit reads a few tenths of a decibel high on this grid whatever it does. The
residual error is that bias and almost nothing else.

