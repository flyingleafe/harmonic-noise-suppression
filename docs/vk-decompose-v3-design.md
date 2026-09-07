# The joint decomposition (v3) — design

**Module**: `src/tracking/joint_decompose.py` · **Driver**: `scripts/vk_decompose.py --joint` ·
**Continues**: `docs/experiments/vk-decomposition.md` (v1, v2)

The v2 decomposition (`tracking.decompose`) splits a recording into per-(rotor, harmonic,
microphone) Vold-Kalman envelopes plus a per-microphone residual. It does this under two
conditions that it never states, and both conditions are incorrect. First, one fixed envelope
band holds every timing deviation. A shaft wanders by about 0.6 rev/s, so the identity
`k (phi + theta)` makes harmonic `k` about `0.6 k` Hz wide. That is more than the 3 Hz envelope
band from about `k` 5 up, and the flanks of every line become residual by construction. Second,
an unweighted least squares takes the floor to be white. Drone noise is strongly colored, so the
unweighted fit is tolerant of comb structure exactly where the floor is loud. v3 removes both
conditions.

## 1. The model

```
y_c(t) = sum_{r,k} Re[ g_{r,k,c}(t) e^{j(k phi_r(t) + k theta_r(t) + psi_{r,k}(t))} ] + n_c(t)
```

**`g_{r,k,c}`** is the residual envelope. Every timing deviation moves into the two phase terms,
so `g` only needs AMPLITUDE bandwidth. That is what lets the tuned v2 bandwidth law stay
unchanged.

**`theta_r`** is a slow coherent shaft correction, in radians, common to every harmonic of one
rotor. It has two parts. A rig-common part comes from every rotor's tracks. A small per-rotor
part comes from that rotor's own tracks. This hierarchy is the measured structure of the
deviation.

**`psi_{r,k}`** is a slow per-track phase correction. It holds what one harmonic does that
`k theta_r` cannot explain. Its allowed band increases with `k` (`bw_psi_hz`), because a high
harmonic wanders more.

**`n_c`** is colored noise with a smooth log spectrum `S_c(f, t)`. Smooth is the important word:
because `S` is smooth and the comb lines are sparse, `S` stays identifiable from BETWEEN the
lines.

There is NO per-microphone arrival term in this model. It was measured, and it is not there.
Every microphone sees the same phase deviation up to one constant of its own, and
`_combine_channels` removes that constant before the phase split.

## 2. The three blocks

`joint_solve_window` alternates three blocks. Each block is linear-Gaussian given the other two,
so the alternation is block-coordinate descent on one MAP objective. One iteration is about one
v2 solve, because the other two blocks are very cheap.

Each block is a function `JointState -> JointState` (`solve_block`, `split_block`,
`floor_block`) with a Frame stage beside it (`vk_solve_stage`, `phase_split_stage`,
`floor_stage`), and the alternation is their composition:

```
floor -> (iters - 1) x (solve -> phase split -> floor) -> solve
```

`joint_solve_window` IS that composition, so there is no second path to keep in agreement. The
state the blocks pass along is `JointState` — the carrier, `theta`, `psi`, the floor model and
the last solve's products — and it travels in the frame as the seam `meta["joint"]`. The
catalog of every primitive is `src/tracking/AGENTS.md` § 3.1.

### Block A — the whitened VK solve

The coherent phases fold into the CARRIER, which is an exact reparametrization: `k (phi +
theta)` keeps the rotor-major power recursion, so a corrected carrier needs one array add only.
The three hooks of `tracking.vk_tracking.vk_envelopes` are the whole seam — `phase_offset`
(theta at audio rate), `env_rotation` (psi on the envelope grid) and `data_weight` (the
whitening). Every hook is `None` on the v2 path, and the arithmetic is then bitwise the v2
arithmetic.

The Whittle likelihood of colored noise weights each frequency by `1 / S`. Because `S` is smooth
and one line is narrow, that whole weighting becomes the ONE scalar `1 / S(k r(t), t)` per track
and per envelope frame (`whiten_weights`). Thus the banded structure of the solver stays
unchanged, and the whitening adds no work to the solve. `_whiten_weights` splits the weight into
its diagonal form `u^2` and its cross form `u_a u_b`, because a cross block is a product of two
DIFFERENT tracks' weighted bases.

`bandwidth_neutral` is the correction that makes the whitening safe. Without it a track whose
floor is 15 dB loud gets its data term scaled down and its curvature prior unchanged, so its
effective band becomes narrow by the same factor and its envelope becomes too smooth. The fix
puts each track's mean weight into `rho^2` as well, so the ACHIEVED bandwidth stays at the tuned
v2 value. The whitening then does only what it is for — the relative trust between coupled
tracks and across time.

### Block B — the phase split

`split_phases` reads the solved envelope bank of the CURRENT carrier, so its angle is the phase
error that is left. By the model that angle is `k theta_r + psi_{r,k}` plus noise. Every
harmonic measures `theta` with precision `k^2` times its own, so the shaft estimate is the
`k`-weighted mean of `arg x / k` over the trustable tracks. That estimate is far better
determined than any one harmonic or than the telemetry.

The smoother is `wh_smooth`, a Whittaker-Henderson smoother. Its transfer is
`1 / (1 + lam (2 sin(w/2))^4)`, which IS the VK-2 transfer. Thus `lam = rho^2`, and the
bandwidth relation is the solver's own (`_tuma_rho`). There is one calibration in the package,
not two.

The data weight of that smoother is the solver's own `edge_taper`. Block A fades its data term
at both window ends, so the envelopes there are the prior's extrapolation and not a
measurement. With the same taper as the weight, the shaft estimate extrapolates over that span
instead of fitting the transient there. § 6.1 gives the measurement that made this necessary.

The fit has two levels. The rig-common `theta_rig` comes first, from every rotor's trustable
tracks. Then each rotor's own tracks give a small per-rotor increment on top of it. What is left
per track becomes `psi`, smoothed with a wider band at higher `k`.

Two gates select the trustable set. The annealing cap `k_trust` is the first. The second is a
CONCENTRATION gate: `|mean exp(j d arg x)|` more than `conc_min` — a scale-free signal-to-noise
proxy that reads about 0 for a noise-dominated envelope and near 1 for a locked one. A track
whose phase increment reaches pi at any frame leaves the set whatever its concentration,
because its unwrap is a guess.

### Block C — the masked smooth floor

`masked_smooth_psd` computes a Welch spectrum of the current residual with every predicted comb
line masked out, per short frame, so the mask stays on a moving line. It then fits a smooth log
spectrum through the gaps — a moving median across frequency, then a cepstral lift to
`psd_n_cep` coefficients.

The mask must be several linewidths wide, and it must also be capped. A line whose skirts are
25 dB more than the floor still lifts the fit two linewidths out, so one linewidth is not
sufficient. The cap `mask_frac_of_rate` keeps the rule usable at `k` 80, where
`3 * 0.6 * k` alone is wider than the distance between one rotor's adjacent lines. Too wide is
as bad as too narrow, because the fit must then bridge gaps instead of reading the floor.

The mask is what makes the estimate honest. An unmasked floor fit increases under every line,
and a floor that increases under the lines tells block A not to fit them. That failure mode
hides itself, which is why the mask is not optional.

## 3. The annealing ladder

The ladder `k_trust` starts at 3, and this is the important correction to the original plan.

The limit on which harmonics can measure the shaft is the ENVELOPE BAND, not the phase unwrap. A
shaft that wanders by `sigma_r` rev/s makes harmonic `k` a frequency modulation of bandwidth
about `k sigma_r` Hz. A band of `B` Hz distorts that phase once `k sigma_r` is more than
`B / 2`. At `sigma_r` 0.6 and `B` 3 that is `k` 2.5. Thus the ladder starts at 3 and not at 10.

Each fold decreases the residual wander everywhere, which brings higher harmonics under the
ceiling, so the next rung can be far higher. The shipped ladder is `(3, 12, 80)`. `psi` starts
at iteration `psi_from_iter` (2 by default), after `theta` takes the coherent part.

## 4. The instruments

A verdict is only worth what its instrument is worth, so both instruments live in the same
module.

**`order_cell_profile`** is THE probe. It takes the power spectrogram, averaged over
microphones, and re-expresses each frame's frequency axis in ORDERS of one reference rotor
(frequency over that rotor's instantaneous rate). The comb then stops drifting and its teeth sit
on the integers. The result is averaged onto a fixed order grid, and every unit cell
`[m - 0.5, m + 0.5)` of a harmonic band is folded into ONE profile (`cell_profile`). Each cell
is first divided by the LOCAL trend of the order profile (`_order_trend`, § 4.1), so the fold
measures modulation and not the spectral tilt across the cell. Every rotor is the reference in
turn, and the band reading is the mean over them. `exclude_others` removes every bin near ANY
other rotor's line before the order mapping, which is what makes the reading meaningful on a
multi-rotor rig.

Two readings come back:

- `depth_db` — the folded peak over the folded median. It is a RATIO, so it can increase while
  the residual decreases toward the broadband floor. On a four-rotor rig the other rotors' lines
  also put a floor under it.
- `excess_db` — ten times the log of the summed ABSOLUTE excess `peak - trend` over the band's
  cells, before the division by the trend. It is in power units of the input, so it is
  comparable ACROSS signals. The original audio's `excess_db` minus the residual's `excess_db`
  is how many decibels of comb the decomposition removed, and it does not move when the floor
  moves.

**Read `excess_db` for the verdict.** `depth_db` is the v2 instrument of record and it stays in
the report, but at mid `k` it gives almost no difference between two signals (see § 9).

**`whitened_flatness`** is the second instrument: the spectral flatness of `|N(f)|^2 / S(f)` per
microphone, beside the flatness of `|N|^2` itself. A correct floor model leaves a flat whitened
residual, so the pair (raw, whitened) is the reading.

**Standing policy.** Do a check of an instrument before you accept its verdict. The order cell
is the THIRD instrument of this campaign to fail in the same direction, after the narrow slot
contrast and the rendered-comb metric. Never use a narrow on-order against half-order slot
contrast as a verdict. That instrument reads about zero for a comb whose linewidth is more than
the slot, or whose peak sits outside it. It has already given one reported verdict, and that
verdict was then removed. The two-ends test of § 4.1 is the cheap check for the order cell.

### 4.1 The in-cell trend removal

One unit cell spans a whole order, which is 70 to 85 Hz of frequency. At low harmonics the
broadband floor decreases much across that span. `cell_profile` divided each cell by its own
SCALAR median, which removes the level but not the slope. Thus a cell of pure smooth floor folds
into a monotone ramp, and `argmax` then gives the low edge — which is the half-integer position.
The first production runs reported a "half-order comb" at −0.4962 orders on every rotor of both
rigs. That comb does not exist.

Two tests found the fault, and both are cheap:

- The two ends of a folded cell are the SAME physical half-integer position (order `m−0.5` and
  order `(m+1)−0.5`), so a line must show at both ends. The DREGON residual read **+1.6 dB** at
  the low end and **−0.4 dB** at the high end, in almost every cell.
- The cell profile is monotone from its low edge, through the integer, to its high edge. That is
  a ramp and not a peak.

The fix is `_order_trend`. It divides the order profile by its running median over
`detrend_orders` of order (1.0 by default) before the fold. A smooth tilt goes to unity. A line
much narrower than one order passes through with no change. `cell_profile` takes the trend as a
`trend=` argument and computes `excess_db` against it, so `excess_db` stays absolute power.

A second reading of the SAME production residuals, before and after:

| residual, band | before (cell median) | after (running median) |
|---|---|---|
| DREGON k1-9 | 1.427 dB at −0.4962 | **0.305 dB at −0.030** |
| DREGON k10-24 | 0.333 dB | 0.423 dB |
| DREGON k25-49 | 0.196 dB | 0.104 dB |
| DREGON k50-80 | 0.159 dB | 0.073 dB |
| FLY124 k1-9 | — | **1.029 dB at +0.062** |

**What stays after the trend removal.** On FLY124 two cells stay, and only on the twin pair
(rotors 1 and 3, cells k2 and k3). There cell k2's high end and cell k3's low end are the same
physical order 2.5, and they agree at +5.90 and +5.50 dB on rotor 1. Thus that one IS a line.
The absolute-frequency reading names it: rotor 1's order-2.5 energy sits at 168.89 Hz and rotor
0's rate is 84.483 rev/s, so the line is 2.0000 × r_0, off by 0.0009 orders. It is another
rotor's own `k` 2 harmonic. Neither rig shows a genuine per-rotor `r/2` comb, which agrees with
the physics: the labels are the SHAFT rate, so an `r/2` line needs a mechanism with a period of
two revolutions.

## 5. The knobs

| Name | Default | What it does |
|---|---|---|
| `JointConfig.iters` | 3 | Alternation rounds. One round is about one v2 solve. |
| `JointConfig.k_trust` | `(3, 12, 80)` | The annealing ladder — one trustable harmonic cap per iteration. |
| `JointConfig.psi_from_iter` | 2 | First iteration (1 based) that estimates the per-track `psi`. |
| `JointConfig.bw_theta_hz` | 1.5 | Bandwidth of the shaft correction `theta`, in Hz. |
| `JointConfig.bw_psi_slope` | 0.6 | Slope of `bw_psi_hz(k) = min(slope * k, max)` — the measured linewidth law. |
| `JointConfig.bw_psi_max` | 8.0 | Cap of the same law. It prevents a high-`k` correction that takes in the floor. |
| `JointConfig.bw_psi_min` | 1.5 | Floor of the same law, in Hz. The law `0.6 k` allows a `k` 2 line only 1.2 Hz, which is narrower than its true incoherent linewidth, so a strong low harmonic keeps a skirt that the model cannot follow. This is the indicated correction for the cross-rotor leftover of § 4.1. |
| `JointConfig.conc_min` | 0.5 | Concentration gate on a track's phase increments. |
| `JointConfig.per_rotor_theta` | `True` | Fit the small per-rotor part of `theta` on top of the rig-common part. |
| `JointConfig.whiten` | `True` | Weight block A by `1 / sqrt(S)`. |
| `JointConfig.whiten_clamp_db` | 15.0 | Clamp on the weight, so one quiet band cannot get too much of a solve. |
| `JointConfig.bandwidth_neutral` | `True` | Put each track's mean weight into `rho^2`, so the whitening does not retune the band. |
| `JointConfig.psd_n_fft` | 4096 | Transform length of the floor fit. |
| `JointConfig.psd_blocks` | 4 | Time blocks of the per-window floor. |
| `JointConfig.psd_n_cep` | 40 | Cepstral coefficients kept — how smooth the log floor is. |
| `JointConfig.profile_n_fft` | 8192 | Transform length of the order-cell probe. |
| `JointConfig.profile_order_step` | 0.005 | Order grid step of the probe. |
| `JointConfig.profile_every_iter` | `True` | Profile every iteration's residual, not the last one only. |
| `order_cell_profile.detrend_orders` | 1.0 | Width in orders of the running median that removes the in-cell spectral tilt (§ 4.1). |
| `order_cell_profile.fold` / `cell_profile.fold` | `"mean"` | How the cells of a band combine. `"median"` is a robust variant. |
| `--joint` | off | Turn on the v3 alternation. Off IS the v2 path, call for call. |
| `--iters` | 3 | Sets `JointConfig.iters`. |
| `--k-trust` | `3,12,80` | Sets the ladder. |
| `--bw-psi` | `0.6,8,1.5` | Takes `slope,max[,min]` and sets `bw_psi_slope`, `bw_psi_max` and `bw_psi_min`. |
| `--bw-theta` | 1.5 | Sets `bw_theta_hz`. |
| `--no-whiten` | off | Runs block A on the unweighted misfit. |

`masked_smooth_psd` carries its own mask geometry: `mask_factor` 3.0, `mask_min_hz` 10.0 and
`mask_frac_of_rate` 0.45. The half-width per line is
`clip(mask_factor * LINEWIDTH_HZ_PER_K * k, mask_min_hz, mask_frac_of_rate * r(t))` Hz.

## 6. Windows and the stitch

One window holds its own shaft correction `theta`, and a phase has an additive constant of its
own in each window. Two overlapping windows would then hold one physical correction at two
origins, which is exactly the failure that the envelope stitch already guards against.

Thus the stitch moves the RATE across a window boundary, and not the phase. `theta_rate` gives
`d theta / dt / (2 pi)` in rev/s, which is gauge-free because the additive constant
differentiates away. `global_rate_correction` cross-fades the per-window rates onto one global
envelope grid. `corrected_phase` integrates that into one global corrected carrier
`r_corrected = r_labels + dr_global`. `window_extra_phase` then gives the extra rotor phase that
moves one window onto that global carrier, and each track `m` is multiplied by
`exp(j k_m e[rotor_m])` before the usual cross-fade.

The rotation is slow by construction, because the two carriers are the same trajectory up to the
blend between adjacent windows. That is REPORTED and not taken for granted:
`theta_stitch_max_rate_hz` in `report.json` measures how fast the fastest track's rotation is,
so a reader can see whether it stayed inside the 100 Hz envelope grid. At high `k` a large
disagreement would cause aliasing on that grid.

### 6.1 The window edges, and a metric that read too high

`joint.theta_stitch_max_rate_hz` read 46.08 on the first DREGON production run, against 0.003
on a single-window smoke test. The cause is the window EDGE. Two overlapping estimated windows
reproduce it locally: the maximum sits at frame 0 or at the last frame of a window, where the
cross-fade weight is 0.0025 to 0.005. That is, it sits where the window gives almost nothing to
the stitch. With the fade as the weight the same reproduction reads 0.077 Hz, against a raw
1.86 Hz.

Three changes came from that measurement:

1. `split_phases` weights the shaft smoother by the solver's own `edge_taper`, so the estimate
   extrapolates over the span where the solver faded its data term (§ 2, block B).
2. `theta_rate` holds its first and last value instead of taking `np.gradient`'s one-sided
   difference, so no frame carries a different estimator from the interior.
3. The report carries `theta_stitch_max_rate_hz` (fade weighted, the number to read) beside
   `theta_stitch_max_rate_hz_raw`.

## 7. Outputs

`--joint` adds these to the v2 output set:

- Per unit, in `raw/<uid>.npz`: `theta` (the total shaft correction, radians), `dr` (the same
  correction as a rate, rev/s), `psi` (the total per-track correction), and the floor model as
  `psd_freq`, `psd_t` and `psd_log_s`.
- Per recording, `joint.npz`: `dr_global`, `r_corrected` and `r_labels` on the envelope grid.
- In `report.json`: a `joint` section (the stitch rate statistics), `order_cell` with both a
  `residual` and an `original` band table, and `flatness`.
- Per unit JSON, under `joint.iterations`: every iteration's diagnostics — `residual_fraction`,
  `track_fraction`, `psd_masked_frac`, `flatness`, the phase-split diagnostics and the
  order-cell band table of that iteration's residual.

Two additions came from the first production runs. In `report.json` the `joint` section carries
`theta_stitch_max_rate_hz_raw` beside the fade-weighted `theta_stitch_max_rate_hz` (§ 6.1). The
`phase_model` section carries `max_abs_step_rad_by_band` and `max_step_rad_by_band_worst`, which
say WHERE the phase unwrap saturates. A global maximum that reaches pi says only that SOME track
is ambiguous. A per-band maximum says whether the ambiguous tracks are the weak high harmonics,
which carry almost no energy and are harmless, or the low ones the shaft estimate is built on,
which are not.

**A bug the same pass found.** `r2_ref_mic` read negative on almost every production window while
the stitched ledger was healthy. That per-window check rebuilt the plain label carrier instead of
reading the solver's own `env.phase`, which on the joint path carries the shaft correction, so it
scored the bank against a carrier it was never fitted to. The stitch itself always used the right
phase.

## 8. Measured results on the synthetic fixture

The fixture is 20 s, 16 kHz, 4 rotors, 3 microphones, `k` up to 20. Its shaft rate wanders by
0.5 rev/s at a bandwidth of 0.5 Hz (a rig-common part plus a per-rotor fifth). Per-track phase
noise is 0.02·k radians rms at a bandwidth of min(0.6·k, 8) Hz. The colored floor is smooth and
34 dB less than the comb.

| arm | residual fraction | k1-9 depth / excess dB | k10-24 depth / excess dB |
|---|---|---|---|
| original audio | 1.0000 | 28.17 / 62.03 | 2.54 / 35.93 |
| v2 (flat carrier) | 0.0551 | 6.09 / 43.65 | 2.02 / 34.18 |
| **v3 (`--joint`, 3 rounds)** | **0.0025** | **1.21 / 24.86** | **1.38 / 20.24** |
| oracle (the true shaft folded in) | 0.0024 | 0.89 / 22.90 | 1.62 / 19.85 |

The reading: v2 removes 18.4 dB of comb excess at k1-9 and only 1.75 dB at k10-24. v3 removes
37.2 dB and 15.7 dB, and it is within 2 dB of the oracle in both bands. The correlation of the
recovered shaft phase against the truth is 1.000 (0.999 to four places, error 0.15 rad rms
against a true 3.65 rad rms). The fitted log floor is within 0.5 to 1.0 dB rms of the truth away
from the lines. The whitened residual flatness increases from 0.010 to 0.36.

Three design decisions were measured on the same fixture:

1. **The ladder must start low.** A ladder that starts at `k` 6 recovers only 43 % of the true
   shaft phase in three rounds. A ladder that starts at 3 recovers all of it.
2. **The whitening needs the bandwidth-neutral correction.** Without it the residual comb at
   k1-9 is 12.6 dB, against 4.3 dB unwhitened. A down-weighted track keeps its curvature prior,
   so its band becomes narrow.
3. **The floor mask must be about three linewidths wide.** The log floor error against truth is
   3.5 dB rms at `(1.5, 3 Hz)`, 0.6 dB at `(3, 10 Hz)`, and 6.5 dB again at `(4, 30 Hz)`.

## 9. Open questions and limits

- **`depth_db` is weak at mid `k` on a four-rotor rig.** The original audio reads 2.54 dB at
  k10-24 and a near-perfect decomposition reads 1.62 dB, so the ratio reading gives almost no
  difference there. `excess_db` gives a clear difference, which is why it is the verdict.
- **The whitening weight is pooled over microphones.** `SmoothPSD.pooled` takes the geometric
  mean, so the per-microphone floor LEVEL does not reach the solve. Only the shape does. A
  per-microphone weight makes the banded system channel dependent, and it needs one banded
  factorization for each microphone. The per-microphone surface is still estimated and still
  reported, because a downstream noise model wants it.
- **Foreign non-comb tones still cause damage to the floor fit locally.** The v2 record found
  quasi-stationary structural or aerodynamic resonances at non-integer orders (for example the
  489 Hz line of `free-flight_nosource_room1`). The mask does not cover them, and a smooth log
  spectrum cannot represent them. `report.json -> residual_tones` measures them, and nothing
  removes them.
- **The per-window floor is fitted in a small number of time blocks** (`psd_blocks` 4 by
  default). The fit does not follow a floor that moves faster than one block of a window.


## 10. Implementation reference — the primitive inventory (moved from `src/tracking/AGENTS.md`, 2026-09)

This is the catalog of `tracking/decompose.py` + `tracking/joint_decompose.py` + their stages in
`tracking/top.py`. Every entry is importable as `tracking.<name>`. It is the one part of the
package a reader is expected to REBUILD rather than call, so it is kept beside the model of §1.
The v4 arm's inventory is in `docs/v4-unified-model-design.md` § "Implementation reference".


**The state.** `JointState` is what the alternation accumulates and the one seam the stages pass
along, `meta["joint"]`: the carrier, `theta`, `psi`, the floor model `psd`, the last solve's
`env` / `x_eff` / `residual` / `track_energy` / `n_solves`, and — once regime 3 has run — the
`stochastic` channel beside the residual. It is frozen — every block returns a
NEW state. The carrier is HELD and never re-derived, because the alternation is conditioned on
one carrier for the whole window and `theta` is a correction on top of it.

### Setting up a solve

| Primitive | Purpose |
|---|---|
| `solve_config(k_max, *, sr, mics, bw_rps=1.0, f_max=6000.0) -> FVKConfig` | THE measurement geometry — one construction, so every solve agrees |
| `k_cap(cfg, reference) -> int` | the harmonic set, from the RECORDING's reference trajectory (never the window's) |
| `solve_audio(audio, cfg, mics=None) -> (C, T)` | the one channel-selection rule, float64 and contiguous |
| `to_audio_grid(r, frame_times, n_t, sr) -> (R, T)` | the trajectory on the audio grid — THE carrier every solve is built from |
| `shaft_phase(r_audio, sr) -> (R, T)` | `2 pi cumsum(r) / sr`, the fundamental phase `phi` |
| `BandwidthSchedule(bw0_hz, slope_hz_per_k, cap_frac_of_sep, bw_abs_max)` | the v2 linewidth-matched band law, with its CLI spelling (`.parse` / `.text`) |
| `base_bandwidths(r_audio, k_hi, cfg) -> (M,)` | the band the solver would use with no schedule — the reference the gain is taken against |
| `line_separations(r_audio, rotor, k) -> (M,)` | Hz from each track's line to the nearest other line |
| `track_rho2_gain(r_audio, k_hi, cfg, sched, rho_scale=1.0) -> (M,) \| None` | THE bandwidth law in the solver's own currency; both solve paths read it |
| `group_plan(r_audio, k_hi, cfg) -> dict` | THE memory model — coupling is transitive, `~1e-4 k_hi^2 window_s` GB per worker. Read it before sizing a job |

### Block A — the whitened solve

| Primitive | Purpose |
|---|---|
| `vk_envelopes(audio, r, vk, *, k_hi=, rho2_gain=, phase_offset=, env_rotation=, data_weight=)` | THE solver. The last three arguments are the whole joint seam, and all three `None` is the v2 arithmetic bit for bit |
| `whiten_weights(psd, k, rotor, r_env, t_env, *, clamp_db=15.0) -> (M, J)` | `1 / sqrt(S(k r(t), t))` — the Whittle weighting, collapsed to one scalar per track and frame |
| `solve_block(state, audio) -> JointState` | block A: weights → hooks → solve → `x_eff = x e^{j psi}` → reconstruct → residual |
| `vk_solve_stage(cfg=None, *, profile=None)` | its Frame adapter |
| `solve_window(audio, r_audio, cfg, *, k_hi, mics=, rho_scale=, bw_schedule=) -> Envelopes` | the v2 solve — block A with no corrections and no whitening |
| `reconstruct(x, k, rotor, phase, stride) -> (recon, track_energy)` | the bank back to audio against a GLOBAL phase, chunked in time |

### Block B — the phase split

| Primitive | Purpose |
|---|---|
| `split_phases(x, k, rotor, valid, fs_env, *, k_trust, ...) -> PhaseSplit` | the `k`-weighted shaft estimate (rig-common, then per rotor), then a per-track `psi` |
| `split_block(state) -> (JointState, PhaseSplit)` | block B: the split folded into the accumulated corrections |
| `phase_split_stage(cfg=None)` | its Frame adapter |
| `wh_smooth(y, lam, weight=None) -> ndarray` | THE Whittaker-Henderson smoother — a one-dimensional VK, one banded solve |
| `wh_lambda(bw_hz, fs) -> float` | its weight from a bandwidth, through the solver's OWN Tuma relation |
| `bw_psi_hz(k, slope=0.6, cap=8.0, floor=1.5)` | the per-track correction band, `clip(slope k, floor, cap)` |
| `theta_rate(theta, fs_env) -> ndarray` | `d theta / dt / 2 pi` in rev/s — the GAUGE-FREE form, the only one that crosses a window |
| `upsample_env(vals, n_out, stride)` | the envelope grid back to audio rate, tail held |

### Block C — the floor

| Primitive | Purpose |
|---|---|
| `masked_smooth_psd(audio, sr, r_audio, k_hi, ...) -> SmoothPSD` | Welch with every predicted line masked per frame, then a moving median and a cepstral lift |
| `floor_block(state, audio) -> JointState` | block C: the floor of what is left — the audio before the first solve, the residual after |
| `floor_stage(cfg=None)` | its Frame adapter |
| `SmoothPSD.pooled() -> (B, F)` | the geometric mean over microphones — what the whitening weight reads |
| `stft_power(audio, starts, n_fft, frames_per_chunk=64)` | THE framed Hann power spectrogram of this module, shared with the order-cell probe |
| `frame_starts(n_t, n_fft, hop)` | its frame grid |

### Regime 3 — the stochastic comb channel

Regimes 1 and 2 are the coherent envelope at the annotated carrier and at the CORRECTED one.
Both need a band narrow against the local line spacing, because two coherent envelopes whose
passbands overlap are not identifiable — a cluster run at a cap of 1.5x the line separation went
singular (SuperLU "Not enough memory to perform factorization", `r2 = -1`). Identifiability caps
a coherent band at about 0.4x the local spacing, and above about `k` 10 the measured linewidth
`0.6 k` Hz is wider than that, so the flanks of every line are comb-locked energy that no
coherent envelope can carry. Regime 3 carries it, as a POWER split with no phase model.

| Primitive | Purpose |
|---|---|
| `stochastic_split(residual, sr, r_audio, k_hi, *, psd=None, n_fft=4096, ...) -> StochasticSplit` | THE split: a PER-BIN amplitude gain `a = clip(sqrt(S / P~), 0, 1)` inside the UNION of the comb search regions, `a = 1` outside. Broadband `= a Y`, stochastic `= (1 - a) Y` |
| `comb_lines(rate, k_hi) -> (lines Hz, k)` | one frame's whole comb, `k` beside it because the band law is written in `k` |
| `line_half_widths(lines, k, *, slope_hz_per_k=0.6, min_half_hz=0)` | the COHERENT law: `min(0.6 k, local spacing)`, floored at one bin — the spacing cap is what stops one band reaching over its neighbour |
| `stochastic_half_widths(k, *, width_factor=2.0, min_half_hz=0)` | regime 3's OWN law: `2 x 0.6 k` Hz, NO spacing cap |
| `stochastic_block(state) -> (JointState, StochasticSplit)` | the block, on the state the last block-A solve left |
| `stochastic_stage(cfg=None)` | its Frame adapter (`tracking/top.py`) |
| `_wola_plan(n_t, n_fft, hop)` | the padding and frame grid that make the overlap-add an EXACT identity at any window and any hop |

Six things a caller must know:

- **The gain is per BIN and it is an AMPLITUDE.** Both halves are the fix for a measured failure
  of the flat per-band Wiener gain this replaced (full-scale DREGON, `results/vk_decompose_v3c`).
  A power gain `S / P` is the conditional mean, which is not a typical floor realization: its
  power `S^2 / P` is DENTED below the floor at every strong line, and the acceptance gate — the
  order-cell excess of the BROADBAND channel — cannot tell a dent from a line (k1-9 went UP,
  4.3 % -> 7.8 % retained, the profile peak moving to +/- 0.5 orders). `sqrt(S / P~)` leaves the
  broadband channel at power `S` in expectation instead. And ONE gain over a union that spans many
  lines scales the region uniformly, so the comb PATTERN survives at reduced amplitude (depth
  0.386 -> 0.380 dB at k10-24); the smoothed periodogram carries the line SHAPE, so a per-bin gain
  concentrates the removal at the line cores with no line model at all.
- **The floor is a STEP in time, and that is what the seams are.** `S` is one spectrum per block
  (`~4 s`), so the gain steps at every block boundary — and because much of the comb band sits
  within ~1 dB of the floor, the clip at `a = 1` toggles whole bands between "nothing taken" and
  "something taken" across one boundary. Measured: the rectangular on/off patches of the FLY124
  demo spectrograms have vertical edges at 59.5 s and 63.4 s, which are boundaries 15 and 16 of
  that run's 3.96 s block grid, exactly. `floor_time_interp=True`
  (`JointConfig.stochastic_floor_interp`, `scripts/vk_decompose.py --stochastic-floor-interp`)
  reads `log S` per FRAME instead, linearly interpolated between the block centers
  `SmoothPSD.t_block`. It is OFF by default and the default path is bitwise unchanged — the block
  floor is what every published number was produced with.
- **The two smoothing widths are fixed, and the number that fixes them is chi-square variance.**
  `P_SMOOTH_FRAMES = 5`, `P_SMOOTH_BINS = 3`: one periodogram bin has 100 % relative standard
  deviation, 15 averaged bins bring the power estimate to ~26 % and the amplitude gain, its square
  root, to ~13 %. They are NOT knobs. Measured alternative: dropping the frequency boxcar
  (`bins = 1`) moves DREGON's retained numbers to 6.75 / 8.88 / 15.65 / 7.35 % — better in three
  bands, worse in the one (k1-9) the estimator is weakest in — so the shipped pair stands.
- **The bands are UNIONED, and that is not cosmetic.** Two lines whose bands touch — at
  `2 x 0.6 k` Hz nearly all of them do above `k` 35 — become ONE region, so their shared energy is
  never taken twice. `n_bands_per_frame` in the diagnostics is the count after the union. The
  union now only delimits WHERE the gain may differ from one; inside it the gain is per bin, so a
  bin that sits at floor level passes through whether or not a region claims it.
- **The broadband channel is a SUBTRACTION.** `residual - stochastic`, exactly as the residual
  itself is `audio - recon`, so `coherent + stochastic + broadband = original` holds to float
  roundoff and no consumer has to trust an overlap-add. The state's `residual` is never
  rewritten; `JointState.stochastic` lands beside it.
- **`P` and `S` are both power spectral DENSITIES on `masked_smooth_psd`'s own normalization**
  (`1 / (sr * sum(w^2))`), so the ratio is scale free whatever `n_fft` the split runs at.
- **The floor is per WINDOW on the state and per RECORDING in the driver.** `stochastic_block`
  scores against the alternation's own floor; `stochastic_split(psd=None)` fits a fresh one with
  the same block C, which is what `scripts/vk_decompose.py --stochastic` does on the STITCHED
  residual, because one window's floor does not describe a minute.
- **Do NOT widen block C's comb mask to feed this.** The search regions blanket 91-94 % of
  1-4 kHz on DREGON, so it is tempting to give the floor fit a wider mask and let the cepstral lift
  bridge. Measured: the shipped mask (3.0, 0.45) leaves the between-region residual within
  +0.84 / +0.11 / +0.03 dB of `S` at 1-2 / 2-4 / 4-8 kHz and -0.04 dB above the comb, so the
  extrapolation is already sane; widening it to (5.0, 0.48) biases `S` HIGH by 1.9 dB at 1-2 kHz
  (`band_floor_share` 1.16, the region seeing no excess to remove at all) and costs the gate
  10.25 / 20.03 / 10.85 % against 9.61 / 16.98 / 8.04 %.

**What limits it now.** On the synthetic fixture the per-bin gain is exact where it is applied —
`a^2 |Y|^2` measured on the transform lands -0.14 dB from `S` — while the RESYNTHESIZED broadband
reads +3.3 dB against the same floor. The gap is the analysis-modify-synthesis error of the
weighted overlap-add, not the gain: a 20 dB deep, one-bin-wide notch has an impulse response as
long as the frame, and it wraps. More overlap does not help (hop `n_fft / 8` reads +3.25 dB), a
zero-padded analysis window is worse (+9.3 dB), and smoothing the GAIN in frequency fills the
notch it is supposed to cut (+10 dB at three bins). It matters least where the method is used:
DREGON's residual sits 0.5 to 1.7 dB over its floor inside the regions, not 20 dB.

### The window layer

| Primitive | Purpose |
|---|---|
| `frame_grid(n_t, sr, hop_s)` | the recording's own frame grid |
| `interp_rps(vals, stamps, ft)` | telemetry onto that grid, in float64 |
| `window_bounds(n_frames, window_s, hop_s, hop_frame_s)` | the window tiling, last window right-aligned, every frame covered |
| `window_span(ft, i0, i1, n_t, stride, sr, hop_frame_s)` | one window's sample range, SNAPPED to the envelope stride |
| `window_geometry(sr, window_s, hop_s, fs_env=100.0) -> (stride, ramp)` | the envelope stride and the cross-fade length |
| `fade_weights(n_win, ramp)` | the linear cross-fade, floored so a singly-covered frame still resolves |
| `stitch_bank(windows, phi, stride, ramp) -> dict` | the phase re-reference `exp(-j k Phi(a0-1))` plus the cross-fade |
| `stitch_windows(windows, phi, stride, ramp, *, r_audio=, sr=) -> dict` | THE stitch: `stitch_bank` for v2, and for v3 the rate stitch that puts windows carrying their own `theta` on ONE carrier |
| `global_rate_correction(windows, stride, a_min, a_max, ramp)` | the per-window `dr` cross-faded onto one global envelope grid |
| `corrected_phase(r_audio, dr_env, sr, stride, a_min, a_max)` | `r + dr` and its integral — the carrier the stitched bank belongs to |
| `window_extra_phase(theta_w, phi_hat, phi_tilde, a0, stride, n_env_w)` | the rotation that moves one window onto that carrier |
| `phase_reference_deviation(r_audio, phi, a0, sr)` | the stitch's assumption MEASURED, not assumed |
| `windowed(inner, *, window_s, hop_s)` | the combinator: tile, run `inner`, stitch |

### The readings, and the instruments that judge a decomposition

| Primitive | Purpose |
|---|---|
| `map_objective(residual, sr, psd, *, x, k, bw_track, theta, psi, fs_env, ..., logdet_posterior=None, h_carrier=None)` | THE converged MAP objective, term by term: `data` + `rent` (the Whittle pair on the floor block's own STFT grid) + `phase_priors` + `envelope_prior`, plus `n_cells`. A pure OBSERVER — switching it on cannot move a product. The weights are the ones the run used: `wh_lambda` for the two phase priors, and `rho^2` read back out of `Envelopes.bw_track` through the solver's own Tuma relation. `logdet_posterior` adds the MARGINAL readout, `h_carrier` the H-AWARE one (both below) |
| `prior_logdet(bw_track, n_env, fs_env, p=2)` | `log det'` of the improper envelope prior `blkdiag(rho_m^2 D2^T D2)` — the pseudo-determinant, because `D2` kills a constant and a ramp per track |
| `d2_pseudo_logdet(n_env)` | its length-only part, one banded Cholesky of the pentadiagonal `D2 D2^T` and cached — `O(n)`, not an eigendecomposition |
| `joint_objective(state)` | the same, with every argument taken off a `JointState` (the last solve's residual against the last floor) |
| `energy_ledger(audio, recon, track_energy, k)` | total / tracks / residual / CROSS TERM — the tracks are not orthogonal, and the cross term is the honest statement of that |
| `phase_model_report(...)` | per-rotor drift against `k` plus `rank_one_share` — shaft jitter against per-harmonic drift |
| `solve_report(state, audio, *, profile)` | the reading of one block-A solve: the shares, the flatness, the order cell |
| `order_cell_profile(audio, sr, r_audio, ...) -> dict` | THE probe: the spectrum re-expressed in ORDERS, folded cell by cell. Read `excess_db` (absolute comb power left, comparable across signals) BEFORE `depth_db` (a ratio, which can rise as the residual falls toward the floor) |
| `order_cell_bands(audio, sr, r_audio, **kw)` | the same table without the plot arrays — what a report carries |
| `cell_profile(profile, grid, lo, hi, order_step, ...)` | one band's fold, with the in-cell trend removal that killed a published half-order verdict |
| `whitened_flatness(residual, sr, psd)` | flatness of `|N|^2 / S` beside flatness of `|N|^2` — a correct floor leaves a flat residual |
| `residual_tones(residual, sr, r_audio, ...)` | the NON-comb tonal peaks left, with their distance to the nearest rotor order. Measurement only |
| `welch_psd(audio, sr, nperseg=4096)` | THE Welch of this module |

### The MARGINAL objective — what profiling does not charge for

`J` as shipped is PROFILED: the envelopes' best value is substituted back, which pays no rent
for their freedom. So a hypothesis whose bands cover more of the spectrum can win by ABSORPTION
alone — measured on 5 frozen windows (`results/joint_rescore/`), the profiled `J` ranks
adversarial coverage fans above the telemetry on 3 of them. `JointConfig.marginal` switches on
the exact Gaussian correction that charges for it:

```
J_marg = J + 0.5 * n_channels * (n_fft / hop) * (log det M - log det' R)
```

- `M` is the whitened banded posterior precision block A already factorized. `Envelopes.logdet`
  carries it — read off the Cholesky diagonal inside `_solve_group_banded` (and off SuperLU's
  `U` on the fallback path), so the readout costs no second factorization. It is ONE channel's
  worth, because every channel is a right-hand side against that same system.
- `R` is IMPROPER — `D2` kills a constant and a ramp, two null directions per track — so it is
  the PSEUDO-determinant `prior_logdet`: `(T - 2) sum_m log rho_m^2 + M log det'(D2^T D2)`.
- The `n_fft / hop` factor is not a knob. `data` + `rent` are summed over frames that overlap,
  so they are that many times ONE likelihood while the correction is one likelihood's worth.
  Without it the correction is out-scaled two to one on the shipped grid and the Occam property
  fails; it is reported as `marginal_redundancy` so a caller can divide it back out.

Four hypothesis-INDEPENDENT constants are dropped, so the readout is valid only for comparing
hypotheses on ONE window with the SAME cells and the SAME track count — which is exactly what
`scripts/joint_rescore.py` pins (`k_hi` comes from the telemetry, never from the candidate):
the Gaussian volume factors; the improper prior's null-space volume (two directions per track);
the real-versus-complex parameterization factor (a complex Gaussian carries 1 rather than 0.5,
which scales the correction and cannot flip its sign); and the solver's own scaling convention
(the data term enters as `w` with a right-hand side of `2 w z`) plus the `1e-8` ridge and any
`diag_scale` PD repair that live inside `M`.

The acceptance property is Occam's, and it is a test (`tests/tracking/test_marginal_objective.py`):
add a whole spurious rotor whose every line sits on pure floor, and the PROFILED objective
improves while the MARGINAL one gets worse.

### The H-AWARE data term — the stochastic comb, in the likelihood

Absorption is only half of what lets a coverage fan win. The other half is in the DATA term:
regime 3 exists because no coherent envelope can carry the `0.6 k` Hz flanks of a line, so `J`
charges EVERY hypothesis for that flank energy alike and the true trajectory has no advantage
over a fan that misses the humps entirely. Measured on the same five frozen windows, the exact
envelope marginalization (`--marginal`) moved no ranking at all. `JointConfig.h_aware` puts the
stochastic comb where it belongs, in the noise model:

```
data_h = sum_{c,f,t} [ P / (S + H) + log(S + H) - log S ] ,   total_h = total - data + data_h
H(f, frame) = max(0, P~(f, frame) - S(f))   inside the hypothesis's own comb SEARCH REGIONS
H(f, frame) = 0                             everywhere else
```

- The REGIONS are the hypothesis's only degree of freedom: per frame of the objective's own STFT
  grid, `k r_r(t)` for every `k` the pinned track set names, half widths from
  `stochastic_half_widths` (the 3.0-linewidth law, floored at one bin), unioned by `_line_mask`.
  Regime 3's law, not the coherent one — same code, so the two readouts cannot drift apart.
- `H` inside them is the PROFILED nuisance, bounded by the estimator the split already uses:
  `P~` is the measured power on the same grid, smoothed with the split's own fixed boxcar
  (`P_SMOOTH_FRAMES` x `P_SMOOTH_BINS`, edge mode nearest). The numerator `P` stays the
  UNSMOOTHED measured power — the smoothing belongs to the nuisance, not to the likelihood.
- `data_h` folds the `log(S + H) - log S` half in, so `rent` keeps its meaning, no logarithm is
  counted twice, and `total_h = total - data + data_h` is exact. At `H = 0` it IS `data`.

The asymmetry is honest and it is the mechanism: `H` is fitted from the same data it explains,
so inside a hypothesis's regions the term is nearly hypothesis independent (at floor level `H`
is zero up to the small positive bias of clipping a noisy `P~ - S`). The discrimination is where
a real hump exists and a hypothesis's regions MISS it — those cells pay `P / S + log S` in full.
A fan that opens regions on empty floor buys nothing; a trajectory whose regions sit on the humps
stops paying for them.

Acceptance is `tests/tracking/test_h_aware_objective.py`, on the regime-3 fixture at a PINNED
floor: the true rates take `data_h` to 0.19-0.23 of `data`, rates shifted by 5 rev/s (regions
disjoint from the lines, measured) leave it at 0.78-0.97, and the profiled total cannot separate
the two AT ALL. On pure floor the term charges under 2 % whatever the carrier is.


### Three things that will bite a caller

- **The annealing ladder starts at `k` 3**, and the reason is the ENVELOPE BAND, not the phase
  unwrap: harmonic `k` of a shaft wandering `sigma_r` rev/s is a modulation of bandwidth about
  `k sigma_r` Hz, and a `B` Hz band distorts its phase once `k sigma_r > B / 2`.
- **Whitening is bandwidth-neutral by default** (`JointConfig.bandwidth_neutral`). Without it a
  down-weighted track keeps its curvature prior, so its achieved band narrows by the same factor
  and its envelope is over-smoothed.
- **The floor mask must be about three linewidths wide, and capped.** Too wide is as bad as too
  narrow, because the fit then bridges gaps instead of reading the floor.


### Names the 2026-08 consolidation changed

| Old | New | Why |
|---|---|---|
| `joint_decompose.track_rho2_gain` | `decompose.track_rho2_gain` | it was the v2 construction copied; now both solve paths read the one |
| the loop body of `joint_solve_window` | `solve_block` / `split_block` / `floor_block` | the three blocks are separately callable and separately composable |
| `joint_decompose.joint_solve_window` | `top.joint_solve_window` (re-exported as `tracking.joint_solve_window`) | it is a composition of stages now, and stages live only in `top.py` |
| `scripts/vk_decompose.stitch_envelopes` (the joint half) | `joint_decompose.stitch_windows` | numerics out of the driver; the driver keeps the file I/O |
| `scripts/vk_decompose._order_cell_bands` | `joint_decompose.order_cell_bands` | it was written twice |
| — (new) | `JointState`, `joint_state`, `joint_result`, `solve_report`, `joint_iterations` | the state, its seed, its read-out and the report view of the log |
| — (new) | `iterate`, `windowed`, `window_geometry`, `stft_power`, `frame_starts`, `solve_audio` | the combinators and the shared kernels they and the driver both need |

