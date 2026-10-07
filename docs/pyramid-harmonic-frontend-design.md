# Multi-resolution STFT pyramid with a harmonic gather — design note

2026-10-07. Record of the design discussion that led to
`src/models/harmonic_ports/hppnet_pyramid.py` (registry key `hppnet_pyramid`).
Read with `docs/harmonic-ports-design.md` (the original gather argument) and
`docs/experiments/paper-regime-matrix.md` § B (why the L3 ports were retired).

## 1. What was wrong with the L3 ports, and what was right about L2

Review experiment B (one seed, historical R2 pool): HPPNet L2 2.27, HPPNet L3
4.30; HarmoF0 L2 2.45, HarmoF0 L3 11.54 (did not train). The record reads this
as "the comb gather is not the adaptation". Re-reading the code against the
published architectures shows that B never tested the gather in HPPNet's
position; L3 bundled eight changes, of which only the axis change was forced:

| # | L3 change | verdict |
|---|---|---|
| 1 | CQT → linear STFT 4096 | forced by the axis choice; the "k cancels" argument is a sufficiency argument, not evidence that high harmonics dominate the error |
| 2 | `HarmonicDilatedConv` → `CombGather` + `Conv2d(K, C, (3,1))` | **misread**: a `(1,3)` kernel at dilation `d` has taps at `p-d, p, p+d` — sub-harmonic, fundamental, harmonic. The port reads `k·r` only, at rate neighbours `r±½` |
| 3 | 7×7 blocks moved *after* the gather, gather on raw power | justified by memory (16× activations); overstated — a per-harmonic loop costs the dilated conv's footprint. Consequence: the gather reads raw periodogram bins instead of learned line features |
| 4 | `log1p(power/floor)/count` evidence | comb-stack philosophy ("start at the Whittle score"); discards level; floor window (120 Hz) swallows lines below ~20 rev/s |
| 5 | `block_4/5` octave dilations → ±15 rev/s context | "octave has no meaning on a rate axis" — wrong direction: an octave up is the rate `2r`, i.e. a gather |
| 6 | pool removed | forced; same in L2 |
| 7 | output grid 0–150/300 directly | fine |
| 8 | lstm 64 | legacy rows only |

HPPNet-L2 (CQT front end kept, per-rotor layers + CRF added) is the best
learned cell of the frozen split. Its front-end plots (`rps_tracking.ipynb`,
cells 16–19) show *why*: the CQT resolves Michael's BPF lines where HarmoF0's
interpolated 1024-STFT shows 10-bin blocks. The resolution advantage is the
CQT's **window length per frequency** (`Q = 68.8` cycles: 1.7 s at 40 Hz,
0.38 s at 180 Hz, 30 ms at 2.3 kHz), not the log axis per se.

L2's own limits, all structural:

- Output axis clamps below 27.5 rev/s (bins 0–54 read one CQT column) and
  caps at 150.
- Resolution in rate units is `0.0145 r` at every harmonic (constant-Q):
  1.2 rev/s at 80 rev/s; two rotors closer than ~1 bin merge on every row.
- On 1 s training clips the CQT's bottom two octaves (< 69 Hz) are 50–75 %
  reflect padding (window 2–4 s); at 8 s validation they are properly defined.
  Train/validation mismatch below ~70 rev/s.
- `FreqGroupLSTM` is a bank of per-bin recurrences: rotors move 0.48 rev/s
  per frame on average (2.65 at the 99th percentile, 23.8 extreme), so outside
  cruise a bin holds a rotor for ~2 frames. The LSTM is a cruise denoiser, not a
  tracker; tracking happens afterwards in the CRF band.

## 2. Resolution requirements

Fundamentals 50–150 rev/s (occasionally to ~1000 for small racing airframes).
Rate-unit resolution of harmonic `k` read at bandwidth `Δf` is `Δf / k`; slew
smear of harmonic `k` over window `T` is `k ṙ T` Hz = `ṙ T` rev/s
(independent of `k`). Balancing resolution `1/(kT)` against smear `ṙT` gives
`T*_k = 1/sqrt(k ṙ)`: windows should shrink with harmonic order roughly as
`k^-1/2` under slew, be as long as possible in cruise. The CQT shrinks as
`k^-1` (too fast at the top); a single STFT does not shrink at all.

## 3. The pyramid

Nested per-level STFTs on one hop (512 → shared 32 ms frame grid):

| level | `n_fft` | window | `Δf` | kept range | bins | line width |
|---|---|---|---|---|---|---|
| 0 | 16384 | 1.02 s | 0.977 Hz | 0–1200 Hz | 1229 | ~1.4 bins |
| 1 | 8192 | 0.51 s | 1.95 | 0–2400 | 1229 | 1.4 |
| 2 | 4096 | 0.26 s | 3.9 | 0–4800 | 1229 | 1.4 |
| 3 | 2048 | 0.13 s | 7.8 | 0–8000 (Nyquist) | 1025 | 1.4 |

Range and bin width double together, so every level has the same bin count and
a line is ~1.4 bins wide at every level: **one shared kernel bank serves all
levels**, as HPPNet's serves all CQT octaves. Nesting (every level starts at
0 Hz) is what makes fusion possible; disjoint bands have no overlap to fuse and
the sliCQ demixing experience (xumx-sliCQ, −2 dB vs STFT despite a better
oracle) says unfused ragged bands lose.

Rate-unit resolution (`Δf_ℓ / k`) at `r` = 50/100/150 rev/s is 0.98 at `k=1`
falling to 0.08/0.16/0.25 at level 0's ceiling, then a constant-Q sawtooth
0.08–0.16 / 0.16–0.33 / 0.24–0.49 above — 4.5–9× finer than the CQT at every
harmonic. Holding a level over two octaves (0–1200 at 0.98 Hz) is what makes
resolution improve with `k` through the low harmonics.

Cost: ~4.7 k bins total against 352 (CQT), 8193 (full-band stack on a common
grid), 16 k (full-band stack per level). Feature maps at 16 channels, batch 16,
2 s: ~150 MB per layer. The transform itself (4 FFTs per frame) is negligible.
Level 0 needs ≥ 2 s clips, hence the unified regime's 2 s chunks.

Alternative considered and deferred: full-band stack on the common 0.977 Hz
grid (`torch.stft(n_fft=16384, win_length=N_ℓ)` — exact alignment by zero
padding, `c_in = L` in `block_1`). Simpler, 1.8× the memory, needs per-level
frequency dilation in `block_1` (`N_0/N_ℓ`) because a 7-bin kernel sees a
coarse level's 16–64-bin lobe as a constant. Its learned `block_1` weights
per input channel would reveal the band→window schedule empirically; worth
running as the schedule-discovery experiment if the pyramid's seams show.

## 4. Fusion (FPN top-down)

`block_1` (shared) per level → for `ℓ = L-2 … 0`: take level `ℓ+1`'s first
`ceil(F_ℓ/2)` bins (they cover exactly level `ℓ`'s range), upsample ×2 along
frequency with **linear interpolation, endpoints aligned** (both grids have bin
0 at DC, so fine bin `i` ↔ coarse `i/2`; `F.interpolate(align_corners=True)`
with `F_ℓ = 2·ceil(F_ℓ/2) − 1`), 1×1 lateral conv, add. Nearest-neighbour
(`i // 2`) would inject a 2-bin staircase at 0.5 cycles/bin — the spatial
frequency `block_1`'s line detectors respond to. After the pass, level `ℓ`'s
map at `f` carries its own resolution and every coarser one; nothing flows
upward (no finer information exists above a level's ceiling). `block_2`,
`block_2_5` (shared) run on the fused maps.

What HPPNet-L2's `block_1` actually learned (`/tmp/hpp_block1_*.png`, from
`hppnet_l2_r2_s0`): 14 of 16 kernels are zero-DC contrast detectors — a
1–2-bin positive stripe with negative flanks, integrated over ±1–3 frames
(responses peak at 0.2–0.35 cycles/bin); 2 carry level (sum ≈ −5 on a dB
input); 4 are time–frequency diagonals (chirp/slew detectors, ±3 bins over
±3 frames ≈ 30 rev/s² at 80 rev/s). Kernel semantics are "1 line width", which
is what the per-level native grids preserve.

## 5. The gather as a strided convolution

With `Δr = Δf_0` (output grid = level 0's grid, 0.977 rev/s; 1229 bins to
1199 rev/s, or a lower subset), harmonic `k` of rate `g` sits at in-level
position `k g / 2^ℓ − offset`. Stride `k / 2^ℓ`: integer on level 0 for every
`k`; on level `ℓ` for `2^ℓ | k`, otherwise split `g` into `2^ℓ / gcd(k, 2^ℓ)`
residue classes each with integer stride and a **constant** fractional offset
`α`. Linear interpolation commutes with the per-harmonic mixing, so
`W_k ((1−α) x[lo] + α x[lo+1])` is one `Conv1d` over frequency with kernel
`[(1−α) W_k, α W_k]` and that stride — HPPNet's "dilation that expands with
position" expressed as a stride on a linear axis. Memory `O(x) + O(y)`; the
`(B, C, T, K, G)` tensor is never materialised.

The implementation shipped first uses the equivalent gather form (index tables
with `lo`, `frac`, `valid` per tap and rate; `index_select` + lerp; the whole
harmonic layer under `torch.utils.checkpoint` so the per-tap tensors are
recomputed in backward). Same arithmetic; swap for the strided-conv form if
the loop shows up in a profile.

Interpolation: linear on power has a −0.9…−1.4 dB chord bias under a Hann main
lobe (4 bins wide); on the fused level maps the coarse levels' lobes are smooth
over 8–64 fine-grid-equivalent bins and the bias vanishes. Nearest-bin reads
give the same max bias with a rate staircase at low `k`; they do not save
memory (the output size is what costs). Exact DTFT reads (zero-padded FFT or a
DFT matrix at `k r_g`) are available on raw spectra only, not on feature maps.

Taps per rate hypothesis: harmonics `k·r` for `k = 1..K` and sub-harmonics
`r/k` for `k = 2..K_sub` (HPPNet's `p−d` taps; the half-rate guard). Each tap
reads from the **finest level whose ceiling exceeds its frequency**; above
`f_max` or Nyquist the tap is masked. For a fixed `k` the rates reading one
level form a contiguous interval; the crossings `r = f_ℓ^hi / k` fall inside
the rotor band for `k ≥ 8` and give the evidence a kink per harmonic per
ceiling. Expected to be diluted over 32 harmonics; diagnostic is the
error-vs-true-rate plot; mitigation is a cross-faded margin above each ceiling
(read both levels, blend with a smoothstep). Each level keeps one extra bin
above its ceiling so `lo+1` never leaves the level.

## 6. Rate-axis blocks and the output

`block_4` (dilation 48 = one octave on the CQT) and `block_5` (12 = quarter
octave with the pool off) are second-order harmonic priors; on the rate axis an
octave up is the rate `2r`, so they become 3-tap gathers at `r·{½, 1, 2}` and
`r·{2^-¼, 1, 2^¼}` with per-tap `(C, C)` weights — the same proportional-tap
operator as the harmonic layer, on the uniform output grid. `block_6–8` stay
temporal `[5,1]` convs. Head: 1×1 to `n_maps` per-rotor layers; readout is the
existing `LayerCRFReadout`.

Output grid inherits level 0's step: `Δr = Δf_0 = 0.977` rev/s, over a lower
subset of its bins (`out_bins`, a config number; first run `1229 // 4 = 307`
bins → 0–299 rev/s, which covers every airframe in the current data and keeps
the loss, readout and rate-axis blocks at the cost of the old 300-bin grid, so
batch sizes are preserved). Expanding the range later is a config change, not
a front-end change. Readout discretisation cost (`h σ 10^(−SNR/20)`) doubles
against the 0.5 grid: 0.025 / 0.078 / 0.25 rev/s at 40 / 30 / 20 dB peak SNR
— under the 0.2 rev/s floor to ~22 dB, under DREGON's ±0.6 label jitter to
~12 dB. Rates from ~1 rev/s are first-class (no clamp). Below `Δf_0` the
harmonics collapse into one bin (`r_min` = one bin); fundamentals under ~4 Hz
sit in the DC lobe.

## 7. Temporal modelling (C shipped; A implemented)

`FreqGroupLSTM` cannot move state across rate bins, which is what a
continuously changing pitch needs. Options, minimal to redesign:

- **A** ConvLSTM over time with gate convolutions along rate; state follows
  the rotor. **Implemented** as `RateConvLSTM` (`head: convlstm`,
  `conf/model/hppnet_pyramid_convlstm.yaml`): the input-to-gate map is 1×1
  (the trunk already supplies rate context), the hidden-to-gate map is a
  `2·half + 1`-tap 1-D convolution, so the state's whole reach is `± half`
  bins per frame. `half = ceil(max_slew / Δr)`, `max_slew` the largest
  **sustained** per-frame speed change in the raw telemetry of the training
  pools (DREGON in-flight + FLY125 resampled to the 31.25 Hz loss grid,
  motor-on frames; 5-frame mean): `updown` 14.6 rev/s per frame (a throttle
  punch, ~460 rev/s²), FLY125 7.3, `rectangle` 4.9, others ≈ 3. Single-frame
  diffs run to 53 on `updown` but reverse within a frame — 900 Hz ESC jitter,
  not motion — so the sustained figure is the one to cover: `max_slew = 15` →
  half-width 16, kernel 33. The earlier "±3 bins covers the 99th percentile"
  figure was the mean-slew view; the kernel must cover the ramps, which are
  exactly the frames the per-bin LSTM loses. Cost: 1.15 M params (33-tap
  recurrent conv on 64 + 64 hidden), 126 sequential steps per 2 s clip.
- **B** dual-path (rate ↔ time) blocks, as `edge_bs_rof` does for bands/time.
- **C** no recurrence: temporal convs + the CRF as the temporal model; the
  principled form is a trainable linear-chain CRF (forward-algorithm NLL
  replacing the Gaussian BCE). **Shipped first** as the null hypothesis: it
  measures what the LSTM was worth.
- **D** four rotor slots: per-slot GRU, predict rate → read the grid in a
  window around the prediction (rate-equivariant soft gather) → update →
  emit a Gaussian layer. A differentiable Kalman/Vold–Kalman tracker;
  identity and crossing native; causal form is what a drone would run.

## 8. Literature anchors (library ids in the bibliography)

VQT (Schörkhuber et al. 2014; PESTO replaced CQT by VQT), MS-SB-CQT (three
`Q`'s in parallel, vocoder), MR-CQTdiff / CQT-Diff (per-octave resolution with
level-matched U-Net fusion — the closest fused-pyramid precedent, generative),
xumx-sliCQ (unfused ragged bands lose), Schlüter & Böck 2014 / madmom (three
windows stacked as channels), Parallel WaveGAN / UnivNet / EnCodec (full-band
multi-resolution as loss/critic), FPN (Lin et al. 2017). No fused frequency
pyramid exists in the pitch-estimation literature.
