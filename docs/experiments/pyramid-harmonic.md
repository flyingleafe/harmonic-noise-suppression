# Pyramid harmonic front end: HPPNet on a multi-resolution STFT

**Configs:** `conf/experiment/real_r4_hppnet_pyr_unified.yaml`
**Design note:** [`docs/pyramid-harmonic-frontend-design.md`](../pyramid-harmonic-frontend-design.md)
**Code:** `src/models/harmonic_ports/hppnet_pyramid.py` (registry `hppnet_pyramid`)

## Motivation

Review experiment B retired the comb-gather ports (L3) because they lost to
HPPNet-L2 with the published CQT front end. Re-reading the ports against the
paper showed that B never tested the gather in HPPNet's position: L3 moved the
gather in front of the convolutions, read raw floor-normalised power, dropped
the fundamental and sub-harmonic taps and replaced the octave dilations with
±15 rev/s context. The front-end plots of `rps_tracking.ipynb` showed where
L2's advantage comes from — the CQT's long windows at the BPF lines — and
where it pays: constant-Q resolution `0.0145 r` at every harmonic, a 2–4 s
window below 70 Hz that is padding on training clips, an output axis clamped
below 27.5 rev/s and capped at 150, and a per-bin LSTM that cannot follow a
moving pitch.

The pyramid keeps HPPNet's block order and changes the coordinate system:
nested STFT levels (1 s window to 1200 Hz, halving per level), FPN fusion,
harmonic taps gathered at `k·r` and `r/k` from the level whose window suits the
frequency, octave taps on the rate axis. The first arm is variant C (no LSTM)
— the null hypothesis for the temporal model.

## Protocol

Unified regime of `real_r4_hppnet_l2_unified`: real rung R4
(`hb_m3s2_dload`, 2 s clips), full deterministic panel every 500 updates,
effective batch 128, AdamW 1e-3, monitor `rps_mae`. Comparison row:
`real_r4_hppnet_l2_unified` (and `hppnet_l2_r2_s0`, 2.27 on the frozen real
split under the historical recipe).

## Arms

**Round 1 (k ≤ 32 / k ≤ 9, stopped 2026-10-08).** Best `val/real_overall`
(rev/s, unified panel, one seed) at the stop:

| experiment | model | temporal model | best real_overall (epoch) | status |
|---|---|---|---|---|
| `real_r4_hppnet_pyr_unified` | `hppnet_pyramid` (4 levels, 307-bin grid, k ≤ 32, 0.45 M params) | temporal convs + CRF (no LSTM) | 2.69 (5; never improved after) | `hppnet-pyr-unified-64a67e`, 1.74 it/s, cancelled at round 68 |
| `real_r4_hppnet_pyrlstm_unified` | `hppnet_pyramid` + `RateConvLSTM` head (1.60 M) | bidirectional ConvLSTM, 33-tap gate conv along rate (±16 bins per frame = the telemetry's largest sustained slew, 15 rev/s per frame) | **2.17** (34) | `hppnet-pyrlstm-c5c30d`, 1.21 it/s, cancelled at round 40 |
| `real_r4_hppnet_l2nolstm_unified` | HPPNet-L2 (CQT, k ≤ 9) with `head: conv1x1` (0.43 M) | temporal convs + CRF (no LSTM) | 2.45 (24) | `hppnet-l2nolstm-88d4e8`, 3.6 it/s, saturation stop at round 77 |
| `real_r4_hppnet_l2convlstm_unified` | HPPNet-L2 (CQT, k ≤ 9) with `head: convlstm` (1.51 M) | `RateConvLSTM` on the log axis, half-width 15 CQT bins (= the telemetry's largest sustained slew at r ≥ 27.5: `updown` 14.9 log-bins/frame) | **2.16** (23) | `hppnet-l2convlstm-4051fc` (restart of `-e09290`, slicing bug below), cancelled at round 77 |

Reading: the ConvLSTM head is worth ~0.3–0.5 on both front ends (2.16/2.17
vs 2.45/2.69) and lands where the published L2 sits (`hppnet_l2_r2_s0` 2.27
on the frozen split, `nv2_mixed_hppnet_l2` 2.25 on this panel); the pyramid
shows no margin over the CQT. The round was stopped on a diagnosis, not on
these numbers: at the data's 60–90 rev/s, k_max 32 reads up to 1.9–2.9 kHz,
so pyramid levels 2 and 3 (2.4–8 kHz) were never read by the taps, and on
the CQT side the grid ends at 4.37 kHz with k ≤ 9 — neither family saw
DREGON's strong rotor comb around 6 kHz (harmonics ~40–80). The loss grids
also differed (0–299 vs 0–150), which made the training losses only roughly
comparable.

**Round 2 (k ≤ 84, conv-only heads, submitted 2026-10-08).**

| experiment | model | status |
|---|---|---|
| `real_r4_hppnet_pyrk84_unified` | `hppnet_pyramid` on 0–150 rev/s (154 bins), taps k ≤ 84 + 1/2..1/8 (91), `f_max` 7.5 kHz, `head: conv1x1` (0.56 M) | `hppnet-pyrk84-400534`, `vast` A100, 2.29 it/s (~7.3 min/epoch) |
| `real_r4_hppnet_l2k84nolstm_unified` | HPPNet-L2, CQT 385 bins (to 7.04 kHz), `HarmonicDilatedConv` k = 2..84 (83 branches, the published eight first and unchanged), `head: conv1x1` (0.90 M) | `hppnet-l2k84nolstm-375978` at 1.23 s/it (~10 min/epoch) for 2 epochs, then resumed from its R2 checkpoint as `hppnet-l2k84nolstm-r-e9732d` on the fused operator below (W&B `g8egadgl` continues) |

Same k set, same sub-harmonics, same head, same loss range; the only
difference is the axis (CQT: Δf/k = 0.0145·r at every k, so high harmonics
add evidence, not resolution; pyramid level 3: 7.8 Hz bins → 0.13 rev/s per
harmonic at k = 60). The pyramid step is ~15 % cheaper than round 1's
despite 2.3× the taps (halving G pays for them). The published
`real_r4_hppnet_l2_unified` (CQT + `FreqGroupLSTM`) has still never been run.

**`HarmonicDilatedConv` with 83 branches, fused.** The CQT arm went from
3.6 it/s (k ≤ 9) to 1.23 s/it (k ≤ 84): the published operator sums 83
`Conv2d((1,3), dilation=d_k)` outputs. Algebraically that is one sparse
convolution with taps at `{0} ∪ {±d_k}` (167 taps; the 83 centre matrices
collapse into their sum), so it now runs through the same `tap_conv` kernel
as the pyramid (`models/harmonic_ports/tap_conv.py`) **on the unchanged
branch parameters** — same state-dict keys, same function, same
per-parameter gradients (fp64 tests against the upstream branch sum, k ≤ 9
and k ≤ 84), so every earlier L2 result and checkpoint stands. Measured
(T4, B=32, full step): k = 84 1.65 → 1.33 s (1.24×); k = 9 0.35 → 0.40 s
(cuDNN's eight small convs win there), so the branch sum is kept below 16
branches. Predicted ~2.5× from output-write traffic; the gather kernel,
not cuDNN, turned out to be the limiter — the same ~6 ms per tap per 700k
rows the pyramid's `conv_3` shows.

Placement notes. `uni-gpushort` was full (two `kla-loglinear` jobs holding both
slots behind a 53-job backlog), so the smoke was repinned to `vast`. The first
vast rental was a **Tesla P40** (Pascal, no fp16 tensor cores): 10 s/step at
batch 64, GPU at 100 % — cancelled. Resubmitted with `--gpu-type A100`
(the project's vast convention): 1.2–1.3 s/step at batch 64 → ~21 min per
1000-step epoch, 5–10× HPPNet-L2's (`prior_v1_hppnet_l2`: 2–4 min), with the
GPU at 99–100 % and the CPU at 7 % — the model, not the data.

**Where the time went** (`scripts/bench.py --target hppnet_pyramid`, T4,
B=64, 2 s, fwd+bwd per stage): `conv_3` harmonic taps 4.44 s, `block_4/5`
3.94 s, `block_2/2_5` 0.57 s, `block_6–8` + head 0.38 s, front end 0.31 s —
the two `ProportionalTapConv` layers were 87 % of a 9.64 s step, and three
taps at 128 channels cost as much as 38 at 16. `BENCH_PROFILE=1` (B=32):
`index_add_` (`indexFuncLargeIndex`) alone was 62 % of CUDA time — the
backward scatter of `index_select` on the innermost axis has inner size 1,
one uncoalesced `atomicAdd` per element. Fusing the taps into one autograd
Function with a single dense input gradient changed nothing (same kernel).
The fix is the layout: `x` transposed once to `(F, B·T, C)`, so every
`index_select`/`index_add_` moves contiguous rows of `B·T·C` and the per-tap
contraction is a plain `(G·B·T, C) @ (C, O)` GEMM with no permute copies.
T4 step 9.64 → 2.71 s (B=64); taps 8.4 → 1.6 s; `index_add_` 62 → 7.5 %.

Further exact rounds (T4, B=64 unless noted): chunking the taps so one
gather feeds one `(G·B·T, J_c·C) @ (J_c·C, O)` GEMM per chunk, 2.71 →
2.50 s; an `(F, C, N)` layout that reads the gathered block as a transposed
GEMM operand, block_4/5 0.76 → 1.85 s (reverted); `channels_last` for the
conv stack, no change (B=32: 1.19 → 1.26 s). Profile after all of it (B=32,
1.21 s) is flat: cuDNN conv bwd 23 %, copies 18 %, elementwise 19 %,
`index_select` 9 %, `index_add_` 8 %, ReLU bwd 8 %, cuDNN conv fwd 7.5 %.
That is the exact-optimisation floor without a fused gather-GEMM kernel;
the structural levers (drop level 3 — only read for `k·r ≥ 4800 Hz`, i.e.
r ≥ 150 rev/s at `k_max` 32 — ~−22 %; `block_2/2_5` only on the half of
each coarse level the taps read, ~−35 % of that stage) change the model and
are for the next generation, not the running arm. Variant C was therefore
left running (the "replace if another 2×" rule was not met).

**`RateConvLSTM` was 6× slower than it should be.** The L2 ConvLSTM arm ran
at 2 s/it at batch 128 (GPU 100 %). Bench (`--target rate_convlstm`, T4,
B=16, T=63, G=352): dense vs separable recurrent conv, `Conv1d` vs
`Conv2d((1,k))` vs `unfold+matmul` vs fp32, eager vs TorchScript vs
Inductor cell — all within 15 % (1.06–1.36 s). Profile: `gx[:, :, s]` per
step — its backward (`SliceBackward`/`SelectBackward`) materialises a
full-size zero copy of the `(B, 4H, T, G)` gate tensor every step: `fill_`
25 % + `copy_` 36 % + `add_` 24 % = 85 % of CUDA time; the conv was 9 %.
One `gx.unbind(2)` before the loop: 1131 → 190 ms (B=16); 593 ms at the A
arm's shape (B=64, G=307). Same maths (verified against the slicing form in
fp64, forward and all gradients).

**Fused Triton `tap_conv` (2026-10-08).** `models/harmonic_ports/tap_conv_triton.py`,
dispatched on CUDA by `tap_conv` (`TAP_CONV_TORCH=1` forces the torch path).
Forward: one program per `(frame, 64 rate bins)` tile (16 bins at C = 128),
per tap the two interpolation rows are read straight from `x` in `(N, F, C)`
layout, lerped in registers and fed to `tl.dot`; the `(bins, O)` accumulator
is written once into the model's `(B, O, T, G)`. Backward: `dx` — per tile,
`gy · W_jᵀ` scattered with **relaxed** fp32 atomics onto the two source rows
(acq-rel atomics were 2× slower); `dw` — per `(tap, 32 row splits)`,
`gyᵀ · a_j` accumulated in registers from a row-major copy of `gy` (reading
`gy` in `(B, O, T, G)` per row cost 16× in sectors: 96 → 4 ms), partials
summed in torch. fp32 operands use `input_precision="ieee"` (TF32 failed the
fp32 dW test). GPU tests (`tests/models/test_tap_conv_triton.py`, CUDA only)
compare forward and all three gradients to the fp32 torch path on random
tables with duplicates/invalid entries and on the real k ≤ 84 pyramid and
CQT tables. A100, B=64, fwd+bwd: conv_3 (91 taps) 127 → 34 ms, block_4
(C=128) 37 → 11.6 ms; full pyramid step 498 → 293 ms (1.70×); the
taps are now ~25 % of the step. The T4 is not a target (sm_75: fp16 dot
failed to lower in Triton 3.3). On the CQT arm the first (T4) measurement
tied cuDNN's 83 branch convs; the A100 one (`--target hppnet_l2`, B=64,
385 bins, k ≤ 84) did not: branch sum 551 ms vs fused 239 ms per step, so
`HarmonicDilatedConv.FUSED_FROM = 16` — the published k ≤ 9 stays on the
branch sum, the k ≤ 84 arm was resumed on the fused path at epoch 20
(`hppnet-l2k84nolstm-r3-939d4b`, 1.24 s/it → 1.9 it/s).
The `dx` atomics (19 ms) are the remaining tap cost; an inverse-table gather
(no atomics, ~2.6× the forward's dot work) is the next step if needed.

**Round-2 diagnosis (2026-10-08, probes on the held-out
`free-flight_nosource_room1`, mic 0, cruise 2 s clips; `best_real_overall`
checkpoints at pyrk84 epoch 13 / l2k84 epoch 6).** At the time of the probe
the panel read pyr 2.69 → pyrk84 2.67 and l2nolstm 2.45 → l2k84 2.31
(`val/real_overall`; per view pyrk84 is *worse* on the DREGON views r1/r2
3.73/3.42 vs 3.58/3.28 and better only on `real_nosource`/FLY124).

1. *Band knock-out.* Low-passing the clips at 1.2 kHz (k ≤ 15 at 80 rev/s)
   leaves pyrk84 unchanged (2.99 → 2.73 PIT-MAE) and costs l2k84 0.7
   (2.57 → 3.26); the k ≤ 32 pyramid collapses (3.10 → 13.6) and l2nolstm
   loses 3 (1.72 → 4.72). Every model outputs zero on a clip with the band
   below 1.2 kHz removed (the empty low band reads as "no rotor"), so only
   the low-pass row is interpretable: **pyrk84 does not use anything above
   1.2 kHz; the CQT k ≤ 84 arm does.**
2. *Tap knock-out by order* (zeroing `conv_3.weight[k]` / the `+d_k` branch
   tap, 12 clips, mean PIT-MAE): pyrk84 2.85 all taps → 2.76 without k ≥ 16
   → **1.94 without k ≥ 31 → 1.56 without k ≥ 61**; l2k84 2.55 → 19.6 (k ≥ 16
   removed: it leans on 1.2–2.4 kHz) → 2.41 → **1.88 without k ≥ 61**. In
   both families the taps that read above 4.8 kHz (pyramid level 3) are
   *harmful* on DREGON cruise, twice as much on the pyramid (−1.29 vs −0.67
   rev/s when removed); the k = 31–60 taps help both by a similar amount
   (pyramid 0.38, CQT 0.53). The null panel delta is level 3's harm
   cancelling level 2's gain. The k ≤ 32 pyramid is also better without its
   top taps (k ≥ 16: 3.00 → 2.39).
3. *The signal is there.* Rate-synchronous average spectrum of the recording
   (8 mics, telemetry × 0.9935 — the acoustic rate is 0.65 % below
   `motors_measured`): per-rotor lines at **k = 14, 42, 70** (motor-pole
   multiples; 3–5 dB per frame at the tap position), weak at 28/56, nothing
   at 84. Per-frame tap contrast at k = 70 is 4.3 dB and *does not depend on
   the window* (4.7 at 1 s, 5.1 at 16 ms; contrast = dB at the tap over the
   median of the ±300 Hz band): the slew-smear argument for short windows up
   top (design note § 2) buys nothing at the tap position on DREGON cruise.
   The k = 70 tap discriminates the right hypothesis from one 1 rev/s off by
   3.9 dB at level 3, so the level-3 harm is learned, not inherent.
4. *Grid-offset sweep* (resampling the clips so the rates move through one
   0.977 rev/s cell): no dependence of |err| or of the logit at the true bin
   on the distance to the nearest grid point, for any arm. The pyramid's
   tap misalignment (harmonic k of a mid-cell rate sits `k·0.49/Δf_l` bins
   from the nearest hypothesis' tap — 3–7 bins at the top of every level
   against a 2-bin Hann half-lobe; the CQT's is ≤ 0.5 bin for every k) is
   therefore absorbed by the learned blur, not visible at the output.
5. *Code checks, no bug found:* `build_tap_table` level assignment, ceilings
   (`lo + 1` always inside its level), `f_min`/`f_max` masks (91/91 taps
   valid at 80 rev/s), per-level `InstanceNorm` (per call, no train/eval
   mismatch), FPN upsampling alignment, shared frame grid. **One config
   regression:** on the 154-bin grid `block_4`'s `2r` tap is masked for
   r > 73.2 rev/s (ceiling `(G−2)·Δr`), i.e. across the whole DREGON cruise
   band; the 307-bin k ≤ 32 arm had it to 148 rev/s and the CQT's octave
   dilation always has it. Not the cause of item 2 but a confound of the
   k ≤ 32 → k ≤ 84 comparison.
6. *Full frozen real split, four regimes*
   (`scripts/tap_ablation_regimes.py`, `uni-cpu` job
   `tap-ablation-regimes-660f0b`, `results/regime_decomp/tap_ablation.json`;
   current `best_real_overall` of both arms re-fetched from R2, 296 clips,
   per-frame PIT MAE, rev/s; shares zero 12.7 % / standby 11.6 / ramp 3.8 /
   cruise 72.0; dregon/standby and michaels/ramp are too thin to read):

   | taps zeroed | pyrk84 all / dregon / michaels | zero · standby · ramp · cruise | l2k84 all / dregon / michaels | zero · standby · ramp · cruise |
   |---|---|---|---|---|
   | none | 2.68 / 3.33 / 1.73 | 1.98 · 2.00 · 8.39 · 2.62 | 2.31 / 2.70 / 1.74 | 0.37 · 2.72 · 9.78 · 2.20 |
   | k ≥ 61 | **2.22** / 2.59 / 1.69 | 1.72 · 2.25 · 8.28 · **1.99** | 2.31 / 2.58 / 1.92 | 1.09 · 2.66 · 10.36 · 2.05 |
   | k ≥ 31 | 3.31 / 3.59 / 2.90 | 0.66 · 3.44 · 15.0 · 3.14 | 4.53 / 5.79 / 2.68 | 2.01 · 3.37 · 13.6 · 4.69 |
   | k ≥ 16 | 4.65 / 5.65 / 3.20 | 0.52 · 3.70 · 20.7 · 4.70 | 11.1 / 17.3 / 2.03 | 0.41 · 3.43 · 15.0 · 14.0 |

   The 12-clip picture holds on the split: the pyramid's k ≥ 61 rows cost
   0.46 overall and 0.74 on DREGON (cruise 3.47 → 2.50), nothing on
   michaels; the CQT's k ≥ 61 taps are neutral (DREGON −0.12, michaels
   +0.18). The k = 31–60 taps are worth 1.1 (pyramid) and 2.2 (CQT); the
   k = 16–30 taps 1.3 and 6.6. Without its level-3 rows the pyramid conv
   arm (2.22) is ahead of the CQT conv arm (2.31) and at the ConvLSTM arms'
   2.16/2.17. Zero-regime errors move the other way for both families
   (pyramid 1.98 → 1.72 but CQT 0.37 → 1.09): the top taps also carry the
   "all rotors stopped" evidence.

Reading: the pyramid reads its high harmonics as well as the CQT does up to
level 2 (k ≤ 60, < 4.8 kHz); level 3 (128 ms window, 7.8 Hz bins) is what
it does not profit from, and since the k = 70 line keeps its contrast at a
1 s window, level 3's short window buys nothing on this data. Next arms
(not run): (a) pyrk84 with `k_max` 60 (reads to 4.8 kHz, drops level 3 from
the taps) and `out_bins` 307 to restore the `2r` tap, (b) a pyramid whose
top level keeps the 4096 window to 8 kHz.

## Results

Round 2 finished (2026-10-09): `real_r4_hppnet_pyrk84_unified` best
`val/real_overall` **2.67** @ epoch 13 (79 epochs, last five ~2.95; W&B
`rnzenk2s`), `real_r4_hppnet_l2k84nolstm_unified` **2.31** @ 6 (66 epochs,
last five ~2.5; `g8egadgl`). With round 1: pyramid conv-only 2.69 → 2.67
for k ≤ 32 → 84; CQT conv-only 2.45 → 2.31 for k ≤ 9 → 84; ConvLSTM heads
2.17 (pyramid) / 2.16 (CQT) at k ≤ 32 / 9. Per-rotor room ownership of the
k ≈ 70 line (rate-synchronous spectra, 8-mic mean, cruise): room 1 rotor 0
(fastest), room 2 rotor 3 (third fastest); room 2's command labels sit
×0.978 / ×0.969 / ×0.983 / ×0.995 above the acoustic rate per rotor.

## Conclusion

The pyramid front end reads high harmonics as well as the CQT up to 4.8 kHz
(its level-2 taps are worth 1.1 rev/s) and is hurt by its level-3 taps
(−0.46 overall, −0.74 on DREGON), which the CQT's equivalents are not; the
two cancel on the panel. The level-3 harm is a learned rotor-identity cue
(the k ≈ 70 motor-pole line sits on a different rotor in the training room
than in the validation room) amplified on the pyramid by room 2's 1–3 %
per-rotor command-label error, which a linear 7.8 Hz grid turns into a
10+-bin tap misplacement at k = 70 where the CQT's log grid keeps it under
two bins. Zeroing the k ≥ 61 rows of the trained pyramid gives 2.22, ahead
of the CQT conv arm and level with the ConvLSTM arms. Handoff and next
arms: `docs/handoff-pyramid-harmonic.md`.
