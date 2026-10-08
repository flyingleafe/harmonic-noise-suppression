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
| `real_r4_hppnet_pyrk84_unified` | `hppnet_pyramid` on 0–150 rev/s (154 bins), taps k ≤ 84 + 1/2..1/8 (91), `f_max` 7.5 kHz, `head: conv1x1` (0.56 M) | `vast` A100 |
| `real_r4_hppnet_l2k84nolstm_unified` | HPPNet-L2, CQT 385 bins (to 7.04 kHz), `HarmonicDilatedConv` k = 2..84 (83 branches, the published eight first and unchanged), `head: conv1x1` (0.90 M) | `vast` A100 |

Same k set, same sub-harmonics, same head, same loss range; the only
difference is the axis (CQT: Δf/k = 0.0145·r at every k, so high harmonics
add evidence, not resolution; pyramid level 3: 7.8 Hz bins → 0.13 rev/s per
harmonic at k = 60). The pyramid step is ~15 % cheaper than round 1's
despite 2.3× the taps (halving G pays for them). The published
`real_r4_hppnet_l2_unified` (CQT + `FreqGroupLSTM`) has still never been run.

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

## Results

Pending.

## Conclusion

Pending.
