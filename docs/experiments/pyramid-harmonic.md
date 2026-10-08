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

| experiment | model | temporal model | status |
|---|---|---|---|
| `real_r4_hppnet_pyr_unified` | `hppnet_pyramid` (4 levels, 307-bin grid, 0.45 M params) | temporal convs + CRF (no LSTM) | restarted on `vast` A100 2026-10-08 on the fixed tap conv (`hppnet-pyr-unified-64a67e`, 1.74 it/s, ~9.6 min/epoch); the first A100 run (`-6980ca`, slow kernel, 3 epochs) was cancelled |
| `real_r4_hppnet_pyrlstm_unified` | `hppnet_pyramid` + `RateConvLSTM` head (1.60 M params) | bidirectional ConvLSTM, 33-tap gate conv along rate (±16 bins per frame = the telemetry's largest sustained slew, 15 rev/s per frame) | submitted 2026-10-08 to `vast` A100 after the optimisation rounds below |
| `real_r4_hppnet_l2nolstm_unified` | HPPNet-L2 (CQT) with `head: conv1x1` (0.43 M) | temporal convs + CRF (no LSTM) — the CQT-side counterpart of variant C | `vast` A100 `hppnet-l2nolstm-88d4e8`, 3.6 it/s at batch 128 |
| `real_r4_hppnet_l2convlstm_unified` | HPPNet-L2 (CQT) with `head: convlstm` (1.51 M) | `RateConvLSTM` on the log axis, half-width 15 CQT bins (= the telemetry's largest sustained slew at r ≥ 27.5: `updown` 14.9 log-bins/frame) — the CQT-side counterpart of variant A | first run `-e09290` (2 s/it, slicing bug below) cancelled at epoch 3; resubmitted on the fix |

The 2×2 — {CQT, pyramid} × {no recurrence, ConvLSTM} — plus the published
`real_r4_hppnet_l2_unified` (CQT + `FreqGroupLSTM`) separates the front
end's contribution from the temporal model's.

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
