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
| `real_r4_hppnet_pyr_unified` | `hppnet_pyramid` (4 levels, 307-bin grid, 0.45 M params) | temporal convs + CRF (no LSTM) | running on `vast` A100 since 2026-10-07 22:55 UTC (`hppnet-pyr-unified-6980ca`, W&B `real_r4_hppnet_pyr_unified`) |
| `real_r4_hppnet_pyrlstm_unified` | `hppnet_pyramid` + `RateConvLSTM` head (1.60 M params) | bidirectional ConvLSTM, 33-tap gate conv along rate (±16 bins per frame = the telemetry's largest sustained slew, 15 rev/s per frame) | prepared 2026-10-08, not submitted |

Placement notes. `uni-gpushort` was full (two `kla-loglinear` jobs holding both
slots behind a 53-job backlog), so the smoke was repinned to `vast`. The first
vast rental was a **Tesla P40** (Pascal, no fp16 tensor cores): 10 s/step at
batch 64, GPU at 100 % — cancelled. Resubmitted with `--gpu-type A100`
(the project's vast convention): 1.2–1.3 s/step at batch 64 → ~21 min per
1000-step epoch.

## Results

Pending.

## Conclusion

Pending.
