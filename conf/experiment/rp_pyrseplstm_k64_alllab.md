---
experiment: rp_pyrseplstm_k64_alllab
training_config: conf/experiment/rp_pyrseplstm_k64_alllab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_pyrseplstm_k64_alllab`

## Motivation

Harmonic-range arm of the HRSPNet candidate (Dmitrii, 2026-10-18): `rp_pyrseplstm_alllab` (separable ConvLSTM, no `block_4`/`block_5`, instance norm, labelaug stream; 2.74) with the taps cut to k ≤ 64 at training time (`conf/model/hppnet_pyrseplstm_k64.yaml`). Zeroing taps k ≥ 61 in the trained k ≤ 84 checkpoint at eval took `dregon` 4.18 → 3.30 (DREGON cruise 3.43 → 2.29, the best DREGON row on the panel) but cost MD2 (1.31 → 1.79), because the model had trained its MD2 reading on those taps. Training without them tests whether the DREGON gain is reachable with MD2 kept — the honest version of the knock-out diagnostic, on the stream where the knock-out helps.

## Conclusion

Defined 2026-10-18. Ran on `vast` (A100-80, job `rp-pyrseplstm-k64-alllab-c5228f`, HEAD `ec86c6d0`), stopped at step 32000 (saturation). Best median round (step 17000, `scripts/rp_history.py --best`): real_macro **2.52**, real 2.30, dregon 3.89, md2 1.14; real_zero 2.08, real_standby 3.65, real_ramp 6.12, real_cruise 1.76. Against the k ≤ 84 twin `rp_pyrseplstm_alllab` (2.74; dregon 4.18, md2 1.31), training without the k > 64 taps improves both drones, unlike the eval-time knock-out, which bought DREGON at an MD2 cost; the best real_macro of the rp grid so far. Its `all` twin: `rp_pyrseplstm_k64_all`.
