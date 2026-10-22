---
experiment: rp_pyrsep_in_alllab
training_config: conf/experiment/rp_pyrsep_in_alllab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_pyrsep_in_alllab`

## Motivation

The missing cell of the head × stream square (Dmitrii, 2026-10-18): the pyramid with the separable rate-kernel head, no `block_4`/`block_5`, instance norm (`conf/model/hppnet_pyrsep_in.yaml`, as `rp_pyrsep_in_all`, 2.59 on the plain stream) trained on the `rp_all_labelaug` stream (`rate_warp` p 1). Its three siblings: `rp_pyrsep_in_all` (sepconv / plain), `rp_pyrseplstm_all` (ConvLSTM / plain, 3.01), `rp_pyrseplstm_alllab` (ConvLSTM / labelaug, 2.74). On the rp_real panel every labelaug-trained model so far gains on DREGON ramps and improves further when its taps k ≥ 61 are zeroed at eval, while the plain-stream models collapse under the same knock-out; this arm says whether the stream or the recurrence is the ramp factor, and whether sepconv + labelaug reaches the old dense ConvLSTM's 2.57 without recurrence.

## Conclusion

Defined 2026-10-18. Ran on `uni` (job `rp-pyrsep-in-alllab-eb97f2`), stopped at step 33500 (saturation); pulled 2026-10-22. Best median round (step 24500, `scripts/rp_history.py --best`): real_macro **2.73**, real 2.50, dregon 4.22, md2 1.23; real_zero 1.86, real_standby 3.39, real_ramp 7.07, real_cruise 2.09. Against `rp_pyrsep_in_all` (2.59; dregon 3.78, md2 1.39, ramp 9.93, cruise 1.57): labelaug trades DREGON and cruise for MD2 and the ramps, the pattern of every other family. The sepconv and ConvLSTM heads end level on labelaug overall (2.73 vs 2.74); most of the ramp gain is the stream's (9.93 → 7.07), the recurrence adds the rest (6.34 in `rp_pyrseplstm_alllab`).
