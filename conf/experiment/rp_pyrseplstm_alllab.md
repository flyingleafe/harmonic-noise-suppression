---
experiment: rp_pyrseplstm_alllab
training_config: conf/experiment/rp_pyrseplstm_alllab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_pyrseplstm_alllab`

## Motivation

Claim 3, the HRSPNet candidate final form (Dmitrii, 2026-10-16): `conf/model/hppnet_pyrseplstm.yaml` — the pyramid with a **separable `RateConvLSTM`** head, `block_4`/`block_5` dropped, instance norm, k ≤ 84 + sub-harmonics, no tap ceiling. Against `rp_pyrsep_in_all` (same model with the same 33-tap rate kernel applied to the current frame instead of the recurrent state) it is the recurrence-only ablation; the old dense ConvLSTM checkpoint scored 2.57 on this panel with ramp 5.71 vs the sepconv row's 9.93. This arm trains on the `rp_all_labelaug` stream (label-altering (rate_warp) augmentation), the comparison to the earlier label-altering-augmented runs.

## Conclusion

Defined 2026-10-16; not yet run.
