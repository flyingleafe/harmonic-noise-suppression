---
experiment: hppnet_l2_r4_s2
training_config: conf/experiment/hppnet_l2_r4_s2.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hppnet_l2_r4_s2`

## Motivation

Review experiment B: hppnet L2, seed 2, real-data fine-tuning.
Matched counterpart: `hppnet_l3_r4_s2`. Warm-starts only from `hppnet_l2_comb_s2`; uses the frozen real development split.
Both levels use four Gaussian output layers and the same linear output grid,
CRF, training policy, budget and `rps_mae` selection; HPPNet's recurrent width
is matched at 128. L2 retains the native log-input harmonic architecture; L3
uses the linear-STFT comb gather. See the batch record's “Review experiments
B/C” section for controls and the remaining front-end differences.

## Conclusion

Results pending. Preflight is not evidence of comparative accuracy.
The shared batch record will hold the cross-seed comparison after completion.
