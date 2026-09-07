---
experiment: hppnet_l3_r4_s1
training_config: conf/experiment/hppnet_l3_r4_s1.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hppnet_l3_r4_s1`

## Motivation

Review experiment B: hppnet L3, seed 1, real-data fine-tuning.
Matched counterpart: `hppnet_l2_r4_s1`. Warm-starts only from `hppnet_l3_comb_s1`; uses the frozen real development split.
Both levels use four Gaussian output layers and the same linear output grid,
CRF, training policy, budget and `rps_mae` selection; HPPNet's recurrent width
is matched at 128. L2 retains the native log-input harmonic architecture; L3
uses the linear-STFT comb gather. See the batch record's “Review experiments
B/C” section for controls and the remaining front-end differences.

## Conclusion

Results pending. Preflight is not evidence of comparative accuracy.
The shared batch record will hold the cross-seed comparison after completion.
