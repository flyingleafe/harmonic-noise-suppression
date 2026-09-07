---
experiment: hf0_l3_comb_s2
training_config: conf/experiment/hf0_l3_comb_s2.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hf0_l3_comb_s2`

## Motivation

Review experiment B: hf0 L3, seed 2, static-comb pretraining.
Matched counterpart: `hf0_l2_comb_s2`. Trains from initialization on the shared static-comb policy.
Both levels use four Gaussian output layers and the same linear output grid,
CRF, training policy, budget and `rps_mae` selection; HPPNet's recurrent width
is matched at 128. L2 retains the native log-input harmonic architecture; L3
uses the linear-STFT comb gather. See the batch record's “Review experiments
B/C” section for controls and the remaining front-end differences.

## Conclusion

Results pending. Preflight is not evidence of comparative accuracy.
The shared batch record will hold the cross-seed comparison after completion.
