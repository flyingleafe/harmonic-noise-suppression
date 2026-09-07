---
experiment: hf0_l2_r4_s0
training_config: conf/experiment/hf0_l2_r4_s0.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `hf0_l2_r4_s0`

## Motivation

Review experiment B: hf0 L2, seed 0, real-data fine-tuning.
Matched counterpart: `hf0_l3_r4_s0`. Warm-starts only from `hf0_l2_comb_s0`; uses the frozen real development split.
Both levels use four Gaussian output layers and the same linear output grid,
CRF, training policy, budget and `rps_mae` selection; HPPNet's recurrent width
is matched at 128. L2 retains the native log-input harmonic architecture; L3
uses the linear-STFT comb gather. See the batch record's “Review experiments
B/C” section for controls and the remaining front-end differences.

## Conclusion

Results pending. Preflight is not evidence of comparative accuracy.
The shared batch record will hold the cross-seed comparison after completion.
