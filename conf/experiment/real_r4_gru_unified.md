---
experiment: real_r4_gru_unified
training_config: conf/experiment/real_r4_gru_unified.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `real_r4_gru_unified`

## Motivation

Fresh augmented-R4 causal-GRU rerun with 2-second clips, batch 128, PIT-MSE and the unified full-panel validation/stopping protocol.

## Conclusion

First submission (2026-09-08, float16 autocast) diverged: R1 went non-finite at update 1,500, R3 hit persistent non-finite gradients after update 500; R2/R4 were cancelled and the rung resubmitted with `amp_dtype: bfloat16` (see the config comment). Not yet complete. Authorized as one of the sixteen parallel R1–R4 regressor reruns after handoff.
