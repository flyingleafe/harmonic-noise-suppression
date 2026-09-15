---
experiment: real_r2_scv2_unified
training_config: conf/experiment/real_r2_scv2_unified.yaml
batch: docs/experiments/paper-regime-matrix.md
---

# `real_r2_scv2_unified`

## Motivation

Fresh R2 SCv2 rerun with 2-second clips, batch 128, PIT-MSE and the unified full-panel validation/stopping protocol.

## Conclusion

First submission (2026-09-08, float16 autocast) degraded from round 8 and stopped non-finite at update 5,500; resubmitted with `amp_dtype: bfloat16` (see the config comment). Not yet complete. Authorized as one of the sixteen parallel R1–R4 regressor reruns after handoff.
