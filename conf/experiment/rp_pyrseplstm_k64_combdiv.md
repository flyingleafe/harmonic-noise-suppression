---
experiment: rp_pyrseplstm_k64_combdiv
training_config: conf/experiment/rp_pyrseplstm_k64_combdiv.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_k64_combdiv`

## Motivation

First synthetic-only arm of the rp protocol (Dmitrii, 2026-10-22): `rp_pyrseplstm_k64_all` (pyramid + separable ConvLSTM, k ≤ 64, real_macro 2.52 on `alllab`) with every real noise recording replaced by the diverse static comb, rendered online (`conf/online_mix/rp_combdiv.yaml`; draws in `docs/explainers/diverse-comb.qmd`). Model, loss, LR schedule, batch and speech mixing are the `all` run's. Validation is the real panel plus 512 pre-drawn combdiv items (`conf/validation/rp_real_combdiv.yaml`); the run stops only when neither `real_macro` nor `static_comb` has improved its 5-round median for 30 rounds, so it keeps learning the synthetic task while real scores drift.

## Conclusion

Defined 2026-10-22; not yet run.
