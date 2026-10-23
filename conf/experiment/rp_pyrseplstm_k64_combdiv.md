---
experiment: rp_pyrseplstm_k64_combdiv
training_config: conf/experiment/rp_pyrseplstm_k64_combdiv.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_k64_combdiv`

## Motivation

First synthetic-only arm of the rp protocol (Dmitrii, 2026-10-22): `rp_pyrseplstm_k64_all` (pyramid + separable ConvLSTM, k ≤ 64, real_macro 2.52 on `alllab`) with every real noise recording replaced by the diverse static comb, rendered online (`conf/online_mix/rp_combdiv.yaml`; draws in `docs/explainers/diverse-comb.qmd`). Model, loss, LR schedule, batch and speech mixing are the `all` run's. Validation is the real panel plus 512 pre-drawn combdiv items (`conf/validation/rp_real_combdiv.yaml`); the run stops only when neither `real_macro` nor `static_comb` has improved its 5-round median for 30 rounds, so it keeps learning the synthetic task while real scores drift.

## Conclusion

Ran 2026-10-23 (job `rp-pyrseplstm-k64-combdi-a48dbc`, W&B `al9s52jz`) to the 100 000-step cap; the stop never fired because `static_comb` kept improving. The model learns its own comb (`static_comb` 0.387 rev/s at step 97 000) but does not read real recordings: `real_macro` 16.20 at that checkpoint and 15.39 at the best real round (step 100 000), against 2.52 for the real-trained `rp_pyrseplstm_k64_alllab`. It keeps the stopped rotors (zero 0.73) and fails on standby, ramps and cruise (27.60 / 31.77 / 13.13). Paper: the static-comb row of the simplistic-synthetic ablation. Details in `docs/experiments/rp-synthetic.md`.
