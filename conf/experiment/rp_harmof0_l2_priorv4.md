---
experiment: rp_harmof0_l2_priorv4
training_config: conf/experiment/rp_harmof0_l2_priorv4.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_harmof0_l2_priorv4`

## Motivation

HarmoF0-L2 in the paper grid's HARD synthetic setup (approved by Dmitrii, 2026-10-26): `rp_harmof0_l2_all` with every real noise recording replaced by drones of the wide prior in noise model v4 (`prior-v4-banks` `prior_v4_n16384.npz`: 16 384 gated draws, rules `prior_v4/4`), flown on the reduced trajectory hyperprior, rendered online with per-clip wind (`wind_sc`) and 3.4 dB per-clip line scatter (`conf/online_mix/rp_priorv4.yaml`). Model, loss, LR schedule, batch and speech mixing are the `all` run's; no label augmentation. Validation is the real panel, `static_comb` and the six synthetic sectors of `validation-sectors-v4` (`conf/validation/rp_real_priorv4.yaml`); the run stops only when neither `real_macro` nor `uniform` has improved its 5-round median for 30 rounds. The synthetic-selected checkpoint (`best_uniform_median`) seeds `rp_harmof0_l2_priorv4_ft`. Read against the real-only `rp_harmof0_l2_all` / `rp_harmof0_l2_alllab` and `rp_pyrseplstm_k64_combdiv`.

## Conclusion

Pending.
