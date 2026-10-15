---
experiment: rp_scv2_alllab
training_config: conf/experiment/rp_scv2_alllab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_scv2_alllab`

## Motivation

Label-altering augmentation reference: the SimpleConv-v2 regressor (`hb_scv2_mag_nogate`, `pit_mse` loss, `rps` metrics), trained on the `rp_all_labelaug` stream; compared against `rp_scv2_all`. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 1e-3 — the LR every architecture we build on was published with (docs/experiments/paper-real-protocol.md § "Budget and LR"); the family's LR range test is reported in the appendix.
