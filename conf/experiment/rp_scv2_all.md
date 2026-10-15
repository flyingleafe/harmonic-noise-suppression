---
experiment: rp_scv2_all
training_config: conf/experiment/rp_scv2_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_scv2_all`

## Motivation

Regressor row (pilot run of the grid): the best SimpleConv modification, SimpleConv-v2 with a BiGRU head and magnitude front end (`hb_scv2_mag_nogate`), trained as a rate regressor (`pit_mse` loss, `rps` metrics) on the `rp_all` stream. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 1e-3 — the LR every architecture we build on was published with (docs/experiments/paper-real-protocol.md § "Budget and LR"); the family's LR range test is reported in the appendix.
