---
experiment: rp_scv2_all
training_config: conf/experiment/rp_scv2_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_scv2_all`

## Motivation

Regressor row (pilot run of the grid): the best SimpleConv modification, SimpleConv-v2 with a BiGRU head and magnitude front end (`hb_scv2_mag_nogate`), trained as a rate regressor (`pit_mse` loss, `rps` metrics) on the `rp_all` stream. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 1.0e-3 from the family's LR range test (`results/rp_lr_range/rp_scv2_all.json`, read 2026-10-15: loss non-finite at 3.5e-3, a third of it; every earlier SCv2 run trained at 1e-3).
