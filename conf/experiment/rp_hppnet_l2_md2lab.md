---
experiment: rp_hppnet_l2_md2lab
training_config: conf/experiment/rp_hppnet_l2_md2lab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_hppnet_l2_md2lab`

## Motivation

Claim 6 (labelaug does not buy transfer): HPPNet at ladder level L2 (rp linear grid, `salience_layers_rp`), trained on the `rp_md2_labelaug` stream and compared against `rp_hppnet_l2_md2`. This file and `rp_hppnet_l2_dregonlab` cover both directions, but only the direction with the nominally better transfer in claim 5 will run. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; not yet run. Max LR 1e-3 — the LR every architecture we build on was published with (docs/experiments/paper-real-protocol.md § "Budget and LR"); the family's LR range test is reported in the appendix.
