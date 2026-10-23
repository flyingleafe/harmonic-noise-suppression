---
experiment: rp_scv2_md2
training_config: conf/experiment/rp_scv2_md2.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_scv2_md2`

## Motivation

Paper, "Transfer between drones" (claim 5): the SimpleConv-v2 regressor (Conv+BiGRU in the paper; `hb_scv2_mag_nogate`, `pit_mse` loss, `rps` metrics), trained on the `rp_md2` stream only and scored on both drones of the rp_real panel. Since 2026-10-23 the run stops on the MD2 score only (`validation.control: [md2]`), so DREGON never steers it; the transfer cell is DREGON (all, cruise) at the round with the best five-round median of `md2`. Reference: `rp_scv2_all` (DREGON 3.82, DREGON cruise 2.64). Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; submitted to `uni` 2026-10-23. Max LR 1e-3 — the LR every architecture we build on was published with (docs/experiments/paper-real-protocol.md § "Budget and LR"); the family's LR range test is reported in the appendix.
