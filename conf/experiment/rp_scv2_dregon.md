---
experiment: rp_scv2_dregon
training_config: conf/experiment/rp_scv2_dregon.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_scv2_dregon`

## Motivation

Paper, "Transfer between drones" (claim 5): the SimpleConv-v2 regressor (Conv+BiGRU in the paper; `hb_scv2_mag_nogate`, `pit_mse` loss, `rps` metrics), trained on the `rp_dregon` stream only and scored on both drones of the rp_real panel. Since 2026-10-23 the run stops on the DREGON score only (`validation.control: [dregon]`), so MD2 never steers it; the transfer cell is MD2 (all, cruise) at the round with the best five-round median of `dregon`. Reference: `rp_scv2_all` (MD2 1.49, MD2 cruise 0.93). Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-14; submitted to `uni` 2026-10-23. Max LR 1e-3 — the LR every architecture we build on was published with (docs/experiments/paper-real-protocol.md § "Budget and LR"); the family's LR range test is reported in the appendix.
