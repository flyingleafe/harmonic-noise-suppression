---
experiment: rp_pyrseplstm_k64_dregonlab
training_config: conf/experiment/rp_pyrseplstm_k64_dregonlab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_pyrseplstm_k64_dregonlab`

## Motivation

Paper, "Transfer between drones": HRSPNet (`hppnet_pyrseplstm_k64`, as `rp_pyrseplstm_k64_alllab`, its better stream on both drones) trained on the DREGON recordings alone (`rp_dregon_labelaug`) and scored on MD2. The run stops on the DREGON score only (`validation.control: [dregon]`), so MD2 never steers it; the transfer cell is MD2 (all, cruise) at the round with the best five-round median of `dregon`. Reference: `rp_pyrseplstm_k64_alllab` (MD2 1.14, MD2 cruise 0.60). Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-10-23; submitted to `uni`.
