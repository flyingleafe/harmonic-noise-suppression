---
experiment: rp_pyrseplstm_k64_alllab
training_config: conf/experiment/rp_pyrseplstm_k64_alllab.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_pyrseplstm_k64_alllab`

## Motivation

Harmonic-range arm of the HRSPNet candidate (Dmitrii, 2026-10-18): `rp_pyrseplstm_alllab` (separable ConvLSTM, no `block_4`/`block_5`, instance norm, labelaug stream; 2.74) with the taps cut to k ≤ 64 at training time (`conf/model/hppnet_pyrseplstm_k64.yaml`). Zeroing taps k ≥ 61 in the trained k ≤ 84 checkpoint at eval took `dregon` 4.18 → 3.30 (DREGON cruise 3.43 → 2.29, the best DREGON row on the panel) but cost MD2 (1.31 → 1.79), because the model had trained its MD2 reading on those taps. Training without them tests whether the DREGON gain is reachable with MD2 kept — the honest version of the knock-out diagnostic, on the stream where the knock-out helps.

## Conclusion

Defined 2026-10-18; not yet run.
