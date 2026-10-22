---
experiment: rp_pyrseplstm_k64_all
training_config: conf/experiment/rp_pyrseplstm_k64_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_pyrseplstm_k64_all`

## Motivation

The `all`-stream twin of `rp_pyrseplstm_k64_alllab` (Dmitrii, 2026-10-22): pyramid + separable ConvLSTM, no `block_4`/`block_5`, instance norm, taps k ≤ 64 at training time (`conf/model/hppnet_pyrseplstm_k64.yaml`), on `rp_all` (no label-altering augmentation). The labelaug arm reached real_macro 2.52 (best of the rp grid at the time); this run completes its with/without-augmentation pair and is read against `rp_pyrseplstm_all` (k ≤ 84, 3.01) for the harmonic-range effect on the plain stream.

## Conclusion

Defined 2026-10-22; not yet run.
