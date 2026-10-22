---
experiment: rp_pyrseplstm_k64_all
training_config: conf/experiment/rp_pyrseplstm_k64_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_pyrseplstm_k64_all`

## Motivation

The `all`-stream twin of `rp_pyrseplstm_k64_alllab` (Dmitrii, 2026-10-22): pyramid + separable ConvLSTM, no `block_4`/`block_5`, instance norm, taps k ≤ 64 at training time (`conf/model/hppnet_pyrseplstm_k64.yaml`), on `rp_all` (no label-altering augmentation). The labelaug arm reached real_macro 2.52 (best of the rp grid at the time); this run completes its with/without-augmentation pair and is read against `rp_pyrseplstm_all` (k ≤ 84, 3.01) for the harmonic-range effect on the plain stream.

## Conclusion

Job `rp-pyrseplstm-k64-all-d5d991` (vast A100-80, 2026-10-22, commit `459a84d6`). Best `real_macro` median **2.81** at step 5 000 (round 9); saturation stop at step 25 000 (round 50, `min_rounds`), the lowest single round 2.57. At the selected round: dregon 3.50, md2 2.12; pooled zero 4.56, standby 5.45, ramp 10.78, cruise 1.30. Against the k ≤ 84 twin `rp_pyrseplstm_all` (3.01) the harmonic cap helps on the plain stream too, but the labelaug arm (2.52, md2 1.14) stays ahead: the plain-stream run selects early and is weak at zero / standby and on MD2.
