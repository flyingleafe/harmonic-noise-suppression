---
experiment: rp_marcoux_grondin_all
training_config: conf/experiment/rp_marcoux_grondin_all.yaml
batch: docs/experiments/paper-real-protocol.md
---

# `rp_marcoux_grondin_all`

## Motivation

Table IV row (no label augmentation) for the BPF estimator of Marcoux & Grondin (ICRA 2026, arXiv:2602.09991): mel + power-cepstrum front end, four parallel CNN blocks, separate 3-layer BiGRU speed and activity branches. Adapted to the protocol's four-rotor output (`marcoux_grondin_rps4`): four unsorted tracks trained with clip-level PIT in BPF Hz, activity BCE on "any rotor turns", speeds multiplied by the 0/1 activity decision. Trained on the `rp_all` stream; compared against `rp_marcoux_grondin_alllab` and the other Table IV rows. Shared settings: up to 100k steps, 5k warm-up + cosine LR, rp_real panel every 500 steps, best_<score>_{median,raw} checkpoints.

## Conclusion

Defined 2026-11-26; submitted to `isambard`. Max LR 1e-3: the paper states none, so the regressor rows' value.
