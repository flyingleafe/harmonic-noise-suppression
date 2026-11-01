---
experiment: rp_pyrseplstm_k64_stochcomb
training_config: conf/experiment/rp_pyrseplstm_k64_stochcomb.yaml
batch: docs/experiments/rp-synthetic.md
---

# `rp_pyrseplstm_k64_stochcomb`

## Motivation

The stochastic-comb row of the simplistic-synthetic ablation (paper Table VIII; Dmitrii, 2026-11-01). `rp_pyrseplstm_k64_combdiv` with the diverse static comb replaced by the stochastic rotor-noise family at its module defaults (`conf/online_mix/rp_stochcomb.yaml`). That family has Lorentzian lines on a smooth coloured floor, and every amplitude drifts as a Gaussian process independent of the trajectory. Trajectories, level law, render reuse and everything outside the noise source are the static comb's, so the two rows differ only in the audio. Validation: the real panel plus 512 pre-drawn stochcomb items (`conf/validation/rp_real_stochcomb.yaml`); the run stops when neither `real_macro` nor `stoch_comb` improves its 5-round median. Read against `rp_pyrseplstm_k64_combdiv` (`real_macro` 16.20 at the synthetic-selected checkpoint) and the hard setup (6.15).

## Conclusion

Pending.
