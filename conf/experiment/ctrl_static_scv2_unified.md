---
experiment: ctrl_static_scv2_unified
training_config: conf/experiment/ctrl_static_scv2_unified.yaml
batch: docs/experiments/stochastic-fit.md
---

## Motivation

C1, the simple control for the whole fitting campaign. Everything outside the
noise family is byte-identical to `rig_fitted_scv2_unified` — clip length,
batch, panel, stopping rule, speech corpus, SNR prior, augmentation block,
silence arm, trajectory generator, even the renderer — so a difference in
transfer is attributable to the family and to nothing else.

The family is **static**: amplitudes frozen for the clip, no shaft wander, no
line diffusion, no floor breathing. It is not naive in its timbre, which is the
part that makes it a fair control rather than a strawman. Roll-off, blade
count, blade-pass emphasis, a boost on the blade-passing fundamental alone,
per-order irregularity and a rotor-to-rotor spread are all sampled wide
(`src/experiments/stochastic_fit/control_streams.py` documents every range),
and two further arms render the fitted DREGON and Michael's profiles themselves
with the dynamics switched off — so the family provably contains the measured
timbres, and a failure cannot be blamed on never having drawn the right comb.

The measured per-rotor shaft-minus-label offset (0.4-1.4 rev/s) stays on. It is
a property of the telemetry rather than of the noise family, and removing it
would hand this control a label precision no real recording carries.

What it decides: whether the stochastic apparatus is load bearing. If a static
comb transfers to real data, the apparatus is unnecessary and that is the best
available outcome. The campaign's claim is that it will not, because a real
high-order comb is broadened and amplitude modulated, and a model trained on
razor-thin static lines reads a cue real audio does not carry.

## Conclusion

Pending — launched alongside `ctrl_diverse_scv2_unified`.
