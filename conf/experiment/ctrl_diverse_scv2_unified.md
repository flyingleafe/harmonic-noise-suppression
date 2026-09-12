---
experiment: ctrl_diverse_scv2_unified
training_config: conf/experiment/ctrl_diverse_scv2_unified.yaml
batch: docs/experiments/stochastic-fit.md
---

## Motivation

C2, which is not a control but a hypothesis test. Same model and same regime as
`rig_fitted_scv2_unified`, with the parameters sampled over a wide space
instead of the tight neighbourhood of two fitted rigs. Only what the
measurements actually pin is held fixed: the line-width bracket (bench shaft
jitter 0.02 rev/s to flight 0.5), the amplitude process (OU, std 2.90 dB,
tau 0.82 s, cross-order coherence 0.191), the bench speed laws (line
2.196 +- 0.125, floor 1.400 +- 0.076) and the label error. Timbre is wide.

Hypothesis: a model forced to disentangle a broad distribution learns the
disentangling operation rather than one rig's fingerprint, and therefore
transfers if the family is a faithful description of real rotor noise.

The failure mode to avoid is the original stochastic-comb stream, which was
diverse in the wrong way — it occupied regions no real rig does (lines wide
enough to merge the comb, floors that buried every order) and was simply a
harder, different task. Every range here is anchored to a measurement, and two
guards keep a draw learnable: the per-rotor floor-coverage guard, so no rotor's
speed is a label with no evidence in the clip, and a width prior whose upper
end is the widest the flight fits ever asked for.

## Conclusion

Pending — launched alongside `ctrl_static_scv2_unified`.
