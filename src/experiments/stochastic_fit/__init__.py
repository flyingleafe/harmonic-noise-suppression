"""Fit the stochastic rotor-noise family to REAL recordings, RPS held fixed.

Contract-fenced research sandbox (``src/experiments``): may import anything,
nothing imports it.

The question. Models trained on the stochastic synthetic family
(:mod:`data_processing.stochastic_rotor_noise`) do not transfer to real drone
audio unless real audio is in the training set. Is that because the family's
*parameter ranges* are off, or because the family is *structurally* a bad model
of real rotor noise? The two have different fixes (retune the sampler vs.
change the model), and neither has been measured: every prior calibration was
a range on a summary statistic, never a fit of the model to a recording.

The method. Maximum-a-posteriori fit of the family's own spectral model to one
recording's multichannel periodogram under a Whittle likelihood, with the
rotor-speed trajectories fixed to the best available reference::

    M_m(f, t) = B(f, t) + sum_r g_mr sum_k P_rk(t) L(f - k r_r(t); gamma_rk)
    -log p(I | M) = sum I / M + log M            (I: Hann-2048/512 periodogram)

with the family's exact parametrization: ``10 log10 P_rk(t) = profile_rk +
h_rk(t)`` (static timbre + squared-exponential GP drift), ``gamma_rk = gamma0_r
+ slope_r k``, ``B`` a smooth log-frequency shape times slow level/tilt GPs and
the speed law ``(r/80)^2.5``, per-microphone line gains and a common floor.
:mod:`.model` is that model in PyTorch; :mod:`.fit` fits it in stages
(floor-only, then a harmonic ladder) and evaluates the references that turn a
likelihood into an *explained fraction*: the floor-only fit below and a
leave-one-out nonparametric periodogram smoother above. :mod:`.diagnostics`
then reads the residual field ``I / M`` along the fitted lines to say *where*
the family fails (line shape, width law, amplitude pattern, temporal drift,
floor, coherence), and the extended variants in :mod:`.model` (free width per
order, Gaussian lines, per-microphone floors, refined carriers) measure how
much each structural change buys.

Data: :mod:`.clips` is the only source of real audio — the published
``*-frames`` datasets at their native 44.1 kHz, with the refined rotor-speed
label held fixed.

Entry points: ``python scripts/stochastic_fit.py --help`` (one fit, regime
selected: ``cruise``, ``standby``, ``bench``) and
``python -m experiments.stochastic_fit.run --help`` (the population/gate/export
pipeline).
"""

from __future__ import annotations
