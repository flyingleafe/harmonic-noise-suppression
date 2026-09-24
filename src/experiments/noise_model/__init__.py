"""Noise model v2 — the Pyro rotor-noise model of the 2026-09 campaign.

A SANDBOX package (it may import anything under ``src/``) that owns the v2
parameterisation, its expected-periodogram forward model, the Pyro model and
guide, the MAP fit and the exact discrete-time renderer. It does NOT change
:mod:`experiments.stochastic_fit`: the C2/C4 code is imported for the pieces
whose math is unchanged (the finite-window kernel, the integrated-OU structure
function, the coloured floor, the composite risk and its exposure weights) and
never forked.

What v2 changes against C4 (``docs/explainers/noise-model-v2-plan.qmd``
section "What changes against C4"):

* the shaft speed error keeps its one OU process, but ``(sigma_nu, lam)`` are
  free with priors measured on the bench and in the telemetry;
* the flat per-harmonic Wiener ``D`` is replaced by ONE free Lorentzian
  half-width ``gamma_rk`` (Hz) per rotor and order — Model R3, which retired
  R1's per-order OU pair ``(sigma_eps, lam_eps)``, its exponent ``p`` and the
  never-fitted path term (:mod:`.lag`);
* DREGON is fitted on the single-motor bench, on the whole-segment
  periodogram, with the carrier FROZEN at the support index's window-refined
  value (:mod:`.spectrum` BENCH mode) instead of in free flight.

Modules: :mod:`.supports` (the fit supports and their cache), :mod:`.lag` (the
residual-phase autocorrelation), :mod:`.spectrum` (the expected periodogram),
:mod:`.model` (the Pyro model), :mod:`.fit` (MAP) and :mod:`.render` (the
exact discrete-time renderer).

NOISE MODEL v3 (``docs/explainers/noise-model-v3-wander.qmd``) is the fit mode
``flight_v3`` of the same stack: :class:`.model.PriorsV3` (half-normal linear
width/shaft priors, one profile prior per order, the floor as the spline alone
with a measured ``sigma_B``, no mic sites — the channels are normalised in the
data by :func:`.fit.load_channel_gains` — and DREGON's per-mic static wind
term), per-window per-block OU wander latents with MEASURED fixed
hyperparameters (:class:`data_processing.noise_model.v3.Wander`, read from
``results/noise_v3/wander/<rig>.json``), and the §3.4 alternation
:func:`.fit.fit_v3`. It writes ``noise-v3-fit/1``, which
:func:`.render.render_noise` renders with FRESH wander per clip. The v2 modes
are unchanged.

WHAT MOVED DOWN. The renderer and the primitives it stands on now live in
:mod:`data_processing.noise_model`, so an online-mixing noise source can
synthesise fitted rotor noise without importing :mod:`experiments`
(import-linter: "nothing imports experiments"). :mod:`.lag`, :mod:`.render`,
:mod:`.spectrum` and the schema tags below re-export those names unchanged;
the Pyro model, the MAP fit and the expected-periodogram forward model stay
here.
"""

from __future__ import annotations

from data_processing.noise_model import FIT_SCHEMA, FIT_SCHEMA_V3, READABLE_FIT_SCHEMAS

__all__ = ["FIT_SCHEMA", "FIT_SCHEMA_V3", "READABLE_FIT_SCHEMAS"]
