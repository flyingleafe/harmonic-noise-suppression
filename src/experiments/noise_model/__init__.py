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
"""

from __future__ import annotations

FIT_SCHEMA = "noise-v2-fit/2"
"""The fit-JSON schema tag every writer in this package stamps.

``/2`` replaced the four ``*_eps`` scalars and ``p`` of ``/1`` by the
``gamma_hz`` block; :func:`.model.gamma_from_params` still reads ``/1``
payloads by mapping their per-order OU onto an equivalent width, so old fits
render and can be frozen into a new one.
"""

READABLE_FIT_SCHEMAS: tuple[str, ...] = (FIT_SCHEMA, "noise-v2-fit/1")
"""Every fit-JSON schema the readers of this campaign accept.

This round's and R1/R2's: a ``/1`` payload's per-order OU is mapped onto an
equivalent width by :func:`.model.gamma_from_params`, so the renderer, the
regime study and the round score read both. One tuple so a new schema tag is
accepted in one place instead of five.
"""
