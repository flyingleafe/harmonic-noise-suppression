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
* the flat per-harmonic Wiener ``D`` is replaced by an independent per-order
  OU ON THE PHASE driven ``propto k^{p/2}`` with ``p = 1`` fixed and one
  ``(sigma_eps, lam_eps)`` pair per PARITY of the order (:mod:`.lag`);
* DREGON is fitted on the single-motor bench, on the whole-segment
  periodogram, with the carrier a constant fit parameter (:mod:`.spectrum`
  BENCH mode) instead of in free flight;
* the per-microphone path phase term is NAMED and not fitted in R1.

Modules: :mod:`.supports` (the fit supports and their cache), :mod:`.lag` (the
residual-phase autocorrelation), :mod:`.spectrum` (the expected periodogram),
:mod:`.model` (the Pyro model), :mod:`.fit` (MAP) and :mod:`.render` (the
exact discrete-time renderer).
"""

from __future__ import annotations

FIT_SCHEMA = "noise-v2-fit/1"
"""The fit-JSON schema tag every writer in this package stamps."""
