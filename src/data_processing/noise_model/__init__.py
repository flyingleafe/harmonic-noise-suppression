"""The rotor-noise model v2/v3/v4 — the parts a DATA STREAM needs.

This package holds the renderer of the 2026-09 noise-model-v2 campaign and the
handful of primitives it stands on, so that an online-mixing noise source can
synthesise fitted rotor noise without importing :mod:`experiments` (which the
import-linter contract "nothing imports experiments" forbids, and which would
drag Pyro into every training job).

WHAT LIVES HERE, and what does not:

* :mod:`.constants`, :mod:`.ou`, :mod:`.floor`, :mod:`.resample` — the shared
  primitives of the C4/v2 model: the exact integrated-OU transition and its
  path simulator, the coloured floor's geometry and power spectrum, and the
  render anti-alias/decimation chain. They were MOVED here from
  :mod:`experiments.stochastic_fit`, which imports them back, so the fit and
  the renderer still read one definition of each law.
* :mod:`.lag`, :mod:`.spectrum`, :mod:`.params`, :mod:`.render` — the v2
  residual-phase autocorrelation, the geometry the renderer reads, the fit-JSON
  params reader and the exact discrete-time renderer.
  :mod:`experiments.noise_model` re-exports every one of them.
* :mod:`.v3` — what noise model v3 adds to a render: the measured block-wander
  hyperparameters (:class:`.v3.Wander`), their stationary OU block draw and the
  fixed per-mic wind shape. :func:`.render.render_noise` reads a
  ``noise-v3-fit/1`` payload with them.
* :mod:`.v4` — the reader of a ``noise-v4-fit/3`` payload
  (``docs/explainers/noise-model-v4.qmd``): line powers, pedestal, shaft law,
  per-rotor floor, microphone response, the flight wind block (static
  Gaussian, or the SC gust generator of :mod:`.wind_sc` per capsule), and the
  constant-speed expected periodogram of a render. :func:`.render.render_noise`
  draws v4 payloads through it.
* NOT here: the Pyro model, the MAP fit, the supports, the moving-frame
  expected-periodogram forward model. Those stay in
  :mod:`experiments.noise_model`; nothing a training stream does needs them.

:mod:`data_processing.noise_v2_pool` is the consumer: the ``noise_v2`` online
mixing source.
"""

from __future__ import annotations

FIT_SCHEMA = "noise-v2-fit/2"
"""The fit-JSON schema tag every writer of the v2 campaign stamps.

``/2`` replaced the four ``*_eps`` scalars and ``p`` of ``/1`` by the
``gamma_hz`` block; :func:`.params.gamma_from_params` still reads ``/1``
payloads by mapping their per-order OU onto an equivalent width, so old fits
render and can be frozen into a new one.
"""

FIT_SCHEMA_V3 = "noise-v3-fit/1"
"""The fit-JSON schema tag of noise model v3 (``flight_v3``).

Its ``params`` carry no microphone block (the channels are normalised in the
data), the floor as the spline alone (``floor_shape_sd_db`` = the measured
``sigma_B``), an optional per-mic ``wind`` level and the measured ``wander``
hyperparameters the renderer draws fresh block latents from (:mod:`.v3`).
"""

READABLE_FIT_SCHEMAS: tuple[str, ...] = (FIT_SCHEMA, "noise-v2-fit/1", FIT_SCHEMA_V3)
"""Every fit-JSON schema the readers of this campaign accept.

This round's and R1/R2's: a ``/1`` payload's per-order OU is mapped onto an
equivalent width by :func:`.params.gamma_from_params`, so the renderer, the
regime study and the round score read both; and v3's, which
:func:`.render.render_noise` draws with fresh wander latents. One tuple so a
new schema tag is accepted in one place instead of five.
"""

FIT_SCHEMA_V4 = "noise-v4-fit/3"
"""The fit-JSON schema tag of noise model v4: ONE schema for the DREGON bench
fit and the MD2 flight fit (``docs/explainers/noise-model-v4.qmd``).

Its ``params`` are read by :mod:`.v4` only. It is NOT in
:data:`READABLE_FIT_SCHEMAS`: the v2/v3 readers (:func:`.params.check_schema`,
the identifiability gate, the ``noise_v2`` bank checks) read keys a v4 payload
does not carry, and must refuse it by its tag rather than fail on a missing key.
:data:`.render.READABLE_SCHEMAS` lists it: :func:`.render.render_noise` draws it.
"""
