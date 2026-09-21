"""The rotor-noise model v2 — the parts a DATA STREAM needs.

This package holds the primitives the v2 renderer stands on, so that an
online-mixing noise source can synthesise fitted rotor noise without importing
:mod:`experiments` (which the import-linter contract "nothing imports
experiments" forbids, and which would drag Pyro into every training job).

:mod:`.constants`, :mod:`.ou`, :mod:`.floor` and :mod:`.resample` are the
shared primitives of the C4/v2 model: the exact integrated-OU transition and
its path simulator, the coloured floor's geometry and power spectrum, and the
render anti-alias/decimation chain. They were MOVED here from
:mod:`experiments.stochastic_fit`, which imports them back, so the fit and the
renderer still read one definition of each law.
"""

from __future__ import annotations

FIT_SCHEMA = "noise-v2-fit/2"
"""The fit-JSON schema tag every writer of the v2 campaign stamps.

``/2`` replaced the four ``*_eps`` scalars and ``p`` of ``/1`` by the
``gamma_hz`` block; :func:`.params.gamma_from_params` still reads ``/1``
payloads by mapping their per-order OU onto an equivalent width, so old fits
render and can be frozen into a new one.
"""

READABLE_FIT_SCHEMAS: tuple[str, ...] = (FIT_SCHEMA, "noise-v2-fit/1")
"""Every fit-JSON schema the readers of this campaign accept.

This round's and R1/R2's: a ``/1`` payload's per-order OU is mapped onto an
equivalent width by :func:`.params.gamma_from_params`, so the renderer, the
regime study and the round score read both. One tuple so a new schema tag is
accepted in one place instead of five.
"""
