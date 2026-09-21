"""Constants shared by the rotor-noise forward model and its renderer.

MOVED here from :mod:`experiments.stochastic_fit.model` (the floor's geometry
constants and the speed-law reference) and
:mod:`experiments.stochastic_fit.revised_phase` (the speed floor and the floor
shape prior's scale and correlation length), which import them back, so there
is exactly one definition of each. Values unchanged.
"""

from __future__ import annotations

#: Reference rotor speed of the amplitude speed law, ``(rps / AMP_RPS_REF) ** amp_exp``.
AMP_RPS_REF = 80.0

#: Lowest frequency the floor's shape is parameterised at; below it the shape
#: clamps to the first control point.
FLOOR_SHAPE_F_MIN = 30.0

#: Reference frequency of the floor's tilt term, in Hz.
FLOOR_TILT_REF_HZ = 500.0

#: Number of control points of the floor's log-frequency shape.
FLOOR_SHAPE_N_CTRL = 14

#: ONE positive speed floor, shared by the prediction and the renderer. The
#: speed laws carry FITTED exponents, so a stopped rotor at exactly 0 raises 0
#: to an unconstrained power: non-finite audio on one path, a finite prediction
#: on the other. Both paths clamp here instead. It is a constant, not a knob.
SPEED_FLOOR_RPS = 1e-6

#: Floor-shape prior scale (dB) and correlation length (octaves). These are
#: also :class:`experiments.stochastic_fit.model.Spec`'s defaults for
#: ``floor_shape_std_db`` / ``floor_shape_oct``, which read them from here.
FLOOR_SHAPE_STD_DB = 5.0
FLOOR_SHAPE_OCT = 1.5
