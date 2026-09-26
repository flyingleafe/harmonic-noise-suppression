"""Parameter-neighbourhood sampler for noise-model-v2 (and v3) fits.

What this is
------------
Given the campaign's fitted v2 rigs — DREGON's room-2 flight profile (round 5)
and Michael's FLY125 cruise/standby (round 3) — this module draws perturbed
``noise-v2-fit/2`` payloads that are plausible OTHER drones of the same kind.
Every result is a complete fit payload, renderable by
:func:`data_processing.noise_model.render.render_noise` and loadable by
:class:`data_processing.noise_v2_pool.NoiseV2Pool` unchanged, so a bank of them
is a training stream (``scripts/noise_v2_build_bank.py``). Two modes:

* :func:`sample_rig` — a NEIGHBOURHOOD of one fit at width ``strength``. The
  easy bank: 1024 draws per rig, each carrying its own rig's ``traj_rig``.
* :func:`sample_path` — a CLOUD along the path between the two CRUISE fits,
  mixing coordinate ``t ~ U[0, 1]``. The hard bank: 2048 draws, ``traj_rig``
  null so the policy's own trajectory mixture (the rig hyperprior) flies them.

This is a generator of varied synthetic rigs. It is NOT campaign scoring and it
makes no gate claim: nothing here says a sampled rig would pass any acceptance
probe, only that it satisfies the stated plausibility guards.

This is the v2 twin of :mod:`experiments.stochastic_fit.rig_sampler`, which
samples the PRE-REVISION stage-2 family. The DESIGN is taken from there — the
profile decomposition, the correlated residual redraw, reject-and-redraw
guards, the two modes, the measured-width discipline — and the 1/3-octave band
helper :func:`~experiments.stochastic_fit.rig_sampler.third_octave` is imported
from it so both samplers measure a spectrum the same way. None of its
PARAMETRISATION is reused: the v2 model has no ``coherence_k_half``, no
``gamma0``/``gamma_slope`` pair, no per-clip level scale, and its profile
carries no separate parity envelope (see below).

The profile decomposition
-------------------------
Per rotor, over orders ``k = 1..K`` with ``x = log10 k``:

    profile_db[r, k] = gain_r + slope_r * (x - mean x) + resid_r(k)

with the order axis centred, so gain and slope are orthogonal and perturbing
the slope does not move the level. The legacy sampler carries a fourth term,
an even/odd PARITY envelope, because its residual redraw is large enough to
destroy the blade-passing split. This one does not, and deliberately: the
measured squared-exponential correlation length of the v2 residual in ``log10``
order is 0.015 decades (variogram, median over the 16 measured rotor profiles),
i.e. the residual is essentially WHITE in order — a positive kernel cannot
represent the alternating component at all, which is why the variogram collapses
to the grid edge. The parity split therefore lives INSIDE ``resid_r`` and is
preserved by the redraw being a MIXTURE: ``resid' = rho resid + sqrt(1 - rho^2)
sigma z`` with ``rho = 1 - r^2 / 2``, ``r = strength * resid_redraw_frac``. At
the strength this campaign builds at, ``rho >= 0.97``, so an entry keeps its
anchor's parity structure to within a few percent while its fine structure
still moves. That is measured and reported in ``structure.json``; no parity
guard is needed and none is claimed.

The three tiers, and what ``strength`` means
--------------------------------------------
Every width is measured in :func:`measure_structure` (phase A) and cached in
``results/noise_v2/rig_sampler/structure.json``; the frozen numbers in
:class:`Widths` are that file's ``widths_strength1`` block, and ``main()``
re-measures and prints frozen against measured so the two cannot drift apart.

* **between-restart** — the SAME fit, the same data, a different optimiser
  seed (R5 restarts ``s0..s3``; the DREGON bench ``Motor{1..4}_70`` restarts).
  This is the ``strength = 1`` target. It is fit reproducibility, not rig
  variability, and it is TIGHT: 0.36 dB on a rotor's profile gain, 0.038 dB on
  the floor level.
* **between-rotor** — the four DREGON bench motors at throttle 70, four
  separate single-rotor fits of four separate recordings of the same airframe's
  four rotors.
* **between-rig** — DREGON room-2 cruise against Michael's FLY125 cruise, on
  their common order range. Reported as ``|rig mean difference| / sqrt(2)``,
  the per-draw sigma two independent draws would need to differ by that much.

The ladder (``structure.json:ladder``) is ``tier / between-restart`` per
coordinate, and its headline is that ONE scalar cannot walk all coordinates up
the tiers together: the between-rotor spread is reached at strength 4.7 on a
rotor's profile gain, 4.1 on the trend slope, 4.1 on the line widths and 41 on
the floor level, because a 255-frame flight fit reproduces its floor level to
0.04 dB across restarts while four real rotors differ by 1.5 dB. The sampler
does not paper over this. It adds ONE coordinate whose width is measured on a
different tier and says so:

* **the level coordinate** (:attr:`Widths.level_db`) moves the whole rig —
  every rotor's ``profile_db`` and ``floor_mean_db`` together — and its
  ``strength = 1`` width is the BETWEEN-CLIP standard deviation of the
  anchors' own real windows' band level (300 Hz - 7.9 kHz): 2.01 dB over
  DREGON's ten frozen room-2 windows, 1.16 dB over Michael's eight cruise
  windows, 0.44 dB over Michael's three standby windows, 1.64 dB pooled. The
  between-restart spread of the same quantity is 0.038 dB, which is not a
  width but a convergence tolerance: used as one it would clone every entry of
  the bank at one level, which is precisely the point-ness the transfer pair
  exists to break. This is the only coordinate whose tier differs, it is
  labelled ``between_clip`` in ``structure.json``, and it is the coordinate the
  coverage check below turns out to depend on.

What is NOT drawn
-----------------
The speed laws. ``amp_exp``, ``floor_exp`` and ``floor_static_rel`` are PINNED
on every entry to the v2 short-span contract — 2.0, 2.0, 2.5e-3, the model's
own prior medians (:data:`SPAN_PIN`) — and :func:`pin_speed_laws` records the
anchor's pre-pin values in ``params.span_pin_record``. Reason: a cruise-only
pool spans too little carrier to identify them (the fit itself pins them when
the span is under 1.5, which is why Michael's cruise fit already carries
exactly these three values), and R5's fitted ``amp_exp = -0.87`` /
``floor_exp = 15.32`` are that non-identifiability showing. The pin is applied
at the 80 rev/s reference, where every speed factor is exactly 1, so a cruise
profile is untouched by it; a standby payload IS moved by it, at its own
20-40 rev/s (Michael's standby fit's own ``floor_exp = 4.93`` becomes 2.0,
raising its floor about 12 dB at 30 rev/s), and that is recorded per entry
rather than hidden. A negative sampled exponent is impossible by construction
and :func:`check_sample` refuses one anyway.

Guards
------
:func:`check_sample` returns every guard result and both modes apply it
internally, redrawing up to ``max_attempts`` times and raising
:class:`SampleRejected` rather than returning an invalid entry. Nothing is ever
clipped into shape.

* ``finite`` — the v2 EXPECTED periodogram (``experiments.noise_model.render.
  expected_periodogram``, the fit's own forward model, not a render) is finite
  and strictly positive at the cruise probe (80 rev/s, :data:`PROBE_CRUISE_RPS`)
  and at the idle probe (30 rev/s, :data:`PROBE_IDLE_RPS`). The idle probe is
  evaluated on the entry's STANDBY payload when it has one, because that is the
  payload the regime composition renders near idle. Every frame of a
  constant-carrier window has the same expectation, so one frame of the
  declared 2 s window is computed and not 59 identical copies (verified to
  0.0 dB against the full window).
* ``trend_falls`` — each rotor's fitted trend must still FALL by at least
  :data:`TREND_MARGIN_DB` = 3 dB from ``k = 1`` to ``k = K``. Measured: the
  smallest total trend drop over the 16 measured v2 rotor profiles is 9.7 dB.
* ``gamma_excursion`` — the per-rotor line-width level may move by at most
  :data:`GAMMA_EXCURSION_CAP` = 1.5x in either direction. Stated, not measured:
  it keeps a sampled width inside the anchor's own decade, well short of the
  measured between-rotor spread (1.28x per rotor) and of the 1.01x between-rig
  level ratio's per-ORDER scatter (3.42 in log).
* ``ltas`` — the draw's expected periodogram, in 1/3-octave bands over
  :data:`LEVEL_BAND_HZ` (300 Hz - 7.9 kHz), within ``LTAS_ENVELOPE_X = 2`` times
  the measured REAL clip-to-clip band RMS of the anchor's own real windows:
  2.66 dB over DREGON's ten frozen room-2 windows and 2.33 dB over Michael's
  eight cruise windows, so the tolerances are 5.32 and 4.65 dB. The reference
  is the ANCHOR for :func:`sample_rig` and the INTERPOLATED POINT for
  :func:`sample_path`; on the path the tolerance is interpolated in ``t`` with
  everything else. The envelope is strength-INDEPENDENT on purpose: one that
  widened with the request could never refuse anything. Below 300 Hz the band
  is not guarded — one 1/3-octave band there holds one or two rotor orders, so
  the per-order freedom this sampler exists to create necessarily moves
  individual low bands by tens of dB.
* ``speed_law`` — ``amp_exp >= 0`` and ``floor_exp >= 0``. A negative exponent
  is ``0 ** negative`` at the exact zero of a full flight's ground phase, i.e.
  infinite line power and NaN audio. Pinned entries cannot fail it; it is
  checked because a caller may hand in an unpinned anchor.

Coverage: how ``strength`` is chosen
------------------------------------
Not by taste. :func:`coverage` measures, per rig, the fraction of 1/3-octave
bands in which the CLOUD (the min-to-max envelope of the accepted draws'
expected periodograms) brackets each real window's measured band level — the
same criterion the legacy transfer pair used
(``docs/experiments/rig-sampler-transfer-pair.md``). ``strength = 2.0`` is kept
unless coverage above 300 Hz falls under 90 %, in which case the ladder
``LADDER_STRENGTHS`` is walked and the first setting that reaches it is used.
The measured answer for this pair is in ``results/noise_v2/rig_sampler/
findings.md``.

Noise model v3 (``generation="v3"``)
------------------------------------
The same two modes, seed, strength, widths, guards and LTAS envelope, drawn
around the round-2 v3 fits with their static latent part folded into the rig
(:data:`ANCHORS_V3`). A ``noise-v3-fit/1`` payload shares ``profile_db``,
``gamma_hz``, ``sigma_nu``, ``lam``, ``floor_mean_db`` and the speed laws with
v2, and those are drawn and interpolated exactly as above. The rest maps like
this:

* **floor shape and tilt** — v3's floor is the spline alone,
  ``mu + sigma_B (L z)_j`` with the MEASURED ``sigma_B``. v2 draws a shape
  step in its GP coordinate (scale ``FLOOR_SHAPE_STD_DB`` = 5 dB) and a tilt
  step in dB/octave. v3 takes the same two draws and writes their exact dB
  move at every control point into ``z`` (:func:`_v3_floor_step`). On the path
  the control values are interpolated linearly in dB, which is v2's rule.
* **microphone blocks** — v3 has none, because it normalises the channels in
  the data, so nothing is drawn there. DREGON's per-mic ``wind`` level takes
  the whole-rig level move plus v2's per-mic floor width (``mic_floor_db``).
  On the path it mixes linearly in power towards Michael's absent term.
* **wander** — the ``(sigma, tau)`` hyperparameters are NOT perturbed; every
  neighbourhood entry keeps its anchor's fitted block. On the path the block is
  CARRIED like the standby slot: Michael's cruise block with probability ``t``,
  DREGON's otherwise (:data:`PATH_INTERP_V3`).
* **trend guard** — the folded Michael's cruise fit falls by only 0.6 and
  0.2 dB on rotors 2 and 3, so v2's absolute 3 dB rule would refuse the
  anchor itself. The v3 bound is ``min(3 dB, reference drop - 3 dB)``
  (:func:`trend_floor_db`): v2's rule on every rotor whose reference falls by
  at least 6 dB, and elsewhere no flattening by more than 3 dB below the fit.
* **probe** — each payload's expected periodogram is evaluated at its own work
  rate (32 kHz for v3), with the wander at its mean.

CLI
---
``PYTHONPATH=src python -m experiments.noise_model.rig_sampler measure`` runs
phase A over the fits and the real windows, writes ``structure.json`` and
prints the tier table plus a frozen-against-measured comparison.
``... ladder`` runs the strength ladder and writes the coverage block and the
figures.
"""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.noise_model import FIT_SCHEMA_V3
from data_processing.noise_model.constants import (
    AMP_RPS_REF,
    FLOOR_SHAPE_F_MIN,
    FLOOR_SHAPE_STD_DB,
    FLOOR_TILT_REF_HZ,
)
from data_processing.noise_model.spectrum import FLIGHT_SR, floor_ctrl_hz, floor_shape_chol
from data_processing.noise_v2_pool import PRESET_BANK_FORMAT
from experiments.stochastic_fit.rig_sampler import third_octave

__all__ = [
    "ANCHORS",
    "GAMMA_EXCURSION_CAP",
    "LEVEL_BAND_HZ",
    "LTAS_ENVELOPE_X",
    "PROBE_CRUISE_RPS",
    "PROBE_IDLE_RPS",
    "SPAN_PIN",
    "STRUCTURE_PATH",
    "TREND_MARGIN_DB",
    "ModelProbe",
    "SampleRejected",
    "Widths",
    "check_sample",
    "coverage",
    "decompose",
    "draw_fit",
    "interpolate_fits",
    "measure_structure",
    "pin_speed_laws",
    "sample_batch",
    "sample_path",
    "sample_rig",
    "strip_payload",
]

# ---------------------------------------------------------------------------
# provenance: the fits phase A measures
# ---------------------------------------------------------------------------

ROUND5 = "results/noise_v2/rounds/round5/fits"
ROUND3 = "results/noise_v2/rounds/round3/fits"

#: The two rigs' anchors. ``standby`` is ``None`` for a single-regime rig, and
#: DREGON has no standby fit (its room-2 recordings hold no standby segment).
ANCHORS: dict[str, dict[str, Any]] = {
    "dregon": {
        "cruise": f"{ROUND5}/dregon_room2_floor__flight_profile.json",
        "standby": None,
        "traj_rig": "dregon",
    },
    "michaels": {
        "cruise": f"{ROUND3}/michaels_fly125_cruise__flight.json",
        "standby": f"{ROUND3}/michaels_fly125_standby__flight.json",
        "traj_rig": "michaels",
    },
}

#: Round 2 of the noise-model-v3 campaign (``docs/experiments/noise-model-v3.md``
#: § "Round 2") with the STATIC part of its fitted block latents folded into the
#: rig (``scripts/noise_v3_diag.py fold_static``, commit ``1b698d78``): profile +=
#: the window-mean ``d`` and ``v``, floor mean += mean ``u``, ``z`` += ``L^-1``
#: mean ``u_j`` / sigma_B. An exact reparametrisation (the fit's block spectra
#: move by <= 8e-4 dB, no parameter added); without it the renderer, which
#: draws zero-mean wander, drops the static part and adds the prior's Jensen
#: term on top of it (DREGON +0.4..+1.8 dB above 300 Hz, Michael's cruise
#: +2.4..+3.5 dB, render minus fit; ``results/noise_v3/diag/ltas_bias_r2.json``).
ROUND_V3_R2 = "results/noise_v3/diag/folded_r2"

#: The v3 anchors: the same two rigs, the same regime pairs, their folded
#: round-2 v3 fits. Michael's standby is a v3 fit too, so a v3 entry is v3 in
#: both slots.
ANCHORS_V3: dict[str, dict[str, Any]] = {
    "dregon": {
        "cruise": f"{ROUND_V3_R2}/dregon_room2_floor__flight_v3.json",
        "standby": None,
        "traj_rig": "dregon",
    },
    "michaels": {
        "cruise": f"{ROUND_V3_R2}/michaels_fly125_cruise__flight_v3.json",
        "standby": f"{ROUND_V3_R2}/michaels_fly125_standby__flight_v3.json",
        "traj_rig": "michaels",
    },
}

#: The model generations a bank can be drawn in, and the anchors of each.
GENERATIONS: dict[str, dict[str, dict[str, Any]]] = {"v2": ANCHORS, "v3": ANCHORS_V3}


def anchors_of(generation: str) -> dict[str, dict[str, Any]]:
    """The anchor table of one model generation (``"v2"`` or ``"v3"``)."""
    if generation not in GENERATIONS:
        raise ValueError(f"generation must be one of {tuple(GENERATIONS)}, got {generation!r}")
    return GENERATIONS[generation]


def is_v3(fit: dict[str, Any]) -> bool:
    """Whether ``fit`` is a ``noise-v3-fit/1`` payload."""
    return fit.get("schema") == FIT_SCHEMA_V3


#: R5's four optimiser restarts: the between-restart tier for everything the
#: flight profile fit leaves free (profile, floor, microphones).
R5_RESTARTS = tuple(
    f"{ROUND5}/restarts/dregon_room2_floor__flight_profile__s{i}.json" for i in range(4)
)

#: The four DREGON bench motors at throttle 70 — the between-ROTOR tier — and
#: their own restarts, which carry the between-restart tier for the quantities
#: R5 FROZE from the bench (``gamma_hz``, ``sigma_nu``, ``lam``:
#: ``frozen_from.gamma_rotor_matched`` in the R5 payload).
BENCH_MOTORS = tuple(f"{ROUND3}/bench_dregon_Motor{i}_70__bench.json" for i in (1, 2, 3, 4))
BENCH_RESTARTS = {
    f"Motor{i}": tuple(
        f"{ROUND3}/restarts/bench_dregon_Motor{i}_70__bench__s{s}.json" for s in range(4)
    )
    for i in (1, 2, 3, 4)
}

STRUCTURE_PATH = Path("results/noise_v2/rig_sampler/structure.json")

# ---------------------------------------------------------------------------
# the pinned speed laws (campaign decision, see the module docstring)
# ---------------------------------------------------------------------------

#: ``experiments.noise_model.model``'s prior medians for the three speed-law
#: quantities, which the fit itself pins whenever a pool's carrier span is
#: under ``priors.speed_span_pin`` = 1.5. Michael's cruise fit carries exactly
#: these; every bank entry does too.
SPAN_PIN: dict[str, float] = {
    "amp_exp": 2.0,
    "floor_exp": 2.0,
    "floor_static_rel": 2.5e-3,
}

#: Cruise probe: the model's own amplitude reference speed, where every speed
#: factor is exactly 1. The two rigs' real cruise windows sit on it — the
#: rotor-mean carrier is 80.4 rev/s over DREGON's frozen room-2 windows and
#: 80.7 rev/s over Michael's eight cruise windows — so one constant-carrier
#: probe is representative of both and comparable across the path.
PROBE_CRUISE_RPS = AMP_RPS_REF
#: Idle probe, the divergent-floor trap. Michael's three standby windows run at
#: a 34.9 rev/s rotor mean and the standby regime band is 20-45 rev/s.
PROBE_IDLE_RPS = 30.0

#: The guarded band. Below 300 Hz a 1/3-octave band holds one or two rotor
#: orders, so it moves by tens of dB under exactly the per-order freedom this
#: sampler exists to create.
LEVEL_BAND_HZ = (300.0, 7900.0)

#: Each rotor's trend must fall by at least this much from ``k = 1`` to
#: ``k = K``. Measured: the smallest total drop over the 16 v2 rotor profiles
#: (2 cruise rigs x 4 rotors, Michael's standby x 4, 4 bench motors) is 9.7 dB.
TREND_MARGIN_DB = 3.0

#: The trend guard around a v3 reference, as recorded in a v3 bank's guards.
TREND_RULE_V3 = (
    "per rotor: drop >= min(TREND_MARGIN_DB, reference drop - TREND_MARGIN_DB), i.e. v2's rule "
    "wherever the reference falls by at least twice the margin; a standby payload keeps v2's "
    "absolute margin"
)

#: Total per-rotor line-width excursion cap. STATED, not measured.
GAMMA_EXCURSION_CAP = 1.5

#: The LTAS envelope multiple: a sampled rig may differ from its reference by
#: up to twice as much as two REAL windows of that rig differ from each other.
#: NOT scaled by strength.
LTAS_ENVELOPE_X = 2.0

#: The strengths the coverage ladder walks, in order. Stops at the first that
#: reaches :data:`COVERAGE_TARGET` above 300 Hz on every rig.
LADDER_STRENGTHS = (1.0, 2.0, 3.0, 4.0, 6.0)
COVERAGE_TARGET = 0.90

#: The path bank's common order range: DREGON's 88 orders against Michael's 81.
#: The extra orders are DROPPED rather than padded, because padding a profile
#: past its own fit's order support would invent data.
PATH_K_MAX = 81


class SampleRejected(RuntimeError):
    """No draw satisfied the guards within the attempt budget."""


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def git_head() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # pragma: no cover - a checkout without git
        return "unknown"


def canonical_digest(payload: Any) -> str:
    """SHA-256 of a JSON payload written canonically — the content digest."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    ).hexdigest()


def load_fit(path: Path | str) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def _round(a: Any, sig: int = 7) -> Any:
    """Arrays to nested lists at ``sig`` significant digits.

    A bank holds thousands of ``(R, K)`` blocks; full float64 repr doubles the
    file for digits no dB quantity carries. Deterministic, so a bank stays
    bit-reproducible.
    """
    arr = np.asarray(a, dtype=np.float64)
    if arr.ndim == 0:
        return float(f"{float(arr):.{sig}g}")
    return [_round(v, sig) for v in arr]


# ---------------------------------------------------------------------------
# the decomposition
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Parts:
    """One rotor's curve split into gain, log-order trend and residual.

    ``curve == gain + slope * (x - x.mean()) + resid`` exactly, so the sampler
    can perturb the parts and reassemble without drift. Used for ``profile_db``
    in dB and for ``log gamma_hz`` in nats, which have the same shape.
    """

    x: np.ndarray
    gain: float
    slope: float
    resid: np.ndarray

    @property
    def drop(self) -> float:
        """How far the trend falls from ``k = 1`` to ``k = K``. Positive = falls."""
        return float(-self.slope * (self.x[-1] - self.x[0]))

    @property
    def resid_std(self) -> float:
        return float(self.resid.std(ddof=2))


def order_axis(n: int) -> np.ndarray:
    """``log10 k`` for ``k = 1..n``."""
    return np.log10(np.arange(1, int(n) + 1, dtype=np.float64))


def decompose(curve: np.ndarray) -> Parts:
    """Split one rotor's ``(K,)`` curve into gain, trend and residual."""
    p = np.asarray(curve, dtype=np.float64).ravel()
    x = order_axis(p.size)
    design = np.stack([np.ones(p.size), x - x.mean()], axis=1)
    coef, *_ = np.linalg.lstsq(design, p, rcond=None)
    return Parts(x=x, gain=float(coef[0]), slope=float(coef[1]), resid=p - design @ coef)


def decompose_block(block: np.ndarray) -> list[Parts]:
    """:func:`decompose` per row of an ``(R, K)`` block."""
    return [decompose(row) for row in np.atleast_2d(np.asarray(block, dtype=np.float64))]


def variogram_length(resid: np.ndarray, x: np.ndarray, *, max_lag: float = 0.3) -> float:
    """Squared-exponential correlation length of ``resid`` in decades of order.

    The short-lag variogram ``E(dr)^2 = 2 sigma^2 (1 - rho(h))`` rather than the
    autocorrelation, because the empirical ACF of ONE detrended realisation is
    driven negative by the detrending and no positive kernel can fit that.
    """
    r = np.asarray(resid, dtype=np.float64)
    iu = np.triu_indices(r.size, 1)
    lag = np.abs(x[:, None] - x[None, :])[iu]
    d2 = ((r[:, None] - r[None, :]) ** 2)[iu]
    near = lag <= float(max_lag)
    if near.sum() < 8:
        return float("nan")
    grid = np.geomspace(0.005, 1.0, 400)
    var = float(np.var(r))
    pred = 2.0 * var * (1.0 - np.exp(-0.5 * (lag[near][None, :] / grid[:, None]) ** 2))
    err = ((pred - d2[near][None, :]) ** 2).mean(axis=1)
    return float(grid[int(np.argmin(err))])


# ---------------------------------------------------------------------------
# the payload: pinning, stripping, rebuilding
# ---------------------------------------------------------------------------

#: What a bank entry keeps of a fit payload. The fit's ``objective``,
#: ``optimiser``, ``diagnostics``, ``priors`` and ``restarts`` blocks are
#: per-fit bookkeeping; carried 2048 times they would be most of the bank.
KEEP_FIELDS = ("schema", "kind", "mode", "n_rotors", "n_mics", "k_max", "sr", "front_end")


def strip_payload(fit: dict[str, Any]) -> dict[str, Any]:
    """The renderable core of a fit: :data:`KEEP_FIELDS` plus ``params``."""
    out = {k: fit[k] for k in KEEP_FIELDS if k in fit}
    out["params"] = json.loads(json.dumps(fit["params"]))
    return out


def pin_speed_laws(fit: dict[str, Any]) -> dict[str, Any]:
    """``fit`` with the three speed-law quantities pinned to :data:`SPAN_PIN`.

    The pre-pin values are recorded in ``params.span_pin_record`` so an entry
    says what was changed and from what. A deep copy: the anchor is never
    mutated.
    """
    out = strip_payload(fit)
    p = out["params"]
    record = {
        "amp_exp": float(p["profile"]["amp_exp"]),
        "floor_exp": float(p["floor"]["floor_exp"]),
        "floor_static_rel": float(p["floor"]["floor_static_rel"]),
        "pinned_to": dict(SPAN_PIN),
        "reference_rps": float(AMP_RPS_REF),
        "rule": (
            "the v2 short-span contract: a cruise-only pool cannot identify the speed laws, "
            "so they are the model's prior medians on every bank entry. Applied at the "
            "80 rev/s reference, where every speed factor is 1, so the cruise spectrum is "
            "unchanged; a standby payload IS moved at its own 20-40 rev/s."
        ),
    }
    p["profile"]["amp_exp"] = SPAN_PIN["amp_exp"]
    p["floor"]["floor_exp"] = SPAN_PIN["floor_exp"]
    p["floor"]["floor_static_rel"] = SPAN_PIN["floor_static_rel"]
    p["span_pin_record"] = record
    return out


def truncate_orders(fit: dict[str, Any], k_max: int) -> dict[str, Any]:
    """``fit`` restricted to orders ``1..k_max``; the rest are DROPPED."""
    out = strip_payload(fit)
    k = int(k_max)
    p = out["params"]
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    if prof.shape[1] < k:
        raise ValueError(f"fit carries {prof.shape[1]} orders, asked to keep {k}")
    p["profile"]["profile_db"] = _round(prof[:, :k])
    p["gamma_hz"] = _round(np.asarray(p["gamma_hz"], dtype=np.float64)[:, :k])
    out["k_max"] = k
    return out


# ---------------------------------------------------------------------------
# the expected-periodogram probe
# ---------------------------------------------------------------------------


class ModelProbe:
    """The v2 expected periodogram of a fit at one constant carrier.

    Holds the flight grid and the work-grid carrier of each probe speed, both
    of which cost more to build than the model they are used for; a bank builds
    ONE probe and evaluates thousands of draws through it.

    One frame of the declared 2 s window is evaluated, not 59: every frame of a
    CONSTANT-carrier window has the same expectation, verified to 0.0 dB against
    ``experiments.noise_model.render.expected_periodogram`` on the full window.

    Each fit is evaluated at ITS OWN work rate (``front_end.sr_work``, as
    :func:`~data_processing.noise_model.render.fit_work_rate` reads it): 64 kHz
    for every v2 fit — the grid this probe always built — and 32 kHz for a
    ``noise-v3-fit/1`` payload, whose likelihood read that rate. A v3 payload is
    evaluated with its block wander at its mean (latents zero), which is the
    forward model its fit wrote.
    """

    def __init__(
        self,
        *,
        n_rotors: int = 4,
        n_mics: int = 8,
        speeds: Sequence[float] = (PROBE_CRUISE_RPS, PROBE_IDLE_RPS),
        window_s: float = 2.0,
    ) -> None:
        import torch

        from experiments.noise_model import spectrum as SP

        torch.set_num_threads(1)
        self.n_rotors = int(n_rotors)
        self.n_mics = int(n_mics)
        self.window_s = float(window_s)
        self.freqs_hz = np.fft.rfftfreq(SP.FLIGHT_N_FFT, d=1.0 / float(SP.FLIGHT_SR))
        self._grids: dict[int, Any] = {}
        self._rate: dict[tuple[int, float], Any] = {}
        for s in speeds:
            self._rate_for(SP.SAMPLE_RATE_WORK, float(s))
        self._centres: np.ndarray | None = None

    def _rate_for(self, sr_work: int, speed_rps: float) -> tuple[Any, Any]:
        """``(grid, carrier)``: one work-rate grid and one constant speed on it."""
        from experiments.noise_model import spectrum as SP

        grid = self._grids.get(int(sr_work))
        if grid is None:
            grid = SP.flight_grid(sr_work=int(sr_work))
            self._grids[int(sr_work)] = grid
        key = (int(sr_work), float(speed_rps))
        rate = self._rate.get(key)
        if rate is None:
            rate = SP.flight_rate_work(
                grid,
                np.full((self.n_rotors, SP.FLIGHT_N_FFT), float(speed_rps)),
                np.array([0], dtype=np.int64),
            )
            self._rate[key] = rate
        return grid, rate

    def periodogram(self, fit: dict[str, Any], speed_rps: float) -> np.ndarray:
        """``(M, F)`` expected periodogram of ``fit`` at ``speed_rps``."""
        import torch

        from data_processing.noise_model.render import fit_work_rate
        from experiments.noise_model import model as MD
        from experiments.noise_model import spectrum as SP

        p = fit["params"]
        grid, rate = self._rate_for(fit_work_rate(fit), float(speed_rps))
        prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
        k_max = min(
            int(prof.shape[1]),
            SP.k_max_for_carrier(float(speed_rps), SP.FLIGHT_SR, k_cap=int(prof.shape[1])),
        )
        with torch.no_grad():
            m = SP.flight_model(
                grid, MD.params_from_dict(p, n_mics=self.n_mics), rate_work=rate, k_max=k_max
            )
        return m.cpu().numpy()[: self.n_mics, 0, :]

    def bands(self, fit: dict[str, Any], speed_rps: float) -> tuple[np.ndarray, np.ndarray, bool]:
        """``(centres, band levels dB, finite)`` — mic-mean 1/3-octave levels."""
        m = self.periodogram(fit, speed_rps)
        finite = bool(np.all(np.isfinite(m)) and np.all(m > 0.0))
        power = m.mean(axis=0)
        if not finite:
            power = np.where(np.isfinite(power) & (power > 0.0), power, 1e-300)
        cen, band = third_octave(self.freqs_hz, 10.0 * np.log10(np.maximum(power, 1e-300)))
        if self._centres is None:
            self._centres = cen
        return cen, band, finite

    @property
    def band_centres(self) -> np.ndarray:
        if self._centres is None:
            raise RuntimeError("no band has been measured yet")
        return self._centres


def band_mask(centres: np.ndarray, band: tuple[float, float] = LEVEL_BAND_HZ) -> np.ndarray:
    c = np.asarray(centres, dtype=np.float64)
    return (c >= band[0]) & (c <= band[1])


def band_level_db(centres: np.ndarray, levels: np.ndarray) -> float:
    """Power-mean of the 1/3-octave levels inside :data:`LEVEL_BAND_HZ`, dB."""
    m = band_mask(centres)
    return float(
        10.0 * np.log10(max(float(np.mean(10.0 ** (np.asarray(levels)[m] / 10.0))), 1e-300))
    )


# ---------------------------------------------------------------------------
# phase B: the widths, frozen from phase A
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Widths:
    """Per-draw sigmas at ``strength = 1``. Every field traces to phase A.

    Every tier is ``between_restart`` — the same fit at a different optimiser
    seed — EXCEPT :attr:`level_db`, which is ``between_clip`` for the reason in
    the module docstring.
    """

    #: per-rotor profile gain, dB. R5 restarts, RMS over the four rotors.
    rotor_gain_db: float = 0.366170
    #: profile trend slope, dB per decade of order. R5 restarts.
    slope_db_dec: float = 0.369979
    #: fraction of the anchor's own residual std that is redrawn, MIXED in.
    resid_redraw_frac: float = 0.090300
    #: squared-exponential correlation length of the residual, decades of
    #: order. Median of the 16 measured rotor profiles; twelve of them sit at
    #: 0.005-0.045, i.e. the residual is near-WHITE in order, which is why the
    #: parity split has to be carried by the mixture rather than by a kernel.
    resid_length_dec: float = 0.023695
    #: whole-rig level, dB, on comb and floor together. BETWEEN-CLIP tier:
    #: RMS of the three real-window sets' band-level standard deviations
    #: (DREGON 2.006, Michael's cruise 1.157, Michael's standby 0.438).
    level_db: float = 1.360854
    #: per-rotor line-width level, ln. DREGON bench restarts.
    gamma_level_ln: float = 0.061758
    #: per-order line-width scatter, ln. DREGON bench restarts.
    gamma_order_ln: float = 0.306010
    #: shaft OU amplitude, ln. DREGON bench restarts.
    sigma_nu_ln: float = 0.012086
    #: shaft OU relaxation, ln. DREGON bench restarts.
    lam_ln: float = 0.249454
    #: floor level, dB. R5 restarts. NOT the rig's level width — see
    #: :attr:`level_db`; this is the floor moving AGAINST the comb.
    floor_mean_db: float = 0.037720
    #: floor tilt, dB per octave. R5 restarts.
    floor_tilt_db_oct: float = 0.147022
    #: floor shape GP coordinate, per control point. R5 restarts.
    floor_shape_z: float = 0.182397
    #: lag-1 correlation of the shape perturbation across control points. The
    #: measured restart value is -0.330 — the restarts trade neighbouring
    #: control points against each other — and a NEGATIVE lag-1 correlation is
    #: a fit-noise signature, not a rig property, so the perturbation is drawn
    #: white (0.0) and the measured number is kept in ``structure.json``.
    floor_shape_ar1: float = 0.0
    #: per-(mic, rotor) line gain, dB. R5 restarts.
    mic_line_gain_db: float = 0.145868
    #: per-mic floor gain, dB. R5 restarts.
    mic_floor_db: float = 0.076714
    #: per-mic overall gain, dB. R5 restarts.
    mic_gains_db: float = 0.126590

    def as_dict(self) -> dict[str, float]:
        return {k: float(v) for k, v in asdict(self).items()}


WIDTHS = Widths()

#: Where each frozen width comes from. Checked against ``structure.json`` by
#: ``main()``.
PROVENANCE: dict[str, str] = {
    "rotor_gain_db": "between_restart: R5 restarts s0-s3, RMS over the 4 rotors",
    "slope_db_dec": "between_restart: R5 restarts s0-s3, RMS over the 4 rotors",
    "resid_redraw_frac": "between_restart: R5 restart residual RMS / the anchor's own residual std",
    "resid_length_dec": "variogram in log10 order, median over the 16 measured rotor profiles",
    "level_db": (
        "between_clip: std of the 300-7900 Hz band level over each rig's real windows "
        "(DREGON 10, Michael's cruise 8, Michael's standby 3), pooled as an RMS. The "
        "between_restart spread of the same quantity is 0.038 dB and is a convergence "
        "tolerance, not a width"
    ),
    "gamma_level_ln": "between_restart: DREGON bench Motor{1..4}_70 restarts, mean log gamma",
    "gamma_order_ln": "between_restart: DREGON bench Motor{1..4}_70 restarts, per-order log gamma",
    "sigma_nu_ln": "between_restart: DREGON bench Motor{1..4}_70 restarts",
    "lam_ln": "between_restart: DREGON bench Motor{1..4}_70 restarts",
    "floor_mean_db": "between_restart: R5 restarts s0-s3",
    "floor_tilt_db_oct": "between_restart: R5 restarts s0-s3",
    "floor_shape_z": "between_restart: R5 restarts s0-s3, mean over the 14 control points",
    "floor_shape_ar1": "between_restart: lag-1 correlation of the R5 restart shape deltas",
    "mic_line_gain_db": "between_restart: R5 restarts s0-s3, mean over the (8, 4) block",
    "mic_floor_db": "between_restart: R5 restarts s0-s3, mean over the 8 microphones",
    "mic_gains_db": "between_restart: R5 restarts s0-s3, mean over the 8 microphones",
}


# ---------------------------------------------------------------------------
# phase A: measuring the tiers
# ---------------------------------------------------------------------------


def _pooled_std(values: np.ndarray, axis: int = 0) -> float:
    """RMS of the per-column standard deviations of ``values``."""
    return float(np.sqrt((np.std(values, axis=axis, ddof=1) ** 2).mean()))


def _restart_spreads(paths: Sequence[str], k_max: int | None = None) -> dict[str, float]:
    """Every coordinate's between-restart spread over one restart family."""
    fits = [load_fit(p) for p in paths]
    prof = np.stack(
        [
            np.asarray(f["params"]["profile"]["profile_db"], dtype=np.float64)[
                :, : (k_max or 10**9)
            ]
            for f in fits
        ]
    )
    parts = [decompose_block(p) for p in prof]
    gains = np.array([[q.gain for q in row] for row in parts])
    slopes = np.array([[q.slope for q in row] for row in parts])
    resid = np.stack([np.stack([q.resid for q in row]) for row in parts])
    own = float(np.mean([q.resid_std for q in parts[0]]))
    lg = np.log(
        np.stack(
            [
                np.asarray(f["params"]["gamma_hz"], dtype=np.float64)[:, : (k_max or 10**9)]
                for f in fits
            ]
        )
    )
    shape = np.stack([np.asarray(f["params"]["floor"]["floor_shape_z"], float) for f in fits])
    d_shape = shape - shape.mean(axis=0)
    ar1 = float(np.corrcoef(d_shape[:, :-1].ravel(), d_shape[:, 1:].ravel())[0, 1])
    out = {
        "rotor_gain_db": _pooled_std(gains),
        "slope_db_dec": _pooled_std(slopes),
        "resid_rms_db": float(np.sqrt((np.std(resid, axis=0, ddof=1) ** 2).mean())),
        "resid_own_std_db": own,
        "gamma_level_ln": _pooled_std(lg.mean(axis=2)),
        "gamma_order_ln": float(np.mean(np.std(lg, axis=0, ddof=1))),
        "sigma_nu_ln": float(np.std([math.log(f["params"]["sigma_nu"]) for f in fits], ddof=1)),
        "lam_ln": float(np.std([math.log(f["params"]["lam"]) for f in fits], ddof=1)),
        "floor_mean_db": float(
            np.std([f["params"]["floor"]["floor_mean_db"] for f in fits], ddof=1)
        ),
        "floor_tilt_db_oct": float(
            np.std([f["params"]["floor"]["floor_tilt_db_oct"] for f in fits], ddof=1)
        ),
        "floor_shape_z": _pooled_std(shape),
        "floor_shape_ar1": ar1,
        "mic_line_gain_db": _pooled_std(
            np.stack(
                [
                    np.asarray(f["params"]["profile"]["mic_line_gain_db"], float).ravel()
                    for f in fits
                ]
            )
        ),
        "mic_floor_db": _pooled_std(
            np.stack([np.asarray(f["params"]["floor"]["mic_floor_db"], float) for f in fits])
        ),
        "mic_gains_db": _pooled_std(
            np.stack([np.asarray(f["params"]["mic_gains_db"], float) for f in fits])
        ),
    }
    out["resid_redraw_frac"] = out["resid_rms_db"] / max(out["resid_own_std_db"], 1e-9)
    return out


def _rig_coords(fit: dict[str, Any], k_max: int) -> dict[str, Any]:
    """The coordinates of one fit on a common order range."""
    p = fit["params"]
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)[:, :k_max]
    parts = decompose_block(prof)
    lg = np.log(np.asarray(p["gamma_hz"], dtype=np.float64)[:, :k_max])
    mic_line = np.asarray(p["profile"]["mic_line_gain_db"], dtype=np.float64)
    mic_gains = np.asarray(p["mic_gains_db"], dtype=np.float64)
    return {
        "gain": np.array([q.gain for q in parts]),
        "slope": np.array([q.slope for q in parts]),
        "resid": np.stack([q.resid for q in parts]),
        "resid_std": float(np.mean([q.resid_std for q in parts])),
        "drop": np.array([q.drop for q in parts]),
        "log_gamma": lg,
        "sigma_nu": math.log(float(p["sigma_nu"])),
        "lam": math.log(float(p["lam"])),
        "floor_mean_db": float(p["floor"]["floor_mean_db"]),
        "floor_tilt_db_oct": float(p["floor"]["floor_tilt_db_oct"]),
        "floor_shape_z": np.asarray(p["floor"]["floor_shape_z"], dtype=np.float64),
        "mic_line_gain_db": mic_line - mic_line.mean(axis=0, keepdims=True),
        "mic_floor_db": np.asarray(p["floor"]["mic_floor_db"], dtype=np.float64),
        "mic_gains_db": mic_gains - mic_gains.mean(),
    }


_SQRT2 = math.sqrt(2.0)


def _between_rig(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    """Per-draw sigma two rigs' difference implies: ``|delta| / sqrt(2)``."""

    def d(x: float, y: float) -> float:
        return abs(float(x) - float(y)) / _SQRT2

    def rms(x: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(((np.asarray(x) - np.asarray(y)) ** 2).mean())) / _SQRT2

    return {
        "rotor_gain_db": d(a["gain"].mean(), b["gain"].mean()),
        "slope_db_dec": d(a["slope"].mean(), b["slope"].mean()),
        "resid_redraw_frac": min(
            rms(a["resid"], b["resid"]) / max(0.5 * (a["resid_std"] + b["resid_std"]), 1e-9), 1.0
        ),
        "gamma_level_ln": d(a["log_gamma"].mean(), b["log_gamma"].mean()),
        "gamma_order_ln": rms(a["log_gamma"], b["log_gamma"]),
        "sigma_nu_ln": d(a["sigma_nu"], b["sigma_nu"]),
        "lam_ln": d(a["lam"], b["lam"]),
        "floor_mean_db": d(a["floor_mean_db"], b["floor_mean_db"]),
        "floor_tilt_db_oct": d(a["floor_tilt_db_oct"], b["floor_tilt_db_oct"]),
        "floor_shape_z": rms(a["floor_shape_z"], b["floor_shape_z"]),
        "mic_line_gain_db": rms(a["mic_line_gain_db"], b["mic_line_gain_db"]),
        "mic_floor_db": rms(a["mic_floor_db"], b["mic_floor_db"]),
        "mic_gains_db": rms(a["mic_gains_db"], b["mic_gains_db"]),
    }


def _between_rotor() -> dict[str, float]:
    """The four DREGON bench motors at throttle 70, on their common range."""
    fits = [load_fit(p) for p in BENCH_MOTORS]
    k = min(int(np.asarray(f["params"]["profile"]["profile_db"]).shape[1]) for f in fits)
    prof = np.stack(
        [np.asarray(f["params"]["profile"]["profile_db"], dtype=np.float64)[0, :k] for f in fits]
    )
    parts = [decompose(row) for row in prof]
    gains = np.array([q.gain for q in parts])
    slopes = np.array([q.slope for q in parts])
    resid = np.stack([q.resid for q in parts])
    own = float(np.mean([q.resid_std for q in parts]))
    lg = np.stack(
        [np.log(np.asarray(f["params"]["gamma_hz"], dtype=np.float64)[0, :k]) for f in fits]
    )
    return {
        "rotor_gain_db": float(gains.std(ddof=1)),
        "slope_db_dec": float(slopes.std(ddof=1)),
        "resid_redraw_frac": min(_pooled_std(resid) / max(own, 1e-9), 1.0),
        "gamma_level_ln": float(lg.mean(axis=1).std(ddof=1)),
        "gamma_order_ln": _pooled_std(lg),
        "sigma_nu_ln": float(np.std([math.log(f["params"]["sigma_nu"]) for f in fits], ddof=1)),
        "lam_ln": float(np.std([math.log(f["params"]["lam"]) for f in fits], ddof=1)),
        "floor_mean_db": float(
            np.std([f["params"]["floor"]["floor_mean_db"] for f in fits], ddof=1)
        ),
        "floor_tilt_db_oct": float(
            np.std([f["params"]["floor"]["floor_tilt_db_oct"] for f in fits], ddof=1)
        ),
        "floor_shape_z": _pooled_std(
            np.stack([np.asarray(f["params"]["floor"]["floor_shape_z"], float) for f in fits])
        ),
        "mic_line_gain_db": _pooled_std(
            np.stack(
                [
                    np.asarray(f["params"]["profile"]["mic_line_gain_db"], float).ravel()
                    for f in fits
                ]
            )
        ),
        "mic_floor_db": _pooled_std(
            np.stack([np.asarray(f["params"]["floor"]["mic_floor_db"], float) for f in fits])
        ),
        "mic_gains_db": _pooled_std(
            np.stack([np.asarray(f["params"]["mic_gains_db"], float) for f in fits])
        ),
    }


#: The real windows of each rig: what the LTAS envelope and the coverage check
#: are measured against. DREGON's ten are the five FROZEN 4 s scoring windows
#: plus the five 8 s windows the R5 fit itself used (adjacent material of the
#: same five recordings, 3.5 dB apart in level — a spread that is real and
#: belongs in a rig-to-rig envelope).
REAL_WINDOW_SETS: dict[str, tuple[str, ...]] = {
    "dregon": ("dregon_score", "dregon_fit"),
    "michaels": ("michaels_cruise",),
    "michaels_standby": ("michaels_standby",),
}


def _real_window_specs() -> dict[str, list[Any]]:
    from experiments.noise_model import supports as SUP

    return {
        "dregon_score": [
            SUP.flight_dregon(r, s, SUP.DREGON_SCORED_DUR_S) for r, s in SUP.DREGON_SCORED
        ],
        "dregon_fit": [
            SUP.flight_dregon(
                r, s + SUP.DREGON_SCORED_DUR_S + SUP.DREGON_FLOOR_GUARD_S, SUP.DREGON_FLOOR_DUR_S
            )
            for r, s in SUP.DREGON_SCORED
        ],
        "michaels_cruise": [
            SUP.flight_michaels("FLY125", s, SUP.MICHAELS_FLIGHT_DUR_S)
            for s in SUP.MICHAELS_CRUISE_STARTS
        ],
        "michaels_standby": SUP.set_michaels_standby(),
    }


def measure_real_windows() -> dict[str, Any]:
    """1/3-octave band levels of every real window, per set, and their spread.

    Reads the frozen supports (cached ``.npz`` when present, the datasets
    otherwise) and reduces each to its mic- and frame-mean band levels. Phase A
    only: the numbers are frozen into ``structure.json`` so a bank build needs
    no dataset.
    """
    from experiments.noise_model import supports as SUP

    out: dict[str, Any] = {"sets": {}, "rigs": {}}
    for tag, specs in _real_window_specs().items():
        levels, names, carriers = [], [], []
        centres: np.ndarray | None = None
        for spec in specs:
            sup = SUP.load_support(spec)
            power = sup.power.mean(axis=(0, 1))
            cen, band = third_octave(sup.freqs_hz, 10.0 * np.log10(np.maximum(power, 1e-300)))
            centres = cen if centres is None else centres
            levels.append(band)
            names.append(sup.name)
            carriers.append(float(sup.carrier_rev_s.mean()))
        B = np.stack(levels)
        assert centres is not None
        m = band_mask(centres)
        pairs = np.stack([B[i] - B[j] for i in range(len(B)) for j in range(i + 1, len(B))])
        lvl = np.array([band_level_db(centres, b) for b in B])
        out["sets"][tag] = {
            "supports": names,
            "n": int(len(B)),
            "carrier_rev_s_mean": float(np.mean(carriers)),
            "band_centres_hz": [float(v) for v in centres],
            "band_levels_db": _round(B, 6),
            "clip_to_clip_rms_db": float(np.sqrt((pairs[:, m] ** 2).mean())),
            "clip_to_clip_rms_db_full": float(np.sqrt((pairs**2).mean())),
            "level_db": [float(v) for v in lvl],
            "level_std_db": float(lvl.std(ddof=1)),
        }
    for rig, tags in REAL_WINDOW_SETS.items():
        B = np.concatenate([np.asarray(out["sets"][t]["band_levels_db"]) for t in tags])
        centres = np.asarray(out["sets"][tags[0]]["band_centres_hz"])
        m = band_mask(centres)
        pairs = np.stack([B[i] - B[j] for i in range(len(B)) for j in range(i + 1, len(B))])
        lvl = np.array([band_level_db(centres, b) for b in B])
        rms = float(np.sqrt((pairs[:, m] ** 2).mean()))
        out["rigs"][rig] = {
            "sets": list(tags),
            "n": int(len(B)),
            "clip_to_clip_rms_db": rms,
            "level_std_db": float(lvl.std(ddof=1)),
            "ltas_tol_db": LTAS_ENVELOPE_X * rms,
        }
    return out


def measure_structure(*, real_windows: bool = True) -> dict[str, Any]:
    """Phase A: every spread the sampler needs, over every listed fit."""
    dregon = load_fit(ANCHORS["dregon"]["cruise"])
    michaels = load_fit(ANCHORS["michaels"]["cruise"])
    standby = load_fit(ANCHORS["michaels"]["standby"])

    restart_r5 = _restart_spreads(R5_RESTARTS)
    restart_bench = {
        name: _restart_spreads(paths, k_max=115) for name, paths in BENCH_RESTARTS.items()
    }
    bench_pool = {
        key: float(np.sqrt(np.mean([v[key] ** 2 for v in restart_bench.values()])))
        for key in ("gamma_level_ln", "gamma_order_ln", "sigma_nu_ln", "lam_ln")
    }
    rotor = _between_rotor()
    rig = _between_rig(_rig_coords(dregon, PATH_K_MAX), _rig_coords(michaels, PATH_K_MAX))

    # residual correlation length over every measured rotor profile
    lengths: list[float] = []
    drops: list[float] = []
    for fit in (dregon, michaels, standby, *[load_fit(p) for p in BENCH_MOTORS]):
        for parts in decompose_block(fit["params"]["profile"]["profile_db"]):
            lengths.append(variogram_length(parts.resid, parts.x))
            drops.append(parts.drop)

    between_restart = dict(restart_r5)
    between_restart.update(bench_pool)
    widths = {
        "rotor_gain_db": between_restart["rotor_gain_db"],
        "slope_db_dec": between_restart["slope_db_dec"],
        "resid_redraw_frac": between_restart["resid_redraw_frac"],
        "resid_length_dec": float(np.median(lengths)),
        "gamma_level_ln": bench_pool["gamma_level_ln"],
        "gamma_order_ln": bench_pool["gamma_order_ln"],
        "sigma_nu_ln": bench_pool["sigma_nu_ln"],
        "lam_ln": bench_pool["lam_ln"],
        "floor_mean_db": between_restart["floor_mean_db"],
        "floor_tilt_db_oct": between_restart["floor_tilt_db_oct"],
        "floor_shape_z": between_restart["floor_shape_z"],
        "floor_shape_ar1": max(0.0, between_restart["floor_shape_ar1"]),
        "mic_line_gain_db": between_restart["mic_line_gain_db"],
        "mic_floor_db": between_restart["mic_floor_db"],
        "mic_gains_db": between_restart["mic_gains_db"],
    }

    out: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "git": git_head(),
        "anchors": {
            name: {
                "cruise": spec["cruise"],
                "cruise_sha256": sha256(spec["cruise"]),
                "standby": spec["standby"],
                "standby_sha256": None if spec["standby"] is None else sha256(spec["standby"]),
                "traj_rig": spec["traj_rig"],
            }
            for name, spec in ANCHORS.items()
        },
        "between_restart": {
            "r5_flight_profile": restart_r5,
            "bench_motor_70": restart_bench,
            "bench_pooled": bench_pool,
            "note": (
                "R5 FROZE gamma_hz, sigma_nu and lam from the bench fits, so their "
                "between-restart spread in the R5 family is identically zero and the bench "
                "restarts carry that tier instead"
            ),
        },
        "between_rotor": rotor,
        "between_rig": rig,
        "residual": {
            "length_dec_median": float(np.median(lengths)),
            "length_dec_per_profile": [float(v) for v in lengths],
            "trend_drop_db_min": float(np.min(drops)),
            "trend_drop_db_per_profile": [float(v) for v in drops],
            "n_profiles": len(lengths),
        },
        "widths_strength1": widths,
        "provenance": dict(PROVENANCE),
    }
    if real_windows:
        rw = measure_real_windows()
        out["real_windows"] = rw
        level_stds = [rw["rigs"][r]["level_std_db"] for r in ("dregon", "michaels")]
        level_stds.append(rw["sets"]["michaels_standby"]["level_std_db"])
        widths["level_db"] = float(np.sqrt(np.mean(np.square(level_stds))))
        out["between_clip"] = {
            "level_db": widths["level_db"],
            "per_rig": {
                "dregon": rw["rigs"]["dregon"]["level_std_db"],
                "michaels": rw["rigs"]["michaels"]["level_std_db"],
                "michaels_standby": rw["sets"]["michaels_standby"]["level_std_db"],
            },
            "between_restart_of_the_same_quantity_db": between_restart["floor_mean_db"],
        }
    out["ladder"] = {
        key: {
            "between_restart": between_restart.get(key),
            "strength_to_rotor": (
                None
                if key not in rotor or not between_restart.get(key)
                else rotor[key] / between_restart[key]
            ),
            "strength_to_rig": (
                None
                if key not in rig or not between_restart.get(key)
                else rig[key] / between_restart[key]
            ),
            "between_rotor": rotor.get(key),
            "between_rig": rig.get(key),
        }
        for key in sorted(rig)
    }
    return out


# ---------------------------------------------------------------------------
# phase B: the draw
# ---------------------------------------------------------------------------


def _se_draw(rng: np.random.Generator, x: np.ndarray, length: float) -> np.ndarray:
    """One unit-variance squared-exponential draw on the ``log k`` axis.

    PROJECTED off the trend basis ``{1, x}`` and rescaled to unit variance: the
    residual coordinate is by construction orthogonal to that basis, and an
    unprojected draw would leak its own tilt into the gain and slope.
    """
    d = x[:, None] - x[None, :]
    cov = np.exp(-0.5 * (d / float(length)) ** 2) + 1e-8 * np.eye(x.size)
    z = np.linalg.cholesky(cov) @ rng.standard_normal(x.size)
    design = np.stack([np.ones(x.size), x - x.mean()], axis=1)
    coef, *_ = np.linalg.lstsq(design, z, rcond=None)
    z = z - design @ coef
    return z / max(float(z.std()), 1e-12)


def _ar1_draw(rng: np.random.Generator, n: int, rho: float) -> np.ndarray:
    """Unit-variance AR(1) sequence with lag-1 correlation ``rho``."""
    out = np.empty(int(n))
    out[0] = rng.standard_normal()
    for i in range(1, int(n)):
        out[i] = rho * out[i - 1] + math.sqrt(max(1.0 - rho * rho, 0.0)) * rng.standard_normal()
    return out


def draw_fit(
    fit: dict[str, Any],
    rng: np.random.Generator,
    strength: float,
    widths: Widths = WIDTHS,
    *,
    level_db: float | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """One unguarded draw around ``fit``. Returns the payload and what was drawn.

    ``level_db`` overrides the level draw, which is how the standby payload of
    an entry takes the SAME whole-rig level move as its cruise payload instead
    of an independent one.
    """
    s = float(strength)
    out = strip_payload(fit)
    p = out["params"]
    prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
    n_rotors, n_orders = prof.shape
    x = order_axis(n_orders)

    # --- the whole-rig level: comb and floor together ----------------------
    d_level = float(rng.normal(0.0, s * widths.level_db)) if level_db is None else float(level_db)

    # --- harmonic profile: gain, trend, correlated residual ----------------
    d_gain = rng.normal(0.0, s * widths.rotor_gain_db, size=n_rotors)
    d_slope = rng.normal(0.0, s * widths.slope_db_dec, size=n_rotors)
    # residual MIXING, not addition: keeps the fine-structure variance at the
    # anchor's own measured level at every strength, and saturates at a fully
    # independent draw instead of diverging.
    r_mix = s * widths.resid_redraw_frac
    rho = float(np.clip(1.0 - 0.5 * r_mix * r_mix, 0.0, 1.0))
    new_profile = np.empty_like(prof)
    for i in range(n_rotors):
        parts = decompose(prof[i])
        z = _se_draw(rng, x, widths.resid_length_dec)
        fresh = rho * parts.resid + math.sqrt(max(1.0 - rho * rho, 0.0)) * parts.resid_std * z
        new_profile[i] = (
            parts.gain + d_gain[i] + d_level + (parts.slope + d_slope[i]) * (x - x.mean()) + fresh
        )
    p["profile"]["profile_db"] = _round(new_profile)

    # --- line widths: a per-rotor level in log, plus per-order scatter -----
    gamma = np.asarray(p["gamma_hz"], dtype=np.float64)
    g_level = rng.normal(0.0, s * widths.gamma_level_ln, size=n_rotors)
    g_order = rng.normal(0.0, s * widths.gamma_order_ln, size=gamma.shape)
    p["gamma_hz"] = _round(np.exp(np.log(np.maximum(gamma, 1e-12)) + g_level[:, None] + g_order))

    # --- the shaft OU ------------------------------------------------------
    p["sigma_nu"] = float(p["sigma_nu"] * math.exp(rng.normal(0.0, s * widths.sigma_nu_ln)))
    p["lam"] = float(p["lam"] * math.exp(rng.normal(0.0, s * widths.lam_ln)))

    # --- the broadband floor ----------------------------------------------
    shape = np.asarray(p["floor"]["floor_shape_z"], dtype=np.float64)
    d_shape = _ar1_draw(rng, shape.size, widths.floor_shape_ar1) * s * widths.floor_shape_z
    if is_v3(fit):
        # the SAME three floor draws as v2, in the same order and at the same
        # widths, mapped onto the v3 floor c_j = mu + sigma_B (L z)_j, which has
        # no tilt and scales its GP by the MEASURED sigma_B instead of v2's
        # fixed FLOOR_SHAPE_STD_DB (:func:`_v3_floor_step`). No channel block
        # follows: v3 has none. The wind, a per-mic low-band floor level, takes
        # the whole-rig level and v2's per-mic floor width.
        d_mean = float(rng.normal(0.0, s * widths.floor_mean_db))
        d_tilt = float(rng.normal(0.0, s * widths.floor_tilt_db_oct))
        p["floor"]["floor_shape_z"] = _round(
            shape + _v3_floor_step(d_shape - d_shape.mean(), d_tilt, p["floor"])
        )
        p["floor"]["floor_mean_db"] = float(p["floor"]["floor_mean_db"] + d_level + d_mean)
        wind = p.get("wind")
        if wind is not None:
            base = np.asarray(wind["wind_db"], dtype=np.float64)
            wind["wind_db"] = _round(
                base + d_level + rng.normal(0.0, s * widths.mic_floor_db, size=base.shape)
            )
    else:
        p["floor"]["floor_shape_z"] = _round(shape + (d_shape - d_shape.mean()))
        p["floor"]["floor_mean_db"] = float(
            p["floor"]["floor_mean_db"] + d_level + rng.normal(0.0, s * widths.floor_mean_db)
        )
        p["floor"]["floor_tilt_db_oct"] = float(
            p["floor"]["floor_tilt_db_oct"] + rng.normal(0.0, s * widths.floor_tilt_db_oct)
        )

        # --- channel structure ---------------------------------------------
        for holder, key, sigma in (
            (p["profile"], "mic_line_gain_db", widths.mic_line_gain_db),
            (p["floor"], "mic_floor_db", widths.mic_floor_db),
            (p, "mic_gains_db", widths.mic_gains_db),
        ):
            base = np.asarray(holder[key], dtype=np.float64)
            holder[key] = _round(base + rng.normal(0.0, s * sigma, size=base.shape))

    drawn = {
        "level_db": d_level,
        "d_gain_db": [float(v) for v in d_gain],
        "d_slope_db_dec": [float(v) for v in d_slope],
        "resid_mix_rho": rho,
        "gamma_level_ln": [float(v) for v in g_level],
        "sigma_nu": float(p["sigma_nu"]),
        "lam": float(p["lam"]),
        "floor_mean_db": float(p["floor"]["floor_mean_db"]),
    }
    return out, drawn


@lru_cache(maxsize=1)
def _tilt_as_ctrl() -> tuple[np.ndarray, np.ndarray]:
    """``(L, tilt)``: the floor GP's Cholesky and ``log2(f_j / 500 Hz)`` at its
    control points, the octave axis v2's ``floor_tilt_db_oct`` multiplies."""
    ctrl = floor_ctrl_hz(FLIGHT_SR)
    tilt = np.log2(np.maximum(ctrl, FLOOR_SHAPE_F_MIN) / FLOOR_TILT_REF_HZ)
    return floor_shape_chol(ctrl), tilt


def _v3_floor_step(
    d_shape_z: np.ndarray, d_tilt_db_oct: float, floor: dict[str, Any]
) -> np.ndarray:
    """The v3 ``floor_shape_z`` step equal, in dB at every control point, to a
    v2 floor step of ``d_shape_z`` (v2 GP coordinate) plus ``d_tilt_db_oct``.

    v2 puts ``FLOOR_SHAPE_STD_DB (L dz)_j + tilt log2(f_j / 500)`` on the
    control values; v3's control values are ``sigma_B (L z)_j``, so the same dB
    move is ``dz' = (FLOOR_SHAPE_STD_DB dz + tilt L^-1 log2(f_j / 500)) /
    sigma_B``. The tilt is exact between control points as well — the floor
    interpolates its control values linearly in octaves, and a tilt IS linear
    in octaves — and below the 30 Hz clamp both floors are flat.
    """
    chol, tilt = _tilt_as_ctrl()
    if d_shape_z.size != tilt.size:
        raise ValueError(f"floor carries {d_shape_z.size} control points, the ladder {tilt.size}")
    sigma_b = float(floor["floor_shape_sd_db"])
    step_db = FLOOR_SHAPE_STD_DB * (chol @ d_shape_z) + float(d_tilt_db_oct) * tilt
    return np.linalg.solve(chol, step_db) / sigma_b


# ---------------------------------------------------------------------------
# phase B: the guards
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Reference:
    """What a draw is guarded AGAINST, measured once per anchor or path point.

    Holding it as a small object is what makes the redraw loop cheap: the
    reference's own expected periodogram is evaluated once, not once per
    attempt.
    """

    name: str
    bands: np.ndarray  # (B,) cruise band levels of the reference payload
    gamma_level_ln: np.ndarray  # (R,) per-rotor mean log gamma
    ltas_tol_db: float
    #: ``(R,)`` per-rotor lower bound on the CRUISE payload's trend drop, or
    #: ``None``: :data:`TREND_MARGIN_DB` on every rotor (every v2 reference).
    #: See :func:`trend_floor_db`.
    trend_floor_db: np.ndarray | None = None


def trend_floor_db(drops: Any) -> np.ndarray:
    """``(R,)`` lower bound on each rotor's trend drop around a v3 reference.

    ``min(TREND_MARGIN_DB, reference drop - TREND_MARGIN_DB)``. On every rotor
    whose REFERENCE falls by at least twice the margin, this is v2's rule:
    every rotor must fall by :data:`TREND_MARGIN_DB`. Every v2 rotor but one
    falls that far, and so do all four DREGON v3 rotors. A rotor that falls by
    less may not be flattened by more than the margin below its reference.
    The folded round-2 Michael's cruise falls by only 0.6 and 0.2 dB on rotors
    2 and 3, against v2's measured minimum of 9.7. v2's absolute rule would
    refuse that anchor itself, and every draw around it.

    The bound is continuous in the reference's drop. A step rule — v2's 3 dB
    wherever the reference clears 3 dB — refuses about half of all draws
    around a path point that falls by just over 3 dB. It exhausted 3 of 2048
    hard draws near t = 0.84, where the interpolated rotors 2 and 3 fall by
    about 3.3 dB.
    """
    d = np.asarray(drops, dtype=np.float64)
    return np.minimum(TREND_MARGIN_DB, d - TREND_MARGIN_DB)


def reference_of(
    fit: dict[str, Any], probe: ModelProbe, *, ltas_tol_db: float, name: str = ""
) -> Reference:
    """The :class:`Reference` of one payload."""
    _, bands, finite = probe.bands(fit, PROBE_CRUISE_RPS)
    if not finite:
        raise ValueError(f"{name or 'reference'}: its own expected periodogram is not finite")
    gamma = np.log(np.maximum(np.asarray(fit["params"]["gamma_hz"], dtype=np.float64), 1e-12))
    return Reference(
        name=name,
        bands=bands,
        gamma_level_ln=gamma.mean(axis=1),
        ltas_tol_db=float(ltas_tol_db),
        trend_floor_db=(
            trend_floor_db(
                [q.drop for q in decompose_block(fit["params"]["profile"]["profile_db"])]
            )
            if is_v3(fit)
            else None
        ),
    )


def check_sample(
    fit: dict[str, Any],
    reference: Reference,
    probe: ModelProbe,
    *,
    standby: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Every guard result for one draw, plus the numbers behind them.

    ``reference`` is the anchor for :func:`sample_rig` and the INTERPOLATED
    POINT for :func:`sample_path`. ``standby``, when given, is the entry's
    standby payload: it carries the IDLE probe, because it is what the regime
    composition renders near idle, and its profile is trend-guarded too.
    """
    out: dict[str, Any] = {}
    cen, band, finite_cruise = probe.bands(fit, PROBE_CRUISE_RPS)
    idle_fit = standby if standby is not None else fit
    _, idle_band, finite_idle = probe.bands(idle_fit, PROBE_IDLE_RPS)
    out["finite_at_cruise"] = bool(finite_cruise)
    out["finite_at_idle"] = bool(finite_idle)
    out["finite"] = bool(finite_cruise and finite_idle)

    payloads = [fit] if standby is None else [fit, standby]
    exps = [
        (float(q["params"]["profile"]["amp_exp"]), float(q["params"]["floor"]["floor_exp"]))
        for q in payloads
    ]
    out["amp_exp"] = exps[0][0]
    out["floor_exp"] = exps[0][1]
    out["speed_law"] = bool(all(a >= 0.0 and f >= 0.0 for a, f in exps))

    drops = [
        parts.drop
        for q in payloads
        for parts in decompose_block(q["params"]["profile"]["profile_db"])
    ]
    out["trend_drop_db"] = [float(v) for v in drops]
    if reference.trend_floor_db is None:
        out["trend_falls"] = bool(min(drops) >= TREND_MARGIN_DB)
    else:
        # the cruise rotors against the reference's own per-rotor bound, the
        # standby payload (if any) against v2's absolute margin
        n_cruise = int(np.asarray(reference.trend_floor_db).size)
        out["trend_falls"] = bool(
            np.all(np.asarray(drops[:n_cruise]) >= reference.trend_floor_db)
            and all(v >= TREND_MARGIN_DB for v in drops[n_cruise:])
        )

    level = np.log(np.maximum(np.asarray(fit["params"]["gamma_hz"], dtype=np.float64), 1e-12)).mean(
        axis=1
    )
    ratio = np.exp(level - reference.gamma_level_ln)
    out["gamma_ratio"] = [float(v) for v in ratio]
    out["gamma_excursion"] = bool(
        np.all((ratio >= 1.0 / GAMMA_EXCURSION_CAP) & (ratio <= GAMMA_EXCURSION_CAP))
    )

    m = band_mask(cen)
    dev = (band - reference.bands)[m]
    ok_dev = bool(np.all(np.isfinite(dev)))
    out["ltas_rms_db"] = float(np.sqrt((dev**2).mean())) if ok_dev else float("inf")
    out["ltas_band_max_db"] = float(np.abs(dev).max()) if ok_dev else float("inf")
    out["ltas_level_db"] = float(dev.mean()) if ok_dev else float("inf")
    out["ltas_tol_rms_db"] = float(reference.ltas_tol_db)
    out["ltas"] = bool(out["ltas_rms_db"] <= float(reference.ltas_tol_db))

    out["bands_db"] = band
    out["idle_bands_db"] = idle_band
    guards = ("finite", "speed_law", "trend_falls", "gamma_excursion", "ltas")
    out["failed"] = [g for g in guards if not out[g]]
    out["ok"] = not out["failed"]
    return out


def _guard_record(guards: dict[str, Any]) -> dict[str, Any]:
    """The JSON-safe part of a guard dict (the band curves are dropped)."""
    return {k: v for k, v in guards.items() if not k.endswith("bands_db")}


# ---------------------------------------------------------------------------
# phase B, mode 1: the neighbourhood
# ---------------------------------------------------------------------------


def sample_rig(
    anchor: dict[str, Any],
    rng: np.random.Generator,
    reference: Reference,
    probe: ModelProbe,
    *,
    strength: float = 2.0,
    standby: dict[str, Any] | None = None,
    widths: Widths = WIDTHS,
    max_attempts: int = 16,
) -> dict[str, Any]:
    """One perturbed entry: a plausible other drone of the same kind.

    ``anchor`` and ``standby`` must already be PINNED
    (:func:`pin_speed_laws`). The standby payload, when there is one, takes the
    SAME whole-rig level move as the cruise payload and an independent draw of
    everything else, so the entry stays one rig across its two regimes.

    Guards are applied by :func:`check_sample` and a failing draw is REDRAWN,
    never clipped, so an accepted entry is an honest draw from the truncated
    distribution. Raises :class:`SampleRejected` after ``max_attempts``.
    """
    attempts: list[dict[str, Any]] = []
    for _ in range(int(max_attempts)):
        cand, drawn = draw_fit(anchor, rng, strength, widths)
        cand_standby = (
            None
            if standby is None
            else draw_fit(standby, rng, strength, widths, level_db=drawn["level_db"])[0]
        )
        guards = check_sample(cand, reference, probe, standby=cand_standby)
        attempts.append(guards)
        if guards["ok"]:
            cand["_sampler"] = {
                "mode": "neighbourhood",
                "strength": float(strength),
                "reference": reference.name,
                "attempts": len(attempts),
                "rejected": [a["failed"] for a in attempts[:-1]],
                "drawn": drawn,
                "guards": _guard_record(guards),
            }
            if cand_standby is not None:
                cand["_standby"] = cand_standby
            cand["_bands_db"] = guards["bands_db"]
            return cand
    counts: dict[str, int] = {}
    for a in attempts:
        for g in a["failed"]:
            counts[g] = counts.get(g, 0) + 1
    raise SampleRejected(
        f"no draw satisfied the guards in {max_attempts} attempts at strength {strength:g}; "
        "guards fired: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )


def sample_batch(
    anchor: dict[str, Any],
    n: int,
    seed: int,
    reference: Reference,
    probe: ModelProbe,
    *,
    strength: float = 2.0,
    standby: dict[str, Any] | None = None,
    widths: Widths = WIDTHS,
    max_attempts: int = 16,
) -> list[dict[str, Any]]:
    """``n`` neighbourhood draws, each from its own substream of ``seed``.

    Every draw is reproducible on its own — ``default_rng([seed, i])`` — so a
    bank can be rebuilt in any order, in parallel, and still be bit-identical.
    """
    return [
        sample_rig(
            anchor,
            np.random.default_rng([int(seed), int(i)]),
            reference,
            probe,
            strength=strength,
            standby=standby,
            widths=widths,
            max_attempts=max_attempts,
        )
        for i in range(int(n))
    ]


# ---------------------------------------------------------------------------
# phase B, mode 2: the path cloud between the two cruise fits
# ---------------------------------------------------------------------------

#: Interpolation convention per path coordinate, recorded in the bank.
PATH_INTERP: dict[str, str] = {
    "rotor_gain_db": "linear in dB (geometric in power), per rotor",
    "slope_db_dec": "linear (already a log-log slope), per rotor",
    "resid_db": "linear in dB, elementwise on the shared order grid K = 1..81",
    "gamma_hz": "log-linear, elementwise (strictly positive in both fits)",
    "sigma_nu": "log-linear",
    "lam": "log-linear",
    "floor_mean_db": "linear in dB",
    "floor_tilt_db_oct": "linear in dB/octave",
    "floor_shape_z": "linear in the GP coordinate (one shared 14-point ladder)",
    "mic_line_gain_db": "linear in dB",
    "mic_floor_db": "linear in dB",
    "mic_gains_db": "linear in dB",
    "amp_exp/floor_exp/floor_static_rel": "NOT interpolated: pinned on both endpoints",
    "standby": (
        "NOT interpolated: the standby SLOT is a regime policy carried with probability t, "
        "so an entry at t gets Michael's standby payload with probability t and none with "
        "probability 1 - t"
    ),
}

#: The v3 path's conventions: every coordinate v3 shares with v2 as in
#: :data:`PATH_INTERP`, and these for what v3 changed. Recorded in a v3 bank.
PATH_INTERP_V3: dict[str, str] = {
    **{
        k: v
        for k, v in PATH_INTERP.items()
        if not k.startswith(("floor_tilt", "floor_shape", "mic_"))
    },
    "floor_shape": (
        "linear in dB at every control point: sigma_B(t) = (1 - t) sigma_B,a + t sigma_B,b and "
        "z(t) = ((1 - t) sigma_B,a z_a + t sigma_B,b z_b) / sigma_B(t), so sigma_B(t) (L z(t)) "
        "is the convex combination of the two control-value curves (v2's linear-in-z with one "
        "fixed scale is the same rule)"
    ),
    "wind_db": (
        "linear in POWER with an absent endpoint as zero (only DREGON has a wind term): "
        "wind_db(t) = wind_db,a + 10 log10(1 - t); linear in dB when both endpoints carry one"
    ),
    "wander": (
        "NOT interpolated and NOT perturbed: a FITTED block carried as a policy, like the "
        "standby slot — Michael's cruise block with probability t, DREGON's with 1 - t"
    ),
    "mic blocks": "none: v3 normalises the channels in the data and renders unit gains",
}


def interpolate_fits(
    fit_a: dict[str, Any], fit_b: dict[str, Any], t: float, *, k_max: int = PATH_K_MAX
) -> dict[str, Any]:
    """The point at mixing coordinate ``t`` on the CRUISE-to-CRUISE path.

    ``t = 0`` is A and ``t = 1`` is B. Both payloads are truncated to the
    shared order range ``1..k_max`` first: the longer fit's extra orders are
    DROPPED, never padded, because padding a profile past its own fit's order
    support would invent data. The profile is mixed in the DECOMPOSED
    coordinates (per-rotor gain, trend slope, residual), everything else by the
    convention in :data:`PATH_INTERP`.

    A falling trend survives by construction — a convex combination of two
    negative slopes is negative — so only the perturbation on top of the point
    can break it.
    """
    t = float(t)
    a = truncate_orders(fit_a, k_max)
    b = truncate_orders(fit_b, k_max)
    pa, pb = a["params"], b["params"]
    out = a
    p = out["params"]
    x = order_axis(k_max)

    prof_a = decompose_block(pa["profile"]["profile_db"])
    prof_b = decompose_block(pb["profile"]["profile_db"])
    if len(prof_a) != len(prof_b):
        raise ValueError(f"rotor counts differ: {len(prof_a)} against {len(prof_b)}")
    profile = np.stack(
        [
            ((1.0 - t) * qa.gain + t * qb.gain)
            + ((1.0 - t) * qa.slope + t * qb.slope) * (x - x.mean())
            + ((1.0 - t) * qa.resid + t * qb.resid)
            for qa, qb in zip(prof_a, prof_b, strict=True)
        ]
    )
    p["profile"]["profile_db"] = _round(profile)
    p["gamma_hz"] = _round(
        np.exp(
            (1.0 - t) * np.log(np.maximum(np.asarray(pa["gamma_hz"], float), 1e-12))
            + t * np.log(np.maximum(np.asarray(pb["gamma_hz"], float), 1e-12))
        )
    )
    for key in ("sigma_nu", "lam"):
        p[key] = float(math.exp((1.0 - t) * math.log(pa[key]) + t * math.log(pb[key])))
    if is_v3(a) != is_v3(b):
        raise ValueError("the path mixes fits of one generation; got a v2 and a v3 payload")
    if is_v3(a):
        _interpolate_v3_floor(pa, pb, p, t)
    else:
        for holder_a, holder_b, holder, key in (
            (pa["floor"], pb["floor"], p["floor"], "floor_mean_db"),
            (pa["floor"], pb["floor"], p["floor"], "floor_tilt_db_oct"),
        ):
            holder[key] = float((1.0 - t) * holder_a[key] + t * holder_b[key])
        for holder_a, holder_b, holder, key in (
            (pa["floor"], pb["floor"], p["floor"], "floor_shape_z"),
            (pa["floor"], pb["floor"], p["floor"], "mic_floor_db"),
            (pa["profile"], pb["profile"], p["profile"], "mic_line_gain_db"),
            (pa, pb, p, "mic_gains_db"),
        ):
            holder[key] = _round(
                (1.0 - t) * np.asarray(holder_a[key], dtype=np.float64)
                + t * np.asarray(holder_b[key], dtype=np.float64)
            )
    for key, value in SPAN_PIN.items():
        holder = p["profile"] if key == "amp_exp" else p["floor"]
        if float(holder[key]) != float(value):
            raise ValueError(f"path endpoints must be pinned; {key} is {holder[key]}")
    out["_path"] = {
        "t": t,
        "k_max": int(k_max),
        "orders_dropped": {
            "a": int(np.asarray(fit_a["params"]["profile"]["profile_db"]).shape[1] - k_max),
            "b": int(np.asarray(fit_b["params"]["profile"]["profile_db"]).shape[1] - k_max),
        },
        "coords": {
            "gain_db": [
                float((1.0 - t) * qa.gain + t * qb.gain)
                for qa, qb in zip(prof_a, prof_b, strict=True)
            ],
            "slope_db_dec": [
                float((1.0 - t) * qa.slope + t * qb.slope)
                for qa, qb in zip(prof_a, prof_b, strict=True)
            ],
        },
    }
    return out


def _interpolate_v3_floor(
    pa: dict[str, Any], pb: dict[str, Any], p: dict[str, Any], t: float
) -> None:
    """The v3 floor and wind at ``t``, written into ``p`` (:data:`PATH_INTERP_V3`)."""
    fa, fb, f = pa["floor"], pb["floor"], p["floor"]
    f["floor_mean_db"] = float((1.0 - t) * fa["floor_mean_db"] + t * fb["floor_mean_db"])
    sa, sb = float(fa["floor_shape_sd_db"]), float(fb["floor_shape_sd_db"])
    sigma = (1.0 - t) * sa + t * sb
    f["floor_shape_sd_db"] = float(sigma)
    f["floor_shape_z"] = _round(
        (
            (1.0 - t) * sa * np.asarray(fa["floor_shape_z"], dtype=np.float64)
            + t * sb * np.asarray(fb["floor_shape_z"], dtype=np.float64)
        )
        / sigma
    )
    wa, wb = pa.get("wind"), pb.get("wind")
    if wa is not None and wb is not None:
        wind = json.loads(json.dumps(wa))
        wind["wind_db"] = _round(
            (1.0 - t) * np.asarray(wa["wind_db"], dtype=np.float64)
            + t * np.asarray(wb["wind_db"], dtype=np.float64)
        )
    elif wa is None and wb is None:
        wind = None
    else:
        present, weight = (wa, 1.0 - t) if wa is not None else (wb, t)
        assert present is not None
        wind = None
        if weight > 0.0:
            wind = json.loads(json.dumps(present))
            wind["wind_db"] = _round(
                np.asarray(present["wind_db"], dtype=np.float64) + 10.0 * math.log10(weight)
            )
    p["wind"] = wind


def sample_path(
    fit_a: dict[str, Any],
    fit_b: dict[str, Any],
    rng: np.random.Generator,
    probe: ModelProbe,
    *,
    t: float | None = None,
    spread: float = 2.0,
    ltas_tol_db: tuple[float, float],
    standby_b: dict[str, Any] | None = None,
    k_max: int = PATH_K_MAX,
    widths: Widths = WIDTHS,
    max_attempts: int = 16,
) -> dict[str, Any]:
    """A draw from the CLOUD along the path between the two cruise anchors.

    ``t`` is uniform on ``[0, 1]`` when not given. The neighbourhood
    perturbation at width ``spread`` is applied AROUND the interpolated point
    and the guards are evaluated against THAT point, which is what lets the
    cloud reach a DREGON-like rig from a Michael-like one while still refusing
    a corrupted one. ``ltas_tol_db`` are the two endpoints' tolerances and the
    point's is interpolated in ``t`` with everything else.

    The standby SLOT is drawn as a regime POLICY, not interpolated: with
    probability ``t`` the entry carries ``standby_b`` (perturbed, sharing the
    entry's level move), and with probability ``1 - t`` it carries none. At
    ``t = 0`` that reproduces DREGON's single-regime convention and at ``t = 1``
    Michael's two-regime one.
    """
    tt = float(rng.uniform()) if t is None else float(t)
    mid = interpolate_fits(fit_a, fit_b, tt, k_max=k_max)
    path = mid.pop("_path")
    tol = (1.0 - tt) * float(ltas_tol_db[0]) + tt * float(ltas_tol_db[1])
    reference = reference_of(mid, probe, ltas_tol_db=tol, name=f"path@t={tt:.3f}")
    standby = standby_b if (standby_b is not None and bool(rng.uniform() < tt)) else None
    path["standby_carried"] = standby is not None
    if is_v3(mid):
        # a v3 wander block is a FITTED block and is never perturbed: it is
        # carried as a policy, like the standby slot, rather than interpolated
        carried = bool(rng.uniform() < tt)
        if carried:
            mid["params"]["wander"] = json.loads(json.dumps(fit_b["params"]["wander"]))
        path["wander_from"] = "b" if carried else "a"
    attempts: list[dict[str, Any]] = []
    for _ in range(int(max_attempts)):
        cand, drawn = draw_fit(mid, rng, spread, widths)
        cand_standby = (
            draw_fit(standby, rng, spread, widths, level_db=drawn["level_db"])[0]
            if standby is not None
            else None
        )
        guards = check_sample(cand, reference, probe, standby=cand_standby)
        attempts.append(guards)
        if guards["ok"]:
            cand["_sampler"] = {
                "mode": "path",
                "strength": float(spread),
                "reference": reference.name,
                "attempts": len(attempts),
                "rejected": [a["failed"] for a in attempts[:-1]],
                "drawn": drawn,
                "guards": _guard_record(guards),
                "path": path,
            }
            if cand_standby is not None:
                cand["_standby"] = cand_standby
            cand["_bands_db"] = guards["bands_db"]
            return cand
    counts: dict[str, int] = {}
    for a in attempts:
        for g in a["failed"]:
            counts[g] = counts.get(g, 0) + 1
    raise SampleRejected(
        f"no draw satisfied the guards in {max_attempts} attempts at t={tt:.3f}, spread "
        f"{spread:g}; guards fired: " + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    )


# ---------------------------------------------------------------------------
# the bank: entries, and the parallel draw pool
# ---------------------------------------------------------------------------

#: The bank-file tag :func:`data_processing.noise_v2_pool.load_preset_bank`
#: accepts is imported rather than restated, so the two cannot drift.

PRESETS = ("easy", "hard")

#: Every file whose CONTENT changes what a bank would contain. Their SHA-256s
#: go into :attr:`BankSpec.digest`, so editing the sampler or the renderer
#: invalidates an existing bank and a rerun of unchanged code skips it. Paths
#: are repository-relative; a missing one digests as ``"absent"`` rather than
#: raising, so the digest is defined in a partial checkout too.
CODE_FILES: tuple[str, ...] = (
    "scripts/noise_v2_build_bank.py",
    "src/experiments/noise_model/rig_sampler.py",
    "src/data_processing/noise_model/render.py",
    "src/data_processing/noise_model/params.py",
    "src/data_processing/noise_model/spectrum.py",
    "src/data_processing/noise_model/lag.py",
    "src/data_processing/noise_model/v3.py",
)


def code_digest() -> dict[str, str]:
    """``{path: sha256}`` over :data:`CODE_FILES`, plus ``structure.json``."""
    out = {p: (sha256(p) if Path(p).is_file() else "absent") for p in CODE_FILES}
    out[str(STRUCTURE_PATH)] = sha256(STRUCTURE_PATH) if STRUCTURE_PATH.is_file() else "absent"
    return out


@dataclass(frozen=True)
class BankSpec:
    """Everything that determines a bank's CONTENT, and nothing else.

    Its :func:`canonical_digest` is what makes a rebuild idempotent: the
    builder skips when the bank on disk carries the same digest.
    """

    preset: str
    n: int
    seed: int
    strength: float
    ltas_tol_db: dict[str, float]
    widths: dict[str, float]
    k_max: int = PATH_K_MAX
    max_attempts: int = 16
    #: ``"v2"``: the v2 anchors (:data:`ANCHORS`); ``"v3"``: the round-2 v3
    #: anchors (:data:`ANCHORS_V3`), drawn by the SAME construction and widths.
    generation: str = "v2"

    def __post_init__(self) -> None:
        if self.preset not in PRESETS:
            raise ValueError(f"preset must be one of {PRESETS}, got {self.preset!r}")
        if self.preset == "easy" and int(self.n) % 2:
            raise ValueError(f"an easy bank splits evenly between two rigs; {self.n} is odd")
        anchors_of(self.generation)

    @property
    def digest(self) -> str:
        anchors = anchors_of(self.generation)
        payload = {
            "generation": self.generation,
            "preset": self.preset,
            "n": int(self.n),
            "seed": int(self.seed),
            "strength": float(self.strength),
            "k_max": int(self.k_max),
            "max_attempts": int(self.max_attempts),
            "ltas_tol_db": {k: round(float(v), 9) for k, v in sorted(self.ltas_tol_db.items())},
            "widths": {k: round(float(v), 9) for k, v in sorted(self.widths.items())},
            "anchors": {
                name: {
                    "cruise": anchors[name]["cruise"],
                    "cruise_sha256": sha256(anchors[name]["cruise"]),
                    "standby": anchors[name]["standby"],
                    "standby_sha256": (
                        None
                        if anchors[name]["standby"] is None
                        else sha256(anchors[name]["standby"])
                    ),
                }
                for name in sorted(anchors)
            },
            "span_pin": dict(SPAN_PIN),
            "guards": {
                "trend_margin_db": TREND_MARGIN_DB,
                "gamma_excursion_cap": GAMMA_EXCURSION_CAP,
                "ltas_envelope_x": LTAS_ENVELOPE_X,
                "probe_rps": [PROBE_CRUISE_RPS, PROBE_IDLE_RPS],
                "level_band_hz": list(LEVEL_BAND_HZ),
                **({"trend_rule": TREND_RULE_V3} if self.generation == "v3" else {}),
            },
            "code": code_digest(),
        }
        return canonical_digest(payload)


def default_tolerances(structure: dict[str, Any] | None = None) -> dict[str, float]:
    """``{rig: LTAS tolerance in dB}`` from ``structure.json``'s real windows."""
    st = structure if structure is not None else load_fit(STRUCTURE_PATH)
    return {rig: float(row["ltas_tol_db"]) for rig, row in st["real_windows"]["rigs"].items()}


def pinned_anchors(generation: str = "v2") -> dict[str, dict[str, Any] | None]:
    """Every anchor payload of one generation, stripped and speed-law pinned.

    Keyed ``<rig>`` and ``<rig>_standby``; the standby slot is ``None`` for a
    single-regime rig (DREGON's room-2 recordings hold no standby segment).
    """
    out: dict[str, dict[str, Any] | None] = {}
    for name, spec in anchors_of(generation).items():
        out[name] = pin_speed_laws(load_fit(spec["cruise"]))
        out[f"{name}_standby"] = (
            None if spec["standby"] is None else pin_speed_laws(load_fit(spec["standby"]))
        )
    return out


def _entry(
    sample: dict[str, Any], *, name: str, traj_rig: str | None, anchor: str
) -> dict[str, Any]:
    """One ``noise-v2-bank/1`` entry from one accepted draw."""
    sampler = sample.pop("_sampler")
    standby = sample.pop("_standby", None)
    sample.pop("_bands_db", None)
    if standby is not None:
        standby.pop("_sampler", None)
    guards = sampler["guards"]
    return {
        "name": name,
        "cruise": sample,
        "standby": standby,
        "traj_rig": traj_rig,
        "provenance": {
            "anchor": anchor,
            "mode": sampler["mode"],
            "strength": sampler["strength"],
            "attempts": sampler["attempts"],
            "rejected": sampler["rejected"],
            "level_db": sampler["drawn"]["level_db"],
            "ltas_rms_db": guards["ltas_rms_db"],
            "ltas_level_db": guards["ltas_level_db"],
            "gamma_ratio": guards["gamma_ratio"],
            "trend_drop_db_min": float(min(guards["trend_drop_db"])),
            "span_pin": dict(SPAN_PIN),
            **({"t": sampler["path"]["t"]} if "path" in sampler else {}),
            **(
                {"wander_from": sampler["path"]["wander_from"]}
                if "wander_from" in sampler.get("path", {})
                else {}
            ),
        },
    }


_WORKER: dict[str, Any] = {}


def _require(payload: dict[str, Any] | None, name: str) -> dict[str, Any]:
    """The payload, or a raise naming what was missing."""
    if payload is None:
        raise ValueError(f"{name}: expected a fit payload, got none")
    return payload


def _init_worker(spec: BankSpec) -> None:
    """One probe and one set of anchors per process, built once."""
    anchors = pinned_anchors(spec.generation)
    probe = ModelProbe()
    widths = Widths(**spec.widths)
    refs = {
        name: reference_of(
            _require(anchors[name], name), probe, ltas_tol_db=spec.ltas_tol_db[name], name=name
        )
        for name in ("dregon", "michaels")
    }
    _WORKER.clear()
    _WORKER.update(spec=spec, anchors=anchors, probe=probe, widths=widths, refs=refs)


def draw_index(index: int) -> dict[str, Any]:
    """Draw entry ``index`` of the worker's bank. Reproducible on its own.

    An EXHAUSTED draw — ``max_attempts`` rejections in a row — is returned as a
    row rather than raised, so a ladder run can measure where acceptance
    collapses instead of dying on the first casualty. A bank build refuses an
    exhausted row (:func:`build_entries` with ``allow_exhausted=False``).
    """
    spec: BankSpec = _WORKER["spec"]
    anchors = _WORKER["anchors"]
    probe: ModelProbe = _WORKER["probe"]
    widths: Widths = _WORKER["widths"]
    rng = np.random.default_rng([int(spec.seed), int(index)])
    try:
        return _draw_index(index, spec, anchors, probe, widths, rng)
    except SampleRejected as exc:
        return {"index": int(index), "exhausted": str(exc)}


def _draw_index(
    index: int,
    spec: BankSpec,
    anchors: dict[str, dict[str, Any] | None],
    probe: ModelProbe,
    widths: Widths,
    rng: np.random.Generator,
) -> dict[str, Any]:
    # a v2 entry keeps its historical name; a v3 one says so in the stream meta
    prefix = "" if spec.generation == "v2" else f"{spec.generation}_"
    if spec.preset == "easy":
        rig = "dregon" if index < spec.n // 2 else "michaels"
        sample = sample_rig(
            _require(anchors[rig], rig),
            rng,
            _WORKER["refs"][rig],
            probe,
            strength=spec.strength,
            standby=anchors[f"{rig}_standby"],
            widths=widths,
            max_attempts=spec.max_attempts,
        )
        name, traj_rig, anchor = (
            f"{prefix}{rig}_s{spec.strength:g}_{index:05d}",
            anchors_of(spec.generation)[rig]["traj_rig"],
            rig,
        )
    else:
        sample = sample_path(
            _require(anchors["dregon"], "dregon"),
            _require(anchors["michaels"], "michaels"),
            rng,
            probe,
            spread=spec.strength,
            ltas_tol_db=(spec.ltas_tol_db["dregon"], spec.ltas_tol_db["michaels"]),
            standby_b=anchors["michaels_standby"],
            k_max=spec.k_max,
            widths=widths,
            max_attempts=spec.max_attempts,
        )
        name, traj_rig, anchor = (
            f"{prefix}path_s{spec.strength:g}_{index:05d}",
            None,
            "dregon->michaels",
        )
    bands = [float(v) for v in sample["_bands_db"]]
    entry = _entry(sample, name=name, traj_rig=traj_rig, anchor=anchor)
    if spec.preset == "hard":
        entry["provenance"]["k_max"] = int(spec.k_max)
    return {"index": int(index), "entry": entry, "bands_db": bands}


def build_entries(
    spec: BankSpec,
    *,
    workers: int | None = None,
    log: Any = None,
    allow_exhausted: bool = False,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Every entry of ``spec``'s bank, plus the draw statistics.

    Draws are independent — draw ``i`` reads ``default_rng([seed, i])`` — so
    they are farmed to processes and re-sorted by index, and the bank is
    bit-identical at any worker count.

    ``allow_exhausted`` decides what happens to a draw that was rejected
    ``max_attempts`` times in a row: a BANK refuses (the default), because a
    short bank silently changes the training distribution; the strength LADDER
    allows, because measuring where acceptance collapses is the point of it.
    """
    import multiprocessing as mp

    n = int(spec.n)
    n_workers = int(workers) if workers else max(1, (mp.cpu_count() or 2) - 2)
    started = datetime.now(UTC)
    if n_workers == 1:
        _init_worker(spec)
        rows = [draw_index(i) for i in range(n)]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=(spec,)) as pool:
            rows = []
            for row in pool.imap_unordered(draw_index, range(n), chunksize=4):
                rows.append(row)
                if log is not None and len(rows) % 128 == 0:
                    log(f"  {len(rows)}/{n} draws")
    rows.sort(key=lambda r: r["index"])
    exhausted = [r for r in rows if "exhausted" in r]
    if exhausted and not allow_exhausted:
        raise SampleRejected(
            f"{len(exhausted)} of {n} draws exhausted {spec.max_attempts} attempts at strength "
            f"{spec.strength:g}; first was index {exhausted[0]['index']}: "
            f"{exhausted[0]['exhausted']}"
        )
    rows = [r for r in rows if "exhausted" not in r]
    entries = [r["entry"] for r in rows]
    attempts = np.array([e["provenance"]["attempts"] for e in entries], dtype=np.float64)
    fired: dict[str, int] = {}
    for e in entries:
        for failed in e["provenance"]["rejected"]:
            for g in failed:
                fired[g] = fired.get(g, 0) + 1
    # NOTHING wall-clock-dependent goes into ``stats`` entries that reach the
    # BANK: the bank's bytes must be reproducible from its inputs alone. The
    # timing and the worker count are RUN facts and are filtered out by
    # :func:`bank_payload`, which keeps them for the sidecar report only.
    total_attempts = float(attempts.sum() + len(exhausted) * spec.max_attempts)
    stats: dict[str, Any] = {
        "n_entries": len(entries),
        "n_requested": n,
        "n_exhausted": len(exhausted),
        "n_workers": n_workers,
        "wall_s": (datetime.now(UTC) - started).total_seconds(),
        "attempts_mean": float(attempts.mean()) if attempts.size else float("nan"),
        "attempts_max": int(attempts.max()) if attempts.size else 0,
        "first_try_frac": float((attempts == 1).mean()) if attempts.size else 0.0,
        "rejected_draws": int(total_attempts - attempts.size),
        "rejection_rate": float((total_attempts - attempts.size) / max(total_attempts, 1.0)),
        "guards_fired": dict(sorted(fired.items())),
        "bands_db": [r["bands_db"] for r in rows],
        "level_db": [float(e["provenance"]["level_db"]) for e in entries],
    }
    if spec.preset == "hard":
        ts = [float(e["provenance"]["t"]) for e in entries]
        stats["t"] = ts
        stats["ks_uniform"] = ks_uniform(ts)
        stats["t_decile_counts"] = [int(v) for v in np.histogram(ts, bins=10, range=(0.0, 1.0))[0]]
        stats["standby_carried_frac"] = float(np.mean([e["standby"] is not None for e in entries]))
    else:
        stats["per_rig"] = {
            rig: int(sum(1 for e in entries if e["provenance"]["anchor"] == rig))
            for rig in ("dregon", "michaels")
        }
    return entries, stats


def bank_payload(
    entries: list[dict[str, Any]], spec: BankSpec, stats: dict[str, Any]
) -> dict[str, Any]:
    """The whole ``noise-v2-bank/1`` file, provenance included."""
    anchors = anchors_of(spec.generation)
    v3 = spec.generation == "v3"
    return {
        "format": PRESET_BANK_FORMAT,
        "entries": entries,
        "provenance": {
            "builder": "scripts/noise_v2_build_bank.py",
            "sampler": "experiments.noise_model.rig_sampler",
            "generation": spec.generation,
            # NOTHING here is a RUN fact. No build timestamp, no git HEAD, no
            # worker count, no wall time: two builds of the same spec must be
            # byte-identical, and an unrelated commit must not change a bank's
            # bytes. Those facts live in the sidecar build report, which names
            # the bank by its sha256. What IS recorded is CONTENT: the anchor
            # and code digests below, which is also what the skip digest reads.
            "code": code_digest(),
            "digest": spec.digest,
            "preset": spec.preset,
            "mode": "neighbourhood" if spec.preset == "easy" else "path",
            "n": int(spec.n),
            "seed": int(spec.seed),
            "strength": float(spec.strength),
            "k_max": int(spec.k_max) if spec.preset == "hard" else None,
            "orders_dropped": (
                None
                if spec.preset == "easy"
                else {
                    "dregon": int(
                        np.asarray(
                            load_fit(anchors["dregon"]["cruise"])["params"]["profile"]["profile_db"]
                        ).shape[1]
                        - spec.k_max
                    ),
                    "michaels": int(
                        np.asarray(
                            load_fit(anchors["michaels"]["cruise"])["params"]["profile"][
                                "profile_db"
                            ]
                        ).shape[1]
                        - spec.k_max
                    ),
                    "rule": (
                        "the path mixes CRUISE against CRUISE on the common order range only; "
                        "the longer fit's extra orders are dropped, never padded"
                    ),
                }
            ),
            "span_pin": dict(SPAN_PIN),
            **({"path_interp": PATH_INTERP_V3} if v3 and spec.preset == "hard" else {}),
            "widths_strength1": dict(spec.widths),
            "ltas_tol_db": dict(spec.ltas_tol_db),
            "structure": str(STRUCTURE_PATH),
            "anchors": {
                name: {
                    "cruise": row["cruise"],
                    "cruise_sha256": sha256(row["cruise"]),
                    "standby": row["standby"],
                    "standby_sha256": (None if row["standby"] is None else sha256(row["standby"])),
                    "traj_rig": row["traj_rig"],
                }
                for name, row in anchors.items()
            },
            "guards": {
                "trend_margin_db": TREND_MARGIN_DB,
                "gamma_excursion_cap": GAMMA_EXCURSION_CAP,
                "ltas_envelope_x": LTAS_ENVELOPE_X,
                "probe_cruise_rps": PROBE_CRUISE_RPS,
                "probe_idle_rps": PROBE_IDLE_RPS,
                "level_band_hz": list(LEVEL_BAND_HZ),
                **({"trend_rule": TREND_RULE_V3} if v3 else {}),
            },
            "statistics": {
                k: v for k, v in stats.items() if k not in ("bands_db", "t", "wall_s", "n_workers")
            },
        },
    }


# ---------------------------------------------------------------------------
# coverage: how the strength is chosen
# ---------------------------------------------------------------------------


def coverage(
    cloud_bands: np.ndarray, real_bands: np.ndarray, centres: np.ndarray
) -> dict[str, Any]:
    """Does the cloud BRACKET the real windows, band by band?

    ``cloud_bands`` is ``(n_draws, B)`` of accepted draws' expected-periodogram
    band levels, ``real_bands`` is ``(n_clips, B)`` of measured ones. A band
    counts as covered for a clip when the clip's level lies between the cloud's
    minimum and maximum there. Reported over all bands and over the guarded
    band above 300 Hz, the same criterion the legacy transfer pair used.
    """
    cloud = np.asarray(cloud_bands, dtype=np.float64)
    real = np.asarray(real_bands, dtype=np.float64)
    lo, hi = cloud.min(axis=0), cloud.max(axis=0)
    inside = (real >= lo[None, :]) & (real <= hi[None, :])
    m = band_mask(np.asarray(centres, dtype=np.float64))
    per_clip = inside[:, m].mean(axis=1)
    return {
        "n_draws": int(cloud.shape[0]),
        "n_clips": int(real.shape[0]),
        "all_bands": float(inside.mean()),
        "above_300hz": float(inside[:, m].mean()),
        "above_300hz_worst_clip": float(per_clip.min()),
        "per_clip_above_300hz": [float(v) for v in per_clip],
        "cloud_span_db": [float(v) for v in (hi - lo)],
        "band_centres_hz": [float(v) for v in centres],
    }


def ks_uniform(values: Any) -> dict[str, float]:
    """Kolmogorov-Smirnov distance of ``values`` to ``U(0, 1)``, and its 95 %
    threshold ``1.358 / sqrt(n)``."""
    v = np.sort(np.asarray(values, dtype=np.float64))
    n = v.size
    i = np.arange(1, n + 1)
    d = float(max(np.max(i / n - v), np.max(v - (i - 1) / n)))
    return {
        "ks": d,
        "threshold_95": float(1.358 / math.sqrt(n)),
        "mean": float(v.mean()),
        "n": int(n),
        "passes": bool(d <= 1.358 / math.sqrt(n)),
    }


def real_bands_of(structure: dict[str, Any], rig: str) -> tuple[np.ndarray, np.ndarray]:
    """``(centres, (n_clips, B) band levels)`` of one rig's real windows."""
    sets = REAL_WINDOW_SETS[rig]
    rows = structure["real_windows"]["sets"]
    centres = np.asarray(rows[sets[0]]["band_centres_hz"], dtype=np.float64)
    levels = np.concatenate([np.asarray(rows[s]["band_levels_db"], dtype=np.float64) for s in sets])
    return centres, levels


def run_ladder(
    structure: dict[str, Any],
    *,
    n: int = 256,
    seed: int = 20260921,
    strengths: Sequence[float] = LADDER_STRENGTHS,
    workers: int | None = None,
    log: Any = print,
) -> dict[str, Any]:
    """Walk :data:`LADDER_STRENGTHS` and measure coverage and acceptance.

    Returns the per-strength block plus the CHOSEN strength: 2.0 when it
    already reaches :data:`COVERAGE_TARGET` above 300 Hz on every rig, else the
    first setting on the ladder that does, else the best measured.
    """
    tol = default_tolerances(structure)
    real = {rig: real_bands_of(structure, rig) for rig in ("dregon", "michaels")}
    rows: dict[str, Any] = {}
    for strength in strengths:
        block: dict[str, Any] = {}
        for preset in PRESETS:
            spec = BankSpec(
                preset=preset,
                n=int(n),
                seed=int(seed),
                strength=float(strength),
                ltas_tol_db=tol,
                widths=WIDTHS.as_dict(),
            )
            entries, stats = build_entries(spec, workers=workers, allow_exhausted=True)
            bands = np.asarray(stats["bands_db"], dtype=np.float64)
            cov: dict[str, Any] = {}
            for rig, (centres, levels) in real.items():
                sel = (
                    bands
                    if preset == "hard"
                    else bands[
                        [i for i, e in enumerate(entries) if e["provenance"]["anchor"] == rig]
                    ]
                )
                cov[rig] = coverage(sel, levels, centres) if sel.size else None
            block[preset] = {
                "coverage": cov,
                "n_entries": stats["n_entries"],
                "n_exhausted": stats["n_exhausted"],
                "attempts_mean": stats["attempts_mean"],
                "rejection_rate": stats["rejection_rate"],
                "first_try_frac": stats["first_try_frac"],
                "guards_fired": stats["guards_fired"],
                "wall_s": stats["wall_s"],
                "level_db_std": (
                    float(np.std(stats["level_db"], ddof=1)) if len(stats["level_db"]) > 1 else 0.0
                ),
            }
            if log is not None:
                log(
                    f"  strength {strength:g} {preset:4s}: coverage>300Hz "
                    + ", ".join(
                        f"{r} {c['above_300hz']:.3f}" if c else f"{r} --" for r, c in cov.items()
                    )
                    + f"  accept {1 - stats['rejection_rate']:.3f}"
                    + f"  attempts {stats['attempts_mean']:.2f}"
                    + f"  exhausted {stats['n_exhausted']}/{spec.n}"
                )
        rows[f"{strength:g}"] = block

    def reaches(block: dict[str, Any]) -> bool:
        return all(
            block[p]["n_exhausted"] == 0
            and block[p]["coverage"][r]
            and block[p]["coverage"][r]["above_300hz"] >= COVERAGE_TARGET
            for p in PRESETS
            for r in ("dregon", "michaels")
        )

    ordered = [f"{s:g}" for s in strengths]
    default = f"{2.0:g}"
    if default in rows and reaches(rows[default]):
        chosen, reason = default, "the default 2.0 already reaches the coverage target"
    else:
        above = [k for k in ordered if reaches(rows[k])]
        if above:
            chosen = above[0]
            reason = f"the first ladder setting reaching {COVERAGE_TARGET:.0%} above 300 Hz"
        else:
            chosen = max(
                ordered,
                key=lambda k: min(
                    rows[k][p]["coverage"][r]["above_300hz"]
                    for p in PRESETS
                    for r in ("dregon", "michaels")
                ),
            )
            reason = (
                f"no ladder setting reaches {COVERAGE_TARGET:.0%} above 300 Hz; this is the "
                "best measured worst-rig coverage"
            )
    return {
        "n_per_strength": int(n),
        "seed": int(seed),
        "strengths": [float(s) for s in strengths],
        "target": COVERAGE_TARGET,
        "rows": rows,
        "chosen_strength": float(chosen),
        "reason": reason,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def print_tier_table(st: dict[str, Any]) -> None:
    """The tier table and the strength ladder."""
    print(
        f"{'coordinate':22s} {'restart':>9s} {'rotor':>9s} {'rig':>9s} {'->rotor':>8s} {'->rig':>8s}"
    )
    for key, row in st["ladder"].items():

        def f(v: Any, w: int = 9, p: int = 4) -> str:
            return " " * w if v is None else f"{float(v):{w}.{p}f}"

        print(
            f"{key:22s} {f(row['between_restart'])} {f(row['between_rotor'])} "
            f"{f(row['between_rig'])} {f(row['strength_to_rotor'], 8, 1)} "
            f"{f(row['strength_to_rig'], 8, 1)}"
        )
    if "between_clip" in st:
        bc = st["between_clip"]
        print(
            f"{'level_db (clip)':22s} {bc['between_restart_of_the_same_quantity_db']:9.4f} "
            f"{'':>9s} {'':>9s}   between-clip width {bc['level_db']:.4f} dB"
        )
    res = st["residual"]
    print(
        f"\nresidual: SE length {res['length_dec_median']:.4f} decades (median of "
        f"{res['n_profiles']} rotor profiles), smallest trend drop "
        f"{res['trend_drop_db_min']:.1f} dB (guard {TREND_MARGIN_DB:g} dB)"
    )
    if "real_windows" in st:
        print("\nreal windows:")
        for rig, row in st["real_windows"]["rigs"].items():
            print(
                f"  {rig:18s} n={row['n']:2d} clip-to-clip rms {row['clip_to_clip_rms_db']:.3f} dB"
                f" -> ltas tolerance {row['ltas_tol_db']:.3f} dB, level std "
                f"{row['level_std_db']:.3f} dB"
            )


def print_width_check(st: dict[str, Any]) -> None:
    """Frozen widths against freshly measured ones."""
    frozen = WIDTHS.as_dict()
    measured = st["widths_strength1"]
    worst = 0.0
    print(f"\n{'width':22s} {'frozen':>10s} {'measured':>10s} {'drift':>8s}")
    for key, value in frozen.items():
        got = measured.get(key)
        if got is None:
            print(f"{key:22s} {value:10.4f} {'--':>10s}")
            continue
        drift = abs(got - value) / max(abs(value), 1e-9)
        worst = max(worst, drift)
        print(f"{key:22s} {value:10.4f} {got:10.4f} {100 * drift:7.2f}%")
    print(f"worst frozen-vs-measured drift: {100 * worst:.3f}%")


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("measure", "ladder"), nargs="?", default="measure")
    ap.add_argument("--out", default=str(STRUCTURE_PATH))
    ap.add_argument(
        "--no-real-windows",
        action="store_true",
        help="measure: skip the support reads (the fit-only tiers still measure)",
    )
    ap.add_argument("--n", type=int, default=256, help="ladder: draws per strength per preset")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--workers", type=int, default=None)
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.command == "measure":
        st = measure_structure(real_windows=not args.no_real_windows)
        out.write_text(json.dumps(st, indent=2, sort_keys=False) + "\n")
        print(f"wrote {out}")
        print_tier_table(st)
        print_width_check(st)
        return 0

    st = load_fit(STRUCTURE_PATH)
    print(f"strength ladder, {args.n} draws per strength per preset:")
    ladder = run_ladder(st, n=args.n, seed=args.seed, workers=args.workers)
    st["coverage_ladder"] = ladder
    STRUCTURE_PATH.write_text(json.dumps(st, indent=2, sort_keys=False) + "\n")
    print(f"\nchosen strength {ladder['chosen_strength']:g}: {ladder['reason']}")
    print(f"wrote {STRUCTURE_PATH}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
