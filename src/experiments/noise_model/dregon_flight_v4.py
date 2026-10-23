"""Noise model v4, the DREGON FLIGHT STAGE: the wind, the floor, a per-rotor
and a per-mic line gain from free flight, on top of a frozen bench rig.

WHY. A v4 BENCH payload (``kind="bench"``, :mod:`.bench_v4`) rendered on
DREGON free flight is 6-23 dB too quiet below 2 kHz, for three reasons: it has
no flight wind (the bench wind is per RECORDING, not a rig term), its
broadband floor is the bench floor while the flight floor is higher, and its
lines are quieter than the flight lines (HPPNet pitch scoring: +6 dB on every
line profile takes the flight PIT from 15.8 to 2.7 rev/s). The line SHAPES
(the profiles across orders, the pedestal, the shaft law and the pair EQ) are
properties of the rotors and the array, and the bench is where they are seen
best (one rotor at a time, 0.03-0.1 Hz bins); the flight wind, the flight
floor and the flight line LEVEL exist only in flight. So this stage refits the
floor, the wind, one level per rotor and one per mic, from flight recordings
only, and holds every bench line-shape / pedestal / shaft / pair-EQ shape
parameter fixed (``coherence="shaft"`` / ``"full"`` also refit the line
COHERENCE: see below). The v4 model is unchanged: the stage is a procedure, and its payload
is the bench payload with ``kind="flight"``, the rotor gains written into the
line profiles, the mic gains into the pair EQ, the floor refitted and a static
per-mic wind block, which
:func:`data_processing.noise_model.v4.expected_periodogram` and the renderer
read as they stand.

WHY THE COHERENCE IN FLIGHT (``coherence="full"``, CLI ``--coherence full``).
The bench fitted its phase pedestal and shaft wander against a MEASURED
carrier (bench sigma_nu 0.73 rad/s at f_ref, lam 49 /s). The flight scorer's
carrier is the ``motors_command`` LABEL, not the true speed: in flight its
error acts as phase wander that the bench never saw. Real DREGON flight lines
stand out of the floor only at k = 1 (7.7 dB) and k = 2 (4.1 dB) and are
smeared to noise level from k = 3 on; nv3r4, which fitted its coherence in
flight (sigma_nu 5.2 rad/s, lam 100.5 /s), reproduces that (6.0 / 5.7 dB, flat
above), while v4 with the frozen bench coherence is far too coherent (HPPNet
reads two mics' rotor as off). So, optionally, the stage frees the phase
pedestal per aerodynamic order (``sigma2_phase_k``, ``W_k`` in the bench's
capped coordinates under the bench prior
:func:`.bench_v4.phase_pedestal_neg_log_prior`) and the shaft scale ``eps``
(``sigma_ref = 2 pi f_ref eps``, the DREGON ``sigma_nu`` at the payload's
``f_ref``, ``~ LogNormal(`` :data:`.v4_priors.FLIGHT_SIGMA_NU_LN` ``)``),
holding the amplitude pedestal, ``a_sigma``, ``lam_ref`` and ``b_lam`` at the
bench's. ``L_r`` is then rendered per evaluation by :class:`LineRenderer`, a
torch port of the line part of ``expected_periodogram`` (``renderer_check``
holds it to 1e-6 of the numpy renderer at random pedestal, ``eps`` and gains),
started at the bench pedestal (clamped under its caps) and the bench ``eps``.
The payload carries the fitted ``sigma2_phase``, ``gamma_phase_hz``,
``width_phase_hz`` and ``eps``.

WHY SHAFT ONLY (``coherence="shaft"``, CLI ``--coherence shaft``; the
2026-10-23 probe). Under ``"full"`` the flight fit drove every phase pedestal
to its caps (``sigma2_phase`` 1-3, ``W`` 24 Hz) and ``eps`` from 0.0014 to
0.049, and the HPPNet scorer got WORSE (PIT 29 vs 16 rev/s). The phase
pedestal smears each order independently, which breaks the harmonic coherence
HPPNet reads; the shaft wander smears all orders of a rotor coherently (FM),
which is how nv3r4 absorbed the flight smear (``sigma_nu`` 5.2 rad/s). So
``"shaft"`` frees ``eps`` alone (the same ``sigma_ref`` prior as ``"full"``)
and holds the phase pedestal at the bench payload's own values (no gradient,
no pedestal prior term); the flight smear can only go into the shaft. The
payload carries the bench pedestal unchanged and the fitted ``eps``.

WHY PER ROTOR. The bench fitted each rotor's profiles from that rotor's own
recordings, alone on the rig, so each rotor carries its own bench-to-flight
level error (its own load at flight thrust, its own mounting); one shared gain
would force the four errors equal. The gain is ONE number per rotor, applied
to both line families, because what changes from bench to flight is that
rotor's level, not the shape of its spectrum. The four gains are identifiable:
the rotors differ in their pair EQ across the eight mics and in their speeds
per frame.

WHY PER MIC. HPPNet reads the rotors as OFF on DREGON mics 2 and 6 of a v4
stage render: the v4 rotor content at low frequency there is 3-5 dB under the
recording (nv3r4 within +-2 dB), and +6 dB on those mics' pair-EQ rows fixes
mic 6 (PIT 80 -> 2.3 rev/s). In flight a mic's line level differs from the
bench pair EQ (mounting, airflow over the capsule), by mic, for every rotor.
So ``h_m`` adds to ``E_rm(f)`` of every rotor at every frequency: it scales
rotor ``r``'s lines AND the per-rotor floor term on mic ``m`` (the renderer
weighs both by the pair EQ), not the static share or the wind. ``g_r + h_m``
has one common-level degeneracy, broken by ``sum_m h_m = 0`` (free
``h_1 .. h_{M-1}``, ``h_M = -sum``, as :meth:`.flight_v4.FlightV4Model._full_c`
pins the rotor offsets).

THE POOL. nv3r4's DREGON flight pool: support set ``dregon-floor``'s five
DISJOINT 8 s room-2 segments ``flight_dregon_<rec>@<t>+8_motors_command``
(free-flight, hovering, updown, rectangle, spinning), cut after each 4 s
scoring window plus a 2 s guard; never the scoring windows themselves. The
frames are the supports' own periodograms ``|rfft(w x)|^2 / sum w^2`` (16 kHz,
2048 / 512, periodic Hann: :func:`experiments.stochastic_fit.data.periodogram`,
the units of ``expected_periodogram``), thinned by 4 to disjoint frames (~310)
and weighted by nv3r4's composite exposure weights; a frame's rotor speeds
are the frame-mean ``motors_command`` label. NO channel normalisation: the
bench pair EQ carries each mic's absolute level. The likelihood band is the
bench payload's ``band_hz`` (the bench Whittle's 30 Hz-7.2 kHz, clear of the
16 kHz decimator, which ``expected_periodogram`` does not model).

THE MODEL, mic ``m``, frame ``n``::

    E_mn = 10^{h_m/10} [ sum_r 10^{g_r/10} L_rmn + T(sum_r h_rm F s_rn^b) ]
           + T[ s F + 10^{w_m/10} W ]

``L_r`` (:func:`frozen_lines`) is rotor ``r``'s expected periodogram at the
frame's speeds from the bench payload with the floor switched off and the
other rotors' profiles silenced: its lines, pedestal, shaft and pair EQ,
computed once per frame by ``expected_periodogram`` (``sum_r L_r`` is asserted
to equal the all-rotor periodogram). ``T`` is that function's window
transform of a PSD on its ``floor_oversample``-times finer grid (linear in the
PSD), done in torch, so the floor and wind terms are exactly what
``expected_periodogram`` adds for them (:func:`renderer_check` holds the whole
``E`` to 1e-6 of the renderer's at random parameters, gains included). ``h``
is the frozen pair EQ, ``F`` the 14-point floor spline on the bench's
``ctrl_hz`` and ``shape_sd_db``, ``b(f) = b0 + b1 log2(f / 2 kHz)``, ``s`` the
static share, ``W`` the nv3 wind shape. The payload writes ``g_r`` into
``profile_aero_db[r]`` and ``profile_motor_db[r]`` and ``h_m`` into
``mic_response.pair_eq.db[:, m, :]`` (linearly interpolated, so exact).

FREE: ``g_r`` (4, ``line_gain_db``; held at 0 by ``line_gain=False``),
``h_m`` (8 with ``sum = 0``, ``mic_gain_db``; held at 0 by ``mic_gain=False``),
``mean_db``, ``shape_z`` (14), ``static_rel``, ``b0``/``b1`` (in the ``b >= 0``
band-edge coordinates of :func:`.v4_priors.floor_exponents`) and ``wind_db``
(8); under ``coherence="shaft"`` also ``eps``, under ``"full"`` also
``sigma2_phase``/``W`` per aerodynamic order and ``eps``. FROZEN: the profile
shapes across orders, ``alpha_aero``/``alpha_motor``, the motor orders, the
hierarchy, the pedestal and the shaft (but for what the coherence mode frees),
the pair EQ's shape (up to ``h_m``) and the
floor's ``ctrl_hz``/``shape_sd_db``/rule. PRIORS: ``g_r ~ N(LINE_GAIN_MU_DB,
LINE_GAIN_SD_DB)``; ``h_m ~ N(0, MIC_GAIN_SD_DB)`` on all eight; the bench's floor terms
(:meth:`.bench_v4.RigModel.neg_log_prior`): ``mean_db ~ N(measured,
FLOOR_MEAN_SD_DB)``, the centre measured on THIS pool as the bench measured
its own, ``z ~ N(0, I)``, ``b0 ~ N(B0)``, ``b1 ~ N(B1)``; the flight static
share ``static_rel ~ LogNormal(flight_v4.STATIC_SHARE_LN)``; nv3r4's wind
prior ``wind_db_m ~ N(measured low-band excess, PRIOR_SETS["v4"].wind_db_sd)``
(:func:`.fit._wind_centres`, here on the unnormalised pool); and the floor
push ``-kappa sum_j ctrl_j`` with the bench payload's ``floor_push_kappa``.

OPTIMISER. Start: the floor measured from the pool (low quantile of
``(I - L - W0) / (sum_r h s^b)`` per bin, projected on the 14 controls), the
bench's exponents, the static share at its prior median, ``g_r = 0``, ``h_m = 0`` and the
nv3r4 DREGON wind (a START only). Then double-precision strong-Wolfe L-BFGS
(:class:`.lbfgs.LBFGS`) in fresh-memory passes until one gains under
``1e-4`` nats per cell AND moves the floor controls under 0.1 dB, the
exponents under 0.02, the wind under 0.1 dB and the line and mic gains under 0.1 dB
(and ``log eps`` under 0.05 under ``"shaft"``; ``log sigma2_phase``, ``log W``
and ``log eps`` under 0.05 under ``"full"``: :data:`COHERENCE_MOVE_TOL`). The
L-BFGS cost under ``"shaft"``/``"full"`` is the line renderer's (R x K' x n_fft per frame): ~19 s
per objective + gradient on the 310 frames on 2 CPU threads, so the full
stage runs on a GPU (``--device cuda``).
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from data_processing.noise_model import v4 as V4
from data_processing.noise_model.constants import SPEED_FLOOR_RPS
from data_processing.noise_model.floor import floor_geometry, floor_power_spectrum
from data_processing.noise_model.ou import integrated_ou_increment_var
from data_processing.noise_model.spectrum import floor_shape_chol
from data_processing.noise_model.v3 import wind_shape, wind_shape_spec
from experiments.stochastic_fit.revised_phase import (
    FLOOR_INIT_DB_CORRECTION,
    FLOOR_INIT_QUANTILE,
    composite_risk,
    composite_weights,
)

from . import bench_v4 as BV
from . import fit as FT
from . import flight_v4 as FV
from . import model as MD
from . import supports as SU
from . import v4_priors as PR
from .lbfgs import LBFGS

__all__ = [
    "FlightPool",
    "FlightStageModel",
    "LineRenderer",
    "Start",
    "coherence_start",
    "coherence_values",
    "fit_stage",
    "flight_payload",
    "frozen_lines",
    "held_pedestal",
    "load_pool",
    "measure_start",
    "renderer_check",
    "run_stage",
    "shaft_values",
    "start_coherence_values",
    "wind_start",
]

STAGE = "dregon_flight_v4"
#: The output name and nv3r4's name of the same pool.
NAME = "dregon_room2"
POOL = "dregon_room2_floor"
SUPPORT_SET = "dregon-floor"
FRAME_STRIDE = 4
#: nv3r4's DREGON flight fit (round 4 = r4b): its wind is the stage's START.
WIND_START = "results/noise_v3/fits_r4/dregon_room2_floor__flight_v3.json"
#: The renderer's floor PSD grid factor: ``expected_periodogram``'s own default,
#: read from it so the stage cannot drift from the renderer.
FLOOR_OVERSAMPLE = int(
    inspect.signature(V4.expected_periodogram).parameters["floor_oversample"].default
)
#: ``mean_db`` of the silenced floor and the profile rows of the silenced
#: rotors: ``10^{-1000}`` underflows to exactly 0.
SILENT_DB = -1.0e4
#: ``sum_r L_r`` against the all-rotor periodogram (max relative error).
SPLIT_RTOL = 1e-9
#: The per-rotor flight line gain prior ``g_r ~ N(mu, sd)`` (dB): centred on the
#: bench level; a 6 dB scale because bench -> flight changes the level by that
#: order (four rotors loading the airframe in flight vs one on the bench).
LINE_GAIN_MU_DB = 0.0
LINE_GAIN_SD_DB = 6.0
#: The per-mic flight line gain prior ``h_m ~ N(0, sd)`` (dB, ``sum_m h_m = 0``):
#: the flight vs bench per-mic level (mounting, airflow over the capsule), on
#: the scale of nv3r4's per-mic level spread.
MIC_GAIN_SD_DB = 6.0
#: nv3r4's DREGON prior set (``--priors v4``): the wind prior's sd.
WIND_PRIORS = MD.PRIOR_SETS["v4"]
TOL_NATS_PER_CELL = FV.TOL_NATS_PER_CELL
#: Convergence movement: floor controls (dB), exponents (b0, b1), wind levels,
#: line gains and mic gains (dB, held to the line-level tolerance).
MOVE_TOL = dict(
    floor_ctrl_db=FV.MOVE_TOL["floor_ctrl_db"],
    exponents=FV.MOVE_TOL["exponents"],
    wind_db=FV.MOVE_TOL["levels_db"],
    line_gain_db=FV.MOVE_TOL["levels_db"],
    mic_gain_db=FV.MOVE_TOL["levels_db"],
)
#: The coherence movement tolerances (in logs): the per-order phase pedestal
#: variance and width (``coherence="full"``) and the shaft wander scale
#: (``"shaft"`` and ``"full"``).
COHERENCE_MOVE_TOL = dict(log_sigma2_phase=0.05, log_width_phase=0.05, log_eps=0.05)
#: :func:`renderer_check`'s tolerance (max relative error of ``E``).
CHECK_RTOL = 1e-6
PARAM_KEYS = ("floor_mean_db", "floor_z", "u_blo", "u_bhi", "log_static", "wind_db")
#: The line gains' key: free (in ``params``) or held at 0 (``line_gain=False``).
GAIN_KEY = "line_gain_db"
#: The mic gains' key: the first ``M - 1`` of ``h_m`` (the last is ``-sum``),
#: free (in ``params``) or held at 0 (``mic_gain=False``).
MIC_GAIN_KEY = "mic_gain_free_db"
#: The coherence modes: ``"none"`` (frozen ``L_r``), ``"shaft"`` (free ``eps``,
#: the bench phase pedestal held) and ``"full"`` (free phase pedestal and ``eps``).
COHERENCE_MODES = ("none", "shaft", "full")
#: The full-coherence keys: the phase pedestal's variance and width per
#: aerodynamic order in the bench's capped coordinates (``sigma2 = _capped(u_s2P,
#: PHASE_SIGMA2_MAX)``, ``W = _capped(u_W, W_MAX_HZ)``) and ``log eps``.
COHERENCE_KEYS = ("u_s2P", "u_W", "log_eps")
#: Per mode: the free coherence parameters, their snapshot groups (movement
#: test) and the payload fields they set.
COHERENCE_FREE = dict(none=(), shaft=("log_eps",), full=COHERENCE_KEYS)
COHERENCE_MOVES = dict(none=(), shaft=("log_eps",), full=tuple(COHERENCE_MOVE_TOL))
COHERENCE_FIELDS = dict(
    none=(),
    shaft=("shaft.eps",),
    full=("pedestal.sigma2_phase", "pedestal.width_phase_hz", "shaft.eps"),
)
#: The full-coherence start holds the bench pedestal at most this fraction of
#: its caps (the bench's own reload margin, ``bench_v4._under``'s 0.99).
START_CAP_FRACTION = 0.99


# ── the pool ────────────────────────────────────────────────────────────────


@dataclass
class FlightPool:
    """The thinned flight frames the stage fits, in the supports' units."""

    names: tuple[str, ...]
    power: np.ndarray  #: (M, N, F) ``|rfft(w x)|^2 / sum w^2``, NOT channel-normalised
    rps: np.ndarray  #: (R, N) frame-mean ``motors_command`` label, rev/s
    weights: np.ndarray  #: (N,) composite exposure weights x the thinning factor
    frame_window: np.ndarray  #: (N,) index into ``names``
    frame_start: np.ndarray  #: (N,) first sample of the frame in its window
    freqs_hz: np.ndarray  #: (F,)
    sr: int
    n_fft: int
    hop: int
    stride: int
    n_frames_read: int  #: every frame of the windows read, before thinning
    exposure_scale: float

    @property
    def n_frames(self) -> int:
        return int(self.power.shape[1])

    def diagnostics(self) -> dict[str, Any]:
        return dict(
            support_set=SUPPORT_SET,
            supports=list(self.names),
            rps_key=SU.DREGON_RPS_KEY,
            speeds="frame-mean label (experiments.stochastic_fit.data.periodogram)",
            sr=self.sr,
            n_fft=self.n_fft,
            hop=self.hop,
            window="periodic Hann",
            units="|rfft(w x)|^2 / sum w^2, one-sided; no channel gains",
            frame_stride=self.stride,
            n_frames_read=self.n_frames_read,
            n_frames_used=self.n_frames,
            exposure_scale=self.exposure_scale,
            weights="revised_phase.composite_weights x exposure_scale (nv3r4 flight_batch)",
            frames=[
                [int(w), int(s)] for w, s in zip(self.frame_window, self.frame_start, strict=True)
            ],
            carrier_min_rev_s=float(self.rps.min()),
            carrier_max_rev_s=float(self.rps.max()),
        )


def pool_specs() -> list[SU.SupportSpec]:
    """The five 8 s fit segments of ``dregon-floor`` (not its 4 s scoring windows)."""
    return [
        s
        for s in SU.support_set(SUPPORT_SET)
        if float(s.args["dur_s"]) == float(SU.DREGON_FLOOR_DUR_S)
    ]


def load_pool(*, stride: int = FRAME_STRIDE, max_frames: int | None = None) -> FlightPool:
    """Every ``stride``-th frame of the five segments, in segment order;
    ``max_frames`` keeps the first frames only and reads no further segment
    (a smoke run)."""
    powers, rpss, keys, fws, fss, names = [], [], [], [], [], []
    n_read = n_thin = 0
    first: Any = None
    for w, spec in enumerate(pool_specs()):
        if max_frames is not None and n_thin >= int(max_frames):
            break
        s = SU.load_support(spec)
        if (s.sr, s.n_fft, s.hop) != (SU.SR, SU.OBS_N_FFT, SU.OBS_HOP):
            raise ValueError(f"{s.name}: front end {(s.sr, s.n_fft, s.hop)} is not the pool's")
        first = first or s
        sel = np.arange(0, s.n_frames, max(1, int(stride)))
        starts = np.asarray(s.frame_starts, dtype=np.int64)[sel]
        n_read += s.n_frames
        n_thin += sel.size
        powers.append(np.asarray(s.power, dtype=np.float64)[:, sel])
        rpss.append(np.asarray(s.carrier_rev_s, dtype=np.float64)[:, sel])
        keys.extend((s.name, int(st)) for st in starts)
        fws.append(np.full(sel.size, w, dtype=np.int64))
        fss.append(starts)
        names.append(s.name)
    if first is None:
        raise ValueError(f"support set {SUPPORT_SET!r} has no 8 s segment")
    scale = float(n_read) / float(n_thin)
    weights = composite_weights(keys, hop=int(first.hop), n_fft=int(first.n_fft)) * scale
    keep = slice(None) if max_frames is None else slice(0, int(max_frames))
    return FlightPool(
        names=tuple(names),
        power=np.concatenate(powers, axis=1)[:, keep],
        rps=np.concatenate(rpss, axis=1)[:, keep],
        weights=weights[keep],
        frame_window=np.concatenate(fws)[keep],
        frame_start=np.concatenate(fss)[keep],
        freqs_hz=np.asarray(first.freqs_hz, dtype=np.float64),
        sr=int(first.sr),
        n_fft=int(first.n_fft),
        hop=int(first.hop),
        stride=int(stride),
        n_frames_read=n_read,
        exposure_scale=scale,
    )


# ── the frozen part ─────────────────────────────────────────────────────────


def _bench_params(bench: dict[str, Any]) -> V4.V4Params:
    if bench.get("kind") != "bench" or bench.get("rig") != "dregon":
        raise ValueError(
            f"the flight stage starts from a DREGON v4 BENCH payload, got kind "
            f"{bench.get('kind')!r} rig {bench.get('rig')!r}"
        )
    par = V4.V4Params.from_payload(bench)
    if par.floor.rule != V4.FLOOR_RULE_SUM:
        raise ValueError(
            f"the bench floor rule must be {V4.FLOOR_RULE_SUM!r}, got {par.floor.rule!r}"
        )
    return par


def silent_floor(bench: dict[str, Any]) -> dict[str, Any]:
    """The bench payload with a floor (and a static share) that adds exactly nothing."""
    out = copy.deepcopy(bench)
    out["params"]["floor"].update(mean_db=SILENT_DB, static_rel=0.0)
    return out


def rotor_payload(payload: dict[str, Any], rotor: int) -> dict[str, Any]:
    """``payload`` with every rotor's line profiles but ``rotor``'s silenced
    (``SILENT_DB``: their line powers are exactly 0)."""
    out = copy.deepcopy(payload)
    p = out["params"]
    for key in ("profile_aero_db", "profile_motor_db"):
        rows = np.array(p[key], dtype=np.float64)
        rows[np.arange(rows.shape[0]) != int(rotor)] = SILENT_DB
        p[key] = rows.tolist()
    return out


def frozen_lines(
    bench: dict[str, Any], rps: np.ndarray, *, n_fft: int, sr: int, progress: int = 0
) -> np.ndarray:
    """``(R, M, N, n_fft // 2 + 1)``: per rotor ``r``, ``expected_periodogram``
    of the bench payload at each frame's speeds ``rps (R, N)`` with the floor
    off and the other rotors silenced: rotor ``r``'s lines, pedestal, shaft and
    pair EQ only. Asserts that nothing of the floor, its static share, a wind
    or another rotor can leak in, and that ``sum_r L_r`` is the all-rotor
    periodogram to :data:`SPLIT_RTOL`."""
    silent = silent_floor(bench)
    par = _bench_params(silent)
    f_psd = np.fft.rfftfreq(FLOOR_OVERSAMPLE * int(n_fft), d=1.0 / float(sr))
    if (
        np.any(par.floor.psd(f_psd) != 0.0)
        or par.floor.static_rel != 0.0
        or par.wind_db is not None
    ):
        raise AssertionError("the silenced bench floor still adds power")
    rotors = [rotor_payload(silent, r) for r in range(par.n_rotors)]
    for r, pay in enumerate(rotors):
        pr = V4.V4Params.from_payload(pay)
        others = np.arange(par.n_rotors) != r
        if np.any(10.0 ** (pr.profile_aero_db[others] / 10.0) != 0.0) or np.any(
            10.0 ** (pr.profile_motor_db[others] / 10.0) != 0.0
        ):
            raise AssertionError(f"rotor {r}'s payload still has power on another rotor")
    rps = np.asarray(rps, dtype=np.float64)
    shape = (par.n_rotors, par.n_mics, rps.shape[1], int(n_fft) // 2 + 1)
    out = np.empty(shape)
    split_err = 0.0
    t0 = time.time()
    for i in range(rps.shape[1]):
        for r, pay in enumerate(rotors):
            out[r, :, i] = V4.expected_periodogram(pay, rps[:, i], n_fft=int(n_fft), sr=int(sr))
        full = V4.expected_periodogram(silent, rps[:, i], n_fft=int(n_fft), sr=int(sr))
        split_err = max(split_err, float(np.max(np.abs(out[:, :, i].sum(axis=0) - full) / full)))
        if progress and (i + 1) % int(progress) == 0:
            print(f"  frozen lines {i + 1}/{rps.shape[1]}  {time.time() - t0:.1f}s", flush=True)
    if not split_err < SPLIT_RTOL:
        raise AssertionError(
            f"sum_r L_r differs from the all-rotor periodogram by {split_err:.3g} (relative)"
        )
    return out


def held_pedestal(bench: dict[str, Any]) -> dict[str, np.ndarray]:
    """The bench payload's own phase pedestal per order: ``sigma2_phase``,
    ``gamma_phase_hz`` (what the renderer reads) and ``width_phase_hz`` (a
    payload without it gets :func:`.v4_priors.pedestal_width`)."""
    par = _bench_params(bench)
    ped, raw = par.pedestal, bench["params"]["pedestal"]
    width = (
        np.asarray(raw["width_phase_hz"], dtype=np.float64)
        if "width_phase_hz" in raw
        else PR.pedestal_width(ped.gamma_phase_hz, ped.sigma2_phase)
    )
    return dict(
        sigma2_phase=np.asarray(ped.sigma2_phase, dtype=np.float64),
        gamma_phase_hz=np.asarray(ped.gamma_phase_hz, dtype=np.float64),
        width_phase_hz=width,
    )


def coherence_start(bench: dict[str, Any]) -> dict[str, np.ndarray]:
    """The coherence start: the bench payload's phase pedestal
    (:func:`held_pedestal`) clamped to at most :data:`START_CAP_FRACTION` of its
    caps, in the capped coordinates, and the bench ``log eps``. The bench parks
    orders a hair under a cap (``sigma2 = 2.999998``), where the capped
    coordinate's slope is ~1e-6: the clamp gives them a usable gradient
    (``coherence="full"``; ``"shaft"`` reads only ``log_eps``)."""
    par = _bench_params(bench)
    held = held_pedestal(bench)
    s2 = np.minimum(held["sigma2_phase"], START_CAP_FRACTION * PR.PHASE_SIGMA2_MAX)
    w = np.minimum(held["width_phase_hz"], START_CAP_FRACTION * PR.W_MAX_HZ)
    if np.any(s2 <= 0.0) or np.any(w <= 0.0) or par.shaft.eps <= 0.0:
        raise ValueError("the bench phase pedestal and eps must be positive to free them")
    return dict(
        u_s2P=BV._uncapped(s2, PR.PHASE_SIGMA2_MAX),
        u_W=BV._uncapped(w, PR.W_MAX_HZ),
        log_eps=np.asarray(math.log(par.shaft.eps)),
    )


def shaft_values(log_eps: Tensor, *, f_ref: float) -> dict[str, Tensor]:
    """``eps`` and ``sigma_ref = 2 pi f_ref eps`` (``sigma_nu`` at ``f_ref``, rad/s)."""
    eps = torch.exp(log_eps)
    return dict(eps=eps, sigma_ref=V4.TWO_PI * float(f_ref) * eps)


def coherence_values(u_s2P: Tensor, u_W: Tensor, log_eps: Tensor, *, f_ref: float) -> dict:
    """The full-coherence coordinates' values: ``sigma2_phase`` and
    ``width_phase_hz`` under their caps, ``gamma_phase_hz`` of that width
    (eq-width) and :func:`shaft_values`."""
    s2 = BV._capped(u_s2P, math.log(PR.PHASE_SIGMA2_MAX))
    width = BV._capped(u_W, math.log(PR.W_MAX_HZ))
    return dict(
        sigma2_phase=s2,
        width_phase_hz=width,
        gamma_phase_hz=BV._gamma_from_width(width, s2),
        **shaft_values(log_eps, f_ref=f_ref),
    )


class LineRenderer:
    """The LINE part of :func:`data_processing.noise_model.v4.expected_periodogram`
    in torch, differentiable in the phase pedestal ``(sigma2_phase_k,
    gamma_phase_k)``, the shaft scale ``eps`` and a per-rotor power scale.

    Exactly the numpy renderer's arithmetic, per frame ``n`` at the CONSTANT
    speeds ``f_rn = max(rps_rn, SPEED_FLOOR_RPS)`` (the frame-mean label): lags
    ``tau = 0 .. n_fft - 1``; every aerodynamic order ``k <= K`` and every motor
    order with ``k f_rn`` under the Nyquist (the others weigh 0);
    ``line_lag_law`` with the amplitude and phase pedestals and the shaft OU
    (``sigma_nu = 2 pi f_ref eps s^a_sigma``, ``lam = lam_ref s^b_lam``) for
    an aerodynamic line, the shaft alone for a motor line; times
    ``cos(2 pi k f_rn tau)``, the line power ``P_rk(s_rn)`` and the pair EQ
    ``10^{E_rm(k f_rn)/10}``; summed into ``c_mn(tau)``, weighted by the window
    autocorrelation ``(w * w)(tau) / sum w^2`` and read at the bins.
    Everything that no free parameter touches (the shaft's unit structure
    function, the amplitude pedestal, the line weights) is computed ONCE, on
    the device; ``cos`` is recomputed per chunk (an ``(n_c, R, K', n)`` table
    of all 310 frames would hold ~3 GB)."""

    def __init__(
        self, par: V4.V4Params, rps: np.ndarray, *, n_fft: int, sr: int, device: Any = "cpu"
    ) -> None:
        self.par = par
        self.dd: dict[str, Any] = dict(dtype=torch.float64, device=torch.device(device))
        n, sr = int(n_fft), int(sr)
        self.n = n
        R, M, K = par.n_rotors, par.n_mics, par.k_max
        rps = np.asarray(rps, dtype=np.float64)
        if rps.ndim != 2 or rps.shape[0] != R:
            raise ValueError(f"rps must be (R, N) with R = {R}, got {rps.shape}")
        f_r = np.maximum(rps, SPEED_FLOOR_RPS)  # (R, N)
        s = par.speed(f_r)
        tau = np.arange(n) / float(sr)
        k = np.concatenate([np.arange(1, K + 1, dtype=np.float64), par.motor_orders.astype(float)])
        self.n_aero, self.n_orders = K, int(k.size)
        p_aero, p_motor = par.line_power(s)  # (R, K, N), (R, n_m, N)
        power = np.concatenate([p_aero, p_motor], axis=1)  # (R, K', N)
        weights = np.zeros((rps.shape[1], M, R, k.size))
        for i in range(rps.shape[1]):
            for r in range(R):
                fk = k * f_r[r, i]
                keep = fk < 0.5 * sr
                gain = 10.0 ** (par.mic.db(fk[keep])[r] / 10.0)  # (M, K'_kept)
                weights[i, :, r][:, keep] = gain * power[r, keep, i][None, :]
        #: (N, M, R, K') pair EQ x line power, 0 above the Nyquist
        self.weights = self._t(weights)
        unit = replace(par.shaft, eps=1.0)
        #: (N, R) ``sigma_nu / eps`` and the unit-``sigma`` structure function (N, R, n)
        self.sig_unit = self._t(unit.sigma_nu(s).T)
        lam = par.shaft.lam(s).T[:, :, None]
        self.v_unit = self._t(integrated_ou_increment_var(tau[None, None, :], lam=lam, sigma=1.0))
        self.half_k2 = self._t(-0.5 * k**2)  # (K',)
        self.k = self._t(k)
        self.omega = self._t(V4.TWO_PI * f_r.T)  # (N, R)
        self.tau = self._t(tau)
        ped = par.pedestal
        amp = np.zeros((k.size, n))
        amp[:K] = ped.sigma2_amp[:, None] * np.expm1(-V4.TWO_PI * ped.gamma_amp_hz[:, None] * tau)
        self.amp = self._t(amp)  # (K', n) the held amplitude pedestal term
        self.motor_zero = torch.zeros((k.size - K, n), **self.dd)
        w = V4.hann_window(n)
        ww = np.correlate(w, w, mode="full")[n - 1 :]
        self.hw = self._t(ww / float(np.sum(w**2)))

    def _t(self, x: Any) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(**self.dd)
        return torch.as_tensor(np.asarray(x, dtype=np.float64), **self.dd)

    def expected(
        self,
        frames: slice | Tensor,
        *,
        sigma2_phase: Tensor,
        gamma_phase_hz: Tensor,
        eps: Tensor,
        scale: Tensor,
    ) -> Tensor:
        """``(M, n_c, n_fft // 2 + 1)`` the lines' expected periodogram of the
        frames ``frames``, rotor ``r``'s power times ``scale[r]``."""
        phase = sigma2_phase[:, None] * torch.expm1(-V4.TWO_PI * gamma_phase_hz[:, None] * self.tau)
        ped = self.amp + torch.cat([phase, self.motor_zero], dim=0)  # (K', n)
        sig2 = (eps * self.sig_unit[frames]) ** 2  # (n_c, R)
        shaft = (sig2[:, :, None] * self.half_k2)[..., None] * self.v_unit[frames][:, :, None, :]
        with torch.no_grad():
            arg = (self.omega[frames][:, :, None] * self.k)[..., None] * self.tau
            carrier = torch.cos(arg)  # (n_c, R, K', n)
        x = torch.exp(shaft + ped) * carrier
        wt = self.weights[frames] * scale[None, None, :, None]  # (n_c, M, R, K')
        n_c = wt.shape[0]
        c = torch.bmm(wt.reshape(n_c, wt.shape[1], -1), x.reshape(n_c, -1, self.n))
        return BV._lag_dtft(c.transpose(0, 1) * self.hw, self.n)

    def all_frames(self, *, chunk: int, **values: Tensor) -> np.ndarray:
        """``(M, N, n_fft // 2 + 1)`` :meth:`expected` on every frame, no gradient."""
        n, chunk = int(self.weights.shape[0]), max(1, int(chunk))
        with torch.no_grad():
            parts = [
                self.expected(slice(i, min(i + chunk, n)), **values).cpu().numpy()
                for i in range(0, n, chunk)
            ]
        return np.concatenate(parts, axis=1)


# ── the model ───────────────────────────────────────────────────────────────


class FlightStageModel:
    """The flight stage's forward model and MAP objective: ``E_m = 10^{h_m/10}
    [sum_r 10^{g_r/10} L_rm + T(sum_r h_rm F s_r^b)] + T[static + wind]`` with
    ``T`` the renderer's window transform; ``g_r`` free unless
    ``line_gain=False`` and ``h_m`` (``sum_m h_m = 0``) free unless
    ``mic_gain=False`` (then held at 0, with no prior term). ``h_m`` is a pair-EQ
    change, so it scales the per-rotor floor term (which the renderer weighs by
    the pair EQ) with the lines. ``L_r`` is frozen (``lines``) under
    ``coherence="none"``; under ``"shaft"`` / ``"full"`` :class:`LineRenderer`
    computes it from the shaft scale ``eps`` (free) and the phase pedestal (the
    bench's, held, under ``"shaft"``; free under ``"full"``) (``lines`` must be
    ``None``; ``renderer`` reuses one built on the pool's speeds, else one is
    built)."""

    def __init__(
        self,
        bench: dict[str, Any],
        pool: FlightPool,
        lines: np.ndarray | None,
        *,
        band_hz: tuple[float, float],
        kappa: float,
        floor_centre_db: float,
        wind_centre_db: np.ndarray,
        init: dict[str, Any],
        line_gain: bool = True,
        mic_gain: bool = True,
        coherence: str = "none",
        renderer: LineRenderer | None = None,
        chunk_frames: int = 16,
        device: Any = "cpu",
    ) -> None:
        par = _bench_params(bench)
        self.bench, self.par, self.pool = bench, par, pool
        self.dd: dict[str, Any] = dict(dtype=torch.float64, device=torch.device(device))
        self.n = int(pool.n_fft)
        self.n_psd = FLOOR_OVERSAMPLE * self.n
        self.kappa = float(kappa)
        self.chunk = max(1, int(chunk_frames))
        self.band_hz = (float(band_hz[0]), float(band_hz[1]))
        f_psd = np.fft.rfftfreq(self.n_psd, d=1.0 / float(pool.sr))
        shape, tilt = floor_geometry(f_psd, par.floor.ctrl_hz)
        self.shape, self.tilt = self._t(shape), self._t(tilt)
        self.chol = self._t(floor_shape_chol(par.floor.ctrl_hz))
        self.sigma_b = float(par.floor.shape_sd_db)
        # b(f) reads the frequency no lower than the payload's band edge (FloorLaw.exponent)
        self.l_f = self._t(np.log2(np.maximum(f_psd, par.floor.band_lo_hz) / V4.FLOOR_EXP_REF_HZ))
        self.gain = self._t(10.0 ** (par.mic.db(f_psd) / 10.0))  # (R, M, F_psd)
        self.array = self._t(10.0 ** (par.mic.array_db(f_psd) / 10.0))  # (M, F_psd)
        w = V4.hann_window(self.n)
        ww = np.correlate(w, w, mode="full")[self.n - 1 :]
        self.hw = self._t(ww / float(np.sum(w**2)))
        f = np.asarray(pool.freqs_hz, dtype=np.float64)
        self.band = np.flatnonzero((f >= self.band_hz[0]) & (f <= self.band_hz[1]))
        self.band_t = torch.as_tensor(self.band, device=self.dd["device"])
        self.log_s = self._t(np.log(np.maximum(par.speed(pool.rps), SPEED_FLOOR_RPS)))  # (R, N)
        if coherence not in COHERENCE_MODES:
            raise ValueError(f"coherence must be one of {COHERENCE_MODES}, got {coherence!r}")
        self.coherence = coherence
        if self.coherence != "none":
            if lines is not None:
                raise ValueError(f"coherence={coherence!r} renders the lines: pass lines=None")
            self.lines = None
            self.renderer = renderer or LineRenderer(
                par, pool.rps, n_fft=self.n, sr=pool.sr, device=self.dd["device"]
            )
            if self.renderer.weights.shape[0] != pool.n_frames or self.renderer.n != self.n:
                raise ValueError("the line renderer was built on another pool")
        else:
            if lines is None:
                raise ValueError("frozen coherence needs the frozen per-rotor lines")
            self.renderer = None
            self.lines = self._t(lines)  # (R, M, N, n // 2 + 1)
            if self.lines.shape[0] != par.n_rotors:
                raise ValueError(
                    f"lines must be per rotor (R={par.n_rotors}, ...), got {lines.shape}"
                )
        self.power = self._t(pool.power[:, :, self.band])
        self.weights = self._t(pool.weights)
        self.t_wind = self._transform(self._t(wind_shape(f_psd))[None, :] * self.array)
        self.floor_centre = float(floor_centre_db)
        self.wind_centre = self._t(wind_centre_db)
        self.n_cells = int(par.n_mics * pool.n_frames * self.band.size)
        self.params: dict[str, Tensor] = {
            k: torch.tensor(np.asarray(init[k], dtype=np.float64), **self.dd).requires_grad_(True)
            for k in PARAM_KEYS
        }
        self.line_gain = bool(line_gain)
        g0 = np.asarray(init.get(GAIN_KEY, np.zeros(par.n_rotors)), dtype=np.float64)
        if g0.shape != (par.n_rotors,):
            raise ValueError(f"{GAIN_KEY} must be (R,) = ({par.n_rotors},), got {g0.shape}")
        if not self.line_gain and np.any(g0 != 0.0):
            raise ValueError(f"line_gain=False holds {GAIN_KEY} at 0, got {g0.tolist()}")
        #: ``g_r`` (dB): in ``params`` when free, else a constant zero tensor
        self.line_gain_db = torch.tensor(g0, **self.dd).requires_grad_(self.line_gain)
        if self.line_gain:
            self.params[GAIN_KEY] = self.line_gain_db
        self.mic_gain = bool(mic_gain)
        h0 = np.asarray(init.get("mic_gain_db", np.zeros(par.n_mics)), dtype=np.float64)
        if h0.shape != (par.n_mics,) or abs(float(h0.sum())) > 1e-9:
            raise ValueError(f"mic_gain_db must be (M,) = ({par.n_mics},) summing to 0, got {h0}")
        if not self.mic_gain and np.any(h0 != 0.0):
            raise ValueError(f"mic_gain=False holds mic_gain_db at 0, got {h0.tolist()}")
        #: ``h_1 .. h_{M-1}`` (dB): in ``params`` when free, else constant zeros
        self.mic_gain_free_db = torch.tensor(h0[:-1], **self.dd).requires_grad_(self.mic_gain)
        if self.mic_gain:
            self.params[MIC_GAIN_KEY] = self.mic_gain_free_db
        if self.coherence != "none":
            c0 = coherence_start(bench)
            for key in COHERENCE_FREE[self.coherence]:
                self.params[key] = torch.tensor(
                    np.asarray(init.get(key, c0[key]), dtype=np.float64), **self.dd
                ).requires_grad_(True)
        if self.coherence == "full":
            #: the phase-width half-normal scales per aerodynamic order
            self.c_w = self._t(PR.width_scale(np.arange(1, par.k_max + 1)))
        elif self.coherence == "shaft":
            #: the held phase pedestal: the bench payload's own values, no gradient
            self.held_pedestal = {k: self._t(a) for k, a in held_pedestal(bench).items()}

    @property
    def move_tol(self) -> dict[str, float]:
        """The convergence movement tolerance per snapshot group."""
        return dict(MOVE_TOL, **{k: COHERENCE_MOVE_TOL[k] for k in COHERENCE_MOVES[self.coherence]})

    def _t(self, x: Any) -> Tensor:
        if isinstance(x, Tensor):
            return x.to(**self.dd)
        return torch.as_tensor(np.asarray(x, dtype=np.float64), **self.dd)

    def _transform(self, psd: Tensor) -> Tensor:
        """``expected_periodogram``'s window transform: the PSD's autocovariance
        on the fine grid, times ``(w * w)(tau) / sum w^2``, read at the bins."""
        c = torch.fft.irfft(psd.to(torch.complex128), n=self.n_psd, dim=-1)[..., : self.n]
        return BV._lag_dtft(c * self.hw, self.n)

    def values(self) -> dict[str, Tensor]:
        p = self.params
        b0, b1 = PR.floor_exponents(p["u_blo"], p["u_bhi"])
        ctrl_rel = self.sigma_b * (self.chol @ p["floor_z"])
        out = dict(
            p,
            line_gain_db=self.line_gain_db,
            mic_gain_db=FV.FlightV4Model._full_c(self.mic_gain_free_db),
            b0=b0,
            b1=b1,
            static_rel=torch.exp(p["log_static"]),
            ctrl_rel=ctrl_rel,
            ctrl_db=p["floor_mean_db"] + ctrl_rel,
        )
        f_ref = self.par.shaft.f_ref_rev_s
        if self.coherence == "full":
            out.update(coherence_values(p["u_s2P"], p["u_W"], p["log_eps"], f_ref=f_ref))
        elif self.coherence == "shaft":
            out.update(self.held_pedestal, **shaft_values(p["log_eps"], f_ref=f_ref))
        return out

    def expected(self, frames: slice | Tensor, *, all_bins: bool = False) -> Tensor:
        """``(M, n_c, F)`` expected periodogram of the pool frames ``frames``
        (the band's bins unless ``all_bins``)."""
        v = self.values()
        fl = floor_power_spectrum(
            self.shape,
            self.tilt,
            mean_db=v["floor_mean_db"],
            ctrl_db=v["ctrl_rel"],
            tilt_db_oct=0.0,
            rate_factor=1.0,
        )
        b = v["b0"] + v["b1"] * self.l_f
        rotor = fl * torch.exp(self.log_s[:, frames, None] * b)  # (R, n_c, F_psd)
        # sum over the rotors of h_rm F s_rn^b (3x faster than the einsum on CPU)
        floor = self._transform((self.gain[:, :, None, :] * rotor[:, None, :, :]).sum(dim=0))
        fixed = (
            v["static_rel"] * self._transform(fl[None, :] * self.array)
            + 10.0 ** (v["wind_db"][:, None] / 10.0) * self.t_wind
        )
        scale = 10.0 ** (v["line_gain_db"] / 10.0)
        if self.renderer is not None:
            lines = self.renderer.expected(
                frames,
                sigma2_phase=v["sigma2_phase"],
                gamma_phase_hz=v["gamma_phase_hz"],
                eps=v["eps"],
                scale=scale,
            )
        else:
            lines = torch.tensordot(scale, self.lines[:, :, frames], dims=1)  # (M, n_c, F)
        # h_m is a pair-EQ change: the lines and the per-rotor floor (both weighed by
        # the pair EQ, in either line path), not the static share or the wind
        mic = 10.0 ** (v["mic_gain_db"] / 10.0)
        out = mic[:, None, None] * (lines + floor) + fixed[:, None, :]
        return out if all_bins else out.index_select(-1, self.band_t)

    def log_prior(self) -> tuple[Tensor, dict[str, float]]:
        """``log p`` of the free values plus the floor push ``kappa sum_j ctrl_j``
        (the line and mic gains' terms only when they are free; the shaft
        scale's under ``coherence="shaft"``/``"full"``, the phase pedestal's only
        under ``"full"``)."""
        v = self.values()
        nd, ln = torch.distributions.Normal, torch.distributions.LogNormal

        def n_(loc: Any, sd: float, x: Tensor) -> Tensor:
            return nd(self._t(loc), float(sd)).log_prob(x).sum()

        parts = dict(
            floor_mean=n_(self.floor_centre, BV.FLOOR_MEAN_SD_DB, v["floor_mean_db"]),
            floor_z=n_(0.0, 1.0, v["floor_z"]),
            floor_exp=n_(PR.B0[0], PR.B0[1], v["b0"]) + n_(PR.B1[0], PR.B1[1], v["b1"]),
            static=ln(self._t(FV.STATIC_SHARE_LN[0]), FV.STATIC_SHARE_LN[1]).log_prob(
                v["static_rel"]
            ),
            wind=n_(self.wind_centre, WIND_PRIORS.wind_db_sd, v["wind_db"]),
            floor_push=self.kappa * v["ctrl_db"].sum(),
        )
        if self.line_gain:
            parts["line_gain"] = n_(LINE_GAIN_MU_DB, LINE_GAIN_SD_DB, v["line_gain_db"])
        if self.mic_gain:
            parts["mic_gain"] = n_(0.0, MIC_GAIN_SD_DB, v["mic_gain_db"])
        if self.coherence == "full":
            parts["phase_pedestal"] = -BV.phase_pedestal_neg_log_prior(
                v["sigma2_phase"], v["width_phase_hz"], self.c_w
            )
        if self.coherence != "none":
            parts["sigma_ref"] = ln(
                self._t(PR.FLIGHT_SIGMA_NU_LN[0]), PR.FLIGHT_SIGMA_NU_LN[1]
            ).log_prob(v["sigma_ref"])
        total = torch.stack([t.reshape(()) for t in parts.values()]).sum()
        return total, {k: float(t.detach()) for k, t in parts.items()}

    def _chunks(self) -> list[slice]:
        n = self.pool.n_frames
        return [slice(i, min(i + self.chunk, n)) for i in range(0, n, self.chunk)]

    def objective(self, *, grad: bool) -> Tensor:
        """``Whittle - log p``; the Whittle term chunk by chunk with the
        gradient accumulated per chunk (one chunk's graph alive at a time)."""
        with torch.set_grad_enabled(grad):
            lp, _ = self.log_prior()
            nlp = -lp
            if grad:
                nlp.backward()
            total = nlp.detach()
            for sl in self._chunks():
                w = composite_risk(self.power[:, sl], self.expected(sl), self.weights[sl])
                if grad:
                    w.backward()
                total = total + w.detach()
        return total

    @torch.no_grad()
    def whittle(self) -> float:
        return float(
            sum(
                float(composite_risk(self.power[:, sl], self.expected(sl), self.weights[sl]))
                for sl in self._chunks()
            )
        )

    @torch.no_grad()
    def snapshot(self) -> dict[str, np.ndarray]:
        v = self.values()
        out = dict(
            floor_ctrl_db=v["ctrl_db"].cpu().numpy().copy(),
            exponents=np.array([float(v["b0"]), float(v["b1"])]),
            wind_db=v["wind_db"].cpu().numpy().copy(),
            log_static=np.array([float(v["log_static"])]),
            line_gain_db=v["line_gain_db"].cpu().numpy().copy(),
            mic_gain_db=v["mic_gain_db"].cpu().numpy().copy(),
        )
        if self.coherence == "full":
            out.update(
                log_sigma2_phase=torch.log(v["sigma2_phase"]).cpu().numpy(),
                log_width_phase=torch.log(v["width_phase_hz"]).cpu().numpy(),
            )
        if self.coherence != "none":
            out.update(log_eps=np.array([float(v["log_eps"])]))
        return out

    @torch.no_grad()
    def exported(self) -> tuple[dict[str, Any], list[float], list[float], list[float]]:
        """``(floor fields, wind_db, line_gain_db, mic_gain_db)`` of the payload at
        the current values."""
        v = self.values()
        floor = dict(
            mean_db=float(v["floor_mean_db"]),
            shape_z=v["floor_z"].cpu().numpy().tolist(),
            ctrl_db=v["ctrl_db"].cpu().numpy().tolist(),
            b0=float(v["b0"]),
            b1=float(v["b1"]),
            static_rel=float(v["static_rel"]),
        )
        return (
            floor,
            v["wind_db"].cpu().numpy().tolist(),
            v["line_gain_db"].cpu().numpy().tolist(),
            v["mic_gain_db"].cpu().numpy().tolist(),
        )

    @torch.no_grad()
    def exported_coherence(self) -> dict[str, Any] | None:
        """The payload's coherence fields at the current values (``None`` under
        ``coherence="none"``): the shaft ``eps``, and under ``"full"`` the phase
        pedestal ``sigma2_phase``, ``gamma_phase_hz`` and ``width_phase_hz`` per
        order (under ``"shaft"`` the pedestal is the bench's, left as it is)."""
        if self.coherence == "none":
            return None
        v = self.values()
        out: dict[str, Any] = dict(shaft=dict(eps=float(v["eps"])))
        if self.coherence == "full":
            out["pedestal"] = {
                k: v[k].cpu().numpy().tolist()
                for k in ("sigma2_phase", "gamma_phase_hz", "width_phase_hz")
            }
        return out


def flight_payload(
    bench: dict[str, Any],
    *,
    floor: dict[str, Any],
    wind_db: list[float],
    line_gain_db: list[float],
    mic_gain_db: list[float],
    coherence: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The bench payload with ``kind="flight"``, the line gains written into the
    profiles (``profile_aero_db[r] += g_r``, ``profile_motor_db[r] += g_r``),
    the mic gains into the pair EQ (``pair_eq.db[:, m, :] += h_m``: the pair EQ
    is linearly interpolated, so ``E_rm(f)`` moves by exactly ``h_m``), the
    floor fields ``floor`` and the static per-mic wind ``wind_db``; ``ctrl_hz``,
    ``shape_sd_db`` and the per-rotor-sum floor rule are the bench's.
    ``coherence`` (:meth:`FlightStageModel.exported_coherence`) replaces the
    shaft ``eps`` and, when it carries one (``coherence="full"``), the phase
    pedestal (then dropping the bench's ``held_by_evidence``, which no longer
    describes it)."""
    pay = copy.deepcopy(bench)
    pay["kind"] = "flight"
    p = pay["params"]
    g = np.asarray(line_gain_db, dtype=np.float64)
    for key in ("profile_aero_db", "profile_motor_db"):
        rows = np.array(p[key], dtype=np.float64)
        if rows.shape[0] != g.size:
            raise ValueError(f"{key} has {rows.shape[0]} rotors, line_gain_db {g.size}")
        p[key] = (rows + g[:, None]).tolist()
    mic = p["mic_response"]
    if mic.get("kind") != "pair_eq":
        raise ValueError(f"the mic gains go into a pair EQ, got kind {mic.get('kind')!r}")
    eq = np.array(mic["pair_eq"]["db"], dtype=np.float64)  # (R, M, Fq)
    h = np.asarray(mic_gain_db, dtype=np.float64)
    if eq.shape[1] != h.size:
        raise ValueError(f"pair_eq.db has {eq.shape[1]} mics, mic_gain_db {h.size}")
    mic["pair_eq"]["db"] = (eq + h[None, :, None]).tolist()
    p["floor"].update(floor, floor_rule=V4.FLOOR_RULE_SUM)
    p["wind"] = dict(
        kind="static_per_mic", wind_db=[float(x) for x in wind_db], shape=wind_shape_spec()
    )
    if coherence is not None:
        if "pedestal" in coherence:
            p["pedestal"].update(coherence["pedestal"])
            p["pedestal"].pop("held_by_evidence", None)
        p["shaft"].update(coherence["shaft"])
    return pay


def renderer_check(
    model: FlightStageModel, frames: list[int], *, seed: int = 0, rtol: float = CHECK_RTOL
) -> dict[str, Any]:
    """The torch ``E`` against ``expected_periodogram`` of the exported payload,
    on every bin of the pool frames ``frames``, at RANDOM parameter values,
    nonzero line and mic gains included even when they are held (restored afterwards);
    under ``coherence="shaft"``/``"full"`` also a random ``eps`` and under
    ``"full"`` a random phase pedestal per order, drawn from their priors."""
    rng = np.random.default_rng(int(seed))
    p = model.params
    saved = {k: t.detach().clone() for k, t in p.items()}
    saved_gain = model.line_gain_db.detach().clone()
    saved_mic = model.mic_gain_free_db.detach().clone()
    try:
        with torch.no_grad():
            p["floor_mean_db"].add_(float(rng.uniform(-6.0, 6.0)))
            p["floor_z"].copy_(model._t(rng.standard_normal(p["floor_z"].numel())))
            p["u_blo"].fill_(float(rng.uniform(-1.0, 8.0)))
            p["u_bhi"].fill_(float(rng.uniform(-1.0, 4.0)))
            p["log_static"].fill_(math.log(float(rng.uniform(1e-4, 0.1))))
            p["wind_db"].add_(model._t(rng.normal(0.0, 6.0, p["wind_db"].numel())))
            model.line_gain_db.copy_(
                model._t(rng.normal(0.0, LINE_GAIN_SD_DB, model.line_gain_db.numel()))
            )
            h = rng.normal(0.0, MIC_GAIN_SD_DB, model.par.n_mics)
            model.mic_gain_free_db.copy_(model._t((h - h.mean())[:-1]))
            if model.coherence == "full":
                k = p["u_s2P"].numel()
                s2 = np.exp(rng.normal(PR.PHASE_SIGMA2_LN[0], PR.PHASE_SIGMA2_LN[1], k))
                width = np.abs(rng.normal(0.0, model.c_w.cpu().numpy()))
                p["u_s2P"].copy_(
                    model._t(BV._uncapped(BV._under(s2, PR.PHASE_SIGMA2_MAX), PR.PHASE_SIGMA2_MAX))
                )
                p["u_W"].copy_(
                    model._t(
                        BV._uncapped(BV._under(np.maximum(width, 1e-3), PR.W_MAX_HZ), PR.W_MAX_HZ)
                    )
                )
            if model.coherence != "none":
                sigma_ref = math.exp(rng.normal(*PR.FLIGHT_SIGMA_NU_LN))
                p["log_eps"].fill_(math.log(sigma_ref / (V4.TWO_PI * model.par.shaft.f_ref_rev_s)))
            idx = torch.as_tensor(np.asarray(frames, dtype=np.int64), device=model.dd["device"])
            mine = model.expected(idx, all_bins=True).cpu().numpy()
        floor, wind, gain, mic_gain = model.exported()
        coherence = model.exported_coherence()
        pay = flight_payload(
            model.bench,
            floor=floor,
            wind_db=wind,
            line_gain_db=gain,
            mic_gain_db=mic_gain,
            coherence=coherence,
        )
        if V4.V4Params.from_payload(pay).wind_db is None:
            raise AssertionError("the exported flight payload carries no wind")
        ref = np.stack(
            [
                V4.expected_periodogram(pay, model.pool.rps[:, i], n_fft=model.n, sr=model.pool.sr)
                for i in frames
            ],
            axis=1,
        )
    finally:
        with torch.no_grad():
            for k, t in saved.items():
                p[k].copy_(t)
            model.line_gain_db.copy_(saved_gain)
            model.mic_gain_free_db.copy_(saved_mic)
    rel = np.abs(mine - ref) / np.abs(ref)
    return dict(
        frames=[int(i) for i in frames],
        seed=int(seed),
        n_fft=model.n,
        max_rel_err=float(rel.max()),
        band_max_rel_err=float(rel[..., model.band].max()),
        rtol=float(rtol),
        passed=bool(rel.max() < rtol),
        at=dict(
            floor=floor,
            wind_db=wind,
            line_gain_db=gain,
            mic_gain_db=mic_gain,
            coherence=coherence,
        ),
    )


# ── the start and the prior centres ─────────────────────────────────────────


@dataclass
class Start:
    init: dict[str, Any]
    floor_centre_db: float
    wind_centre_db: np.ndarray
    record: dict[str, Any]


def wind_start(path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """nv3r4's DREGON wind in the UNNORMALISED units: its ``wind_db`` plus the
    channel gain its data were divided by."""
    fit = json.loads(Path(path).read_text())
    w = np.asarray(fit["params"]["wind"]["wind_db"], dtype=np.float64)
    gains = ((fit.get("diagnostics") or {}).get("batch") or {}).get("channel_gains") or {}
    g = np.asarray(gains.get("gains_db", np.zeros_like(w)), dtype=np.float64)
    return w + g, dict(
        source=str(path),
        wind_db=w.tolist(),
        channel_gains_db=g.tolist(),
        rule="nv3r4 wind_db + its channel gain (its data were divided by 10^{g/10}); a start, "
        "not a prior",
    )


def measure_start(
    bench: dict[str, Any],
    pool: FlightPool,
    lines: np.ndarray,
    *,
    band_hz: tuple[float, float],
    wind_db: np.ndarray,
) -> Start:
    """The floor measured from the pool at the bench exponents and ``g_r = 0``:
    per band bin, the nv3r4 floor quantile (+ its exponential-percentile
    correction) over mics and frames of ``max(I - L - W0, 0.1 I) / sum_r h_rm
    s_rn^b`` with ``L = sum_r L_r`` (``lines (R, M, N, F)``, or their sum as
    ``(1, M, N, F)``), least squares on the 14 controls; the wind prior centres
    by nv3r4's rule."""
    par = _bench_params(bench)
    f_all = np.asarray(pool.freqs_hz, dtype=np.float64)
    band = (f_all >= band_hz[0]) & (f_all <= band_hz[1])
    f = f_all[band]
    power = pool.power[:, :, band]
    wind = 10.0 ** (np.asarray(wind_db)[:, None] / 10.0) * wind_shape(f)[None, :]
    resid = np.maximum(power - lines[:, :, :, band].sum(axis=0) - wind[:, None, :], 0.1 * power)
    ls = np.log(np.maximum(par.speed(pool.rps), SPEED_FLOOR_RPS))
    drive = np.einsum(
        "rmf,rnf->mnf",
        10.0 ** (par.mic.db(f) / 10.0),
        np.exp(ls[:, :, None] * par.floor.exponent(f)[None, None, :]),
    )
    db = 10.0 * np.log10(resid / drive).reshape(-1, f.size)
    level = np.quantile(db, FLOOR_INIT_QUANTILE, axis=0) + FLOOR_INIT_DB_CORRECTION
    shape, _ = floor_geometry(f, par.floor.ctrl_hz)
    ctrl = np.linalg.lstsq(shape, level, rcond=None)[0]
    mean = float(ctrl.mean())
    z = BV._ridge_z(floor_shape_chol(par.floor.ctrl_hz), float(par.floor.shape_sd_db), ctrl - mean)
    u_lo, u_hi = PR.floor_coords(par.floor.b0, par.floor.b1)
    # nv3r4's measured per-mic low-band excess, read off the pool's own frames
    centres = FT._wind_centres(
        SimpleNamespace(power=torch.as_tensor(pool.power), grid=SimpleNamespace(freqs_hz=f_all))
    )
    init = dict(
        floor_mean_db=mean,
        floor_z=z,
        u_blo=u_lo,
        u_bhi=u_hi,
        log_static=FV.STATIC_SHARE_LN[0],
        wind_db=np.asarray(wind_db, dtype=np.float64),
        line_gain_db=np.zeros(par.n_rotors),
        mic_gain_db=np.zeros(par.n_mics),
    )
    return Start(
        init=init,
        floor_centre_db=mean,
        wind_centre_db=centres,
        record=dict(
            floor_ctrl_db=ctrl.tolist(),
            floor_rule="per band bin: quantile FLOOR_INIT_QUANTILE (+ FLOOR_INIT_DB_CORRECTION) "
            "over mics and frames of max(I - L - W0, 0.1 I) / sum_r h_rm s_rn^b at the bench b; "
            "least squares on the controls; z the bench's ridge solve",
            floor_quantile=float(FLOOR_INIT_QUANTILE),
            floor_db_correction=float(FLOOR_INIT_DB_CORRECTION),
            exponents="the bench payload's b0, b1",
            line_gain_db=np.zeros(par.n_rotors).tolist(),
            mic_gain_db=np.zeros(par.n_mics).tolist(),
            static_rel=math.exp(FV.STATIC_SHARE_LN[0]),
            wind_centre_db=np.asarray(centres).tolist(),
        ),
    )


# ── the optimiser ───────────────────────────────────────────────────────────


#: Per-order groups that move by their median over orders (as in the bench test,
#: :func:`.bench_v4.group_movement`); every other group by its largest change.
PER_ORDER_GROUPS = ("log_sigma2_phase", "log_width_phase")


def movement(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    """Change per snapshot group: the largest absolute change, or the median over
    orders for :data:`PER_ORDER_GROUPS` (a pedestal on an order the data do not
    see drifts towards its prior mode at W = 0 without end in log W)."""
    return {
        k: float((np.median if k in PER_ORDER_GROUPS else np.max)(np.abs(b[k] - a[k]))) for k in a
    }


def _pass(model: FlightStageModel, iters: int, history: int) -> dict[str, Any]:
    """One fresh-memory strong-Wolfe L-BFGS pass, with its gain and movement."""
    opt = LBFGS(
        list(model.params.values()),
        max_iter=int(iters),
        history_size=int(history),
        line_search_fn="strong_wolfe",
    )
    evals = 0

    def closure() -> Tensor:
        nonlocal evals
        opt.zero_grad(set_to_none=False)
        evals += 1
        return model.objective(grad=True)

    before_v = model.snapshot()
    t0 = time.time()
    before = float(model.objective(grad=False))
    opt.step(closure)
    after = float(model.objective(grad=False))
    moved = movement(before_v, model.snapshot())
    gain_cell = max(0.0, before - after) / max(1, model.n_cells)
    move_ok = {k: bool(moved[k] < tol) for k, tol in model.move_tol.items()}
    return dict(
        loss_before=before,
        loss_after=after,
        iters=int(opt.n_iter),
        evals=int(evals),
        wall_s=time.time() - t0,
        gain_per_cell=gain_cell,
        gain_ok=bool(gain_cell < TOL_NATS_PER_CELL),
        movement=moved,
        movement_ok=move_ok,
        trace=[float(x) for x in opt.trace],
    )


def fit_stage(
    model: FlightStageModel,
    *,
    iters: int = 200,
    restart_iters: int = 100,
    history: int = 10,
    max_passes: int = 20,
    progress: bool = False,
) -> dict[str, Any]:
    """L-BFGS passes until a FRESH pass (the second or later) passes the test:
    gain per cell under :data:`TOL_NATS_PER_CELL` and every group's movement
    under its tolerance (``model.move_tol``: :data:`MOVE_TOL`, plus
    :data:`COHERENCE_MOVE_TOL`'s groups the coherence mode frees); every pass recorded."""
    t0 = time.time()
    with torch.no_grad():
        start = float(model.objective(grad=False))
    passes: list[dict[str, Any]] = []
    converged, stop = False, "max_passes"
    while len(passes) < int(max_passes):
        out = _pass(model, iters if not passes else restart_iters, history)
        out["converged"] = bool(passes) and out["gain_ok"] and all(out["movement_ok"].values())
        passes.append(out)
        if progress:
            print(
                f"  pass {len(passes)}: {out['loss_after']:.10g} nats  gain/cell "
                f"{out['gain_per_cell']:.3g}  moved "
                + " ".join(f"{k} {v:.3g}" for k, v in out["movement"].items())
                + f"  {out['iters']} it / {out['evals']} ev  {out['wall_s']:.1f}s",
                flush=True,
            )
        if out["converged"]:
            converged, stop = True, "converged"
            break
    with torch.no_grad():
        lp, lp_parts = model.log_prior()
        whittle = model.whittle()
    total = whittle - float(lp)
    last = passes[-1]
    return dict(
        objective=dict(
            total_nats=total,
            whittle_nats=whittle,
            neg_log_prior_nats=-float(lp),
            log_prior_parts=lp_parts,
            n_cells=model.n_cells,
            total_per_cell=total / max(1, model.n_cells),
            start_total_nats=start,
        ),
        optimiser=dict(
            method="torch L-BFGS (experiments.noise_model.lbfgs.LBFGS), strong Wolfe, float64",
            iters=int(iters),
            restart_iters=int(restart_iters),
            history=int(history),
            max_passes=int(max_passes),
            chunk_frames=model.chunk,
            device=str(model.dd["device"]),
            passes=passes,
            convergence=dict(
                test="a fresh-memory pass after the first: gain per cell < tol AND every group's "
                "max movement < its tolerance",
                tol_nats_per_cell=TOL_NATS_PER_CELL,
                move_tol=model.move_tol,
                gain_per_cell=last["gain_per_cell"],
                movement=last["movement"],
                converged=bool(converged),
                stop=stop,
                n_passes=len(passes),
            ),
            converged=bool(converged),
            wall_s=time.time() - t0,
        ),
    )


# ── the stage ───────────────────────────────────────────────────────────────


def priors_record(model: FlightStageModel, *, kappa_source: str) -> dict[str, Any]:
    out = dict(
        floor_mean=dict(
            family="Normal",
            centre_db=model.floor_centre,
            sd_db=BV.FLOOR_MEAN_SD_DB,
            rule="bench_v4 floor-mean prior; centre the mean of the floor measured on this pool",
        ),
        floor_z="N(0, I); control values mean_db + shape_sd_db (L z)",
        b0=list(PR.B0),
        b1=list(PR.B1),
        b_edges_hz=list(PR.B_EDGES_HZ),
        b_rule="b = softplus(u) >= 0 at both edges (v4_priors.floor_exponents)",
        static_rel_lognormal=list(FV.STATIC_SHARE_LN),
        wind=dict(
            family="Normal",
            centre_db=model.wind_centre.cpu().numpy().tolist(),
            sd_db=float(WIND_PRIORS.wind_db_sd),
            rule="nv3r4 (PRIOR_SETS['v4'], --wind): fit._wind_centres, the per-mic low-band "
            "excess, here on the unnormalised pool",
            shape=wind_shape_spec(),
        ),
        floor_push_kappa=model.kappa,
        kappa_source=kappa_source,
        floor_push="+kappa sum_j ctrl_j on the log prior (dB)",
        line_gain=dict(
            family="Normal",
            mu_db=LINE_GAIN_MU_DB,
            sd_db=LINE_GAIN_SD_DB,
            free=model.line_gain,
            rule="per rotor, on both line families; centred on the bench level, a 6 dB scale "
            "for the bench -> flight level change (four rotors in flight vs one on the bench)"
            if model.line_gain
            else "held at 0 (--no-line-gain): no prior term",
        ),
        mic_gain=dict(
            family="Normal",
            mu_db=0.0,
            sd_db=MIC_GAIN_SD_DB,
            free=model.mic_gain,
            constraint="sum_m h_m = 0 (free h_1..h_{M-1}, h_M = -sum): the common level is g_r's",
            rule="per mic, on every rotor's lines and the per-rotor floor (a pair-EQ change); "
            "flight vs bench per-mic level, on the scale of nv3r4's per-mic level spread"
            if model.mic_gain
            else "held at 0 (--no-mic-gain): no prior term",
        ),
    )
    if model.coherence == "full":
        out.update(
            phase_pedestal=dict(
                sigma2_phase=dict(
                    family="LogNormal (Normal on log sigma2), truncated",
                    ln=list(PR.PHASE_SIGMA2_LN),
                    cap=PR.PHASE_SIGMA2_MAX,
                ),
                width_phase_hz=dict(
                    family="HalfNormal, truncated",
                    scale_hz="v4_priors.width_scale(k)",
                    c_w_hz=PR.C_W,
                    cap_hz=PR.W_MAX_HZ,
                ),
                rule="bench_v4.phase_pedestal_neg_log_prior in the capped coordinates "
                "(bench_v4._capped), per aerodynamic order",
            )
        )
    if model.coherence != "none":
        out.update(
            sigma_ref=dict(
                family="LogNormal",
                ln=list(PR.FLIGHT_SIGMA_NU_LN),
                at_f_ref_rev_s=model.par.shaft.f_ref_rev_s,
                rule="v4_priors.FLIGHT_SIGMA_NU_LN (flight sigma_nu at f_ref, rad/s; flight_v4's "
                "sigma_ref) on sigma_ref = 2 pi f_ref eps, the DREGON sigma_nu at the payload's "
                "f_ref; a_sigma, lam_ref and b_lam held at the bench's",
            ),
        )
    return out


def start_coherence_values(
    bench: dict[str, Any], coherence: str, t: Callable[[Any], Tensor]
) -> dict[str, Tensor]:
    """The line coherence at the start of mode ``coherence`` (``"shaft"`` or
    ``"full"``), as tensors ``t(.)``: :func:`coherence_start` through
    :func:`coherence_values` (``"full"``), or the held bench pedestal and the
    bench ``eps`` (``"shaft"``)."""
    c0, f_ref = coherence_start(bench), _bench_params(bench).shaft.f_ref_rev_s
    if coherence == "full":
        return coherence_values(*(t(c0[k]) for k in COHERENCE_KEYS), f_ref=f_ref)
    if coherence == "shaft":
        held = {k: t(a) for k, a in held_pedestal(bench).items()}
        return dict(held, **shaft_values(t(c0["log_eps"]), f_ref=f_ref))
    raise ValueError(f"no start coherence for coherence={coherence!r}")


def coherence_record(model: FlightStageModel, bench: dict[str, Any]) -> dict[str, Any]:
    """``flight_stage.coherence``: the mode, what of the line coherence was free,
    its start, its priors and its fitted values (the bench's, held, when frozen)."""
    ped, shaft = bench["params"]["pedestal"], bench["params"]["shaft"]
    mode = model.coherence
    if mode == "none":
        return dict(
            mode=mode,
            free=False,
            rule="phase pedestal and shaft held at the bench payload's (frozen L_r)",
        )
    full = mode == "full"
    with torch.no_grad():
        s = start_coherence_values(bench, mode, model._t)
        v = model.values()

    def arrays(d: dict[str, Tensor]) -> dict[str, Any]:
        return dict(
            sigma2_phase=d["sigma2_phase"].cpu().numpy().tolist(),
            width_phase_hz=d["width_phase_hz"].cpu().numpy().tolist(),
            gamma_phase_hz=d["gamma_phase_hz"].cpu().numpy().tolist(),
            eps=float(d["eps"]),
            sigma_ref_rad_s=float(d["sigma_ref"]),
        )

    return dict(
        mode=mode,
        free=True,
        free_keys=list(COHERENCE_FIELDS[mode]),
        derived=["pedestal.gamma_phase_hz = W (1 - e^-sigma2) / sigma2 (eq-width)"] if full else [],
        held=dict(
            pedestal=["sigma2_amp", "gamma_amp_hz"]
            if full
            else ["sigma2_amp", "gamma_amp_hz", "sigma2_phase", "gamma_phase_hz", "width_phase_hz"],
            shaft=dict(
                a_sigma=float(shaft["a_sigma"]),
                lam_ref_per_s=float(shaft["lam_ref_per_s"]),
                b_lam=float(shaft["b_lam"]),
            ),
        ),
        coordinates="u_s2P, u_W: bench_v4._capped under PHASE_SIGMA2_MAX / W_MAX_HZ; log eps"
        if full
        else "log eps",
        f_ref_rev_s=model.par.shaft.f_ref_rev_s,
        why="the scorer's carrier is the motors_command LABEL; its error in flight acts as "
        "phase wander the bench (measured carrier) never saw, so the bench pedestal and shaft "
        "are too coherent for flight"
        if full
        else "the 2026-10-23 probe: the free phase pedestal went to its caps and HPPNet got "
        "worse (PIT 29 vs 16); the pedestal smears each order independently, the shaft "
        "wander all orders of a rotor coherently (FM, how nv3r4 absorbed the flight smear), "
        "so the flight smear goes only into the shaft",
        start=dict(
            arrays(s),
            rule="the bench pedestal (sigma2_phase, width_phase_hz) clamped to at most "
            f"{START_CAP_FRACTION:g} of its cap, and the bench eps"
            if full
            else "the bench pedestal (held, unclamped) and the bench eps",
        ),
        bench=dict(
            sigma2_phase=ped["sigma2_phase"],
            width_phase_hz=ped.get("width_phase_hz"),
            held_by_evidence=ped.get("held_by_evidence"),
            eps=float(shaft["eps"]),
        ),
        fitted=arrays(v),
        move_tol={k: COHERENCE_MOVE_TOL[k] for k in COHERENCE_MOVES[mode]},
        export="params.pedestal.{sigma2_phase, gamma_phase_hz, width_phase_hz} and "
        "params.shaft.eps replaced; pedestal.held_by_evidence dropped"
        if full
        else "params.shaft.eps replaced; params.pedestal the bench's, unchanged",
    )


def run_stage(
    bench: dict[str, Any],
    *,
    bench_path: str,
    kappa: float | None = None,
    stride: int = FRAME_STRIDE,
    max_frames: int | None = None,
    wind_start_path: str = WIND_START,
    chunk_frames: int = 16,
    device: Any = "cpu",
    iters: int = 200,
    restart_iters: int = 100,
    max_passes: int = 20,
    progress: int = 0,
    line_gain: bool = True,
    mic_gain: bool = True,
    coherence: str = "none",
) -> dict[str, Any]:
    """The whole stage: pool, the per-rotor lines (frozen under
    ``coherence="none"``, else the torch line renderer), start, the renderer
    check, the fit and the flight payload (``line_gain=False`` holds ``g_r`` at
    0, ``mic_gain=False`` holds ``h_m`` at 0)."""
    if coherence not in COHERENCE_MODES:
        raise ValueError(f"coherence must be one of {COHERENCE_MODES}, got {coherence!r}")
    t0 = time.time()
    par = _bench_params(bench)
    band_hz = (float(bench["band_hz"][0]), float(bench["band_hz"][1]))
    pool = load_pool(stride=stride, max_frames=max_frames)
    if progress:
        print(
            f"pool {POOL}: {len(pool.names)} segments, {pool.n_frames} frames "
            f"(of {pool.n_frames_read}), band {band_hz} Hz  {time.time() - t0:.1f}s",
            flush=True,
        )
    renderer, c_start = None, {}
    if coherence != "none":
        renderer = LineRenderer(par, pool.rps, n_fft=pool.n_fft, sr=pool.sr, device=device)
        c_start = {k: coherence_start(bench)[k] for k in COHERENCE_FREE[coherence]}
        c_vals = start_coherence_values(bench, coherence, renderer._t)
        # the start's L = sum_r L_r at the start coherence (measure_start sums over axis 0)
        lines = renderer.all_frames(
            chunk=chunk_frames,
            sigma2_phase=c_vals["sigma2_phase"],
            gamma_phase_hz=c_vals["gamma_phase_hz"],
            eps=c_vals["eps"],
            scale=renderer._t(np.ones(par.n_rotors)),
        )[None]
        if progress:
            print(f"start lines (torch renderer)  {time.time() - t0:.1f}s", flush=True)
    else:
        lines = frozen_lines(bench, pool.rps, n_fft=pool.n_fft, sr=pool.sr, progress=progress)
    w0, w0_record = wind_start(wind_start_path)
    start = measure_start(bench, pool, lines, band_hz=band_hz, wind_db=w0)
    init = dict(start.init, **c_start)
    if kappa is None:
        kappa, kappa_source = float(bench["priors"]["floor_push_kappa"]), "bench payload priors"
    else:
        kappa_source = "--kappa"
    model = FlightStageModel(
        bench,
        pool,
        None if coherence != "none" else lines,
        band_hz=band_hz,
        kappa=float(kappa),
        floor_centre_db=start.floor_centre_db,
        wind_centre_db=start.wind_centre_db,
        init=init,
        line_gain=line_gain,
        mic_gain=mic_gain,
        coherence=coherence,
        renderer=renderer,
        chunk_frames=chunk_frames,
        device=device,
    )
    del lines
    check = renderer_check(model, sorted({0, pool.n_frames - 1}))
    if not check["passed"]:
        raise AssertionError(
            f"torch E differs from expected_periodogram by {check['max_rel_err']:.3g} "
            f"(relative) > {check['rtol']:g}"
        )
    if progress:
        print(f"renderer check: max rel err {check['max_rel_err']:.3g}", flush=True)
    record = fit_stage(
        model,
        iters=iters,
        restart_iters=restart_iters,
        max_passes=max_passes,
        progress=bool(progress),
    )
    floor, wind, gain, mic_gain_db = model.exported()
    exported_coh = model.exported_coherence()
    pay = flight_payload(
        bench,
        floor=floor,
        wind_db=wind,
        line_gain_db=gain,
        mic_gain_db=mic_gain_db,
        coherence=exported_coh,
    )
    pay["support"] = POOL
    pay["supports"] = list(pool.names)
    pay["flight_stage"] = dict(
        stage=STAGE,
        module=__name__,
        bench_source=str(bench_path),
        bench_git=bench.get("git"),
        frozen=dict(
            blocks=[
                "profile_aero_db",
                "motor_orders",
                "profile_motor_db",
                "alpha_aero",
                "alpha_motor",
                "hierarchy",
                *dict(
                    none=["pedestal", "shaft"],
                    shaft=["pedestal", "shaft.a_sigma", "shaft.lam_ref_per_s", "shaft.b_lam"],
                    full=[
                        "pedestal.sigma2_amp",
                        "pedestal.gamma_amp_hz",
                        "shaft.a_sigma",
                        "shaft.lam_ref_per_s",
                        "shaft.b_lam",
                    ],
                )[coherence],
                "mic_response",
                "floor.ctrl_hz",
                "floor.shape_sd_db",
                "floor.floor_rule",
            ],
            profiles="frozen up to the per-rotor line gain: profile_*_db[r] = bench + g_r"
            if line_gain
            else "frozen (line gains held at 0)",
            mic_response="pair EQ frozen up to the per-mic gain: pair_eq.db[:, m, :] = bench + h_m"
            if mic_gain
            else "pair EQ frozen (mic gains held at 0)",
            rule="L_r rendered per frame by LineRenderer (the torch port of the line part of "
            "v4.expected_periodogram) at the "
            + (
                "free phase pedestal and eps"
                if coherence == "full"
                else "bench phase pedestal and the free eps"
            )
            if coherence != "none"
            else "L_r = v4.expected_periodogram of the bench payload with the floor off "
            f"(mean_db {SILENT_DB:g}, static_rel 0) and the other rotors' profiles at "
            f"{SILENT_DB:g} dB, per frame; sum_r L_r = all-rotor L to {SPLIT_RTOL:g}",
        ),
        free=(["line_gain_db"] if line_gain else [])
        + (["mic_gain_db"] if mic_gain else [])
        + list(COHERENCE_FIELDS[coherence])
        + ["floor.mean_db", "floor.shape_z", "floor.static_rel", "floor.b0", "floor.b1", "wind"],
        coherence=coherence_record(model, bench),
        line_gain=dict(
            free=bool(line_gain),
            gain_db=[float(x) for x in gain],
            prior=dict(family="Normal", mu_db=LINE_GAIN_MU_DB, sd_db=LINE_GAIN_SD_DB),
            start_db=start.record["line_gain_db"],
            export="profile_aero_db[r] += g_r and profile_motor_db[r] += g_r",
        ),
        mic_gain=dict(
            free=bool(mic_gain),
            gain_db=[float(x) for x in mic_gain_db],
            prior=dict(family="Normal", mu_db=0.0, sd_db=MIC_GAIN_SD_DB),
            constraint="sum_m h_m = 0",
            start_db=start.record["mic_gain_db"],
            export="mic_response.pair_eq.db[:, m, :] += h_m (lines and per-rotor floor on mic m)",
        ),
        band_hz=list(band_hz),
        band_rule="the bench payload's band_hz: the bench Whittle's band (bench_v4.BAND_HZ)",
        floor_oversample=FLOOR_OVERSAMPLE,
        pool=pool.diagnostics(),
        max_frames=max_frames,
        start=dict(start.record, wind_start=w0_record),
        priors=priors_record(model, kappa_source=kappa_source),
        renderer_check=check,
        objective=record["objective"],
        optimiser=record["optimiser"],
        converged=record["optimiser"]["converged"],
        wall_s=time.time() - t0,
        git=FV._git_sha(),
    )
    return pay
