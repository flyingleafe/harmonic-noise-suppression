"""Noise model v4, the DREGON FLIGHT STAGE: the wind and the floor from free
flight, on top of a frozen bench rig.

WHY. A v4 BENCH payload (``kind="bench"``, :mod:`.bench_v4`) rendered on
DREGON free flight is 6-23 dB too quiet below 2 kHz, for two reasons: it has
no flight wind (the bench wind is per RECORDING, not a rig term), and its
broadband floor is the bench floor while the flight floor is higher. The
lines, their pedestal, the shaft law and the pair EQ are properties of the
rotors and the array, and the bench is where they are seen best (one rotor at
a time, 0.03-0.1 Hz bins); the flight wind and the flight floor exist only in
flight. So this stage refits ONLY the floor and the wind, from flight
recordings only, and holds every bench line / pedestal / shaft / pair-EQ
parameter fixed. The v4 model is unchanged: the stage is a procedure, and its
payload is the bench payload with ``kind="flight"``, the floor refitted and a
static per-mic wind block, which :func:`data_processing.noise_model.v4.expected_periodogram`
and the renderer read as they stand.

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

    E_mn = L_mn + T[ sum_r h_rm F s_rn^b + s F + 10^{w_m/10} W ]

``L`` (:func:`frozen_lines`) is the bench payload's own expected periodogram
at the frame's speeds with the floor switched off: lines, pedestal, shaft and
pair EQ, computed once per frame by ``expected_periodogram``. ``T`` is that
function's window transform of a PSD on its ``floor_oversample``-times finer
grid (linear in the PSD), done in torch, so the floor and wind terms are
exactly what ``expected_periodogram`` adds for them (:func:`renderer_check`
holds the whole ``E`` to 1e-6 of the renderer's at random parameters). ``h``
is the frozen pair EQ, ``F`` the 14-point floor spline on the bench's
``ctrl_hz`` and ``shape_sd_db``, ``b(f) = b0 + b1 log2(f / 2 kHz)``, ``s`` the
static share, ``W`` the nv3 wind shape.

FREE: ``mean_db``, ``shape_z`` (14), ``static_rel``, ``b0``/``b1`` (in the
``b >= 0`` band-edge coordinates of :func:`.v4_priors.floor_exponents`) and
``wind_db`` (8). PRIORS: the bench's floor terms
(:meth:`.bench_v4.RigModel.neg_log_prior`): ``mean_db ~ N(measured,
FLOOR_MEAN_SD_DB)``, the centre measured on THIS pool as the bench measured
its own, ``z ~ N(0, I)``, ``b0 ~ N(B0)``, ``b1 ~ N(B1)``; the flight static
share ``static_rel ~ LogNormal(flight_v4.STATIC_SHARE_LN)``; nv3r4's wind
prior ``wind_db_m ~ N(measured low-band excess, PRIOR_SETS["v4"].wind_db_sd)``
(:func:`.fit._wind_centres`, here on the unnormalised pool); and the floor
push ``-kappa sum_j ctrl_j`` with the bench payload's ``floor_push_kappa``.

OPTIMISER. Start: the floor measured from the pool (low quantile of
``(I - L - W0) / (sum_r h s^b)`` per bin, projected on the 14 controls), the
bench's exponents, the static share at its prior median and the nv3r4 DREGON
wind (a START only). Then double-precision strong-Wolfe L-BFGS
(:class:`.lbfgs.LBFGS`) in fresh-memory passes until one gains under
``1e-4`` nats per cell AND moves the floor controls under 0.1 dB, the
exponents under 0.02 and the wind under 0.1 dB.
"""

from __future__ import annotations

import copy
import inspect
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import Tensor

from data_processing.noise_model import v4 as V4
from data_processing.noise_model.constants import SPEED_FLOOR_RPS
from data_processing.noise_model.floor import floor_geometry, floor_power_spectrum
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
    "Start",
    "fit_stage",
    "flight_payload",
    "frozen_lines",
    "load_pool",
    "measure_start",
    "renderer_check",
    "run_stage",
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
#: ``mean_db`` of the silenced floor: ``10^{-1000}`` underflows to exactly 0.
SILENT_DB = -1.0e4
#: nv3r4's DREGON prior set (``--priors v4``): the wind prior's sd.
WIND_PRIORS = MD.PRIOR_SETS["v4"]
TOL_NATS_PER_CELL = FV.TOL_NATS_PER_CELL
#: Convergence movement: floor controls (dB), exponents (b0, b1), wind levels
#: (dB, held to the line-level tolerance).
MOVE_TOL = dict(
    floor_ctrl_db=FV.MOVE_TOL["floor_ctrl_db"],
    exponents=FV.MOVE_TOL["exponents"],
    wind_db=FV.MOVE_TOL["levels_db"],
)
#: :func:`renderer_check`'s tolerance (max relative error of ``E``).
CHECK_RTOL = 1e-6
PARAM_KEYS = ("floor_mean_db", "floor_z", "u_blo", "u_bhi", "log_static", "wind_db")


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


def frozen_lines(
    bench: dict[str, Any], rps: np.ndarray, *, n_fft: int, sr: int, progress: int = 0
) -> np.ndarray:
    """``(M, N, n_fft // 2 + 1)``: ``expected_periodogram`` of the bench payload
    at each frame's speeds ``rps (R, N)`` with the floor off: lines, pedestal,
    shaft and pair EQ only. Asserts that nothing of the floor, its static share
    or a wind can leak in."""
    silent = silent_floor(bench)
    par = _bench_params(silent)
    f_psd = np.fft.rfftfreq(FLOOR_OVERSAMPLE * int(n_fft), d=1.0 / float(sr))
    if (
        np.any(par.floor.psd(f_psd) != 0.0)
        or par.floor.static_rel != 0.0
        or par.wind_db is not None
    ):
        raise AssertionError("the silenced bench floor still adds power")
    rps = np.asarray(rps, dtype=np.float64)
    out = np.empty((par.n_mics, rps.shape[1], int(n_fft) // 2 + 1))
    t0 = time.time()
    for i in range(rps.shape[1]):
        out[:, i] = V4.expected_periodogram(silent, rps[:, i], n_fft=int(n_fft), sr=int(sr))
        if progress and (i + 1) % int(progress) == 0:
            print(f"  frozen lines {i + 1}/{rps.shape[1]}  {time.time() - t0:.1f}s", flush=True)
    return out


# ── the model ───────────────────────────────────────────────────────────────


class FlightStageModel:
    """The flight stage's forward model and MAP objective: ``E = L + T[floor +
    static + wind]`` with ``L`` frozen and ``T`` the renderer's window transform."""

    def __init__(
        self,
        bench: dict[str, Any],
        pool: FlightPool,
        lines: np.ndarray,
        *,
        band_hz: tuple[float, float],
        kappa: float,
        floor_centre_db: float,
        wind_centre_db: np.ndarray,
        init: dict[str, Any],
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
        self.lines = self._t(lines)  # (M, N, n // 2 + 1)
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

    def _t(self, x: Any) -> Tensor:
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
        return dict(
            p,
            b0=b0,
            b1=b1,
            static_rel=torch.exp(p["log_static"]),
            ctrl_rel=ctrl_rel,
            ctrl_db=p["floor_mean_db"] + ctrl_rel,
        )

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
        out = self.lines[:, frames] + floor + fixed[:, None, :]
        return out if all_bins else out.index_select(-1, self.band_t)

    def log_prior(self) -> tuple[Tensor, dict[str, float]]:
        """``log p`` of the free values plus the floor push ``kappa sum_j ctrl_j``."""
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
        return dict(
            floor_ctrl_db=v["ctrl_db"].cpu().numpy().copy(),
            exponents=np.array([float(v["b0"]), float(v["b1"])]),
            wind_db=v["wind_db"].cpu().numpy().copy(),
            log_static=np.array([float(v["log_static"])]),
        )

    @torch.no_grad()
    def exported(self) -> tuple[dict[str, Any], list[float]]:
        """``(floor fields, wind_db)`` of the payload at the current values."""
        v = self.values()
        floor = dict(
            mean_db=float(v["floor_mean_db"]),
            shape_z=v["floor_z"].cpu().numpy().tolist(),
            ctrl_db=v["ctrl_db"].cpu().numpy().tolist(),
            b0=float(v["b0"]),
            b1=float(v["b1"]),
            static_rel=float(v["static_rel"]),
        )
        return floor, v["wind_db"].cpu().numpy().tolist()


def flight_payload(
    bench: dict[str, Any], *, floor: dict[str, Any], wind_db: list[float]
) -> dict[str, Any]:
    """The bench payload with ``kind="flight"``, the floor fields ``floor`` and
    the static per-mic wind ``wind_db``; ``ctrl_hz``, ``shape_sd_db`` and the
    per-rotor-sum floor rule are the bench's."""
    pay = copy.deepcopy(bench)
    pay["kind"] = "flight"
    pay["params"]["floor"].update(floor, floor_rule=V4.FLOOR_RULE_SUM)
    pay["params"]["wind"] = dict(
        kind="static_per_mic", wind_db=[float(x) for x in wind_db], shape=wind_shape_spec()
    )
    return pay


def renderer_check(
    model: FlightStageModel, frames: list[int], *, seed: int = 0, rtol: float = CHECK_RTOL
) -> dict[str, Any]:
    """The torch ``E`` against ``expected_periodogram`` of the exported payload,
    on every bin of the pool frames ``frames``, at RANDOM parameter values
    (restored afterwards)."""
    rng = np.random.default_rng(int(seed))
    p = model.params
    saved = {k: t.detach().clone() for k, t in p.items()}
    try:
        with torch.no_grad():
            p["floor_mean_db"].add_(float(rng.uniform(-6.0, 6.0)))
            p["floor_z"].copy_(model._t(rng.standard_normal(p["floor_z"].numel())))
            p["u_blo"].fill_(float(rng.uniform(-1.0, 8.0)))
            p["u_bhi"].fill_(float(rng.uniform(-1.0, 4.0)))
            p["log_static"].fill_(math.log(float(rng.uniform(1e-4, 0.1))))
            p["wind_db"].add_(model._t(rng.normal(0.0, 6.0, p["wind_db"].numel())))
            idx = torch.as_tensor(np.asarray(frames, dtype=np.int64), device=model.dd["device"])
            mine = model.expected(idx, all_bins=True).cpu().numpy()
        floor, wind = model.exported()
        pay = flight_payload(model.bench, floor=floor, wind_db=wind)
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
    rel = np.abs(mine - ref) / np.abs(ref)
    return dict(
        frames=[int(i) for i in frames],
        seed=int(seed),
        n_fft=model.n,
        max_rel_err=float(rel.max()),
        band_max_rel_err=float(rel[..., model.band].max()),
        rtol=float(rtol),
        passed=bool(rel.max() < rtol),
        at=dict(floor=floor, wind_db=wind),
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
    """The floor measured from the pool at the bench exponents: per band bin,
    the nv3r4 floor quantile (+ its exponential-percentile correction) over
    mics and frames of ``max(I - L - W0, 0.1 I) / sum_r h_rm s_rn^b``, least
    squares on the 14 controls; the wind prior centres by nv3r4's rule."""
    par = _bench_params(bench)
    f_all = np.asarray(pool.freqs_hz, dtype=np.float64)
    band = (f_all >= band_hz[0]) & (f_all <= band_hz[1])
    f = f_all[band]
    power = pool.power[:, :, band]
    wind = 10.0 ** (np.asarray(wind_db)[:, None] / 10.0) * wind_shape(f)[None, :]
    resid = np.maximum(power - lines[:, :, band] - wind[:, None, :], 0.1 * power)
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
            static_rel=math.exp(FV.STATIC_SHARE_LN[0]),
            wind_centre_db=np.asarray(centres).tolist(),
        ),
    )


# ── the optimiser ───────────────────────────────────────────────────────────


def movement(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> dict[str, float]:
    """Largest absolute change per snapshot group."""
    return {k: float(np.max(np.abs(b[k] - a[k]))) for k in a}


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
    move_ok = {k: bool(moved[k] < tol) for k, tol in MOVE_TOL.items()}
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
    gain per cell under :data:`TOL_NATS_PER_CELL` and every :data:`MOVE_TOL`
    group's movement under its tolerance; every pass recorded."""
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
                move_tol=dict(MOVE_TOL),
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
    return dict(
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
) -> dict[str, Any]:
    """The whole stage: pool, frozen lines, start, the renderer check, the fit
    and the flight payload."""
    t0 = time.time()
    _bench_params(bench)
    band_hz = (float(bench["band_hz"][0]), float(bench["band_hz"][1]))
    pool = load_pool(stride=stride, max_frames=max_frames)
    if progress:
        print(
            f"pool {POOL}: {len(pool.names)} segments, {pool.n_frames} frames "
            f"(of {pool.n_frames_read}), band {band_hz} Hz  {time.time() - t0:.1f}s",
            flush=True,
        )
    lines = frozen_lines(bench, pool.rps, n_fft=pool.n_fft, sr=pool.sr, progress=progress)
    w0, w0_record = wind_start(wind_start_path)
    start = measure_start(bench, pool, lines, band_hz=band_hz, wind_db=w0)
    if kappa is None:
        kappa, kappa_source = float(bench["priors"]["floor_push_kappa"]), "bench payload priors"
    else:
        kappa_source = "--kappa"
    model = FlightStageModel(
        bench,
        pool,
        lines,
        band_hz=band_hz,
        kappa=float(kappa),
        floor_centre_db=start.floor_centre_db,
        wind_centre_db=start.wind_centre_db,
        init=start.init,
        chunk_frames=chunk_frames,
        device=device,
    )
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
    floor, wind = model.exported()
    pay = flight_payload(bench, floor=floor, wind_db=wind)
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
                "pedestal",
                "shaft",
                "mic_response",
                "floor.ctrl_hz",
                "floor.shape_sd_db",
                "floor.floor_rule",
            ],
            rule="L = v4.expected_periodogram of the bench payload with the floor off "
            f"(mean_db {SILENT_DB:g}, static_rel 0), per frame",
        ),
        free=["floor.mean_db", "floor.shape_z", "floor.static_rel", "floor.b0", "floor.b1", "wind"],
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
