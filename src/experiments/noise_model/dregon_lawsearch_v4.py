"""Noise model v4, the DREGON SPEED-LAW SEARCH: the flight payload's line and
floor speed laws searched DIRECTLY on the official scorer's metrics.

WHY. The DREGON flight payload carries the BENCH speed laws (per-rotor line
profiles and exponents fitted on the single-rotor bench, the flight floor of
the flight stage). In flight the laws may differ; a Whittle refit optimises the
likelihood, not what the gates read. This search moves six numbers of the
payload and scores every candidate exactly as ``scripts/noise_v2_round_score.py``
scores a support (``measure_support``): the frozen HPPNet-L2 tracker's PIT MAE
over all eight mics on the raw-telemetry cruise support, and the absolute LTAS
error on mic 0.

THE SPACE (:data:`BOX`, in :data:`PARAMS` order), edits of the base payload
(:func:`edit_payload`); nothing else moves (pedestal, shaft, pair EQ, static
share, wind):

* ``g_a``: dB added to every ``params.profile_aero_db`` entry;
* ``alpha_a``: ``params.alpha_aero`` (line power ``10^{p/10} s^alpha``);
* ``g_m``: dB added to every ``params.profile_motor_db`` entry;
* ``alpha_m``: ``params.alpha_motor``;
* ``g_f``: dB added to ``params.floor.mean_db`` (and the ``ctrl_db`` record);
* ``b0``: ``params.floor.b0`` of ``b(f) = b0 + b1 log2(f / 2 kHz)``, ``b1`` the
  base's.

THE TUNING SET: the five 8 s ``dregon-floor`` fit segments
(:func:`.dregon_flight_v4.pool_specs`, read by :func:`.supports.flight_clip`),
never the 4 s scoring windows :data:`.gates.DREGON_CRUISE_SUPPORTS`; one render
seed (:data:`RENDER_SEED`, the scorer's first). The BARS are nv3r4's
(:data:`NV3R4`) mean PIT and mean LTAS over the segments under the same
protocol: a candidate is FEASIBLE when it beats both.

COMPOSITION (:func:`build_components`, :func:`compose`). A render of seed
``s`` is linear in its parts and every part's random draw is independent of
these six numbers, so a segment's parts are rendered ONCE:

* per rotor ``r`` and family (aerodynamic, motor) the lines alone,
  ``render_noise`` of the base payload with both exponents 0, the other
  family and the other rotors at :data:`OFF_DB` (exactly zero in the kernel's
  float32), the floor off and no wind: the same streams, so the same shaft
  paths, start phases and pedestal OUs. A candidate multiplies it by
  ``10^{g/20}`` and by ``s_r^{alpha/2}`` evaluated on the 1 kHz envelope
  knots and interpolated linearly between them, as the line kernel
  interpolates the envelope. EXACT up to (i) the product of two linear
  interpolations against the interpolation of the product, and (ii) the gain
  applied after, not before, the anti-alias filter (and the decimation of the
  orders rendered at the work rate); both are second order in the speed's
  rate of change;
* the wind: :func:`.render.render_v4_wind`, the renderer's own wind path;
* the floor: NOT a sum of per-rotor floors. The renderer draws ONE white noise
  per mic shaped by ``sqrt(F(f))`` and gives it the short-time gain
  ``sum_r 10^{E_rm(f)/10} s_r(t)^{b(f)} + static`` through a WOLA, so the rotor
  floors add in POWER under one square root and a per-rotor ``s^{Delta b0}``
  factor does not factor out. The shaped white noise (the floor stream of the
  seed, base ``F``) is cached; per candidate the WOLA gain is re-applied with
  the candidate's ``b(f)`` (:func:`floor_output`, the renderer's floor block
  on the cached draw, times ``10^{g_f/20}``) and passed through the renderer's
  output stage (``_v4_output``). Exact.

:func:`verification` compares the composition with ``render_noise`` of the
edited payload at the same seed (max and rms error re the direct render's rms,
and the PIT / LTAS differences).

THE SEARCH (the script ``scripts/noise_v4_dregon_lawsearch.py``): stage 1 a
scrambled Sobol sample of the box (:data:`SOBOL_SEEDS`), stage 2 a second
Sobol sample in the box :func:`stage2_box` shrinks around stage 1's best
feasible points; selection (:func:`select`): the feasible point of lowest mean
PIT; the Pareto front (mean PIT vs mean LTAS) and the main-effect spread of
each parameter over the stage-1 sample (:func:`main_effects`) are reported.
"""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.noise_model import FIT_SCHEMA_V4
from data_processing.noise_model.constants import SPEED_FLOOR_RPS
from data_processing.noise_model.render import (
    ENV_RATE_HZ,
    V4_FLOOR_WOLA_FRAME,
    _slow_gain,
    _v4_output,
    _v4_streams,
    fit_work_rate,
    render_noise,
    render_v4_wind,
)
from data_processing.noise_model.v4 import FLOOR_RULE_SUM, V4Params
from experiments.stochastic_fit import revised_eval as RE

from . import dregon_flight_v4 as DF
from . import gates as GT
from . import supports as SU

__all__ = [
    "BOX",
    "NV3R4",
    "PARAMS",
    "RENDER_SEED",
    "SOBOL_SEEDS",
    "STAGE",
    "Components",
    "base_point",
    "bars",
    "build_components",
    "check_base",
    "components",
    "compose",
    "cruise_support",
    "edit_payload",
    "floor_output",
    "load_components",
    "main_effects",
    "mark_feasible",
    "pareto",
    "point_table",
    "read_rows",
    "render_parts",
    "save_components",
    "score",
    "select",
    "sobol_points",
    "stage2_box",
    "verify_points",
]

STAGE = "dregon_lawsearch_v4"
#: The six searched numbers, in the order of every point vector.
PARAMS: tuple[str, ...] = ("g_a", "alpha_a", "g_m", "alpha_m", "g_f", "b0")
#: The stage-1 box. ``b0``'s lower edge keeps ``b(f) >= 0`` on the band (the v4
#: prior's truncation) with the DREGON stage payload's ``b1`` = 0.771:
#: ``b(30 Hz) = b0 - 0.771 * log2(2000 / 30) >= 0`` needs ``b0 >= 4.67``.
BOX: dict[str, tuple[float, float]] = dict(
    g_a=(-6.0, 12.0),
    alpha_a=(0.0, 14.0),
    g_m=(-12.0, 6.0),
    alpha_m=(-4.0, 4.0),
    g_f=(-12.0, 6.0),
    b0=(4.7, 12.0),
)
#: Scrambled Sobol seeds of stage 1 and stage 2.
SOBOL_SEEDS: tuple[int, int] = (20261023, 20261024)
#: The composition check: its seed, its random points (plus the base point).
VERIFY_SEED = 20261025
N_VERIFY = 3
#: The render seed of every arm: the scorer's first (``gates.RENDER_SEEDS``).
RENDER_SEED = int(GT.RENDER_SEEDS[0])
#: nv3r4's DREGON flight fit, the bar.
NV3R4 = "results/noise_v3/fits_r4/dregon_room2_floor__flight_v3.json"
#: The scored band of the DREGON supports.
CRUISE_BAND = GT.REGIME_BANDS["cruise"]
#: A line switched off: 10^{-100} power, zero once the kernel's float32 envelope.
OFF_DB = -1000.0
#: The floor switched off (the flight stage's convention): 10^{-1000} = 0.
FLOOR_OFF_DB = -10000.0
#: Stage 2: the box spans the stage-1 best ``STAGE2_TOP`` feasible points,
#: padded by ``STAGE2_PAD`` of the stage-1 range per side, clipped to the box.
STAGE2_TOP = 8
STAGE2_PAD = 0.1
#: Main effects: equal-width bins of each parameter over its stage-1 range.
MAIN_EFFECT_BINS = 8
FAMILIES = ("aero", "motor")


# ── the payload edits ───────────────────────────────────────────────────────


def check_base(base: Mapping[str, Any]) -> None:
    """The search is defined on a DREGON v4 flight payload with the per-rotor floor."""
    if base.get("schema") != FIT_SCHEMA_V4 or base.get("kind") != "flight":
        raise ValueError(
            f"the law search starts from a v4 FLIGHT payload, got schema {base.get('schema')!r} "
            f"kind {base.get('kind')!r}"
        )
    if base.get("rig") != "dregon":
        raise ValueError(f"the law search is DREGON's, got rig {base.get('rig')!r}")
    par = V4Params.from_payload(base)
    if par.floor.rule != FLOOR_RULE_SUM:
        raise ValueError(f"floor rule {par.floor.rule!r}: the composition is the per-rotor sum's")


def base_point(base: Mapping[str, Any]) -> dict[str, float]:
    """The base payload as a point of the space (gains 0)."""
    p = base["params"]
    return dict(
        g_a=0.0,
        alpha_a=float(p["alpha_aero"]),
        g_m=0.0,
        alpha_m=float(p["alpha_motor"]),
        g_f=0.0,
        b0=float(p["floor"]["b0"]),
    )


def edit_payload(base: Mapping[str, Any], point: Mapping[str, float]) -> dict[str, Any]:
    """The base payload with the point's six numbers written in."""
    out = copy.deepcopy(dict(base))
    p = out["params"]
    p["profile_aero_db"] = (np.asarray(p["profile_aero_db"]) + float(point["g_a"])).tolist()
    p["alpha_aero"] = float(point["alpha_a"])
    p["profile_motor_db"] = (np.asarray(p["profile_motor_db"]) + float(point["g_m"])).tolist()
    p["alpha_motor"] = float(point["alpha_m"])
    fl = p["floor"]
    fl["mean_db"] = float(fl["mean_db"]) + float(point["g_f"])
    if "ctrl_db" in fl:  # the absolute control-value record moves with the mean
        fl["ctrl_db"] = (np.asarray(fl["ctrl_db"]) + float(point["g_f"])).tolist()
    fl["b0"] = float(point["b0"])
    return out


def _lines_payload(base: Mapping[str, Any], rotor: int, family: str) -> dict[str, Any]:
    """Rotor ``rotor``'s ``family`` lines alone, exponents 0, no floor, no wind."""
    out = copy.deepcopy(dict(base))
    p = out["params"]
    keep, drop = (
        ("profile_aero_db", "profile_motor_db")
        if family == "aero"
        else ("profile_motor_db", "profile_aero_db")
    )
    kept = np.asarray(p[keep], dtype=np.float64)
    off = np.full_like(kept, OFF_DB)
    off[rotor] = kept[rotor]
    p[keep] = off.tolist()
    p[drop] = np.full_like(np.asarray(p[drop], dtype=np.float64), OFF_DB).tolist()
    p["alpha_aero"] = 0.0
    p["alpha_motor"] = 0.0
    p["floor"]["mean_db"] = FLOOR_OFF_DB
    p["floor"]["static_rel"] = 0.0
    p.pop("wind", None)
    return out


# ── the components and the composition ─────────────────────────────────────


@dataclass(frozen=True)
class Components:
    """One segment's cached render parts (seed :data:`RENDER_SEED`) and its data."""

    name: str
    recording: str
    start_s: float
    dur_s: float
    sr: int
    seed: int
    rps: np.ndarray  #: (R, T) the motors_command label on the audio grid
    real: np.ndarray  #: (M, T) the raw audio
    aero: np.ndarray  #: (R, M, T) the rotor's aerodynamic lines at alpha 0, post output
    motor: np.ndarray  #: (R, M, T) the rotor's motor lines at alpha 0, post output
    wind: np.ndarray  #: (M, T) the wind, post output
    shaped: np.ndarray  #: (M, T) the floor's shaped white noise at the base F, pre WOLA
    par: V4Params  #: the base payload
    log_s_knots: np.ndarray  #: (R, n_env) log relative speed on the envelope knots
    log_s: np.ndarray  #: (R, T) log relative speed on the output grid
    rotor_lin: np.ndarray  #: (R, M, Fw) 10^{E_rm/10} on the floor WOLA bins
    hop_env: int

    @property
    def n_samples(self) -> int:
        return int(self.real.shape[1])

    @property
    def n_mics(self) -> int:
        return int(self.real.shape[0])


def render_parts(
    base: Mapping[str, Any], rps: np.ndarray, *, sr: int, n_mics: int, seed: int
) -> dict[str, Any]:
    """The base payload's render of ``seed`` on ``rps`` in parts: ``aero`` and
    ``motor`` ``(R, M, T)`` (each rotor's lines alone at exponent 0, post
    output), ``wind`` ``(M, T)`` (post output), ``shaped`` ``(M, T)`` (the
    floor's shaped white noise before its WOLA gain), and ``wall_s`` per part."""
    par = V4Params.from_payload(base)
    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    n = int(rps.shape[1])
    walls: dict[str, float] = {}
    lines = {f: np.zeros((par.n_rotors, n_mics, n), dtype=np.float32) for f in FAMILIES}
    for family in FAMILIES:
        t0 = time.time()
        for r in range(par.n_rotors):
            x = render_noise(_lines_payload(base, r, family), rps, sr=sr, n_mics=n_mics, seed=seed)
            lines[family][r] = np.asarray(x, dtype=np.float32)
        walls[family] = time.time() - t0
    t0 = time.time()
    wind = render_v4_wind(base, n, sr=sr, n_mics=n_mics, seed=seed)
    walls["wind"] = time.time() - t0
    t0 = time.time()
    # the renderer's floor draw: the floor stream of the seed, one white noise per mic
    rng_floor = _v4_streams(seed)[4]
    floor_amp = np.sqrt(par.floor.psd(np.fft.rfftfreq(n, d=1.0 / float(sr))))
    shaped = np.empty((n_mics, n))
    for m in range(n_mics):
        white = rng_floor.standard_normal(n)
        shaped[m] = np.fft.irfft(np.fft.rfft(white) * floor_amp, n=n)
    walls["floor"] = time.time() - t0
    return dict(
        aero=lines["aero"],
        motor=lines["motor"],
        wind=np.asarray(wind, dtype=np.float32),
        shaped=shaped,
        wall_s=walls,
    )


def build_components(
    base: Mapping[str, Any], segment: int, *, seed: int = RENDER_SEED
) -> dict[str, Any]:
    """The arrays :func:`save_components` writes for pool segment ``segment``:
    its data, its identity and :func:`render_parts` on its label."""
    spec = DF.pool_specs()[int(segment)]
    clip = SU.flight_clip(spec)  # the loader of the flight stage's supports
    audio = np.asarray(clip.audio, dtype=np.float32)
    rps = np.asarray(clip.rps, dtype=np.float64)
    if int(audio.shape[0]) != V4Params.from_payload(base).n_mics:
        raise ValueError(f"segment has {audio.shape[0]} mics, the payload does not")
    parts = render_parts(base, rps, sr=int(clip.sr), n_mics=int(audio.shape[0]), seed=seed)
    return dict(
        parts,
        name=np.array(spec.name),
        recording=np.array(str(spec.args["recording"])),
        start_s=np.array(float(spec.args["start_s"])),
        dur_s=np.array(float(spec.args["dur_s"])),
        sr=np.array(int(clip.sr)),
        seed=np.array(int(seed)),
        rps=rps,
        real=audio,
    )


def save_components(path: Path, arrays: Mapping[str, Any]) -> None:
    """Atomic ``.npz`` of :func:`build_components`' arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.stem}.tmp.npz")
    np.savez(tmp, **{k: v for k, v in arrays.items() if k != "wall_s"})
    tmp.replace(path)


def load_components(path: Path, base: Mapping[str, Any]) -> Components:
    """:func:`components` of a :func:`save_components` file."""
    with np.load(path) as z:
        return components({k: z[k] for k in z.files}, base)


def components(a: Mapping[str, Any], base: Mapping[str, Any]) -> Components:
    """A segment's parts plus the speed arrays the composition reads, derived
    from the label exactly as ``_render_noise_v4`` derives them."""
    par = V4Params.from_payload(base)
    sr = int(a["sr"])
    rps = np.asarray(a["rps"], dtype=np.float64)
    n = int(rps.shape[1])
    sr_work = fit_work_rate(base)
    oversample = sr_work // sr
    hop_env = sr // ENV_RATE_HZ
    n_work = n * oversample
    n_env = n // hop_env + 2
    knots = np.minimum(np.arange(n_env) * hop_env * oversample, n_work - 1)
    t_src = np.arange(n) / float(sr)
    t_knots = knots / float(sr_work)
    f_knots = np.maximum(np.stack([np.interp(t_knots, t_src, r) for r in rps]), SPEED_FLOOR_RPS)
    s_knots = np.maximum(par.speed(f_knots), SPEED_FLOOR_RPS / par.f_ref_rev_s)
    f_wola = np.fft.rfftfreq(V4_FLOOR_WOLA_FRAME, d=1.0 / float(sr))
    n_mics = int(a["real"].shape[0])
    return Components(
        name=str(a["name"]),
        recording=str(a["recording"]),
        start_s=float(a["start_s"]),
        dur_s=float(a["dur_s"]),
        sr=sr,
        seed=int(a["seed"]),
        rps=rps,
        real=a["real"],
        aero=a["aero"],
        motor=a["motor"],
        wind=a["wind"],
        shaped=a["shaped"],
        par=par,
        log_s_knots=np.log(s_knots),
        log_s=np.log(par.speed(np.maximum(rps, SPEED_FLOOR_RPS))),
        rotor_lin=10.0 ** (par.mic.rotor_db(f_wola)[:, :n_mics] / 10.0),
        hop_env=hop_env,
    )


def floor_output(comp: Components, *, b0: float, g_f: float) -> np.ndarray:
    """``(M, T)`` the floor of exponent offset ``b0`` and level ``+g_f`` dB as
    the renderer makes it from the cached draw: the WOLA gain
    ``sum_r 10^{E_rm/10} s_r(t)^{b(f)} + static`` (per-rotor sum rule), then
    the output stage."""
    sr, n = comp.sr, comp.n_samples
    floor = dataclasses.replace(comp.par.floor, b0=float(b0))
    b_wola = floor.exponent(np.fft.rfftfreq(V4_FLOOR_WOLA_FRAME, d=1.0 / float(sr)))
    t_out = np.arange(n) / float(sr)
    law: dict[int, np.ndarray] = {}

    def rotor_floor_gain(t_s: np.ndarray) -> np.ndarray:
        if t_s.size not in law:
            ls = np.stack([np.interp(t_s, t_out, row) for row in comp.log_s])
            law[t_s.size] = np.exp(ls[:, :, None] * b_wola[None, None, :])
        return law[t_s.size]

    static = floor.static_rel
    level = 10.0 ** (float(g_f) / 20.0)
    out = np.empty((comp.n_mics, n))
    for m in range(comp.n_mics):

        def gain_db(t_s: np.ndarray, m: int = m) -> np.ndarray:
            g = np.einsum("rf,rtf->tf", comp.rotor_lin[:, m], rotor_floor_gain(t_s)) + static
            return 10.0 * np.log10(g)

        out[m] = level * _slow_gain(comp.shaped[m], gain_db, sr_work=sr, frame=V4_FLOOR_WOLA_FRAME)
    return _v4_output(comp.par, out, sr)


def _envelope_gain(comp: Components, r: int, alpha: float) -> np.ndarray:
    """``(T,)`` ``s_r^{alpha/2}`` on the envelope knots, linear between them."""
    g = np.exp(0.5 * float(alpha) * comp.log_s_knots[r])
    pos = np.arange(g.size) * comp.hop_env
    return np.interp(np.arange(comp.n_samples), pos, g)


def compose(comp: Components, point: Mapping[str, float]) -> np.ndarray:
    """``(M, T)`` the render of the edited payload, from the segment's parts."""
    out = comp.wind.astype(np.float64)
    ga = 10.0 ** (float(point["g_a"]) / 20.0)
    gm = 10.0 ** (float(point["g_m"]) / 20.0)
    for r in range(comp.aero.shape[0]):
        out += (ga * _envelope_gain(comp, r, float(point["alpha_a"])))[None, :] * comp.aero[r]
        out += (gm * _envelope_gain(comp, r, float(point["alpha_m"])))[None, :] * comp.motor[r]
    out += floor_output(comp, b0=float(point["b0"]), g_f=float(point["g_f"]))
    return out


# ── scoring ─────────────────────────────────────────────────────────────────


def cruise_support(comp: Components) -> Any:
    """The scorer's raw-telemetry cruise support of the segment."""
    window = RE.Window(comp.recording, comp.start_s, comp.dur_s, "cruise")
    return RE.regime_support(
        window,
        comp.rps,
        regime="cruise",
        min_rps=CRUISE_BAND[0],
        max_rps=CRUISE_BAND[1],
        sr=comp.sr,
    )


def score(tracker: Any, comp: Components, audio: np.ndarray) -> dict[str, Any]:
    """``noise_v2_round_score.measure_support``'s two numbers on one segment:
    the tracker's PIT MAE (all mics, cruise support) and the mic-0 LTAS error."""
    audio = np.asarray(audio, dtype=np.float64)
    mics = list(range(comp.n_mics))
    t0 = time.time()
    pit = tracker.pit(
        audio,
        comp.rps,
        mics=mics,
        expected_samples=comp.n_samples,
        support=cruise_support(comp),
    )
    t1 = time.time()
    ltas = RE.ltas_deviation_db(comp.real.astype(np.float64), audio, mic=0)
    return dict(
        pit_mae=float(pit["mae"]),
        pit_per_mic=[float(v) for v in pit["per_mic"]],
        n_scored_frames=pit.get("n_scored_frames"),
        ltas_abs_db=float(ltas["mean_abs_db"]),
        ltas_level_offset_db=float(ltas["level_offset_db"]),
        pit_s=t1 - t0,
        ltas_s=time.time() - t1,
    )


# ── the samplers ────────────────────────────────────────────────────────────


def _scale(u: np.ndarray, box: Mapping[str, Sequence[float]]) -> list[dict[str, float]]:
    lo = np.array([float(box[k][0]) for k in PARAMS])
    hi = np.array([float(box[k][1]) for k in PARAMS])
    x = lo + np.asarray(u) * (hi - lo)
    return [dict(zip(PARAMS, (float(v) for v in row), strict=True)) for row in x]


def sobol_points(n: int, box: Mapping[str, Sequence[float]], seed: int) -> list[dict[str, float]]:
    """The first ``n`` points of a scrambled Sobol sequence (``2^ceil(log2 n)``
    drawn, so a prefix of a balanced sample) mapped into ``box``."""
    from scipy.stats import qmc

    if n <= 0:
        return []
    m = max(0, math.ceil(math.log2(n)))
    u = qmc.Sobol(d=len(PARAMS), scramble=True, seed=int(seed)).random_base2(m)[:n]
    return _scale(u, box)


def verify_points(base: Mapping[str, Any]) -> list[dict[str, float]]:
    """The composition check's points: the base, then :data:`N_VERIFY` uniform draws."""
    u = np.random.default_rng(VERIFY_SEED).random((N_VERIFY, len(PARAMS)))
    return [base_point(base), *_scale(u, BOX)]


# ── the table, the bars, the selection ──────────────────────────────────────


def bars(ref_rows: Sequence[Mapping[str, Any]], n_segments: int) -> dict[str, Any]:
    """Per arm (real, nv3r4, base) the per-segment numbers and their means."""
    out: dict[str, Any] = {}
    for arm in ("real", "nv3r4", "base"):
        rows = sorted((r for r in ref_rows if r.get("arm") == arm), key=lambda r: r["segment"])
        if len(rows) != n_segments:
            raise ValueError(f"reference arm {arm}: {len(rows)} of {n_segments} segments")
        out[arm] = dict(
            segments=[r["name"] for r in rows],
            pit_mae=[r["pit_mae"] for r in rows],
            ltas_abs_db=[r["ltas_abs_db"] for r in rows],
            mean_pit_mae=float(np.mean([r["pit_mae"] for r in rows])),
            mean_ltas_abs_db=float(np.mean([r["ltas_abs_db"] for r in rows])),
        )
    return out


def point_table(rows: Sequence[Mapping[str, Any]], n_segments: int) -> list[dict[str, Any]]:
    """One entry per (stage, point): the point, its per-segment numbers and,
    when every segment is in, the means."""
    groups: dict[tuple[int, int], list[Mapping[str, Any]]] = {}
    for r in rows:
        groups.setdefault((int(r["stage"]), int(r["idx"])), []).append(r)
    table = []
    for (stage, idx), rs in sorted(groups.items()):
        rs = sorted(rs, key=lambda r: r["segment"])
        complete = len(rs) == n_segments
        table.append(
            dict(
                stage=stage,
                idx=idx,
                point=dict(rs[0]["point"]),
                segments=[int(r["segment"]) for r in rs],
                pit_mae=[float(r["pit_mae"]) for r in rs],
                ltas_abs_db=[float(r["ltas_abs_db"]) for r in rs],
                complete=complete,
                mean_pit_mae=float(np.mean([r["pit_mae"] for r in rs])) if complete else None,
                mean_ltas_abs_db=(
                    float(np.mean([r["ltas_abs_db"] for r in rs])) if complete else None
                ),
            )
        )
    return table


def _feasible(entry: Mapping[str, Any], bar: Mapping[str, Any]) -> bool:
    return bool(
        entry["complete"]
        and entry["mean_pit_mae"] < bar["mean_pit_mae"]
        and entry["mean_ltas_abs_db"] < bar["mean_ltas_abs_db"]
    )


def mark_feasible(table: list[dict[str, Any]], bar: Mapping[str, Any]) -> None:
    for e in table:
        e["feasible"] = _feasible(e, bar)


def stage2_box(stage1: Sequence[Mapping[str, Any]], bar: Mapping[str, Any]) -> dict[str, Any]:
    """The stage-2 box: per parameter the span of the stage-1 best
    :data:`STAGE2_TOP` FEASIBLE points by mean PIT, padded by
    :data:`STAGE2_PAD` of the stage-1 range per side and clipped to it. With
    no feasible point, the best points by mean PIT alone (``fallback``)."""
    done = [e for e in stage1 if e["complete"]]
    if not done:
        raise ValueError("stage 1 has no complete point")
    feas = [e for e in done if _feasible(e, bar)]
    pool = feas or done
    top = sorted(pool, key=lambda e: e["mean_pit_mae"])[:STAGE2_TOP]
    box: dict[str, list[float]] = {}
    for k in PARAMS:
        lo, hi = BOX[k]
        pad = STAGE2_PAD * (hi - lo)
        vals = [float(e["point"][k]) for e in top]
        box[k] = [max(lo, min(vals) - pad), min(hi, max(vals) + pad)]
    return dict(
        box=box,
        rule=(
            f"per parameter [min - {STAGE2_PAD:g} R, max + {STAGE2_PAD:g} R] of the stage-1 best "
            f"{STAGE2_TOP} feasible points by mean PIT (feasible: beats nv3r4's tuning-set mean "
            "PIT and mean LTAS), R the stage-1 range, clipped to the stage-1 box; with no "
            "feasible point the best by mean PIT alone"
        ),
        fallback=not feas,
        n_feasible_stage1=len(feas),
        n_stage1=len(done),
        top=[dict(stage=e["stage"], idx=e["idx"]) for e in top],
    )


def pareto(table: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """The complete points no other complete point beats on both means."""
    done = sorted(
        (e for e in table if e["complete"]),
        key=lambda e: (e["mean_pit_mae"], e["mean_ltas_abs_db"]),
    )
    front, best_ltas = [], math.inf
    for e in done:
        if e["mean_ltas_abs_db"] < best_ltas:
            front.append(
                dict(
                    stage=e["stage"],
                    idx=e["idx"],
                    point=e["point"],
                    mean_pit_mae=e["mean_pit_mae"],
                    mean_ltas_abs_db=e["mean_ltas_abs_db"],
                    feasible=e.get("feasible"),
                )
            )
            best_ltas = e["mean_ltas_abs_db"]
    return front


def main_effects(
    table: Sequence[Mapping[str, Any]], box: Mapping[str, Sequence[float]] = BOX
) -> dict[str, Any]:
    """Per parameter over the (uniform) stage-1 sample: the spread (max - min)
    of the bin means of mean PIT and mean LTAS over :data:`MAIN_EFFECT_BINS`
    equal-width bins of the parameter, the bin means, and Spearman's rho."""
    from scipy.stats import spearmanr

    done = [e for e in table if e["complete"] and e["stage"] == 1]
    if len(done) < 2 * MAIN_EFFECT_BINS:
        return dict(n_points=len(done), note=f"fewer than {2 * MAIN_EFFECT_BINS} stage-1 points")
    pit = np.array([e["mean_pit_mae"] for e in done])
    ltas = np.array([e["mean_ltas_abs_db"] for e in done])
    out: dict[str, Any] = dict(n_points=len(done), bins=MAIN_EFFECT_BINS)
    for k in PARAMS:
        x = np.array([e["point"][k] for e in done])
        lo, hi = float(box[k][0]), float(box[k][1])
        b = np.clip(((x - lo) / (hi - lo) * MAIN_EFFECT_BINS).astype(int), 0, MAIN_EFFECT_BINS - 1)
        mp = [
            float(pit[b == j].mean()) if np.any(b == j) else None for j in range(MAIN_EFFECT_BINS)
        ]
        ml = [
            float(ltas[b == j].mean()) if np.any(b == j) else None for j in range(MAIN_EFFECT_BINS)
        ]
        vp = [v for v in mp if v is not None]
        vl = [v for v in ml if v is not None]
        out[k] = dict(
            pit_spread=float(max(vp) - min(vp)),
            ltas_spread_db=float(max(vl) - min(vl)),
            pit_bin_means=mp,
            ltas_bin_means=ml,
            spearman_pit=float(spearmanr(x, pit).statistic),
            spearman_ltas=float(spearmanr(x, ltas).statistic),
        )
    return out


def select(table: Sequence[Mapping[str, Any]]) -> dict[str, Any] | None:
    """The feasible point of lowest mean PIT, or ``None``."""
    feas = [e for e in table if e.get("feasible")]
    return dict(min(feas, key=lambda e: e["mean_pit_mae"])) if feas else None


def read_rows(raw_dir: Path) -> list[dict[str, Any]]:
    """Every unit JSON of a grid's ``raw/``."""
    return [json.loads(p.read_text()) for p in sorted(Path(raw_dir).glob("*.json"))]
