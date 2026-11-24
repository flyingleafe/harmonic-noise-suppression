#!/usr/bin/env python3
"""Vold–Kalman residual validation of the November DAT → WAV clock maps.

Referee (label-free): the VK reconstruction residual ratio ``||x - x_hat|| / ||x||``
of the band-limited WAV excerpt ``x``, where ``x_hat`` is the coupled VK-2 envelope
model (:func:`tracking.vk_tracking.vk_envelopes`, banded Cholesky per coupling
group, then :func:`~tracking.vk_tracking.vk_reconstruct`) built ALONG the DAT
rotor speeds mapped onto the WAV by a clock hypothesis ``audio_s = A + tick / f``.
Only the clock hypothesis (and a fixed RPM-scale nuisance) changes; the audio,
the scored samples, the analysis band and the VK model are fixed per window.

Order models (shaft orders, carrier = ``base_order`` x shaft rate, VK harmonics
``k_min..k_max`` of the carrier, k-scaled band ``order * bw_shaft`` Hz):

  FLY013 (no propellers): orders 84 and 98 = harmonics 6, 7 of 14 x shaft rate,
         analysed at 48 kHz (98 x 135 rev/s = 13.2 kHz).
  FLY018 / FLY022 (two-blade propellers): blade-pass orders 2, 4, ..., 14 =
         harmonics 1..7 of 2 x shaft rate, analysed at 8 kHz.

Per window (frozen fit / held-out WAV centres of ``november_motor_clock.py``):

  coarse  local offset ``o`` (WAV minus ``tick / 84 MHz``) scanned +-1 s at 20 ms
          around the transition-based coarse seed (independent of the comb result);
  fine    +-50 ms at 2 ms around the coarse minimum, extended while the minimum
          sits on the scan edge (<= 3 times).

Both stages use the same capture radius, 1 rev/s (track band = order x 1 Hz), with
an amplitude prior (``RIDGE``). Chosen on ONE fit window per flight before any
held-out window was scored: 0.25 and 0.5 rev/s gave a non-convex curve with a
MAXIMUM near alignment on FLY022 (a band narrower than the 60 ms-sampled ESC speed
error is a misspecified line model); 1 rev/s gave a single smooth minimum on both
FLY013 and FLY022.

The nuisance RPM gain is fixed before any lag is fitted: ``--gain comb`` re-derives
the comb's own fit-window gain (``november_motor_clock.scan``), ``--gain 1.002``
fixes it, ``--gain profile`` profiles the pooled fine residual over the comb gain
grid on usable FIT windows only. The affine clock map is fitted on usable fit
windows; held-out windows only test it. Telemetry gaps (> 0.2 s between packets)
carry no rotor comb (speed set to zero, never interpolated) and every sample that
is not covered by real packets under EVERY scanned offset is unscored.

Decision (frozen before any run; see :func:`decide`): the published comb map is kept
unless a well-conditioned VK fit beats it on BOTH the pooled held-out residual and
the held-out local-lag RMS; maps that differ by less than the held-out lag scatter
validate the comb. The selected map is the best supported one, not exact truth.

Run (3-6 CPUs; the cluster command is the same with ``--jobs``)::

    export PYTHONPATH="$PWD/src" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
    python scripts/michaels_calib/november_vk_clock.py --gain comb --jobs 4 \\
        --out results/new_motor_data_alignment/vk_clock
    python scripts/michaels_calib/november_vk_clock.py --recording FLY022 --smoke --jobs 2

Outputs: ``<out>/<FLY>.json`` per recording (``<FLY>_smoke.json`` for ``--smoke``),
``<out>/summary.json`` over every full-run recording present, resumable unit JSONs
under ``<out>/<FLY>/{coarse,gain,fine,maps}/raw/``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
for _p in (str(HERE), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from calib import grid, parab_min  # noqa: E402
from november_motor_clock import GAIN, NOMINAL, WINDOWS  # noqa: E402

from utils.gridrun import Unit, add_gridrun_args, run_grid, unit_path  # noqa: E402

#: The comb maps published on 2026-11-24 (docs/data-catalog.md), frozen here as the
#: comparison arm. Deliberately NOT read from ``new_motor_data._AUDIO_CLOCK``: that
#: table is the publication target and may adopt this script's own output.
COMB_MAPS: dict[str, tuple[float, float]] = {
    "FLY013": (29.091134, 84_000_904.0),
    "FLY018": (-53.894454, 84_000_510.0),
    "FLY022": (-273.865007, 84_001_237.0),
}
RECORDINGS = {Path(rel).stem: rel for rel in WINDOWS}
WAV_SR = 48_000
GAP_S = 0.20  # packets ~60 ms apart; a longer hole is a gap (as in the comb scan)
EDGE_S = 0.5  # unscored guard at both window ends (filter + VK edge transients)
MIN_RPS = 5.0  # VKConfig.min_rps: a slower rotor carries no comb
F_MIN = 60.0
#: Curve flags (frozen before any run): relative rise of the residual from its
#: minimum to the lower scan end; below it the lag is not localised by the scan.
FLAT_REL = {"coarse": 1e-3, "fine": 2e-4}
#: A coarse local minimum outside the main basin whose dip below the median is at
#: least this fraction of the global dip makes the window multimodal.
MULTIMODAL_FRAC = 0.5
#: Energy rise defining the reported sharpness half-width (a descriptor, not a CI).
SHARPNESS_ETA = 1e-3
#: A fit is well-conditioned only with at least this many usable fit windows.
MIN_FIT_WINDOWS = 3
MIN_HELDOUT_WINDOWS = 2
#: Envelope amplitude prior precision (``vk_envelopes(ridge=)``), relative to the unit
#: data weight of one envelope sample. Same-order lines of near-equal rotors overlap
#: for seconds; the improper curvature prior alone then leaves their difference mode
#: free and the coupled solve blows up (residual ratios >> 1 on FLY013 without it).
#: It shrinks a well-determined envelope by 1 / (1 + RIDGE), equally for every clock
#: hypothesis.
RIDGE = 0.1


@dataclass(frozen=True)
class OrderModel:
    family: str
    base_order: int  # carrier = base_order x shaft rate
    k_min: int  # VK harmonics of the carrier
    k_max: int
    fs: int  # analysis rate (Hz), a divisor of 48 kHz
    half_s: float  # window half-length (s), as in november_motor_clock.scan

    @property
    def shaft_orders(self) -> list[int]:
        return [self.base_order * k for k in range(self.k_min, self.k_max + 1)]


BARE = OrderModel("bare_motor_orders_84_98", 14, 6, 7, 48_000, 7.5)
BLADE = OrderModel("two_blade_bpf_orders_2_to_14", 2, 1, 7, 8_000, 5.7)


def model_for(rid: str) -> OrderModel:
    return BARE if rid == "FLY013" else BLADE


@dataclass(frozen=True)
class Stage:
    name: str
    bw_shaft: float  # rev/s capture radius: track band = shaft order x bw_shaft Hz
    halfwidth_s: float
    step_s: float
    max_extend: int


COARSE = Stage("coarse", 1.0, 1.0, 0.02, 0)
FINE = Stage("fine", 1.0, 0.05, 0.002, 3)


def vk_config(model: OrderModel, bw_shaft: float) -> Any:
    """Coupled VK-2 config: k-scaled band, envelope grid wide enough to hold it."""
    from tracking.vk_tracking import VKConfig

    top = model.base_order * model.k_max * bw_shaft
    fs_env = next(e for e in (100, 200, 400, 800) if 0.45 * e >= 1.5 * top)
    bw_rps = bw_shaft * model.base_order
    return VKConfig(
        fs=float(model.fs),
        fs_env=float(fs_env),
        k_min=model.k_min,
        k_max=model.k_max,
        f_min=F_MIN,
        f_max=0.45 * model.fs,
        min_rps=MIN_RPS,
        bw_rps=bw_rps,
        bw_hz=model.k_min * bw_rps,
        # No separation clamp: with four near-equal rotors it pins EVERY track of a
        # coupled group to the k_min band. The coupled solve attributes crossing
        # lines itself (PD repair / splu fallback inside vk_envelopes).
        sep_bw_factor=1e6,
        solver="banded",
    )


# ────────────────────────────────────────────────────────── telemetry
def default_root() -> Path:
    from data_processing.sources import raw_root

    return raw_root("new_motor_data")


@lru_cache(maxsize=4)
def packets(root: str, rid: str) -> tuple[np.ndarray, np.ndarray]:
    """Native motor packets sorted by unwrapped tick: ``(ticks int64, rev/s (N, 4))``."""
    from data_processing.sources.motor_dat import ROTORS, read_motor_dat

    motor = read_motor_dat(Path(root) / RECORDINGS[rid])["Motor"].sort_values("tick", kind="stable")
    ticks = motor["tick"].to_numpy(np.int64)
    if np.any(np.diff(ticks) <= 0):
        raise ValueError(f"{rid}: repeated motor packet ticks")
    return ticks, motor[list(ROTORS)].to_numpy(np.float64) / 60.0


def native_intervals(ticks: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(starts, ends)`` of packet runs (``tick / NOMINAL`` s) without a gap > GAP_S."""
    n = ticks.astype(np.float64) / NOMINAL
    cut = np.flatnonzero(np.diff(n) > GAP_S)
    return n[np.r_[0, cut + 1]], n[np.r_[cut, len(n) - 1]]


def covered_mask(t: np.ndarray, ticks: np.ndarray, o_lo: float, o_hi: float) -> np.ndarray:
    """Samples covered by real packet runs under EVERY offset in ``[o_lo, o_hi]``.

    ``audio = tick / NOMINAL + o``; ``t`` is covered for all ``o`` iff some run
    ``[a, b]`` contains ``[t - o_hi, t - o_lo]``, i.e. ``a + o_hi <= t <= b + o_lo``.
    """
    a, b = native_intervals(ticks)
    lo, hi = a + o_hi, b + o_lo
    keep = (hi > lo) & (hi >= t[0]) & (lo <= t[-1])
    mask = np.zeros(len(t), dtype=bool)
    for x0, x1 in zip(lo[keep], hi[keep], strict=True):
        mask[np.searchsorted(t, x0, "left") : np.searchsorted(t, x1, "right")] = True
    return mask


def trajectory(
    t: np.ndarray, ticks: np.ndarray, rps: np.ndarray, a: float, f: float, gain: float
) -> tuple[np.ndarray, np.ndarray]:
    """Shaft rev/s ``(4, T)`` on WAV times ``t`` under ``audio = a + tick / f``.

    Linear between neighbouring real packets only; outside the packet span and
    inside gaps the rotor speed is ZERO (no comb), never a held or bridged value.
    Returns ``(gain * r, covered)``.
    """
    tau = a + ticks.astype(np.float64) / f
    j = np.searchsorted(tau, t, side="right")
    ok = (j > 0) & (j < len(tau))
    jj = np.clip(j, 1, len(tau) - 1)
    ok &= (tau[jj] - tau[jj - 1]) <= GAP_S
    r = np.stack([np.interp(t, tau, rps[:, i]) for i in range(rps.shape[1])]) * gain
    r[:, ~ok] = 0.0
    return r, ok


def local_offset(a: float, f: float, centre: float) -> float:
    """The map's offset ``WAV - tick / NOMINAL`` at WAV time ``centre``."""
    return centre - (centre - a) * f / NOMINAL


def map_from_line(intercept: float, slope: float) -> tuple[float, float]:
    """``o = intercept + slope * n`` (``n = tick / NOMINAL``) as ``audio = A + tick / f``."""
    return intercept, NOMINAL / (1.0 + slope)


# ────────────────────────────────────────────────────────── one window
def analysis_band(
    model: OrderModel, ticks: np.ndarray, rps: np.ndarray, seed: float, centre: float, gains
) -> list[float]:
    """Hypothesis-independent band covering every modelled order in the window (+-20 %)."""
    n = ticks.astype(np.float64) / NOMINAL + seed
    span = model.half_s + COARSE.halfwidth_s + 0.5
    sel = (n >= centre - span) & (n <= centre + span)
    moving = rps[sel][rps[sel] >= MIN_RPS]
    if moving.size == 0:
        raise ValueError(f"no moving rotor near {centre} s")
    lo = max(F_MIN, 0.8 * min(model.shaft_orders) * float(moving.min()) * min(gains))
    hi = min(0.45 * model.fs, 1.2 * max(model.shaft_orders) * float(moving.max()) * max(gains))
    return [round(lo, 3), round(hi, 3)]


@dataclass
class Context:
    x: np.ndarray  # (2, T) band-limited WAV excerpt at model.fs
    t: np.ndarray  # (T,) WAV seconds
    mask: np.ndarray  # scored samples
    den: float  # ||x||^2 over the scored samples
    ticks: np.ndarray
    rps: np.ndarray
    cfg: Any
    model: OrderModel
    info: dict[str, Any]


def load_context(p: dict[str, Any], stage: Stage) -> Context:
    import soundfile as sf
    from scipy import signal

    rid, centre = p["rid"], float(p["centre"])
    model = model_for(rid)
    root = Path(p["root"])
    (wav,) = (root / RECORDINGS[rid]).parent.glob("*.wav")
    with sf.SoundFile(wav) as audio:
        if audio.samplerate != WAV_SR:
            raise ValueError(f"{wav}: {audio.samplerate} Hz, expected {WAV_SR}")
        start = round((centre - model.half_s) * WAV_SR)
        n = round(2 * model.half_s * WAV_SR)
        if start < 0 or start + n > audio.frames:
            raise ValueError(f"{rid} window at {centre} s leaves the WAV")
        audio.seek(start)
        x = audio.read(n, dtype="float64", always_2d=True).T
    if model.fs != WAV_SR:
        x = signal.resample_poly(x, 1, WAV_SR // model.fs, axis=-1)
    lo, hi = p["band"]
    sos = signal.butter(4, [lo, hi], btype="band", fs=model.fs, output="sos")
    x = signal.sosfiltfilt(sos, x, axis=-1)
    t = start / WAV_SR + np.arange(x.shape[-1]) / model.fs
    ticks, rps = packets(str(root), rid)
    o_lo, o_hi = p["mask_offsets"]
    near = (ticks / NOMINAL >= t[0] - o_hi - 2.0) & (ticks / NOMINAL <= t[-1] - o_lo + 2.0)
    ticks, rps = ticks[near], rps[near]
    guard = (t >= t[0] + EDGE_S) & (t <= t[-1] - EDGE_S)
    covered = covered_mask(t, ticks, o_lo, o_hi)
    mask = guard & covered
    if not mask.any():
        raise ValueError(f"{rid} window at {centre} s has no scored sample")
    info = {
        "scored_s": round(float(mask.sum()) / model.fs, 4),
        "telemetry_uncovered_s": round(float((guard & ~covered).sum()) / model.fs, 4),
        "edge_guard_s": EDGE_S,
        "fs": model.fs,
    }
    return Context(
        x,
        t,
        mask,
        float(np.sum(x[:, mask] ** 2)),
        ticks,
        rps,
        vk_config(model, stage.bw_shaft),
        model,
        info,
    )


def residual_energy(ctx: Context, a: float, f: float, gain: float) -> tuple[float, np.ndarray]:
    """``||x - x_hat||^2`` on the scored samples, plus the per-track VK bands (Hz)."""
    from tracking.vk_tracking import vk_envelopes, vk_reconstruct

    r, _ = trajectory(ctx.t, ctx.ticks, ctx.rps, a, f, gain)
    r *= ctx.model.base_order  # carrier rate; harmonics k_min..k_max = the shaft orders
    top = float(r.max()) * ctx.model.k_max
    if top > ctx.cfg.f_max:
        raise ValueError(f"order {top:.0f} Hz above the analysis band edge {ctx.cfg.f_max:.0f} Hz")
    env = vk_envelopes(ctx.x, r, ctx.cfg, ridge=np.full(4 * (ctx.cfg.k_max - ctx.cfg.k_min + 1), RIDGE))
    e = ctx.x - vk_reconstruct(env, n_samples=ctx.x.shape[-1])
    return float(np.sum(e[:, ctx.mask] ** 2)), env.bw_track


def analyse_curve(xs: Any, ys: Any, kind: str) -> dict[str, Any]:
    """Sub-grid minimum and the flags that make a lag estimate unusable."""
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    xm, ym, edge = parab_min(xs, ys)
    i = int(np.argmin(ys))
    rise = float((min(ys[0], ys[-1]) - ys[i]) / ys[i])
    out: dict[str, Any] = {
        "best": xm,
        "best_ratio": ym,
        "edge": bool(edge),
        "rise_rel": rise,
        "flat": bool(rise < FLAT_REL[kind]),
    }
    if 0 < i < len(xs) - 1:
        h = float(xs[i] - xs[i - 1])
        curv = float(ys[i - 1] - 2 * ys[i] + ys[i + 1]) / (2 * h * h)  # ratio ~ ym + curv d^2
        out["curvature_per_s2"] = curv
        if curv > 0:
            out["sharpness_ms"] = 1e3 * float(np.sqrt(SHARPNESS_ETA * ym / (2 * curv)))
    if kind == "coarse":
        plateau = float(np.median(ys))
        dip = plateau - ys[i]
        basin = ys < ys[i] + 0.5 * dip
        lo = hi = i
        while lo > 0 and basin[lo - 1]:
            lo -= 1
        while hi < len(ys) - 1 and basin[hi + 1]:
            hi += 1
        local = [
            j
            for j in range(1, len(ys) - 1)
            if (j < lo or j > hi) and ys[j] <= ys[j - 1] and ys[j] <= ys[j + 1]
        ]
        frac = max(((plateau - ys[j]) / dip for j in local), default=0.0) if dip > 0 else 1.0
        out |= {
            "basin": [float(xs[lo]), float(xs[hi])],
            "secondary_dip_frac": float(frac),
            "multimodal": bool(frac >= MULTIMODAL_FRAC),
        }
    out["usable"] = not (out["edge"] or out["flat"] or out.get("multimodal", False))
    return out


# ────────────────────────────────────────────────────────── grid units
def _worker(unit: Unit) -> dict[str, Any]:
    import torch

    torch.set_num_threads(1)
    p = dict(unit.params)
    stage = Stage(**p["stage"])
    t0 = time.time()
    ctx = load_context(p, stage)
    gain = float(p["gain"])
    out: dict[str, Any] = {
        "uid": unit.uid,
        "window": p["name"],
        "stage": stage.name,
        "gain": gain,
        "den": ctx.den,
        "window_info": ctx.info,
        "vk": {
            "fs_env": ctx.cfg.fs_env,
            "bw_rps_carrier": ctx.cfg.bw_rps,
            "nominal_band_hz": {
                str(o): round(o * stage.bw_shaft, 4) for o in ctx.model.shaft_orders
            },
        },
    }
    bands: list[np.ndarray] = []

    def energy(a: float, f: float) -> float:
        num, bw = residual_energy(ctx, a, f, gain)
        bands.append(bw)
        return num

    if p["kind"] == "maps":
        out["maps"] = {
            label: {"A": a, "f": f, "num": energy(a, f)} for label, a, f in p["maps"]
        }
    else:
        hw, step = stage.halfwidth_s, stage.step_s
        c = float(p["centre_offset"])
        nums = {float(o): energy(float(o), NOMINAL) for o in grid(c - hw, c + hw, step)}
        extended = 0
        while extended < stage.max_extend:
            lags = sorted(nums)
            i = min(range(len(lags)), key=lambda k: nums[lags[k]])
            if 0 < i < len(lags) - 1:
                break
            new = grid(lags[0] - hw, lags[0] - step, step) if i == 0 else grid(
                lags[-1] + step, lags[-1] + hw, step
            )
            nums |= {float(o): energy(float(o), NOMINAL) for o in new}
            extended += 1
        lags = sorted(nums)
        out["lags"] = lags
        out["nums"] = [nums[o] for o in lags]
        out["ratios"] = [float(np.sqrt(nums[o] / ctx.den)) for o in lags]
        out["extended"] = extended
        out["curve"] = analyse_curve(lags, out["ratios"], stage.name)
    bw = np.concatenate(bands)
    out["vk"]["track_band_hz_range"] = [float(bw.min()), float(bw.max())]
    out["wall_s"] = round(time.time() - t0, 2)
    return out


def _uid(prefix: str, params: dict[str, Any]) -> str:
    digest = hashlib.sha1(json.dumps(params, sort_keys=True).encode()).hexdigest()[:10]
    return f"{prefix}__{params['name']}__{digest}"


def run_phase(units: list[Unit], out_dir: Path, args: argparse.Namespace) -> dict[str, dict]:
    if not units:
        return {}
    run_grid(units, _worker, out_dir, jobs=args.jobs, resume=not args.no_resume)
    rows, missing = {}, []
    for u in units:
        path = unit_path(out_dir, u.uid)
        if path.exists():
            rows[u.uid] = json.loads(path.read_text())
        else:
            missing.append(u.uid)
    if missing:
        raise SystemExit(f"{len(missing)} unit(s) failed in {out_dir}: see raw/*.err ({missing[:3]})")
    return rows


# ────────────────────────────────────────────────────────── analysis
def comb_prior_gain(root: Path, rid: str, seed: float, fit: list[float]) -> dict[str, Any]:
    """The comb's own fit-window gain (``november_motor_clock``), fixed before any VK lag."""
    from november_motor_clock import scan as comb_scan

    ticks, rps = packets(str(root), rid)
    (wav,) = (root / RECORDINGS[rid]).parent.glob("*.wav")
    curves = np.stack(
        [comb_scan(wav, c, seed, ticks / NOMINAL, rps, rid == "FLY013")[1] for c in fit]
    )
    profile = curves.mean(axis=0).max(axis=1)
    k = int(profile.argmax())
    return {
        "gain": round(float(GAIN[k]), 6),
        "source": "november_motor_clock.scan on fit windows (comb prior)",
        "at_grid_edge": bool(k in (0, len(GAIN) - 1)),
        "grid": [round(float(g), 6) for g in GAIN],
        "profile": profile.tolist(),
    }


def fit_line(native: np.ndarray, offset: np.ndarray) -> dict[str, Any]:
    """OLS ``offset = a + b * native`` with SEs (dof >= 1) and leave-one-out jackknife."""
    n = len(native)
    b, a = np.polyfit(native, offset, 1)
    A, f = map_from_line(a, b)
    res = offset - (a + b * native)
    out: dict[str, Any] = {
        "n": n,
        "A_s": float(A),
        "rate_hz": float(f),
        "slope": float(b),
        "fit_residual_ms": (1e3 * res).tolist(),
    }
    if n > 2:
        s2 = float(res @ res) / (n - 2)
        sxx = float(np.sum((native - native.mean()) ** 2))
        se_b = np.sqrt(s2 / sxx)
        se_a = np.sqrt(s2 * (1.0 / n + native.mean() ** 2 / sxx))
        out |= {
            "se_A_ms": 1e3 * float(se_a),
            "se_rate_hz": float(NOMINAL * se_b / (1 + b) ** 2),
            "residual_sd_ms": 1e3 * float(np.sqrt(s2)),
            "se_native_mean": float(native.mean()),
            "sxx": sxx,
            "s2": s2,
        }
        loo = np.array(
            [
                map_from_line(*np.polyfit(np.delete(native, i), np.delete(offset, i), 1)[::-1])
                for i in range(n)
            ]
        )
        jk = np.sqrt((n - 1) / n * np.sum((loo - loo.mean(0)) ** 2, axis=0))
        out |= {
            "jackknife_se_A_ms": 1e3 * float(jk[0]),
            "jackknife_se_rate_hz": float(jk[1]),
            "loo_A_s": loo[:, 0].tolist(),
            "loo_rate_hz": loo[:, 1].tolist(),
        }
    return out


def prediction_se_ms(line: dict[str, Any], native: float) -> float | None:
    if "s2" not in line:
        return None
    m = line["se_native_mean"]
    return 1e3 * float(np.sqrt(line["s2"] * (1.0 / line["n"] + (native - m) ** 2 / line["sxx"])))


def rms(v: Any) -> float | None:
    v = np.asarray(list(v), float)
    return float(np.sqrt(np.mean(v**2))) if v.size else None


def decide(
    *,
    well_conditioned: bool,
    reasons: list[str],
    pooled: dict[str, float],
    lag_rms_ms: dict[str, float | None],
    max_map_diff_ms: float | None,
    scatter_ms: float | None,
) -> dict[str, Any]:
    """Frozen rule: keep the comb map unless a well-conditioned VK fit wins BOTH tests.

    ``indecisive_data``  VK fit not identifiable here: comb kept, NOT validated.
    ``comb_validated``   maps differ by <= the held-out lag scatter about the VK map.
    ``vk_preferred``     VK map lower on pooled held-out residual AND held-out lag RMS.
    ``comb_preferred``   comb lower on both: comb supported over the VK candidate.
    ``mixed_evidence``   each map wins one test: comb kept, but not supported.
    """
    rule = decide.__doc__
    if not well_conditioned:
        verdict = "indecisive_data"
    elif max_map_diff_ms is not None and scatter_ms is not None and max_map_diff_ms <= scatter_ms:
        verdict = "comb_validated"
    else:
        vk_res = pooled["vk"] < pooled["comb"]
        vk_lag = lag_rms_ms["vk"] < lag_rms_ms["comb"]  # type: ignore[operator]
        verdict = (
            "vk_preferred"
            if vk_res and vk_lag
            else "comb_preferred"
            if not (vk_res or vk_lag)
            else "mixed_evidence"
        )
    return {
        "verdict": verdict,
        "selected_map": "vk" if verdict == "vk_preferred" else "comb",
        "comb_supported": verdict in ("comb_validated", "comb_preferred"),
        "not_well_conditioned_because": reasons,
        "rule": rule,
        "label": "best supported map under the VK residual referee; not exact truth",
    }


# ────────────────────────────────────────────────────────── one recording
def window_list(rid: str, smoke: bool) -> list[dict[str, Any]]:
    seed, fit, held = WINDOWS[RECORDINGS[rid]]
    if smoke:
        fit, held = fit[:2], held[:1]
    lo, hi = min(fit), max(fit)
    rows = [{"role": "fit", "centre": c, "position": "fit"} for c in fit]
    rows += [
        {
            "role": "heldout",
            "centre": c,
            "position": "early" if c < lo else "late" if c > hi else "interior",
        }
        for c in held
    ]
    for r in rows:
        r["name"] = f"{rid}_{r['role']}_{r['centre']:.2f}".replace(".", "p")
    return rows


def run_recording(rid: str, args: argparse.Namespace) -> dict[str, Any]:
    root = Path(args.root) if args.root else default_root()
    model = model_for(rid)
    seed, fit_c, _ = WINDOWS[RECORDINGS[rid]]
    coarse, fine = COARSE, FINE
    if args.smoke:
        coarse = replace(COARSE, step_s=0.1)
        fine = replace(FINE, halfwidth_s=0.01, step_s=0.005, max_extend=1)
    tag = f"{rid}_smoke" if args.smoke else rid
    out_dir = Path(args.out) / tag
    ticks, rps = packets(str(root), rid)  # loaded before the pool forks
    comb_a, comb_f = COMB_MAPS[rid]
    wins = window_list(rid, args.smoke)

    # Nuisance gain: fixed before any VK lag is looked at (profile mode: fit only).
    if args.gain == "comb":
        gain_info = comb_prior_gain(root, rid, seed, [w["centre"] for w in wins if w["role"] == "fit"])
    elif args.gain == "profile":
        gain_info = {"gain": 1.0, "source": "VK profile on usable fit windows"}
    else:
        gain_info = {"gain": float(args.gain), "source": "fixed on the command line"}
    gain_grid = [round(float(g), 6) for g in (GAIN[::10] if args.smoke else GAIN)]
    g_lo, g_hi = (min(gain_grid), max(gain_grid)) if args.gain == "profile" else (gain_info["gain"],) * 2

    for w in wins:
        w["band"] = analysis_band(model, ticks, rps, seed, w["centre"], (g_lo, g_hi))
        comb_o = local_offset(comb_a, comb_f, w["centre"])
        lo, hi = seed - coarse.halfwidth_s, seed + coarse.halfwidth_s
        w["mask_offsets"] = [round(min(lo, comb_o) - 0.06, 6), round(max(hi, comb_o) + 0.06, 6)]
        w["comb_offset"] = comb_o

    def params(w: dict[str, Any], stage: Stage, gain: float, **kw: Any) -> dict[str, Any]:
        base = {k: w[k] for k in ("name", "centre", "band", "mask_offsets")}
        return {"rid": rid, "root": str(root), "stage": asdict(stage), "gain": gain, **base, **kw}

    # Coarse: around the independent transition seed, never the comb map.
    g0 = gain_info["gain"]
    units = [Unit(_uid("coarse", p), p) for w in wins for p in [params(w, coarse, g0, kind="scan", centre_offset=seed)]]
    rows = run_phase(units, out_dir / "coarse", args)
    for w, u in zip(wins, units, strict=True):
        w["coarse"] = rows[u.uid]

    usable_fit = [w for w in wins if w["role"] == "fit" and w["coarse"]["curve"]["usable"]]
    if args.gain == "profile":
        units = [
            Unit(_uid(f"gain{g:.6f}", p), p)
            for g in gain_grid
            for w in usable_fit
            for p in [params(w, fine, g, kind="scan", centre_offset=w["coarse"]["curve"]["best"])]
        ]
        rows = run_phase(units, out_dir / "gain", args)
        cells = [rows[u.uid] for u in units]
        per_gain, per_window = [], {w["name"]: [] for w in usable_fit}
        for g in gain_grid:
            sel = [c for c in cells if c["gain"] == g]
            per_gain.append(float(np.sqrt(sum(min(c["nums"]) for c in sel) / sum(c["den"] for c in sel))))
            for c in sel:
                per_window[c["window"]].append(min(c["ratios"]))
        gm, gy, gedge = parab_min(gain_grid, per_gain)
        rise = (min(per_gain[0], per_gain[-1]) - min(per_gain)) / min(per_gain) if per_gain else 0.0
        gain_info |= {
            "gain": round(float(gm), 6),
            "grid": gain_grid,
            "pooled_profile": per_gain,
            "per_window_ratio": per_window,
            "per_window_argmin": {k: gain_grid[int(np.argmin(v))] for k, v in per_window.items()},
            "at_grid_edge": bool(gedge),
            "flat": bool(rise < FLAT_REL["fine"]),
            "edge_lag_cells": sum(c["curve"]["edge"] for c in cells),
            "windows": [w["name"] for w in usable_fit],
        }
    gain = gain_info["gain"]

    # Fine: every window at the fixed gain, around its own coarse minimum.
    units = [
        Unit(_uid("fine", p), p)
        for w in wins
        for p in [params(w, fine, gain, kind="scan", centre_offset=w["coarse"]["curve"]["best"])]
    ]
    rows = run_phase(units, out_dir / "fine", args)
    for w, u in zip(wins, units, strict=True):
        w["fine"] = rows[u.uid]
        w["usable"] = bool(w["coarse"]["curve"]["usable"] and w["fine"]["curve"]["usable"])
        w["vk_offset"] = w["fine"]["curve"]["best"]
        w["native_s"] = w["centre"] - w["vk_offset"]
        w["outside_mask_range"] = not (w["mask_offsets"][0] <= w["vk_offset"] <= w["mask_offsets"][1])

    fits = [w for w in wins if w["role"] == "fit" and w["usable"]]
    held = [w for w in wins if w["role"] == "heldout"]
    reasons = []
    if len(fits) < MIN_FIT_WINDOWS:
        reasons.append(f"{len(fits)} usable fit windows < {MIN_FIT_WINDOWS}")
    if gain_info.get("at_grid_edge"):
        reasons.append("nuisance gain at its grid edge")
    if gain_info.get("flat"):
        reasons.append("nuisance gain profile flat")
    if sum(w["usable"] for w in held) < MIN_HELDOUT_WINDOWS:
        reasons.append(f"fewer than {MIN_HELDOUT_WINDOWS} usable held-out windows")
    if any(w["outside_mask_range"] for w in wins):
        reasons.append("a local VK offset left the scored offset range")

    line = None
    vk_map = None
    if len(fits) >= 2:
        line = fit_line(np.array([w["native_s"] for w in fits]), np.array([w["vk_offset"] for w in fits]))
        vk_map = (line["A_s"], line["rate_hz"])

    # Maps: actual residual of the comb map and the VK candidate on every window.
    maps = [["comb", comb_a, comb_f]] + ([["vk", *vk_map]] if vk_map else [])
    units = [Unit(_uid("maps", p), p) for w in wins for p in [params(w, fine, gain, kind="maps", maps=maps)]]
    rows = run_phase(units, out_dir / "maps", args)
    for w, u in zip(wins, units, strict=True):
        m = rows[u.uid]
        w["map_ratio"] = {k: float(np.sqrt(v["num"] / m["den"])) for k, v in m["maps"].items()}
        w["map_num"] = {k: v["num"] for k, v in m["maps"].items()}
        w["den"] = m["den"]
        w["comb_lag_resid_ms"] = 1e3 * (w["vk_offset"] - w["comb_offset"])
        if vk_map:
            w["vk_map_offset"] = local_offset(*vk_map, w["centre"])
            w["vk_lag_resid_ms"] = 1e3 * (w["vk_offset"] - w["vk_map_offset"])
            w["map_diff_ms"] = 1e3 * (w["vk_map_offset"] - w["comb_offset"])
            w["vk_pred_se_ms"] = prediction_se_ms(line, w["native_s"])
    lo_hi = [(w["mask_offsets"], w.get("vk_map_offset")) for w in wins]
    if vk_map and any(not (m[0] <= o <= m[1]) for m, o in lo_hi):
        reasons.append("the VK map leaves a window's scored offset range")

    def pooled(ws: list[dict[str, Any]], key: str) -> float | None:
        if not ws or key not in ws[0]["map_num"]:
            return None
        return float(np.sqrt(sum(w["map_num"][key] for w in ws) / sum(w["den"] for w in ws)))

    held_ok = [w for w in held if w["usable"]]
    evidence: dict[str, Any] = {
        "pooled_heldout_ratio": {"comb": pooled(held, "comb"), "vk": pooled(held, "vk")},
        "pooled_fit_ratio": {
            "comb": pooled([w for w in wins if w["role"] == "fit"], "comb"),
            "vk": pooled([w for w in wins if w["role"] == "fit"], "vk"),
        },
        "pooled_heldout_local_min_ratio": float(
            np.sqrt(sum(min(w["fine"]["nums"]) for w in held) / sum(w["den"] for w in held))
        ),
        "heldout_lag_rms_ms": {
            "comb": rms(w["comb_lag_resid_ms"] for w in held_ok),
            "vk": rms(w["vk_lag_resid_ms"] for w in held_ok) if vk_map else None,
        },
        "heldout_lag_rms_ms_by_position": {
            pos: {
                "comb": rms(w["comb_lag_resid_ms"] for w in held_ok if w["position"] == pos),
                "vk": rms(w["vk_lag_resid_ms"] for w in held_ok if w["position"] == pos)
                if vk_map
                else None,
                "n": sum(w["position"] == pos for w in held_ok),
            }
            for pos in ("early", "interior", "late")
        },
        "heldout_windows_lower_residual": {
            "vk": sum(w["map_ratio"].get("vk", np.inf) < w["map_ratio"]["comb"] for w in held),
            "comb": sum(w["map_ratio"]["comb"] < w["map_ratio"].get("vk", -np.inf) for w in held),
        },
        "max_heldout_map_diff_ms": max(abs(w["map_diff_ms"]) for w in held) if vk_map else None,
        "map_diff_ms_over_windows": [w.get("map_diff_ms") for w in wins],
    }
    scatter = evidence["heldout_lag_rms_ms"]["vk"]
    evidence["heldout_lag_scatter_ms"] = scatter
    decision = decide(
        well_conditioned=vk_map is not None and not reasons,
        reasons=reasons if vk_map else reasons + ["no VK line (fewer than 2 usable fit windows)"],
        pooled=evidence["pooled_heldout_ratio"],
        lag_rms_ms=evidence["heldout_lag_rms_ms"],
        max_map_diff_ms=evidence["max_heldout_map_diff_ms"],
        scatter_ms=scatter,
    )
    sel = vk_map if decision["selected_map"] == "vk" else (comb_a, comb_f)
    decision["selected"] = {"A_s": sel[0], "rate_hz": sel[1]}

    keep = ("name", "role", "position", "centre", "band", "mask_offsets", "usable", "comb_offset",
            "vk_offset", "native_s", "outside_mask_range", "map_ratio", "comb_lag_resid_ms",
            "vk_map_offset", "vk_lag_resid_ms", "map_diff_ms", "vk_pred_se_ms", "den")  # fmt: skip
    windows_out = []
    for w in wins:
        row = {k: w[k] for k in keep if k in w}
        for st in ("coarse", "fine"):
            s = w[st]
            row[st] = {k: s[k] for k in ("lags", "ratios", "curve", "extended", "vk", "wall_s")}
        row["window_info"] = w["fine"]["window_info"]
        windows_out.append(row)

    all_ok = [w for w in wins if w["usable"]]
    report = {
        "recording": rid,
        "smoke": bool(args.smoke),
        "root": str(root),
        "metric": "||x - x_hat|| / ||x|| on scored samples of the band-limited stereo WAV excerpt",
        "model": {
            "family": model.family,
            "shaft_orders": model.shaft_orders,
            "carrier_base_order": model.base_order,
            "vk_harmonics": [model.k_min, model.k_max],
            "fs": model.fs,
            "window_half_s": model.half_s,
            "vk": "tracking.vk_tracking.vk_envelopes (coupled VK-2, k-scaled bw_rps, banded "
            "Cholesky per coupling group) + vk_reconstruct; 2 channels",
            "stages": {s.name: asdict(s) for s in (coarse, fine)},
            "vk_config": {s.name: asdict(vk_config(model, s.bw_shaft)) for s in (coarse, fine)},
            "gap_s": GAP_S,
            "edge_s": EDGE_S,
            "flat_rel": FLAT_REL,
            "multimodal_frac": MULTIMODAL_FRAC,
            "sharpness_eta": SHARPNESS_ETA,
        },
        "seed_offset_s": seed,
        "nominal_hz": NOMINAL,
        "gain": gain_info,
        "comb_map": {"A_s": comb_a, "rate_hz": comb_f, "source": "published 2026-11-24, frozen"},
        "vk_fit": line,
        "vk_map": None if not vk_map else {
            "A_s": vk_map[0],
            "rate_hz": vk_map[1],
            "A_minus_comb_ms": 1e3 * (vk_map[0] - comb_a),
            "rate_minus_comb_ppm": 1e6 * (vk_map[1] / comb_f - 1),
        },
        "diagnostic_all_window_fit": fit_line(
            np.array([w["native_s"] for w in all_ok]), np.array([w["vk_offset"] for w in all_ok])
        ) if len(all_ok) >= 2 else None,
        "evidence": evidence,
        "decision": decision,
        "windows": windows_out,
    }
    path = Path(args.out) / f"{tag}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=1))
    print(f"{rid}: {decision['verdict']} -> {decision['selected_map']} ({path})", flush=True)
    return report


def write_summary(out: Path) -> None:
    rows = {}
    for path in sorted(out.glob("FLY*.json")):
        if path.stem.endswith("_smoke"):
            continue
        r = json.loads(path.read_text())
        rows[r["recording"]] = {
            "verdict": r["decision"]["verdict"],
            "selected": r["decision"]["selected"],
            "comb_supported": r["decision"]["comb_supported"],
            "reasons": r["decision"]["not_well_conditioned_because"],
            "comb_map": r["comb_map"],
            "vk_map": r["vk_map"],
            "gain": r["gain"]["gain"],
            "evidence": {
                k: r["evidence"][k]
                for k in ("pooled_heldout_ratio", "heldout_lag_rms_ms", "max_heldout_map_diff_ms")
            },
        }
    (out / "summary.json").write_text(json.dumps(rows, indent=1))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--recording", nargs="+", choices=sorted(RECORDINGS), default=sorted(RECORDINGS))
    ap.add_argument("--root", help="raw new_motor_data tree (default: the pinned dload dataset)")
    ap.add_argument("--out", default="results/new_motor_data_alignment/vk_clock")
    ap.add_argument("--gain", default="comb", help="'comb' (prior), 'profile' (VK, fit only) or a number")
    ap.add_argument("--smoke", action="store_true", help="2 fit + 1 held-out window, coarse grids")
    add_gridrun_args(ap, jobs=1)
    args = ap.parse_args(argv)
    if args.gain not in ("comb", "profile"):
        float(args.gain)
    for rid in args.recording:
        run_recording(rid, args)
    write_summary(Path(args.out))


if __name__ == "__main__":
    main()
