"""The whole-flight layer: replayed transients around the cruise model.

Start-up, take-off and landing are controller programs, not noise: on MD2 the
four recorded start-ups coincide to 5 % of the standby level, and every
measured take-off is a ramp at one rate.  So the layer REPLAYS them.  What it
replays was chosen by which part of a transient another flight of the same rig
repeats (campaign doc ``docs/experiments/rps-flight-regimes.md``):

* **common level** — the rotor mean, as a fraction of the plateau it leaves
  or joins: the start-up waveform with its spikes, the take-off overshoot,
  the landing drop, shelf and spin-down.  The take-off ramp itself is one
  rate (rev/s per second), because a smaller step is covered sooner.
* **pattern hand-over** — how the per-rotor split moves from the old
  plateau's pattern to the new one.  During the MD2 start-up spike the four
  rotors run in unison and split only while settling; at take-off the split
  stays the standby one up the ramp and switches at arrival, which is the
  jump of single rotors at the top of the ramp.
* **fan-out** — after touchdown the rotors spread for about two seconds and
  collapse with the spin-down (20-40 rev/s on MD2).  Its time course repeats;
  its size and direction do not, so each landing draws one amplitude in
  (roll, pitch, yaw).

What is left after those is fluctuation of cruise kind, about twice the
settled size for three seconds after arrival: the cruise model, switched on
with the hand-over and held at its loudest level until it has settled.

What is random per flight: the plateau levels, the time on the ground before
take-off, the time airborne, the fan-out amplitude and the cruise process.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, NamedTuple

import numpy as np
from scipy.ndimage import uniform_filter1d

from data_processing.trajectory_model import Params
from experiments.rps_traj.cruise_api import NUM_ROTORS
from experiments.rps_traj.data import RATE_HZ, Flight
from experiments.rps_traj.phases import (
    ARRIVAL_FRACTION,
    CLEAR_BEFORE_LANDING_S,
    SETTLE_AFTER_ARRIVAL_S,
    Landmarks,
    bridge_nans,
    landmarks,
)
from experiments.rps_traj.switching import SwitchingCruise
from tracking.rotors import MIXER

#: The arrival window (s before / after the arrival) and the lead of the
#: landing window before the landing mark (s).
ARRIVAL_PRE_S = 1.5
LANDING_PRE_S = CLEAR_BEFORE_LANDING_S

#: Columns of the mixer that spread the rotors: roll, pitch, yaw.
_SPREAD = MIXER[:, 1:]
_TO_SPREAD = np.linalg.inv(MIXER)[1:]


def _window(x: np.ndarray, at: int, pre_s: float, post_s: float, fs: float) -> np.ndarray:
    """``x[:, at - pre : at + post]``, edge-padded where the track ends."""
    idx = np.arange(at - int(round(pre_s * fs)), at + int(round(post_s * fs)))
    return x[:, np.clip(idx, 0, x.shape[1] - 1)]


def _smooth(y: np.ndarray, seconds: float, fs: float) -> np.ndarray:
    return uniform_filter1d(y, max(int(round(seconds * fs)), 1), mode="nearest")


def _blend_end(y: np.ndarray, target: float, seconds: float, fs: float) -> np.ndarray:
    """``y`` with its last ``seconds`` cross-faded onto ``target``."""
    n = min(int(round(seconds * fs)), y.size)
    out = y.copy()
    if n > 0:
        w = np.linspace(0.0, 1.0, n)
        out[-n:] = (1.0 - w) * out[-n:] + w * target
    return out


def _blend_start(y: np.ndarray, target: float, seconds: float, fs: float) -> np.ndarray:
    return _blend_end(y[::-1], target, seconds, fs)[::-1]


def _lognormal(values: Sequence[float]) -> tuple[float, float]:
    """``(mean, sd)`` of the logs (sd 0 for a single value)."""
    logs = np.log(np.asarray(values, dtype=np.float64))
    return float(logs.mean()), float(logs.std(ddof=1)) if logs.size > 1 else 0.0


def _hand_over(split: np.ndarray, d_from: np.ndarray, d_to: np.ndarray) -> np.ndarray | None:
    """Least-squares hand-over ``a(t)`` of ``split = d_from + a (d_to - d_from)``
    over flights: ``split`` ``(F, 4, T)``, patterns ``(F, 4)``.  ``None`` when
    the two patterns do not differ enough to define one."""
    step = d_to - d_from
    power = float(np.sum(step * step))
    scale = float(np.mean(np.abs(d_from)) + np.mean(np.abs(d_to)))
    if power < 1e-6 * max(scale * scale, 1e-12) * step.size or power < 1e-9:
        return None
    return np.einsum("frt,fr->t", split - d_from[:, :, None], step) / power


@dataclass
class PhaseLayer:
    """The fitted whole-flight layer of one rig; curves on the ``fs`` grid.

    Every ``*_common`` curve is a rotor mean relative to a plateau, every
    ``*_split`` curve a hand-over between 0 (old per-rotor pattern) and 1 (new
    one).  ``landing_*`` is ``None`` when no training flight lands.
    """

    fs: float
    startup_common: np.ndarray
    startup_split: np.ndarray
    standby_level: np.ndarray
    standby_level_sd: np.ndarray
    pre_takeoff_log_s: tuple[float, float]
    takeoff_rate: float
    arrival_common: np.ndarray
    arrival_split: np.ndarray
    airborne_log_s: tuple[float, float]
    landing_common: np.ndarray | None = None
    landing_fan: np.ndarray | None = None
    landing_amp_mean: np.ndarray | None = None
    landing_amp_sd: np.ndarray | None = None
    counts: dict[str, int] = field(default_factory=dict)

    # ── generation ──
    def sample(
        self,
        cruise: SwitchingCruise,
        rng: np.random.Generator,
        *,
        airborne_s: float | None = None,
        pre_takeoff_s: float | None = None,
        ground_s: tuple[float, float] = (1.0, 1.0),
    ) -> dict[str, Any]:
        """One whole flight ``rps (4, n)`` at ``fs`` (shaft speed, no
        measurement process) and the sample indices of its phase changes.

        ``cruise`` supplies the levels (``mu``, ``s_c``, ``s_r``) and the
        fluctuation; ``airborne_s`` fixes the time from arrival to landing and
        ``pre_takeoff_s`` the time from motors-on to take-off instead of
        drawing them (a given time on the ground shorter than the start-up
        curve cuts the curve short).
        """
        fs = self.fs
        params: Params = cruise.params
        hover = params.mu + rng.standard_normal() * params.s_c + rng.standard_normal(4) * params.s_r
        standby = np.maximum(
            self.standby_level + rng.standard_normal(4) * self.standby_level_sd,
            0.25 * self.standby_level,
        )
        sb, h0 = float(standby.mean()), float(hover.mean())
        step = h0 - sb
        d_sb, d_cr = standby - sb, hover - h0

        n_su = self.startup_common.size
        if pre_takeoff_s is None:
            pre_s = float(
                np.exp(
                    self.pre_takeoff_log_s[0] + self.pre_takeoff_log_s[1] * rng.standard_normal()
                )
            )
            n_pre = max(int(round(pre_s * fs)), n_su)
        else:
            n_pre = max(int(round(float(pre_takeoff_s) * fs)), 1)
        air_s = (
            float(np.exp(self.airborne_log_s[0] + self.airborne_log_s[1] * rng.standard_normal()))
            if airborne_s is None
            else float(airborne_s)
        )
        n_ramp = max(int(round(ARRIVAL_FRACTION * step / self.takeoff_rate * fs)), 1)
        n_arr_pre = int(round(ARRIVAL_PRE_S * fs))
        n_arr = self.arrival_common.size  # arrival window, starts ARRIVAL_PRE_S before arrival
        n_air = max(int(round(air_s * fs)), n_arr - n_arr_pre + 1)
        lands = self.landing_common is not None
        n_ld_pre = int(round(LANDING_PRE_S * fs))
        n_tail = (self.landing_common.size - n_ld_pre) if self.landing_common is not None else 0

        g0 = int(round(ground_s[0] * fs))
        i_on = g0
        i_to = i_on + n_pre
        i_arr = i_to + n_ramp
        i_ld = i_arr + n_air
        n = i_ld + n_tail + int(round(ground_s[1] * fs))
        common = np.zeros(n)  # rotor mean
        split = np.zeros((NUM_ROTORS, n))  # per-rotor departure from the rotor mean
        gate = np.zeros(n)

        # start-up and standby
        idx = np.minimum(np.arange(n_pre), n_su - 1)
        common[i_on:i_to] = sb * self.startup_common[idx]
        split[:, i_on:i_to] = d_sb[:, None] * self.startup_split[idx][None, :]
        # take-off ramp, at one rate, to the arrival fraction of the step
        common[i_to:i_arr] = sb + ARRIVAL_FRACTION * step * (np.arange(n_ramp) + 1) / n_ramp
        split[:, i_to:i_arr] = d_sb[:, None]
        # arrival window: overshoot and settling of the level, hand-over of the pattern
        a = np.ones(n)
        a[:i_to] = 0.0
        w0 = i_arr - n_arr_pre
        lo = max(w0, i_to)
        a[lo : w0 + n_arr] = self.arrival_split[lo - w0 : n_arr]
        a[i_to:lo] = 0.0
        post = n_arr - n_arr_pre
        common[i_arr : i_arr + post] = sb + step * self.arrival_common[n_arr_pre:]
        common[i_arr + post : i_ld] = h0
        split[:, i_to:i_ld] = d_sb[:, None] * (1.0 - a[i_to:i_ld]) + d_cr[:, None] * a[i_to:i_ld]
        gate[i_to:i_ld] = np.clip(a[i_to:i_ld], 0.0, 1.0)
        # landing
        if lands and self.landing_common is not None and self.landing_fan is not None:
            w1 = i_ld - n_ld_pre
            frac = self.landing_common
            common[w1 : w1 + frac.size] = h0 * frac
            split[:, w1 : w1 + frac.size] = d_cr[:, None] * frac[None, :]
            amp = np.zeros(3)
            if self.landing_amp_mean is not None and self.landing_amp_sd is not None:
                amp = self.landing_amp_mean + self.landing_amp_sd * rng.standard_normal(3)
            split[:, w1 : w1 + frac.size] += (h0 * (_SPREAD @ amp))[:, None] * self.landing_fan[
                None, :
            ]
            gate[w1 : w1 + frac.size] = np.clip((frac - 0.5) / 0.4, 0.0, 1.0)
            gate[w1 + frac.size :] = 0.0
        # the cruise process, loudest until the take-off has settled
        n_fl = (i_ld + n_tail) - i_to
        forced = (i_arr + int(round(SETTLE_AFTER_ARRIVAL_S * fs))) - i_to
        fluct = cruise.sample(n_fl, rng, rate_hz=fs, forced=forced, shaft_only=True)
        rps = common[None, :] + split
        rps[:, i_to : i_to + n_fl] += gate[None, i_to : i_to + n_fl] * fluct
        return {
            "rps": np.clip(rps, 0.0, None),
            "motors_on": i_on,
            "takeoff": i_to,
            "arrival": i_arr,
            "landing": i_ld if lands else None,
            "hover": hover,
            "standby": standby,
        }

    # ── serialisation ──
    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in self.__dict__.items():
            out[key] = value.tolist() if isinstance(value, np.ndarray) else value
        return out

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> PhaseLayer:
        kw: dict[str, Any] = {}
        for key, value in payload.items():
            if key in ("pre_takeoff_log_s", "airborne_log_s"):
                kw[key] = (float(value[0]), float(value[1]))
            elif isinstance(value, list):
                kw[key] = np.asarray(value, dtype=np.float64)
            else:
                kw[key] = value
        return cls(**kw)


class _Up(NamedTuple):
    """One recorded way up: the gap-bridged track and the marks it needs."""

    x: np.ndarray
    on: int
    settled: int
    takeoff: int
    arrival: int
    n: int
    level: np.ndarray
    hover: np.ndarray


class _Down(NamedTuple):
    """One recorded landing; ``off`` is motors-off or the end of the track."""

    x: np.ndarray
    landing: int
    off: int
    hover: np.ndarray


def _marked(flights: Sequence[Flight]) -> list[tuple[np.ndarray, Landmarks]]:
    out = []
    for flight in flights:
        marks = landmarks(flight.rps, flight.fs)
        if marks is not None:
            out.append((bridge_nans(np.asarray(flight.rps, dtype=np.float64)), marks))
    return out


def _ups(marked: Sequence[tuple[np.ndarray, Landmarks]]) -> list[_Up]:
    out = []
    for x, m in marked:
        if (
            m.motors_on is None
            or m.standby is None
            or m.standby_level is None
            or m.takeoff is None
            or m.arrival is None
            or m.hover_in is None
        ):
            continue
        out.append(
            _Up(
                x, m.motors_on, m.standby[0], m.takeoff, m.arrival, m.n, m.standby_level, m.hover_in
            )
        )
    return out


def _downs(marked: Sequence[tuple[np.ndarray, Landmarks]]) -> list[_Down]:
    out = []
    for x, m in marked:
        if m.landing is None or m.hover_out is None:
            continue
        out.append(_Down(x, m.landing, m.n if m.motors_off is None else m.motors_off, m.hover_out))
    return out


def fit_phase_layer(flights: Sequence[Flight], fs: float = RATE_HZ) -> PhaseLayer:
    """Fit the layer to the training flights of one rig (all on the ``fs``
    grid): medians and least squares over aligned windows, nothing iterative.

    Needs at least one flight with a standby and a take-off; the landing part
    is fitted from the flights that land.
    """
    marked = _marked(flights)
    ups = _ups(marked)
    if not ups:
        raise ValueError("no flight with a standby and a take-off: nothing to fit a layer to")

    # start-up and standby: from motors-on, as long as every flight is still on the ground
    spans = [(u.takeoff - u.on) / fs for u in ups]
    settle = [(u.settled - u.on) / fs for u in ups]
    t_su = max(min(min(spans), max(settle) + 2.0), 0.2)
    win = np.stack([_window(u.x, u.on, 0.0, t_su, fs) for u in ups])
    level = np.stack([u.level for u in ups])
    mean_level = level.mean(axis=1)
    c = win.mean(axis=1)
    startup_common = np.median(c / mean_level[:, None], axis=0)
    a = _hand_over(win - c[:, None, :], np.zeros_like(level), level - mean_level[:, None])
    startup_split = np.clip(startup_common, 0.0, 1.0) if a is None else a
    startup_common = _blend_end(startup_common, 1.0, 0.2, fs)
    startup_split = _blend_end(startup_split, 1.0, 0.2, fs)

    # take-off: one rate up the ramp, then the arrival window
    hover_in = np.stack([u.hover for u in ups])
    h0 = hover_in.mean(axis=1)
    step = h0 - mean_level
    rise = np.array([(u.arrival - u.takeoff) / fs for u in ups])
    rate = float(np.median(ARRIVAL_FRACTION * step / np.maximum(rise, 1.0 / fs)))
    win = np.stack(
        [_window(u.x, u.arrival, ARRIVAL_PRE_S, SETTLE_AFTER_ARRIVAL_S, fs) for u in ups]
    )
    c = win.mean(axis=1)
    progress = np.median((c - mean_level[:, None]) / step[:, None], axis=0)
    n_pre = int(round(ARRIVAL_PRE_S * fs))
    post = _smooth(progress, 0.1, fs)
    post[n_pre] = ARRIVAL_FRACTION  # the ramp ends exactly where this window takes over
    arrival_common = _blend_end(post, 1.0, 0.5, fs)
    d_sb, d_cr = level - mean_level[:, None], hover_in - h0[:, None]
    a = _hand_over(win - c[:, None, :], d_sb, d_cr)
    if a is None:
        a = np.clip(progress, 0.0, 1.0)
    a = _smooth(a, 0.1, fs)
    arrival_split = _blend_end(_blend_start(a, 0.0, 0.3, fs), 1.0, 0.5, fs)

    downs = _downs(marked)
    air = [
        (m.landing - m.arrival) / fs
        for _, m in marked
        if m.landing is not None and m.arrival is not None
    ]
    if not air:  # no flight is recorded from take-off to landing: the recorded airborne spans
        air = [(u.n - u.arrival) / fs for u in ups]
    layer = PhaseLayer(
        fs=float(fs),
        startup_common=startup_common,
        startup_split=startup_split,
        standby_level=level.mean(axis=0),
        standby_level_sd=level.std(axis=0, ddof=1) if len(ups) > 1 else np.zeros(NUM_ROTORS),
        pre_takeoff_log_s=_lognormal(spans),
        takeoff_rate=rate,
        arrival_common=arrival_common,
        arrival_split=arrival_split,
        airborne_log_s=_lognormal(air),
        counts={"takeoffs": len(ups), "landings": len(downs)},
    )
    if downs:
        tail = max((d.off - d.landing) / fs for d in downs)
        win = np.stack([_window(d.x, d.landing, LANDING_PRE_S, tail + 0.5, fs) for d in downs])
        hover_out = np.stack([d.hover for d in downs])
        h1 = hover_out.mean(axis=1)
        c = win.mean(axis=1)
        frac = np.median(c / h1[:, None], axis=0)
        frac = _blend_end(_blend_start(frac, 1.0, 0.5, fs), 0.0, 0.3, fs)
        rest = win - c[:, None, :] - (hover_out - h1[:, None])[:, :, None] * frac[None, None, :]
        rest = rest / h1[:, None, None]
        _, _, vt = np.linalg.svd(rest.reshape(-1, rest.shape[-1]), full_matrices=False)
        fan = vt[0] * np.sign(vt[0][np.argmax(np.abs(vt[0]))])
        fan = _blend_end(_blend_start(fan / np.abs(fan).max(), 0.0, 0.3, fs), 0.0, 0.3, fs)
        amp = np.stack([_TO_SPREAD @ (r @ fan / float(fan @ fan)) for r in rest])
        layer.landing_common = frac
        layer.landing_fan = fan
        layer.landing_amp_mean = amp.mean(axis=0)
        layer.landing_amp_sd = amp.std(axis=0, ddof=1) if len(downs) > 1 else np.zeros(3)
    return layer


def replay_error(layer: PhaseLayer, flight: Flight) -> dict[str, float]:
    """How far one (held-out) flight's transients are from the layer's replay
    of them, given that flight's own plateau levels: RMS in rev/s of the rotor
    mean (``*_common``) and of the per-rotor split (``*_split``) over each
    transient window; ``*_split_prop`` is the same with the split simply
    following the common curve (no hand-over curve, no fan-out), the form a
    plain per-rotor rescaling of one template would give.  The landing split
    is scored with the fan-out amplitude that fits this landing best, so it
    measures the time course, not the draw.
    """
    fs = layer.fs
    marks = landmarks(flight.rps, flight.fs)
    out: dict[str, float] = {}
    if marks is None:
        return out
    x = bridge_nans(np.asarray(flight.rps, dtype=np.float64))

    def rms(v: np.ndarray) -> float:
        return float(np.sqrt(np.mean(v * v)))

    if (
        marks.standby is not None
        and marks.standby_level is not None
        and marks.motors_on is not None
        and marks.takeoff is not None
    ):
        level = marks.standby_level
        sb = float(level.mean())
        t_su = layer.startup_common.size / fs
        if (marks.takeoff - marks.motors_on) / fs >= t_su:
            w = _window(x, marks.motors_on, 0.0, t_su, fs)
            c = w.mean(axis=0)
            out["startup_common"] = rms(c - sb * layer.startup_common)
            out["startup_split"] = rms(w - c - (level - sb)[:, None] * layer.startup_split[None, :])
            out["startup_split_prop"] = rms(w - c - (level - sb)[:, None] * (c / sb)[None, :])
        if marks.arrival is not None and marks.hover_in is not None:
            h0 = float(marks.hover_in.mean())
            w = _window(x, marks.arrival, 0.0, SETTLE_AFTER_ARRIVAL_S, fs)
            n_pre = int(round(ARRIVAL_PRE_S * fs))
            c = w.mean(axis=0)
            out["arrival_common"] = rms(c - sb - (h0 - sb) * layer.arrival_common[n_pre:])
            d_sb, d_cr = level - sb, marks.hover_in - h0
            a = layer.arrival_split[n_pre:]
            out["arrival_split"] = rms(w - c - d_sb[:, None] * (1.0 - a) - d_cr[:, None] * a)
            s = (c - sb) / (h0 - sb)
            out["arrival_split_prop"] = rms(w - c - d_sb[:, None] * (1.0 - s) - d_cr[:, None] * s)
    if (
        marks.landing is not None
        and marks.hover_out is not None
        and layer.landing_common is not None
        and layer.landing_fan is not None
    ):
        h1 = float(marks.hover_out.mean())
        n_ld = layer.landing_common.size
        w = _window(x, marks.landing, LANDING_PRE_S, n_ld / fs - LANDING_PRE_S, fs)[:, :n_ld]
        c = w.mean(axis=0)
        out["landing_common"] = rms(c - h1 * layer.landing_common)
        rest = w - c - (marks.hover_out - h1)[:, None] * layer.landing_common[None, :]
        out["landing_split_prop"] = rms(rest)
        fan = layer.landing_fan
        out["landing_split"] = rms(rest - np.outer(rest @ fan / float(fan @ fan), fan))
    return out


__all__ = ["ARRIVAL_PRE_S", "LANDING_PRE_S", "PhaseLayer", "fit_phase_layer", "replay_error"]
