"""Stage 2: Michael's FLY125 in CRUISE, fitted as coherent tones + stochastic floor.

Stage 1 established the line model on the DREGON bench: one power per order,
split by a coherent fraction ``w_k = exp(-(k/k_half)^2)`` between a coherent
needle carrying the analysis window's own power response ``|W(f-f0)|^2`` and a
Rayleigh pedestal of the fitted width. That model beat a single-component line
by about 1030 nats for one parameter, and its fitted ``w_k`` reproduced the
coherent share that two independent descriptive statistics imply.

Stage 2 applies the same model to real flight. Four differences from the bench,
each forced by the data rather than chosen:

* **Four rotors, not one.** Every rotor gets its own carrier from the telemetry,
  its own profile offset and its own per-microphone gains.
* **Eight microphones.** The acceptance probe reads cross-channel agreement, so
  the fit must own the microphone structure: per-(mic, rotor) line gains, a
  per-mic floor, and one per-mic gain on everything (Michael's rig).
* **The speed moves.** ``chirp_width`` adds the width a line acquires because
  the rotor sweeps ``k * ds`` hertz across one analysis window. It is computed
  from the telemetry with NO free parameter, so a moving shaft cannot be
  mistaken for a wide line.
* **Cruise only.** Windows are selected where all four rotors are above
  ``CRUISE_MIN_RPS``, so one stationary model has a stationary target.

Native 44.1 kHz audio is the only source. Every clip is decimated by
``native.decimate``, so the published 16 kHz brick wall at 7.9 kHz never enters.

Run a fit with ``scripts/_stage2_fit.py``.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from experiments.stochastic_fit import native, rig
from experiments.stochastic_fit.data import Clip, Periodogram, periodogram
from experiments.stochastic_fit.model import BASE_VARIANT, make_spec
from experiments.stochastic_fit.stage1_bayes import HOP, N_FFT

SR = 16000
F_MAX = 7900.0
#: Cruise is defined by the telemetry, not by the clip list: every rotor above
#: this rate for the whole window. FLY125's four rotors sit near 70-95 rev/s in
#: cruise and near 35-40 rev/s in standby, so the two regimes are far apart.
CRUISE_MIN_RPS = 65.0
CRUISE_SECONDS = 16.0
#: 7900 Hz / 65 rev/s = 121 orders at the slowest cruise rate.
K_CAP = 130
FIT_RECORDING = "FLY125"
#: Held out, never fitted and never used to choose a threshold.
HELD_OUT_RECORDING = "FLY124"

S2_VARIANT: dict[str, Any] = {
    **BASE_VARIANT,
    # the line model Stage 1 selected
    "fit_coherence": True,
    "needle_window_shape": True,
    "line_shape": "lorentz_bucket",
    # the rotor speed moves in flight, and the sweep across one window is known
    # from the telemetry, so it enters as a covariate with no free parameter
    "chirp_width": True,
    "fit_speed_law": True,
    # microphone structure: the probe reads cross-channel agreement
    "mic_floor": True,
    "gain_all": True,
    # Dynamics stay OFF for the first fit. On stationary bench data a free
    # drift term absorbed static per-order level (+4.97 dB mean over orders
    # 19-64), which breaks the map from the fitted vector to the renderer.
    # Cruise windows are selected to be stationary, so the static profile is
    # the right owner of the level.
    "gp_std_db": 0.0,
    "floor_gp_std_db": 0.0,
    "floor_tilt_gp_std": 0.0,
    "umod_std_db": 0.0,
}


def _recording(recording_id: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    return native._load_recording(recording_id)  # noqa: SLF001 - the only loader


def cruise_windows(
    recording_id: str = FIT_RECORDING,
    *,
    seconds: float = CRUISE_SECONDS,
    min_rps: float = CRUISE_MIN_RPS,
    max_clips: int = 12,
    stride_s: float | None = None,
) -> list[tuple[float, float]]:
    """``[(start_s, duration_s)]`` windows where every rotor stays in cruise.

    The selection reads the telemetry only. A window is kept when the minimum
    over rotors and over time of the rotor rate is above ``min_rps``, so no
    window straddles a transition.
    """
    _, audio_t, rps, sr = _recording(recording_id)
    step = float(seconds if stride_s is None else stride_s)
    t0 = float(audio_t[0])
    total = float(audio_t[-1] - audio_t[0])
    out: list[tuple[float, float]] = []
    n = int(round(seconds * sr))
    start = 0.0
    while start + seconds <= total and len(out) < max_clips:
        i0 = int(np.searchsorted(audio_t, t0 + start))
        block = rps[:, i0 : i0 + n]
        if block.shape[-1] == n and float(np.nanmin(block)) >= min_rps:
            out.append((t0 + start, seconds))
        start += step
    return out


def cruise_clips(
    recording_id: str = FIT_RECORDING, **kwargs: Any
) -> list[tuple[str, Clip, Periodogram]]:
    """``[(clip_id, clip_16k, periodogram)]`` for the cruise windows."""
    rows: list[tuple[str, Clip, Periodogram]] = []
    for i, (start_s, dur) in enumerate(cruise_windows(recording_id, **kwargs)):
        cid = f"{recording_id.lower()}_cruise_{i:02d}"
        clip = native.decimate(native.load_native_clip(recording_id, start_s, dur, clip_id=cid), SR)
        rows.append((cid, clip, periodogram(clip, n_fft=N_FFT, hop=HOP)))
    return rows


def fit(
    recording_id: str = FIT_RECORDING,
    *,
    max_clips: int = 8,
    seconds: float = CRUISE_SECONDS,
    k_cap: int = K_CAP,
    n_mics: int | None = None,
    iters: tuple[int, int, int] = (120, 120, 300),
    ladder: tuple[int, ...] = (16, 48),
    rotor_delta: bool = True,
    device: str = "cpu",
    log: Any = print,
) -> dict[str, Any]:
    """Joint MAP over the cruise windows.

    ``rotor_delta=True`` gives each rotor its own profile offset on top of the
    shared shape, which is the point of a four-rotor rig: the rotors differ in
    level and in the microphone pattern, not in the physics.
    """
    rows = cruise_clips(recording_id, seconds=seconds, max_clips=max_clips)
    if not rows:
        raise ValueError(f"{recording_id}: no cruise window of {seconds} s found")
    staged: list[tuple[str, str, Periodogram, Any]] = []
    for cid, clip, pg in rows:
        m = int(clip.audio.shape[0] if n_mics is None else n_mics)
        spec = make_spec(pg, n_mics=m, f_max=F_MAX, k_cap=k_cap, variant=S2_VARIANT)
        staged.append((cid, clip.group, pg, spec))
        log(
            f"  {cid}: {clip.audio.shape[0]} mics, {pg.power.shape[1]} frames, "
            f"rotors {np.round(np.asarray(pg.rps).mean(axis=1), 1).tolist()} rev/s"
        )
    k = max(int(s.n_harm) for _, _, _, s in staged)
    staged = [(cid, grp, pg, replace(s, n_harm=k)) for cid, grp, pg, s in staged]
    rig_spec = rig.RigSpec(rotor_delta=rotor_delta)
    return rig.fit_rig(staged, rig_spec, device=device, ladder=ladder, iters=iters, lr=0.1, log=log)


def save(summary: dict[str, Any], path: str | Path) -> Path:
    """Write a fit summary, with arrays converted to lists."""

    def plain(x: Any) -> Any:
        if isinstance(x, np.ndarray):
            return x.tolist()
        if isinstance(x, dict):
            return {k: plain(v) for k, v in x.items()}
        if isinstance(x, (list, tuple)):
            return [plain(v) for v in x]
        if isinstance(x, (np.floating, np.integer)):
            return x.item()
        return x

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    keep = {k: v for k, v in summary.items() if k != "rig_state"}
    p.write_text(json.dumps(plain(keep), indent=1))
    return p
