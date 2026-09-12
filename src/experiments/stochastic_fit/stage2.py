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
#: Standby (Stage 3): every rotor spinning but below this rate. FLY125's
#: standby sits near 35-40 rev/s, so the two regimes do not overlap. A slower
#: shaft puts MORE orders under Nyquist - 7900 / 35 = 226 against 121 in cruise
#: - so standby needs a taller comb, not a shorter one.
STANDBY_MAX_RPS = 45.0
STANDBY_MIN_RPS = 20.0
STANDBY_K_CAP = 230
#: FLY125 holds exactly ONE contiguous standby run, 13.5 s starting at 1.78 s
#: (13.9 s of standby in the whole 178 s recording). So Stage 3 fits a single
#: window, not a population: its clips would otherwise be consecutive slices of
#: one event, which is not a sample of anything.
STANDBY_SECONDS = 12.5
REGIMES = {
    "cruise": dict(min_rps=CRUISE_MIN_RPS, max_rps=None, k_cap=130, stride_s=None),
    "standby": dict(
        min_rps=STANDBY_MIN_RPS, max_rps=STANDBY_MAX_RPS, k_cap=STANDBY_K_CAP, stride_s=1.0
    ),
}
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
    max_rps: float | None = None,
    max_clips: int = 12,
    stride_s: float | None = None,
) -> list[tuple[float, float]]:
    """``[(start_s, duration_s)]`` windows where every rotor stays in one regime.

    The selection reads the telemetry only. A window is kept when every rotor
    stays inside ``[min_rps, max_rps]`` for the whole window, so no window
    straddles a transition and one stationary model has a stationary target.
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
        ok = block.shape[-1] == n and float(np.nanmin(block)) >= min_rps
        if ok and max_rps is not None:
            ok = float(np.nanmax(block)) <= max_rps
        if ok:
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


#: Floor-only dynamics. Real cruise gusts: the broadband floor's level and tilt
#: drift over a few seconds, and a stationary model cannot produce that. The
#: LINE drift stays off, because that is the term which absorbed +4.97 dB of
#: static per-order level on the bench and broke the map to the renderer. A
#: floor GP cannot steal per-order line level, so this is the one dynamic the
#: identifiability argument permits without a reparameterisation.
FLOOR_DYNAMICS: dict[str, Any] = {
    "floor_gp_std_db": 2.0,
    "floor_gp_tau_s": 3.0,
    "floor_tilt_gp_std": 0.5,
    "floor_tilt_gp_tau_s": 6.0,
}


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
    regime: str = "cruise",
    floor_dynamics: bool = False,
    min_rps: float | None = None,
    device: str = "cpu",
    log: Any = print,
) -> dict[str, Any]:
    """Joint MAP over the cruise windows.

    ``rotor_delta=True`` gives each rotor its own profile offset on top of the
    shared shape, which is the point of a four-rotor rig: the rotors differ in
    level and in the microphone pattern, not in the physics.
    """
    band = REGIMES[regime]
    rows = cruise_clips(
        recording_id,
        seconds=seconds,
        max_clips=max_clips,
        min_rps=float(band["min_rps"] if min_rps is None else min_rps),
        max_rps=band["max_rps"],
        stride_s=band["stride_s"],
    )
    if not rows:
        raise ValueError(f"{recording_id}: no {regime} window of {seconds} s found")
    k_cap = k_cap if k_cap != K_CAP else int(band["k_cap"])
    staged: list[tuple[str, str, Periodogram, Any]] = []
    for cid, clip, pg in rows:
        m = int(clip.audio.shape[0] if n_mics is None else n_mics)
        variant = {**S2_VARIANT, **(FLOOR_DYNAMICS if floor_dynamics else {})}
        spec = make_spec(pg, n_mics=m, f_max=F_MAX, k_cap=k_cap, variant=variant)
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


def params_from_export(
    export: dict[str, Any],
    rates: np.ndarray,
    *,
    sample_rate: int = SR,
    n_mics: int = 8,
) -> Any:
    """Renderer parameters from one cruise clip's MAP export.

    The four-rotor, eight-microphone counterpart of
    :func:`stage1_bayes.params_from_export`. Every field is copied, not
    converted: the fit emits the renderer's own coordinates. The microphone
    structure is copied as FIXED vectors, so a synthetic clip carries the
    fitted rig's own channel pattern and cross-channel agreement can be
    measured the way the acceptance probe measures it on real clips.
    """
    from data_processing import stochastic_rotor_noise as srn

    profile = np.atleast_2d(np.asarray(export["profile_db"], dtype=np.float64))
    h = np.asarray(export.get("h_db", 0.0), dtype=np.float64)
    if h.ndim == 3:
        profile = profile + h.mean(axis=-1)[: profile.shape[0]]
    n_rotors = profile.shape[0]
    rates = np.atleast_1d(np.asarray(rates, dtype=np.float64))
    # The comb must reach Nyquist for the SLOWEST rotor, or the fastest rotor's
    # high orders are cut while the slowest keeps padding.
    k_max = max(2, int(np.floor((sample_rate / 2) / max(float(rates.min()), 1.0))))
    k_use = min(k_max, profile.shape[1])
    gamma0 = np.atleast_1d(np.asarray(export["gamma0"], dtype=np.float64))
    slope = np.atleast_1d(np.asarray(export["gamma_slope"], dtype=np.float64))

    def per_rotor(x: np.ndarray) -> np.ndarray:
        return np.resize(x, n_rotors).astype(np.float64)

    mic_gain = export.get("mic_gain_db")
    mic_gain_db = None
    if mic_gain is not None:
        g = np.atleast_2d(np.asarray(mic_gain, dtype=np.float64))
        if g.shape[0] >= n_mics and g.shape[1] >= n_rotors:
            mic_gain_db = g[:n_mics, :n_rotors].copy()
    mic_floor = export.get("mic_floor_db")
    mic_floor_db = (
        np.asarray(mic_floor, dtype=np.float64)[:n_mics].copy() if mic_floor is not None else None
    )
    gain_all = export.get("gain_all_db")
    gain_all_db = (
        np.asarray(gain_all, dtype=np.float64)[:n_mics].copy() if gain_all is not None else None
    )
    return srn.StochasticParams(
        sample_rate=int(sample_rate),
        n_rotors=n_rotors,
        n_harmonics=k_use,
        profile_db=profile[:n_rotors, :k_use].copy(),
        gamma0=per_rotor(gamma0),
        gamma_slope=per_rotor(slope),
        floor_ctrl_hz=np.asarray(export["floor_ctrl_hz"], dtype=np.float64),
        floor_ctrl_db=np.asarray(export["floor_shape_db"], dtype=np.float64),
        floor_tilt_db_oct=float(export.get("floor_tilt_db_oct", 0.0)),
        harm_mean_db=0.0,
        floor_mean_db=float(export.get("floor_mean_db", 0.0)),
        harm_gp_std_db=0.0,
        harm_gp_tau_s=1.0,
        harm_coherence=0.0,
        floor_gp_std_db=0.0,
        floor_gp_tau_s=1.0,
        floor_tilt_gp_std=0.0,
        floor_tilt_gp_tau_s=1.0,
        line_bin_integrate=True,
        floor_static_rel=float(export.get("floor_static_rel", 0.0)),
        amp_rps_exponent=float(export.get("amp_exp", 0.0)),
        amp_rps_exponent_floor=float(export.get("floor_exp", export.get("amp_exp", 0.0))),
        amp_rps_ref=80.0,
        shaft_jitter_rps=0.0,
        shaft_jitter_tau_s=2.0,
        phase_diffusion_hz_per_order=0.0,
        shaft_offset_rps=0.0,
        umod_std_db=0.0,
        umod_tau_s=1.0,
        umod_corner_hz=200.0,
        mic_gain_all_db=0.0,
        mic_floor_std_db=0.0,
        fixed_mic_gain_db=mic_gain_db,
        fixed_mic_floor_db=mic_floor_db,
        fixed_mic_gain_all_db=gain_all_db,
        gamma_min_bins=0.01,
        coherence_k_half=float(export.get("coherence_k_half", 0.0)),
    )


def render_from_export(
    export: dict[str, Any],
    rps: np.ndarray,
    *,
    sample_rate_native: int = native.NATIVE_SR,
    n_mics: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """``(M, T)`` synthetic cruise audio at 16 kHz from a MAP export.

    Rendered at the native rate and decimated on the real clips' own path, so
    the synthetic clip carries no band edge the real clips do not have.
    """
    from data_processing import stochastic_rotor_noise as srn

    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    rates = rps.mean(axis=1)
    params = params_from_export(export, rates, sample_rate=sample_rate_native, n_mics=n_mics)
    # the trajectory is resampled to the native grid the renderer works on
    n_native = int(round(rps.shape[-1] / SR * sample_rate_native))
    t_src = np.linspace(0.0, 1.0, rps.shape[-1])
    t_dst = np.linspace(0.0, 1.0, n_native)
    rps_native = np.stack([np.interp(t_dst, t_src, r) for r in rps])
    audio, _ = srn.synthesize(
        params,
        rps_native,
        rng=np.random.default_rng(seed),
        n_mics=n_mics,
        line_mode="fm",
        n_fft=1 << 16,
    )
    clip = Clip(
        "synthetic", "synthetic", np.asarray(audio, np.float32), rps_native, sample_rate_native
    )
    return np.asarray(native.decimate(clip, SR).audio, dtype=np.float64)
