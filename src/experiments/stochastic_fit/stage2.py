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

Native 44.1 kHz audio is the only source: every clip comes out of the published
frames datasets (`experiments.stochastic_fit.clips`) and is decimated here, so
the 16 kHz training sets' brick wall at 7.9 kHz never enters, and the rotor
track the fit holds fixed is the published refined label.

The fitting entry point is `campaign.fit` (regimes ``cruise``/``standby``),
driven by ``scripts/stochastic_fit.py``; this module owns the flight window
selection, the flight variant and the map to the renderer's parameters.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.stochastic_fit import clips
from experiments.stochastic_fit.data import Clip, Periodogram, periodogram
from experiments.stochastic_fit.model import BASE_VARIANT
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
#: The published frames datasets the flight fit reads.
FIT_DATASET = "michaels-frames"
DREGON_DATASET = "DREGON-frames"

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


def cruise_windows(
    recording_id: str = FIT_RECORDING,
    *,
    dataset: str = FIT_DATASET,
    version: str | None = None,
    rps_key: str = clips.DEFAULT_RPS_KEY,
    seconds: float = CRUISE_SECONDS,
    min_rps: float = CRUISE_MIN_RPS,
    max_rps: float | None = None,
    max_clips: int = 12,
    stride_s: float | None = None,
) -> list[tuple[float, float]]:
    """``[(start_s, duration_s)]`` windows where every rotor stays in one regime.

    The selection reads the rotor-speed LABEL only (``rps_key``, the refined
    one by default). A window is kept when every rotor stays inside
    ``[min_rps, max_rps]`` for the whole window, so no window straddles a
    transition and one stationary model has a stationary target.
    """
    rec = clips.load_recording(dataset, recording_id, version, rps_key)
    return clips.windows(
        rec,
        seconds=seconds,
        max_clips=max_clips,
        min_rps=min_rps,
        max_rps=max_rps,
        stride_s=stride_s,
    )


def cruise_clips(
    recording_id: str = FIT_RECORDING,
    *,
    dataset: str = FIT_DATASET,
    version: str | None = None,
    rps_key: str = clips.DEFAULT_RPS_KEY,
    channels: str | tuple[int, ...] | None = None,
    tag: str = "cruise",
    **kwargs: Any,
) -> list[tuple[str, Clip, Periodogram]]:
    """``[(clip_id, clip_16k, periodogram)]`` for the windows of one recording.

    Native 44.1 kHz audio out of the published frames, decimated here: the
    published 16 kHz training sets carry an 88-90 dB brick wall at 7.9 kHz,
    which a fit would read as structure. ``tag`` names the regime in the clip
    id, so a rig fit over several recordings and regimes has unique ids.
    """
    rows: list[tuple[str, Clip, Periodogram]] = []
    found = cruise_windows(
        recording_id, dataset=dataset, version=version, rps_key=rps_key, **kwargs
    )
    for i, (start_s, dur) in enumerate(found):
        cid = f"{recording_id.lower()}_{tag}_{i:02d}"
        clip = clips.decimate(
            clips.load_clip(
                dataset,
                recording_id,
                start_s,
                dur,
                version=version,
                channels=channels,
                rps_key=rps_key,
                clip_id=cid,
            ),
            SR,
        )
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
    sample_rate_native: int = clips.NATIVE_SR,
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
    return np.asarray(clips.decimate(clip, SR).audio, dtype=np.float64)
