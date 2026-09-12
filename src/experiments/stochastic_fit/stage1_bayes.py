"""S1 as a Bayesian fit of the bench spectrum — the project's own machinery.

The first S1 attempt (``stage1.py``, retired with the frames cutover) fitted
DERIVED SUMMARIES: per-order band-integrated excess over a local floor,
thresholded at 6 dB of peak margin, then regressions on those summaries, then
an iterative calibration of the renderer against the estimator. Every step
needed a patch — median-to-mean debiasing, two selection-bias corrections, a
Tobit regression for the thresholded orders, and finally an affine calibration
that could not represent a non-monotone residual. None of that is a fit.

The right object was already in the repository. For one stationary rotor the
periodogram bins are independent with mean equal to the model spectrum, so the
exact log-likelihood is the Whittle form ``sum I/M + log M`` that
:meth:`model.CombSpectrum.whittle` computes, with
``M(f) = floor(f) + sum_k A_k L(f - k s; gamma_k)`` — the renderer's own
spectrum in the renderer's own coordinates. :func:`rig.fit_rig` does joint MAP
over clips with rig-level parameters tied. That removes, at once:

* the detection threshold and its censoring — a quiet line contributes little
  likelihood and keeps a wide posterior instead of vanishing from the data;
* the median-versus-mean debiasing — the likelihood is the noise model;
* the calibration loop — the fitted parameters ARE renderer parameters, so
  there is no convention to map;
* the point estimate for unmeasured orders — they fall to their prior, which is
  the correct generative statement.

The bench is the simplest case this machinery handles: one rotor, one
microphone, a constant rate, no trajectory, no per-mic terms.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import rig
from experiments.stochastic_fit.data import Clip, Periodogram, periodogram
from experiments.stochastic_fit.model import AMP_RPS_REF, BASE_VARIANT

#: The bench cells. Setpoints are NOT rev/s — the rate is measured per cell.
MOTORS = (1, 2, 3, 4)
SPEEDS = (50, 60, 70, 80, 90)
FIT_MOTORS = (1, 2, 3)
HELD_OUT_MOTORS = (4,)

#: Channel with the largest comb margin.
CHANNEL = 7
SR = 16000
#: The published frames dataset that carries the single-motor recordings, as
#: ``split="motor"`` samples with ids ``motor_Motor1_80``. They have no
#: telemetry, so the rate is a fitted parameter seeded by comb evidence.
BENCH_DATASET = "DREGON-frames"
#: 9 s is what the tightest recording offers: the rotors spin up between 1.25
#: and 5.38 s and stop between 14.7 and 17.2 s, so a fixed 16 s window from
#: 3 s held up to 34 % silence, which depresses the measured floor and makes
#: the span non-stationary.
BENCH_SECONDS = 9.0

#: The bench is stationary, so a long window costs nothing in time resolution
#: and buys frequency resolution: 16384 points at 16 kHz is a 0.98 Hz bin, and
#: the model's ``line_bin_integrate`` + window kernel handle lines narrower
#: than that correctly rather than pretending to resolve them.
N_FFT = 16384
# NON-overlapping frames. The Whittle likelihood treats every cell as an
# independent draw, and 50 % overlapped frames are not independent -- the same
# samples enter two frames, so an overlapped analysis double counts the data and
# scores correlated cells as if they were fresh. It also doubles the cost and
# the memory of every gradient step, which is what let a five-clip fit be killed
# twice. One frame per 16384 samples at 16 kHz is 1.024 s, and a 9 s bench span
# still gives eight independent frames per cell.
HOP = N_FFT
F_MAX = 7900.0
K_CAP = 200
#: OLA window for rendering at 44.1 kHz: 0.67 Hz bins.
OLA_N_FFT = 1 << 16

#: The variant the corrected forward model uses everywhere else, minus the
#: pieces a clamped bench motor cannot have: no rate trajectory, so no chirp
#: width, and the telemetry-offset latent is meaningless when the rate is
#: itself a fitted parameter of a stationary clip.
BENCH_VARIANT: dict[str, Any] = {
    **BASE_VARIANT,
    "chirp_width": False,
    "rps_offset": False,
    "fit_speed_law": True,
    # Fit the line shape the RENDERER draws, not an idealisation of it:
    # ``build_psd`` renders each line as a Lorentzian truncated to a
    # power-of-two bucket of bins and renormalised by the fixed 87.4 % factor.
    # Fitting a Gaussian of equal HWHM instead leaves a few dB of band-
    # integrated power on the table, which is what the k 2-15 residual showed.
    "line_shape": "lorentz_bucket",
    # A clamped motor at a fixed setpoint has no slow amplitude drift, so the
    # drift processes are pinned OFF. Left free they absorb static structure the
    # profile should own: on Motor1_80 the fitted ``h_db`` carried a static
    # +4.97 dB mean over orders 19-64 (up to +17.96 dB), which is a
    # profile-versus-drift identifiability leak, not a measurement.
    "gp_std_db": 0.0,
    "floor_gp_std_db": 0.0,
    "floor_tilt_gp_std": 0.0,
    "umod_std_db": 0.0,
    # A bench line is a MIXTURE: a coherent needle carrying the window's own
    # power response |W(f - f0)|^2, plus a Rayleigh pedestal of the fitted
    # width, with needle share w_k = exp(-(k/k_half)^2). Measured on Motor1_80,
    # the fitted w_k matches the share that the equivalent width and the
    # centre-bin share independently imply, and the variant is worth about
    # 1030 nats over a single-component line (-1.1462 against -1.1404 per cell).
    "fit_coherence": True,
    "needle_window_shape": True,
}


def bench_recording_id(motor: int, setpoint: int) -> str:
    """The published frames id of one bench cell."""
    return f"motor_Motor{int(motor)}_{int(setpoint)}"


def bench_span(audio: np.ndarray, sr: float) -> tuple[float, float]:
    """``(start_s, duration_s)`` of the steady span of one bench recording.

    The rotors spin up between 1.25 s and 5.38 s and stop between 14.7 s and
    17.2 s depending on the cell, so no fixed window works: a 16 s window from
    3 s held up to 34 % silence. Onset and offset are the first and last times
    the 50 ms envelope clears half the 95th percentile; the window starts 1 s
    after onset and ends 0.5 s before offset.
    """
    y = np.asarray(audio, dtype=np.float64)
    n = int(0.05 * sr)
    env = np.sqrt(np.convolve(y**2, np.ones(n) / n, mode="same"))
    on = np.flatnonzero(env > 0.5 * np.percentile(env, 95))
    start = float(on[0]) / sr + 1.0
    end = float(on[-1]) / sr - 0.5
    return start, min(BENCH_SECONDS, max(end - start, 1.0))


def _rate_seed(clip: Clip) -> float:
    """A starting rate for the fit, from the proper-prior comb evidence.

    The fit refines the rate itself; this only has to land in the right octave.
    Read at 16 kHz because the same estimator at 44.1 kHz returns half the rate
    on three of the four motors.
    """
    from experiments.stochastic_fit import bench as bench_mod

    x = np.asarray(clip.audio[0], dtype=np.float64)
    n = 1 << 16
    psd = bench_mod._welch(x - x.mean(), n)  # noqa: SLF001 - the bench instrument's own
    return float(bench_mod.estimate_rate(psd, clip.sr / n))


def _cache_path(
    dataset: str, version: str | None, motor: int, setpoint: int, channels: tuple[int, ...] | None
) -> Path:
    stamp = "all" if channels is None else "-".join(str(c) for c in channels)
    return C.CACHE_DIR / (
        f"bench_{dataset}_{(version or 'pinned')[:12]}_Motor{motor}_{setpoint}_ch{stamp}_16k.npz"
    )


def bench_cells(
    motors: tuple[int, ...] = FIT_MOTORS,
    speeds: tuple[int, ...] = SPEEDS,
    *,
    dataset: str = BENCH_DATASET,
    version: str | None = None,
    channels: str | tuple[int, ...] | None = (CHANNEL,),
) -> dict[tuple[int, int], Clip]:
    """``{(motor, setpoint): clip_16k}`` — the steady span of each cell.

    Every missing cell is read in ONE pass over the frames dataset (272
    samples, of which 262 are bench cells), because a pass per cell would
    re-scan the shards twenty times. Cut cells are cached at 16 kHz, so a
    second run touches no dataset at all.
    """
    dataset, pinned = C.split_dataset(dataset)
    version = version or pinned
    spec = C.channel_spec(channels)
    want = [(int(m), int(s)) for m in motors for s in speeds]
    out: dict[tuple[int, int], Clip] = {}
    missing: dict[str, tuple[int, int]] = {}
    for motor, speed in want:
        path = _cache_path(dataset, version, motor, speed, spec)
        cached = C.read_cached_clip(path, f"Motor{motor}_{speed}") if path.exists() else None
        if cached is not None:
            out[(motor, speed)] = cached
        else:
            missing[bench_recording_id(motor, speed)] = (motor, speed)
    if missing:
        for rec in C.iter_recordings(dataset, tuple(missing), version, rps_key="auto"):
            motor, speed = missing[rec.recording_id]
            chans = C.resolve_channels(spec, rec.n_channels)
            start_s, duration_s = bench_span(rec.audio[chans[0]], rec.sr)
            clip = C.decimate(
                rec.cut(
                    rec.t_start + start_s,
                    duration_s,
                    channels=chans,
                    clip_id=f"Motor{motor}_{speed}",
                ),
                SR,
            )
            rate = _rate_seed(clip)
            clip.rps = np.full((1, clip.audio.shape[1]), rate)
            clip.rps_original = clip.rps
            clip.meta.update(motor=motor, setpoint=speed, rate_seed=rate, span_start_s=start_s)
            C.write_cached_clip(_cache_path(dataset, version, motor, speed, spec), clip)
            out[(motor, speed)] = clip
    return {cell: out[cell] for cell in want}


def bench_clip(
    motor: int,
    setpoint: int,
    *,
    dataset: str = BENCH_DATASET,
    version: str | None = None,
    channels: str | tuple[int, ...] | None = (CHANNEL,),
) -> Clip:
    """One bench cell: native frames audio, steady span, decimated to 16 kHz."""
    return bench_cells(
        (int(motor),), (int(setpoint),), dataset=dataset, version=version, channels=channels
    )[(int(motor), int(setpoint))]


def bench_periodogram(motor: int, setpoint: int, **kwargs: Any) -> tuple[Clip, Periodogram]:
    clip = bench_clip(motor, setpoint, **kwargs)
    return clip, periodogram(clip, n_fft=N_FFT, hop=HOP)


def bench_clips(
    motors: tuple[int, ...] = FIT_MOTORS,
    speeds: tuple[int, ...] = SPEEDS,
    *,
    dataset: str = BENCH_DATASET,
    version: str | None = None,
    channels: str | tuple[int, ...] | None = (CHANNEL,),
    n_harm: int | None = None,
) -> list[tuple[str, str, Periodogram, Any]]:
    """The ``fit_rig`` input list for the chosen cells.

    The fitting entry point is :func:`campaign.fit` with ``regime="bench"``;
    this is the same staging, kept here for the routines that re-stage a
    SUBSET of cells under an already fitted rig (:func:`reconstruct`,
    :func:`held_out_nll`), which must inherit that rig's ladder.
    """
    cells = bench_cells(motors, speeds, dataset=dataset, version=version, channels=channels)
    rows = [
        (clip.clip_id, f"motor{motor}", clip, periodogram(clip, n_fft=N_FFT, hop=HOP))
        for (motor, _speed), clip in cells.items()
    ]
    return rig.stage_clips(rows, variant=BENCH_VARIANT, f_max=F_MAX, k_cap=K_CAP, n_harm=n_harm)


def reconstruct(
    summary: dict[str, Any],
    motor: int,
    setpoint: int,
    *,
    iters: tuple[int, int, int] = (15, 15, 30),
    device: str = "cpu",
) -> tuple[Any, Any]:
    """``(clip_model, periodogram)`` for one cell under a fitted rig.

    The rig-level parameters are frozen and only this clip's nuisances are
    fitted, exactly as :func:`rig.fit_heldout` does. A subset must inherit the
    ladder the rig was fitted under, or the tied vectors no longer line up.
    """
    import torch

    state = {
        k: torch.as_tensor(np.asarray(v, dtype=np.float32)) for k, v in summary["rig_state"].items()
    }
    staged = bench_clips(motors=(motor,), speeds=(setpoint,), n_harm=int(summary["spec"]["n_harm"]))
    rig_params = rig.RigParams(
        staged[0][3], rig.RigSpec(**summary["rig_spec"]), staged[0][2].rps.shape[0], device
    )
    rig_params.load_state_dict(state)
    for p in rig_params.parameters():
        p.requires_grad_(False)
    rcs = rig._prepare(clips, rig_params, device)
    with torch.no_grad():
        for rc in rcs:
            rc.model.level_db.copy_(
                (rc.model.profile_db - rig_params.profile_db[None, :]).median(dim=1).values
            )
    rig._stages(
        rig_params,
        rcs,
        rig_free=False,
        ladder=(16, 48),
        iters=iters,
        lr=0.1,
        log=lambda *_: None,
        t0=0.0,
    )
    return rcs[0].model, staged[0][2]


def sample_from_fit(clip_model: Any, *, seed: int = 0, mic: int = 0) -> np.ndarray:
    """A waveform drawn from the FITTED spectrum itself.

    The fit's ``forward()`` is the expected periodogram on its own STFT grid, so
    a draw from the model is white noise filtered through ``sqrt(M)`` frame by
    frame — one call to the renderer's own overlap-add filter. Nothing is
    mapped, converted or calibrated: this is the fitted model's own sample, and
    it is the object to look at and listen to when asking whether the fit is
    good.

    The line magnitudes come out Rayleigh (the ``stochastic`` line statistics).
    Whether a real rotor line is that noisy or more coherent is a separate
    question about the line process, not about the spectrum.
    """
    import torch

    from data_processing.stochastic_rotor_noise import _ola_filter

    with torch.no_grad():
        M = clip_model().detach().cpu().numpy()  # (mics, frames, freqs)
    gain = np.sqrt(np.maximum(M[mic : mic + 1], 0.0))
    n_fft = int(clip_model.spec.extra.get("n_fft", N_FFT)) if clip_model.spec.extra else N_FFT
    hop = HOP
    n_samples = (gain.shape[1] - 1) * hop + n_fft
    rng = np.random.default_rng(seed)
    white = rng.standard_normal((1, n_samples))
    out = _ola_filter(white, gain, n_fft, hop)
    return np.asarray(out[0], dtype=np.float64)


def params_from_export(export: dict[str, Any], rate_rps: float, *, sample_rate: int = SR) -> Any:
    """Renderer parameters from one clip's MAP export.

    No conversion and no calibration: :meth:`model.CombSpectrum.export` already
    emits the renderer's own coordinates, which is the whole point of fitting
    the forward spectrum instead of a derived summary. Dynamics are left at
    zero — a clamped bench motor has no drift to render, and the fit's ``h_db``
    for a stationary clip is prior-scale noise.
    """
    from data_processing import stochastic_rotor_noise as srn

    profile = np.asarray(export["profile_db"], dtype=np.float64)
    if profile.ndim == 1:
        profile = profile[None, :]
    # The static line level is the profile PLUS the time mean of the drift: a
    # free ``h_db`` does not come out zero-mean, and dropping it cost 4-10.5 dB
    # of rendered line power over orders 19-64. With the drift pinned off this
    # term is zero and the line is a no-op.
    h = np.asarray(export.get("h_db", 0.0), dtype=np.float64)
    if h.ndim == 3:
        profile = profile + h.mean(axis=-1)[: profile.shape[0]]
    k_max = max(2, int(np.floor((sample_rate / 2) / max(rate_rps, 1.0))))
    k_use = min(k_max, profile.shape[1])
    gamma0 = np.atleast_1d(np.asarray(export["gamma0"], dtype=np.float64))
    slope = np.atleast_1d(np.asarray(export["gamma_slope"], dtype=np.float64))
    return srn.StochasticParams(
        sample_rate=int(sample_rate),
        n_rotors=1,
        n_harmonics=k_use,
        profile_db=profile[:1, :k_use].copy(),
        gamma0=gamma0[:1].copy(),
        gamma_slope=slope[:1].copy(),
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
        amp_rps_ref=AMP_RPS_REF,  # the fit's own reference; both must agree
        shaft_jitter_rps=float(slope[0]) / 1.177,
        shaft_jitter_tau_s=2.0,
        phase_diffusion_hz_per_order=0.0,
        shaft_offset_rps=0.0,
        umod_std_db=0.0,
        umod_tau_s=1.0,
        umod_corner_hz=200.0,
        mic_gain_all_db=0.0,
        mic_floor_std_db=0.0,
        # Render under the same width floor the fit used, or every line comes
        # out at least 12.9 Hz wide at 44.1 kHz against 0.2-2 Hz measured.
        gamma_min_bins=0.01,
        # The fitted coherence transition. The renderer puts the coherent share
        # w_k = exp(-(k/k_half)^2) of each order through the tone bank and the
        # rest through narrowband noise, which is the generative form of the
        # needle-plus-pedestal line the fit identified.
        coherence_k_half=float(export.get("coherence_k_half", 0.0)),
    )


def render_from_export(
    export: dict[str, Any], rate_rps: float, *, seconds: float, seed: int = 0
) -> np.ndarray:
    """One synthetic clip from a MAP export, on the real clips' own path.

    Rendered at 44.1 kHz and decimated by ``clips.decimate`` so the synthetic
    clip carries the same resampler transition band the real one does.
    """
    from data_processing import stochastic_rotor_noise as srn

    params = params_from_export(export, rate_rps, sample_rate=C.NATIVE_SR)
    n = int(round(seconds * C.NATIVE_SR))
    rps = np.full((1, n), float(rate_rps))
    audio, _ = srn.synthesize(
        params, rps, rng=np.random.default_rng(seed), n_mics=1, line_mode="fm"
    )
    clip = Clip(
        "synthetic",
        "synthetic",
        np.asarray(audio, dtype=np.float32),
        rps,
        C.NATIVE_SR,
        None,
        {"synthetic": True},
    )
    return np.asarray(C.decimate(clip, SR).audio[0], dtype=np.float64)


def held_out_nll(
    summary: dict[str, Any], motors: tuple[int, ...] = HELD_OUT_MOTORS, **kwargs: Any
) -> dict[str, Any]:
    """Score the fitted rig on cells it never saw (``rig.fit_heldout``)."""
    staged = bench_clips(motors, SPEEDS, n_harm=int(summary["spec"]["n_harm"]))
    return rig.fit_heldout(summary, staged, **kwargs)


def save(summary: dict[str, Any], path: str | Path) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)

    def plain(obj: Any) -> Any:
        if hasattr(obj, "detach"):  # torch tensors in rig_state
            return plain(obj.detach().cpu().numpy())
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: plain(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [plain(v) for v in obj]
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        return obj

    p.write_text(json.dumps(plain(summary), indent=1))
    return p


__all__ = [
    "BENCH_VARIANT",
    "MOTORS",
    "BENCH_DATASET",
    "bench_cells",
    "bench_clip",
    "bench_clips",
    "bench_periodogram",
    "fit",
    "held_out_nll",
    "save",
]
