"""Render one clip from a MAP fit of *all* the model's parameters.

A draw from the rig population shares only the rotor-speed trajectory with the
real clip beside it: every amplitude, width, level and floor detail is
resampled from the population, which is what training data needs and what a
"can the model represent THIS recording?" question does not.

This module answers that question honestly. It does **not** decompose the clip
and hand the pieces back (a Vold-Kalman decomposition would do that, and a
decomposition is not a generative model). It maximises the posterior of the
model of Part I over every one of its parameters for this one clip —

    theta_hat = argmax  log p(I | theta) + log p(theta)

with ``log p(I | theta)`` the same Whittle likelihood §4.1 fits and
``log p(theta)`` the same priors: the amplitude drift is still a GP with the
fitted timescale, the floor level and tilt are still GPs, the carrier
correction still has its 0.3 rev/s prior, the widths still follow the affine
law. Nothing is free-form, nothing is integrated away, and the fitted
amplitudes are whatever the *model's own parameterisation* can produce.

The render is then a sample from that fitted model: the Whittle model is a
locally stationary Gaussian process with expected periodogram
``M_m(f, t) = model.forward()``, so a draw is an overlap-add of frames whose
magnitude spectrum is ``sqrt(M)`` and whose phases are fresh. That is exactly
what the fitted model says the clip is, with no waveform detail copied.

Reading the result:

* the conditional render sounds and looks close to the real clip -> the model
  *can* represent it, and every remaining discrepancy in the population arm is
  a fitting problem;
* it still differs -> the gap is structural (the comb model, the phase model,
  the floor model), and no amount of better population fitting will close it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from experiments.stochastic_fit.data import Clip, Periodogram, periodogram
from experiments.stochastic_fit.fit import fit_clip
from experiments.stochastic_fit.model import make_spec


@dataclass
class ConditionalFit:
    """A per-clip MAP fit and a draw from it."""

    clip_id: str
    audio: np.ndarray  # (M, T) the draw
    spectrum: np.ndarray  # (M, N, F) the fitted expected periodogram
    power: np.ndarray  # (M, N, F) the clip's own periodogram
    nll_fit: float  # Whittle nats per cell of the MAP fit
    nll_loo: float  # the leave-one-out smoother's, i.e. the achievable floor
    params: dict[str, Any]  # every fitted latent
    n_harm: int


def fit_conditional(
    clip: Clip,
    *,
    variant: dict[str, Any] | None = None,
    f_max: float | None = 8000.0,
    k_cap: int = 200,
    device: str = "cpu",
    iters: tuple[int, int, int, int] = (150, 150, 300, 60),
    log: Any = print,
) -> tuple[Periodogram, dict[str, Any]]:
    """MAP-fit every parameter of the model to one clip.

    ``variant`` selects the model structure exactly as the fitting CLI does
    (e.g. ``dict(line_shape="gauss", rps_offset=True, mic_floor=True)`` for
    DREGON's M5); the default is the plain Gaussian-line model.
    """
    pg = periodogram(clip)
    spec = make_spec(
        pg,
        n_mics=clip.audio.shape[0],
        f_max=f_max,
        k_cap=k_cap,
        variant=variant or {"line_shape": "gauss"},
    )
    res = fit_clip(pg, spec, device=device, iters=iters, log=log)
    return pg, res


def sample_from_spectrum(
    spectrum: np.ndarray,
    n_fft: int,
    hop: int,
    n_samples: int,
    *,
    seed: int = 0,
) -> np.ndarray:
    """Draw ``(M, T)`` audio whose measured periodogram is ``spectrum``.

    The Whittle likelihood treats each periodogram ordinate as an independent
    exponential with mean ``M(f, t)``; the matching time-domain object is a
    locally stationary Gaussian process, and drawing one is white noise shaped
    frame by frame. That is exactly what the renderer's floor path does, so it
    is done with the same function (``_ola_filter``) rather than a second
    implementation of overlap-add.

    Two normalisations have to be respected and neither is guessable, so the
    constant is MEASURED once per (n_fft, hop) by shaping white noise with a
    flat target and reading the result back through the very periodogram the
    fit used: ``data.periodogram`` divides by ``sum(w^2)``, and a Hann
    analysis/synthesis pair at a quarter-window hop reconstructs unit gain
    only for a *deterministic* filter, not for a fresh noise draw.
    """
    from data_processing.stochastic_rotor_noise import _ola_filter

    n_mic, n_frames, _ = spectrum.shape
    rng = np.random.default_rng(seed)
    need = (n_frames - 1) * hop + n_fft
    gain = np.sqrt(np.maximum(spectrum, 0.0)) * _ola_calibration(n_fft, hop)
    white = rng.standard_normal((n_mic, need))
    out = _ola_filter(white, gain, n_fft, hop)
    return out[:, :n_samples].astype(np.float32)


_OLA_CAL: dict[tuple[int, int], float] = {}


def _ola_calibration(n_fft: int, hop: int, n_frames: int = 64, seed: int = 12345) -> float:
    """Gain that makes ``_ola_filter`` of white noise land on its target PSD.

    Shapes white noise to a flat unit target, measures the result with
    :func:`data.periodogram`'s normalisation, and returns the correction. Cached
    per geometry; the value is a property of the window pair, not of the data.
    """
    key = (int(n_fft), int(hop))
    if key in _OLA_CAL:
        return _OLA_CAL[key]
    from data_processing.stochastic_rotor_noise import _ola_filter

    rng = np.random.default_rng(seed)
    need = (n_frames - 1) * hop + n_fft
    white = rng.standard_normal((1, need))
    flat = np.ones((1, n_frames, n_fft // 2 + 1))
    y = _ola_filter(white, flat, n_fft, hop)[0]
    window = np.hanning(n_fft + 1)[:n_fft]
    starts = np.arange(1 + (y.size - n_fft) // hop) * hop
    frames = np.stack([y[s : s + n_fft] for s in starts]) * window
    power = np.abs(np.fft.rfft(frames, axis=-1)) ** 2 / float(np.sum(window**2))
    # trim the ramp-up/ramp-down frames, where the overlap-add is incomplete
    inner = power[2:-2] if power.shape[0] > 6 else power
    cal = float(1.0 / np.sqrt(np.mean(inner)))
    _OLA_CAL[key] = cal
    return cal


def conditional_render(
    clip: Clip,
    *,
    variant: dict[str, Any] | None = None,
    seed: int = 0,
    device: str = "cpu",
    iters: tuple[int, int, int, int] = (150, 150, 300, 60),
    log: Any = print,
) -> ConditionalFit:
    """MAP-fit the model to ``clip`` and draw one sample from the fit."""
    pg, res = fit_conditional(clip, variant=variant, device=device, iters=iters, log=log)
    spectrum = np.asarray(res["spectrum"], dtype=np.float64)
    # ``fit_clip`` scales the periodogram to unit mean before fitting, so the
    # fitted spectrum lives in that scale; put it back on the clip's.
    scale = float(np.mean(pg.power))
    audio = sample_from_spectrum(
        spectrum * scale,
        pg.n_fft,
        pg.hop,
        clip.audio.shape[1],
        seed=seed,
    )
    return ConditionalFit(
        clip_id=clip.clip_id,
        audio=audio,
        spectrum=spectrum,
        power=np.asarray(pg.power, dtype=np.float64),
        nll_fit=float(res["scores"]["nll_fit"]),
        nll_loo=float(res["scores"]["nll_loo"]),
        params=res["params"],
        n_harm=int(res["spec"]["n_harm"]),
    )
