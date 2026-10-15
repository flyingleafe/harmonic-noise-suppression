"""Pre-mix noise-chunk augmentations and the augmentation-block vocabulary.

``policy.noise_augmentations`` (one block or a LIST of blocks applied in
order) runs on the sampled noise chunk *before* speech mixing, after the
optional ``noise_rate_warp`` (:mod:`data_processing.rate_warp`); the
post-mix ``policy.augmentations`` block lives in ``online_mixing``. Both use
the ``probability`` + ``choices`` block schema::

    noise_augmentations:
      - {probability: 0.5, choices: [{spectral_recolor: {}}]}
      - {probability: 0.5, choices: [{random_reverb: {drr_db_low: 3.0, drr_db_high: 15.0}}]}

Every numeric parameter may be a severity ramp (``online_mixing``
materialises it per sample id before this module sees it). The two
transforms are label-preserving and audio-only — the chunk's rotor track and
timeline are left untouched:

* ``spectral_recolor`` — multiply the STFT magnitude (model grid 2048/512) by
  a smooth random curve: gains ``U(-gain_db, +gain_db)`` (8 dB) at
  ``n_anchors`` (10) log-spaced anchors ``f_low..f_high`` (30 Hz..8 kHz),
  interpolated over bins in log-frequency, independent per channel.
* ``random_reverb`` — convolve each channel with a synthetic RIR drawn for
  this chunk: one room per chunk (``RT60 ~ U(rt60_low, rt60_high)``, 0.1..0.8 s;
  direct-to-reverberant ratio ``DRR ~ U(drr_db_low, drr_db_high)``, 3..15 dB;
  one tail tilt), an independent exponentially-decaying coloured-noise tail per
  channel (the incoherence of diffuse reverb across mics). Output
  renormalised to the input's global RMS. Draw order: rt60, drr, tilt, then
  one tail noise vector per channel.

The rp cutover (2026-10-14, ``docs/experiments/paper-real-protocol.md``)
removed ``freq_scale`` (replaced by ``noise_rate_warp``), ``tooth_dropout``,
``spec_mask``, ``floor_inject``, ``random_polarity``, ``channel_drop`` and the
``noise_time_warp`` key; pre-rp policies naming them are archival and fail at
build time with :func:`removed_error`.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import numpy as np
import scipy.signal as sps
import tdseries as td

#: STFT grid for ``spectral_recolor`` — the model's own analysis grid.
AUG_N_FFT = 2048
AUG_HOP = 512

#: Pre-mix choices and the parameters each accepts.
NOISE_AUGMENTATION_PARAMS: dict[str, tuple[str, ...]] = {
    "spectral_recolor": ("gain_db", "n_anchors", "f_low", "f_high"),
    "random_reverb": ("rt60_low", "rt60_high", "drr_db_low", "drr_db_high"),
}

#: Choice names and policy keys removed by the rp cutover -> their replacement.
REMOVED_AUGMENTATIONS: dict[str, str | None] = {
    "freq_scale": "noise_rate_warp",
    "noise_time_warp": "noise_rate_warp",
    "tooth_dropout": None,
    "spec_mask": None,
    "floor_inject": None,
    "random_polarity": None,
    "channel_drop": None,
}


def removed_error(name: str) -> ValueError:
    """The build-time error for an augmentation the rp cutover removed."""
    repl = REMOVED_AUGMENTATIONS.get(name)
    hint = f" (replaced by {repl})" if repl else ""
    return ValueError(
        f"augmentation {name!r} was removed in the rp cutover (2026-10-14){hint}; "
        "pre-rp online_mix policies are archival and not renderable — use "
        "conf/online_mix/rp_*.yaml (docs/experiments/paper-real-protocol.md)"
    )


def parse_choice(choice: Any) -> tuple[str, Mapping[str, Any]]:
    """``"name"`` or ``{name: params}`` -> ``(name, params)``."""
    if isinstance(choice, str):
        return choice, {}
    if isinstance(choice, Mapping):
        if len(choice) != 1:
            raise ValueError(f"augmentation choice must have one key, got {choice!r}")
        name, params = next(iter(choice.items()))
        return str(name), cast(Mapping[str, Any], params or {})
    raise ValueError(f"unsupported augmentation choice: {choice!r}")


def check_block(block: Any, known: Mapping[str, tuple[str, ...]], key: str) -> None:
    """Validate one ``probability`` + ``choices`` block against ``known``
    choice names and their parameter names (removed names get
    :func:`removed_error`)."""
    if not isinstance(block, Mapping):
        raise ValueError(f"{key}: block must be a mapping, got {block!r}")
    for choice in block.get("choices", []) or []:
        name, params = parse_choice(choice)
        if name in REMOVED_AUGMENTATIONS:
            raise removed_error(name)
        if name not in known:
            raise ValueError(f"{key}: unsupported augmentation {name!r} (known: {sorted(known)})")
        unknown = sorted(set(params) - set(known[name]))
        if unknown:
            raise ValueError(f"{key}.{name}: unknown parameters {unknown} (known: {known[name]})")


def draw_augmentation(
    spec: Mapping[str, Any] | None, rng: np.random.Generator
) -> tuple[str, Mapping[str, Any]] | None:
    """Fire the block and pick a choice — ``None`` when it does not fire.

    Consumes one fire-decision draw (none for an absent block or
    ``probability <= 0``) and, on a fire, one choice-index draw — the
    structure the ``check_stream`` control-stream methodology relies on.
    """
    if not spec:
        return None
    probability = float(spec.get("probability", 0.0))
    if probability <= 0.0 or rng.random() >= probability:
        return None
    choices = list(spec.get("choices", []))
    if not choices:
        return None
    return parse_choice(choices[int(rng.integers(0, len(choices)))])


# ── the two transforms: (audio (C, T), params, rng) -> audio (C, T) ─────────


def _spectral_recolor(
    audio: np.ndarray, params: Mapping[str, Any], rng: np.random.Generator, *, sample_rate: int
) -> np.ndarray:
    gain_db = float(params.get("gain_db", 8.0))
    n_anchors = int(params.get("n_anchors", 10))
    f_lo = float(params.get("f_low", 30.0))
    f_hi = float(params.get("f_high", 8000.0))
    T = audio.shape[-1]
    f, _, Z = sps.stft(audio, fs=sample_rate, nperseg=AUG_N_FFT, noverlap=AUG_N_FFT - AUG_HOP)
    anchors = np.geomspace(f_lo, f_hi, n_anchors)
    log_f = np.log(np.maximum(f, f_lo * 0.5))  # sub-f_low bins clamp to the first anchor
    for ch in range(Z.shape[0]):
        gains = rng.uniform(-gain_db, gain_db, size=n_anchors)
        curve_db = np.interp(log_f, np.log(anchors), gains)  # clamps beyond the anchor span
        Z[ch] *= (10.0 ** (curve_db / 20.0))[:, None]
    _, x = sps.istft(Z, fs=sample_rate, nperseg=AUG_N_FFT, noverlap=AUG_N_FFT - AUG_HOP)
    x = np.asarray(x, dtype=np.float32)
    if x.shape[-1] >= T:
        return np.ascontiguousarray(x[..., :T])
    return np.pad(x, ((0, 0), (0, T - x.shape[-1])))


def reverb_rirs(
    rng: np.random.Generator, n_channels: int, params: Mapping[str, Any], *, sample_rate: int
) -> list[np.ndarray]:
    """One chunk's unit-energy RIRs ``[direct, tail...]``, one per channel."""
    rt60 = float(
        rng.uniform(float(params.get("rt60_low", 0.1)), float(params.get("rt60_high", 0.8)))
    )
    drr_db = float(
        rng.uniform(float(params.get("drr_db_low", 3.0)), float(params.get("drr_db_high", 15.0)))
    )
    tilt = float(rng.uniform(0.0, 1.0))  # mild pink-ish tail coloration
    n = max(int(round(rt60 * sample_rate)), 8)
    freqs = np.fft.rfftfreq(n, 1.0 / sample_rate)
    shape = np.zeros_like(freqs)
    shape[1:] = freqs[1:] ** (-tilt / 2.0)
    envelope = np.exp(-6.9078 * np.arange(n) / (rt60 * sample_rate))  # -60 dB at RT60
    rirs = []
    for _ in range(n_channels):
        tail = np.fft.irfft(np.fft.rfft(rng.standard_normal(n)) * shape, n) * envelope
        e_tail = float(np.sum(tail**2)) or 1.0
        direct = np.sqrt(e_tail * 10.0 ** (drr_db / 10.0))
        rir = np.concatenate([[direct], tail])
        rirs.append(rir / np.sqrt(np.sum(rir**2)))
    return rirs


def _random_reverb(
    audio: np.ndarray, params: Mapping[str, Any], rng: np.random.Generator, *, sample_rate: int
) -> np.ndarray:
    T = audio.shape[-1]
    rms_in = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2))) or 1.0
    out = np.empty_like(audio)
    for ch, rir in enumerate(reverb_rirs(rng, audio.shape[0], params, sample_rate=sample_rate)):
        out[ch] = sps.fftconvolve(audio[ch].astype(np.float64), rir)[:T].astype(np.float32)
    rms_out = float(np.sqrt(np.mean(out.astype(np.float64) ** 2))) or 1.0
    return out * np.float32(rms_in / rms_out)


_AUG_FUNCS = {"spectral_recolor": _spectral_recolor, "random_reverb": _random_reverb}
assert set(_AUG_FUNCS) == set(NOISE_AUGMENTATION_PARAMS)


def maybe_apply_noise_augmentation(
    frame: td.Frame,
    spec: Mapping[str, Any] | None,
    rng: np.random.Generator,
    *,
    sample_rate: int,
) -> td.Frame:
    """Fire-and-apply one noise-chunk block from ``spec`` (resolved floats).

    No fire returns ``frame`` itself; a fire returns ``frame`` with only its
    ``audio`` entry replaced (same length, rate, dims and timeline), so the
    rotor track is bit-identical either way.
    """
    drawn = draw_augmentation(spec, rng)
    if drawn is None:
        return frame
    name, params = drawn
    func = _AUG_FUNCS.get(name)
    if func is None:
        if name in REMOVED_AUGMENTATIONS:
            raise removed_error(name)
        raise ValueError(f"unsupported noise augmentation: {name!r} (known: {sorted(_AUG_FUNCS)})")
    audio = cast(td.Series, frame["audio"])
    sr = float(cast(td.GridIndex, audio.tindex).sr)
    if int(round(sr)) != int(sample_rate):
        raise ValueError(f"noise augmentation needs {sample_rate} Hz audio, got {sr}")
    data = np.asarray(audio.data, dtype=np.float32)
    is_mono = data.ndim == 1
    out = func(data[None, :] if is_mono else data, params, rng, sample_rate=int(sample_rate))
    new_data = np.ascontiguousarray(out[0] if is_mono else out, dtype=np.float32)
    return frame.with_entry("audio", audio.with_data(new_data))
