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
from dataclasses import dataclass
from functools import lru_cache
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


#: The render's own anti-alias low-pass, specified rather than inherited.
#: Passband edge (the fit's own band edge, F_MAX), stopband edge (the output
#: Nyquist), stopband attenuation and the passband ripple the fit band must
#: keep; the filter length follows from them through ``kaiserord``, so the
#: transition is as narrow as the specification requires instead of as narrow
#: as a fixed length allows.
AA_PASS_HZ, AA_STOP_HZ, AA_STOP_DB, AA_RIPPLE_DB = 7900.0, 8000.0, 100.0, 0.01


def _ripple_attenuation_db(ripple_db: float) -> float:
    """Stopband attenuation a Kaiser design needs for a passband ripple.

    A Kaiser window's deviation ``delta`` is the SAME in both bands, so a
    passband tolerance of ``ripple_db`` (peak-to-nominal, dB) is the
    attenuation ``-20 log10(10**(ripple_db/20) - 1)``. Stating it separately
    is what makes the passband a specification rather than a by-product of the
    stopband number.
    """
    delta = 10.0 ** (float(ripple_db) / 20.0) - 1.0
    if delta <= 0.0:
        raise ValueError(f"ripple_db must be positive, got {ripple_db}")
    return float(-20.0 * np.log10(delta))


@dataclass(frozen=True)
class AntialiasFilter:
    """The render's low-pass, as a designed object rather than a side effect."""

    taps: np.ndarray  # (L,) odd length, symmetric: type-I linear phase
    sample_rate: float
    pass_hz: float
    stop_hz: float
    stop_db: float
    ripple_db: float
    beta: float

    @property
    def delay(self) -> int:
        """Group delay in samples — exactly the centre tap of a type-I FIR."""
        return (self.taps.size - 1) // 2

    def spec(self) -> dict[str, float]:
        return dict(
            sample_rate=float(self.sample_rate),
            pass_hz=float(self.pass_hz),
            stop_hz=float(self.stop_hz),
            stop_db=float(self.stop_db),
            ripple_db=float(self.ripple_db),
            beta=float(self.beta),
            n_taps=int(self.taps.size),
            delay=int(self.delay),
        )


@lru_cache(maxsize=8)
def antialias_filter(
    sample_rate: float,
    pass_hz: float = AA_PASS_HZ,
    stop_hz: float = AA_STOP_HZ,
    stop_db: float = AA_STOP_DB,
    ripple_db: float = AA_RIPPLE_DB,
) -> AntialiasFilter:
    """Design the render low-pass from its specification (cached per spec).

    ``kaiserord`` sizes the filter from the transition width and the stricter
    of the two band specifications, so both the stopband attenuation AND the
    passband ripple are honoured by construction.
    """
    from scipy.signal import firwin, kaiserord

    nyquist = float(sample_rate) / 2.0
    width = (float(stop_hz) - float(pass_hz)) / nyquist
    if not 0.0 < float(pass_hz) < float(stop_hz) <= nyquist or not 0.0 < width < 1.0:
        raise ValueError(
            f"transition [{pass_hz}, {stop_hz}] Hz is not inside the Nyquist band "
            f"of {sample_rate} Hz (Nyquist {nyquist} Hz)"
        )
    attenuation = max(float(stop_db), _ripple_attenuation_db(ripple_db))
    n_taps, beta = kaiserord(attenuation, width)
    n_taps = int(n_taps) | 1  # firwin wants an odd length for a type-I linear phase
    taps = firwin(n_taps, (pass_hz + stop_hz) / 2.0, window=("kaiser", beta), fs=sample_rate)
    return AntialiasFilter(
        taps=np.asarray(taps, dtype=np.float64),
        sample_rate=float(sample_rate),
        pass_hz=float(pass_hz),
        stop_hz=float(stop_hz),
        stop_db=float(stop_db),
        ripple_db=float(ripple_db),
        beta=float(beta),
    )


def antialias(
    x: np.ndarray,
    sample_rate: float,
    *,
    pass_hz: float = AA_PASS_HZ,
    stop_hz: float = AA_STOP_HZ,
    stop_db: float = AA_STOP_DB,
    ripple_db: float = AA_RIPPLE_DB,
) -> np.ndarray:
    """Low-pass ``x`` so that nothing above ``stop_hz`` survives decimation.

    A rendered comb carries lines above the OUTPUT Nyquist; the shared
    decimator's own filter is a fixed-length Kaiser whose stop band is about
    40-50 dB, and raising its beta widens the transition instead of deepening
    it at constant length. So the render removes its own out-of-band energy
    first, with a filter whose passband, stopband, attenuation and ripple are
    stated (:func:`antialias_filter`).

    ONE application, by overlap-add FFT convolution, centred on the type-I
    filter's own group delay. Three consequences, each a property the previous
    ``filtfilt`` route did not have:

    * the realized response is ``H``, not ``|H|^2``, so the designed 0.01 dB
      passband ripple and 100 dB stop band are the ones the render gets
      instead of their squares;
    * the cost is one FFT convolution, not two direct passes over 2827 taps;
    * the edges are a deterministic zero-extension (a linear convolution
      centred on the delay), defined for EVERY input length. ``filtfilt``
      refuses any input shorter than ``3 * (n_taps - 1)`` — 8478 samples,
      0.19 s at 44.1 kHz — and its odd extension makes short-input behaviour a
      function of the length.

    Zero phase comes from the symmetric odd-length kernel: the centred slice is
    aligned sample-for-sample with the input, so a rendered clip's rotor track
    still describes its own audio.
    """
    from scipy.signal import oaconvolve

    filt = antialias_filter(
        float(sample_rate), float(pass_hz), float(stop_hz), float(stop_db), float(ripple_db)
    )
    a = np.asarray(x, dtype=np.float64)
    n = a.shape[-1]
    kernel = filt.taps.reshape((1,) * (a.ndim - 1) + (filt.taps.size,))
    full = oaconvolve(a, kernel, mode="full", axes=-1)
    return np.ascontiguousarray(full[..., filt.delay : filt.delay + n])


#: The DECLARED power transfer of the synthetic render chain, stated so that a
#: model fitted on real 16 kHz data and a renderer working on an oversampled
#: grid agree on what the observation is. Two components, attributed
#: separately because they are physically different filters:
#:
#: * the render's own anti-alias FIR (:func:`antialias_filter`), designed at
#:   the WORK rate — flat through 7900 Hz within its 0.01 dB ripple spec and
#:   100 dB down at 8000 Hz, so it contributes essentially nothing in band;
#: * the unchanged ``scipy.signal.resample_poly`` decimator of
#:   :func:`clips.decimate` — an 81-tap Kaiser(5.0) design at 64 -> 16 kHz
#:   whose own rolloff is what costs ~4.9 dB at 7900 Hz.
#:
#: The product is the multiplier an expected spectrum must carry. It is a
#: WINDOW/FILTER-COMMUTATION approximation: the real chain filters before the
#: analysis window, this scales an already-windowed expected spectrum.
#:
#: KEYS ARE ADD-ONLY AND MUST STAY PLAIN JSON (addendum 13 §66-67): a candidate
#: export copies this dict verbatim into its ``training_provenance``, and
#: ``stage2.save``'s ``plain()`` would raise on a dataclass or a numpy scalar.
#: Never rename an existing key — an export written under today's keys must
#: still be readable — and never store a numpy float.
RENDER_TRANSFER_SPEC: dict[str, Any] = dict(
    components=["render anti-alias FIR at the work rate", "resample_poly decimator"],
    antialias=dict(
        pass_hz=AA_PASS_HZ, stop_hz=AA_STOP_HZ, stop_db=AA_STOP_DB, ripple_db=AA_RIPPLE_DB
    ),
    decimator=dict(
        source="scipy.signal.resample_poly defaults, as clips.decimate calls it",
        window=["kaiser", 5.0],
        half_len_rule="10 * max(up, down)",
        cutoff_rule="1 / max(up, down) of the internal (up * sr_in) Nyquist = min(sr_in, sr_out) / 2",
        upsampling_gain="firwin taps are multiplied by `up`, as resample_poly does; the response "
        "is then normalized to unit DC gain, so a future up != 1 cannot silently rescale",
    ),
    normalization="unit gain at DC; power transfer is |H_aa|^2 * |H_dec|^2",
    approximation=(
        "window/filter commutation: the physical chain filters before the Hann window, this "
        "multiplies an already-windowed expected spectrum"
    ),
    #: ONE centered overlap-add application of the AA FIR, so the AA enters the
    #: transfer as |H_aa|^2 and never |H_aa|^4. This is what replacing
    #: ``filtfilt`` bought, and recording it means no export can be ambiguous
    #: about which transfer was multiplied in.
    aa_application="single_pass_centered_fir",
    #: COMPUTED below (never asserted): the worst-case attenuation, in dB, of
    #: anything that folds into 0..8000 Hz under 4:1 decimation.
    alias_floor_db=None,
)


def _resample_poly_taps(up: int, down: int, beta: float = 5.0) -> np.ndarray:
    """The FIR ``scipy.signal.resample_poly`` builds for ``up``/``down``.

    Reproduced from its documented default design — ``firwin(2 * half_len + 1,
    1 / max(up, down), window=('kaiser', beta)) * up`` with
    ``half_len = 10 * max(up, down)`` — because scipy exposes no accessor for
    it. The ``* up`` is ``resample_poly``'s own upsampling gain; it is kept so
    the convention is explicit, and the response is normalized to unit DC gain
    afterwards so a future ``up != 1`` cannot silently rescale anything.
    :func:`render_transfer_power`'s regression test measures the real
    ``resample_poly`` response against this, so a scipy change surfaces as a
    test failure rather than as a silent mismatch.
    """
    from scipy.signal import firwin

    max_rate = max(int(up), int(down))
    half_len = 10 * max_rate
    taps = firwin(2 * half_len + 1, 1.0 / max_rate, window=("kaiser", float(beta))) * int(up)
    return np.asarray(taps, dtype=np.float64)


@lru_cache(maxsize=8)
def _transfer_pieces(
    sample_rate_work: int, sample_rate_out: int
) -> tuple[np.ndarray, np.ndarray, int]:
    from math import gcd

    g = gcd(int(sample_rate_work), int(sample_rate_out))
    up, down = int(sample_rate_out) // g, int(sample_rate_work) // g
    return antialias_filter(float(sample_rate_work)).taps, _resample_poly_taps(up, down), up


def render_transfer_power(
    freqs_hz: np.ndarray, *, sample_rate_work: int, sample_rate_out: int = SR
) -> np.ndarray:
    """``|H(f)|^2`` of the whole render chain at OUTPUT-band frequencies.

    ``freqs_hz`` is real ``(F,)`` in ``0 .. sample_rate_out/2`` (normally the
    ``n_fft_analysis//2 + 1`` analysis bins); the return is ``(F,)`` real,
    dimensionless POWER gain — already squared, linear, not dB, and about 1 in
    the passband.

    Composed in the order the renderer actually applies them: (1) the render
    anti-alias FIR designed at ``sample_rate_work`` (:func:`antialias_filter`,
    reused — no second filter is designed), applied ONCE
    (``aa_application="single_pass_centered_fir"``, so ``|H_aa|^2`` and never
    ``|H_aa|^4``); (2) the UNCHANGED ``resample_poly`` FIR for
    ``sample_rate_work -> sample_rate_out``, derived from the rates rather than
    hardcoded. Both are normalized to unit gain at DC and evaluated on the
    grid each filter actually lives on.

    This is a WINDOW/FILTER-COMMUTATION APPROXIMATION, not an exact
    moving-filtered periodogram: the real renderer filters BEFORE the Hann
    window, while a consumer multiplies an already-windowed expected spectrum
    by this gain. The finite-window kernel it multiplies remains exact.
    ``clips.decimate`` is only MODELLED here, never replaced.

    Attribution, so nothing is double-counted: at 7900 Hz the AA contributes
    ~0 dB (its 0.01 dB ripple spec; its -100 dB figure is at 8000 Hz) and the
    ~-4.9 dB is the decimator's own rolloff. Apply the result EXACTLY ONCE.
    """
    from scipy.signal import freqz

    f = np.asarray(freqs_hz, dtype=np.float64)
    aa, dec, up = _transfer_pieces(int(sample_rate_work), int(sample_rate_out))
    # the decimator's design lives on the internal up * sr_work grid; with
    # up == 1 for an integer decimation that is the work grid itself
    w_aa = 2.0 * np.pi * f / float(sample_rate_work)
    w_dec = 2.0 * np.pi * f / (float(sample_rate_work) * up)
    h_aa = np.abs(freqz(aa, worN=w_aa)[1]) / np.abs(freqz(aa, worN=[0.0])[1][0])
    h_dec = np.abs(freqz(dec, worN=w_dec)[1]) / np.abs(freqz(dec, worN=[0.0])[1][0])
    return (h_aa * h_dec) ** 2


def alias_floor_db(
    sample_rate_work: int = 4 * SR, sample_rate_out: int = SR, *, n: int = 4096
) -> float:
    """Worst-case attenuation, in dB, of anything that FOLDS into the output band.

    A COMPUTED number, not an assertion (addendum 13 §66): under
    ``sample_rate_work / sample_rate_out`` decimation, an output frequency
    ``f`` also receives everything at ``k * sample_rate_out +- f`` above the
    output Nyquist. The designed chain's response is evaluated at every such
    image frequency inside the work band and the LEAST attenuated one is
    returned, which is what justifies treating the transfer as purely
    multiplicative in the output band.
    """
    ratio = int(round(float(sample_rate_work) / float(sample_rate_out)))
    nyq_out = float(sample_rate_out) / 2.0
    base = np.linspace(0.0, nyq_out, int(n))
    images: list[float] = []
    for k in range(1, ratio + 1):
        for sign in (-1.0, 1.0):
            f = k * float(sample_rate_out) + sign * base
            images.extend(f[(f > nyq_out) & (f <= float(sample_rate_work) / 2.0)].tolist())
    if not images:
        return float("inf")
    gains = render_transfer_power(
        np.asarray(images), sample_rate_work=sample_rate_work, sample_rate_out=sample_rate_out
    )
    return float(10.0 * np.log10(float(np.max(gains))))


RENDER_TRANSFER_SPEC["alias_floor_db"] = round(alias_floor_db(), 3)
RENDER_TRANSFER_SPEC["alias_floor_expectation_db"] = -100.0


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
    # high orders are cut while the slowest keeps padding. Lines above the
    # OUTPUT Nyquist are therefore rendered on purpose and removed by
    # :func:`antialias` before decimation, never by truncating the ladder: one
    # order is above the band for the fast rotor and inside it for the slow
    # one, and the profile is a single shared (rotor, order) grid.
    k_max = max(2, int(np.floor((sample_rate / 2) / max(float(rates.min()), 1.0))))
    k_use = min(k_max, profile.shape[1])
    profile = profile[:n_rotors, :k_use].copy()
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
        profile_db=profile,
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
    sample_rate_work: int = clips.NATIVE_SR,
    n_mics: int = 8,
    seed: int = 0,
    normalize_rms: float | None = 0.1,
) -> np.ndarray:
    """``(M, T)`` synthetic cruise audio at 16 kHz from a MAP export.

    Rendered on a WORK grid and decimated on the real clips' own path, so the
    synthetic clip carries no band edge the real clips do not have. The word
    "native" is retired as ambiguous (addendum 11 §55): ``sample_rate_work`` is
    the model/render grid and 16 kHz is the analysis grid. The evaluator passes
    the declared 64000 explicitly; the default stays this module's historical
    44.1 kHz so every existing caller renders exactly what it always did.

    The comb runs past the OUTPUT Nyquist — at these rotor speeds the fitted
    ladder reaches about 10.8 kHz against an 8 kHz output Nyquist — so the
    render low-passes ITSELF (:func:`antialias`) before the shared decimator
    touches it. Without that step the out-of-band lines folded down: decomposing
    the accepted raw-label render showed the above-8 kHz component contributing
    14.5 dB to the decimated 7.5-8 kHz band against 5.7 dB from genuine in-band
    content, and a 9 kHz tone arriving only 31 dB down. The power gain that
    chain applies in the output band is :func:`render_transfer_power`, and any
    expected spectrum compared against this render must carry it exactly once.

    ``normalize_rms`` is handed to
    :func:`data_processing.stochastic_rotor_noise.synthesize` unchanged. The
    default 0.1 is that function's own default and is what every existing
    caller (``_stage2_probe.py``, ``_stage2_panels.py``, ``_ab_render.py``, the
    training streams) has always got: the clip is rescaled to a fixed RMS and
    its ABSOLUTE level is arbitrary. ``normalize_rms=None`` keeps the model's
    own level, which is what an absolute-level comparison needs — see
    :func:`experiments.stochastic_fit.revised_eval.to_renderer_units` for the
    declared conversion from stored fit power to renderer units that must be
    applied with it.
    """
    from data_processing import stochastic_rotor_noise as srn

    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    rates = rps.mean(axis=1)
    params = params_from_export(export, rates, sample_rate=sample_rate_work, n_mics=n_mics)
    # the trajectory is resampled onto the work grid the renderer runs on
    n_work = int(round(rps.shape[-1] / SR * sample_rate_work))
    t_src = np.linspace(0.0, 1.0, rps.shape[-1])
    t_dst = np.linspace(0.0, 1.0, n_work)
    rps_work = np.stack([np.interp(t_dst, t_src, r) for r in rps])
    audio, _ = srn.synthesize(
        params,
        rps_work,
        rng=np.random.default_rng(seed),
        n_mics=n_mics,
        line_mode="fm",
        n_fft=1 << 16,
        normalize_rms=normalize_rms,
    )
    audio = antialias(np.asarray(audio, dtype=np.float64), sample_rate_work)
    clip = Clip("synthetic", "synthetic", np.asarray(audio, np.float32), rps_work, sample_rate_work)
    return np.asarray(clips.decimate(clip, SR).audio, dtype=np.float64)
