"""The v2 noise model's SUPPORTS: the observation windows every fit and gate reads.

A :class:`Support` is one frozen observation: a periodogram in the evaluator's
absolute units, the per-rotor carrier that produced it, and the identity of the
material it was cut from. Nothing here fits anything — this module only decides
*what is observed*, so that a fit, a render and a gate can all name the same
bytes.

Two kinds:

``bench``
    One stationary segment of a bench/static recording, as ONE periodogram of
    the whole segment (``n_fft = hop = n_samples``, so ``n_frames == 1``). The
    resolution is therefore ``1/T`` — 0.03-0.25 Hz — which is what makes the
    decohered line SHAPE visible at all; a 128 ms analysis frame smears every
    line to 12 Hz and the bench material's whole point is lost.
``flight``
    A flight window on the model's front-end grid (NFFT 2048, hop 512 at
    16 kHz, periodic Hann), with the published rotor label as the carrier.

**Units.** ``power`` is always ``|rfft(x * w)|^2 / sum(w^2)`` with a periodic
Hann ``w`` — literally :func:`experiments.stochastic_fit.revised_eval.window_periodogram`,
one-sided (rfft bins, NOT doubled), white noise of variance ``s^2`` reading
``s^2``. Every support is taken at ``sr = 16000`` after
:func:`experiments.stochastic_fit.clips.decimate`, native rate or not, so a
consumer never applies a rate factor: the renderer runs at 16 kHz, the frozen
observation band tops out at 7900 Hz, and the floor fit reads 16 kHz flight
material. (The convention is not rate-free: a bin-centred tone reads
``A^2 * n_fft / 6`` and broadband reads ``S * sr / 2``, so for a fixed segment
DURATION both scale with the rate and a native-rate periodogram needs the
scalar ``16000 / sr_native`` to sit at the same level. Decimating first removes
that scalar instead of propagating it —
``tests/experiments/test_noise_model_supports.py`` plants a sinusoid and
asserts the two routes agree.)

**Bench stationarity rule.** For a DREGON single-motor recording there is no
telemetry at all, so the carrier is the corpus survey's speed estimate and the
stationary span has to be measured from the audio. The approved rule
demodulates one high order (the brightest in 60-80, chosen with the survey
speed) and keeps the longest span where the residual frequency stays inside
+-1 Hz. Four things the rule needs to be usable, all recorded per recording:

1. The survey speed is quantised to ~0.01 rev/s, which at order 70 misplaces
   the line by up to 2.8 Hz (measured) — more than the whole tolerance, and
   enough to smear every order above ~60 in a fit. The carrier is therefore
   REFINED from the survey speed by integrating the demodulated residual
   (:func:`refine_carrier`); ``survey_rev_s`` keeps the manifest value and
   ``carrier_shift_rev_s`` the correction (measured: up to 0.076 rev/s).
2. "Brightest" is measured as the order's MARGIN over the floor 6-18 Hz away
   (:func:`line_margins`), not as raw level, so the same number that picks the
   order also says whether that order carries a line at all. Measured over the
   21 DREGON recordings: 3.4-12.1 dB.
3. The residual is read from a NARROW 1.5 Hz band, averaged over the
   microphones and then over :data:`BENCH_RESIDUAL_SMOOTH_S`, not
   instantaneous. At order ~70 the raw instantaneous residual has a 20-60 Hz
   standard deviation and even a 50 ms average keeps ~0.45 Hz: that is the
   phase noise the model exists to describe (the decoherence study's ``D_k``),
   not a speed drift, and a +-1 Hz test on it accepts nothing longer than
   30 ms. Mic-averaged over a 1.5 Hz band and smoothed over 2 s the jitter is
   0.16-0.33 Hz, so +-1 Hz is the 3-6 sigma DRIFT test it was meant to be.
4. A narrow band alone is not enough: a carrier that drifts FASTER than the
   band can follow leaves it altogether, and the residual of the filtered
   noise that remains sits near zero — an absent line reads as perfect
   stationarity. Every accepted window must therefore ALSO keep the residual
   of a WIDE (+-0.45 f0) band inside :data:`BENCH_WIDE_TOL_HZ`, which still
   contains a drifting line and reports its drift.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from data_processing.derivations import NOISE_V2_MANIFEST
from experiments.stochastic_fit import clips as C
from experiments.stochastic_fit import revised_eval as RE
from experiments.stochastic_fit.data import Clip
from utils.demod import demodulate, residual_frequency

#: Every support lives at this rate — the renderer's and the evaluator's.
SR = 16000
#: The flight front end (``data.N_FFT``/``data.HOP``), periodic Hann.
OBS_N_FFT = 2048
OBS_HOP = 512

#: Round-1 cache root. ``<name>.npz`` per support plus ``index.json``.
CACHE_DIR = Path("results/noise_v2/rounds/round1/supports")

DREGON_DATASET = "DREGON-frames"
MICHAELS_DATASET = "michaels-frames"
BENCH_POINT_DATASET = "noise-v2-bench-points"

#: The RAW carrier DREGON's frozen evaluator measures against.
DREGON_RPS_KEY = "motors_command"

# ── the bench stationarity rule (see the module docstring) ──────────────────

#: Orders searched for the demodulation carrier — the approved 60-80 window.
BENCH_ORDER_RANGE = (60, 80)
#: Half-width of the peak search around ``k * survey_speed``, in Hz. Covers the
#: survey speed's own quantisation at order 80 (0.005 rev/s -> 0.4 Hz) with
#: room for the measured +-2.8 Hz misplacement.
BENCH_ORDER_TOL_HZ = 4.0
#: The floor annulus the order's peak is measured against, in Hz from the line.
BENCH_MARGIN_BAND_HZ = (6.0, 18.0)
#: How far the order's peak must clear that floor for the order to carry a LINE
#: at all. Measured over the 21 DREGON bench recordings: 3.4-12.1 dB, so every
#: recording clears 3 dB and a recording with no usable line at any order in
#: 60-80 fails the rule outright instead of having its residual read from
#: filtered noise.
BENCH_LINE_MARGIN_DB = 3.0
#: Demodulation half-bands of the carrier refinement, narrowing each pass.
BENCH_REFINE_BANDS = (8.0, 4.0, 2.0)
#: Half-band of the NARROW residual read-out. The decoherence pedestal is
#: ~1.6 Hz wide (``D_k/(2 pi)`` at k ~ 70), so 1.5 Hz keeps the line and little
#: else: widening it to 3 Hz doubles the jitter and costs 17 of 21 recordings.
BENCH_DEMOD_BAND_HZ = 1.5
#: Half-band of the WIDE locator read-out, as a fraction of the rotor speed
#: (about +-30 Hz), and the tolerance its residual must hold. A line that
#: drifts FASTER than the narrow band can follow leaves that band within a
#: fraction of the averaging window and the narrow residual then reads filtered
#: noise, i.e. near-perfect stationarity. The wide band still contains the
#: drifting line, so its residual reports the drift; a real bench recording
#: keeps both residuals near zero.
BENCH_WIDE_BAND_FRAC = 0.45
BENCH_WIDE_TOL_HZ = 3.0
#: Averaging length of the residual FREQUENCY, in seconds. The residual is
#: averaged over the microphones first and then over this window: at order ~70
#: the raw instantaneous residual has a 20-60 Hz standard deviation and even
#: 50 ms of averaging keeps ~0.45 Hz, which is the phase noise the model exists
#: to describe, not a speed drift. Mic-averaged and smoothed over 2 s the
#: jitter is 0.16-0.33 Hz (measured), so +-1 Hz is a 3-6 sigma drift test.
BENCH_RESIDUAL_SMOOTH_S = 2.0
#: The approved tolerance.
BENCH_RESIDUAL_TOL_HZ = 1.0
#: The approved minimum stationary span.
BENCH_MIN_SEGMENT_S = 4.0
#: Trimmed from both ends: ``sosfiltfilt`` transients and the moving average's
#: own end padding.
BENCH_EDGE_S = 2.0
#: Grid the fallback window search walks, in seconds (a failed recording only
#: needs a defensible window, not a sample-exact optimum).
BENCH_FALLBACK_STEP_S = 0.1


@dataclass
class Support:
    """One frozen observation window. Shapes are asserted, not assumed."""

    kind: str  # "bench" | "flight"
    name: str
    sr: int
    n_mics: int
    freqs_hz: np.ndarray  # (F,) == rfftfreq(n_fft, 1/sr)
    power: np.ndarray  # (n_mics, n_frames, F), |X|^2/sum(w^2)
    carrier_rev_s: np.ndarray  # (R, n_frames) frame-mean rotor speed
    carrier_rev_s_audio: np.ndarray  # (R, n_samples) the label on the audio grid
    frame_starts: np.ndarray  # (n_frames,) int sample offsets into the segment
    frame_centres_s: np.ndarray  # (n_frames,) == (frame_starts + n_fft/2)/sr
    n_fft: int
    hop: int
    segment: tuple[float, float]  # (start, end) on the RECORDING clock
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in ("bench", "flight"):
            raise ValueError(f"{self.name}: kind must be bench|flight, not {self.kind!r}")
        m, n, f = self.power.shape
        if m != self.n_mics:
            raise ValueError(f"{self.name}: power has {m} mics, n_mics={self.n_mics}")
        if f != self.freqs_hz.size:
            raise ValueError(f"{self.name}: power has {f} bins, freqs_hz {self.freqs_hz.size}")
        if self.carrier_rev_s.shape[1] != n or self.frame_centres_s.size != n:
            raise ValueError(f"{self.name}: {n} frames, carrier/centres disagree")
        if self.carrier_rev_s_audio.shape[0] != self.carrier_rev_s.shape[0]:
            raise ValueError(f"{self.name}: audio-grid carrier has a different rotor count")
        if self.kind == "bench" and n != 1:
            raise ValueError(f"{self.name}: a bench support is ONE frame, got {n}")

    @property
    def n_rotors(self) -> int:
        return int(self.carrier_rev_s.shape[0])

    @property
    def n_frames(self) -> int:
        return int(self.power.shape[1])

    @property
    def n_samples(self) -> int:
        return int(self.carrier_rev_s_audio.shape[1])

    @property
    def duration_s(self) -> float:
        return self.n_samples / float(self.sr)


# ── specs ───────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SupportSpec:
    """What to load, as data: a family plus its arguments.

    ``text`` round-trips through :func:`parse_spec`; ``name`` is the cache stem
    and the key every record (fits, index, round JSON) uses.
    """

    family: str  # "bench_dregon_motor" | "bench_point" | "flight_michaels" | "flight_dregon"
    kind: str  # "bench" | "flight"
    name: str
    text: str
    args: dict[str, Any]


def bench_dregon_motor(rotor: str, throttle: int) -> SupportSpec:
    """A DREGON bench recording, ``rotor`` in ``Motor1..4``/``allMotors``."""
    rid = f"motor_{rotor}_{int(throttle)}"
    return SupportSpec(
        "bench_dregon_motor",
        "bench",
        f"bench_dregon_{rotor}_{int(throttle)}",
        f"bench_dregon_motor:{rotor}:{int(throttle)}",
        dict(rotor=str(rotor), throttle=int(throttle), recording_id=rid),
    )


def bench_point(point_id: str) -> SupportSpec:
    """One published ``noise-v2-bench-points`` frame, by its manifest key."""
    return SupportSpec(
        "bench_point",
        "bench",
        f"bench_point_{point_id}",
        f"bench_point:{point_id}",
        dict(point_id=str(point_id)),
    )


def _michaels_rps_key(recording: str) -> str:
    """FLY125 fits on the refined label (as the legacy cruise export did);
    FLY124 is the frozen evaluation cohort and stays on RAW ``rps``."""
    return "rps" if str(recording).upper() == "FLY124" else "rps_refined"


def flight_michaels(
    recording: str, start_s: float, dur_s: float, *, rps_key: str | None = None
) -> SupportSpec:
    key = rps_key or _michaels_rps_key(recording)
    return SupportSpec(
        "flight_michaels",
        "flight",
        f"flight_michaels_{recording}@{float(start_s):.3f}+{float(dur_s):g}_{key}",
        f"flight_michaels:{recording}:{start_s!r}:{dur_s!r}:{key}",
        dict(
            recording=str(recording),
            start_s=float(start_s),
            dur_s=float(dur_s),
            rps_key=key,
            dataset=MICHAELS_DATASET,
        ),
    )


def flight_dregon(
    recording: str, start_s: float, dur_s: float, *, rps_key: str = DREGON_RPS_KEY
) -> SupportSpec:
    return SupportSpec(
        "flight_dregon",
        "flight",
        f"flight_dregon_{recording}@{float(start_s):.3f}+{float(dur_s):g}_{rps_key}",
        f"flight_dregon:{recording}:{start_s!r}:{dur_s!r}:{rps_key}",
        dict(
            recording=str(recording),
            start_s=float(start_s),
            dur_s=float(dur_s),
            rps_key=str(rps_key),
            dataset=DREGON_DATASET,
        ),
    )


def parse_spec(text: str) -> SupportSpec:
    """Inverse of :attr:`SupportSpec.text`."""
    head, _, rest = str(text).partition(":")
    parts = rest.split(":") if rest else []
    if head == "bench_dregon_motor" and len(parts) == 2:
        return bench_dregon_motor(parts[0], int(parts[1]))
    if head == "bench_point" and len(parts) == 1:
        return bench_point(parts[0])
    if head in ("flight_michaels", "flight_dregon") and len(parts) in (3, 4):
        rec, start, dur = parts[0], float(parts[1]), float(parts[2])
        key = parts[3] if len(parts) == 4 else None
        if head == "flight_michaels":
            return flight_michaels(rec, start, dur, rps_key=key)
        return flight_dregon(rec, start, dur, rps_key=key or DREGON_RPS_KEY)
    raise ValueError(f"unparseable support spec {text!r}")


def as_spec(spec: SupportSpec | str) -> SupportSpec:
    return spec if isinstance(spec, SupportSpec) else parse_spec(spec)


# ── the bench survey manifest ───────────────────────────────────────────────


def bench_manifest() -> dict[str, dict[str, Any]]:
    """The committed fit-point manifest, keyed by point id.

    This is the label set for every bench support: the speeds are estimator
    output (``scripts/noise_v2_bench_speed.py``), not telemetry, and the bench
    recordings publish no rotor track at all.
    """
    points = json.loads(NOISE_V2_MANIFEST.read_text())["points"]
    return {str(p["key"]): p for p in points}


def dregon_bench_point_key(recording_id: str) -> str:
    """``motor_Motor1_70`` -> its manifest key ``DREGON-bench__motor_Motor1_70``."""
    return f"DREGON-bench__{recording_id}"


# ── the bench stationarity rule ─────────────────────────────────────────────


def _moving_mean(v: np.ndarray, n: int) -> np.ndarray:
    """Centred boxcar mean — the same average
    :func:`utils.demod.residual_frequency` applies, in O(N) instead of the
    O(N*n) convolution (a 1 s kernel on a 45 s 44.1 kHz track is 2e11 ops)."""
    n = int(n)
    if n <= 1:
        return np.asarray(v, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    pad = np.concatenate([np.full(n // 2, v[0]), v, np.full(n - n // 2, v[-1])])
    cs = np.cumsum(np.concatenate([[0.0], pad]))
    return (cs[n:] - cs[:-n])[: v.size] / n


def _longest_run(mask: np.ndarray) -> tuple[int, int]:
    """``(start, stop)`` of the longest True run; ``(0, 0)`` if there is none."""
    edges = np.diff(np.concatenate([[0], np.asarray(mask, dtype=np.int8), [0]]))
    starts = np.flatnonzero(edges == 1)
    stops = np.flatnonzero(edges == -1)
    if starts.size == 0:
        return 0, 0
    i = int(np.argmax(stops - starts))
    return int(starts[i]), int(stops[i])


def block_periodogram(
    x: np.ndarray, sr: float, *, n_fft: int = 1 << 17
) -> tuple[np.ndarray, np.ndarray]:
    """Block-averaged periodogram of ``(M, T)`` or ``(T,)``, averaged over mics.

    0.34 Hz bins at 44.1 kHz — fine enough to separate a decoherence pedestal
    (~1.6 Hz wide) from the floor 6-18 Hz away, which is what makes the
    order's margin measurable at all.
    """
    x = np.asarray(x, dtype=np.float64)
    x = x[None, :] if x.ndim == 1 else x
    n_fft = int(min(n_fft, 1 << int(np.log2(max(x.shape[-1], 2)))))
    window = np.hanning(n_fft + 1)[:n_fft]
    n_blocks = max(1, x.shape[-1] // n_fft)
    acc = np.zeros(n_fft // 2 + 1)
    for b in range(n_blocks):
        block = x[:, b * n_fft : (b + 1) * n_fft] * window
        acc += np.mean(np.abs(np.fft.rfft(block, axis=-1)) ** 2, axis=0)
    acc /= n_blocks * float(np.sum(window**2))
    return np.fft.rfftfreq(n_fft, 1.0 / float(sr)), acc


def line_margins(
    x: np.ndarray,
    sr: float,
    f0: float,
    *,
    order_range: tuple[int, int] = BENCH_ORDER_RANGE,
    tol_hz: float = BENCH_ORDER_TOL_HZ,
) -> dict[int, float]:
    """Per-order ``peak / neighbouring floor`` in dB, for orders of ``f0``.

    The peak is the largest bin within ``k*f0 +- tol_hz`` (the survey speed's
    quantisation cannot move a line out of its own search window) and the floor
    is the MEDIAN over :data:`BENCH_MARGIN_BAND_HZ` either side.
    """
    freqs, power = block_periodogram(x, sr)
    lo, hi = BENCH_MARGIN_BAND_HZ
    out: dict[int, float] = {}
    for k in range(int(order_range[0]), int(order_range[1]) + 1):
        centre = k * float(f0)
        if centre + hi > 0.45 * sr:
            continue
        peak_sel = (freqs >= centre - tol_hz) & (freqs <= centre + tol_hz)
        off = np.abs(freqs - centre)
        floor_sel = (off >= lo) & (off <= hi)
        if not np.any(peak_sel) or not np.any(floor_sel):
            continue
        out[k] = float(
            10.0 * np.log10(float(power[peak_sel].max()) / float(np.median(power[floor_sel])))
        )
    return out


def select_order(x: np.ndarray, sr: float, f0: float) -> tuple[int, float]:
    """``(order, margin_db)``: the brightest order of ``f0`` in 60-80.

    "Brightest" is measured as the MARGIN over the neighbouring floor rather
    than as raw level, so the same number that picks the order also says
    whether that order carries a line at all.
    """
    margins = line_margins(x, sr, f0)
    if not margins:
        raise ValueError(f"no order in {BENCH_ORDER_RANGE} of {f0} rev/s fits under Nyquist")
    order = max(margins, key=lambda k: margins[k])
    return int(order), float(margins[order])


def refine_carrier(
    x: np.ndarray,
    sr: float,
    f0: float,
    order: int,
    *,
    bands: Sequence[float] = BENCH_REFINE_BANDS,
    edge_s: float = 0.5,
) -> float:
    """Survey speed -> the speed the order-``order`` line actually sits at.

    Each pass demodulates at the current estimate, reads the MEAN residual
    frequency and divides it back by the order; the band narrows every pass, so
    a 2.8 Hz misplacement is captured by the 8 Hz pass and the 2 Hz pass then
    measures the residual with the floor mostly excluded.
    """
    x = np.asarray(x, dtype=np.float64)
    edge = int(round(float(edge_s) * sr))
    f = float(f0)
    for band in bands:
        z = demodulate(x, np.full(x.size, f), float(band), sr, order=order)
        res = residual_frequency(z, sr, smooth_s=0.0)
        f += float(np.mean(res[edge : x.size - edge])) / float(order)
    return f


def stationary_segment(
    audio: np.ndarray, sr: float, survey_rev_s: Sequence[float]
) -> dict[str, Any]:
    """The approved bench rule, per recording.

    Per rotor: pick the brightest order in 60-80 (by its margin over the
    neighbouring floor), refine that rotor's carrier onto the line, and read
    the residual frequency from a NARROW band and a WIDE band, both averaged
    over the microphones and then over :data:`BENCH_RESIDUAL_SMOOTH_S`.

    A window is stationary when, for EVERY rotor, the narrow residual is inside
    +-1 Hz and the wide residual is inside :data:`BENCH_WIDE_TOL_HZ` — the
    second test is what stops a carrier that drifts straight out of the narrow
    band from reading as perfectly steady filtered noise. A recording whose
    order carries no line at all (margin below :data:`BENCH_LINE_MARGIN_DB`)
    fails the rule outright, and a recording whose longest stationary span is
    shorter than :data:`BENCH_MIN_SEGMENT_S` fails it too, falling back to the
    most stationary window of the minimum length so the material stays usable
    with its failure on the record.
    """
    x = np.asarray(audio, dtype=np.float64)
    x = x[None, :] if x.ndim == 1 else x
    n = x.shape[-1]
    edge = int(round(BENCH_EDGE_S * sr))
    if n <= 2 * edge + int(round(BENCH_MIN_SEGMENT_S * sr)):
        raise ValueError(f"recording of {n / sr:.2f} s is too short for the bench rule")
    smooth = int(round(BENCH_RESIDUAL_SMOOTH_S * sr))
    inner = slice(edge, n - edge)

    orders: list[int] = []
    carriers: list[float] = []
    line_margin_db: list[float] = []
    narrow: list[np.ndarray] = []
    wide: list[np.ndarray] = []
    for f_survey in survey_rev_s:
        k, margin_db = select_order(x, sr, float(f_survey))
        f0 = refine_carrier(x[0], sr, float(f_survey), k)
        carrier = np.full(n, f0)
        res = []
        for band in (BENCH_DEMOD_BAND_HZ, BENCH_WIDE_BAND_FRAC * f0):
            z = demodulate(x, carrier, float(band), sr, order=k)
            mic_mean = np.mean(residual_frequency(z, sr, smooth_s=0.0), axis=0)
            res.append(_moving_mean(mic_mean, smooth)[inner])
        orders.append(k)
        carriers.append(f0)
        line_margin_db.append(margin_db)
        narrow.append(res[0])
        wide.append(res[1])

    worst = np.max(np.abs(np.stack(narrow)), axis=0)
    worst_wide = np.max(np.abs(np.stack(wide)), axis=0)
    line_present = min(line_margin_db) >= BENCH_LINE_MARGIN_DB
    inside = (worst <= BENCH_RESIDUAL_TOL_HZ) & (worst_wide <= BENCH_WIDE_TOL_HZ)
    a, b = _longest_run(inside)
    longest_inside_s = (b - a) / sr
    passed = bool(line_present and longest_inside_s >= BENCH_MIN_SEGMENT_S)
    if longest_inside_s < BENCH_MIN_SEGMENT_S:
        # Still give the recording a defensible window: the most stationary
        # span of the minimum length, i.e. the sliding BENCH_MIN_SEGMENT_S that
        # minimises the WORST residual in it, walked on a 0.1 s grid.
        m = min(int(round(BENCH_MIN_SEGMENT_S * sr)), worst.size)
        step = max(1, int(round(BENCH_FALLBACK_STEP_S * sr)))
        offsets = np.arange(0, worst.size - m + 1, step)
        a = int(offsets[int(np.argmin([worst[i : i + m].max() for i in offsets]))])
        b = a + m
    start = (edge + a) / sr
    stop = (edge + b) / sr
    keep = slice(a, b)
    return dict(
        passed=passed,
        line_present=bool(line_present),
        start_s=float(start),
        end_s=float(stop),
        duration_s=float(stop - start),
        longest_inside_s=float(longest_inside_s),
        frac_inside=float(np.mean(inside)),
        frac_narrow_ok=float(np.mean(worst <= BENCH_RESIDUAL_TOL_HZ)),
        frac_wide_ok=float(np.mean(worst_wide <= BENCH_WIDE_TOL_HZ)),
        orders=[int(k) for k in orders],
        line_margin_db=[float(v) for v in line_margin_db],
        carrier_rev_s=[float(v) for v in carriers],
        survey_rev_s=[float(v) for v in survey_rev_s],
        carrier_shift_rev_s=[float(c - s) for c, s in zip(carriers, survey_rev_s)],
        residual_std_hz=[float(np.std(r[keep])) for r in narrow],
        residual_mean_hz=[float(np.mean(r[keep])) for r in narrow],
        residual_max_abs_hz=float(np.max(worst[keep])),
        residual_wide_max_abs_hz=float(np.max(worst_wide[keep])),
        smooth_s=float(BENCH_RESIDUAL_SMOOTH_S),
        tol_hz=float(BENCH_RESIDUAL_TOL_HZ),
        wide_tol_hz=float(BENCH_WIDE_TOL_HZ),
        margin_tol_db=float(BENCH_LINE_MARGIN_DB),
        n_mics_averaged=int(x.shape[0]),
    )


# ── building supports ───────────────────────────────────────────────────────


def _bench_clip(
    audio: np.ndarray, sr: float, carriers: Sequence[float], *, clip_id: str, group: str
) -> Clip:
    """A native-rate clip whose rotor track is the bench's constant carrier."""
    audio = np.ascontiguousarray(np.asarray(audio, dtype=np.float32))
    rps = np.repeat(np.asarray(carriers, dtype=np.float64)[:, None], audio.shape[-1], axis=1)
    return Clip(clip_id, group, audio, rps, int(round(sr)), rps, {})


def _support_from_clip(
    clip: Clip,
    *,
    kind: str,
    name: str,
    n_fft: int,
    hop: int,
    segment: tuple[float, float],
    meta: dict[str, Any],
) -> Support:
    pg = RE.window_periodogram(clip, n_fft=int(n_fft), hop=int(hop))
    n_frames = int(pg.power.shape[1])
    starts = np.arange(n_frames, dtype=np.int64) * int(hop)
    return Support(
        kind=kind,
        name=name,
        sr=int(clip.sr),
        n_mics=int(clip.audio.shape[0]),
        freqs_hz=np.asarray(pg.freqs, dtype=np.float64),
        power=np.asarray(pg.power, dtype=np.float64),
        carrier_rev_s=np.asarray(pg.rps, dtype=np.float64),
        carrier_rev_s_audio=np.asarray(clip.rps, dtype=np.float64),
        frame_starts=starts,
        frame_centres_s=(starts + int(n_fft) / 2.0) / float(clip.sr),
        n_fft=int(n_fft),
        hop=int(hop),
        segment=(float(segment[0]), float(segment[1])),
        meta=meta,
    )


def bench_support(
    name: str,
    audio: np.ndarray,
    sr: float,
    carriers: Sequence[float],
    *,
    segment: tuple[float, float],
    meta: dict[str, Any] | None = None,
) -> Support:
    """The bench route: cut ``segment``, decimate to 16 kHz, ONE periodogram.

    Public because a self-check needs it: rendering a candidate on the same
    carrier and pushing the result through THIS function is the only way to
    compare a render against a bench support in the same units.
    ``segment`` is in seconds from the recording's first audio sample.
    """
    audio = np.asarray(audio)
    audio = audio[None, :] if audio.ndim == 1 else audio
    i0 = int(round(segment[0] * sr))
    i1 = int(round(segment[1] * sr))
    if not 0 <= i0 < i1 <= audio.shape[-1]:
        raise ValueError(
            f"{name}: segment {segment} -> samples [{i0}, {i1}) is not inside the "
            f"{audio.shape[-1]}-sample ({audio.shape[-1] / sr:.3f} s) recording"
        )
    clip = _bench_clip(audio[:, i0:i1], sr, carriers, clip_id=name, group="bench")
    clip = C.decimate(clip, SR)
    n = int(clip.audio.shape[-1])
    return _support_from_clip(
        clip,
        kind="bench",
        name=name,
        n_fft=n,
        hop=n,
        segment=segment,
        meta=dict(meta or {}, sr_native=int(round(sr))),
    )


def _dregon_bench_support(spec: SupportSpec, rec: C.Recording, point: dict[str, Any]) -> Support:
    survey = [float(v) for v in point["speed_rev_s"]]
    rule = stationary_segment(rec.audio, rec.sr, survey)
    meta = dict(
        recording_id=rec.recording_id,
        dataset=rec.dataset,
        dataset_version=rec.version,
        rig=point.get("rig"),
        corpus=point.get("corpus"),
        condition=point.get("condition"),
        throttle=point.get("throttle"),
        rotor_ids=[spec.args["rotor"]]
        if spec.args["rotor"] != "allMotors"
        else ["1", "2", "3", "4"],
        n_rotors=len(survey),
        rps_key=None,
        speed_source=point.get("speed_source"),
        stationarity=rule,
        stationary_pass=rule["passed"],
    )
    return bench_support(
        spec.name,
        rec.audio,
        rec.sr,
        rule["carrier_rev_s"],
        segment=(rule["start_s"], rule["end_s"]),
        meta=dict(meta, spec=spec.text, family=spec.family),
    )


def _bench_point_support(spec: SupportSpec, rec: C.Recording, point: dict[str, Any]) -> Support:
    """A published bench point: the WHOLE frame, already stationary (<= 30 s)."""
    audio = rec.audio if rec.audio.ndim > 1 else rec.audio[None, :]
    meta = dict(
        recording_id=rec.recording_id,
        dataset=rec.dataset,
        dataset_version=rec.version,
        rig=point.get("rig"),
        corpus=point.get("corpus"),
        source_id=point.get("source_id"),
        condition=point.get("condition"),
        throttle=point.get("throttle"),
        throttle_level=point.get("throttle_level"),
        rotor_ids=[str(i + 1) for i in range(len(point["speed_rev_s"]))],
        n_rotors=len(point["speed_rev_s"]),
        n_resolved=point.get("n_resolved"),
        multiplicity_unresolved=point.get("multiplicity_unresolved"),
        rps_key=None,
        speed_source=point.get("speed_source"),
        speed_tolerance_rev_s=point.get("speed_tolerance_rev_s"),
        margin_db=point.get("margin_db"),
        stationary_pass=True,
        stationarity=dict(
            rule="published stationary frame (manifest tolerance rule, <= 30 s)",
            publish_start_s=point.get("publish_start_s"),
            publish_s=point.get("publish_s"),
        ),
    )
    return bench_support(
        spec.name,
        audio,
        rec.sr,
        [float(v) for v in point["speed_rev_s"]],
        segment=(0.0, audio.shape[-1] / float(rec.sr)),
        meta=dict(meta, spec=spec.text, family=spec.family),
    )


def _flight_support(spec: SupportSpec) -> Support:
    """One flight window on the frozen front-end grid, all channels, 16 kHz."""
    args = spec.args
    clip = C.load_clip(
        args["dataset"],
        args["recording"],
        args["start_s"],
        args["dur_s"],
        version=None,
        channels=None,
        rps_key=args["rps_key"],
        clip_id=spec.name,
    )
    clip = C.decimate(clip, SR)
    meta = dict(
        recording_id=args["recording"],
        dataset=args["dataset"],
        dataset_version=clip.meta.get("dataset_version"),
        rps_key=args["rps_key"],
        group=clip.group,
        rotor_ids=[str(i + 1) for i in range(int(clip.rps.shape[0]))],
        n_rotors=int(clip.rps.shape[0]),
        throttle=None,
        sr_native=int(clip.meta.get("decimated_from", clip.sr)),
        spec=spec.text,
        family=spec.family,
        stationary_pass=True,
    )
    return _support_from_clip(
        clip,
        kind="flight",
        name=spec.name,
        n_fft=OBS_N_FFT,
        hop=OBS_HOP,
        segment=(float(args["start_s"]), float(args["start_s"]) + float(args["dur_s"])),
        meta=meta,
    )


def load_support(
    spec: SupportSpec | str, *, cache_dir: Path | None = None, use_cache: bool = True
) -> Support:
    """One support, from the ``.npz`` cache if it is there, else from the data."""
    spec = as_spec(spec)
    if use_cache:
        cached = load_cached(spec.name, cache_dir=cache_dir)
        if cached is not None:
            return cached
    if spec.family == "bench_dregon_motor":
        rec = C.load_recording(DREGON_DATASET, spec.args["recording_id"], None, DREGON_RPS_KEY)
        point = bench_manifest()[dregon_bench_point_key(spec.args["recording_id"])]
        return _dregon_bench_support(spec, rec, point)
    if spec.family == "bench_point":
        point = bench_manifest()[spec.args["point_id"]]
        rec = next(
            iter(C.iter_recordings(BENCH_POINT_DATASET, (spec.args["point_id"],), None, "auto"))
        )
        return _bench_point_support(spec, rec, point)
    return _flight_support(spec)


def iter_supports(specs: Iterable[SupportSpec | str]) -> Iterator[Support]:
    """Every support of ``specs``, reading each dataset at most once.

    The bench-point dataset is 135 multichannel recordings in one stream, so
    resolving them one at a time would decode it 135 times over; the flight
    specs are grouped per recording because :func:`clips.load_recording`
    caches only two.
    """
    specs = [as_spec(s) for s in specs]
    points = [s for s in specs if s.family == "bench_point"]
    others = [s for s in specs if s.family != "bench_point"]
    if points:
        manifest = bench_manifest()
        by_id = {s.args["point_id"]: s for s in points}
        for rec in C.iter_recordings(BENCH_POINT_DATASET, tuple(by_id), None, "auto"):
            spec = by_id[rec.recording_id]
            yield _bench_point_support(spec, rec, manifest[spec.args["point_id"]])
    for spec in sorted(
        others, key=lambda s: (s.family, s.args.get("recording_id") or s.args.get("recording", ""))
    ):
        yield load_support(spec, use_cache=False)


# ── the npz cache ───────────────────────────────────────────────────────────

_SCALARS = ("kind", "name", "sr", "n_mics", "n_fft", "hop")


def save_support(support: Support, *, out_dir: Path | None = None) -> Path:
    """Write ``<name>.npz``. ``power`` travels as float32 — it comes out of
    :func:`data.periodogram` as float32 anyway, and a 30 s whole-segment bench
    periodogram is 240k bins per mic."""
    out = Path(out_dir or CACHE_DIR)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{support.name}.npz"
    header = {k: getattr(support, k) for k in _SCALARS}
    header["segment"] = list(support.segment)
    header["meta"] = support.meta
    np.savez(
        path,
        power=support.power.astype(np.float32),
        freqs_hz=support.freqs_hz.astype(np.float64),
        carrier_rev_s=support.carrier_rev_s.astype(np.float64),
        carrier_rev_s_audio=support.carrier_rev_s_audio.astype(np.float32),
        frame_starts=support.frame_starts.astype(np.int64),
        frame_centres_s=support.frame_centres_s.astype(np.float64),
        header=json.dumps(header),
    )
    return path


def load_cached(name: str, *, cache_dir: Path | None = None) -> Support | None:
    path = Path(cache_dir or CACHE_DIR) / f"{name}.npz"
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        header = json.loads(str(z["header"]))
        return Support(
            kind=str(header["kind"]),
            name=str(header["name"]),
            sr=int(header["sr"]),
            n_mics=int(header["n_mics"]),
            freqs_hz=np.asarray(z["freqs_hz"], dtype=np.float64),
            power=np.asarray(z["power"], dtype=np.float64),
            carrier_rev_s=np.asarray(z["carrier_rev_s"], dtype=np.float64),
            carrier_rev_s_audio=np.asarray(z["carrier_rev_s_audio"], dtype=np.float64),
            frame_starts=np.asarray(z["frame_starts"], dtype=np.int64),
            frame_centres_s=np.asarray(z["frame_centres_s"], dtype=np.float64),
            n_fft=int(header["n_fft"]),
            hop=int(header["hop"]),
            segment=(float(header["segment"][0]), float(header["segment"][1])),
            meta=dict(header["meta"]),
        )


def index_row(support: Support) -> dict[str, Any]:
    """The ``index.json`` row: identity, geometry, carriers, stationarity."""
    st = support.meta.get("stationarity") or {}
    return dict(
        name=support.name,
        spec=support.meta.get("spec"),
        kind=support.kind,
        family=support.meta.get("family"),
        recording_id=support.meta.get("recording_id"),
        dataset=support.meta.get("dataset"),
        rps_key=support.meta.get("rps_key"),
        sr=support.sr,
        n_mics=support.n_mics,
        n_frames=support.n_frames,
        n_rotors=support.n_rotors,
        n_fft=support.n_fft,
        hop=support.hop,
        n_bins=int(support.freqs_hz.size),
        duration_s=round(support.duration_s, 6),
        segment=[support.segment[0], support.segment[1]],
        carriers_rev_s=[round(float(v), 6) for v in support.carrier_rev_s.mean(axis=1)],
        carrier_min_rev_s=[round(float(v), 6) for v in support.carrier_rev_s.min(axis=1)],
        carrier_max_rev_s=[round(float(v), 6) for v in support.carrier_rev_s.max(axis=1)],
        throttle=support.meta.get("throttle"),
        rig=support.meta.get("rig"),
        rotor_ids=support.meta.get("rotor_ids"),
        stationary_pass=support.meta.get("stationary_pass"),
        residual_std_hz=(
            [round(v, 6) for v in st["residual_std_hz"]] if "residual_std_hz" in st else None
        ),
        residual_max_abs_hz=st.get("residual_max_abs_hz"),
        longest_inside_s=st.get("longest_inside_s"),
        orders=st.get("orders"),
        survey_rev_s=st.get("survey_rev_s"),
        carrier_shift_rev_s=(
            [round(v, 6) for v in st["carrier_shift_rev_s"]]
            if "carrier_shift_rev_s" in st
            else None
        ),
    )


# ── the named sets ──────────────────────────────────────────────────────────

#: The 20 DREGON single-motor cells plus the quad validation recording.
DREGON_BENCH_ROTORS = ("Motor1", "Motor2", "Motor3", "Motor4")
DREGON_BENCH_THROTTLES = (50, 60, 70, 80, 90)

#: FLY125's cruise windows, as ``results/S2/cruise_8clip_refined.json`` defines
#: them (``data.starts_s``, 16 s each); R1 takes the first 8 s of each, so the
#: eight supports are disjoint and share the legacy export's material.
MICHAELS_CRUISE_STARTS = (16.0, 32.0, 48.0, 64.0, 96.0, 112.0, 128.0, 144.0)
#: The frozen FLY124 evaluation supports (standby, standby, ramp, cruise,
#: cruise), 8 s each — ``scripts/noise_v2_likelihood_window.py:115-121``.
MICHAELS_FROZEN = (
    (8.0, "standby"),
    (16.0, "standby"),
    (27.68, "ramp"),
    (40.0, "cruise"),
    (56.0, "cruise"),
)

#: The five frozen DREGON room-2 scoring supports (4 s, all mics) —
#: ``scripts/noise_v2_likelihood_window.py:105-111``.
DREGON_SCORED = (
    ("free-flight_nosource_room2", 1512727397.2050455),
    ("hovering_nosource_room2", 1511903905.3944898),
    ("updown_nosource_room2", 1511903578.348311),
    ("rectangle_nosource_room2", 1511905725.952559),
    ("spinning_nosource_room2", 1511905200.978012),
)
DREGON_SCORED_DUR_S = 4.0
#: The floor-fit segments: 8 s of the SAME five recordings, starting after the
#: scored window plus a guard, so no floor parameter is fitted on scored
#: material. ``scripts/noise_v2_likelihood_window.py`` uses the same 2 s guard.
DREGON_FLOOR_GUARD_S = 2.0
DREGON_FLOOR_DUR_S = 8.0


def set_dregon_bench() -> list[SupportSpec]:
    specs = [
        bench_dregon_motor(rotor, throttle)
        for rotor in DREGON_BENCH_ROTORS
        for throttle in DREGON_BENCH_THROTTLES
    ]
    specs.append(bench_dregon_motor("allMotors", 70))
    return specs


def set_bench_points() -> list[SupportSpec]:
    return [bench_point(key) for key in sorted(bench_manifest())]


def set_michaels_cruise() -> list[SupportSpec]:
    specs = [flight_michaels("FLY125", start, 8.0) for start in MICHAELS_CRUISE_STARTS]
    specs += [flight_michaels("FLY124", start, 8.0) for start, _ in MICHAELS_FROZEN]
    return specs


def set_dregon_floor() -> list[SupportSpec]:
    specs = [flight_dregon(rec, start, DREGON_SCORED_DUR_S) for rec, start in DREGON_SCORED]
    specs += [
        flight_dregon(rec, start + DREGON_SCORED_DUR_S + DREGON_FLOOR_GUARD_S, DREGON_FLOOR_DUR_S)
        for rec, start in DREGON_SCORED
    ]
    return specs


SUPPORT_SETS: dict[str, Callable[[], list[SupportSpec]]] = {
    "dregon-bench": set_dregon_bench,
    "bench-points": set_bench_points,
    "michaels-cruise": set_michaels_cruise,
    "dregon-floor": set_dregon_floor,
}


def support_set(name: str) -> list[SupportSpec]:
    if name not in SUPPORT_SETS:
        raise KeyError(f"unknown support set {name!r}; have {sorted(SUPPORT_SETS)}")
    return SUPPORT_SETS[name]()


__all__ = [
    "BENCH_MIN_SEGMENT_S",
    "BENCH_RESIDUAL_TOL_HZ",
    "CACHE_DIR",
    "OBS_HOP",
    "OBS_N_FFT",
    "SR",
    "SUPPORT_SETS",
    "Support",
    "SupportSpec",
    "as_spec",
    "bench_dregon_motor",
    "bench_manifest",
    "bench_point",
    "bench_support",
    "block_periodogram",
    "dregon_bench_point_key",
    "flight_dregon",
    "flight_michaels",
    "index_row",
    "iter_supports",
    "line_margins",
    "load_cached",
    "load_support",
    "parse_spec",
    "refine_carrier",
    "save_support",
    "select_order",
    "stationary_segment",
    "support_set",
]
