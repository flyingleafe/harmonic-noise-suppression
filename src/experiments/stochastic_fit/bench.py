"""Per-rotor harmonic profiles and the speed law from DREGON's single-motor
bench recordings.

``Motor{1-4}_{50,60,70,80,90}`` each spin ONE rotor at a fixed setpoint in
front of the 8-mic array, 25-45 s at 44.1 kHz. Nothing moves, so the estimator
is the closed-form one derived in the explainer's Part V rather than a
Vold-Kalman decomposition:

* the rate comes from the proper-prior comb evidence (Part V §11.3), which
  charges every amplitude an Occam factor and therefore does not fall into the
  octave trap a harmonic-sum score falls into (a flat-prior score returns
  s/2 or s/3 on these recordings, every time);
* the per-order power is the BAND-INTEGRATED excess over the local floor with
  the derived ``2 sigma^2`` bias correction (§10.3), not a peak value: the
  lines are phase-broadened, and peak-picking loses 3 dB at the bottom of the
  comb and 6-8 dB above it;
* the same band gives each line's width, which on stationary data is a clean
  measurement of the width law with no flight or tracking confound;
* the five setpoints per rotor give the SPEED LAW, which is the parameter the
  4 s flight crops cannot identify at all (§4.1) and which DREGON currently
  inherits as ``q = q_fl = 2.5``.

What transfers to flight: the profile SHAPE, the per-rotor deviations, the
width law and the speed exponents. What does not: absolute level and floor
shape, since a motor on a bench has no inflow, a different loading and a
different room.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

#: The raw tree, kept as a fallback; the published frames are preferred.
BENCH_DIR = (
    Path.home()
    / ".cache/dload/materialized/DREGON/db39bcf762d0/DREGON_individual_motors_recordings"
)
MOTORS = ("Motor1", "Motor2", "Motor3", "Motor4")
SPEEDS = (50, 60, 70, 80, 90)
SR = 44100.0
#: Analysis length. 2**19 at 44.1 kHz = 11.9 s, comfortably inside the
#: shortest recording (25 s) and giving a 0.084 Hz bin.
N_ANALYSIS = 1 << 19
#: Zero-padding for the rate search only, so h_k(omega) is available off-bin.
N_PAD = 1 << 22
#: Orders read. At 49 rev/s the 100th order is 4.9 kHz; at 89 rev/s it is
#: 8.9 kHz, above which nothing is modelled anyway.
K_MAX = 110


@dataclass(frozen=True)
class BenchSpectrum:
    """One recording's comb, per microphone and order."""

    motor: str
    setpoint: int
    rate_rps: float
    #: (M, K) band-integrated line power over the local floor, dB. NaN above
    #: the last order that fits below Nyquist.
    line_db: np.ndarray
    #: (M, K) the local floor level in the same units, dB.
    floor_db: np.ndarray
    #: (K,) RMS line width about the line centre, Hz, microphone-averaged.
    width_hz: np.ndarray
    #: (M, K) peak margin over the local floor, dB — the detection statistic.
    margin_db: np.ndarray
    n_seconds: float
    #: Per order, the microphone-mean floor-subtracted band excess and its
    #: frequency offsets: ``{k: (f_off, excess)}``. Kept because a per-order
    #: width fitted on one line is far too noisy to regress (the planted
    #: control showed +-2x scatter); the LAW is fitted to every line at once.
    bands: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)


def _load_frames() -> dict[tuple[str, int], np.ndarray]:
    """``{(motor, setpoint): (M, T)}`` from the published DREGON frames."""
    from data_processing.frames import meta_dict
    from data_processing.streams import iter_published_frames

    out: dict[tuple[str, int], np.ndarray] = {}
    for frame in iter_published_frames("DREGON-frames"):
        meta = meta_dict(frame)
        if meta.get("split") != "motor":
            continue
        m = re.match(r"motor_(Motor\d)_(\d+)$", str(meta.get("recording_id")))
        if not m:
            continue  # allMotors_70: four rotors at once, not a per-rotor probe
        audio = np.asarray(frame["audio"].data, dtype=np.float64)
        out[(m.group(1), int(m.group(2)))] = audio
    return out


def _welch(x: np.ndarray, n_fft: int) -> np.ndarray:
    """Hann periodogram averaged over 50 %-overlapped segments, (F,) per call.

    No detrending and no scaling: every readout below is a RATIO against the
    local floor of the same spectrum, so the normalisation cancels.
    """
    win = np.hanning(n_fft + 1)[:n_fft]
    starts = range(0, max(x.size - n_fft, 0) + 1, n_fft // 2)
    acc = None
    n = 0
    for s in starts:
        p = np.abs(np.fft.rfft(x[s : s + n_fft] * win)) ** 2
        acc = p if acc is None else acc + p
        n += 1
    if acc is None:
        raise ValueError("segment shorter than one analysis window")
    return acc / n


def comb_evidence(
    psd: np.ndarray, df: float, rate: float, k_max: int, *, delta2: float, sigma2: float
) -> float:
    """``log p(D | rate)`` with a proper N(0, delta^2) prior on every amplitude.

    Part V §11.3: with orthonormal model functions each amplitude contributes
    ``-0.5 log(1 + delta^2/sigma^2)`` whether the data want it or not, plus
    ``h^2 delta^2 / (2 sigma^2 (sigma^2 + delta^2))``. An order whose line is
    absent contributes only the penalty, so a half-rate comb pays for its
    empty orders instead of getting them free.
    """
    ks = np.arange(1, k_max + 1)
    idx = np.round(ks * rate / df).astype(int)
    idx = idx[idx < psd.size - 2]
    if idx.size < 8:
        return -np.inf
    h2 = np.maximum(psd[idx - 1], np.maximum(psd[idx], psd[idx + 1]))
    occam = -0.5 * (2 * idx.size) * np.log1p(delta2 / sigma2)
    fit = float(h2.sum()) * delta2 / (2.0 * sigma2 * (sigma2 + delta2))
    return occam + fit


def estimate_rate(psd: np.ndarray, df: float, *, lo: float = 20.0, hi: float = 120.0) -> float:
    """The shaft rate by maximising the proper-prior comb evidence."""
    sigma2 = float(np.median(psd))
    delta2 = float(np.percentile(psd, 99.9))
    grid = np.arange(lo, hi, 0.02)
    scores = [comb_evidence(psd, df, float(s), 40, delta2=delta2, sigma2=sigma2) for s in grid]
    coarse = float(grid[int(np.argmax(scores))])
    fine = np.arange(coarse - 0.05, coarse + 0.05, 0.0005)
    scores = [comb_evidence(psd, df, float(s), 80, delta2=delta2, sigma2=sigma2) for s in fine]
    return float(fine[int(np.argmax(scores))])


_WINDOW_RMS: dict[tuple[int, int], float] = {}


def _window_rms_hz(n_fft: int, df: float, span: int) -> float:
    """RMS width of the Hann window's own power spectrum, in Hz.

    A second moment read off a periodogram measures the line convolved with
    the analysis window, so a line narrower than the window reports the
    WINDOW's width — constant in k, which flattens any width law fitted at low
    orders (it is what the planted control caught, and it is the origin of the
    0.3-0.4 Hz intercept the real measurements show). The window's own moment
    is a property of the geometry, so it is computed once and removed in
    quadrature.
    """
    key = (int(n_fft), int(span))
    if key in _WINDOW_RMS:
        return _WINDOW_RMS[key] * df
    w = np.hanning(n_fft + 1)[:n_fft]
    spec = np.abs(np.fft.fftshift(np.fft.fft(w, n_fft))) ** 2
    centre = n_fft // 2
    lo, hi = centre - span, centre + span + 1
    band = spec[lo:hi]
    bins = np.arange(lo, hi) - centre
    rms_bins = float(np.sqrt((band * bins**2).sum() / band.sum()))
    _WINDOW_RMS[key] = rms_bins
    return rms_bins * df


def _local_floor(psd: np.ndarray, centre: float, half: float) -> float:
    a = psd[int(round(centre + 0.25 * half)) : int(round(centre + 0.75 * half))]
    b = psd[int(round(centre - 0.75 * half)) : int(round(centre - 0.25 * half))]
    if not a.size or not b.size:
        return float("nan")
    return float(np.median(np.concatenate([a, b])))


def _fit_line_width(
    excess: np.ndarray, f_off: np.ndarray, df: float, n_fft: int, *, shape: str = "lorentz"
) -> float:
    """Half width at half maximum of one line, by fitting its SHAPE.

    Neither of the cheap statistics survives contact with a real periodogram:
    a second moment is tail-dominated for a Lorentzian and grows with the
    integration window (a planted 0.05 Hz/order law came back as 0.106), and a
    half-power crossing keys off the single loudest bin, which for a broadened
    line is a noise spike and biases the width DOWN (planted 1.8 Hz came back
    as 0.95). Fitting the shape uses every bin in the band, which is what the
    derivation in the explainer's §11.4 recommends.

    The amplitude profiles out analytically, so the fit is a one-dimensional
    scan over ``gamma``: for each candidate the best amplitude is
    ``sum(e L) / sum(L^2)`` and the residual follows. The analysis window's own
    half width is then removed in quadrature.
    """
    total = float(excess.sum())
    if total <= 0 or excess.size < 5:
        return float("nan")
    grid = np.geomspace(0.2 * df, max(f_off[-1], 2.0 * df), 48)
    best, best_gamma = np.inf, float("nan")
    for gamma in grid:
        if shape == "gauss":
            model = np.exp(-np.log(2.0) * (f_off / gamma) ** 2)
        else:
            model = 1.0 / (1.0 + (f_off / gamma) ** 2)
        denom = float((model * model).sum())
        if denom <= 0:
            continue
        amp = float((excess * model).sum() / denom)
        resid = float(((excess - amp * model) ** 2).sum())
        if resid < best:
            best, best_gamma = resid, float(gamma)
    if not np.isfinite(best_gamma):
        return float("nan")
    win = _window_hwhm_hz(n_fft, df)
    return float(np.sqrt(max(best_gamma**2 - win**2, 0.0)))


_WINDOW_HWHM: dict[int, float] = {}


def _window_hwhm_hz(n_fft: int, df: float) -> float:
    """The Hann analysis window's own half-power half width, in Hz."""
    if n_fft not in _WINDOW_HWHM:
        w = np.hanning(n_fft + 1)[:n_fft]
        spec = np.abs(np.fft.fftshift(np.fft.fft(w, 16 * n_fft))) ** 2
        centre = 8 * n_fft
        band = spec[centre : centre + 16 * 4]
        half = band[0] / 2.0
        idx = int(np.argmax(band < half))
        # sub-bin interpolation, in units of the 1/16-bin fine grid
        y0, y1 = band[idx - 1], band[idx]
        frac = (y0 - half) / (y0 - y1) if y1 != y0 else 0.0
        _WINDOW_HWHM[n_fft] = (idx - 1 + frac) / 16.0
    return _WINDOW_HWHM[n_fft] * df


def read_orders(
    psd_mic: np.ndarray, df: float, rate: float, k_max: int = K_MAX
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[int, tuple[np.ndarray, np.ndarray]]
]:
    """Per order: band-integrated line power, local floor, width, peak margin.

    The integration band is +-3 times the width law's own prediction, floored
    at 4 bins, so a broadened line is not cut off and a narrow one does not
    collect a hundred bins of floor variance. The floor is subtracted per bin
    WITHOUT clipping, so its fluctuations cancel instead of accumulating.
    """
    n_mic = psd_mic.shape[0]
    line = np.full((n_mic, k_max), np.nan)
    floor = np.full((n_mic, k_max), np.nan)
    margin = np.full((n_mic, k_max), np.nan)
    width = np.full(k_max, np.nan)
    bands: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    half = rate / df / 2.0
    for i, k in enumerate(range(1, k_max + 1)):
        c0 = k * rate / df
        span = max(4, int(round(3.0 * (0.4 + 0.23 * k) / df)))
        lo, hi = int(round(c0)) - span, int(round(c0)) + span + 1
        if hi >= psd_mic.shape[1] - 2 or lo < 1:
            break
        f_off = (np.arange(lo, hi) - c0) * df
        for m in range(n_mic):
            fl = _local_floor(psd_mic[m], c0, half)
            if not np.isfinite(fl):
                continue
            excess = float((psd_mic[m, lo:hi] - fl).sum())
            floor[m, i] = 10.0 * np.log10(max(fl, 1e-30))
            # A non-positive band excess means the line is below its own floor
            # in this band: that is an absent measurement, not a -300 dB line.
            line[m, i] = 10.0 * np.log10(excess) if excess > 0.0 else np.nan
            peak = float(psd_mic[m, int(round(c0)) - 2 : int(round(c0)) + 3].max())
            margin[m, i] = 10.0 * np.log10(max(peak / fl, 1e-30))
        mean_excess = np.clip(
            psd_mic[:, lo:hi].mean(axis=0) - np.nanmean(10 ** (floor[:, i] / 10)), 0.0, None
        )
        width[i] = _fit_line_width(mean_excess, f_off, df, 2 * (psd_mic.shape[1] - 1))
        bands[k] = (f_off.copy(), mean_excess.copy())
    return line, floor, width, margin, bands


def bench_spectra(
    recordings: dict[tuple[str, int], np.ndarray] | None = None,
    *,
    sr: float = SR,
    n_analysis: int = N_ANALYSIS,
    k_max: int = K_MAX,
) -> list[BenchSpectrum]:
    """One :class:`BenchSpectrum` per single-motor recording.

    The analysis geometry is explicit rather than read off module constants,
    so a planted control can run this same code at a lower sample rate and a
    shorter window without patching globals.
    """
    recs = _load_frames() if recordings is None else recordings
    out: list[BenchSpectrum] = []
    for (motor, setpoint), audio in sorted(recs.items()):
        n = min(audio.shape[1], n_analysis * 2)
        seg = audio[:, :n] - audio[:, :n].mean(axis=1, keepdims=True)
        psd_mic = np.stack([_welch(seg[m], n_analysis) for m in range(seg.shape[0])])
        df = sr / n_analysis
        rate = estimate_rate(psd_mic.mean(axis=0), df)
        line, floor, width, margin, bands = read_orders(psd_mic, df, rate, k_max)
        out.append(
            BenchSpectrum(
                motor=motor,
                setpoint=setpoint,
                rate_rps=rate,
                line_db=line,
                floor_db=floor,
                width_hz=width,
                margin_db=margin,
                n_seconds=n / sr,
                bands=bands,
            )
        )
    return out


# =============================================================================
# The three quantities DREGON's flight fit could not identify
# =============================================================================

#: A line is used only where it is actually detected. 10 dB of peak margin is
#: the bar the gate's topology features use, and it is right for the SPEED law
#: (which needs the same cell at five speeds). The PROFILE bar is lower,
#: because a band-integrated excess is unbiased wherever it is positive and
#: the high orders of this rig sit at 6-10 dB — insisting on 10 dB there would
#: throw away half the comb and leave the flight fit's flat hold in place.
MARGIN_MIN_DB = 10.0
PROFILE_MARGIN_MIN_DB = 6.0
#: The width is a shape fit over a whole band, so it needs a line that stands
#: clear of the floor across that band, not merely a detectable peak.
WIDTH_MARGIN_MIN_DB = 18.0


@dataclass
class BenchFit:
    profile_db: np.ndarray  # (K,) rig mean, relative to order 2
    rotor_delta_db: np.ndarray  # (R, K) per-rotor deviation
    profile_n: np.ndarray  # (K,) how many (rotor, speed, mic) cells fed each order
    width_law: dict[str, Any] = field(default_factory=dict)
    speed_law: dict[str, Any] = field(default_factory=dict)
    rates: dict[str, float] = field(default_factory=dict)


def _nanmedian(x: np.ndarray, *, axis: int, size: int) -> np.ndarray:
    """``np.nanmedian`` that returns NaN for an all-NaN slice without warning.

    Orders above a recording's own detection limit have no cells at all, which
    is information (the comb ends there), not an error.
    """
    if x.size == 0:
        return np.full(size, np.nan)
    out = np.full(size, np.nan)
    finite = np.isfinite(x)
    for i in range(x.shape[1 - axis] if x.ndim == 2 else 0):
        col = x[:, i] if axis == 0 else x[i]
        m = finite[:, i] if axis == 0 else finite[i]
        if m.any():
            out[i] = float(np.median(col[m]))
    return out


def _centre(line_db: np.ndarray, margin_db: np.ndarray, k_ref: int) -> np.ndarray:
    """Remove each microphone's own level, so what is left is timbre.

    The reference is the order-``k_ref`` line of that microphone; cells below
    the detection bar become NaN.
    """
    x = np.where(margin_db >= PROFILE_MARGIN_MIN_DB, line_db, np.nan)
    ref = x[:, k_ref - 1 : k_ref]
    return x - ref


def fit_profile(spectra: list[BenchSpectrum], k_ref: int = 2) -> BenchFit:
    """Rig profile and per-rotor deviations, pooled over speeds and mics."""
    k_max = spectra[0].line_db.shape[1]
    per_rotor: dict[str, list[np.ndarray]] = {m: [] for m in MOTORS}
    for s in spectra:
        per_rotor[s.motor].append(_centre(s.line_db, s.margin_db, k_ref))
    rotor_curves = []
    for motor in MOTORS:
        stack = (
            np.concatenate(per_rotor[motor], axis=0) if per_rotor[motor] else np.zeros((0, k_max))
        )
        rotor_curves.append(_nanmedian(stack, axis=0, size=k_max))
    rotor = np.vstack(rotor_curves)
    mean = _nanmedian(rotor, axis=0, size=k_max)
    counts = np.sum(
        [np.sum(np.isfinite(_centre(s.line_db, s.margin_db, k_ref)), axis=0) for s in spectra],
        axis=0,
    )
    return BenchFit(
        profile_db=mean,
        rotor_delta_db=rotor - mean[None, :],
        profile_n=np.asarray(counts, dtype=float),
        rates={f"{s.motor}_{s.setpoint}": round(s.rate_rps, 4) for s in spectra},
    )


def fit_width_law(spectra: list[BenchSpectrum], k_hi: int = 24) -> dict[str, Any]:
    """``gamma = gamma0 + c k`` and the log-log slope, over detected orders.

    Restricted to ``k <= k_hi`` and to orders whose microphone-median margin
    clears the bar: the second moment of a wide window is floor-dominated
    otherwise, which is what made a first pass at this measurement return an
    exponent of 0.5.
    """
    ks, ws = [], []
    for s in spectra:
        med = np.nanmedian(s.margin_db, axis=0)
        for i in range(min(k_hi, s.width_hz.size)):
            # A width that deconvolves to zero means "narrower than the
            # analysis window": it carries no information about the law and
            # cannot enter a log-log fit at all.
            if np.isfinite(s.width_hz[i]) and s.width_hz[i] > 0.0 and med[i] >= WIDTH_MARGIN_MIN_DB:
                ks.append(i + 1.0)
                ws.append(s.width_hz[i])
    if len(ks) < 12:
        return {}
    k = np.asarray(ks)
    w = np.asarray(ws)
    slope = float(np.polyfit(np.log(k), np.log(w), 1)[0])
    c, g0 = (float(v) for v in np.polyfit(k, w, 1))
    return dict(
        n_points=len(ks),
        loglog_slope=round(slope, 3),
        gamma0_hz=round(g0, 3),
        gamma_slope_hz_per_order=round(c, 4),
        implied_sigma_rps=round(c / np.sqrt(2.0 * np.log(2.0)), 4),
        k_hi=k_hi,
    )


def fit_width_law_joint(
    spectra: list[BenchSpectrum],
    *,
    k_hi: int = 32,
    shape: str = "lorentz",
) -> dict[str, Any]:
    """``gamma_k = gamma0 + c k`` fitted to every line of every recording at once.

    One line of one recording pins its own width only to within a factor of
    two (the planted control), because a broadened line's periodogram is
    chi-squared with as many looks as the Welch average has segments. The law
    has two parameters and hundreds of lines, so it is identified far better
    than any single line: for a candidate ``(gamma0, c)`` each line's amplitude
    profiles out analytically, and the objective is the total residual over
    all lines. That is the same profiling trick §10.3 uses for the amplitudes,
    applied to the width law.
    """
    items: list[tuple[np.ndarray, np.ndarray]] = []
    for s in spectra:
        med = np.nanmedian(s.margin_db, axis=0)
        for k, (f_off, excess) in s.bands.items():
            usable = k <= k_hi and k - 1 < med.size and med[k - 1] >= WIDTH_MARGIN_MIN_DB
            if usable and excess.sum() > 0:
                items.append((f_off / max(k, 1), excess))  # scale to k=1 units
    if len(items) < 20:
        return {}
    # gamma_k = gamma0 + c k  =>  in k-scaled offsets the width is gamma0/k + c,
    # so the scan is over (gamma0, c) with the per-line k restored below.
    ks = np.array(
        [k for s in spectra for k in s.bands if k <= k_hi and s.bands[k][1].sum() > 0], dtype=float
    )
    ks = ks[: len(items)]
    g0_grid = np.geomspace(1e-3, 5.0, 24)
    c_grid = np.geomspace(1e-3, 2.0, 40)
    best = (np.inf, np.nan, np.nan)
    for g0 in g0_grid:
        for c in c_grid:
            resid = 0.0
            for (f_scaled, excess), k in zip(items, ks, strict=True):
                gamma = (g0 + c * k) / k  # in the k-scaled offset units
                if shape == "gauss":
                    model = np.exp(-np.log(2.0) * (f_scaled / gamma) ** 2)
                else:
                    model = 1.0 / (1.0 + (f_scaled / gamma) ** 2)
                denom = float((model * model).sum())
                if denom <= 0:
                    continue
                amp = float((excess * model).sum() / denom)
                # normalise per line so a loud low order does not dominate
                scale = float((excess**2).sum()) or 1.0
                resid += float(((excess - amp * model) ** 2).sum()) / scale
            if resid < best[0]:
                best = (resid, float(g0), float(c))
    _, g0, c = best
    return dict(
        n_lines=len(items),
        gamma0_hz=round(g0, 4),
        gamma_slope_hz_per_order=round(c, 4),
        implied_sigma_rps=round(c / np.sqrt(2.0 * np.log(2.0)), 4),
        k_hi=k_hi,
        shape=shape,
    )


def fit_speed_law(spectra: list[BenchSpectrum], k_hi: int = 64) -> dict[str, Any]:
    """The line and floor speed exponents, with a free intercept per cell.

    Five setpoints per rotor span 49-89 rev/s, i.e. 2.6 dB of
    ``10 log10 s`` — a real lever, which a 4 s flight crop does not have. The
    regression is

        y_{rotor,mic,k,speed} = alpha_{rotor,mic,k} + q * 10 log10 s + e ,

    so ``q`` is identified purely by how each cell moves with speed, and the
    per-cell level (which carries the microphone, the timbre and the bench
    geometry) is absorbed. The standard error is clustered on the cell,
    because the five speeds of one cell share whatever the model gets wrong
    about it. The floor exponent is the same regression on the local floor.
    """

    def regress(get: Any) -> tuple[float, float, int]:
        rows: dict[tuple[str, int, int], list[tuple[float, float]]] = {}
        for s in spectra:
            values = get(s)
            med = np.nanmedian(s.margin_db, axis=0)
            x = 10.0 * np.log10(s.rate_rps)
            for m in range(values.shape[0]):
                for i in range(min(k_hi, values.shape[1])):
                    if np.isfinite(values[m, i]) and med[i] >= MARGIN_MIN_DB:
                        rows.setdefault((s.motor, m, i), []).append((x, float(values[m, i])))
        cells = [v for v in rows.values() if len(v) >= 4]
        if len(cells) < 20:
            return float("nan"), float("nan"), 0
        # Within-cell centring removes alpha exactly, leaving one slope.
        num = den = 0.0
        per_cell = []
        for cell in cells:
            xs = np.array([a for a, _ in cell])
            ys = np.array([b for _, b in cell])
            xc, yc = xs - xs.mean(), ys - ys.mean()
            num += float((xc * yc).sum())
            den += float((xc * xc).sum())
            if (xc**2).sum() > 0:
                per_cell.append(float((xc * yc).sum() / (xc * xc).sum()))
        q = num / den
        # Cluster-robust SE: the spread of per-cell slopes over sqrt(n_cells).
        arr = np.asarray(per_cell)
        se = float(arr.std(ddof=1) / np.sqrt(arr.size)) if arr.size > 1 else float("nan")
        return q, se, len(cells)

    q_line, se_line, n_line = regress(lambda s: s.line_db)
    q_floor, se_floor, n_floor = regress(lambda s: s.floor_db)
    return dict(
        line_exponent=round(q_line, 4),
        line_exponent_se=round(se_line, 4),
        line_cells=n_line,
        floor_exponent=round(q_floor, 4),
        floor_exponent_se=round(se_floor, 4),
        floor_cells=n_floor,
        k_hi=k_hi,
    )


def fit_bench(spectra: list[BenchSpectrum] | None = None) -> BenchFit:
    """The whole bench instrument: profile, rotor deviations, width, speed."""
    spec = bench_spectra() if spectra is None else spectra
    fit = fit_profile(spec)
    fit.width_law = fit_width_law_joint(spec) or fit_width_law(spec)
    fit.speed_law = fit_speed_law(spec)
    return fit


def bench_summary(fit: BenchFit) -> dict[str, Any]:
    """JSON-able view of a :class:`BenchFit`."""
    return dict(
        source="DREGON single-motor bench recordings",
        n_orders=int(fit.profile_db.size),
        reference_order=2,
        profile_db=[None if not np.isfinite(v) else round(float(v), 4) for v in fit.profile_db],
        rotor_delta_db=[
            [None if not np.isfinite(v) else round(float(v), 4) for v in row]
            for row in fit.rotor_delta_db
        ],
        profile_cells=[int(v) for v in fit.profile_n],
        width_law=fit.width_law,
        speed_law=fit.speed_law,
        rates_rps=fit.rates,
    )


# =============================================================================
# Merging the bench measurements into a rig summary
# =============================================================================


def apply_bench(
    summary: dict[str, Any], fit: BenchFit, *, support: int | None = None
) -> dict[str, Any]:
    """Return ``summary`` with the three quantities the bench identifies.

    1. **The profile above the flight fit's measured support.** The rig fit
       sizes its harmonic ladder to the fastest clips, so its top orders carry
       no likelihood term and fall to the prior (DREGON's mean collapses 126 dB
       above order 91); the renderer therefore truncates and holds. The bench
       measures those orders directly — at 89 rev/s its 100th order is 8.9 kHz
       — so the hold is replaced by the measured shape, level-matched on the
       last 12 orders both instruments share.
    2. **Both speed exponents.** Free-intercept regressions over five
       setpoints, which 4 s flight crops cannot do at all.
    3. **The line width.** The bench's ``gamma = gamma0 + c k`` is measured
       where nothing moves, so it carries no tracking error; only the slope
       transfers (the intercept is the analysis window's, per §2.2).

    Everything else — levels, floor shape, microphone structure, the amplitude
    process — stays as the flight fit left it, because a bench motor has no
    inflow and a different room.
    """
    from experiments.stochastic_fit.raw_predictive import measured_profile_support

    out = {k: v for k, v in summary.items()}
    rig = dict(out["rig"])
    flight = np.asarray(rig["profile_db"], dtype=np.float64)
    sup = int(measured_profile_support(flight.tolist()) if support is None else support)
    bench_profile = np.asarray(fit.profile_db, dtype=np.float64)

    n = min(flight.size, bench_profile.size)
    overlap = slice(max(sup - 12, 1), min(sup, n))
    both = np.isfinite(bench_profile[overlap]) & np.isfinite(flight[overlap])
    if both.any():
        shift = float(np.median(flight[overlap][both] - bench_profile[overlap][both]))
        extended = flight.copy()
        for i in range(sup, flight.size):
            if i < bench_profile.size and np.isfinite(bench_profile[i]):
                extended[i] = bench_profile[i] + shift
        rig["profile_db"] = extended.tolist()
        out["bench_profile_shift_db"] = round(shift, 3)
        out["bench_profile_orders_replaced"] = int(
            sum(
                1
                for i in range(sup, flight.size)
                if i < bench_profile.size and np.isfinite(bench_profile[i])
            )
        )

    speed = fit.speed_law
    if speed and np.isfinite(speed.get("line_exponent", np.nan)):
        rig["amp_exp"] = float(speed["line_exponent"])
        rig["floor_exp"] = float(speed["floor_exponent"])
        out["bench_speed_law"] = speed

    width = fit.width_law
    if width:
        # Only the SLOPE transfers: the renderer's quasi-static mapping is
        # sigma = w_rho * gamma_slope / sqrt(2 ln 2), and gamma0 belongs to the
        # analysis window, not to the shaft.
        rig["gamma_slope"] = float(width["gamma_slope_hz_per_order"])
        out["bench_width_law"] = width

    out["rig"] = rig
    out["bench_applied"] = True
    return out
