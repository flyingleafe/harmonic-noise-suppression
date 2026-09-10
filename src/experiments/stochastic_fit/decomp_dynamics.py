"""Amplitude dynamics fitted from decomposed Vold-Kalman envelopes.

The dense-comb Whittle fit cannot see the line-level amplitude process: its
drift GP is a *prior* with a fixed standard deviation and correlation time
(``Spec.gp_std_db`` / ``gp_tau_s``), integrated out rather than estimated. The
decomposed envelopes measure it directly — one amplitude series per
``(rotor, harmonic)`` against the exact carrier the solve used — so the
renderer's amplitude process is fitted here.

What is identifiable, and what is not:

* the **speed law** ``power ~ rps**p`` — a regression of envelope dB on
  ``10 log10 rps`` with per-``(rotor, order)`` intercepts, with a
  cluster-robust standard error (clusters = time chunks, because the residual
  is strongly autocorrelated);
* the **slow amplitude structure** — the residual's spectral density inside a
  band that the Vold-Kalman analysis passes untouched. The decomposition's
  bandwidth is ``bw_rps * k`` Hz, so orders ``k >= k_min`` are unfiltered up to
  ``band_hz[1]``; nothing above that band is measurable from envelopes and the
  fit does not pretend otherwise;
* the **cross-order coherence per component** — the rotor-common share of the
  variance, which the renderer draws as ``harm_coherence``. It is estimated
  frequency-resolved and then projected onto the fitted components, because the
  slow structure is markedly more coherent across orders than the fast one.

A flat (white) term is carried in the model to absorb envelope estimation
noise. It is *not* transferred to the renderer: it is an artefact of the
analysis, not a property of the rotor.

Model comparison over the number of components uses held-out Whittle score on
time-disjoint segments plus the one-standard-error rule, as everywhere else in
this campaign.
"""

from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize

#: Envelope orders below this are dropped: the Vold-Kalman bandwidth of order
#: ``k`` is ``bw_rps * k`` Hz, so the lowest orders are smoothed inside the
#: analysis band and would bias every timescale upwards.
K_MIN = 4
#: The band the analysis passes for every retained order, in Hz.
BAND_HZ = (0.03, 1.5)


@dataclass(frozen=True)
class DynamicsComponent:
    std_db: float
    tau_s: float
    coherence: float


@dataclass(frozen=True)
class DecompDynamics:
    """The fitted amplitude process of one rig."""

    floor: dict[str, Any] | None
    speed_exponent: float
    speed_exponent_se: float
    components: list[dict[str, float]]
    white_floor_db2_per_hz: float
    selected_components: int
    component_valid_whittle: list[float]
    component_valid_whittle_se: list[float]
    component_fold_scores: list[list[float]]
    residual_std_db: list[float]
    band_hz: list[float]
    env_rate_hz: float
    k_min: int
    n_train_segments: int
    n_valid_segments: int
    recording_id: str


def envelope_arrays(
    frame: Any, *, k_max: int = 64, min_rps: float = 30.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """``(y, x, valid, env_rate)`` — mic-incoherent line power in dB and log speed."""
    amp = np.asarray(frame["amp"].data, dtype=np.float64)  # (M, R, K, T)
    amp_valid = np.asarray(frame["amp_valid"].data, dtype=bool)  # (R, K, T)
    rps = np.asarray(frame["rps"].data, dtype=np.float64)  # (R, n_samples)
    n_time = amp.shape[3]
    k = min(int(k_max), amp.shape[2])
    stride = int(round(rps.shape[-1] / n_time))
    sample_rate = float(frame["rps"].tindex.rate)
    rps_env = rps[:, : n_time * stride].reshape(rps.shape[0], n_time, stride).mean(axis=2)
    power = np.square(amp[:, :, :k, :]).mean(axis=0)  # (R, K, T)
    valid = amp_valid[:, :k, :] & (rps_env[:, None, :] >= min_rps) & (power > 0.0)
    y = np.where(valid, 10.0 * np.log10(np.maximum(power, 1e-30)), np.nan)
    x = 10.0 * np.log10(np.maximum(rps_env, 1e-9))
    return y, x, valid, sample_rate / stride


def fit_speed_law(
    y: np.ndarray, x: np.ndarray, valid: np.ndarray, *, chunk: int, min_frames: int = 50
) -> tuple[float, float, np.ndarray]:
    """Common ``dB per dB rps`` slope with per-``(rotor, order)`` intercepts.

    The standard error is clustered on time chunks: the residual's correlation
    time is seconds, so treating frames as independent would understate it by
    more than an order of magnitude.
    """
    speeds = np.broadcast_to(x[:, None, :], y.shape)
    count = valid.sum(axis=2)
    usable = count > min_frames
    if not usable.any():
        raise ValueError("no (rotor, order) series has enough valid frames")
    mean_y = np.where(valid, y, 0.0).sum(axis=2) / np.maximum(count, 1)
    mean_x = np.where(valid, speeds, 0.0).sum(axis=2) / np.maximum(count, 1)
    keep = valid & usable[:, :, None]
    y_c = np.where(keep, y - mean_y[:, :, None], 0.0)
    x_c = np.where(keep, speeds - mean_x[:, :, None], 0.0)
    sxx = float((x_c * x_c).sum())
    if sxx <= 0.0:
        raise ValueError("the speed regressor has no within-series variation")
    slope = float((x_c * y_c).sum() / sxx)
    score = x_c * (y_c - slope * x_c)
    n_blocks = y.shape[2] // chunk
    if n_blocks < 2:
        raise ValueError("need at least two time chunks for a clustered standard error")
    block = np.asarray(
        [float(score[:, :, b * chunk : (b + 1) * chunk].sum()) for b in range(n_blocks)]
    )
    se = float(np.sqrt(np.square(block).sum()) / sxx * np.sqrt(n_blocks / (n_blocks - 1)))
    residual = np.where(
        keep, y - mean_y[:, :, None] - slope * (speeds - mean_x[:, :, None]), np.nan
    )
    return slope, se, residual


#: Octave-ish bands the residual floor law is pooled over.
FLOOR_BANDS = (
    (60.0, 120.0),
    (120.0, 250.0),
    (250.0, 500.0),
    (500.0, 1000.0),
    (1000.0, 2000.0),
    (2000.0, 4000.0),
    (4000.0, 7000.0),
)


def fit_floor_law(
    frame: Any,
    *,
    min_rps: float = 30.0,
    n_fft: int = 2048,
    hop: int = 512,
    reference_rps: float = 80.0,
    bands: tuple[tuple[float, float], ...] = FLOOR_BANDS,
    block_s: float = 20.0,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Speed law of the broadband floor, from the decomposition's residual.

    The Vold-Kalman residual is the recording with every tracked line removed,
    which is exactly what the renderer's floor stands for. The renderer's floor
    gain is ``(rps / ref)**q + s``: a rotor-driven part plus the recording
    chain's static share, so both are fitted jointly with one intercept per
    band. The uncertainty is a moving-block bootstrap, because consecutive
    frames of a flight are anything but independent.
    """
    residual = np.asarray(frame["residual"].data, dtype=np.float64)
    rps = np.asarray(frame["rps"].data, dtype=np.float64)
    window = np.hanning(n_fft + 1)[:n_fft]
    n_frames = 1 + (residual.shape[1] - n_fft) // hop
    starts = np.arange(n_frames) * hop
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(frame["rps"].tindex.rate))
    power = np.zeros((n_frames, freqs.size))
    for mic in range(residual.shape[0]):
        frames = np.stack([residual[mic, s : s + n_fft] for s in starts]) * window
        spectrum = np.fft.rfft(frames, axis=1)
        power += (spectrum.real**2 + spectrum.imag**2) / float(np.square(window).sum())
    power /= residual.shape[0]
    levels = np.stack(
        [
            10.0 * np.log10(np.maximum(power[:, (freqs >= lo) & (freqs < hi)].mean(axis=1), 1e-30))
            for lo, hi in bands
        ]
    )
    speed = np.stack(
        [np.mean([rps[r, s : s + n_fft] for s in starts], axis=1) for r in range(rps.shape[0])]
    ).mean(axis=0)
    keep = speed >= min_rps
    if keep.sum() < 100:
        raise ValueError("too few frames above the speed threshold for a floor law")

    def solve(index: np.ndarray) -> tuple[float, float, float]:
        ratio = speed[index] / reference_rps
        observed = levels[:, index]

        def objective(theta: np.ndarray) -> float:
            # (q, s) trade off along a ridge — a large static share can be paid
            # for by a steeper exponent — so both are bounded to the physically
            # meaningful range instead of left to wander during a bootstrap.
            exponent = float(np.clip(theta[0], 0.0, 8.0))
            static = float(np.clip(np.exp(theta[1]), 0.0, 1.0))
            gain = 10.0 * np.log10(np.maximum(ratio**exponent + static, 1e-12))
            resid = observed - gain[None, :]
            resid = resid - resid.mean(axis=1, keepdims=True)
            return float(np.mean(np.square(resid)))

        best = None
        for exponent in (1.0, 2.0, 3.0, 4.0):
            for log_static in (-8.0, -4.0, -1.0):
                result = minimize(
                    objective,
                    [exponent, log_static],
                    method="Nelder-Mead",
                    options=dict(maxiter=5000, fatol=1e-10, xatol=1e-8),
                )
                if best is None or result.fun < best.fun:
                    best = result
        assert best is not None
        return (
            float(np.clip(best.x[0], 0.0, 8.0)),
            float(np.clip(np.exp(best.x[1]), 0.0, 1.0)),
            float(np.sqrt(best.fun)),
        )

    index = np.flatnonzero(keep)
    exponent, static, rms = solve(index)
    rate = float(frame["rps"].tindex.rate) / hop
    block = max(int(round(block_s * rate)), 1)
    n_blocks = max(index.size // block, 2)
    rng = np.random.default_rng(seed)
    draws = [
        solve(
            np.concatenate(
                [
                    index[start : start + block]
                    for start in rng.integers(0, index.size - block, size=n_blocks)
                ]
            )
        )[0]
        for _ in range(n_bootstrap)
    ]
    log_speed = 10.0 * np.log10(speed[keep] / reference_rps)
    return dict(
        floor_exponent=exponent,
        floor_exponent_q05_q95=[float(value) for value in np.quantile(draws, (0.05, 0.95))],
        floor_static_rel=static,
        floor_rms_db=rms,
        band_slopes=[
            float(np.polyfit(log_speed, levels[band, keep], 1)[0])
            for band in range(levels.shape[0])
        ],
        n_frames=int(keep.sum()),
    )


def _fill_gaps(series: np.ndarray, max_gap: int) -> np.ndarray | None:
    finite = np.isfinite(series)
    if finite.all():
        return series
    if finite.mean() < 0.95:
        return None
    edges = np.diff(np.r_[0, (~finite).astype(np.int8), 0])
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    if starts.size and int((ends - starts).max()) > max_gap:
        return None
    out = series.copy()
    index = np.arange(series.size)
    out[~finite] = np.interp(index[~finite], index[finite], series[finite])
    return out


def _segment_spectra(
    series: np.ndarray, *, seg: int, stride: int, env_rate: float, max_gap: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hann periodograms of every complete segment: ``(spectra, freqs, t0)``."""
    window = np.hanning(seg + 1)[:seg]
    rows: list[np.ndarray] = []
    starts: list[int] = []
    flat = series.reshape(-1, series.shape[-1])
    for row in range(flat.shape[0]):
        for t0 in range(0, flat.shape[1] - seg + 1, stride):
            filled = _fill_gaps(flat[row, t0 : t0 + seg], max_gap)
            if filled is None:
                continue
            rows.append(filled)
            starts.append(t0)
    if not rows:
        raise ValueError("no complete envelope segment survived the validity mask")
    block = np.stack(rows)
    spectrum = np.fft.rfft((block - block.mean(axis=1, keepdims=True)) * window, axis=1)
    power = (spectrum.real**2 + spectrum.imag**2) / (env_rate * float(np.square(window).sum()))
    return power, np.fft.rfftfreq(seg, 1.0 / env_rate), np.asarray(starts)


def _ou_density(freqs: np.ndarray, std: float, tau: float) -> np.ndarray:
    return 2.0 * std * std * tau / (1.0 + np.square(2.0 * np.pi * freqs * tau))


def _unpack(
    theta: np.ndarray, tau_bounds: tuple[float, float]
) -> tuple[np.ndarray, np.ndarray, float]:
    """Positive parameters with the timescales clipped to what is resolvable.

    A segment of length ``L`` carries no information about a corner frequency
    below ``1 / (2 pi L)``; leaving ``tau`` free there lets the optimizer buy
    in-band fit by extrapolating an unmeasured Lorentzian tail, which inflates
    the component variance (on the planted control, by 40%). The cap keeps
    every reported variance inside the measured band.
    """
    # Nelder-Mead wanders freely in log space; clip before exponentiating so a
    # scouting step cannot turn the density into inf and poison the simplex.
    value = np.exp(np.clip(theta, -80.0, 80.0))
    return value[0:-1:2], np.clip(value[1:-1:2], *tau_bounds), float(value[-1])


def _model_density(
    theta: np.ndarray, freqs: np.ndarray, tau_bounds: tuple[float, float]
) -> np.ndarray:
    std, tau, white = _unpack(theta, tau_bounds)
    density = np.full_like(freqs, white)  # white envelope-estimation noise
    for j in range(std.size):
        density = density + _ou_density(freqs, std[j], tau[j])
    return density


def _whittle(
    theta: np.ndarray,
    freqs: np.ndarray,
    observed: np.ndarray,
    tau_bounds: tuple[float, float],
) -> float:
    density = _model_density(theta, freqs, tau_bounds)
    return float(np.mean(observed / density + np.log(density)))


def _fit_components(
    freqs: np.ndarray,
    train: np.ndarray,
    n_components: int,
    tau_bounds: tuple[float, float],
    *,
    restarts: int = 8,
    seed: int = 0,
) -> np.ndarray:
    """Whittle fit of ``n_components`` OU terms plus a white floor."""
    rng = np.random.default_rng(seed)
    best: np.ndarray | None = None
    best_score = np.inf
    scale = float(np.sqrt(np.maximum(np.trapezoid(train, freqs) * 2.0, 1e-6)))
    for _ in range(restarts):
        theta = np.empty(2 * n_components + 1)
        for j in range(n_components):
            theta[2 * j] = np.log(max(scale * float(rng.uniform(0.3, 1.2)), 1e-3))
            theta[2 * j + 1] = float(
                rng.uniform(np.log(tau_bounds[0] * 2.0), np.log(tau_bounds[1]))
            )
        theta[-1] = np.log(max(float(train[-5:].mean()) * 0.5, 1e-9))
        result = minimize(
            _whittle,
            theta,
            args=(freqs, train, tau_bounds),
            method="Nelder-Mead",
            options=dict(maxiter=40000, maxfev=40000, fatol=1e-12, xatol=1e-9),
        )
        if float(result.fun) < best_score:
            best_score, best = float(result.fun), np.asarray(result.x, dtype=np.float64)
    assert best is not None
    order = np.argsort(_unpack(best, tau_bounds)[1])[::-1]  # slowest component first
    packed = np.empty_like(best)
    for slot, source in enumerate(order):
        packed[2 * slot] = best[2 * int(source)]
        packed[2 * slot + 1] = best[2 * int(source) + 1]
    packed[-1] = best[-1]
    return packed


def _component_coherence(
    freqs: np.ndarray,
    theta: np.ndarray,
    tau_bounds: tuple[float, float],
    individual: np.ndarray,
    common: np.ndarray,
    n_orders: int,
) -> list[float]:
    """Project the frequency-resolved rotor-common share onto the components.

    With variance share ``rho`` shared by all orders of a rotor, the spectrum of
    the order-average is ``rho S + (1 - rho) S / K``, which inverts to a
    frequency-resolved ``rho(f)``. Least squares against the fitted component
    densities turns that curve into one coherence per component.
    """
    share = (common / np.maximum(individual, 1e-12) - 1.0 / n_orders) / (1.0 - 1.0 / n_orders)
    std, tau, white = _unpack(theta, tau_bounds)
    columns = np.stack([_ou_density(freqs, std[j], tau[j]) for j in range(std.size)], axis=1)
    total = columns.sum(axis=1) + white
    # weight each frequency by the components' share of the density: the white
    # term carries no coherence and must not drag the estimate down
    weight = np.sqrt(np.maximum(columns.sum(axis=1) / np.maximum(total, 1e-12), 0.0))
    design = columns / np.maximum(total, 1e-12)[:, None] * weight[:, None]
    target = share * weight
    solution, *_ = np.linalg.lstsq(design, target, rcond=None)
    return [float(np.clip(value, 0.0, 1.0)) for value in solution]


def fit_decomp_dynamics(
    frame: Any,
    *,
    recording_id: str,
    k_max: int = 64,
    min_rps: float = 30.0,
    k_min: int = K_MIN,
    band_hz: tuple[float, float] = BAND_HZ,
    segment_s: float = 40.96,
    max_components: int = 3,
    valid_fraction: float = 0.25,
    floor_law: bool = True,
) -> DecompDynamics:
    """Fit the speed law and the amplitude process of one decomposed recording."""
    y, x, valid, env_rate = envelope_arrays(frame, k_max=k_max, min_rps=min_rps)
    slope, slope_se, residual = fit_speed_law(y, x, valid, chunk=int(round(4.0 * env_rate)))
    per_order = [float(np.nanstd(residual[:, k, :])) for k in range(residual.shape[1])]
    del slope_se  # recomputed below, once the residual's correlation time is known

    seg = int(round(segment_s * env_rate))
    n_time = residual.shape[2]
    if n_time < seg:
        raise ValueError("recording shorter than one spectral segment")
    lines = residual[:, k_min - 1 :, :]
    max_gap = int(round(0.5 * env_rate))
    # Cross-validate by leaving one rotor out. The kernel is a rig-level
    # population and every rotor carries its OWN common component, so a
    # held-out rotor is genuinely independent of the training rotors — which a
    # held-out slice of time is not, and (through the common component) a
    # held-out order is not either. It also keeps the whole record in every
    # periodogram, which the slow component needs.
    per_rotor = [
        _segment_spectra(
            lines[rotor : rotor + 1], seg=seg, stride=seg // 2, env_rate=env_rate, max_gap=max_gap
        )
        for rotor in range(lines.shape[0])
    ]
    freqs = per_rotor[0][1]
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    freqs_band = freqs[band]
    rotor_power = [power[:, band] for power, _, _ in per_rotor]
    # a Lorentzian corner below 1/(2 pi L) is not in the data; see ``_unpack``
    tau_bounds = (2.0 / env_rate, segment_s / (2.0 * np.pi))

    scores: list[float] = []
    score_se: list[float] = []
    fold_scores: list[list[float]] = []
    for n_components in range(1, max_components + 1):
        folds: list[float] = []
        for held_out in range(len(rotor_power)):
            train = np.concatenate(
                [power for rotor, power in enumerate(rotor_power) if rotor != held_out]
            ).mean(axis=0)
            theta = _fit_components(freqs_band, train, n_components, tau_bounds, seed=n_components)
            density = _model_density(theta, freqs_band, tau_bounds)
            folds.append(float(-np.mean(rotor_power[held_out] / density + np.log(density))))
        fold_scores.append(folds)
        scores.append(float(np.mean(folds)))
        score_se.append(
            float(np.std(folds, ddof=1) / np.sqrt(len(folds))) if len(folds) > 1 else 0.0
        )
    best = int(np.argmax(scores))
    selected = next(
        index for index, score in enumerate(scores) if score >= scores[best] - score_se[best]
    )
    n_components = selected + 1
    train_mean = np.concatenate(rotor_power).mean(axis=0)
    theta = _fit_components(freqs_band, train_mean, n_components, tau_bounds, seed=n_components)

    common_series = np.nanmean(lines, axis=1)  # rotor-common process estimate
    common_power, _, _ = _segment_spectra(
        common_series[:, None, :],
        seg=seg,
        stride=seg // 2,
        env_rate=env_rate,
        max_gap=max_gap,
    )
    coherence = _component_coherence(
        freqs_band,
        theta,
        tau_bounds,
        train_mean,
        common_power[:, band].mean(axis=0),
        lines.shape[1],
    )
    std_db, tau_s, white = _unpack(theta, tau_bounds)
    components = [
        asdict(
            DynamicsComponent(
                std_db=float(std_db[j]),
                tau_s=float(tau_s[j]),
                coherence=float(coherence[j]),
            )
        )
        for j in range(n_components)
    ]
    # The clustered standard error is only valid when the clusters are long
    # against the residual's correlation time; size them from the fitted
    # slowest component instead of assuming.
    cluster = int(round(max(4.0, 5.0 * float(tau_s.max())) * env_rate))
    slope, slope_se, _ = fit_speed_law(y, x, valid, chunk=min(cluster, n_time // 3))
    return DecompDynamics(
        floor=fit_floor_law(frame, min_rps=min_rps) if floor_law else None,
        speed_exponent=slope,
        speed_exponent_se=slope_se,
        components=components,
        white_floor_db2_per_hz=white,
        selected_components=n_components,
        component_valid_whittle=scores,
        component_valid_whittle_se=score_se,
        component_fold_scores=fold_scores,
        residual_std_db=per_order,
        band_hz=[float(band_hz[0]), float(band_hz[1])],
        env_rate_hz=float(env_rate),
        k_min=int(k_min),
        n_train_segments=int(sum(power.shape[0] for power in rotor_power)),
        n_valid_segments=int(len(rotor_power)),
        recording_id=str(recording_id),
    )


def apply_decomp_dynamics(summary: dict[str, Any], dynamics: DecompDynamics) -> dict[str, Any]:
    """Attach the fitted amplitude process and speed law to a candidate summary."""
    out = copy.deepcopy(summary)
    out["dynamics"] = asdict(dynamics)
    # Both exponents are measured directly — line power against speed from the
    # envelopes, floor power against speed from the residual — so they replace
    # the values the Whittle fit carried pinned.
    rig = dict(out["rig"], amp_exp=float(dynamics.speed_exponent))
    if dynamics.floor is not None:
        rig["floor_exp"] = float(dynamics.floor["floor_exponent"])
        rig["floor_static_rel"] = float(dynamics.floor["floor_static_rel"])
    out["rig"] = rig
    return out


__all__ = [
    "BAND_HZ",
    "K_MIN",
    "DecompDynamics",
    "DynamicsComponent",
    "apply_decomp_dynamics",
    "envelope_arrays",
    "fit_decomp_dynamics",
    "fit_floor_law",
    "fit_speed_law",
]
