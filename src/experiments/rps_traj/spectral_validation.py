"""Mean speed plus cross-spectral dynamics of the four cruise motor speeds.

Window distance is Euclidean in the joint embedding. Retain the spectral
block separately for the diagnostics that specifically target fluctuations.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.distance import pdist

from experiments.rps_traj.spectral_inputs import RATE_HZ_OUT as FS
from experiments.rps_traj.spectral_inputs import WINDOW_SAMPLES as N

EDGES = (0.25, 0.5, 1.0, 2.0, 4.0)
BANDS = tuple(slice(round(lo * N / FS), round(hi * N / FS)) for lo, hi in zip(EDGES, EDGES[1:]))
SEED = 2026
# (diagnostic component, octave or None = all octaves)
CONTROLS = {
    "shared_phase": ("total", None),
    "independent_phase": ("coupling", None),
    "delay_one_80ms": ("coupling", None),
    "slow_common": ("power", 0),
    "fast_independent": ("power", 3),
}
COMPONENT = {"total": slice(None), "power": slice(0, 4), "coupling": slice(4, 16)}


def spectra(waves: np.ndarray) -> np.ndarray:
    """(segment, motor, time) -> three-bin-smoothed one-sided cross-periodogram.

    No time taper: common Fourier-phase changes are exact negative controls.
    Only interior bins are used. Smooth outer products, NOT Fourier amplitudes.
    """
    if waves.ndim != 3 or waves.shape[1:] != (4, N):
        raise ValueError(f"expected (n,4,{N}) windows at {FS} Hz; got {waves.shape}")
    z = np.fft.rfft(waves - waves.mean(axis=-1, keepdims=True), axis=-1)
    # Keep one guard bin on each side. Global Fourier-bin 4 becomes index 0.
    z = z[:, :, BANDS[0].start - 1 : BANDS[-1].stop + 1].transpose(0, 2, 1)
    s = z[..., :, None] * z[..., None, :].conj()
    return (s[:, :-2] + s[:, 1:-1] + s[:, 2:]) * (2 / (3 * FS * N))


def _octave(j: int) -> slice:
    return slice(BANDS[j].start - BANDS[0].start, BANDS[j].stop - BANDS[0].start)


def calibrate(s: np.ndarray, rows: list[dict]) -> np.ndarray:
    """Common band-specific power scale per motor, median across reserved rigs."""
    rigs = np.array([row["rig"] for row in rows])
    auto = s.diagonal(axis1=-2, axis2=-1).real
    powers = [
        [auto[rigs == rig, _octave(j)].mean(axis=(0, 1)) for j in range(len(BANDS))]
        for rig in np.unique(rigs)
    ]
    scale = np.median(powers, axis=0)
    if not np.all(np.isfinite(scale) & (scale > 0)):
        raise ValueError("non-positive or missing calibration power")
    return scale


def embed(s: np.ndarray, power: np.ndarray) -> np.ndarray:
    """Pack the four auto- and six complex cross-spectra without duplicating pairs."""
    i, k = np.triu_indices(4, 1)
    parts = []
    for j in range(len(BANDS)):
        ix = _octave(j)
        scale = np.sqrt(power[j])
        a = s[:, ix] / np.outer(scale, scale)
        off = a[..., i, k] * np.sqrt(2)
        entries = np.concatenate([a.diagonal(axis1=-2, axis2=-1).real, off.real, off.imag], axis=-1)
        parts.append(entries / np.sqrt(len(BANDS) * (ix.stop - ix.start)))
    return np.concatenate(parts, axis=1)


@dataclass(frozen=True)
class CruiseMetric:
    band_power: np.ndarray
    level_scale_rps: float
    spectrum_scale: float

    @classmethod
    def fit(cls, calibration_waves: np.ndarray, calibration_rows: list[dict]) -> CruiseMetric:
        """Match typical block distances on whole reserved recordings, one vote per pair."""
        if not all(r["calibration"] for r in calibration_rows):
            raise ValueError("the distance scales require calibration recordings only")
        s = spectra(calibration_waves)
        power = calibrate(s, calibration_rows)
        spec = embed(s, power).reshape(len(s), -1)
        levels = calibration_waves.mean(axis=-1)
        groups: dict[tuple, list[int]] = {}
        for i, r in enumerate(calibration_rows):
            groups.setdefault((r["rig"], r["flight"]), []).append(i)
        if len(groups) < 2:
            raise ValueError("at least two calibration recordings are needed")
        level_scale = np.median(pdist([levels[ix].mean(axis=0) for ix in groups.values()])) / 2
        spectrum_scale = np.median(pdist([spec[ix].mean(axis=0) for ix in groups.values()]))
        if (
            not np.all(np.isfinite(np.asarray([level_scale, spectrum_scale])))
            or min(level_scale, spectrum_scale) <= 0
        ):
            raise ValueError("calibration recordings have no usable between-flight variation")
        return cls(power, float(level_scale), float(spectrum_scale))

    def combine(self, means_rps: np.ndarray, spectral: np.ndarray) -> np.ndarray:
        # L2 of the first four differences = per-motor RMS difference / reference RMS.
        return np.column_stack(
            (
                means_rps / (2 * self.level_scale_rps),
                spectral.reshape(len(spectral), -1) / self.spectrum_scale,
            )
        )

    def transform(self, waves: np.ndarray) -> np.ndarray:
        return self.combine(waves.mean(axis=-1), embed(spectra(waves), self.band_power))


def _offset_controls(metric: CruiseMetric, waves: np.ndarray, clean: np.ndarray) -> dict:
    """A common level shift and a motor trim shift must both move the joint point."""
    out = {}
    for name, offset in (
        ("common", [30, 30, 30, 30]),
        ("one_motor", [30, 0, 0, 0]),
        ("zero_mean_trim", [30, -30, 30, -30]),
    ):
        step = np.array(offset, dtype=float)
        difference = metric.transform(waves + step[None, :, None]) - clean
        distances = np.linalg.norm(difference, axis=1)
        expected = np.sqrt(np.mean(step**2)) / metric.level_scale_rps
        spectral_residual = np.linalg.norm(difference[:, 4:], axis=1).max()
        np.testing.assert_allclose(distances, expected, rtol=1e-10)
        np.testing.assert_allclose(spectral_residual, 0, atol=1e-10)
        out[name] = {
            "offset_rps": offset,
            "windows": len(waves),
            "distance": float(np.median(distances)),
            "expected": float(expected),
            "max_spectral_change": float(spectral_residual),
        }
    return out


def augment(x: np.ndarray, name: str, rng: np.random.Generator) -> np.ndarray:
    if name in ("shared_phase", "independent_phase"):
        z = np.fft.rfft(x, axis=-1)
        channels = 1 if name == "shared_phase" else 4
        phase = rng.uniform(-np.pi, np.pi, size=(len(x), channels, N // 2 - 1))
        z[..., 1:-1] *= np.exp(1j * phase)
        return np.fft.irfft(z, n=N, axis=-1)
    if name == "delay_one_80ms":
        y = x.copy()
        y[:, 0] = np.roll(x[:, 0], 2, axis=-1)
        return y
    hover = np.median(x, axis=(1, 2), keepdims=True)
    if name == "slow_common":
        phase = rng.uniform(-np.pi, np.pi, size=(len(x), 1, 1))
        return x + 0.04 * hover * np.sin(2 * np.pi * 5 * np.arange(N) / N + phase)
    if name == "fast_independent":
        z = np.zeros((len(x), 4, N // 2 + 1), dtype=complex)
        shape = z[..., BANDS[-1]].shape
        z[..., BANDS[-1]] = rng.standard_normal(shape) + 1j * rng.standard_normal(shape)
        y = np.fft.irfft(z, n=N, axis=-1)
        return x + 0.02 * hover * y / y.std(axis=-1, keepdims=True)
    raise KeyError(name)


def _norm(x: np.ndarray, component: str, band: int | None = None) -> np.ndarray:
    v = x[:, slice(None) if band is None else _octave(band), COMPONENT[component]]
    # Per-band distances do not include the 1/sqrt(n_bands) global weight.
    return np.sqrt((1 if band is None else len(BANDS)) * np.einsum("nfc,nfc->n", v, v))


def _controls(rows: list[dict], waves: np.ndarray, features: np.ndarray, power: np.ndarray) -> dict:
    rng = np.random.default_rng(SEED)
    rigs = np.array([r["rig"] for r in rows])
    flights = np.array([r["flight"] for r in rows])
    results = {}
    for name, (component, band) in CONTROLS.items():
        rig_stats = {}
        for rig in np.unique(rigs):
            means = {
                f: features[(rigs == rig) & (flights == f)].mean(axis=0)
                for f in np.unique(flights[rigs == rig])
            }
            total = np.sum(list(means.values()), axis=0)
            pairs = []
            invariant = []
            for flight, mean in means.items():
                ids = (rigs == rig) & (flights == flight)
                clean = features[ids]
                augmented = embed(spectra(augment(waves[ids], name, rng)), power)
                shift = augmented - clean
                if name in ("shared_phase", "independent_phase", "delay_one_80ms"):
                    invariant_component = "total" if name == "shared_phase" else "power"
                    base_norm = np.maximum(_norm(clean, invariant_component), 1e-15)
                    invariant.extend((_norm(shift, invariant_component) / base_norm).tolist())
                item: dict = {
                    "flight": str(flight),
                    "windows": len(clean),
                    "self_distance": float(np.median(_norm(shift, component, band))),
                }
                if len(means) > 1:
                    ref = (total - mean) / (len(means) - 1)
                    before, after = clean - ref, augmented - ref
                    base = _norm(before, component, band)
                    item["clean"] = float(np.median(base))
                    item["penalty"] = float(np.median(_norm(after, component, band) - base))
                    item["octave_penalties"] = [
                        float(np.median(_norm(after, component, j) - _norm(before, component, j)))
                        for j in range(len(BANDS))
                    ]
                pairs.append(item)
            info: dict = {"flights": pairs}
            if invariant:
                info["max_relative_invariance_error"] = max(invariant)
            if len(means) > 1:
                clean_scores = np.array([v["clean"] for v in pairs])
                penalties = np.array([v["penalty"] for v in pairs])
                mad = float(np.median(np.abs(clean_scores - np.median(clean_scores))))
                info.update(
                    clean_mad=mad,
                    clean_median=float(np.median(clean_scores)),
                    median_penalty=float(np.median(penalties)),
                    worse=int(np.sum(penalties > 0)),
                    octave_penalties=np.median(
                        [v["octave_penalties"] for v in pairs], axis=0
                    ).tolist(),
                )
                if name != "shared_phase":
                    info["passed"] = bool(
                        np.median(penalties) > mad and np.mean(penalties > 0) > 0.5
                    )
                    if band is not None:
                        profile = np.asarray(info["octave_penalties"])
                        info["target_band_largest"] = bool(
                            profile[band] > np.delete(profile, band).max()
                        )
                        info["passed"] = info["passed"] and info["target_band_largest"]
            rig_stats[str(rig)] = info
        eligible = [r for r in rig_stats.values() if "passed" in r]
        invariance = [
            r["max_relative_invariance_error"]
            for r in rig_stats.values()
            if "max_relative_invariance_error" in r
        ]
        inv_ok = not invariance or max(invariance) < 1e-10
        passed = sum(r["passed"] for r in eligible)
        results[name] = {
            "component": component,
            "octave_index": band,
            "rigs": rig_stats,
            "eligible_rigs": len(eligible),
            "rigs_passing_effect_gate": passed,
            "passed": bool(inv_ok and (name == "shared_phase" or passed > len(eligible) / 2)),
        }
    return results


def run(out: Path) -> dict:
    from experiments.rps_traj.spectral_inputs import load_segments
    from experiments.rps_traj.spectral_map import analyze, map_all_flights

    all_rows, waves, inventory = load_segments()
    calibration = np.array([r["calibration"] for r in all_rows], dtype=bool)
    metric = CruiseMetric.fit(waves[calibration], [r for r in all_rows if r["calibration"]])
    rows = [r for r in all_rows if not r["calibration"]]
    test = waves[~calibration]
    means = waves.mean(axis=-1)
    spectral = embed(spectra(waves), metric.band_power)
    joint = metric.combine(means, spectral)
    out.mkdir(parents=True, exist_ok=True)
    controls = _controls(rows, test, spectral[~calibration], metric.band_power)
    offsets = _offset_controls(metric, test, joint[~calibration])
    real = analyze(rows, joint[~calibration], out)
    flight_map = map_all_flights(all_rows, joint, means, out)
    report = {
        "sample_rate_hz": FS,
        "samples": N,
        "band_edges_hz": EDGES,
        "seed": SEED,
        "metric": "per_motor_mean_plus_cross_spectrum_v1",
        "level_scale_rps": metric.level_scale_rps,
        "spectral_distance_scale": metric.spectrum_scale,
        "calibration_band_power": metric.band_power.tolist(),
        "inventory": inventory,
        "segments": all_rows,
        "spectral_controls": controls,
        "offset_controls": offsets,
        "real": real,
        "flight_map": flight_map,
    }
    (out / "spectral_validation.json").write_text(
        json.dumps(report, indent=1, allow_nan=False) + "\n"
    )
    return report
