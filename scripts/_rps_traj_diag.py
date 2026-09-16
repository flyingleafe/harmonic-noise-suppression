#!/usr/bin/env python3
"""Why the round-2 fits get the ACF and the cross-correlation wrong.

Round 2 beat the baseline on 30 of 35 family discrepancies but still failed
every rig, and the blocking family is ``acf`` on six of seven rigs: the model's
autocorrelation decays far too slowly (michaels model 0.50 at 1.6 s against a
real 0.16) and its rotor cross-correlation is far too weak on DREGON and
neurobem.  Both point at the same suspect — the fit putting its variance at the
wrong time scales — but "points at" is not a measurement, so this script
measures it, per MODE (``tracking.rotors.modes_from_rps``), for the three rigs
that matter.

What it produces under ``results/rps_traj/diag``:

``<rig>_spectra.png``
    Per mode, the real averaged periodogram EXACTLY as the fit sees it (Hann
    20 s blocks at 50 % overlap, block-demeaned, taper-power corrected) with
    the fitted model mode spectrum on top, log-log over the fit band.  Plus a
    per-mode variance table in the bands < 0.2 / 0.2-1 / 1-5 / 5-40 Hz, so a
    mismatch can be attributed to a band rather than guessed at.
``<rig>_acf.png``
    Per mode, the frozen-style ACF (per-segment demeaned, biased normalisation,
    duration weighted) of the real telemetry and of samples drawn from the fit,
    with the model's analytic ACF, on ``stats.LAGS_S``.  A third real curve is
    computed on 20 s BLOCK-demeaned data: if the real ACF only looks fast
    because the estimator sees whole segments while the FIT only ever sees 20 s
    blocks, that curve is where it shows.
``summary.json``
    Every number behind the figures, plus:
    * neurobem: the real per-segment mode variances against the model's
      EXPECTED within-segment variances for the same segment lengths, using
      ``var_T = sigma^2 (1 - 2 tau/T (1 - tau/T (1 - e^(-T/tau))))`` — the test
      of whether the components sitting on the 10 s bound carry more in-segment
      variance than the real modes do.

Usage: ``python scripts/_rps_traj_diag.py [--rigs michaels,dregon,neurobem_quad]``
(reads ``results/rps_traj/fits/new/<rig>.json``; it refits nothing).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy.signal.windows import hann  # noqa: E402

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parent if (_HERE.parent / "src").is_dir() else Path.cwd().resolve()
sys.path.insert(0, str(_ROOT / "src"))

from data_processing.trajectory_model import (  # noqa: E402
    NewFit,
    Params,
    rotation,
)
from experiments.rps_traj.data import (  # noqa: E402
    RATE_HZ,
    Flight,
    airborne_segments,
    load_rig,
)
from experiments.rps_traj.stats import (  # noqa: E402
    LAG_SAMPLES,
    LAGS_S,
)
from tracking.rotors import MIXER, MODE_NAMES, NUM_ROTORS  # noqa: E402

#: Rigs worth the diagnostic: the two developed ones plus the worst offender.
DIAG_RIGS = ("michaels", "dregon", "neurobem_quad")

#: Variance-attribution bands (Hz).
BANDS = ((0.0, 0.2), (0.2, 1.0), (1.0, 5.0), (5.0, 40.0))

#: Repetitions when drawing model statistics for the ACF panel.
N_REP = 10

MODE_COLORS = ("#d62728", "#1f77b4", "#2ca02c", "#9467bd")

#: The DIAGNOSTIC's own periodogram recipe, deliberately local: it is a picture
#: of the data, not the likelihood, and round 4 replaced the frequency-domain
#: likelihood entirely, so there are no longer any block constants in the model
#: to borrow.  20 s Hann blocks at 50 % overlap resolve 0.05 Hz, which is what
#: it takes to see the sub-0.5 Hz half of the variance.
BLOCK_S = 20.0
BLOCK_OVERLAP = 0.5
MIN_BLOCK_S = 5.0


# ─── real mode periodograms, exactly as the fit sees them ─────────────────────


def mode_periodogram(
    flights: list[Flight],
    fs: float = RATE_HZ,
    block_s: float = BLOCK_S,
    detrend: str = "block_mean",
) -> tuple[np.ndarray, np.ndarray, int]:
    """``(f, P, n_blocks)`` with ``P`` ``(4, F)`` one-sided mode PSDs.

    The same geometry as :func:`model.whittle_groups` — Hann blocks at 50 %
    overlap, taper-power corrected, ``sum_f P df`` = block variance — but
    expressed per MODE (``m = M^T w / 4``) rather than as the ``M^T I M`` the
    likelihood consumes, so it is directly comparable to a mode spectrum.
    """
    full = int(round(block_s * fs))
    hop = max(int(round(full * (1.0 - BLOCK_OVERLAP))), 1)
    min_len = int(round(MIN_BLOCK_S * fs))
    acc: dict[int, dict[str, Any]] = {}
    for flight in flights:
        rps = np.asarray(flight.rps, dtype=np.float64)
        for sl in airborne_segments(rps, fs):
            seg = rps[:, sl]
            n = seg.shape[1]
            length = full if n >= full else n
            if length < min_len:
                continue
            seg_mean = np.nanmean(seg, axis=1)
            entry = acc.setdefault(length, {"n": 0, "sum": None})
            for start in range(0, max(n - length + 1, 1), hop):
                block = seg[:, start : start + length]
                if block.shape[1] != length or not np.isfinite(block).all():
                    continue
                if detrend == "segment_mean":
                    centred = block - seg_mean[:, None]
                else:
                    centred = block - block.mean(axis=1, keepdims=True)
                w = hann(length, sym=False)
                spec = np.fft.rfft((MIXER.T @ centred) / NUM_ROTORS * w, axis=-1)
                power = (2.0 / (fs * float(np.sum(w**2)))) * np.abs(spec) ** 2
                entry["sum"] = power if entry["sum"] is None else entry["sum"] + power
                entry["n"] += 1
    usable = {k: v for k, v in acc.items() if v["n"]}
    if not usable:
        raise ValueError("no usable blocks")
    # Report the dominant (longest, most populated) block length: the shorter
    # whole-segment blocks have a different frequency grid.
    length = max(usable, key=lambda k: (usable[k]["n"] * k, k))
    entry = usable[length]
    f = np.fft.rfftfreq(length, d=1.0 / fs)
    return f, entry["sum"] / entry["n"], int(entry["n"])


def model_mode_spectrum(p: Params, f: np.ndarray, fs: float = RATE_HZ) -> np.ndarray:
    """``(4, F)`` model spectrum of the MODE series ``m = M^T w / 4``.

    ``m = M^T (mu + delta) / 4 + R(theta) v + M^T e / 4``, so the mode
    cross-spectrum is ``R diag(S_v) R^T + S_e / 4 I`` — the measurement process
    picks up ``M^T M / 16 = I / 4``.  The per-flight offset is DC and does
    not appear.  ``S_v`` comes from :meth:`Params.component_psd`, i.e. each
    mode's slow OU plus its CAR2 oscillator.
    """
    comp = p.component_psd(f, fs)  # (4, F) per COMPONENT (pre-rotation)
    r = rotation(p.theta)
    out = np.einsum("ij,jf,ij->if", r, comp, r)  # diagonal of R diag R^T
    return out + p.measurement_psd(f, fs) / 4.0


def band_variance(f: np.ndarray, psd: np.ndarray) -> dict[str, list[float]]:
    """Per-mode variance in each of :data:`BANDS`, by trapezoid integration."""
    out: dict[str, list[float]] = {}
    for lo, hi in BANDS:
        keep = (f >= lo) & (f <= hi)
        out[f"{lo:g}-{hi:g}Hz"] = [
            float(np.trapezoid(psd[i][keep], f[keep])) for i in range(psd.shape[0])
        ]
    return out


# ─── frozen-style ACF, per mode ───────────────────────────────────────────────


def frozen_acf(series: list[np.ndarray]) -> np.ndarray:
    """``(4, len(LAGS_S))`` frozen-style ACF of a list of ``(4, n)`` segments.

    Exactly ``stats.compute_stats``'s estimator, per mode: each segment is
    demeaned, the biased sum ``sum c[n] c[n+k]`` is normalised by the segment's
    FULL sum of squares, and segments are weighted by their length.
    """
    num = np.zeros((NUM_ROTORS, LAG_SAMPLES.size))
    den = np.zeros((NUM_ROTORS, LAG_SAMPLES.size))
    for seg in series:
        c = seg - seg.mean(axis=1, keepdims=True)
        n = c.shape[1]
        c0 = (c * c).sum(axis=1)
        ok = c0 > 0.0
        for j, lag in enumerate(LAG_SAMPLES):
            k = int(lag)
            if n < 2 * k:
                continue
            r = (c[:, : n - k] * c[:, k:]).sum(axis=1)
            num[ok, j] += float(n) * r[ok] / c0[ok]
            den[ok, j] += float(n)
    with np.errstate(invalid="ignore"):
        return np.where(den > 0, num / np.maximum(den, 1.0), np.nan)


def mode_segments(
    flights: list[Flight], fs: float = RATE_HZ, block_s: float | None = None
) -> list[np.ndarray]:
    """Airborne mode series; with ``block_s`` each segment is cut into blocks
    (which is what makes the ACF see only what the FIT sees)."""
    out: list[np.ndarray] = []
    for flight in flights:
        rps = np.asarray(flight.rps, dtype=np.float64)
        for sl in airborne_segments(rps, fs):
            m = (MIXER.T @ np.nan_to_num(rps[:, sl], nan=0.0)) / NUM_ROTORS
            if block_s is None:
                out.append(m)
                continue
            length = int(round(block_s * fs))
            if m.shape[1] < length:
                out.append(m)
                continue
            for start in range(0, m.shape[1] - length + 1, length):
                out.append(m[:, start : start + length])
    return out


def analytic_mode_acf(p: Params, fs: float = RATE_HZ) -> np.ndarray:
    """``(4, L)`` analytic ACF of the model's mode series on ``LAGS_S``.

    Straight from :meth:`Params.mode_acf`, which evaluates
    ``C(k) = H Phi^k P H^T`` on the components' own state spaces — the same
    objects the likelihood and the sampler use, so there is no separate ACF
    formula that could drift (the CAR2 ACF has three closed forms depending on
    whether ``zeta`` is below, at, or above 1; the state space has one).
    """
    return p.mode_acf(LAG_SAMPLES, fs)


def model_mode_segments(
    fit: NewFit, durations: list[float], fs: float = RATE_HZ, n_rep: int = N_REP, seed: int = 0
) -> list[np.ndarray]:
    """Mode series of model samples, after the frozen airborne rule."""
    sampler = fit.sampler(fs)
    out: list[np.ndarray] = []
    for rep in range(n_rep):
        for i, duration in enumerate(durations):
            n = int(round(duration * fs))
            rps = sampler(n, np.random.default_rng([seed, rep, i]))
            for sl in airborne_segments(rps, fs):
                out.append((MIXER.T @ rps[:, sl]) / NUM_ROTORS)
    return out


# ─── neurobem: expected within-segment variance ───────────────────────────────


def expected_segment_mode_var(p: Params, duration_s: float, fs: float = RATE_HZ) -> np.ndarray:
    """``(4,)`` model per-MODE variance a ``duration_s`` segment would show.

    :meth:`Params.mode_var_in_window` computes ``C(0) - Var(window mean)``
    exactly on the grid from the components' state spaces.  Round 2 used the
    OU-only closed form ``sigma^2 (1 - 2 tau/T (1 - tau/T (1 - e^{-T/tau})))``,
    which is its continuous-time special case and does not cover a CAR2.
    """
    if not np.isfinite(duration_s):  # T -> inf recovers the stationary variance
        r = rotation(p.theta)
        return np.einsum("ij,j,ij->i", r, p.mode_var, r) + p.sigma_e**2 / 4.0
    return p.mode_var_in_window(int(round(float(duration_s) * fs)), fs)


# ─── figures ──────────────────────────────────────────────────────────────────


def plot_spectra(
    rig: str,
    f: np.ndarray,
    real: np.ndarray,
    model: np.ndarray,
    band: tuple[float, float],
    out: Path,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0), sharex=True, sharey=True)
    keep = (f >= band[0]) & (f <= band[1]) & (f > 0)
    for i, ax in enumerate(axes.ravel()):
        ax.loglog(f[keep], real[i][keep], color="0.35", lw=1.3, label="real (as fitted)")
        ax.loglog(f[keep], model[i][keep], color=MODE_COLORS[i], lw=2.0, label="model")
        ax.set_title(f"{MODE_NAMES[i]}", fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        ax.legend(fontsize=8, loc="lower left")
    for ax in axes[1]:
        ax.set_xlabel("frequency (Hz)")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"mode PSD ((rev/s)$^2$/Hz)")
    fig.suptitle(
        f"{rig}: real mode periodogram (Hann {BLOCK_S:g} s blocks, block-demeaned) "
        "vs fitted model spectrum",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def plot_acf(
    rig: str,
    real_seg: np.ndarray,
    real_block: np.ndarray,
    model_sample: np.ndarray,
    model_analytic: np.ndarray,
    out: Path,
) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.4), sharex=True, sharey=True)
    for i, ax in enumerate(axes.ravel()):
        ax.semilogx(
            LAGS_S,
            real_seg[i],
            "o-",
            ms=3.5,
            color="0.15",
            lw=1.6,
            label="real, per-segment demeaned (frozen)",
        )
        ax.semilogx(
            LAGS_S,
            real_block[i],
            "s--",
            ms=3.0,
            color="0.55",
            lw=1.2,
            label=f"real, {BLOCK_S:g} s block demeaned",
        )
        ax.semilogx(
            LAGS_S,
            model_sample[i],
            "o-",
            ms=3.5,
            color=MODE_COLORS[i],
            lw=1.8,
            label="model samples (frozen estimator)",
        )
        ax.semilogx(
            LAGS_S, model_analytic[i], ":", color=MODE_COLORS[i], lw=1.6, label="model analytic"
        )
        ax.axhline(0.0, color="k", lw=0.6)
        ax.axhline(np.exp(-1.0), color="k", lw=0.6, ls=":")
        ax.set_title(MODE_NAMES[i], fontsize=11)
        ax.grid(True, which="both", alpha=0.25)
        ax.set_ylim(-0.35, 1.05)
    for ax in axes[1]:
        ax.set_xlabel("lag (s)")
    for ax in axes[:, 0]:
        ax.set_ylabel("autocorrelation")
    # One shared legend: per-panel legends land on top of the curves, which in
    # the yaw panel is exactly where the disagreement is.  The two model
    # entries are drawn in the panel's own colour, so the shared key uses grey
    # proxies and the colour is read off the title.
    handles = [
        Line2D(
            [],
            [],
            color="0.15",
            marker="o",
            ms=4,
            lw=1.6,
            label="real, per-segment demeaned (frozen)",
        ),
        Line2D(
            [],
            [],
            color="0.55",
            marker="s",
            ms=3.5,
            lw=1.2,
            ls="--",
            label=f"real, {BLOCK_S:g} s block demeaned",
        ),
        Line2D(
            [],
            [],
            color="#444444",
            marker="o",
            ms=4,
            lw=1.8,
            label="model samples, in mode colour (frozen estimator)",
        ),
        Line2D([], [], color="#444444", lw=1.6, ls=":", label="model analytic, in mode colour"),
        Line2D([], [], color="k", lw=0.6, ls=":", label="1/e"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=9, frameon=False)
    fig.suptitle(f"{rig}: per-mode autocorrelation, real vs fitted model", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.075, 1.0, 1.0))
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


def sample_durations(flights: list[Flight], fs: float = RATE_HZ) -> list[float]:
    """Real airborne segment lengths plus the 2 s the airborne rule erodes.

    Model samples are drawn at these lengths so that what survives the frozen
    airborne rule is exactly as long as the real segments it is compared with.
    """
    return [
        (sl.stop - sl.start) / fs + 2.0
        for flight in flights
        for sl in airborne_segments(flight.rps, fs)
    ]


# ─── main ─────────────────────────────────────────────────────────────────────


def run_rig(rig: str, root: Path, seed: int) -> dict[str, Any]:
    flights = load_rig(rig)
    fit = NewFit.from_json((root / "fits" / "new" / f"{rig}.json").read_text())
    durations = sample_durations(flights)
    diag = root / "diag"
    diag.mkdir(parents=True, exist_ok=True)

    # (1) spectra
    f, real_psd, n_blocks = mode_periodogram(flights)
    model_psd = model_mode_spectrum(fit.params, f)
    top = min(0.5 * fit.fit_rate_hz, 40.0)
    spectra_png = plot_spectra(
        rig, f, real_psd, model_psd, (0.02, top), diag / f"{rig}_spectra.png"
    )

    # (2)+(3) ACFs
    real_seg = frozen_acf(mode_segments(flights))
    real_block = frozen_acf(mode_segments(flights, block_s=BLOCK_S))
    model_samp = frozen_acf(model_mode_segments(fit, durations, seed=seed))
    model_ana = analytic_mode_acf(fit.params)
    acf_png = plot_acf(rig, real_seg, real_block, model_samp, model_ana, diag / f"{rig}_acf.png")

    report: dict[str, Any] = {
        "rig": rig,
        "figures": {"spectra": spectra_png, "acf": acf_png},
        "n_blocks": n_blocks,
        "fit_rate_hz": fit.fit_rate_hz,
        "modes": list(MODE_NAMES),
        "band_variance": {
            "real": band_variance(f, real_psd),
            "model": band_variance(f, model_psd),
        },
        "acf": {
            "lags_s": LAGS_S.tolist(),
            "real_segment_demeaned": real_seg.tolist(),
            "real_block_demeaned": real_block.tolist(),
            "model_samples": model_samp.tolist(),
            "model_analytic": model_ana.tolist(),
        },
        "fitted_params": fit.params.to_dict(),
    }

    print(f"\n[{rig}] {n_blocks} periodogram blocks, fit rate {fit.fit_rate_hz:g} Hz")
    print("  in-band mode variance (rev/s)^2, real | model")
    header = "  " + "band".ljust(12) + "".join(f"{m:>22}" for m in MODE_NAMES)
    print(header)
    for key in report["band_variance"]["real"]:
        cells = "".join(
            f"{report['band_variance']['real'][key][i]:10.3f} |"
            f"{report['band_variance']['model'][key][i]:10.3f}"
            for i in range(NUM_ROTORS)
        )
        print("  " + key.ljust(12) + cells)
    print("  ACF at 0.05 / 0.32 / 1.99 / 10.0 s (rotor-mode mean):")
    for name, curve in (
        ("real segment", real_seg),
        ("real block  ", real_block),
        ("model sample", model_samp),
        ("model analyt", model_ana),
    ):
        print(
            f"    {name} "
            + "  ".join(
                f"{MODE_NAMES[i][:5]}:" + "/".join(f"{curve[i, j]:6.3f}" for j in (0, 8, 16, 23))
                for i in range(NUM_ROTORS)
            )
        )

    if rig == "neurobem_quad":
        seg_lengths = [d - 2.0 for d in durations]
        median_t = float(np.median(seg_lengths))
        real_var = _real_segment_mode_var(flights)
        model_var = expected_segment_mode_var(fit.params, median_t)
        model_stat = expected_segment_mode_var(fit.params, float("inf"))
        report["segment_variance"] = {
            "median_segment_s": median_t,
            "n_segments": len(seg_lengths),
            "real_mode_var": real_var.tolist(),
            "model_expected_in_segment": model_var.tolist(),
            "model_stationary": model_stat.tolist(),
            "component_tau_slow": fit.params.tau_slow.tolist(),
            "component_f0": fit.params.f0.tolist(),
            "component_zeta": fit.params.zeta.tolist(),
            "component_stationary_var_slow": (fit.params.sigma_slow**2).tolist(),
            "component_stationary_var_osc": (fit.params.sigma_osc**2).tolist(),
        }
        print(f"  median segment {median_t:.1f} s over {len(seg_lengths)} segments")
        print(f"    real per-segment mode var      {np.round(real_var, 1)}")
        print(f"    model expected in-segment var  {np.round(model_var, 1)}")
        print(f"    model stationary var           {np.round(model_stat, 1)}")
    return report


def _real_segment_mode_var(flights: list[Flight], fs: float = RATE_HZ) -> np.ndarray:
    """Duration-weighted mean of the per-segment, within-segment mode variance."""
    num = np.zeros(NUM_ROTORS)
    den = 0.0
    for seg in mode_segments(flights, fs):
        n = seg.shape[1]
        num += n * seg.var(axis=1)
        den += n
    return num / max(den, 1.0)


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    ap.add_argument("--rigs", default=",".join(DIAG_RIGS))
    ap.add_argument("--root", type=Path, default=Path("results/rps_traj"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.root)
    reports = [
        run_rig(rig, root, args.seed)
        for rig in (r.strip() for r in str(args.rigs).split(","))
        if rig
    ]
    out = root / "diag" / "summary.json"
    out.write_text(
        json.dumps(
            {"bands_hz": [list(b) for b in BANDS], "n_rep": N_REP, "rigs": reports},
            indent=2,
            allow_nan=False,
        )
    )
    print(f"\nwrote {out}")
    for report in reports:
        for path in report["figures"].values():
            print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
