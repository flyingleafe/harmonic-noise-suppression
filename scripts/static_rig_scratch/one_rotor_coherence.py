"""Phase coherence of one rotor's harmonics, one recording, one microphone.

The instructive version of ``scripts/noise_v2_order_decoherence.py``: every
order of ``motor_Motor1_70`` is demodulated on one quiet microphone, the
phase increment over a lag is read with the circular estimator, the shared
shaft part (fitted on k <= 10) is removed, and the additive-noise floor
``2 log(1 + 1/SNR)`` is subtracted.  Orders run to 100 so the motor family
(k = 42, 63, 84) is covered.  Figures and a JSON go to
``results/noise_v2/decoherence/one_rotor/``; the explainer
``docs/explainers/order-decoherence.qmd`` reads them.

    PYTHONPATH=src .venv/bin/python scripts/static_rig_scratch/one_rotor_coherence.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from utils.demod import demodulate, residual_frequency  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "decoh", ROOT / "scripts" / "noise_v2_order_decoherence.py"
)
assert _spec is not None and _spec.loader is not None
decoh = importlib.util.module_from_spec(_spec)
sys.modules["decoh"] = decoh
_spec.loader.exec_module(decoh)

OUT = ROOT / "results" / "noise_v2" / "decoherence" / "one_rotor"
RECORDING = "motor_Motor1_70"
FS = 44100.0
K_MAX = 100
SHAFT_ORDERS = np.arange(1, 11)
BAND_FRAC = 0.4
BB_DECIM = 220  # 44100 / 220 = 200.45 Hz phase grid
FS_PHASE = FS / BB_DECIM
LAGS_MS = (5, 10, 20, 50, 100, 200, 500, 1000, 2000)
SHOW_ORDERS = (2, 4, 10, 20, 40, 42, 63, 84)
MOTOR_ORDERS = (42, 63, 84)
GATE_DB = 10.0  # peak / median of the in-band spectrum


def pick_mic(seg: np.ndarray) -> tuple[int, np.ndarray]:
    """The microphone with the least 20-200 Hz (wind) power relative to 1-4 kHz."""
    f, p = signal.welch(seg, fs=FS, nperseg=8192, axis=-1)
    low = p[:, (f >= 20) & (f <= 200)].mean(axis=-1)
    mid = p[:, (f >= 1000) & (f <= 4000)].mean(axis=-1)
    ratio_db = 10 * np.log10(low / mid)
    return int(np.argmin(ratio_db)), ratio_db


def demod_orders(x: np.ndarray, rate: float, orders: np.ndarray) -> np.ndarray:
    carrier = np.full(x.shape[-1], rate)
    band = BAND_FRAC * rate
    return np.stack(
        [demodulate(x, carrier, band, FS, order=int(k))[..., ::BB_DECIM] for k in orders]
    )


def circ_var(prod: np.ndarray) -> float:
    """``-2 log |gamma|`` of the amplitude-weighted circular mean of ``prod``."""
    num = np.abs(prod.sum())
    den = max(np.abs(prod).sum(), 1e-300)
    return float(-2.0 * np.log(np.clip(num / den, 1e-8, 1.0)))


def amp_var(zz: np.ndarray, lag: int) -> float:
    """Variance of the log-amplitude increment ``log|z(t+lag)| - log|z(t)|``,
    robust (1.4826 MAD squared) so a few noise-collapsed samples do not own it."""
    d = np.log(np.abs(zz[lag:]) + 1e-300) - np.log(np.abs(zz[:-lag]) + 1e-300)
    return float((1.4826 * np.median(np.abs(d - np.median(d)))) ** 2)


def mc_floor(
    snr_db_grid: np.ndarray, lags: list[int], band: float, seconds: float = 120.0, seed: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """``(phase, log-amplitude)`` floors, each ``(n_snr, n_lags)``, of the two
    estimators for a steady line in band-limited noise, by simulation on the
    phase grid with the SAME estimators - a miniature of the paired control
    of the 12-recording study. The noise is white inside ``+-band`` (the
    demodulation passband), so two samples closer than ~1/(2 band) share part
    of it and the floors are lower at the shortest lags."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS_PHASE)
    white = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    sos = signal.butter(6, band, btype="low", fs=FS_PHASE, output="sos")
    noise = signal.sosfiltfilt(sos, white)
    noise /= np.sqrt(np.mean(np.abs(noise) ** 2))
    out = np.empty((snr_db_grid.size, len(lags)))
    out_a = np.empty_like(out)
    for i, s in enumerate(snr_db_grid):
        zz = np.sqrt(10 ** (s / 10)) + noise
        out[i] = [circ_var(zz[lag:] * np.conj(zz[:-lag])) for lag in lags]
        out_a[i] = [amp_var(zz, lag) for lag in lags]
    return out, out_a


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [j for j in decoh.load_bench() if j.name == RECORDING]
    if not jobs:
        raise SystemExit(f"{RECORDING} not in the local DREGON-frames")
    job = jobs[0]
    job.max_seg_s = 30.0
    seg, start_s, span_s = decoh.job_segment(job)
    mic, wind_db = pick_mic(seg)
    x = seg[mic]
    rate0 = decoh.survey_rates()[RECORDING]
    rate = rate0
    work = decoh.pre_decimate(seg[mic : mic + 1])
    for k in (10, 40):
        rate, _resid = decoh.refine_rate(work, rate, k)
    # final refinement at k = 84 on the full-rate signal
    z84 = demodulate(x, np.full(x.size, rate), BAND_FRAC * rate, FS, order=84)
    trim = int(0.3 * FS)
    fr = residual_frequency(z84[trim:-trim], FS)
    w = np.abs(z84[trim:-trim]) ** 2
    rate += float(np.sum(fr * w) / np.sum(w)) / 84

    orders = np.arange(1, K_MAX + 1)
    z = demod_orders(x, rate, orders)  # (K, T)
    edge = int(0.5 * FS_PHASE)
    z = z[:, edge:-edge]
    t = np.arange(z.shape[-1]) / FS_PHASE
    band = BAND_FRAC * rate
    snr_pm, snr_eff = decoh.line_snr(z[:, None, :], band)
    snr_pm, snr_eff = snr_pm[:, 0], snr_eff[:, 0]
    gated = snr_pm >= GATE_DB

    # shaft phase from orders 1..10 of ALL microphones (the shaft is shared;
    # pooling the mics shrinks the estimate's own error): SNR-weighted LS of
    # unwrapped phase / k, in three disjoint order groups so the error of the
    # estimate can be measured from their disagreement (as the 12-recording
    # study does) and the combined estimate is inverse-variance weighted.
    z_lo = demod_orders(seg, rate, SHAFT_ORDERS)[:, :, edge:-edge]  # (10, M, T)
    pm_lo, _ = decoh.line_snr(z_lo, band)
    use_lo = pm_lo >= GATE_DB
    phi_lo, _slip = decoh.unwrap_censored(z_lo)
    wgt = np.where(use_lo, pm_lo, 0.0)  # (10, M)
    group = (SHAFT_ORDERS - 1) % 3
    shaft_g = []
    for g in range(3):
        wg = wgt * (group == g)[:, None]
        shaft_g.append(
            np.einsum("km,k,kmt->t", wg, SHAFT_ORDERS, phi_lo)
            / np.einsum("km,k->", wg, SHAFT_ORDERS**2)
        )
    shaft_g = np.stack(shaft_g)  # (3, T)

    lags = [int(round(ms * 1e-3 * FS_PHASE)) for ms in LAGS_MS]
    snr_grid = np.arange(-15.0, 35.1, 1.0)
    floor_grid, floor_grid_a = mc_floor(snr_grid, lags, band)  # (n_snr, n_lags)
    snr_db = 10 * np.log10(np.maximum(snr_eff, 1e-9))
    floor = np.stack(
        [np.interp(snr_db, snr_grid, floor_grid[:, i]) for i in range(len(lags))]
    )  # (n_lags, K)
    floor_a = np.stack([np.interp(snr_db, snr_grid, floor_grid_a[:, i]) for i in range(len(lags))])
    floor_analytic = decoh.phase_floor(snr_eff)
    v_raw = np.full((len(lags), K_MAX), np.nan)
    v_derot = np.full((len(lags), K_MAX), np.nan)
    v_amp = np.full((len(lags), K_MAX), np.nan)
    v_shaft = np.full(len(lags), np.nan)
    b_err = np.full(len(lags), np.nan)
    theta = None
    for i, lag in enumerate(lags):
        if lag >= z.shape[-1] // 3:
            continue
        d = shaft_g[:, lag:] - shaft_g[:, :-lag]
        pair = {(a, b): float(np.var(d[a] - d[b])) for a in range(3) for b in range(3) if a < b}
        bg = np.array(
            [
                0.5 * (pair[(0, 1)] + pair[(0, 2)] - pair[(1, 2)]),
                0.5 * (pair[(0, 1)] + pair[(1, 2)] - pair[(0, 2)]),
                0.5 * (pair[(0, 2)] + pair[(1, 2)] - pair[(0, 1)]),
            ]
        )
        bg = np.maximum(bg, 1e-12)
        inv = 1.0 / bg
        b_err[i] = float(1.0 / inv.sum())
        dth = (d * inv[:, None]).sum(0) * b_err[i]
        v_shaft[i] = float(np.var(dth)) - b_err[i]
        if i == 0:
            theta = np.cumsum(np.concatenate([[0.0], dth[::lag][: z.shape[-1] // lag]]))
        for j, k in enumerate(orders):
            prod = z[j, lag:] * np.conj(z[j, :-lag])
            v_raw[i, j] = circ_var(prod)
            v_derot[i, j] = circ_var(prod * np.exp(-1j * k * dth))
            v_amp[i, j] = amp_var(z[j], lag)
    leak = (orders[None, :] ** 2) * b_err[:, None]
    v_net = v_derot - leak - floor
    v_amp_net = v_amp - floor_a
    # cross-order correlation of the log-amplitude increments at 200 ms among
    # the gated orders up to 12: pure gain modulation would make it ~1
    lag_c = lags[LAGS_MS.index(200)]
    la = np.log(np.abs(z[:12]) + 1e-300)
    dla = la[:, lag_c:] - la[:, :-lag_c]
    amp_corr = np.corrcoef(dla)
    keep12 = gated[:12]
    # a display-only shaft phase on the full grid (inverse-variance weights of
    # the shortest lag applied to the three group phases)
    assert theta is not None
    theta = np.interp(np.arange(z.shape[-1]), np.arange(theta.size) * lags[0], theta)

    # ── figures ────────────────────────────────────────────────────────────
    taus = np.array(LAGS_MS) / 1e3
    # 1. spectrum with the orders
    f, p = signal.welch(x, fs=FS, nperseg=16384)
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.semilogy(f, p, lw=0.6, color="k")
    for k in orders:
        ax.axvline(k * rate, color="C1" if k in MOTOR_ORDERS else "C0", lw=0.4, alpha=0.5)
    for k in MOTOR_ORDERS:
        ax.annotate(f"k={k}", (k * rate, p[np.argmin(np.abs(f - k * rate))] * 3), color="C1")
    ax.set(
        xlim=(0, 8000), xlabel="Hz", ylabel="PSD", title=f"{RECORDING} mic {mic}, {rate:.2f} rev/s"
    )
    fig.tight_layout()
    fig.savefig(OUT / "spectrum.png", dpi=130)
    plt.close(fig)

    # 2. phase tracks (de-rotated, unwrapped for display)
    fig, axes = plt.subplots(len(SHOW_ORDERS), 1, figsize=(11, 1.4 * len(SHOW_ORDERS)), sharex=True)
    for ax, k in zip(axes, SHOW_ORDERS):
        zz = z[k - 1] * np.exp(-1j * k * theta)
        ax.plot(t, np.unwrap(np.angle(zz)), lw=0.6, color="C1" if k in MOTOR_ORDERS else "C0")
        ax.set_ylabel(f"k={k}\nrad", fontsize=8)
        ax.text(
            0.99,
            0.85,
            f"SNR {10 * np.log10(snr_eff[k - 1]):.0f} dB",
            transform=ax.transAxes,
            ha="right",
            fontsize=8,
        )
    axes[-1].set_xlabel("s")
    axes[0].set_title("phase of each order after removing k × shaft phase (unwrapped for display)")
    fig.tight_layout()
    fig.savefig(OUT / "phase_tracks.png", dpi=130)
    plt.close(fig)

    # 3. increment variance vs lag per order, with each order's noise floor
    colors = {k: f"C{i}" for i, k in enumerate(SHOW_ORDERS)}
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True)
    for ax, (vv, title) in zip(
        axes, ((v_raw, "raw increment"), (v_derot, "after removing k × shaft"))
    ):
        for k in SHOW_ORDERS:
            c = colors[k]
            ax.loglog(
                taus,
                vv[:, k - 1],
                "o-",
                ms=3,
                color=c,
                lw=2 if k in MOTOR_ORDERS else 1,
                label=f"k={k} ({10 * np.log10(snr_eff[k - 1]):.0f} dB)",
            )
            ax.loglog(taus, floor[:, k - 1], ":", color=c, lw=0.8)
        ax.loglog(taus, v_shaft, "k--", lw=1, label="Var[Δθ] shaft (k = 1)")
        ax.axhline(3.0, color="r", lw=0.5, label="estimator ceiling")
        ax.set(xlabel="lag τ [s]", title=title, ylim=(1e-4, 10))
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("Var[Δφ] (circular), rad²   (dotted: that order's noise floor)")
    axes[1].legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "variance_vs_lag.png", dpi=130)
    plt.close(fig)

    # 4. net independent term vs k at three lags, motor orders flagged
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for ms, c in ((50, "C0"), (200, "C2"), (500, "C3")):
        i = LAGS_MS.index(ms)
        ok = gated & (v_derot[i] < 3.0)
        ax.plot(orders[ok], v_net[i, ok], "o", ms=4, color=c, label=f"τ = {ms} ms")
        kk = np.arange(4, 60)
        ax.plot(kk, 0.09148 * kk**1.082 * (ms / 1e3) ** 0.434, color=c, lw=0.8, alpha=0.6)
    for k in MOTOR_ORDERS:
        ax.axvline(k, color="C1", lw=0.8, alpha=0.6)
    ax.axhline(0, color="k", lw=0.5)
    ax.set(
        xlabel="order k",
        ylabel="V_ε net, rad²",
        xlim=(0, 60),
        ylim=(-0.6, 4),
        title="independent per-order term after floor and shaft removal (lines: the 12-recording power law; orange: motor orders)",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "veps_vs_k.png", dpi=130)
    plt.close(fig)

    # 5. the motor order against aerodynamic orders: coherence |gamma| vs lag
    fig, ax = plt.subplots(figsize=(8, 4.8))
    for k in (2, 4, 10, 20, 40, 42):
        g = np.exp(-0.5 * np.maximum(v_net[:, k - 1], 0))
        ax.semilogx(
            taus,
            g,
            "o-",
            ms=4,
            lw=2.5 if k in MOTOR_ORDERS else 1,
            color=colors[k],
            label=f"k={k} ({10 * np.log10(snr_eff[k - 1]):.0f} dB){'  motor' if k in MOTOR_ORDERS else ''}",
        )
    law42 = np.exp(-0.5 * 0.09148 * 42**1.082 * taus**0.434)
    ax.semilogx(
        taus, law42, "--", color=colors[42], lw=1, label="k=42 if it followed the aerodynamic law"
    )
    ax.set(
        xlabel="lag τ [s]",
        ylabel="|γ| = exp(−V_ε/2): coherence left after floor and shaft removal",
        ylim=(0, 1.05),
        title="is the motor order more phase-coherent?",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, loc="lower left")
    fig.tight_layout()
    fig.savefig(OUT / "motor_orders.png", dpi=130)
    plt.close(fig)

    # 6. simulation: why additive noise does not hide a random walk
    rng = np.random.default_rng(0)
    n = int(60 * FS_PHASE)
    sim_lags = lags[:-1]
    sim_taus = np.array(sim_lags) / FS_PHASE
    d_walk = 0.5  # rad^2/s
    walk = np.cumsum(rng.standard_normal(n) * np.sqrt(d_walk / FS_PHASE))
    sos = signal.butter(6, band, btype="low", fs=FS_PHASE, output="sos")
    fig, ax = plt.subplots(figsize=(8, 5))
    for snr_db, c in ((20, "C0"), (6, "C1"), (0, "C3")):
        amp = np.sqrt(10 ** (snr_db / 10))
        noise = signal.sosfiltfilt(
            sos, (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
        )
        noise /= np.sqrt(np.mean(np.abs(noise) ** 2))
        for walk_on, ls in ((False, ":"), (True, "-")):
            zz = amp * np.exp(1j * (walk if walk_on else 0.0)) + noise
            v = [circ_var(zz[lag:] * np.conj(zz[:-lag])) for lag in sim_lags]
            ax.loglog(
                sim_taus,
                v,
                ls,
                color=c,
                marker="o",
                ms=3,
                label=f"SNR {snr_db} dB, {'walk + noise' if walk_on else 'noise only'}",
            )
    ax.loglog(sim_taus, 2 * d_walk * sim_taus, "k--", lw=0.8, label="2 D τ (the walk alone)")
    ax.axhline(3.0, color="r", lw=0.5, label="estimator ceiling")
    ax.set(
        xlabel="lag τ [s]",
        ylabel="Var[Δφ] (circular), rad²",
        ylim=(1e-3, 10),
        title="simulated line in band-limited noise: noise is flat in τ, a random walk grows",
    )
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "noise_vs_walk.png", dpi=130)
    plt.close(fig)

    # 7. phase vs log-amplitude increments: is the per-order term a shape
    # change (both equal), a gain (amplitude only) or a timing (phase only)?
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for k in (4, 10, 20, 40, 42):
        axes[0].loglog(
            taus,
            np.maximum(v_net[:, k - 1], 1e-4),
            "o-",
            ms=4,
            color=colors[k],
            lw=2.5 if k in MOTOR_ORDERS else 1,
            label=f"phase k={k}",
        )
        axes[0].loglog(
            taus,
            np.maximum(v_amp_net[:, k - 1], 1e-4),
            "s--",
            ms=4,
            color=colors[k],
            lw=2.5 if k in MOTOR_ORDERS else 1,
            label=f"log-amp k={k}",
        )
    axes[0].set(
        xlabel="lag τ [s]",
        ylabel="increment variance, floor removed",
        title="phase (solid) vs log-amplitude (dashed), same order",
        ylim=(1e-3, 10),
    )
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=7, ncol=2)
    im = axes[1].imshow(
        np.where(keep12[:, None] & keep12[None, :], amp_corr, np.nan),
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
        extent=(0.5, 12.5, 12.5, 0.5),
    )
    axes[1].set(
        xlabel="order",
        ylabel="order",
        title="correlation of log-amplitude increments (200 ms), orders 1-12",
    )
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(OUT / "phase_vs_amplitude.png", dpi=130)
    plt.close(fig)

    # 8. line shapes: baseband spectrum of each order before and after the
    # shaft phase is removed. A needle that survives at 0 Hz is the part of
    # the line phase-locked to the shaft; what is left around it is the
    # per-order pedestal.
    nper = 2048
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5))
    for ax, k in zip(axes.ravel(), (4, 10, 20, 30, 40, 42)):
        zz = z[k - 1]
        zd = zz * np.exp(-1j * k * theta)
        fr, pr = signal.welch(zz, fs=FS_PHASE, nperseg=nper, return_onesided=False)
        fd, pd = signal.welch(zd, fs=FS_PHASE, nperseg=nper, return_onesided=False)
        fr, pr, fd, pd = (np.fft.fftshift(a) for a in (fr, pr, fd, pd))
        ax.semilogy(fr, pr, lw=0.6, color="grey", label="constant carrier")
        ax.semilogy(
            fd,
            pd,
            lw=0.8,
            color="C1" if k in MOTOR_ORDERS else "C0",
            label="k × shaft phase removed",
        )
        ax.set(
            xlim=(-12, 12),
            ylim=(pd.max() * 1e-3, pd.max() * 2),
            xlabel="Hz off k × shaft rate",
            title=f"k={k} ({10 * np.log10(snr_eff[k - 1]):.0f} dB)",
        )
        ax.axvline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("baseband spectrum of each order on mic 2")
    fig.tight_layout()
    fig.savefig(OUT / "line_shapes.png", dpi=120)
    plt.close(fig)

    # 9. coherence left at long lags per order, from the lag law: |gamma|^2 at
    # 1 s and 2 s after floor and shaft-leak removal is the fraction of the
    # line's power still phase-locked to the shaft at that lag.
    i1, i2 = LAGS_MS.index(1000), LAGS_MS.index(2000)
    share1 = np.exp(-np.maximum(v_net[i1], 0.0))
    share2 = np.exp(-np.maximum(v_net[i2], 0.0))
    share = share2
    fig, ax = plt.subplots(figsize=(10, 4))
    ok = gated & (v_derot[i2] < 3.0)
    ax.plot(orders[gated], share1[gated], "s", ms=4, color="C2", alpha=0.6, label="at 1 s")
    ax.plot(orders[gated], share2[gated], "o", ms=4, color="C0", label="at 2 s")
    for k in MOTOR_ORDERS:
        if gated[k - 1]:
            ax.plot(
                [k],
                [share2[k - 1]],
                "o",
                ms=10,
                mfc="none",
                mec="C1",
                mew=2,
                label=f"motor order k={k}",
            )
    ax.set(
        xlabel="order k",
        ylabel="|γ|² = coherent fraction of the line",
        ylim=(0, 1.05),
        xlim=(0, 60),
        title="fraction of each order's power still phase-locked to the shaft after 1 s and 2 s",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "coherent_share.png", dpi=130)
    plt.close(fig)
    print(
        "coherence^2 at 1 s / 2 s:",
        {
            int(k): (round(float(share1[k - 1]), 2), round(float(share2[k - 1]), 2))
            for k in (2, 4, 6, 8, 10, 12, 16, 20, 24, 30, 36, 40, 42)
            if gated[k - 1]
        },
    )

    rows = {
        "recording": RECORDING,
        "mic": mic,
        "wind_ratio_db": wind_db.round(1).tolist(),
        "rate_survey": rate0,
        "rate": rate,
        "segment": [start_s, span_s],
        "lags_ms": list(LAGS_MS),
        "orders": orders.tolist(),
        "snr_eff_db": (10 * np.log10(snr_eff)).round(1).tolist(),
        "gated": gated.tolist(),
        "floor": floor.round(4).tolist(),
        "v_shaft": v_shaft.round(5).tolist(),
        "b_err": b_err.round(6).tolist(),
        "v_raw": v_raw.round(4).tolist(),
        "v_derot": v_derot.round(4).tolist(),
        "v_net": v_net.round(4).tolist(),
        "v_amp": v_amp.round(4).tolist(),
        "v_amp_net": v_amp_net.round(4).tolist(),
        "floor_amp": floor_a.round(4).tolist(),
        "amp_corr_200ms_orders_1_12": amp_corr.round(3).tolist(),
        "coherent_share": [None if np.isnan(s) else round(float(s), 3) for s in share],
    }
    (OUT / "one_rotor.json").write_text(json.dumps(rows))
    print(
        f"mic {mic}, wind ratio dB per mic {wind_db.round(1)}, rate {rate0:.3f} -> {rate:.3f} rev/s, segment {span_s:.1f} s"
    )
    print(
        f"V_shared(50 ms) {v_shaft[LAGS_MS.index(50)]:.2e}  (500 ms) {v_shaft[LAGS_MS.index(500)]:.2e}"
    )
    i200 = LAGS_MS.index(200)
    print("phase vs log-amp net at 50 / 200 / 500 ms:")
    for k in (2, 4, 6, 8, 10, 20, 30, 40, 42):
        print(
            f"  k={k:2d} phase {v_net[LAGS_MS.index(50), k - 1]:.3f} {v_net[i200, k - 1]:.3f} {v_net[LAGS_MS.index(500), k - 1]:.3f}   log-amp {v_amp_net[LAGS_MS.index(50), k - 1]:.3f} {v_amp_net[i200, k - 1]:.3f} {v_amp_net[LAGS_MS.index(500), k - 1]:.3f}"
        )
    print("log-amp increment corr (200 ms) orders 1-12 (gated only):")
    print(np.where(keep12[:, None] & keep12[None, :], amp_corr, np.nan).round(2))
    for k in (2, 4, 10, 20, 30, 40, 41, 42, 43, 44, 62, 63, 64, 82, 83, 84, 85, 86):
        i50, i500 = LAGS_MS.index(50), LAGS_MS.index(500)
        print(
            f"k={k:3d} SNR {10 * np.log10(snr_eff[k - 1]):5.1f} dB gate {gated[k - 1]!s:5} "
            f"floor50 {floor[i50, k - 1]:.3f} ana {floor_analytic[k - 1]:.3f}  derot50 {v_derot[i50, k - 1]:.3f} net50 {v_net[i50, k - 1]:.3f} "
            f"derot500 {v_derot[i500, k - 1]:.3f} net500 {v_net[i500, k - 1]:.3f}  shaft50 {k**2 * v_shaft[i50]:.3f}"
        )


if __name__ == "__main__":
    main()
