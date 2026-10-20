"""Build the assets of ``docs/explainers/noise-v4-bench-fit.qmd``: what the v4
bench fits of the DREGON single-rotor ladders look like and sound like.

Two fits per motor, both one shaft law per ladder (``noise-v4-fit/2``):

* ``r3``  Round 3, the pedestal ``sigma2_k`` uncapped, 1500 Adam + 1200 + 600
  L-BFGS (``{motor}__s0__nocap__long.json``);
* ``r2``  Round 2, the pedestal capped at 0.6 (``{motor}__s0.json``).

For three settings of each ladder: the real recording and a time-domain render
of each fit (:func:`bench_v4.render_setting`, mic 0, the same 6 s of the span,
ONE gain per setting so the levels are comparable), spectrograms, long-term
spectra, zooms at the native periodogram resolution with the fit split into its
parts, the inter-line fold, the line-level flicker, and the fitted parameters.

Run from the repo root::

    PYTHONPATH=src python docs/explainers/noise-v4-bench-fit/build.py
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf
import torch
from scipy.signal import stft

from experiments.noise_model import bench_v4 as B

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FITS = Path("results/noise_v4/bench_law/restarts")
ARMS = {
    "r3": dict(file="{motor}__s0__nocap__long.json", label="R3 fit (pedestal uncapped)"),
    "r2": dict(file="{motor}__s0.json", label="R2 fit (pedestal capped at 0.6)"),
}
#: the listening arms, in display order
CLIP_ARMS = {
    "real": "real recording",
    "r3": "R3 fit, Gaussian-pedestal render",
    "r2": "R2 fit, Gaussian-pedestal render",
    "r3_logamp": "R3 fit, library log-amplitude render",
}
MOTORS = ("Motor1", "Motor3")
THROTTLES = (50, 70, 90)
MIC = 0
CLIP_S = 6.0
SR = B.SR
#: one colour per quantity, on every figure
COL = dict(
    data="0.55",
    r3="k",
    r2="tab:red",
    core="tab:blue",
    pedestal="tab:orange",
    floor="tab:green",
    wind="tab:purple",
    ambient="tab:brown",
    render="tab:cyan",
)
plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 9,
        "figure.dpi": 110,
    }
)


def db(x: Any) -> Any:
    return 10.0 * np.log10(np.maximum(x, 1e-30))


def load_model(lad: B.Ladder, motor: str, arm: str) -> tuple[B.LadderModel, dict[str, Any]]:
    pay = json.loads((FITS / ARMS[arm]["file"].format(motor=motor)).read_text())
    m = B.LadderModel(lad, sigma2_cap=pay["shared"]["pedestal"]["cap"])
    m.load_payload(pay)
    return m, pay


def parts(m: B.LadderModel, j: int) -> dict[str, np.ndarray]:
    """``(M, F)`` expected-periodogram parts of setting ``j``: the lines split
    into the coherent core (share ``e^{-sigma2}``, plus the motor family) and
    the pedestal, the floor, the wind, the ambient and the total."""
    with torch.no_grad():
        c = {k: v.cpu().numpy() for k, v in m.components(j).items()}
        u, pa = m.params["u_sigma2"].clone(), m.params["p_aero"].clone()
        s2 = m.sigma2().clone()
        m.params["p_aero"].sub_(10.0 * s2 / math.log(10.0))
        m.params["u_sigma2"].fill_(-60.0)  # sigma2 ~ e^-60 under either cap rule
        core = m.components(j)["lines"].cpu().numpy()
        m.params["u_sigma2"].copy_(u)
        m.params["p_aero"].copy_(pa)
    amb = m.lad.settings[j].ambient
    return dict(
        lines=c["lines"],
        core=core,
        pedestal=np.maximum(c["lines"] - core, 0.0),
        floor=c["floor"],
        wind=c["wind"],
        ambient=amb,
        total=c["lines"] + c["floor"] + c["wind"] + amb,
    )


def frac_oct(f: np.ndarray, x: np.ndarray, lo: float, hi: float, n_per_oct: int = 24):
    """Mean of ``x`` (last axis on ``f``) in ``1/n_per_oct``-octave bands."""
    edges = lo * 2.0 ** (np.arange(0, math.log2(hi / lo) * n_per_oct + 1) / n_per_oct)
    idx = np.digitize(f, edges) - 1
    ok = (idx >= 0) & (idx < edges.size - 1)
    cnt = np.bincount(idx[ok], minlength=edges.size - 1)
    out = np.stack(
        [
            np.bincount(idx[ok], weights=row[ok], minlength=edges.size - 1)
            for row in np.atleast_2d(x)
        ]
    ) / np.maximum(cnt, 1)
    centres = np.sqrt(edges[:-1] * edges[1:])
    good = cnt > 0
    return centres[good], out[:, good].squeeze()


def line_offset(f: np.ndarray, carrier: float) -> np.ndarray:
    return np.abs(f - np.round(f / carrier) * carrier)


def spectrogram_db(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    f, t, z = stft(x, fs=SR, nperseg=2048, noverlap=2048 - 512, window="hann")
    return f, t, db(np.abs(z) ** 2)


def flicker(x: np.ndarray, carrier: float, orders: range) -> float:
    """Median over ``orders`` of the frame-to-frame sd (dB) of the 64 ms STFT
    power in the bin of ``k f_r``: how much a line's level jumps."""
    f, _t, z = stft(x, fs=SR, nperseg=1024, noverlap=512, window="hann")
    p = db(np.abs(z) ** 2)
    sds = []
    for k in orders:
        i = int(np.argmin(np.abs(f - k * carrier)))
        if 0 < i < f.size - 1 and k * carrier < B.BAND_HZ[1]:
            sds.append(float(np.std(p[i])))
    return float(np.median(sds)) if sds else float("nan")


def _gauss_envelope(rho: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Complex Gaussian sequence with autocorrelation ``rho`` (lags 0..n-1,
    ``rho[0] = 1``) by circulant embedding of length ``2 n``."""
    n = rho.size
    r = np.concatenate([rho, [0.0], rho[:0:-1]])
    S = np.maximum(np.fft.fft(r).real, 0.0)
    w = (rng.standard_normal(2 * n) + 1j * rng.standard_normal(2 * n)) / math.sqrt(2.0)
    return (np.fft.ifft(np.sqrt(S) * w) * math.sqrt(2 * n))[:n]


def render_gaussian(m: B.LadderModel, j: int, n: int, rng: np.random.Generator) -> np.ndarray:
    """``(M, n)`` render of setting ``j`` that realises the fitted spectrum
    with a GAUSSIAN pedestal: per aerodynamic order a steady tone of power
    ``P e^{-sigma2}`` plus a complex Gaussian envelope of power
    ``P (1 - e^{-sigma2})`` and autocorrelation
    ``expm1(sigma2 e^{-2 pi gamma tau}) / expm1(sigma2)`` (the v4 lag law's
    pedestal), both on one integrated-OU shaft phase; the motor family a
    steady tone on the same shaft; Gaussian floor + wind + ambient. Same second
    moments as ``bench_v4.render_setting``, no log-amplitude bursts."""
    s = m.lad.settings[j]
    with torch.no_grad():
        wa, wm, motor_k = m.line_weights(j)
        w_aero, w_motor = wa.cpu().numpy(), wm.cpu().numpy()
        comp = m.components(j)
        sig, lam = (float(v) for v in m.shaft(j))
    s2 = m.sigma2().detach().cpu().numpy()
    gam = m.gamma_am().detach().cpu().numpy()
    dt = 1.0 / SR
    a = math.exp(-lam * dt)
    nu = np.empty(n)
    nu[0] = rng.standard_normal() * sig
    xi = rng.standard_normal(n) * sig * math.sqrt(1.0 - a * a)
    for i in range(1, n):
        nu[i] = a * nu[i - 1] + xi[i]
    t = np.arange(n) * dt
    phase = 2.0 * math.pi * s.carrier * t + np.cumsum(nu) * dt
    tau = np.arange(n) * dt
    x = np.zeros((w_aero.shape[0], n))
    for k in range(1, s.k_max + 1):
        e = s2[k - 1]
        coh = math.exp(-e)
        env = math.sqrt(coh) * np.exp(1j * rng.uniform(0, 2 * math.pi))
        if e > 1e-9:
            rho = np.expm1(e * np.exp(-2.0 * math.pi * gam[k - 1] * tau)) / np.expm1(e)
            env = env + math.sqrt(1.0 - coh) * _gauss_envelope(rho, rng)
        carrier = np.real(env * np.exp(1j * k * phase))
        x += np.sqrt(2.0 * w_aero[:, k - 1])[:, None] * carrier[None, :]
    for i, k in enumerate(motor_k):
        carrier = np.cos(k * phase + rng.uniform(0, 2 * math.pi))
        x += np.sqrt(2.0 * w_motor[:, i])[:, None] * carrier[None, :]
    f_clip = np.fft.rfftfreq(n, d=dt)
    for spec in ((comp["floor"] + comp["wind"]).cpu().numpy(), s.ambient):
        sp = np.stack([np.interp(f_clip, s.freqs, row) for row in spec])
        white = np.fft.rfft(rng.standard_normal(x.shape), axis=-1)
        x += np.fft.irfft(white * np.sqrt(np.maximum(sp, 0.0)), n=n, axis=-1)
    return x


def main() -> None:
    os.chdir(ROOT)
    index: dict[str, Any] = dict(arms={a: v["label"] for a, v in ARMS.items()}, motors={})
    for motor in MOTORS:
        lad = B.build_ladder(motor)
        models = {a: load_model(lad, motor, a) for a in ARMS}
        calm = [m for m in range(lad.settings[0].power.shape[0]) if m not in B.WIND_MICS[motor]]
        rec: dict[str, Any] = dict(calm_mics=calm, settings={})
        rng = np.random.default_rng(0)
        js = [next(j for j, s in enumerate(lad.settings) if s.throttle == t) for t in THROTTLES]

        spec_fig, spec_ax = plt.subplots(
            len(CLIP_ARMS),
            len(js),
            figsize=(5.6 * len(js), 13),
            sharey=True,
            constrained_layout=True,
        )
        ltas_fig, ltas_ax = plt.subplots(
            2,
            len(js),
            figsize=(5.4 * len(js), 8.5),
            sharex=True,
            height_ratios=[3, 1.3],
            constrained_layout=True,
        )
        fold_fig, fold_ax = plt.subplots(3, len(js), figsize=(5.2 * len(js), 10), sharex=True)
        cache: dict[int, dict[str, Any]] = {}
        for col, j in enumerate(js):
            s = lad.settings[j]
            f = s.freqs
            band = (f >= B.BAND_HZ[0]) & (f <= B.BAND_HZ[1])
            pr = {a: parts(models[a][0], j) for a in ARMS}
            cache[j] = dict(pr=pr)
            x_real, span = B.load_audio(motor, s.throttle)
            seg = x_real[:, int(round(span[0] * SR)) : int(round(span[0] * SR)) + s.n]
            start = int(round(max(0.0, (s.n / SR - CLIP_S) / 2.0) * SR))
            stop = start + int(CLIP_S * SR)
            full = {
                "real": seg[:, start:stop],
                "r3": render_gaussian(models["r3"][0], j, stop - start, rng),
                "r2": render_gaussian(models["r2"][0], j, stop - start, rng),
                "r3_logamp": B.render_setting(models["r3"][0], j, rng)[:, start:stop],
            }
            clips = {name: x[MIC] for name, x in full.items()}
            gain = 0.5 / float(np.max(np.abs(clips["real"])))
            clip_info = {}
            for name, x in clips.items():
                y = x * gain
                clip_info[name] = dict(
                    n_clipped=int(np.sum(np.abs(y) > 1.0)),
                    rms_dbfs=float(db(np.mean(y**2))),
                )
                sf.write(HERE / f"{motor}_{s.throttle}_{name}.wav", np.clip(y, -1, 1), SR)
            # spectrograms on the real clip's colour scale
            specs = {name: spectrogram_db(x * gain) for name, x in clips.items()}
            vmax = float(np.percentile(specs["real"][2], 99.5))
            for row, name in enumerate(CLIP_ARMS):
                fs_, ts_, S = specs[name]
                ax = spec_ax[row, col]
                im = ax.pcolormesh(
                    ts_, fs_ / 1e3, S, vmin=vmax - 55, vmax=vmax, shading="auto", cmap="magma"
                )
                ax.set_ylim(0, B.BAND_HZ[1] / 1e3)
                ttl = CLIP_ARMS[name]
                ax.set_title(f"{s.throttle} % ({s.carrier:.0f} rev/s): {ttl}", fontsize=10)
                if col == 0:
                    ax.set_ylabel("frequency (kHz)")
                if row == len(CLIP_ARMS) - 1:
                    ax.set_xlabel("time (s)")
            spec_fig.colorbar(im, ax=spec_ax[:, col], shrink=0.6, label="dB (re real clip)")

            # long-term spectra, calm-mic mean, 1/24 octave
            data_m = s.power[calm].mean(axis=0)
            fc, d24 = frac_oct(f[band], data_m[band], 30.0, B.BAND_HZ[1])
            ax = ltas_ax[0, col]
            ax.semilogx(fc, db(d24), color=COL["data"], lw=2.4, label="data")
            curves = {}
            for a in ARMS:
                _, m24 = frac_oct(
                    f[band], pr[a]["total"][calm].mean(axis=0)[band], 30.0, B.BAND_HZ[1]
                )
                curves[a] = m24
                ax.semilogx(
                    fc,
                    db(m24),
                    color=COL[a],
                    lw=1.4,
                    ls="-" if a == "r3" else "--",
                    label=f"{a.upper()} expected",
                )
            f_clip = np.fft.rfftfreq(stop - start, d=1.0 / SR)
            b_clip = (f_clip >= B.BAND_HZ[0]) & (f_clip <= B.BAND_HZ[1])
            rend = {}
            for name, ls in (("r3", "-"), ("r3_logamp", ":")):
                _, rend[name] = frac_oct(
                    f_clip[b_clip],
                    B._periodogram(full[name])[calm].mean(axis=0)[b_clip],
                    30.0,
                    B.BAND_HZ[1],
                )
                ax.semilogx(
                    fc, db(rend[name]), color=COL["render"], lw=1.0, ls=ls, label=CLIP_ARMS[name]
                )
            for part in ("lines", "floor", "wind", "ambient"):
                _, q = frac_oct(
                    f[band], pr["r3"][part][calm].mean(axis=0)[band], 30.0, B.BAND_HZ[1]
                )
                ax.semilogx(
                    fc,
                    db(q),
                    color=COL["pedestal" if part == "lines" else part],
                    lw=0.9,
                    alpha=0.8,
                    label=f"R3 {part}",
                )
            ax.set_title(f"{motor} {s.throttle} %: long-term spectrum (1/24 oct)")
            ax.set_ylim(db(d24).min() - 25, db(d24).max() + 5)
            if col == 0:
                ax.set_ylabel("power (dB)")
            ax.grid(alpha=0.3, which="both")
            ax = ltas_ax[1, col]
            for a in ARMS:
                ax.semilogx(
                    fc,
                    db(d24) - db(curves[a]),
                    color=COL[a],
                    ls="-" if a == "r3" else "--",
                    label=a.upper(),
                )
            ax.semilogx(
                fc,
                db(rend["r3"]) - db(curves["r3"]),
                color=COL["render"],
                ls="-",
                label="Gaussian render − R3",
            )
            ax.semilogx(
                fc,
                db(rend["r3_logamp"]) - db(curves["r3"]),
                color=COL["render"],
                ls=":",
                label="log-amp render − R3",
            )
            ax.axhline(0, color="0.3", lw=0.8)
            ax.set_ylim(-6, 6)
            ax.set_xlabel("frequency (Hz)")
            if col == 0:
                ax.set_ylabel("data − model (dB)")
            ax.grid(alpha=0.3, which="both")

            # the fold: power against the distance to the nearest line
            off = line_offset(f, s.carrier)
            edges = np.geomspace(0.03, s.carrier / 2.0, 40)
            for row, (lo, hi) in enumerate(((300, 1000), (1000, 2000), (3000, 5000))):
                ax = fold_ax[row, col]
                sel = (f >= lo) & (f < hi)
                idx = np.digitize(off[sel], edges) - 1
                ok = (idx >= 0) & (idx < edges.size - 1)
                xc = np.sqrt(edges[:-1] * edges[1:])

                def fold(
                    arr: np.ndarray, sel=sel, idx=idx, ok=ok, edges=edges, calm=calm
                ) -> np.ndarray:
                    v = arr[calm].mean(axis=0)[sel]
                    tot = np.bincount(idx[ok], weights=v[ok], minlength=edges.size - 1)
                    n = np.bincount(idx[ok], minlength=edges.size - 1)
                    return np.where(n > 0, tot / np.maximum(n, 1), np.nan)

                ax.semilogx(xc, db(fold(s.power)), color=COL["data"], lw=2.6, label="data")
                ax.semilogx(xc, db(fold(pr["r3"]["total"])), color=COL["r3"], label="R3 total")
                ax.semilogx(
                    xc, db(fold(pr["r2"]["total"])), color=COL["r2"], ls="--", label="R2 total"
                )
                for part in ("core", "pedestal", "floor", "ambient"):
                    ax.semilogx(
                        xc, db(fold(pr["r3"][part])), color=COL[part], lw=1.0, label=f"R3 {part}"
                    )
                top = np.nanmax(db(fold(s.power)))
                ax.set_ylim(top - 45, top + 4)
                ax.set_title(f"{s.throttle} %, {lo}–{hi} Hz")
                if col == 0:
                    ax.set_ylabel("power (dB)")
                if row == 2:
                    ax.set_xlabel("|f − k·f_r| (Hz)")
                if row == 0 and col == 0:
                    ax.legend(loc="lower left", ncol=2)
                ax.grid(alpha=0.3, which="both")

            # zooms at the native resolution, three order windows
            zoom_fig, zoom_ax = plt.subplots(3, 1, figsize=(13, 11))
            k_hi = min(60, s.k_max - 3)
            for ax, (k0, k1) in zip(zoom_ax, ((0.5, 4.5), (15.5, 19.5), (k_hi - 0.5, k_hi + 3.5))):
                sel = (f >= k0 * s.carrier) & (f <= k1 * s.carrier)
                kern = np.ones(5) / 5.0
                dsm = np.convolve(s.power[calm].mean(axis=0), kern, mode="same")
                ax.plot(f[sel], db(dsm[sel]), color=COL["data"], lw=0.8, label="data (5-bin mean)")
                ax.plot(
                    f[sel],
                    db(pr["r3"]["total"][calm].mean(axis=0)[sel]),
                    color=COL["r3"],
                    lw=1.6,
                    label="R3 total",
                )
                ax.plot(
                    f[sel],
                    db(pr["r2"]["total"][calm].mean(axis=0)[sel]),
                    color=COL["r2"],
                    lw=1.2,
                    ls="--",
                    label="R2 total",
                )
                for part in ("core", "pedestal", "floor", "wind", "ambient"):
                    ax.plot(
                        f[sel],
                        db(pr["r3"][part][calm].mean(axis=0)[sel]),
                        color=COL[part],
                        lw=1.0,
                        label=f"R3 {part}",
                    )
                hi_db = db(dsm[sel]).max()
                ax.set_ylim(hi_db - 60, hi_db + 5)
                ax.set_xlim(f[sel][0], f[sel][-1])
                ax.set_title(
                    f"{motor} {s.throttle} %: orders {math.ceil(k0)}–{math.floor(k1)} "
                    f"({f[sel][0]:.0f}–{f[sel][-1]:.0f} Hz), mean of calm mics {calm}"
                )
                ax.set_ylabel("power (dB)")
                ax.grid(alpha=0.3)
            zoom_ax[0].legend(loc="upper right", ncol=4)
            zoom_ax[-1].set_xlabel("frequency (Hz)")
            zoom_fig.tight_layout()
            zoom_fig.savefig(HERE / f"zoom_{motor}_{s.throttle}.png")
            plt.close(zoom_fig)

            def ltas_err(curve: np.ndarray, ref: np.ndarray = d24) -> float:
                return float(np.mean(np.abs(db(ref) - db(curve))))

            rec["settings"][str(s.throttle)] = dict(
                key=s.key,
                carrier_rev_s=s.carrier,
                span_s=list(span),
                clip_s=[start / SR, stop / SR],
                gain_db=float(db(gain**2)),
                clips=clip_info,
                ltas_mean_abs_db=dict(r3=ltas_err(curves["r3"]), r2=ltas_err(curves["r2"])),
                render_vs_expected_ltas_mean_abs_db={
                    name: float(np.mean(np.abs(db(rend[name]) - db(curves["r3"])))) for name in rend
                },
                flicker_db={
                    name: {
                        f"k{a}-{b}": flicker(x, s.carrier, range(a, b + 1))
                        for a, b in ((1, 4), (10, 20), (40, 80))
                    }
                    for name, x in clips.items()
                },
            )
        spec_fig.savefig(HERE / f"spec_{motor}.png", bbox_inches="tight")
        handles: dict[str, Any] = {}
        for row in range(2):
            for h, lab in zip(*ltas_ax[row, 0].get_legend_handles_labels()):
                handles.setdefault(lab if row == 0 else f"residual: {lab}", h)
        ltas_fig.legend(
            list(handles.values()), list(handles), loc="outside lower center", ncol=4, fontsize=9
        )
        ltas_fig.savefig(HERE / f"ltas_{motor}.png")
        fold_fig.tight_layout()
        fold_fig.savefig(HERE / f"fold_{motor}.png")
        for fig in (spec_fig, ltas_fig, fold_fig):
            plt.close(fig)

        # the fitted parameters
        fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
        cmap = plt.get_cmap("viridis")
        for jj, s in enumerate(lad.settings):
            c = cmap(jj / (len(lad.settings) - 1))
            meas = db((10 ** (lad.tables[jj]["level_db"][calm] / 10)).mean(axis=0))
            m3 = models["r3"][0]
            with torch.no_grad():
                wa, wm, mk = m3.line_weights(jj)
            w = wa.cpu().numpy()[calm].mean(axis=0)
            if mk.size:
                w[mk - 1] += wm.cpu().numpy()[calm].mean(axis=0)
            k = np.arange(1, w.size + 1)
            kk = np.arange(1, meas.size + 1)
            ax[0, 0].semilogx(kk, meas, "o", ms=2.5, color=c, alpha=0.6)
            ax[0, 0].semilogx(k, db(w), color=c, lw=1.2, label=f"{s.throttle} %")
        ax[0, 0].set_title("line power: measured (dots) vs R3 fit (lines)")
        ax[0, 0].set_xlabel("order k")
        ax[0, 0].set_ylabel("line power, calm-mic mean (dB)")
        ax[0, 0].legend(ncol=2)
        k_all = np.arange(1, lad.k_prof + 1)
        mu_s, sd_s = B._log_sigma2_prior(k_all.astype(float))
        mu_g, sd_g = B._log_gamma_prior(k_all.astype(float))
        for a in ARMS:
            sh = models[a][1]["shared"]["pedestal"]
            ls = "-" if a == "r3" else "--"
            ax[0, 1].loglog(k_all, sh["sigma2"], color=COL[a], ls=ls, label=a.upper())
            ax[0, 2].loglog(k_all, sh["gamma_hz"], color=COL[a], ls=ls, label=a.upper())
        ax[0, 1].fill_between(
            k_all, np.exp(mu_s - sd_s), np.exp(mu_s + sd_s), color="0.85", label="prior ±1 sd"
        )
        ax[0, 1].axhline(B.SIGMA2_CAP, color=COL["r2"], lw=0.8, ls=":", label="R2 cap 0.6")
        ax[0, 1].set_title("pedestal σ²ₖ (coherent share e^−σ²)")
        ax[0, 1].set_xlabel("order k")
        ax[0, 1].legend()
        ax[0, 2].fill_between(
            k_all, np.exp(mu_g - sd_g), np.exp(mu_g + sd_g), color="0.85", label="prior ±1 sd"
        )
        ax[0, 2].set_title("pedestal width γₖ (Hz)")
        ax[0, 2].set_xlabel("order k")
        ax[0, 2].legend()
        car = np.array([s.carrier for s in lad.settings])
        quick = [B._quick_shaft(s.key) for s in lad.settings]
        for a in ARMS:
            st = models[a][1]["settings"]
            ls = "-o" if a == "r3" else "--s"
            ax[1, 0].plot(car, [x["sigma_nu_rad_s"] for x in st], ls, color=COL[a], label=a.upper())
            ax[1, 1].semilogy(car, [x["lam_per_s"] for x in st], ls, color=COL[a], label=a.upper())
        qc = [(c, q) for c, q in zip(car, quick) if q is not None]
        ax[1, 0].plot(
            [c for c, _ in qc],
            [q[0] for _, q in qc],
            "D",
            color="0.4",
            label="quick fit (per recording)",
        )
        ax[1, 1].semilogy(
            [c for c, _ in qc], [q[1] for _, q in qc], "D", color="0.4", label="quick fit"
        )
        ax[1, 0].set_title("shaft speed jitter σ_ν (rad/s)")
        ax[1, 1].set_title("shaft rate λ (1/s)")
        for a_ in (ax[1, 0], ax[1, 1]):
            a_.set_xlabel("rotor speed (rev/s)")
            a_.legend()
        for jj, s in enumerate(lad.settings):
            c = cmap(jj / (len(lad.settings) - 1))
            f = s.freqs
            off = line_offset(f, s.carrier)
            band = (f >= B.BAND_HZ[0]) & (f <= B.BAND_HZ[1]) & (off >= 0.4 * s.carrier)
            fc, mid = frac_oct(f[band], s.power[calm].mean(axis=0)[band], 30.0, B.BAND_HZ[1], 6)
            pr = cache[jj]["pr"]["r3"] if jj in cache else parts(models["r3"][0], jj)
            _, flo = frac_oct(f[band], pr["floor"][calm].mean(axis=0)[band], 30.0, B.BAND_HZ[1], 6)
            _, amb = frac_oct(f[band], s.ambient[calm].mean(axis=0)[band], 30.0, B.BAND_HZ[1], 6)
            ax[1, 2].semilogx(fc, db(mid), "-o", ms=3, color=c, label=f"{s.throttle} % data midway")
            ax[1, 2].semilogx(fc, db(flo), "--", color=c, lw=1.2)
            ax[1, 2].semilogx(fc, db(amb), ":", color=c, lw=1.0)
        ax[1, 2].set_title("between the lines: data (o), R3 floor (--), ambient (:)")
        ax[1, 2].set_xlabel("frequency (Hz)")
        ax[1, 2].set_ylabel("power (dB)")
        ax[1, 2].legend(fontsize=8)
        for a_ in ax.flat:
            a_.grid(alpha=0.3, which="both")
        fig.suptitle(f"{motor}: fitted parameters", fontsize=13)
        fig.tight_layout()
        fig.savefig(HERE / f"params_{motor}.png")
        plt.close(fig)

        rec["fits"] = {
            a: dict(
                path=str(FITS / ARMS[a]["file"].format(motor=motor)),
                total_nats=pay["objective"]["total_nats"],
                converged=pay["optimiser"]["converged"],
                restart_gain_per_cell=pay["optimiser"]["restart_gain_per_cell"],
                shaft=pay["shared"]["shaft"],
                alpha_aero=pay["shared"]["alpha_aero"],
                alpha_motor=pay["shared"]["alpha_motor"],
                floor_b0=pay["shared"]["floor"]["b0"],
                floor_b1=pay["shared"]["floor"]["b1"],
            )
            for a, (_m, pay) in models.items()
        }
        index["motors"][motor] = rec
        print(motor, json.dumps(rec["settings"], indent=1)[:3000])
    (HERE / "index.json").write_text(json.dumps(index, indent=1))


if __name__ == "__main__":
    main()
