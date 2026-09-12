"""Numerical checks for the Bretthorst-style derivations of the explainer's
Part V, run on one DREGON single-motor bench recording.

Three results are checked, each one a claim the derivation makes:

1. the sufficient statistic for the shaft rate is the comb-summed periodogram,
   and its Student-t posterior is sharply peaked -- the width is compared with
   the closed-form  sigma_omega^2 = 24 sigma^2 / (N^3 sum_k k^2 a_k^2);
2. the octave ambiguity is a MODEL SELECTION problem, not a likelihood
   problem: with flat amplitude priors the half-rate comb ties or wins, with
   proper priors its unused orders cost prior volume and it loses by a wide
   margin (the Occam factor of Bretthorst Ch. 5);
3. the maximum-likelihood order power under phase noise is the BAND-INTEGRATED
   line power, and peak-picking is biased low by an amount that grows with the
   order because the line broadens.

Run from the repo root::

    PYTHONPATH=src python docs/explainers/rig-model-listening/bayes_comb.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
RECORDING = "motor_Motor1_70"
SR = 44100.0
N = 1 << 18  # 5.94 s of data: the "N" of the derivation
PAD = 1 << 22  # zero-padding only, so h_k(omega) is available off-bin
K_MAX = 100
DPI = 150


def load_mic0() -> np.ndarray:
    from data_processing.frames import meta_dict
    from data_processing.streams import iter_published_frames

    for f in iter_published_frames("DREGON-frames"):
        m = meta_dict(f)
        if str(m.get("recording_id")) == RECORDING:
            x = np.asarray(f["audio"].data, dtype=np.float64)
            return x[0] if x.ndim > 1 else x
    raise KeyError(RECORDING)


class Comb:
    """The projections h_j(omega) of one data segment onto the comb model.

    The model functions are cos(k omega t), sin(k omega t) on a uniform grid
    with the time origin at the centre of the record, so that (Bretthorst
    6.11-6.12) the g_jk matrix is N/2 times the identity to O(1/N) and the
    orthonormal functions are H = sqrt(2/N) cos, sqrt(2/N) sin. Then
    h_{2k-1} = sqrt(2/N) Re D(k omega), h_{2k} = -sqrt(2/N) Im D(k omega),
    and h_{2k-1}^2 + h_{2k}^2 = (2/N) |D(k omega)|^2.
    """

    def __init__(self, d: np.ndarray) -> None:
        self.d = d - d.mean()
        self.N = self.d.size
        self.dbar2 = float(np.mean(self.d**2))
        # centre the time origin: multiply by the linear-phase that shifts t=0
        # to the middle, which the |D| magnitudes below do not even need
        self.spec = np.fft.rfft(self.d, PAD)
        self.df = SR / PAD

    def line_power(self, rate: float, k_max: int) -> np.ndarray:
        """(2/N)|D(k omega)|^2 per order -- the per-order contribution to m hbar^2."""
        ks = np.arange(1, k_max + 1)
        idx = np.round(ks * rate / self.df).astype(int)
        keep = idx < self.spec.size - 1
        out = np.zeros(k_max)
        out[keep] = (2.0 / self.N) * np.abs(self.spec[idx[keep]]) ** 2
        return out

    def student_t_log(self, rate: float, k_max: int) -> float:
        """log P(omega | D, I) with sigma unknown -- Bretthorst (3.17)."""
        p = self.line_power(rate, k_max)
        m = 2 * k_max
        mh2 = float(p.sum())
        frac = mh2 / (self.N * self.dbar2)
        if frac >= 1.0:
            return -np.inf
        return 0.5 * (m - self.N) * np.log1p(-frac)

    def log_evidence(self, rate: float, k_max: int, delta2: float, sigma2: float) -> float:
        """log p(D | omega, model) with a PROPER N(0, delta^2) prior on every
        amplitude and sigma^2 known -- the Occam-factor form.

        For orthonormal model functions each amplitude contributes
        independently: with lambda = 1 (H already normalised) the marginal of
        one amplitude is N(0, sigma^2 + delta^2), so

            log p = sum_j [ -0.5 log(1 + delta^2/sigma^2)
                            + h_j^2 delta^2 / (2 sigma^2 (sigma^2 + delta^2)) ].

        The first term is the Occam factor: it is paid once per amplitude
        whether the data want it or not.
        """
        p = self.line_power(rate, k_max)  # = h_{2k-1}^2 + h_{2k}^2
        m = 2 * k_max
        occam = -0.5 * m * np.log1p(delta2 / sigma2)
        fit = float(p.sum()) * delta2 / (2 * sigma2 * (sigma2 + delta2))
        return occam + fit


def local_floor(spec_mag2: np.ndarray, centre: float, half: float) -> float:
    a = spec_mag2[int(round(centre + 0.25 * half)) : int(round(centre + 0.75 * half))]
    b = spec_mag2[int(round(centre - 0.75 * half)) : int(round(centre - 0.25 * half))]
    if not a.size or not b.size:
        return np.nan
    return float(np.median(np.concatenate([a, b])))


def _width_law(ks: np.ndarray, width_hz: np.ndarray, margin_db: np.ndarray) -> dict[str, object]:
    """Log-log slope of the measured width over the trustworthy order range,
    plus the affine fit gamma = gamma0 + c k the renderer's law would use.

    A second moment taken over a window a third of the comb spacing wide is
    dominated by the floor residual unless the line is well above it, so only
    orders with a 28 dB peak margin are fitted -- on this rig those are the
    even orders, the odd ones being 20 dB weaker. Above k ~ 24 the moment
    saturates against the window and is reported but not fitted."""
    ok = np.isfinite(width_hz) & (ks >= 2) & (ks <= 24) & (margin_db >= 28.0)
    if ok.sum() < 5:
        return {}
    slope = float(np.polyfit(np.log(ks[ok]), np.log(width_hz[ok]), 1)[0])
    c, g0 = np.polyfit(ks[ok], width_hz[ok], 1)
    sigma = float(c) / np.sqrt(2 * np.log(2.0))
    return dict(
        range="k = 2..16",
        loglog_slope=round(slope, 3),
        gamma0_hz=round(float(g0), 3),
        gamma_slope_hz_per_order=round(float(c), 4),
        implied_sigma_rps=round(sigma, 4),
        saturates_above_k=int(ks[np.isfinite(width_hz)][-1]) if np.isfinite(width_hz).any() else 0,
    )


def main() -> None:
    x = load_mic0()
    start = (x.size - N) // 2
    seg = x[start : start + N]
    comb = Comb(seg)

    # ---- 1. the posterior for the shaft rate -------------------------------
    coarse = np.arange(30.0, 110.0, 0.01)
    lp = np.array([comb.student_t_log(float(s), 40) for s in coarse])
    s_hat0 = float(coarse[int(np.argmax(lp))])
    fine = np.arange(s_hat0 - 0.05, s_hat0 + 0.05, 0.00005)
    lpf = np.array([comb.student_t_log(float(s), K_MAX) for s in fine])
    s_hat = float(fine[int(np.argmax(lpf))])

    # posterior width: fit a parabola to the log posterior near the peak
    w = np.abs(fine - s_hat) < 0.004
    c = np.polyfit(fine[w] - s_hat, lpf[w], 2)
    sigma_post = float(np.sqrt(-1.0 / (2.0 * c[0]))) if c[0] < 0 else np.nan

    # closed-form prediction
    p_k = comb.line_power(s_hat, K_MAX)  # = h_{2k-1}^2 + h_{2k}^2
    mag2 = np.abs(comb.spec) ** 2
    half = s_hat / comb.df / 2.0
    floor = np.array([local_floor(mag2, (k + 1) * s_hat / comb.df, half) for k in range(K_MAX)])
    # White noise of variance sigma^2 contributes E[h_j^2] = sigma^2 to every
    # model function, so E[p_k] = (N/2) a_k^2 + 2 sigma^2 and the noise floor
    # of the padded periodogram is E|D|^2 = N sigma^2 / 2.
    sigma2 = float(np.nanmedian(floor)) * 2.0 / comb.N
    a2 = np.clip((2.0 / comb.N) * (p_k - 2.0 * sigma2), 0.0, None)
    ks = np.arange(1, K_MAX + 1)
    # sigma_omega^2 = 24 sigma^2 / (N^3 sum_k k^2 a_k^2), omega in rad/sample.
    # Only orders whose line is actually detected carry information; a noise
    # order contributes k^2 * 0 in expectation but k^2 * noise in practice.
    detected = p_k > 4.0 * sigma2
    denom = float(np.sum(ks[detected] ** 2 * a2[detected]))
    sigma_omega = np.sqrt(24.0 * sigma2 / (comb.N**3 * denom))
    sigma_rate = sigma_omega * SR / (2 * np.pi)

    # ---- 2. the octave ambiguity as model selection ------------------------
    # The prior scale belongs to the ORTHONORMAL amplitudes A_j (E[h_j] = A_j,
    # Var = sigma^2), not to the physical amplitudes: A_j = sqrt(N/2) a_j. Per
    # order p_k = h_{2k-1}^2 + h_{2k}^2, so the per-amplitude scale is p_k / 2.
    delta2 = float(np.nanmedian(p_k[detected] / 2.0)) if detected.any() else sigma2
    rows = []
    for name, rate, kk in (
        ("omega (fitted)", s_hat, K_MAX),
        ("omega / 2", s_hat / 2, 2 * K_MAX),
        ("omega / 3", s_hat / 3, 3 * K_MAX),
        ("2 omega", s_hat * 2, K_MAX // 2),
    ):
        rows.append(
            dict(
                model=name,
                rate=round(rate, 4),
                k_max=kk,
                student_t=round(comb.student_t_log(rate, kk), 1),
                evidence=round(comb.log_evidence(rate, kk, delta2, sigma2), 1),
            )
        )

    # ---- 3. peak-picked versus band-integrated power, and the line width ---
    # Read on the UNPADDED periodogram (bin 1/T = 0.168 Hz), so neighbouring
    # bins are independent and a sum of excesses is an unbiased power estimate
    # (no clipping: the floor-subtracted noise must be allowed to cancel).
    raw = np.abs(np.fft.rfft(comb.d)) ** 2
    df_raw = SR / comb.N
    peak_db = np.full(K_MAX, np.nan)
    band_db = np.full(K_MAX, np.nan)
    width_hz = np.full(K_MAX, np.nan)
    margin_db = np.full(K_MAX, -np.inf)
    n_band = 8  # +-1.34 Hz: wider than any line this rig shows at low order
    for i, k in enumerate(ks):
        c0 = k * s_hat / df_raw
        if c0 + 3 * n_band >= raw.size - 2:
            break
        fl = local_floor(raw, c0, s_hat / df_raw / 2.0)
        lo, hi = int(round(c0 - 2)), int(round(c0 + 2)) + 1
        peak_db[i] = 10 * np.log10(max(raw[lo:hi].max() - fl, 1e-30))
        margin_db[i] = 10 * np.log10(raw[lo:hi].max() / fl)
        blo, bhi = int(round(c0)) - n_band, int(round(c0)) + n_band + 1
        e = raw[blo:bhi] - fl
        band_db[i] = 10 * np.log10(max(e.sum(), 1e-30))
        # The width needs a window WIDER than the line, so it is read over a
        # third of the comb spacing; the power above is read over a narrow one,
        # where the floor contributes little variance.
        wide = max(n_band, int(0.33 * s_hat / df_raw))
        wlo, whi = int(round(c0)) - wide, int(round(c0)) + wide + 1
        if wlo < 0 or whi >= raw.size:
            continue
        ew = np.clip(raw[wlo:whi] - fl, 0.0, None)
        f_off = (np.arange(wlo, whi) - c0) * df_raw
        if ew.sum() > 0:
            width_hz[i] = float(np.sqrt((ew * f_off**2).sum() / ew.sum()))

    gap = band_db - peak_db
    summary = dict(
        recording=RECORDING,
        seconds=round(N / SR, 3),
        rate_rps=round(s_hat, 5),
        posterior_sigma_rps=round(sigma_post, 6),
        predicted_sigma_rps=float(f"{sigma_rate:.3e}"),
        noise_var_per_function=float(f"{sigma2:.4e}"),
        models=rows,
        band_minus_peak_db={
            "k<=8": round(float(np.nanmean(gap[:8])), 2),
            "k 9-32": round(float(np.nanmean(gap[8:32])), 2),
            "k 33-64": round(float(np.nanmean(gap[32:64])), 2),
            "k 65-100": round(float(np.nanmean(gap[64:])), 2),
        },
        line_width_hz={
            f"k={k}": (round(float(width_hz[k - 1]), 3) if np.isfinite(width_hz[k - 1]) else None)
            for k in (2, 4, 8, 16, 32, 48, 64, 80)
        },
        # The slope is fitted only where the readout is trustworthy: above
        # k ~ 20 the second moment saturates against the window and the
        # floor-subtraction residual, so a slope taken over all orders is a
        # measurement artefact, not a law.
        width_law=_width_law(ks, width_hz, margin_db),
        width_fit_orders=[
            int(k) for k in ks[np.isfinite(width_hz) & (ks <= 24) & (margin_db >= 28.0)]
        ],
    )
    print(json.dumps(summary, indent=1))
    (HERE / "bayes_comb.json").write_text(json.dumps(summary, indent=1))

    # ---- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
    ax = axes[0]
    zoom = np.abs(fine - s_hat) < 0.0012
    ax.plot((fine[zoom] - s_hat) * 1000, lpf[zoom] - lpf.max(), lw=1.8, color="#157F3D")
    ax.axvspan(-sigma_post * 1e3, sigma_post * 1e3, color="#157F3D", alpha=0.13, lw=0)
    ax.axhline(-0.5, color="0.5", lw=0.9, ls=":")
    ax.set_xlabel(r"$s - \hat s$ (milli-rev/s)")
    ax.set_ylabel(r"$\log P(s \mid D, I)$")
    ax.set_title(
        rf"Student-t posterior, $\hat s$={s_hat:.4f} rev/s" + "\n"
        rf"width {sigma_post * 1e3:.3f} vs closed form {sigma_rate * 1e3:.3f} m-rev/s"
    )
    ax.set_ylim(-40, 3)
    ax.grid(alpha=0.3)

    ax = axes[1]
    names = [r["model"] for r in rows]
    st = np.array([r["student_t"] for r in rows])
    ev = np.array([r["evidence"] for r in rows])
    xx = np.arange(len(rows))
    ax.bar(xx - 0.2, st - st.max(), width=0.4, color="#9A6700", label="flat priors (Student-t)")
    ax.bar(xx + 0.2, ev - ev.max(), width=0.4, color="#157F3D", label="proper priors (evidence)")
    ax.set_xticks(xx, names, rotation=20)
    ax.set_ylabel("log probability, relative to best")
    ax.set_title("The octave ambiguity is an Occam question")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    law = _width_law(ks, width_hz, margin_db)
    law_slope = law.get("loglog_slope")
    law_g0 = float(law.get("gamma0_hz", 0.0))  # type: ignore[arg-type]
    law_c = float(law.get("gamma_slope_hz_per_order", 0.0))  # type: ignore[arg-type]
    ax = axes[2]
    ok = np.isfinite(width_hz)
    strong = ok & (margin_db >= 28.0)
    ax.plot(
        ks[ok & ~strong],
        width_hz[ok & ~strong],
        "o",
        ms=3,
        mfc="none",
        color="0.6",
        label="margin < 28 dB (floor-limited readout)",
    )
    ax.plot(
        ks[strong], width_hz[strong], "o", ms=4.5, color="#157F3D", label="measured RMS line width"
    )
    ref = ks[ok]
    w0 = float(np.nanmedian(width_hz[strong][:4]))
    k0 = float(np.median(ks[strong][:4]))
    ax.axvspan(20, ref.max(), color="0.6", alpha=0.15, lw=0)
    ax.plot(
        ref, w0 * (ref / k0), "--", lw=1.2, color="#9A6700", label=r"$\propto k$ (quasi-static)"
    )
    ax.plot(
        ref, w0 * (ref / k0) ** 2, ":", lw=1.4, color="#B42318", label=r"$\propto k^2$ (diffusive)"
    )
    ax.axhline(1.0 / (N / SR), color="0.5", lw=1.0)
    ax.annotate("1/T resolution limit", xy=(ks[ok][0], 1.05 / (N / SR)), fontsize=8, color="0.35")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("harmonic order $k$")
    ax.set_ylabel("line width (Hz, RMS about the line)")
    ax.set_title(
        "The width law, measured on stationary data\n"
        f"$k$=2..16: slope {law_slope}, $\\gamma$={law_g0:+.2f}{law_c:+.3f}$k$ Hz"
    )
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Bayesian comb analysis of one stationary DREGON bench recording "
        f"({RECORDING}, {N / SR:.1f} s, mic 1)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "bayes_comb.png", dpi=DPI)
    plt.close(fig)
    print("wrote", HERE / "bayes_comb.png")


if __name__ == "__main__":
    main()
