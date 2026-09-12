"""Separate the two phase-noise components of one stationary bench recording.

The model of §2.1/§13 gives order k the phase error

    delta_k(t) = k theta(t) + psi_k(t),

a SHAFT term shared by every order (times k) and a PER-HARMONIC term of its
own. From one line the two are indistinguishable -- both broaden it -- so the
separation is done across orders: the increment covariance of the demodulated
phases is

    C_kl = k l Var[dtheta] + delta_kl Var[dpsi_k],

a rank-one matrix plus a diagonal. This script measures C, fits that
structure, and reports how much of the phase noise is the shaft and how much
is per-harmonic, plus the coherence-loss factor each implies.

Run from the repo root::

    PYTHONPATH=src python docs/explainers/rig-model-listening/phase_noise.py
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
N = 1 << 20  # 23.8 s
ENV_RATE = 400.0  # decimated envelope rate (Hz)
#: Increment lags. Additive noise inside the demodulation band contributes a
#: phase jitter that is WHITE, so its increment variance is the same at every
#: lag; a real phase random walk contributes a variance LINEAR in the lag. The
#: two are therefore separated by fitting Var[dphi(tau)] = a + D tau, and only
#: D is phase noise. This is the same white-term separation §4.2c applies to
#: the amplitude envelopes.
LAGS_S = (0.01, 0.02, 0.04, 0.07, 0.12, 0.2, 0.3, 0.45)
DPI = 150


def load_mic0() -> np.ndarray:
    from data_processing.frames import meta_dict
    from data_processing.streams import iter_published_frames

    for f in iter_published_frames("DREGON-frames"):
        if str(meta_dict(f).get("recording_id")) == RECORDING:
            a = np.asarray(f["audio"].data, dtype=np.float64)
            return a[0] if a.ndim > 1 else a
    raise KeyError(RECORDING)


def demodulate(x: np.ndarray, f0: float, bw: float) -> np.ndarray:
    """Complex envelope of the line at f0, decimated to ENV_RATE.

    A rectangular band of half width ``bw`` is kept, so this is the ideal
    band-limited demodulator: no smoothness prior anywhere, and the only
    approximation is the band, which is set to a third of the comb spacing so
    that neighbouring orders cannot leak in.
    """
    n = x.size
    spec = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(n, 1 / SR)
    keep = (freqs > f0 - bw) & (freqs < f0 + bw)
    band = np.zeros_like(spec)
    band[keep] = spec[keep]
    analytic = np.fft.irfft(band, n) + 1j * np.fft.irfft(-1j * band, n)
    t = np.arange(n) / SR
    env = analytic * np.exp(-2j * np.pi * f0 * t)
    step = int(round(SR / ENV_RATE))
    return env[::step]


def main() -> None:
    x = load_mic0()
    start = (x.size - N) // 2
    seg = x[start : start + N]
    seg = seg - seg.mean()

    # the rate, from the Bayesian comb posterior of §11
    import importlib.util

    spec = importlib.util.spec_from_file_location("bc", HERE / "bayes_comb.py")
    assert spec is not None and spec.loader is not None
    bc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bc)
    coarse = np.arange(30.0, 110.0, 0.01)
    comb = bc.Comb(seg)
    s_hat = float(coarse[int(np.argmax([comb.student_t_log(float(s), 40) for s in coarse]))])
    fine = np.arange(s_hat - 0.03, s_hat + 0.03, 0.0002)
    s_hat = float(fine[int(np.argmax([comb.student_t_log(float(s), 80) for s in fine]))])

    # orders strong enough for a phase to mean anything: use the even orders of
    # this two-blade rotor up to 6 kHz
    orders = [k for k in range(2, 61, 2) if k * s_hat < 6000.0]

    # The band must cover the line and little else: too wide and the floor
    # inside it dominates the measured phase. Use four times the width law
    # measured in §11.4, clamped away from the neighbours.
    def band_of(k: int) -> float:
        return float(min(4.0 * (0.39 + 0.227 * k), 0.35 * s_hat))

    phases = []
    amps = []
    for k in orders:
        env = demodulate(seg, k * s_hat, band_of(k))
        edge = int(0.1 * env.size)
        env = env[edge:-edge]
        phases.append(np.unwrap(np.angle(env)))
        amps.append(np.abs(env))
    phi = np.array(phases)  # (K, T_env)
    amp = np.array(amps)
    # remove each order's own mean drift: a residual rate error looks like a
    # ramp shared by all orders, and it is not phase NOISE
    t_env = np.arange(phi.shape[1]) / ENV_RATE
    for i in range(phi.shape[0]):
        phi[i] -= np.polyval(np.polyfit(t_env, phi[i], 1), t_env)

    ks = np.asarray(orders, dtype=float)
    off = ~np.eye(len(ks), dtype=bool)
    kk = np.outer(ks, ks)

    def decompose(C: np.ndarray) -> tuple[float, np.ndarray]:
        """C = kk^T v_theta + diag(v_psi): the off-diagonal identifies the
        rank-one shaft term (the diagonal cannot reach it), the diagonal then
        gives the per-harmonic remainder. Three alternations suffice."""
        v_s = float((C[off] * kk[off]).sum() / (kk[off] ** 2).sum())
        v_i = np.clip(np.diag(C) - v_s * ks**2, 0.0, None)
        for _ in range(3):
            resid = C - np.diag(v_i)
            v_s = float((resid[off] * kk[off]).sum() / (kk[off] ** 2).sum())
            v_i = np.clip(np.diag(C) - v_s * ks**2, 0.0, None)
        return v_s, v_i

    taus, v_shaft_tau, v_indep_tau, corr_last = [], [], [], None
    for lag_s in LAGS_S:
        lag = max(1, int(round(lag_s * ENV_RATE)))
        d = phi[:, lag:] - phi[:, :-lag]
        C = np.cov(d)
        v_s, v_i = decompose(C)
        taus.append(lag / ENV_RATE)
        v_shaft_tau.append(v_s)
        v_indep_tau.append(v_i)
        corr_last = np.corrcoef(d)
    taus = np.asarray(taus)
    v_shaft_tau = np.asarray(v_shaft_tau)
    v_indep_tau = np.asarray(v_indep_tau)  # (n_lag, K)

    # Var(tau) = a + D tau: the intercept a is white (additive noise inside the
    # band plus demodulation error), the slope D is the phase random walk.
    d_shaft, a_shaft = np.polyfit(taus, v_shaft_tau, 1)
    slopes = np.array([np.polyfit(taus, v_indep_tau[:, i], 1) for i in range(len(ks))])
    d_indep, a_indep = np.clip(slopes[:, 0], 0.0, None), slopes[:, 1]
    # the eigen-diagnostic is read at one representative lag
    C = np.cov(phi[:, 20:] - phi[:, :-20])
    w, v = np.linalg.eigh(C)

    # eigen-diagnostic: if the shaft dominates, the top eigenvector of C is
    # parallel to k
    top = v[:, -1]
    cos_k = float(abs(top @ (ks / np.linalg.norm(ks))))
    share = float(w[-1] / w.sum())

    # --- what each component implies for the line width ----------------------
    # Wiener increments over lag tau: Var = D tau, and a Lorentzian of HWHM
    # gamma = D / (4 pi^2) ... expressed directly: the phase-error variance per
    # second is D = Var[dphi] / tau, and the -3 dB half width of the resulting
    # Lorentzian is D / (4 pi) Hz.
    # A Wiener phase of diffusion D (rad^2/s) gives a Lorentzian of HWHM
    # D / (4 pi) Hz; the shaft term enters order k as k^2 D_theta.
    tau = float(taus[-1])
    gamma_shaft = ks**2 * float(d_shaft) / (4 * np.pi)
    gamma_indep_k = d_indep / (4 * np.pi)
    d_indep_med = float(np.median(d_indep))
    gamma_indep = d_indep_med / (4 * np.pi)
    # If instead the shaft wander is QUASI-STATIC on the analysis window, the
    # same diffusion reads as a rate spread: over tau, Var[dtheta] = D tau and
    # dtheta = 2 pi delta_s tau, so sigma_s = sqrt(D / tau) / (2 pi).
    sigma_s = float(np.sqrt(max(d_shaft, 0.0) * tau)) / (2 * np.pi * tau)

    out = dict(
        recording=RECORDING,
        seconds=round(N / SR, 2),
        rate_rps=round(s_hat, 5),
        orders=orders,
        lag_s=round(tau, 4),
        shaft_diffusion_rad2_per_s=float(f"{float(d_shaft):.4e}"),
        shaft_white_intercept_rad2=float(f"{float(a_shaft):.4e}"),
        per_harmonic_diffusion_rad2_per_s_median=float(f"{d_indep_med:.4e}"),
        per_harmonic_white_intercept_rad2_median=float(f"{float(np.median(a_indep)):.4e}"),
        shaft_share_of_diffusion_at={
            f"k={int(k)}": round(
                float(float(d_shaft) * k**2 / (float(d_shaft) * k**2 + d_indep[i] + 1e-30)),
                3,
            )
            for i, k in enumerate(ks)
            if int(k) in (2, 8, 16, 32, 48, 60)
        },
        crossover_order=(
            round(float(np.sqrt(d_indep_med / max(float(d_shaft), 1e-30))), 1)
            if d_shaft > 0
            else None
        ),
        total_gamma_hz={
            f"k={int(k)}": round(float(gamma_shaft[i] + gamma_indep_k[i]), 3)
            for i, k in enumerate(ks)
            if int(k) in (2, 8, 16, 32, 48, 60)
        },
        top_eigenvalue_share=round(share, 3),
        cos_top_eigenvector_with_k=round(cos_k, 4),
        implied_sigma_shaft_rps=round(sigma_s, 4),
        shaft_gamma_hz={
            f"k={int(k)}": round(float(gamma_shaft[i]), 3)
            for i, k in enumerate(ks)
            if int(k) in (2, 16, 32, 60)
        },
        implied_per_harmonic_gamma_hz_median=round(float(gamma_indep), 4),
        per_harmonic_gamma_hz={
            f"k={int(k)}": round(float(gamma_indep_k[i]), 3)
            for i, k in enumerate(ks)
            if int(k) in (2, 8, 16, 32, 48, 60)
        },
        # If the per-harmonic term were truly independent of the shaft its
        # diffusion would not care about k. Fit log D_psi = a + b log k.
        per_harmonic_k_exponent=round(
            float(np.polyfit(np.log(ks[d_indep > 0]), np.log(d_indep[d_indep > 0]), 1)[0]),
            3,
        ),
        amp_rel_std_db=round(
            float(np.median(20 * np.log10(amp.std(axis=1) / amp.mean(axis=1)))), 2
        ),
    )
    print(json.dumps(out, indent=1))
    (HERE / "phase_noise.json").write_text(json.dumps(out, indent=1))

    # --- figure ---------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

    ax = axes[0]
    for k in orders[::4]:
        ax.plot(t_env, phi[orders.index(k)] / k, lw=0.9, label=f"$k$={k}")
    ax.set_xlabel("time (s)")
    ax.set_ylabel(r"$\delta_k(t)\,/\,k$  (rad)")
    ax.set_title("Order phases, divided by their order\n(shaft noise would superimpose them)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)

    ax = axes[1]
    im = ax.imshow(
        corr_last if corr_last is not None else np.eye(len(ks)),
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        extent=[ks[0], ks[-1], ks[-1], ks[0]],
    )
    fig.colorbar(im, ax=ax, fraction=0.046, label="correlation")
    ax.set_title(
        "Increment correlation across orders\n"
        f"top eigenvalue {share:.0%} of trace, $|\\cos(v_1,k)|$={cos_k:.2f}"
    )
    ax.set_xlabel("order $k$")
    ax.set_ylabel("order $k$")
    ax.grid(False)

    ax = axes[2]
    ax.plot(
        ks,
        gamma_shaft + gamma_indep_k,
        "o-",
        ms=4,
        color="0.25",
        label=r"total $\gamma_k$ from the fitted diffusions",
    )
    ax.plot(ks, gamma_shaft, "--", lw=1.6, color="#157F3D", label=r"shaft: $k^2D_\theta/4\pi$")
    ax.plot(
        ks, gamma_indep_k, ":", lw=1.6, color="#B42318", label=r"per-harmonic: $D_{\psi,k}/4\pi$"
    )
    ax.plot(ks, 0.39 + 0.227 * ks, "-", lw=1.1, color="#9A6700", label="width measured in §11.4")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("order $k$")
    ax.set_ylabel("implied line half width (Hz)")
    ax.set_title("Each component's own $k$-law")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle(
        f"Phase noise of one stationary rotor: shaft versus per-harmonic "
        f"({RECORDING}, {N / SR:.0f} s, mic 1)",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(HERE / "phase_noise.png", dpi=DPI)
    plt.close(fig)
    print("wrote", HERE / "phase_noise.png")


if __name__ == "__main__":
    main()
