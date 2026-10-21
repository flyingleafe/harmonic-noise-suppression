"""Step A0 instrument (experiments.static_rig.phase_width): a planted phase
pedestal must come back through the band window, and the pedestal spectrum
used for the floor correction must carry the pedestal's power."""

from __future__ import annotations

import math

import numpy as np
from scipy import integrate, signal

from experiments.static_rig import phase_width as PW

FS = 44100.0
RATE = 68.0
K = 20


def _ou(n: int, var: float, g: float, rng: np.random.Generator) -> np.ndarray:
    a = math.exp(-2.0 * math.pi * g / FS)
    e = rng.standard_normal(n) * math.sqrt(var * (1.0 - a * a))
    return signal.lfilter([1.0], [1.0, -a], e) + rng.standard_normal() * math.sqrt(
        var
    ) * a ** np.arange(1, n + 1)


def _recovered_width(s2p: float, gp: float, seed: int, seconds: float = 20.0) -> float:
    """Fit one planted line (no shaft) at in-band SNR 10 dB, the floor known."""
    rng = np.random.default_rng(seed)
    band = PW.BAND_FRAC * RATE
    win = PW.Window.make(band)
    n_p = PW._fast_len_below(int(seconds * FS) // PW.DECIM)
    n = n_p * PW.DECIM
    t = np.arange(n) / FS
    s2a, ga = 0.1, 1.0
    psi, a = _ou(n, s2p, gp, rng), _ou(n, s2a, ga, rng)
    n0 = 1e-5  # two-sided density of the white floor
    p_line = 4.0 * 10.0 * n0 * win.kappa0  # mean square of the real line
    x = math.sqrt(2.0 * p_line) * np.exp(a - s2a) * np.cos(2.0 * np.pi * K * RATE * t + psi)
    x = (x + rng.standard_normal(n) * math.sqrt(n0 * FS))[None, :]
    fs_p = FS / PW.DECIM
    edge = int(PW.EDGE_S * fs_p)
    lag_n = np.unique(np.rint(np.asarray(PW.LAGS_MS) * 1e-3 * fs_p).astype(int))
    lag_n = lag_n[lag_n < (n_p - 2 * edge) // 3]
    taus = lag_n / fs_p
    y = PW.baseband(np.fft.rfft(x, axis=-1), n, FS, K * RATE, band, True)[:, edge:-edge]
    st = PW.lag_stats(y, lag_n)
    od = PW.OrderData(
        k=K,
        taus=taus,
        rho=np.abs(st["c"]) / st["c0"][:, None],
        sig_c=st["sig"],
        c0=st["c0"],
        n0=np.array([n0]),
        kappa=win.kappa(taus),
        leak=np.zeros(taus.size),
        s2a=s2a,
        ga=ga,
        nu0=np.zeros(1),
        nu=np.zeros((1, taus.size), dtype=np.complex128),
    )
    fit = PW.fit_phase(win, od)
    assert fit["ok"]
    return float(fit["W_hz"])


def test_planted_pedestal_width_is_recovered_through_the_window() -> None:
    # g = 4 Hz decorrelates within ~40 ms, inside the window's ringing: the
    # regime the old floor-subtracting instrument got wrong.
    for s2p, gp in ((1.2, 4.0), (0.4, 1.0)):
        truth = float(PW.pedestal_width(gp, s2p))
        got = [_recovered_width(s2p, gp, seed) for seed in (1, 2, 3)]
        ratio = math.exp(float(np.mean(np.log(got)))) / truth
        assert 0.75 < ratio < 1.33, (s2p, gp, truth, got)


def test_pedestal_spectrum_holds_the_pedestal_power() -> None:
    # the floor correction removes P * pedestal_psd at the gaps; its integral
    # must be the power off the core, 1 - exp(-s2A - s2P)
    for prm in ((0.1, 1.0, 1.2, 4.0), (0.25, 0.4, 3.0, 2.0)):
        val, _ = integrate.quad(
            lambda f, p=prm: float(PW.pedestal_psd(np.array([f]), *p)[0]),
            -np.inf,
            np.inf,
            limit=400,
        )
        assert abs(val - (1.0 - math.exp(-prm[0] - prm[2]))) < 2e-3, (prm, val)
