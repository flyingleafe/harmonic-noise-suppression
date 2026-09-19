"""Core checks of the noise model v2 forward model, lag law and renderer.

Every test here is a DIFFERENTIAL or a PHYSICAL one: each pins the new code to
something already trusted (C4's lag law, C4's shared finite-window kernel,
C4's ``frame_model``) or to a quantity that can be derived independently of the
implementation (the Gaussian linewidth of a frozen-rate shaft, the absolute
level of a render read back through the frozen evaluator periodogram, the
demodulated per-order increment variance). CPU-small by construction.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
import torch

from experiments.noise_model import fit as FT
from experiments.noise_model import model as MD
from experiments.noise_model import render as RD
from experiments.noise_model import spectrum as SP
from experiments.noise_model.lag import r_tau
from experiments.stochastic_fit import revised_eval as RE
from experiments.stochastic_fit import revised_phase as RP
from experiments.stochastic_fit.data import Clip
from experiments.stochastic_fit.phase_kernel import expected_periodogram_from_atoms, hann_window

SR = 16000
NC = SP.FLOOR_SHAPE_N_CTRL


def _t(v) -> torch.Tensor:
    return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)


def _params(
    *,
    n_rotors: int,
    n_mics: int,
    k_cap: int,
    profile_db,
    carrier=None,
    sigma_nu: float = 0.8,
    lam: float = 5.5,
    gamma_hz: Any = 0.0,
    floor_mean_db: float = -40.0,
    shape_z=None,
    tilt: float = -2.0,
    mic_floor_db=None,
    mic_line_db=None,
    gain_all_db=None,
    amp_exp: float = 0.0,
    floor_exp: float = 0.0,
    static_rel: float = 0.0,
) -> SP.V2Params:
    prof = np.broadcast_to(np.asarray(profile_db, dtype=np.float64), (n_rotors, k_cap)).copy()
    return SP.V2Params(
        sigma_nu=_t(sigma_nu),
        lam=_t(lam),
        gamma_hz=_t(np.broadcast_to(np.asarray(gamma_hz, dtype=np.float64), (n_rotors, k_cap))),
        profile_db=_t(prof),
        floor=SP.FloorParams(
            mean_db=_t(floor_mean_db),
            shape_z=_t(np.zeros(NC) if shape_z is None else shape_z),
            tilt_db_oct=_t(tilt),
            mic_floor_db=_t(np.zeros(n_mics) if mic_floor_db is None else mic_floor_db),
            exp=_t(floor_exp),
            static_rel=_t(static_rel),
        ),
        mic_line_gain_db=_t(np.zeros((n_mics, n_rotors)) if mic_line_db is None else mic_line_db),
        gain_all_db=_t(np.zeros(n_mics) if gain_all_db is None else gain_all_db),
        carrier_rev_s=None if carrier is None else _t(carrier),
        amp_exp=_t(amp_exp),
    )


# ── the lag law ─────────────────────────────────────────────────────────────


def test_lag_law_without_a_line_width_is_c4s_prior_lag_law():
    """``gamma_hz = 0`` must leave EXACTLY C4's shaft-only prior ``R_k(tau)``.

    The R3 law adds an independent per-line Lorentzian to C4's shaft factor;
    if the shaft factor itself had drifted (a factor of two in ``V_theta``,
    ``k`` instead of ``k^2``), every line width in the campaign would move and
    no other test here would localise it.
    """
    tau = np.linspace(0.0, 2.0, 257)
    k = np.array([1.0, 2.0, 7.0, 20.0, 21.0, 130.0])[:, None]
    for sigma_nu, lam in ((0.45, 5.5), (1.8, 0.67), (0.31, 14.4)):
        got = r_tau(tau[None, :], k, sigma_nu=sigma_nu, lam=lam, gamma_hz=0.0)
        want = RP.prior_r_tau(tau[None, :], k, lam=lam, sigma=sigma_nu, d=0.0)
        assert np.abs(got - want).max() < 1e-9


def test_lag_law_takes_one_width_per_line_and_agrees_across_backends():
    """``gamma_hz`` is per ORDER (and per rotor), and numpy and torch agree.

    The widths are a ``(R, K)`` block with no law across ``k``, so the test
    plants a different width in every line of two rotors and checks each line
    carries its OWN ``exp(-2 pi gamma |tau|)``. The torch path is the one the
    fit differentiates and the numpy path is the one the diagnostics use; a
    disagreement between them would put the renderer and the fit on different
    laws.
    """
    tau = np.linspace(0.0, 0.5, 65)
    k = np.arange(1.0, 5.0)
    gamma = np.array([[0.3, 1.0, 4.0, 9.0], [0.05, 0.2, 0.5, 2.0]])  # (R, K)
    got = r_tau(
        tau[None, None, :],
        k[None, :, None],
        sigma_nu=1e-12,
        lam=5.5,
        gamma_hz=gamma[:, :, None],
    )
    want = np.exp(-2.0 * math.pi * gamma[:, :, None] * tau[None, None, :])
    assert got.shape == (2, 4, 65)
    assert np.abs(got - want).max() < 1e-9

    tor = r_tau(
        torch.as_tensor(tau)[None, None, :],
        torch.as_tensor(k)[None, :, None],
        sigma_nu=torch.tensor(1e-12, dtype=torch.float64),
        lam=torch.tensor(5.5, dtype=torch.float64),
        gamma_hz=torch.as_tensor(gamma)[:, :, None],
    )
    assert float((tor.detach().numpy() - got).__abs__().max()) < 1e-12


def test_lag_law_rejects_a_width_without_a_lag_axis():
    """A ``(K,)`` width beside a ``(K, 1)`` order column would broadcast into
    a ``(K, K)`` outer product — every order carrying every other order's
    width, silently. The law refuses it instead."""
    tau = np.linspace(0.0, 0.1, 17)
    k = np.arange(1.0, 5.0)[:, None]
    with pytest.raises(ValueError, match="lag axis"):
        r_tau(tau[None, :], k, sigma_nu=0.3, lam=2.0, gamma_hz=np.array([0.1, 0.2, 0.3, 0.4]))


def test_lag_law_is_differentiable_in_every_parameter():
    """The fit needs a gradient in all three dynamics parameters."""
    tau = torch.linspace(0.0, 0.5, 65, dtype=torch.float64)
    k = torch.arange(1.0, 9.0, dtype=torch.float64)[:, None]
    args = {
        "sigma_nu": torch.tensor(0.45, dtype=torch.float64, requires_grad=True),
        "lam": torch.tensor(5.5, dtype=torch.float64, requires_grad=True),
        "gamma_hz": torch.full((8, 1), 0.05, dtype=torch.float64, requires_grad=True),
    }
    r_tau(
        tau[None, :],
        k,
        sigma_nu=args["sigma_nu"],
        lam=args["lam"],
        gamma_hz=args["gamma_hz"],
    ).sum().backward()
    for name, v in args.items():
        assert v.grad is not None, name
        assert torch.isfinite(v.grad).all(), name
        assert float(v.grad.abs().sum()) > 0.0, name


def test_a_planted_lorentzian_width_is_recovered_from_the_line_it_makes():
    """Plant ``gamma`` in ONE order and read its HWHM back off the line.

    The whole point of R3's parameter is that it IS the line's half-width at
    half maximum in Hz: a wrong ``2 pi``, a factor of two in the increment
    variance or a Gaussian-instead-of-Lorentzian convention all move this
    number. Read off the bench forward model's own periodogram with the shaft
    switched off, on a 4 s window whose 0.25 Hz bins resolve a 2 Hz width
    forty-fold.
    """
    n, k, f0, gamma = 1 << 16, 6, 149.0, 2.0
    prof = np.full((1, k), -300.0)
    prof[0, k - 1] = -10.0
    par = _params(
        n_rotors=1,
        n_mics=1,
        k_cap=k,
        profile_db=prof,
        carrier=[f0],
        sigma_nu=1e-9,
        lam=1.0,
        gamma_hz=gamma,
        floor_mean_db=-300.0,
    )
    grid = SP.bench_grid(n=n, sr=SR, floor_lag=1 << 13, apply_transfer=False)
    m = SP.bench_model(grid, par, k_max=k)[0, 0].numpy()
    df = SR / n
    centre = int(round(k * f0 / df))
    half = int(round(40.0 / df))
    seg, freqs = (
        m[centre - half : centre + half + 1],
        grid.freqs_hz[centre - half : centre + half + 1],
    )
    above = np.nonzero(seg >= 0.5 * seg.max())[0]
    hwhm = 0.5 * float(freqs[above[-1]] - freqs[above[0]] + df)
    assert abs(hwhm - gamma) / gamma < 0.05


def test_bench_spectrum_has_a_gradient_for_every_dynamics_parameter():
    """Every dynamics value — including EVERY line's own width — must move the
    expected spectrum.

    This is deliberately a forward-model check, rather than only differentiating
    ``r_tau``: it catches a NumPy/detach conversion while a Pyro sample is being
    carried through the finite-window transform, and it is the evidence that
    the ``(R, K)`` width block reaches the objective line by line rather than
    as one shared number.
    """
    par = _params(
        n_rotors=1,
        n_mics=1,
        k_cap=8,
        profile_db=np.linspace(-14.0, -27.0, 8)[None, :],
        carrier=[173.0],
        gamma_hz=0.05,
    )
    dynamics = {
        "sigma_nu": par.sigma_nu.requires_grad_(),
        "lam": par.lam.requires_grad_(),
        "gamma_hz": par.gamma_hz.requires_grad_(),
    }
    grid = SP.bench_grid(n=1024, sr=SR, apply_transfer=False, floor_lag=1024)
    m = SP.bench_model(grid, par, k_max=8)
    m[..., torch.as_tensor(grid.band)].mean().backward()
    for name, value in dynamics.items():
        assert value.grad is not None, name
        assert torch.isfinite(value.grad).all(), name
        assert float(value.grad.abs().sum()) > 0.0, name


# ── the forward model ───────────────────────────────────────────────────────


def test_bench_matches_the_shared_finite_window_kernel():
    """BENCH mode's lag-domain transform IS the kernel, to float64 roundoff.

    The bench model does NOT call :func:`expected_periodogram_from_atoms`: with
    a constant carrier the kernel collapses to one shifted transform per rotor,
    which is what makes a 30 s whole-segment periodogram affordable. This test
    is the evidence that the collapse is algebra and not an approximation — it
    computes the same support the slow way, order by order through the shared
    kernel, including the coloured floor through its complex-atom convention.
    """
    n, n_mics, k_cap, f0 = 2048, 3, 7, 137.3
    prof = np.linspace(-10.0, -35.0, k_cap)[None, :]
    gamma = np.linspace(0.5, 6.0, k_cap)[None, :]
    par = _params(
        n_rotors=1,
        n_mics=n_mics,
        k_cap=k_cap,
        profile_db=prof,
        carrier=[f0],
        gamma_hz=gamma,
        shape_z=np.random.default_rng(3).standard_normal(NC),
        mic_floor_db=[0.0, 1.5, -0.7],
        mic_line_db=[[0.0], [2.0], [-1.0]],
        gain_all_db=[0.3, -0.2, 0.9],
    )
    grid = SP.bench_grid(n=n, sr=SR, floor_lag=n)
    got = SP.bench_model(grid, par, k_max=k_cap)

    w = _t(hann_window(n))
    sumsq = float((w**2).sum())
    tau = torch.arange(n, dtype=torch.float64) / SR
    k = torch.arange(1.0, k_cap + 1.0, dtype=torch.float64)
    kph = torch.remainder(k[:, None] * (2.0 * np.pi * f0 * tau)[None, :], 2.0 * np.pi)
    amp = torch.sqrt(2.0 * 10.0 ** (_t(prof[0]) / 10.0))
    rho = r_tau(
        tau[None, :],
        k[:, None],
        sigma_nu=0.8,
        lam=5.5,
        gamma_hz=_t(gamma[0])[:, None],
    )
    lines = expected_periodogram_from_atoms(
        torch.polar(amp[:, None] * w[None, :], kph), rho, n_fft=n, window_sumsq=sumsq
    ).sum(0)
    c = torch.fft.irfft(grid.floor.psd(par.floor).to(torch.complex128), n=grid.floor.psd_len).real[
        :n
    ]
    floor = expected_periodogram_from_atoms(
        w * torch.sqrt(c[0]), 2.0 * c / c[0], n_fft=n, window_sumsq=sumsq
    )
    lg = 10.0 ** ((par.mic_line_gain_db - par.mic_line_gain_db.mean(0, keepdim=True)) / 10.0)
    ag = 10.0 ** ((par.gain_all_db - par.gain_all_db.mean()) / 10.0)
    fg = 10.0 ** (par.floor.mic_floor_db / 10.0)
    want = (
        (torch.einsum("mr,f->mf", lg, lines) + fg[:, None] * floor[None, :])
        * grid.transfer_power[None, :]
        * ag[:, None]
    )
    band = torch.as_tensor(grid.band)
    rel = ((got[:, 0, band] - want[:, band]).abs() / want[:, band]).max()
    assert float(rel) < 1e-8


def test_bench_order_truncation_does_not_change_the_prediction():
    """Dropping each order's dead lag tail must be invisible in the band.

    The truncation is what makes the bench affordable; if the threshold were
    too aggressive it would narrow the high orders' pedestals, which is exactly
    the quantity the campaign is trying to measure.
    """
    n, k_cap = 4096, 40
    par = _params(
        n_rotors=1, n_mics=1, k_cap=k_cap, profile_db=-20.0, carrier=[137.0], sigma_nu=1.5
    )
    grid = SP.bench_grid(n=n, sr=SR, floor_lag=n)
    groups = SP.order_groups(k_cap, sigma_nu=1.5, lam=5.5, sr=SR, n=n)
    truncated = SP.bench_model(grid, par, k_max=k_cap, groups=groups)
    full = SP.bench_model(grid, par, k_max=k_cap)
    band = torch.as_tensor(grid.band)
    rel = ((truncated[:, 0, band] - full[:, 0, band]).abs() / full[:, 0, band]).max()
    assert float(rel) < 1e-6


def test_flight_matches_c4_frame_model_where_the_laws_coincide():
    """FLIGHT mode must equal C4's ``frame_model`` to 1e-6 at ``gamma_hz = 0``.

    v2's flight forward model is C4's with ONE substitution — the prior lag law
    at the kernel call. With the line widths at zero and C4's ``D``
    driven to zero the two laws are identical, so any other difference (the
    within-window carrier integral, the phase centring, the speed law, the
    floor's envelope, the transfer, the grid power factor) shows up here.
    """
    n_fft, hop, k_cap, n_frames = 2048, 512, 6, 3
    n = n_fft + (n_frames - 1) * hop + 4096
    rng = np.random.default_rng(11)
    rps = (100.0 + 6.0 * np.sin(np.arange(n) / n * 5.0))[None, :]
    clip = Clip(
        "diff",
        "synthetic",
        rng.standard_normal((2, n)).astype(np.float32) * 1e-3,
        rps,
        SR,
        rps.copy(),
        {},
    )
    cfg = RP.FitConfig(
        n_fft=n_fft,
        hop=hop,
        sr=SR,
        band_hz=(30.0, 7900.0),
        k_cap=k_cap,
        delay_s=(0.0,),
        atom_dtype="float64",
        harmonic_chunk=None,
        fit_method="fixed_carrier_marginal",
        lbfgs_max_iter=1,
        lbfgs_max_eval=1,
    )
    c4 = RP._RevisedModel(  # type: ignore[attr-defined]
        [(clip.clip_id, clip)],
        dynamics=RP.ShaftDynamics(
            lam=5.5, sigma=0.8, d_init=1e-14, identified=True, diagnostics={}
        ),
        config=cfg,
        observe=False,
    )
    with torch.no_grad():
        c4.profile_db.copy_(_t(np.linspace(-12.0, -30.0, k_cap)[None, :]))
        c4.amp_exp.copy_(_t(2.3))
        c4.floor_mean_db.copy_(_t([-46.0]))
        c4.floor_shape_z.copy_(_t(rng.standard_normal(NC)))
        c4.floor_tilt_db_oct.copy_(_t([-1.7]))
        c4.floor_exp.copy_(_t(1.9))
        c4.floor_static_raw.copy_(_t(-4.5))
        c4.mic_floor_db.copy_(_t([0.6, -1.1]))
        c4.mic_gain_db.copy_(_t([[1.3], [-0.4]]))
        c4.gain_all_db.copy_(_t([0.8, -0.2]))
        c4.log_sigma.copy_(_t(math.log(0.8)))
        c4.log_d.copy_(_t(math.log(1e-14)))
        want = c4.frame_model(0, np.arange(n_frames), state=None, kernel="prior")

    grid = SP.flight_grid(sr=SR, n_fft=n_fft, hop=hop)
    par = _params(
        n_rotors=1,
        n_mics=2,
        k_cap=k_cap,
        profile_db=np.linspace(-12.0, -30.0, k_cap)[None, :],
        sigma_nu=0.8,
        lam=5.5,
        gamma_hz=0.0,
        floor_mean_db=-46.0,
        shape_z=np.asarray(c4.floor_shape_z.detach()),
        tilt=-1.7,
        mic_floor_db=[0.6, -1.1],
        mic_line_db=[[1.3], [-0.4]],
        gain_all_db=[0.8, -0.2],
        amp_exp=2.3,
        floor_exp=1.9,
        static_rel=float(torch.nn.functional.softplus(_t(-4.5))),
    )
    with torch.no_grad():
        rate = SP.flight_rate_work(grid, rps, np.arange(n_frames) * hop)
        got = SP.flight_model(grid, par, rate_work=rate, k_max=k_cap, harmonic_chunk=None)
    band = torch.as_tensor(grid.band)
    rel = ((got[:, :, band] - want[:, :, band]).abs() / want[:, :, band]).max()
    assert float(rel) < 1e-6


def test_bench_linewidth_matches_the_frozen_rate_gaussian():
    """A frozen-rate shaft must give the analytic Gaussian linewidth.

    At ``lam |tau| << 1`` the structure function is ``V_theta = sigma_nu^2
    tau^2`` exactly, so ``R_k`` is a Gaussian of scale ``1 / (k sigma_nu)`` and
    the line is a Gaussian of FWHM ``2 sqrt(ln 2 / 2) k sigma_nu / pi =
    0.37470 k sigma_nu`` Hz — a number derived from the law, not from this
    code. The window's own 1.4 Hz main lobe adds in quadrature and is 0.1 % of
    the 30 Hz width measured here.
    """
    n, k, sigma_nu, lam, f0 = 1 << 14, 40, 2.0, 0.5, 137.0
    prof = np.full((1, k), -300.0)
    prof[0, k - 1] = -10.0
    par = _params(
        n_rotors=1,
        n_mics=1,
        k_cap=k,
        profile_db=prof,
        carrier=[f0],
        sigma_nu=sigma_nu,
        lam=lam,
        gamma_hz=0.0,
        floor_mean_db=-200.0,
    )
    grid = SP.bench_grid(n=n, sr=SR, floor_lag=n, apply_transfer=False)
    m = SP.bench_model(grid, par, k_max=k)[0, 0].numpy()
    df = SR / n
    centre = int(round(k * f0 / df))
    half = int(round(120.0 / df))
    sl = slice(centre - half, centre + half + 1)
    seg, freqs = m[sl], grid.freqs_hz[sl]
    peak = float(seg.max())
    above = np.nonzero(seg >= 0.5 * peak)[0]
    fwhm = float(freqs[above[-1]] - freqs[above[0]] + df)
    expect = 2.0 * math.sqrt(math.log(2.0) / 2.0) * k * sigma_nu / math.pi
    assert abs(fwhm - expect) / expect < 0.05


# ── the renderer ────────────────────────────────────────────────────────────


def _planted_fit(*, k_cap: int, f0: float, n_mics: int) -> dict:
    prof = np.linspace(-18.0, -34.0, k_cap)[None, :]
    par = _params(
        n_rotors=1,
        n_mics=n_mics,
        k_cap=k_cap,
        profile_db=prof,
        carrier=[f0],
        sigma_nu=0.45,
        lam=5.5,
        gamma_hz=np.linspace(0.4, 3.0, k_cap)[None, :],
        floor_mean_db=-52.0,
        tilt=-3.0,
        amp_exp=0.0,
        floor_exp=0.0,
        static_rel=1.0,
    )
    return dict(schema="noise-v2-fit/2", params=MD.params_to_dict(par))


def test_render_level_matches_the_model_in_absolute_units():
    """The rendered periodogram must BE the fitted ``M``, with no normalisation.

    This is the open item the last campaign left (an RMS-normalising render
    wrapper, ``revised-phase-campaign.qmd:37-42``): the render is read back
    through the frozen evaluator periodogram and its band-mean compared to the
    model's, in absolute units, on the same carriers. A 1 dB tolerance is what
    one 4 s draw of an exponential-cell periodogram supports.
    """
    k_cap, f0, n_mics, dur = 24, 149.0, 2, 4.0
    fit = _planted_fit(k_cap=k_cap, f0=f0, n_mics=n_mics)
    n = int(dur * SR)
    rps = np.full((1, n), f0)
    audio = RD.render_noise(fit, rps, sr=SR, n_mics=n_mics, seed=5)
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (n_mics, n)

    n_fft = 1 << 14
    pg = RE.window_periodogram(
        Clip("r", "synthetic", audio.astype(np.float32), rps, SR), n_fft=n_fft, hop=n_fft
    )
    grid = SP.bench_grid(n=n_fft, sr=SR)
    par = MD.params_from_dict(fit["params"])
    m = SP.bench_model(grid, par, k_max=k_cap).numpy()
    band = SP.bench_band(grid.freqs_hz, SR)
    got_db = 10.0 * np.log10(pg.power[:, :, band].mean())
    want_db = 10.0 * np.log10(m[:, :, band].mean())
    assert abs(got_db - want_db) < 1.0


def test_render_reproduces_the_per_order_lag_law():
    """One order demodulated out of the render must carry ``4 pi gamma tau``.

    The renderer draws a Wiener phase per line; the fit assumes its increment
    variance is ``4 pi gamma_rk |tau|``, which is what makes the line the
    Lorentzian of half-width ``gamma_rk`` the law describes. A stationary OU
    (the R1 draw, which SATURATES), a missing factor of two or a
    ``2 pi``-instead-of-``4 pi`` all show up as a wrong ``V_psi(k, tau)``:
    measured here with :mod:`utils.demod` on a render whose shaft term is
    switched off, so only the per-line process is left, and at two lags, so a
    saturating process cannot pass by matching one of them.
    """
    from utils.demod import demodulate

    k, f0, dur, gamma = 20, 149.0, 8.0, 1.5
    fit = _planted_fit(k_cap=k, f0=f0, n_mics=1)
    p = fit["params"]
    p["sigma_nu"] = 1e-9
    p["gamma_hz"] = np.full((1, k), gamma).tolist()
    p["floor"]["floor_mean_db"] = -300.0
    prof = np.full((1, k), -300.0)
    prof[0, k - 1] = -6.0
    p["profile"]["profile_db"] = prof.tolist()

    n = int(dur * SR)
    rps = np.full((1, n), f0)
    audio = RD.render_noise(fit, rps, sr=SR, n_mics=1, seed=7)
    # demodulate() takes the FREQUENCY track and the order, not a phase
    z = demodulate(audio[0], np.full(n, f0), 30.0, SR, order=float(k))
    phi = np.unwrap(np.angle(z))
    edge = int(0.25 * SR)
    phi = phi[edge:-edge]
    # remove the residual linear trend the demodulator's own band-pass leaves
    t = np.arange(phi.size) / SR
    phi = phi - np.polyval(np.polyfit(t, phi, 1), t)
    for tau_s in (0.05, 0.2):
        lag = int(round(tau_s * SR))
        got = float(np.var(phi[lag:] - phi[:-lag]))
        want = 4.0 * math.pi * gamma * tau_s
        assert abs(got - want) / want < 0.2, (tau_s, got, want)


def _regime_pair(*, k_cap: int, n_mics: int, gap_db: float) -> dict[str, dict]:
    """A standby/cruise pair of planted fits ``gap_db`` apart in LEVEL.

    Both speed exponents are zero in :func:`_planted_fit`, so within one fit
    the level does not move with the carrier: the only thing that can change
    the composed level is the composition itself.
    """
    standby = _planted_fit(k_cap=k_cap, f0=35.0, n_mics=n_mics)
    cruise = _planted_fit(k_cap=k_cap, f0=80.0, n_mics=n_mics)
    prof = np.asarray(cruise["params"]["profile"]["profile_db"], dtype=np.float64)
    cruise["params"]["profile"]["profile_db"] = (prof + gap_db).tolist()
    cruise["params"]["floor"]["floor_mean_db"] = (
        float(standby["params"]["floor"]["floor_mean_db"]) + gap_db
    )
    return dict(standby=standby, cruise=cruise)


def _block_level_db(audio: np.ndarray, block: int) -> np.ndarray:
    n = (audio.shape[-1] // block) * block
    p = (audio[:, :n] ** 2).reshape(audio.shape[0], -1, block).mean(axis=(0, 2))
    return 10.0 * np.log10(np.maximum(p, 1e-300))


def test_regime_composition_crosses_the_bands_without_a_level_step():
    """A track that crosses 45 -> 65 rev/s must render CONTINUOUSLY, and must
    reproduce each regime's own render where that regime is the only one gated
    in.

    The two planted fits sit 12 dB apart, so the composition's failure modes
    are visible: a hard switch at either threshold would put a ~12 dB step in
    the block-level track (the 3 dB criterion below), and summing the two
    renders from ONE stream (correlated shaft and line phases) would put a
    bump of up to 3 dB in the middle of the blend, where the measured power
    must instead be the weight-interpolated power of the two regimes.
    """
    from data_processing.rps_gating import CRUISE_MIN_RPS, STANDBY_MAX_RPS

    k_cap, n_mics, seed = 12, 2, 4
    fits = _regime_pair(k_cap=k_cap, n_mics=n_mics, gap_db=12.0)
    # 0.5 s of standby, a 2.5 s spool-up 35 -> 80 rev/s, 1 s of cruise
    hold, ramp = int(0.5 * SR), int(2.5 * SR)
    track = np.concatenate([np.full(hold, 35.0), np.linspace(35.0, 80.0, ramp), np.full(SR, 80.0)])
    rps = track[None, :]
    w = RD.regime_blend_weight(rps, sr=SR)
    assert float(w.min()) == 0.0 and float(w.max()) == 1.0
    assert np.all(w[track <= STANDBY_MAX_RPS] == 0.0)
    assert np.all(w[track >= CRUISE_MIN_RPS] == 1.0)

    out, diag = RD.render_noise_regimes(
        fits, rps, sr=SR, n_mics=n_mics, seed=seed, return_diagnostics=True
    )
    assert out.shape == (n_mics, track.size)
    assert diag["per_regime"]["standby"]["rendered"] and diag["per_regime"]["cruise"]["rendered"]

    seeds = RD.regime_seeds(seed)
    alone = {
        regime: np.asarray(
            RD.render_noise(fits[regime], rps, sr=SR, n_mics=n_mics, seed=seeds[regime])
        )
        for regime in ("standby", "cruise")
    }
    # outside the blend the composition IS that regime's own render
    only_standby = track < STANDBY_MAX_RPS
    only_cruise = track > CRUISE_MIN_RPS
    assert only_standby.sum() > SR // 4 and only_cruise.sum() > SR // 2
    assert np.allclose(out[:, only_standby], alone["standby"][:, only_standby], atol=0.0)
    assert np.allclose(out[:, only_cruise], alone["cruise"][:, only_cruise], atol=0.0)

    # continuity: no step between adjacent 64 ms blocks anywhere on the track
    block = 1024
    levels = _block_level_db(out, block)
    steps = np.abs(np.diff(levels))
    assert float(steps.max()) <= 3.0, float(steps.max())
    # and the 12 dB the two regimes differ by IS crossed, so the test is not
    # passing because nothing happens
    assert float(levels.max() - levels.min()) > 8.0

    # the blend is a POWER interpolation of the two regimes' own renders
    p_std = 10.0 ** (_block_level_db(alone["standby"], block) / 10.0)
    p_cru = 10.0 ** (_block_level_db(alone["cruise"], block) / 10.0)
    w_blk = w[: (w.size // block) * block].reshape(-1, block).mean(axis=1)
    want_db = 10.0 * np.log10((1.0 - w_blk) * p_std + w_blk * p_cru)
    assert float(np.abs(levels - want_db).max()) < 1.5, float(np.abs(levels - want_db).max())


def test_regime_expected_periodogram_is_the_same_power_blend():
    """The composition's expected ``M`` must be the per-frame weighted sum, and
    on a single-regime track it must BE that regime's own prediction — the
    likelihood gate reads this, so a silent fall-back to one fit would move
    every margin."""
    k_cap, n_mics = 6, 2
    fits = _regime_pair(k_cap=k_cap, n_mics=n_mics, gap_db=12.0)
    n = 4096 + 512
    cruise_only = np.full((1, n), 80.0)
    got = RD.expected_periodogram_regimes(
        fits, cruise_only, n_fft=2048, hop=512, sr=SR, n_mics=n_mics
    )
    want = RD.expected_periodogram(
        fits["cruise"], cruise_only, n_fft=2048, hop=512, sr=SR, n_mics=n_mics
    )
    assert np.allclose(got, want, rtol=0.0, atol=0.0)

    mid = np.full((1, n), 0.5 * (STANDBY_MAX := 45.0) + 0.5 * (CRUISE_MIN := 65.0))
    assert STANDBY_MAX < float(mid[0, 0]) < CRUISE_MIN
    blend = RD.expected_periodogram_regimes(fits, mid, n_fft=2048, hop=512, sr=SR, n_mics=n_mics)
    half = 0.5 * (
        RD.expected_periodogram(fits["standby"], mid, n_fft=2048, hop=512, sr=SR, n_mics=n_mics)
        + RD.expected_periodogram(fits["cruise"], mid, n_fft=2048, hop=512, sr=SR, n_mics=n_mics)
    )
    # the mid-band carrier gets weight exactly 1/2 from the smoothstep
    assert np.allclose(blend, half, rtol=1e-12, atol=0.0)


# ── the objective and the fit ───────────────────────────────────────────────


def test_bench_map_recovers_planted_dynamics_at_the_frozen_carrier():
    """A planted bench support must be recovered by the MAP fit end to end.

    Small by design (one rotor, eight orders, 1 s, four microphones): what it
    proves is that the Pyro model, the ``AutoDelta`` guide, the measurement
    pass, the Adam phase, the L-BFGS polish and the JSON all compose — a wrong
    sign on the Whittle factor or a mis-scoped frozen block cannot pass it.
    Four microphones, because the tolerance below is a statement about the
    estimator and an R3 line lands in ~one bin: with two exponential cells per
    line the 3 dB band is a coin toss about the draw, with eight it is not.
    """
    k_cap, f0, n_mics = 8, 211.0, 4
    par = _params(
        n_rotors=1,
        n_mics=n_mics,
        k_cap=k_cap,
        profile_db=np.linspace(-14.0, -26.0, k_cap)[None, :],
        carrier=[f0],
        sigma_nu=0.9,
        lam=5.5,
        gamma_hz=0.05,
        floor_mean_db=-46.0,
        tilt=0.0,
    )
    n = 1 << 14
    grid = SP.bench_grid(n=n, sr=SR)
    truth = SP.bench_model(grid, par, k_max=k_cap).numpy()
    rng = np.random.default_rng(4)
    # exponential periodogram cells around the truth: the Whittle likelihood's
    # own sampling distribution, so the MAP is being asked exactly its question
    obs = truth * rng.standard_exponential(truth.shape)

    batch = MD.bench_batch(
        name="planted", power=obs, sr=SR, carrier_mean=np.array([f0]), n_samples=n, k_cap=k_cap
    )
    # the bench carrier is FROZEN at the support's own (window-refined) value:
    # no site, no refinement, and the batch carries it through unchanged
    assert batch.carrier_init is not None
    assert float(batch.carrier_init[0]) == f0
    assert "carrier" not in MD.free_blocks("bench")
    out = FT.fit_support(
        batch, mode="bench", optim=FT.OptimSpec(adam_steps=150, adam_lr=0.05, lbfgs_iters=60)
    )
    got = MD.params_to_dict(out.params)
    assert got["carrier_rev_s"][0] == f0
    assert 0.4 < got["sigma_nu"] < 2.0
    prof = np.asarray(got["profile"]["profile_db"])[0]
    assert np.abs(prof - np.linspace(-14.0, -26.0, k_cap)).max() < 3.0
    assert out.objective["n_cells"] == batch.n_cells
    assert out.objective["per_band"]["comb"] + out.objective["per_band"]["floor"] == pytest.approx(
        out.objective["whittle_nats"], rel=1e-9
    )


def test_a_profile_init_moves_only_the_orders_it_carries_a_width_for():
    """``ProfileInit`` is a PRIOR, per line, and only where it says so.

    The four-motor rig fit enters the multi-rotor estimator's profile this way
    (``scripts/noise_v2_fourmotor.py``), so the contract a consumer sees is:
    an order handed a centre AND a tight width ends at that centre even when
    the data disagrees by 12 dB, while an order handed a centre but NO width
    keeps following the data. (It does not stay put to the decibel: pinning
    half the comb 12 dB up moves the floor and the dynamics with it, so the
    free orders shift a few dB — what must hold is that they follow the data
    and not the offer.) Planted the same way as the MAP test above.
    """
    k_cap, f0, n_mics = 8, 211.0, 4
    truth_db = np.linspace(-14.0, -26.0, k_cap)
    par = _params(
        n_rotors=1,
        n_mics=n_mics,
        k_cap=k_cap,
        profile_db=truth_db[None, :],
        carrier=[f0],
        sigma_nu=0.9,
        lam=5.5,
        gamma_hz=0.05,
        floor_mean_db=-46.0,
        tilt=0.0,
    )
    n = 1 << 14
    grid = SP.bench_grid(n=n, sr=SR)
    obs = SP.bench_model(grid, par, k_max=k_cap).numpy() * np.random.default_rng(
        4
    ).standard_exponential((n_mics, 1, n // 2 + 1))
    batch = MD.bench_batch(
        name="planted", power=obs, sr=SR, carrier_mean=np.array([f0]), n_samples=n, k_cap=k_cap
    )
    optim = FT.OptimSpec(adam_steps=150, adam_lr=0.05, lbfgs_iters=60)
    free = np.asarray(
        MD.params_to_dict(FT.fit_support(batch, mode="bench", optim=optim).params)["profile"][
            "profile_db"
        ]
    )[0]

    # a centre 12 dB above the truth on every order, but a WIDTH only below k = 5
    centre = truth_db + 12.0
    sigma = np.array([0.05] * 4 + [np.nan] * 4)
    init = FT.ProfileInit(profile_db=centre[None, :], sigma_db=sigma[None, :], source="unit-test")
    out = FT.fit_support(batch, mode="bench", optim=optim, profile_init=init)
    got = np.asarray(MD.params_to_dict(out.params)["profile"]["profile_db"])[0]

    assert np.abs(got[:4] - centre[:4]).max() < 1.0
    assert np.abs(got[4:] - free[4:]).max() < np.abs(got[4:] - centre[4:]).min()
    assert out.diagnostics["profile_init"]["n_widths"] == 4
    assert out.diagnostics["profile_init"]["n_centres"] == 4


def test_floor_only_mode_freezes_the_combs_shape_and_frees_only_its_level():
    """``--floor-only`` must create no PER-ORDER comb site: the comb's shape and
    dynamics arrive frozen and the only comb freedom is one shared level."""
    k_cap, n_mics = 6, 2
    n_fft, hop, n_frames = 512, 256, 4
    n = n_fft + (n_frames - 1) * hop
    rps = np.full((1, n), 180.0)
    grid = SP.flight_grid(sr=SR, n_fft=n_fft, hop=hop)
    par = _params(
        n_rotors=1,
        n_mics=n_mics,
        k_cap=k_cap,
        profile_db=np.linspace(-16.0, -24.0, k_cap)[None, :],
        amp_exp=2.0,
        floor_exp=2.0,
        static_rel=0.01,
    )
    starts = np.arange(n_frames) * hop
    with torch.no_grad():
        truth = SP.flight_model(
            grid, par, rate_work=SP.flight_rate_work(grid, rps, starts), k_max=k_cap
        ).numpy()
    obs = truth * np.random.default_rng(6).standard_exponential(truth.shape)
    batch = MD.flight_batch(
        name="floor_only",
        members=[("w0", obs, rps, starts)],
        sr=SR,
        n_fft=n_fft,
        hop=hop,
        k_cap=k_cap,
    )
    frozen = MD.frozen_from_params(MD.params_to_dict(par))
    out = FT.fit_support(
        batch,
        mode="flight_floor_only",
        frozen=frozen,
        optim=FT.OptimSpec(
            adam_steps=20, adam_lr=0.05, adam_batch=None, lbfgs_iters=10, lbfgs_frames=None
        ),
    )
    got = MD.params_to_dict(out.params)
    # the recorded comb is the frozen one shifted by the ONE free scalar, so
    # every order moved by the same amount: no per-order comb site exists
    planted = np.asarray(MD.params_to_dict(par)["profile"]["profile_db"], dtype=np.float64)
    assert out.comb_gain_db is not None
    shift = np.asarray(got["profile"]["profile_db"], dtype=np.float64) - planted
    assert np.abs(shift - out.comb_gain_db).max() < 1e-12
    assert got["sigma_nu"] == pytest.approx(float(par.sigma_nu), rel=1e-12)
    assert got["floor"]["floor_mean_db"] != pytest.approx(float(par.floor.mean_db), rel=1e-6)


def test_floor_only_fit_recovers_the_level_of_a_transplanted_comb():
    """A frozen comb handed over 20 dB below the truth must be re-levelled.

    This is the DREGON transplant in miniature: the comb is frozen from another
    rig, ``render_noise`` mean-centres both mic-gain blocks so ``profile_db`` is
    the only absolute comb scale there is, and without a free level the fit
    renders a comb 20 dB under the one the data carries
    (``results/noise_v2/rounds/round2/render_dregon/findings.md``).
    """
    planted_offset_db = 20.0
    k_cap, n_mics = 6, 2
    n_fft, hop, n_frames = 512, 256, 8
    n = n_fft + (n_frames - 1) * hop
    rps = np.full((1, n), 180.0)
    grid = SP.flight_grid(sr=SR, n_fft=n_fft, hop=hop)
    par = _params(
        n_rotors=1,
        n_mics=n_mics,
        k_cap=k_cap,
        profile_db=np.linspace(-16.0, -24.0, k_cap)[None, :],
        amp_exp=2.0,
        floor_exp=2.0,
        static_rel=0.01,
    )
    starts = np.arange(n_frames) * hop
    with torch.no_grad():
        truth = SP.flight_model(
            grid, par, rate_work=SP.flight_rate_work(grid, rps, starts), k_max=k_cap
        ).numpy()
    obs = truth * np.random.default_rng(17).standard_exponential(truth.shape)
    batch = MD.flight_batch(
        name="transplant",
        members=[("w0", obs, rps, starts)],
        sr=SR,
        n_fft=n_fft,
        hop=hop,
        k_cap=k_cap,
    )
    frozen = MD.frozen_from_params(MD.params_to_dict(par))
    frozen["profile_db"] = (
        np.asarray(frozen["profile_db"], dtype=np.float64) - planted_offset_db
    ).tolist()
    out = FT.fit_support(
        batch,
        mode="flight_floor_only",
        frozen=frozen,
        optim=FT.OptimSpec(
            adam_steps=60, adam_lr=0.05, adam_batch=None, lbfgs_iters=40, lbfgs_frames=None
        ),
    )
    assert out.comb_gain_db == pytest.approx(planted_offset_db, abs=1.0)
    # and the re-levelled comb IS the truth's comb again, to the same tolerance
    got = np.asarray(MD.params_to_dict(out.params)["profile"]["profile_db"], dtype=np.float64)
    assert np.abs(got - np.asarray(MD.params_to_dict(par)["profile"]["profile_db"])).max() < 1.0
    # and the recorded initialisation is the INITIALISATION: AutoDelta adopts
    # the tensors it is initialised from and updates them in place, so a fit
    # that hands it the init dict itself reports every fit as one that never
    # moved (this is what made the R2 DREGON floor fit look frozen at its seed)
    init = FT.initial_values(batch, mode="flight_floor_only", frozen=frozen)
    assert out.diagnostics["init_comb_gain_db"] == pytest.approx(
        float(init["comb_gain_db"]), rel=1e-12
    )
    assert out.diagnostics["init_comb_gain_db"] != pytest.approx(out.comb_gain_db, rel=1e-9)


def test_a_pinned_dynamics_scalar_is_held_and_the_rest_is_still_fitted():
    """``pin`` must hold exactly the named dynamics scalar while the rest of
    the block is fitted.

    This is the identified-ridge recipe's contract: a rate the data cannot
    constrain is a CONSTANT of the model (no guide parameter, no prior term),
    not a parameter under a tight prior, and the recorded parameters carry the
    pinned value verbatim.
    """
    k_cap, f0, n_mics = 8, 211.0, 2
    par = _params(
        n_rotors=1,
        n_mics=n_mics,
        k_cap=k_cap,
        profile_db=np.linspace(-14.0, -26.0, k_cap)[None, :],
        carrier=[f0],
        sigma_nu=0.9,
        lam=5.5,
        gamma_hz=0.05,
        floor_mean_db=-46.0,
        tilt=0.0,
    )
    grid = SP.bench_grid(n=1 << 14, sr=SR)
    truth = SP.bench_model(grid, par, k_max=k_cap).numpy()
    obs = truth * np.random.default_rng(11).standard_exponential(truth.shape)
    batch = MD.bench_batch(
        name="pinned", power=obs, sr=SR, carrier_mean=np.array([f0]), k_cap=k_cap
    )
    pin = MD.dynamics_pin({"lam": 7.25})
    assert pin == {"lam": 7.25}
    with pytest.raises(ValueError, match="cannot pin"):
        MD.dynamics_pin({"lam_eps_odd": 3.5})
    out = FT.fit_support(
        batch,
        mode="bench",
        pin=pin,
        optim=FT.OptimSpec(adam_steps=60, adam_lr=0.05, lbfgs_iters=20),
    )
    got = MD.params_to_dict(out.params)
    assert got["lam"] == pytest.approx(7.25, rel=1e-12)
    # the free coordinates moved off their prior-centre initialisation
    assert got["sigma_nu"] != pytest.approx(math.exp(MD.PRIORS.log_sigma_nu[0]), rel=1e-6)
    assert np.asarray(got["gamma_hz"]).shape == (1, k_cap)
    assert out.optimiser["pinned_dynamics"] == {"lam": 7.25}


def test_a_round_one_fit_still_renders_through_the_schema_loader():
    """A ``noise-v2-fit/1`` payload must still render, with its per-order OU
    mapped onto the equivalent Lorentzian width.

    R1 and R2 published fits in ``/1``; R3 must not orphan them. The mapping
    ``gamma_rk = sigma_eps^2 k^p lam_eps / (2 pi)`` (by the parity of ``k``)
    is the short-lag equivalence of the two laws, and this checks it lands on
    the right numbers AND that the renderer accepts the old schema tag.
    """
    k_cap, f0 = 6, 149.0
    fit = _planted_fit(k_cap=k_cap, f0=f0, n_mics=2)

    params = {kk: v for kk, v in fit["params"].items() if kk != "gamma_hz"}
    params.update(sigma_eps_even=0.2, sigma_eps_odd=0.5, lam_eps_even=3.0, lam_eps_odd=4.0, p=1.0)
    old = dict(schema="noise-v2-fit/1", params=params)

    gamma = MD.gamma_from_params(params)
    k = np.arange(1, k_cap + 1)
    want = np.where(k % 2 == 0, 0.2**2 * 3.0, 0.5**2 * 4.0) * k / (2.0 * math.pi)
    assert gamma.shape == (1, k_cap)
    assert np.abs(gamma[0] - want).max() < 1e-12

    n = SR  # one second is enough: this is a schema check, not a level check
    audio = RD.render_noise(old, np.full((1, n), f0), sr=SR, n_mics=2, seed=3)
    assert isinstance(audio, np.ndarray)
    assert audio.shape == (2, n) and np.isfinite(audio).all()
    # and the frozen mapping a later fit would read carries the same widths
    assert np.allclose(np.asarray(MD.frozen_from_params(params)["gamma_hz"]), gamma)


def test_expected_periodogram_frames_match_the_evaluator_grid():
    """``expected_periodogram`` must use ``data.periodogram``'s own framing."""
    k_cap, f0 = 5, 180.0
    fit = _planted_fit(k_cap=k_cap, f0=f0, n_mics=2)
    n_fft, hop = 512, 256
    n = n_fft + 5 * hop
    rps = np.full((1, n), f0)
    m = RD.expected_periodogram(fit, rps, n_fft=n_fft, hop=hop, sr=SR, n_mics=2)
    pg = RE.window_periodogram(
        Clip("z", "synthetic", np.zeros((2, n), dtype=np.float32), rps, SR), n_fft=n_fft, hop=hop
    )
    assert m.shape == (2, pg.power.shape[1], n_fft // 2 + 1)
    assert np.isfinite(m).all() and (m > 0).all()


def test_the_low_order_gamma_check_names_the_shaft_absorbing_ramp_only():
    """The check must pass unresolved low orders and fail a VISIBLE k^2 ramp.

    Widths under the window's resolution are one statement — "the shaft alone
    sets this line's shape" — and their ratio is noise, so the floor test is
    decisive; a ramp is only a verdict once the widths are above the floor,
    which is the degeneracy the explainer's identification section names.
    """
    n = 1 << 14  # 1.024 s at 16 kHz: resolution 1 / (2 T) = 0.488 Hz
    batch = MD.bench_batch(
        name="check",
        power=np.ones((1, 1, n // 2 + 1)),
        sr=SR,
        carrier_mean=np.array([200.0]),
        n_samples=n,
        k_cap=4,
    )
    res = batch.resolution_hz
    assert res == pytest.approx(0.5 * SR / n)

    tiny = FT.gamma_low_order_check(np.array([[1e-4, 3e-3, 6e-3, 1e-2]]), batch=batch)
    assert tiny["verdict"] == "pass", "all four widths are a fraction of a bin"
    assert tiny["k4_over_k1"] > FT.GAMMA_RAMP_RATIO, "a ratio that must NOT decide it"

    ramp = FT.gamma_low_order_check(res * np.array([[4.0, 16.0, 36.0, 64.0]]), batch=batch)
    assert ramp["verdict"] == "fail_k2_ramp"
    assert not ramp["passed"]

    wide = FT.gamma_low_order_check(res * np.array([[10.0, 10.0, 10.0, 10.0]]), batch=batch)
    assert wide["verdict"] == "fail_above_floor"


def test_flight_floor_lowk_recovers_a_per_order_low_order_gain():
    """``flight_floor_lowk`` must re-level the frozen comb's LOW orders per
    order while the shared scalar stays put.

    The DREGON transplant in miniature: the comb is frozen from another rig and
    its low orders arrive 15 dB off while the rest of it is right. One shared
    scalar cannot fix that without dragging the whole comb (R2: the low orders
    ended ~20 dB too quiet), so the mode frees one gain per order below the
    cut, shared across rotors. What must come back is the per-order gain at
    k <= K_low and a comb_gain_db still near zero.
    """
    planted_low_db, k_low = 15.0, 4
    k_cap, n_mics = 8, 2
    n_fft, hop, n_frames = 512, 256, 8
    n = n_fft + (n_frames - 1) * hop
    rps = np.full((1, n), 180.0)
    grid = SP.flight_grid(sr=SR, n_fft=n_fft, hop=hop)
    par = _params(
        n_rotors=1,
        n_mics=n_mics,
        k_cap=k_cap,
        profile_db=np.linspace(-16.0, -24.0, k_cap)[None, :],
        amp_exp=2.0,
        floor_exp=2.0,
        static_rel=0.01,
    )
    starts = np.arange(n_frames) * hop
    with torch.no_grad():
        truth = SP.flight_model(
            grid, par, rate_work=SP.flight_rate_work(grid, rps, starts), k_max=k_cap
        ).numpy()
    obs = truth * np.random.default_rng(23).standard_exponential(truth.shape)
    batch = MD.flight_batch(
        name="lowk", members=[("w0", obs, rps, starts)], sr=SR, n_fft=n_fft, hop=hop, k_cap=k_cap
    )
    frozen = MD.frozen_from_params(MD.params_to_dict(par))
    prof = np.asarray(frozen["profile_db"], dtype=np.float64)
    prof[:, :k_low] -= planted_low_db  # only the LOW orders are wrong
    frozen["profile_db"] = prof.tolist()

    assert MD.free_blocks("flight_floor_lowk") == ("floor", "mic", "comb_gain", "low_order_gain")
    out = FT.fit_support(
        batch,
        mode="flight_floor_lowk",
        frozen=frozen,
        low_orders=k_low,
        optim=FT.OptimSpec(
            adam_steps=120, adam_lr=0.05, adam_batch=None, lbfgs_iters=60, lbfgs_frames=None
        ),
    )
    assert out.low_order_gain_db is not None
    got = np.asarray(out.low_order_gain_db, dtype=np.float64)
    assert got.shape == (k_low,)
    assert np.abs(got - planted_low_db).max() < 2.0
    assert out.comb_gain_db == pytest.approx(0.0, abs=2.0)
    # and the recorded comb IS the truth again: both gains are folded into it
    fitted = np.asarray(MD.params_to_dict(out.params)["profile"]["profile_db"], dtype=np.float64)
    planted = np.asarray(MD.params_to_dict(par)["profile"]["profile_db"], dtype=np.float64)
    assert np.abs(fitted - planted).max() < 2.5
