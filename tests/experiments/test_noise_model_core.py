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
    sigma_eps=(0.25, 0.6),
    lam_eps=(2.0, 6.0),
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
        sigma_eps=_t(sigma_eps),
        lam_eps=_t(lam_eps),
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


def test_lag_law_without_per_order_term_is_c4s_prior_lag_law():
    """``sigma_eps = 0`` must leave EXACTLY C4's shaft-only prior ``R_k(tau)``.

    The v2 law adds an independent per-order term to C4's; if the shaft factor
    itself had drifted (a factor of two in ``V_theta``, ``k`` instead of
    ``k^2``), every line width in the campaign would move and no other test
    here would localise it.
    """
    tau = np.linspace(0.0, 2.0, 257)
    k = np.array([1.0, 2.0, 7.0, 20.0, 21.0, 130.0])[:, None]
    for sigma_nu, lam in ((0.45, 5.5), (1.8, 0.67), (0.31, 14.4)):
        got = r_tau(tau[None, :], k, sigma_nu=sigma_nu, lam=lam, sigma_eps=0.0, lam_eps=2.0)
        want = RP.prior_r_tau(tau[None, :], k, lam=lam, sigma=sigma_nu, d=0.0)
        assert np.abs(got - want).max() < 1e-9


def test_lag_law_splits_the_per_order_term_by_parity():
    """The parity pair must select on ``k``, and saturate at ``exp(-sigma^2 k^p)``."""
    tau = np.array([0.0, 1e4])  # 0 and "infinity" against lam_eps
    k = np.array([20.0, 21.0])[:, None]
    got = r_tau(tau[None, :], k, sigma_nu=1e-12, lam=5.5, sigma_eps=(0.2, 0.5), lam_eps=(3.0, 3.0))
    assert got[0, 0] == pytest.approx(1.0, abs=1e-12)
    assert got[0, 1] == pytest.approx(math.exp(-(0.2**2) * 20.0), rel=1e-9)
    assert got[1, 1] == pytest.approx(math.exp(-(0.5**2) * 21.0), rel=1e-9)


def test_lag_law_is_differentiable_in_every_parameter():
    """The fit needs a gradient in all six dynamics parameters, not just sigma."""
    tau = torch.linspace(0.0, 0.5, 65, dtype=torch.float64)
    k = torch.arange(1.0, 9.0, dtype=torch.float64)[:, None]
    args = {
        "sigma_nu": torch.tensor(0.45, dtype=torch.float64, requires_grad=True),
        "lam": torch.tensor(5.5, dtype=torch.float64, requires_grad=True),
        "sigma_eps": torch.tensor([0.3, 0.5], dtype=torch.float64, requires_grad=True),
        "lam_eps": torch.tensor([2.0, 1.5], dtype=torch.float64, requires_grad=True),
    }
    r_tau(
        tau[None, :],
        k,
        sigma_nu=args["sigma_nu"],
        lam=args["lam"],
        sigma_eps=args["sigma_eps"],
        lam_eps=args["lam_eps"],
    ).sum().backward()
    for name, v in args.items():
        assert v.grad is not None, name
        assert torch.isfinite(v.grad).all(), name
        assert float(v.grad.abs().sum()) > 0.0, name


def test_bench_spectrum_has_a_gradient_for_every_dynamics_parameter():
    """Every one of the six dynamics values must move the expected spectrum.

    This is deliberately a forward-model check, rather than only differentiating
    ``r_tau``: it catches a NumPy/detach conversion while a Pyro sample is being
    carried through the finite-window transform.
    """
    par = _params(
        n_rotors=1,
        n_mics=1,
        k_cap=8,
        profile_db=np.linspace(-14.0, -27.0, 8)[None, :],
        carrier=[173.0],
    )
    dynamics = {
        "sigma_nu": par.sigma_nu.requires_grad_(),
        "lam": par.lam.requires_grad_(),
        "sigma_eps": par.sigma_eps.requires_grad_(),
        "lam_eps": par.lam_eps.requires_grad_(),
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
    par = _params(
        n_rotors=1,
        n_mics=n_mics,
        k_cap=k_cap,
        profile_db=prof,
        carrier=[f0],
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
        sigma_eps=_t([0.25, 0.6]),
        lam_eps=_t([2.0, 6.0]),
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
    """FLIGHT mode must equal C4's ``frame_model`` to 1e-6 at ``sigma_eps = 0``.

    v2's flight forward model is C4's with ONE substitution — the prior lag law
    at the kernel call. With the per-order term switched off and C4's ``D``
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
        sigma_eps=(0.0, 0.0),
        lam_eps=(2.0, 2.0),
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
        sigma_eps=(0.0, 0.0),
        lam_eps=(2.0, 2.0),
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
        sigma_eps=(0.2, 0.35),
        lam_eps=(2.0, 3.0),
        floor_mean_db=-52.0,
        tilt=-3.0,
        amp_exp=0.0,
        floor_exp=0.0,
        static_rel=1.0,
    )
    return dict(schema="noise-v2-fit/1", params=MD.params_to_dict(par))


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
    """Order 20 demodulated out of the render must carry the fitted ``V_eps``.

    The renderer draws an OU per order; the fit assumes its increment variance
    is ``2 sigma_eps^2 k^p (1 - exp(-lam_eps tau))``. A Wiener draw, a wrong
    parity or a missing ``k^{p/2}`` in the drive all show up as a wrong
    ``V_eps(20, tau)``, measured here with :mod:`utils.demod` on a render whose
    shaft term is switched off so only the per-order process is left.
    """
    from utils.demod import demodulate

    k, f0, dur, sigma_eps, lam_eps = 20, 149.0, 8.0, 0.35, 3.0
    fit = _planted_fit(k_cap=k, f0=f0, n_mics=1)
    p = fit["params"]
    p["sigma_nu"] = 1e-9
    p["sigma_eps_even"] = p["sigma_eps_odd"] = sigma_eps
    p["lam_eps_even"] = p["lam_eps_odd"] = lam_eps
    p["floor_mean_db"] = -300.0
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
        want = 2.0 * sigma_eps**2 * k * (1.0 - math.exp(-lam_eps * tau_s))
        assert abs(got - want) / want < 0.2, (tau_s, got, want)


# ── the objective and the fit ───────────────────────────────────────────────


def test_bench_map_recovers_planted_dynamics_at_the_frozen_carrier():
    """A planted bench support must be recovered by the MAP fit end to end.

    Small by design (one rotor, eight orders, 1 s, two microphones): what it
    proves is that the Pyro model, the ``AutoDelta`` guide, the initialisation,
    the Adam phase, the L-BFGS polish, the carrier refinement and the JSON all
    compose — a wrong sign on the Whittle factor or a mis-scoped frozen block
    cannot pass it.
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
        sigma_eps=(0.2, 0.2),
        lam_eps=(2.0, 2.0),
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


def test_floor_only_mode_freezes_the_comb():
    """``--floor-only`` must create no comb site and leave the comb untouched."""
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
    assert got["profile"]["profile_db"] == MD.params_to_dict(par)["profile"]["profile_db"]
    assert got["sigma_nu"] == pytest.approx(float(par.sigma_nu), rel=1e-12)
    assert got["floor"]["floor_mean_db"] != pytest.approx(float(par.floor.mean_db), rel=1e-6)


def test_pinned_dynamics_coordinates_are_held_and_the_rest_is_still_fitted():
    """``pin`` must hold exactly the named coordinates — including ONE parity of
    the two-vector sites — while the rest of the dynamics block is fitted.

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
        sigma_eps=(0.2, 0.2),
        lam_eps=(2.0, 2.0),
        floor_mean_db=-46.0,
        tilt=0.0,
    )
    grid = SP.bench_grid(n=1 << 14, sr=SR)
    truth = SP.bench_model(grid, par, k_max=k_cap).numpy()
    obs = truth * np.random.default_rng(11).standard_exponential(truth.shape)
    batch = MD.bench_batch(
        name="pinned", power=obs, sr=SR, carrier_mean=np.array([f0]), k_cap=k_cap
    )
    pin = MD.dynamics_pin({"lam": 7.25, "lam_eps_odd": 3.5})
    assert pin == {"lam": 7.25, "lam_eps": [None, 3.5]}
    out = FT.fit_support(
        batch,
        mode="bench",
        pin=pin,
        optim=FT.OptimSpec(adam_steps=60, adam_lr=0.05, lbfgs_iters=20),
    )
    got = MD.params_to_dict(out.params)
    assert got["lam"] == pytest.approx(7.25, rel=1e-12)
    assert got["lam_eps_odd"] == pytest.approx(3.5, rel=1e-12)
    # the free coordinates moved off their prior-centre initialisation
    assert got["lam_eps_even"] != pytest.approx(math.exp(MD.PRIORS.log_lam_eps[0]), rel=1e-6)
    assert got["sigma_nu"] != pytest.approx(math.exp(MD.PRIORS.log_sigma_nu[0]), rel=1e-6)
    assert out.optimiser["pinned_dynamics"] == {"lam": 7.25, "lam_eps": [None, 3.5]}


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
