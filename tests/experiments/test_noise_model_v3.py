"""Noise model v3 (``docs/explainers/noise-model-v3-wander.qmd``): its priors,
its data normalisation, its block wander and a synthetic recovery.

Each test pins a consumer-visible property of the ``flight_v3`` mode against
something derived independently of the implementation: the half-normal
density in closed form, the explicit Gaussian of an OU chain, the v2 forward
model bit for bit, or a planted rig. CPU-small by construction.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Any

import numpy as np
import pyro
import pyro.distributions as dist
import pytest
import torch
from pyro.infer import Trace_ELBO
from pyro.infer.autoguide import AutoDelta, init_to_value

from data_processing.noise_model.floor import floor_geometry
from experiments.noise_model import fit as FT
from experiments.noise_model import model as MD
from experiments.noise_model import spectrum as SP

SR = 16000
NC = SP.FLOOR_SHAPE_N_CTRL


def _t(v: Any) -> torch.Tensor:
    return torch.as_tensor(np.asarray(v, dtype=np.float64), dtype=torch.float64)


def _params(
    *,
    k_cap: int,
    n_mics: int,
    profile_db: Any,
    shape_z: Any = None,
    floor_mean_db: float = -40.0,
    tilt: float = 0.0,
) -> SP.V2Params:
    prof = np.broadcast_to(np.asarray(profile_db, dtype=np.float64), (1, k_cap)).copy()
    return SP.V2Params(
        sigma_nu=_t(0.3),
        lam=_t(0.5),
        gamma_hz=_t(np.full((1, k_cap), 0.02)),
        profile_db=_t(prof),
        floor=SP.FloorParams(
            mean_db=_t(floor_mean_db),
            shape_z=_t(np.zeros(NC) if shape_z is None else shape_z),
            tilt_db_oct=_t(tilt),
            mic_floor_db=_t(np.zeros(n_mics)),
            exp=_t(2.0),
            static_rel=_t(2.5e-3),
        ),
        mic_line_gain_db=_t(np.zeros((n_mics, 1))),
        gain_all_db=_t(np.zeros(n_mics)),
        amp_exp=_t(2.0),
    )


def _flight_batch(
    par: SP.V2Params,
    *,
    k_cap: int,
    n_mics: int,
    n_frames: int = 16,
    seed: int = 3,
    f0: float = 180.0,
    **kw: Any,
) -> MD.SupportBatch:
    """A one-window flight batch whose cells are the model's own ``M`` times
    unit exponentials — the Whittle likelihood's exact generative law."""
    n_fft, hop = 512, 256
    n = n_fft + (n_frames - 1) * hop
    rps = np.full((1, n), f0)
    grid = SP.flight_grid(sr=SR, n_fft=n_fft, hop=hop)
    starts = np.arange(n_frames) * hop
    with torch.no_grad():
        truth = SP.flight_model(
            grid, par, rate_work=SP.flight_rate_work(grid, rps, starts), k_max=k_cap
        ).numpy()
    obs = truth * np.random.default_rng(seed).standard_exponential(truth.shape)
    return MD.flight_batch(
        name="v3",
        members=[("w0", obs[:n_mics], rps, starts)],
        sr=SR,
        n_fft=n_fft,
        hop=hop,
        k_cap=k_cap,
        **kw,
    )


def _measured(batch: MD.SupportBatch, priors: MD.PriorsV3 = MD.PRIORS_V3) -> MD.SupportBatch:
    return dataclasses.replace(
        batch, measured=FT.measure_batch(batch, mode=MD.V3_MODE, priors=priors)
    )


def _trace(batch: MD.SupportBatch, priors: MD.PriorsV3 = MD.PRIORS_V3) -> Any:
    return pyro.poutine.trace(
        lambda: MD.sample_params(batch, mode=MD.V3_MODE, priors=priors)
    ).get_trace()


def _sites(trace: Any) -> set[str]:
    return {
        name
        for name, node in trace.nodes.items()
        if node.get("type") == "sample" and not node.get("is_observed", False)
    }


# ── (a) the dynamics priors ─────────────────────────────────────────────────


def test_line_width_and_shaft_priors_are_light_tailed_in_linear_units():
    """A 13 Hz half width at ``k = 1`` must be ruled out, a bench-law one free.

    ``gamma_rk / (0.01 k) ~ HalfNormal(3)`` charges ``g^2 / (2 s^2)`` with
    ``s = 0.03 k`` Hz: 13 Hz at k = 1 costs ~ 9.4e4 nats, where v2's
    ``LN(log 0.01 k, 1)`` charged ~ 40 and let R4's fit park there. A 0.05 Hz
    line — five bench-law medians — must stay nearly free.
    """
    k_cap = 4
    batch = _measured(
        _flight_batch(_params(k_cap=k_cap, n_mics=2, profile_db=-20.0), k_cap=k_cap, n_mics=2)
    )
    tr = _trace(batch)
    gamma_fn = tr.nodes["gamma_hz"]["fn"].base_dist
    assert isinstance(gamma_fn, dist.HalfNormal)

    def cost(value_hz: float) -> float:
        g = torch.full((1, k_cap), float(value_hz), dtype=torch.float64)
        return -float(gamma_fn.log_prob(g)[0, 0])

    assert cost(13.0) > 1e4
    assert cost(0.05) < 5.0
    # the scale is the bench decoherence law times c_gamma, order by order
    # the scale is the bench decoherence law times c_gamma, order by order
    np.testing.assert_allclose(gamma_fn.scale.numpy()[0], 3.0 * 0.01 * np.arange(1, k_cap + 1))
    sigma_fn = tr.nodes["sigma_nu"]["fn"]
    assert isinstance(sigma_fn, dist.HalfNormal)
    assert float(sigma_fn.scale) == 0.6
    # closed-form half-normal: log(sqrt(2/pi)/s) - x^2 / (2 s^2)
    x = torch.tensor(1.7, dtype=torch.float64)
    want = math.log(math.sqrt(2.0 / math.pi) / 0.6) - 1.7**2 / (2.0 * 0.6**2)
    assert abs(float(sigma_fn.log_prob(x)) - want) < 1e-12


# ── (b) the profile prior ───────────────────────────────────────────────────


def test_every_order_gets_the_same_profile_prior_no_visibility_switch():
    """Visible or not, a line's prior is ``N(pooled level at k f_r, 10 dB)``.

    Half of the planted comb is silent (-300 dB): v2 would park those orders
    at ``N(floor - 15, 8)``. v3 centres them on the pooled level at their own
    frequency, like every other order, with the same 10 dB sd.
    """
    k_cap = 6
    prof = np.array([-15.0, -300.0, -18.0, -300.0, -22.0, -300.0])
    # lines 400 Hz = 13 bins apart, so no order's window reaches a neighbour
    par = _params(k_cap=k_cap, n_mics=2, profile_db=prof)
    batch = _measured(_flight_batch(par, k_cap=k_cap, n_mics=2, f0=400.0))
    meas = batch.measured
    assert meas is not None
    silent = prof[None, :] < -100.0
    # the silent orders really are invisible: v2's switch would have fired
    assert np.all(meas.line_snr_db[silent] < MD.PRIORS.line_visible_snr_db)

    base = _trace(batch).nodes["profile_db"]["fn"].base_dist
    assert isinstance(base, dist.Normal)
    np.testing.assert_array_equal(base.loc.numpy(), np.asarray(meas.profile_db))
    np.testing.assert_array_equal(base.scale.numpy(), np.full((1, k_cap), 10.0))
    loc = np.asarray(meas.profile_db)
    # v2 would have parked the silent orders at N(mu - 15, 8); v3 does not
    park = meas.floor_mean_db + MD.PRIORS.profile_below_offset_db
    assert np.all(np.abs(loc[silent] - park) > 1.0)
    # the silent orders all measure the same thing: the floor under them
    assert float(np.ptp(loc[silent])) < 3.0
    # and a visible one's centre is an UPPER bound on its line: line + floor
    visible = ~silent
    assert np.all(loc[visible] >= prof[None, :][visible] - 1.5)


# ── (c) the floor ───────────────────────────────────────────────────────────


def test_floor_is_the_spline_alone_scaled_by_the_measured_sigma_b():
    """No mean or tilt site; control values ``mu + sigma_B (L z)``.

    A floor planted with a known shape must yield a measured ``sigma_B`` equal
    to the RMS of its control values about the measured level, and the
    forward model's floor must be exactly ``mu + sigma_B L z`` on the spline.
    """
    k_cap = 3
    z_true = np.random.default_rng(5).standard_normal(NC) * 1.5
    par = _params(k_cap=k_cap, n_mics=2, profile_db=-60.0, shape_z=z_true, floor_mean_db=-35.0)
    batch = _measured(_flight_batch(par, k_cap=k_cap, n_mics=2, n_frames=48, seed=9))
    meas = batch.measured
    assert meas is not None and meas.floor_shape_sd_db is not None

    grid = batch.grid
    assert isinstance(grid, SP.FlightGrid)
    c_true = SP.FLOOR_SHAPE_STD_DB * (grid.floor.shape_chol.numpy() @ z_true)
    basis = floor_geometry(grid.freqs_hz[grid.band], grid.floor.ctrl_hz)[0]
    c_rel = c_true - np.median(basis @ c_true)
    want = float(np.sqrt(np.mean(c_rel**2)))
    assert abs(meas.floor_shape_sd_db - want) < 1.0, (meas.floor_shape_sd_db, want)

    tr = _trace(batch)
    sites = _sites(tr)
    assert "floor_shape_z" in sites
    assert sites.isdisjoint({"floor_mean_db", "floor_tilt_db_oct"})

    z = torch.as_tensor(np.random.default_rng(1).standard_normal(NC))
    values = {name: tr.nodes[name]["value"] for name in sites}
    values["floor_shape_z"] = z
    p = MD.sample_params_from_values(batch, mode=MD.V3_MODE, priors=MD.PRIORS_V3, values=values)
    assert float(p.floor.mean_db) == meas.floor_mean_db
    assert float(p.floor.shape_sd_db) == meas.floor_shape_sd_db
    assert float(p.floor.tilt_db_oct) == 0.0
    want_ctrl = meas.floor_shape_sd_db * (grid.floor.shape_chol @ z)
    got_ctrl = grid.floor.shape_db(p.floor.shape_z, p.floor.shape_sd_db)
    assert torch.equal(got_ctrl, want_ctrl)
    psd = grid.floor.psd(p.floor)
    db = meas.floor_mean_db + grid.floor.shape_psd @ want_ctrl
    want_psd = grid.floor.rate_factor * 10.0 ** (db / 10.0)
    assert float(((psd - want_psd).abs() / want_psd).max()) < 1e-12


# ── (d) microphones: normalised in the data; DREGON's static wind term ──────


def test_no_mic_site_the_data_is_channel_normalised_and_wind_is_opt_in(tmp_path):
    """v3 has no microphone parameter; the channels are normalised in the DATA
    by the rank test's >= 500 Hz gains, and the per-mic wind term exists only
    when asked for and is identically zero from 500 Hz up."""
    import json

    k_cap, n_mics = 3, 2
    par = _params(k_cap=k_cap, n_mics=n_mics, profile_db=-20.0)
    # the rank-test record: the >= 500 Hz floor gains of a 3-mic rig; the fit
    # uses mics 0 and 2, re-centred on their own mean (+2 and -1 -> +1.5, -1.5)
    record = {
        "schema": "mic-gain-rank/1",
        "rigs": {"toy": {"wind": {"excess_db_above_500hz": [2.0, 9.0, -1.0]}}},
    }
    path = tmp_path / "mic_gains.json"
    path.write_text(json.dumps(record))
    gains = FT.load_channel_gains(path, rig="toy", mics=[0, 2])
    np.testing.assert_allclose(gains.gains_db, [1.5, -1.5])

    raw = _flight_batch(par, k_cap=k_cap, n_mics=n_mics)
    starts = np.arange(16) * 256
    rps = np.full((1, 512 + 15 * 256), 180.0)
    normed = MD.flight_batch(
        name="v3",
        members=[("w0", raw.power.numpy(), rps, starts)],
        sr=SR,
        n_fft=512,
        hop=256,
        k_cap=k_cap,
        channel_gains=gains,
    )
    ratio = (raw.power / normed.power).numpy()
    np.testing.assert_allclose(ratio[0], 10.0**0.15, rtol=1e-12)
    np.testing.assert_allclose(ratio[1], 10.0**-0.15, rtol=1e-12)
    assert normed.diagnostics["channel_gains"]["gains_db"] == [1.5, -1.5]

    plain = _measured(normed)
    sites = _sites(_trace(plain))
    assert sites.isdisjoint({"mic_line_gain_db", "gain_all_db", "mic_floor_db"})
    assert "wind_db" not in sites

    windy_priors = dataclasses.replace(MD.PRIORS_V3, wind=True)
    windy = _measured(normed, windy_priors)
    tr = _trace(windy, windy_priors)
    assert "wind_db" in _sites(tr)
    assert tr.nodes["wind_db"]["value"].shape == (n_mics,)

    values = {n: tr.nodes[n]["value"] for n in _sites(tr)}
    values["wind_db"] = torch.tensor([-30.0, -45.0], dtype=torch.float64)
    with_wind = MD.sample_params_from_values(
        windy, mode=MD.V3_MODE, priors=windy_priors, values=values
    )
    without = dataclasses.replace(with_wind, wind_db=None)
    with torch.no_grad():
        diff = (MD.forward(windy, with_wind) - MD.forward(windy, without)).numpy()
    f = windy.grid.freqs_hz
    assert np.all(diff[:, :, f >= 500.0] == 0.0)
    low = (f >= 30.0) & (f <= 100.0)
    # on the flat part the term IS its level, through the (unit) transfer there
    transfer = windy.grid.transfer_power.numpy()[low]
    want = np.broadcast_to(transfer[None, :], diff[0][:, low].shape)
    np.testing.assert_allclose(diff[0][:, low], 10.0**-3.0 * want, rtol=1e-9)
    np.testing.assert_allclose(diff[1][:, low], 10.0**-4.5 * want, rtol=1e-9)


def test_measured_wind_centres_follow_a_planted_per_capsule_excess():
    """The wind prior centres are the per-mic low-band POWER excess over the
    quietest capsule: a mic with a planted wind 10 dB over the floor gets a
    centre near its planted level, the quiet one sinks."""
    k_cap, n_mics = 2, 2
    par = _params(k_cap=k_cap, n_mics=n_mics, profile_db=-60.0, floor_mean_db=-40.0)
    planted_db = np.array([-30.0, -300.0])
    par = dataclasses.replace(par, wind_db=_t(planted_db))
    batch = _flight_batch(par, k_cap=k_cap, n_mics=n_mics, n_frames=64, seed=13)
    priors = dataclasses.replace(MD.PRIORS_V3, wind=True)
    meas = FT.measure_batch(batch, mode=MD.V3_MODE, priors=priors)
    assert meas.wind_db is not None
    assert abs(meas.wind_db[0] - planted_db[0]) < 3.0
    assert meas.wind_db[1] < planted_db[0] - 15.0


# ── (e) the OU block prior ──────────────────────────────────────────────────


def test_ou_block_prior_is_the_explicit_gaussian_of_the_chain():
    """The Markov-factorised OU prior of a 4-block chain must equal the
    explicit Gaussian ``N(0, sigma^2 rho^|i - j|)``, log-determinant included,
    and the Pyro site distribution must be that density — per window over
    ITS OWN blocks when windows of different lengths share one padded site."""
    sigma, block_s, tau, n_blocks = 2.5, 0.5, 3.0, 4
    wander = MD.Wander(
        sigma_d_db=sigma,
        tau_d_s=tau,
        sigma_v_db=1.0,
        tau_v_s=1.0,
        sigma_u_db=1.0,
        tau_u_s=1.0,
        block_s=block_s,
    )
    rho = wander.rho("d")
    assert rho == math.exp(-block_s / tau)

    def chain(n: int) -> torch.distributions.MultivariateNormal:
        lag = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
        cov = torch.as_tensor(sigma**2 * rho**lag, dtype=torch.float64)
        return torch.distributions.MultivariateNormal(torch.zeros(n, dtype=torch.float64), cov)

    x = torch.as_tensor(np.random.default_rng(2).normal(0.0, 3.0, (5, n_blocks)))
    np.testing.assert_allclose(
        MD.ou_log_density(x, sigma, rho).numpy(), chain(n_blocks).log_prob(x).numpy(), rtol=1e-12
    )
    # two windows of 4 and 2 blocks, 5 tracks each, on one (W, 5, 4) site;
    # the padding of the short window must not count
    site = MD.OUTracks(sigma, rho, torch.tensor([4, 2]), (5,), 4)
    xs = torch.as_tensor(np.random.default_rng(3).normal(0.0, 3.0, (2, 5, n_blocks)))
    want = [float(chain(4).log_prob(xs[0]).sum()), float(chain(2).log_prob(xs[1, :, :2]).sum())]
    np.testing.assert_allclose(site.log_prob(xs).numpy(), want, rtol=1e-12)


# ── (f) the block forward model ─────────────────────────────────────────────


def _two_window_batch(par: SP.V2Params, *, k_cap: int, n_mics: int) -> MD.SupportBatch:
    n_fft, hop, n_frames = 512, 256, 12
    n = n_fft + (n_frames - 1) * hop
    grid = SP.flight_grid(sr=SR, n_fft=n_fft, hop=hop)
    starts = np.arange(n_frames) * hop
    members = []
    for w, f0 in enumerate((180.0, 195.0)):
        rps = (f0 + 5.0 * np.linspace(0.0, 1.0, n))[None, :]
        with torch.no_grad():
            m = SP.flight_model(
                grid, par, rate_work=SP.flight_rate_work(grid, rps, starts), k_max=k_cap
            ).numpy()
        members.append((f"w{w}", m, rps, starts))
    return MD.flight_batch(name="blocks", members=members, sr=SR, n_fft=n_fft, hop=hop, k_cap=k_cap)


def test_block_forward_model_at_zero_latents_is_flight_model():
    """v3's per-block forward model is v2's ``flight_model`` with two per-block
    multipliers: at zero latents it must BE ``flight_model`` (to rounding: the
    per-block floor curves go through one batched FFT), a latent must move
    only its own block, by exactly its dB, and evaluating the frames in chunks
    must change nothing."""
    k_cap, n_mics = 3, 2
    par = _params(
        k_cap=k_cap,
        n_mics=n_mics,
        profile_db=np.array([-18.0, -21.0, -25.0]),
        shape_z=np.random.default_rng(4).standard_normal(NC),
    )
    par = dataclasses.replace(
        par, floor=dataclasses.replace(par.floor, shape_sd_db=_t(4.0)), wind_db=_t([-50.0, -55.0])
    )
    wander = MD.Wander(
        sigma_d_db=3.0,
        tau_d_s=2.0,
        sigma_v_db=2.0,
        tau_v_s=2.0,
        sigma_u_db=1.0,
        tau_u_s=2.0,
        block_s=0.05,
        sigma_uj_db=1.0,
        tau_uj_s=2.0,
    )
    batch = MD.with_blocks(_two_window_batch(par, k_cap=k_cap, n_mics=n_mics), wander.block_s)
    assert all(nb >= 3 for nb in batch.window_blocks)
    zero = {
        w: MD.zero_latents(wander, n_rotors=1, k_max=k_cap, n_blocks=nb)
        for w, nb in enumerate(batch.window_blocks)
    }
    grid = batch.grid
    assert isinstance(grid, SP.FlightGrid) and batch.rate_work is not None
    with torch.no_grad():
        want = SP.flight_model(grid, par, rate_work=batch.rate_work, k_max=k_cap)
        got = MD.forward_v3(batch, par, zero)
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-13, atol=0.0)

    # +3 dB on rotor 0 in block 1 of window 0: that block's line power x 10^0.3,
    # every other frame untouched
    moved = {w: dataclasses.replace(lat) for w, lat in zero.items()}
    assert zero[0].d is not None
    d = torch.zeros_like(zero[0].d)
    d[0, 1] = 3.0
    moved[0] = dataclasses.replace(zero[0], d=d)
    quiet = dataclasses.replace(par, profile_db=torch.full((1, k_cap), -300.0, dtype=torch.float64))
    with torch.no_grad():
        got = MD.forward_v3(batch, par, moved)
        floor = MD.forward_v3(batch, quiet, zero)
    assert batch.frame_window is not None and batch.frame_block is not None
    hit = (batch.frame_window == 0) & (batch.frame_block == 1)
    assert hit.any()
    np.testing.assert_allclose(got[:, ~hit].numpy(), want[:, ~hit].numpy(), rtol=1e-13, atol=0.0)
    lines_before = (want - floor)[:, hit]
    lines_after = (got - floor)[:, hit]
    tol = 1e-12 * float(floor[:, hit].max())
    np.testing.assert_allclose(
        lines_after.numpy(), 10.0**0.3 * lines_before.numpy(), rtol=1e-9, atol=tol
    )

    # the same batch evaluated 5 frames at a time (chunks straddle the windows)
    chunked = dataclasses.replace(batch, latents=moved, chunk_frames=5)
    assert len(MD.frame_chunks(chunked)) > 2
    with torch.no_grad():
        np.testing.assert_allclose(
            MD.forward(chunked, par).numpy(), got.numpy(), rtol=1e-13, atol=0.0
        )


@pytest.mark.parametrize("amp_fitted", [False, True])
def test_the_unit_autocorrelation_line_kernel_is_the_atom_kernel(amp_fitted):
    """The lines summed through their UNIT-atom autocorrelation (the orders
    summed in the lag domain, one transform per rotor and frame, no autograd
    through the atoms) must be the atom kernel's expected periodogram — value
    and gradient in every line parameter — at per-frame line offsets, over a
    speed ramp and across harmonic chunks. A fitted ``amp_exp`` moves the
    atoms themselves: its gradient (the autocorrelation's tangent) must be the
    atom kernel's too."""
    k_cap, n_mics, n_frames, n_fft, hop = 5, 2, 6, 512, 256
    par = _params(
        k_cap=k_cap, n_mics=n_mics, profile_db=np.array([-18.0, -21.0, -25.0, -30.0, -28.0])
    )
    grid = SP.flight_grid(sr=SR, n_fft=n_fft, hop=hop)
    n = n_fft + (n_frames - 1) * hop
    starts = np.arange(n_frames) * hop
    rate = SP.flight_rate_work(grid, (180.0 + 25.0 * np.linspace(0.0, 1.0, n))[None, :], starts)
    rng = np.random.default_rng(7)
    line_db = _t(3.0 * rng.standard_normal((1, k_cap, n_frames)))
    weights = _t(rng.uniform(0.5, 1.5, (n_mics, n_frames, n_fft // 2 + 1)))

    def run(unit: bool) -> tuple[np.ndarray, list[np.ndarray]]:
        leaves = [
            par.sigma_nu.clone().requires_grad_(),
            _t(np.linspace(0.01, 0.4, k_cap)[None, :]).requires_grad_(),
            par.profile_db.clone().requires_grad_(),
            _t(3.1).requires_grad_(amp_fitted),
        ]
        p = dataclasses.replace(
            par, sigma_nu=leaves[0], gamma_hz=leaves[1], profile_db=leaves[2], amp_exp=leaves[3]
        )
        m = SP.flight_model(
            grid,
            p,
            rate_work=rate,
            k_max=k_cap,
            harmonic_chunk=2,
            line_db=line_db,
            unit_autocorr=unit,
        )
        (m * weights).sum().backward()
        return m.detach().numpy(), [x.grad.numpy() for x in leaves if x.grad is not None]

    want, want_grad = run(False)
    got, got_grad = run(True)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=0.0)
    assert len(got_grad) == len(want_grad) == (4 if amp_fitted else 3)
    for g, w in zip(got_grad, want_grad, strict=True):
        np.testing.assert_allclose(g, w, rtol=1e-9, atol=1e-12 * float(np.abs(w).max()))


def test_the_chunked_rig_objective_is_the_elbo_value_and_gradient():
    """The rig step's frame-chunked objective (the Whittle term accumulated
    chunk by chunk, each chunk's graph released) must be the ``Trace_ELBO``
    loss of the whole batch — value AND gradient in every guide parameter —
    with the latents on the batch and chunks straddling the windows."""
    k_cap, n_mics = 3, 2
    par = _params(
        k_cap=k_cap,
        n_mics=n_mics,
        profile_db=np.array([-18.0, -21.0, -25.0]),
        shape_z=np.random.default_rng(4).standard_normal(NC),
    )
    wander = MD.Wander(
        sigma_d_db=3.0,
        tau_d_s=2.0,
        sigma_v_db=2.0,
        tau_v_s=2.0,
        sigma_u_db=1.0,
        tau_u_s=2.0,
        block_s=0.05,
        sigma_uj_db=1.0,
        tau_uj_s=2.0,
    )
    priors = MD.PriorsV3(wander=wander, wind=True)
    batch = MD.with_blocks(_two_window_batch(par, k_cap=k_cap, n_mics=n_mics), wander.block_s)
    start = FT.seeds(batch, mode=MD.V3_MODE, priors=priors)
    rng = np.random.default_rng(11)
    latents = {}
    for w, nb in enumerate(batch.window_blocks):
        zero = MD.zero_latents(wander, n_rotors=1, k_max=k_cap, n_blocks=nb)
        latents[w] = MD.WindowLatents(
            **{n: _t(rng.normal(0.0, 2.0, x.shape)) for n, x in zero.tracks().items()}
        )
    whole = dataclasses.replace(batch, measured=start.measured, latents=latents)
    init = {k: v * 1.1 if k in ("sigma_nu", "gamma_hz") else v for k, v in start.init.items()}

    def model() -> Any:
        return MD.support_model(whole, mode=MD.V3_MODE, priors=priors)

    pyro.clear_param_store()
    guide = AutoDelta(model, init_loc_fn=init_to_value(values=init))
    loss = Trace_ELBO().differentiable_loss(model, guide)
    loss.backward()
    params = dict(guide.named_parameters())
    want: dict[str, torch.Tensor] = {}
    for k, p in params.items():
        assert p.grad is not None, k
        want[k] = p.grad.clone()
        p.grad = None

    chunked = dataclasses.replace(whole, chunk_frames=5)
    assert len(MD.frame_chunks(chunked)) > 2
    got = FT.chunked_objective(chunked, guide, mode=MD.V3_MODE, priors=priors, grad=True)
    np.testing.assert_allclose(float(got), float(loss), rtol=1e-12)
    for k, p in params.items():
        assert p.grad is not None, k
        np.testing.assert_allclose(
            p.grad.numpy(), want[k].numpy(), rtol=1e-8, atol=1e-10 * float(want[k].abs().max())
        )


def test_the_latent_steps_cached_forward_is_forward_v3_at_any_latents():
    """With the rig fixed, step (ii) caches every line's spectrum and the
    floor's atoms once and moves only their block multipliers: at arbitrary
    ``d, v, u, u_j`` on windows of DIFFERENT block counts it must reproduce
    :func:`forward_v3` (the per-block ``flight_model``) to rounding."""
    k_cap, n_mics = 3, 2
    par = _params(
        k_cap=k_cap,
        n_mics=n_mics,
        profile_db=np.array([-18.0, -21.0, -25.0]),
        shape_z=np.random.default_rng(4).standard_normal(NC),
    )
    par = dataclasses.replace(
        par, floor=dataclasses.replace(par.floor, shape_sd_db=_t(4.0)), wind_db=_t([-50.0, -55.0])
    )
    wander = MD.Wander(
        sigma_d_db=3.0,
        tau_d_s=2.0,
        sigma_v_db=2.0,
        tau_v_s=2.0,
        sigma_u_db=1.0,
        tau_u_s=2.0,
        block_s=0.05,
        sigma_uj_db=1.0,
        tau_uj_s=2.0,
    )
    full = _two_window_batch(par, k_cap=k_cap, n_mics=n_mics)
    assert full.frame_window is not None
    keep = np.flatnonzero(
        ~((full.frame_window == 1) & (np.arange(full.frame_window.size) % 12 >= 8))
    )
    batch = MD.with_blocks(MD.batch_slice(full, keep), wander.block_s)
    assert batch.window_blocks[0] > batch.window_blocks[1] >= 3
    rng = np.random.default_rng(7)
    latents = {}
    for w, nb in enumerate(batch.window_blocks):
        zero = MD.zero_latents(wander, n_rotors=1, k_max=k_cap, n_blocks=nb)
        latents[w] = MD.WindowLatents(
            **{n: _t(rng.normal(0.0, 3.0, x.shape)) for n, x in zero.tracks().items()}
        )
    cache = MD.latent_cache(batch, par)
    tracks = MD.stack_latents(latents, cache.windows, cache.n_block_max)
    with torch.no_grad():
        want = MD.forward_v3(batch, par, latents)
        got = cache.expected(*(tracks[f"wander_{n}"] for n in ("d", "v", "u", "uj")))
    np.testing.assert_allclose(got.numpy(), want.numpy(), rtol=1e-10, atol=0.0)


# ── (g) recovery of a planted v3 rig with wander ────────────────────────────


@pytest.mark.slow
def test_a_rendered_v3_rig_with_wander_is_recovered_with_its_block_tracks():
    """Render two short windows from a known v3 rig WITH wander, fit
    ``flight_v3`` with the true wander hyperparameters, and get back the
    profile (within 3 dB of the rig level the two clips realised) and the
    drawn block tracks of the lines (correlation > 0.7): the per-block latents
    make slow amplitude wander visible to the Whittle likelihood, and the
    renderer draws exactly the process the fit assumes. ~2.5 CPU-min: ``-m slow``."""
    from data_processing.noise_model import FIT_SCHEMA_V3
    from experiments.noise_model import render as RD
    from experiments.noise_model import supports as SU

    k_cap, n_mics, f0, dur = 6, 2, 150.0, 3.0
    wander = MD.Wander(
        sigma_d_db=3.0,
        tau_d_s=2.0,
        sigma_v_db=2.0,
        tau_v_s=2.0,
        sigma_u_db=1.5,
        tau_u_s=2.0,
        block_s=0.5,
    )
    prof = np.linspace(-20.0, -30.0, k_cap)[None, :]
    par = _params(k_cap=k_cap, n_mics=n_mics, profile_db=prof, floor_mean_db=-45.0)
    par = dataclasses.replace(par, floor=dataclasses.replace(par.floor, shape_sd_db=_t(3.0)))
    fit = dict(schema=FIT_SCHEMA_V3, params=MD.params_to_dict_v3(par, wander=wander))
    n = int(dur * SR)
    rps = np.full((1, n), f0)
    members, drawn = [], []
    for w in range(2):
        audio, diag = RD.render_noise(
            fit, rps, sr=SR, n_mics=n_mics, seed=100 + w, return_diagnostics=True
        )
        s = SU.synthetic_support(f"syn{w}", audio, rps, segment=(0.0, dur), meta={})
        members.append((s.name, s.power, s.carrier_rev_s_audio, s.frame_starts))
        drawn.append(diag["wander_tracks"])
    batch = MD.flight_batch(
        name="syn", members=members, sr=SR, n_fft=2048, hop=512, k_cap=k_cap, frame_stride=4
    )
    out = FT.fit_v3(
        batch,
        priors=MD.PriorsV3(wander=wander),
        optim=FT.OptimSpecV3(
            rig=FT.OptimSpec(
                adam_steps=60, adam_lr=0.05, adam_batch=None, lbfgs_iters=30, lbfgs_frames=None
            ),
            rounds=2,
            latent_lbfgs_iters=30,
        ),
    )
    assert out.window_latents is not None
    fitted, truth = [], []
    for w in range(2):
        lat = out.window_latents[w]
        assert lat.d is not None and lat.v is not None
        n_b = int(lat.d.shape[-1])
        fitted.append((lat.d[:, None, :] + lat.v).detach().numpy())
        truth.append(drawn[w]["d"][:, None, :n_b] + drawn[w]["v"][..., :n_b])
    # the profile is the rig's MEAN line level: two 3 s clips pin it only up to
    # the mean their own drawn wander happened to have, which is the target
    realised = np.concatenate(truth, axis=-1).mean(axis=-1)
    got = out.params.profile_db.detach().numpy()
    assert np.abs(got - (prof + realised)).max() < 3.0, (got - prof, realised)

    flat_fit = np.concatenate([x.reshape(-1) for x in fitted])
    flat_true = np.concatenate([x.reshape(-1) for x in truth])
    corr = float(np.corrcoef(flat_fit, flat_true)[0, 1])
    assert corr > 0.7, corr
    assert out.latents is not None and out.latents["summary"]["d"]["prior_sd_db"] == 3.0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA device")
def test_a_v3_fit_on_a_cuda_device_runs_end_to_end_and_writes_its_record(tmp_path):
    """The whole ``flight_v3`` fit on a CUDA device — minibatch Adam and L-BFGS on
    the frame-chunked rig objective, the latent step, the rig refit, the wind
    term — and its record: every number the fit reads back off a device tensor
    must come home first (the first kaggle smoke died in ``fit_support``'s
    diagnostics on ``np.asarray`` of the guide's cuda ``gamma_hz`` start)."""
    import json

    k_cap, n_mics = 4, 2
    par = _params(k_cap=k_cap, n_mics=n_mics, profile_db=np.linspace(-20.0, -30.0, k_cap))
    par = dataclasses.replace(par, floor=dataclasses.replace(par.floor, shape_sd_db=_t(3.0)))
    members = []
    n_frames = 12
    grid = SP.flight_grid(sr=SR, n_fft=512, hop=256)
    starts = np.arange(n_frames) * 256
    n = 512 + (n_frames - 1) * 256
    for w, f0 in enumerate((180.0, 195.0)):
        rps = (f0 + 5.0 * np.linspace(0.0, 1.0, n))[None, :]
        with torch.no_grad():
            m = SP.flight_model(
                grid, par, rate_work=SP.flight_rate_work(grid, rps, starts), k_max=k_cap
            ).numpy()
        obs = m * np.random.default_rng(w).standard_exponential(m.shape)
        members.append((f"w{w}", obs, rps, starts))
    batch = MD.flight_batch(
        name="cuda",
        members=members,
        sr=SR,
        n_fft=512,
        hop=256,
        k_cap=k_cap,
        device="cuda",
        chunk_frames=5,
    )
    wander = MD.Wander(
        sigma_d_db=2.0,
        tau_d_s=1.0,
        sigma_v_db=2.0,
        tau_v_s=1.0,
        sigma_u_db=1.0,
        tau_u_s=1.0,
        block_s=0.05,
    )
    priors = MD.PriorsV3(wind=True, wander=wander)
    out = FT.fit_v3(
        batch,
        priors=priors,
        optim=FT.OptimSpecV3(
            rig=FT.OptimSpec(adam_steps=3, adam_batch=4, lbfgs_iters=2, lbfgs_frames=None),
            rounds=1,
            latent_lbfgs_iters=2,
        ),
    )
    path = FT.write_fit(
        tmp_path / "cuda__flight_v3.json",
        support="cuda",
        kind="flight",
        mode=MD.V3_MODE,
        outcome=out,
        batch=batch,
        priors=priors,
    )
    record = json.loads(path.read_text())
    assert record["optimiser"]["device"].startswith("cuda")
    assert record["optimiser"]["rounds_run"] == 1
    assert np.isfinite(record["objective"]["total_nats"])
    assert np.all(np.isfinite(np.asarray(record["params"]["gamma_hz"], dtype=np.float64)))
    assert record["latents"]["fit"]["windows"] == 2


def test_sigma_v_by_order_sets_each_lines_ou_prior_and_the_render_draw():
    """``sigma_v_db_by_order`` (wander schema 2) gives every order its group's
    sd: ``sigma_db[i]`` for ``k_edges[i] <= k < k_edges[i + 1]``, the last group
    past the last edge; a group at 0 dB is a dead line (no prior term, drawn as
    zero). Without the block the scalar ``sigma_v_db`` holds for every line."""
    from data_processing.noise_model.v3 import ou_blocks

    rec = dict(
        sigma_d_db=1.0,
        tau_d_s=2.0,
        sigma_v_db=1.5,
        tau_v_s=2.0,
        sigma_u_db=1.0,
        tau_u_s=2.0,
        block_s=0.5,
        sigma_v_db_by_order=dict(k_edges=[1, 3, 9, 25, 61, 82], sigma_db=[3.0, 2.0, 0.0, 1.0, 0.5]),
    )
    wander = MD.Wander.from_mapping(rec)
    k_max, n_rot, n_b = 90, 2, 5
    per = np.asarray(wander.track_sigma("v", k_max))
    k = np.arange(1, k_max + 1)
    want = np.select([k < 3, k < 9, k < 25, k < 61], [3.0, 2.0, 0.0, 1.0], default=0.5)
    np.testing.assert_array_equal(per, want)
    assert MD.Wander.from_mapping(MD.Wander.from_mapping(rec).as_params()) == wander
    scalar = MD.Wander.from_mapping({k: v for k, v in rec.items() if k != "sigma_v_db_by_order"})
    assert scalar.track_sigma("v", k_max) == 1.5

    prior = MD.ou_tracks(
        wander, "v", n_rotors=n_rot, k_max=k_max, n_blocks=torch.tensor([n_b]), b_max=n_b
    )
    assert prior.live is not None and not bool(prior.live[:, 8:24].any())
    x = _t(np.random.default_rng(5).normal(0.0, 2.0, (1, n_rot, k_max, n_b)))
    rho = wander.rho("v")
    lag = np.abs(np.arange(n_b)[:, None] - np.arange(n_b)[None, :])
    expl = 0.0
    for r in range(n_rot):
        for kk in range(k_max):
            if per[kk] > 0.0:
                cov = torch.as_tensor(per[kk] ** 2 * rho**lag, dtype=torch.float64)
                mvn = torch.distributions.MultivariateNormal(
                    torch.zeros(n_b, dtype=torch.float64), cov
                )
                expl += float(mvn.log_prob(x[0, r, kk]))
    np.testing.assert_allclose(float(prior.log_prob(x)), expl, rtol=1e-12)

    # the renderer's draw: the same innovations scaled per line, zero on dead lines
    base = ou_blocks(np.random.default_rng(9), (n_rot, k_max), n_b, sigma=1.0, rho=rho)
    got = ou_blocks(np.random.default_rng(9), (n_rot, k_max), n_b, sigma=per, rho=rho)
    np.testing.assert_allclose(got, base * per[None, :, None], rtol=1e-12, atol=0.0)


def _wander_mean_fit(
    *, k_cap: int, n_rotors: int, n_mics: int, floor_mean_db: float
) -> dict[str, Any]:
    """A v3 payload with every wander family live: ``d``, ``v`` in two order
    groups of different sd, ``u`` and the kernel-mixed colour ``u_j``."""
    from data_processing.noise_model import FIT_SCHEMA_V3

    wander = MD.Wander.from_mapping(
        dict(
            sigma_d_db=1.0,
            tau_d_s=2.0,
            sigma_v_db=3.0,
            tau_v_s=2.0,
            sigma_u_db=1.5,
            tau_u_s=2.0,
            sigma_uj_db=1.0,
            tau_uj_s=1.0,
            block_s=0.5,
            uj_corr_oct=1.5,
            sigma_v_db_by_order=dict(k_edges=[1, 11, k_cap + 1], sigma_db=[2.0, 4.0]),
        )
    )
    prof = np.linspace(-20.0, -30.0, k_cap)[None, :]
    par = _params(k_cap=k_cap, n_mics=n_mics, profile_db=prof, floor_mean_db=floor_mean_db)
    par = dataclasses.replace(par, floor=dataclasses.replace(par.floor, shape_sd_db=_t(3.0)))
    p = MD.params_to_dict_v3(par, wander=wander)
    fit: dict[str, Any] = dict(schema=FIT_SCHEMA_V3, params=p)
    p["profile"]["profile_db"] = np.repeat(prof, n_rotors, axis=0).tolist()
    p["gamma_hz"] = np.repeat(np.asarray(p["gamma_hz"]), n_rotors, axis=0).tolist()
    return fit


def test_zero_wander_mean_is_the_default_and_power_moves_the_render():
    """``wander_mean="zero"`` is the default (byte-identical to not passing
    it); ``"power"`` moves the render; anything else is refused."""
    from experiments.noise_model import render as RD

    fit = _wander_mean_fit(k_cap=12, n_rotors=2, n_mics=2, floor_mean_db=-45.0)
    fit["params"]["profile"]["profile_db"] = fit["params"]["profile"]["profile_db"][:1]
    fit["params"]["gamma_hz"] = fit["params"]["gamma_hz"][:1]
    rps = np.stack([np.full(SR, 150.0), np.full(SR, 131.0)])
    kw: dict[str, Any] = dict(sr=SR, n_mics=2, seed=4, sr_work=32000)
    default = RD.render_noise(fit, rps, **kw)
    zero = RD.render_noise(fit, rps, wander_mean="zero", **kw)
    assert zero.tobytes() == default.tobytes()
    assert not np.array_equal(RD.render_noise(fit, rps, wander_mean="power", **kw), zero)
    with pytest.raises(ValueError, match="wander_mean"):
        RD.render_noise(fit, rps, wander_mean="mean", **kw)  # type: ignore[call-overload]


def test_power_wander_mean_renders_the_fitted_mean_line_power():
    """``wander_mean="power"`` shifts every dB-Gaussian wander track by
    ``-sigma^2 ln10 / 20`` (per order group for ``v``, ``diag(A A^T)`` for the
    kernel-mixed ``u_j``) on the SAME draws, so a long render's mean line power
    is the expected periodogram's (the model at zero latents), where the
    zero-mean draw sits above it by ``exp((s ln10 / 10)^2 / 2)``.

    Four rotors at 145-155 rev/s put orders 1-10 (``v`` sd 2 dB) below 1572 Hz
    and orders 11-50 (4 dB) above; the floor is off. 20 s is ~10 wander
    correlation times per line over 40 and 160 lines: the band means scatter
    by ~0.15 dB over seeds, and the linear interpolation between block centres
    leaves the 4 dB group ~0.14 dB low; 0.5 dB covers both."""
    from data_processing.noise_model import spectrum as NSP
    from experiments.noise_model import render as RD
    from experiments.noise_model import supports as SU

    k_cap, n_mics, dur, sr_work = 50, 1, 20.0, 32000
    f0 = np.array([145.0, 148.3, 151.7, 155.0])
    fit = _wander_mean_fit(k_cap=k_cap, n_rotors=f0.size, n_mics=n_mics, floor_mean_db=-300.0)
    rps = np.repeat(f0[:, None], int(dur * SR), axis=1)
    kw: dict[str, Any] = dict(sr=SR, n_mics=n_mics, seed=1, sr_work=sr_work)
    zero, dz = RD.render_noise(fit, rps, return_diagnostics=True, **kw)
    power, dp = RD.render_noise(fit, rps, wander_mean="power", return_diagnostics=True, **kw)

    # the same draws, each track shifted by its own stationary variance
    wander = MD.Wander.from_mapping(fit["params"]["wander"])
    tz, tp = dz["wander_tracks"], dp["wander_tracks"]
    c = math.log(10.0) / 20.0
    sd_v = np.where(np.arange(1, k_cap + 1) < 11, 2.0, 4.0)
    mix = wander.uj_mix(NSP.floor_ctrl_hz(SR))
    assert mix is not None
    np.testing.assert_allclose(tp["d"] - tz["d"], -c * 1.0**2, rtol=1e-12)
    np.testing.assert_allclose(
        tp["v"] - tz["v"], np.broadcast_to(-c * sd_v[None, :, None] ** 2, tz["v"].shape), rtol=1e-9
    )
    np.testing.assert_allclose(tp["u"] - tz["u"], -c * 1.5**2, rtol=1e-9)
    uj_var = 1.0**2 * (mix @ mix.T).diagonal()
    np.testing.assert_allclose(
        tp["uj"] - tz["uj"], np.broadcast_to(-c * uj_var[:, None], tz["uj"].shape), rtol=1e-9
    )

    # constant carriers: one frame of the expected periodogram is every frame's
    m = RD.expected_periodogram(
        fit,
        rps[:, : SU.OBS_N_FFT],
        n_fft=SU.OBS_N_FFT,
        hop=SU.OBS_HOP,
        sr=SR,
        n_mics=n_mics,
        sr_work=sr_work,
    )
    split = int(1572.0 / SR * SU.OBS_N_FFT)
    kk = math.log(10.0) / 10.0
    for lo, hi, s_v in ((1, split, 2.0), (split, m.shape[-1], 4.0)):
        want = float(m[..., lo:hi].sum(-1).mean())
        got = {}
        for mode, audio in (("zero", zero), ("power", power)):
            s = SU.synthetic_support(mode, audio, rps, segment=(0.0, dur), meta={})
            got[mode] = float(np.asarray(s.power)[..., lo:hi].sum(-1).mean())
        assert abs(10.0 * np.log10(got["power"] / want)) < 0.5, (lo, hi, got, want)
        # every line of the band carries the same d + v sd: the zero-mean draw
        # is the same audio scaled by the lognormal excess
        excess = math.exp(kk**2 * (1.0**2 + s_v**2) / 2.0)
        np.testing.assert_allclose(got["zero"] / got["power"], excess, rtol=1e-3)


# ── round 4: the array's per-channel response ────────────────────────────────


def test_array_response_curve_interpolates_in_log_f_and_holds_the_edges():
    """``array_response_db`` is linear in ``log f`` between the band centres,
    flat beyond them, and refuses a malformed block."""
    from data_processing.noise_model import params as MDP

    block = {"band_hz": [100.0, 1000.0, 10000.0], "gain_db": [[1.0, 3.0, -5.0], [0.0, 0.0, 0.0]]}
    f = np.array([10.0, 100.0, 316.2277660168379, 1000.0, 3162.277660168379, 10000.0, 20000.0])
    got = MDP.array_response_db(block, f)
    np.testing.assert_allclose(got[0], [1.0, 1.0, 2.0, 3.0, -1.0, -5.0, -5.0], rtol=1e-9)
    np.testing.assert_allclose(got[1], 0.0)
    with pytest.raises(ValueError, match="array_response"):
        MDP.array_response_db({"band_hz": [1000.0, 100.0], "gain_db": [[0.0, 0.0]]}, f)
    with pytest.raises(ValueError, match="array_response"):
        MDP.array_response_db({"band_hz": [100.0, 1000.0], "gain_db": [[0.0, 0.0, 0.0]]}, f)


def test_a_per_band_transfer_normalises_the_data_and_is_written_for_the_renderer(tmp_path):
    """``load_channel_gains(band="transfer")`` reads each mic's per-band
    deviation from the array mean off the rank record's level table; a batch
    normalised by it is divided by the curve on its own bins (the power-domain
    inverse of what the renderer and the expectation apply back), and the fit
    record carries the block as ``params.array_response`` with the SAME
    numbers, so a render of the record is mic ``m`` of the rig."""
    import json

    from data_processing.noise_model import params as MDP
    from experiments.noise_model import render as RD

    k_cap, n_mics = 3, 2
    par = _params(k_cap=k_cap, n_mics=n_mics, profile_db=-20.0)
    par = dataclasses.replace(par, floor=dataclasses.replace(par.floor, shape_sd_db=_t(3.0)))
    band_hz = [100.0, 1000.0, 5000.0]
    # three windows, three mics, three bands: mic 0 is +2 / 0 / -6 dB against
    # the array mean, mic 2 the mirror image, mic 1 (unused) fills the mean
    full = []
    for w in range(3):
        base = np.array([[-40.0, -50.0, -60.0]]) + w
        full.append(
            (base + np.array([[2.0], [0.0], [-2.0]]) * np.array([[1.0, 0.0, -3.0]])).tolist()
        )
    record = {
        "schema": "mic-gain-rank/1",
        "rigs": {
            "toy": {
                "band_centres_hz": band_hz,
                "levels_db": {"full": full},
                "wind": {"excess_db_above_500hz": [0.0, 0.0, 0.0]},
            }
        },
    }
    path = tmp_path / "mic_gains.json"
    path.write_text(json.dumps(record))
    gains = FT.load_channel_gains(path, rig="toy", mics=[0, 2], band="transfer")
    assert gains.transfer_db is not None and gains.band_hz is not None
    np.testing.assert_allclose(gains.transfer_db, [[2.0, 0.0, -6.0], [-2.0, 0.0, 6.0]], atol=1e-12)
    np.testing.assert_allclose(gains.gains_db, [-4.0 / 3.0, 4.0 / 3.0], atol=1e-12)

    raw = _flight_batch(par, k_cap=k_cap, n_mics=n_mics)
    starts = np.arange(16) * 256
    rps = np.full((1, 512 + 15 * 256), 180.0)
    normed = MD.flight_batch(
        name="v3",
        members=[("w0", raw.power.numpy(), rps, starts)],
        sr=SR,
        n_fft=512,
        hop=256,
        k_cap=k_cap,
        channel_gains=gains,
    )
    block = gains.array_response()
    assert block is not None
    curve = MDP.array_response_db(block, normed.grid.freqs_hz)  # (M, F)
    ratio = (raw.power / normed.power).numpy()  # (M, N, F)
    np.testing.assert_allclose(
        ratio, np.broadcast_to(10.0 ** (curve[:, None, :] / 10.0), ratio.shape), rtol=1e-12
    )

    wander = MD.Wander(
        sigma_d_db=0.0,
        tau_d_s=1.0,
        sigma_v_db=0.0,
        tau_v_s=1.0,
        sigma_u_db=0.0,
        tau_u_s=1.0,
        block_s=0.05,
    )
    priors = MD.PriorsV3(wander=wander)
    out = FT.fit_v3(
        normed,
        priors=priors,
        optim=FT.OptimSpecV3(
            rig=FT.OptimSpec(adam_steps=2, adam_batch=4, lbfgs_iters=1, lbfgs_frames=None),
            rounds=0,
        ),
    )
    rec_path = FT.write_fit(
        tmp_path / "toy__flight_v3.json",
        support="toy",
        kind="flight",
        mode=MD.V3_MODE,
        outcome=out,
        batch=normed,
        priors=priors,
    )
    fit = json.loads(rec_path.read_text())
    assert fit["params"]["array_response"]["gain_db"] == block["gain_db"]
    assert fit["params"]["array_response"]["band_hz"] == block["band_hz"]
    assert fit["diagnostics"]["batch"]["channel_gains"]["transfer_db"] == block["gain_db"]

    # the expectation of mic m is the array-mean model times the response
    plain = json.loads(json.dumps(fit))
    del plain["params"]["array_response"]
    r = np.full((1, 4096), 180.0)
    with_resp = RD.expected_periodogram(fit, r, n_fft=512, hop=256, sr=SR, n_mics=n_mics)
    without = RD.expected_periodogram(plain, r, n_fft=512, hop=256, sr=SR, n_mics=n_mics)
    want = MDP.array_response_db(block, np.fft.rfftfreq(512, 1.0 / SR))
    np.testing.assert_allclose(
        10.0 * np.log10(with_resp / without), np.broadcast_to(want[:, None, :], with_resp.shape)
    )


def test_a_render_carries_each_channels_response_in_its_long_run_spectrum():
    """Rendered mic 0 over mic 1 follows ``params.array_response`` band for band:
    a 20 s floor-only render, its per-mic band levels against the response's
    difference on the same bands, within the periodogram's own scatter."""
    from experiments.noise_model import render as RD
    from experiments.noise_model import supports as SU

    k_cap, n_mics, dur = 12, 2, 20.0
    fit = _wander_mean_fit(k_cap=k_cap, n_rotors=1, n_mics=n_mics, floor_mean_db=-45.0)
    p = fit["params"]
    p["profile"]["profile_db"] = (np.asarray(p["profile"]["profile_db"]) - 300.0).tolist()
    for key in ("sigma_d_db", "sigma_v_db", "sigma_u_db", "sigma_uj_db"):
        p["wander"][key] = 0.0
    p["wander"]["sigma_v_db_by_order"]["sigma_db"] = [0.0, 0.0]
    band_hz = [125.0, 500.0, 2000.0, 6000.0]
    p["array_response"] = {
        "band_hz": band_hz,
        "gain_db": [[3.0, 1.0, -2.0, -5.0], [-3.0, -1.0, 2.0, 5.0]],
    }
    rps = np.full((1, int(dur * SR)), 150.0)
    audio = RD.render_noise(fit, rps, sr=SR, n_mics=n_mics, seed=3, sr_work=32000)
    s = SU.synthetic_support("resp", audio, rps, segment=(0.0, dur), meta={})
    power = np.asarray(s.power).mean(axis=1)  # (M, F)
    f = np.fft.rfftfreq(SU.OBS_N_FFT, 1.0 / SR)
    for lo, hi, want in (
        (110.0, 140.0, 6.0),
        (450.0, 550.0, 2.0),
        (1800.0, 2200.0, -4.0),
        (5500.0, 6500.0, -10.0),
    ):
        sel = (f >= lo) & (f < hi)
        got = 10.0 * np.log10(power[0, sel].mean() / power[1, sel].mean())
        assert abs(got - want) < 0.4, (lo, hi, got, want)
