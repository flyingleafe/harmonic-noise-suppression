"""The revised rotor-phase candidate against answers that are known.

Each test defends one behavioural risk of
:mod:`experiments.stochastic_fit.revised_phase` and its frozen kernel: the
finite-window expected periodogram (where a factor of 2 or an off-by-one bin
hides), its backend dispatch, the exact integrated-OU algebra and its
small-step limits, the conditional/marginal distinction that must not
double-count the shaft, the composite exposure of duplicated windows and the
unbiasedness of its minibatch estimator, the moment gate's per-(mic, frame)
resolution and its run-local lag pairs, the coloured floor's exact windowed
expectation, the renderer's rotor-speed labels and band content, the
sign/scale of the fitted objective, the fit cohort's refusals (inferred rotor
tracks, non-cohort manifests, duplicated training supports), and the declared
observation law — the oversampled work grid, the head-slice bin convention,
the known render/decimation transfer applied exactly once, and the band-edge
continuity that the removed whole-frame order mask destroyed.
"""

from __future__ import annotations

import math
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.stochastic_fit import revised_phase as RP
from experiments.stochastic_fit.data import Clip, periodogram
from experiments.stochastic_fit.model import (
    AMP_RPS_REF,
    FLOOR_SHAPE_F_MIN,
    FLOOR_SHAPE_N_CTRL,
    FLOOR_TILT_REF_HZ,
)
from experiments.stochastic_fit.phase_kernel import expected_periodogram_from_atoms, hann_window
from experiments.stochastic_fit.stage2 import antialias, render_transfer_power

SR = 16000


def _window(n: int) -> tuple[np.ndarray, float]:
    w = hann_window(n)
    return w, float(np.sum(w**2))


def _mean_periodogram(signals: np.ndarray, w: np.ndarray, wss: float) -> np.ndarray:
    spec = np.fft.rfft(signals * w, axis=-1)
    return ((spec.real**2 + spec.imag**2) / wss).mean(axis=0)


def _chirp_atom(n: int, w: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A moving carrier with a time-varying amplitude — the real ramp case."""
    t = np.arange(n, dtype=np.float64)
    phase = 2.0 * np.pi * (7.3 / n * t + 0.5 * 2.5 / n**2 * t**2)
    amp = 1.0 + 0.3 * np.sin(2.0 * np.pi * t / n)
    return w * amp * np.exp(1j * phase), phase, amp


# ── 1-3: the frozen finite-window kernel ────────────────────────────────────


def test_kernel_equals_the_explicit_double_sum():
    """The FFT route must equal ``sum_{t,s} a_t conj(a_s) R(t-s) e^{-2 pi i f (t-s)}``
    folded to the real-signal convention, to machine precision — and the torch
    backend must agree and carry a gradient through ``R``."""
    n = 64
    w, wss = _window(n)
    atom, _, _ = _chirp_atom(n, w)
    lags = np.arange(n)
    r_tau = np.exp(-0.031 * lags)
    lag = lags[:, None] - lags[None, :]
    gram = atom[:, None] * np.conj(atom[None, :]) * r_tau[np.abs(lag)]
    brute = np.array(
        [
            0.25
            * (
                np.sum(gram * np.exp(-2j * np.pi * f * lag)).real
                + np.sum(gram * np.exp(2j * np.pi * f * lag)).real
            )
            / wss
            for f in np.arange(n // 2 + 1) / n
        ]
    )
    fast = expected_periodogram_from_atoms(atom, r_tau, n_fft=n, window_sumsq=wss)
    assert fast.shape == (n // 2 + 1,)
    assert np.allclose(fast, brute, rtol=1e-11, atol=1e-12 * brute.max())
    # real and non-negative by construction (Schur product of two PD kernels)
    assert fast.min() > -1e-12 * fast.max()

    d = torch.tensor(0.031, dtype=torch.float64, requires_grad=True)
    r_t = torch.exp(-d * torch.arange(n, dtype=torch.float64))
    out = expected_periodogram_from_atoms(torch.as_tensor(atom), r_t, n_fft=n, window_sumsq=wss)
    assert np.allclose(out.detach().numpy(), brute, rtol=1e-10, atol=1e-12 * brute.max())
    out.sum().backward()
    assert d.grad is not None and torch.isfinite(d.grad) and float(d.grad) != 0.0


@pytest.mark.parametrize("moving", [False, True])
def test_planted_tone_and_chirp_match_the_monte_carlo_mean(moving: bool):
    """A fractional-bin tone and a planted chirp, with a diffusing residual
    phase: the kernel must match the mean periodogram over random initial
    phases in ``data.periodogram`` units, INCLUDING the amplitude constant
    ``A = sqrt(2 P)``."""
    n, draws = 64, 8000
    w, wss = _window(n)
    t = np.arange(n, dtype=np.float64)
    power = 0.7
    amp_peak = math.sqrt(2.0 * power)
    if moving:
        _, phase, shape = _chirp_atom(n, w)
        amp = amp_peak * shape
    else:
        phase = 2.0 * np.pi * 7.3 / n * t
        amp = np.full(n, amp_peak)
    diffusion = 0.031  # rad^2 per sample
    rng = np.random.default_rng(1)
    steps = math.sqrt(2.0 * diffusion) * rng.standard_normal((draws, n - 1))
    eps = np.concatenate((np.zeros((draws, 1)), np.cumsum(steps, axis=1)), axis=1)
    phi0 = rng.uniform(0.0, 2.0 * np.pi, size=(draws, 1))
    mc = _mean_periodogram(amp * np.cos(phase + eps + phi0), w, wss)
    pred = expected_periodogram_from_atoms(
        w * amp * np.exp(1j * phase),
        np.exp(-diffusion * t),
        n_fft=n,
        window_sumsq=wss,
    )
    assert np.max(np.abs(mc - pred)) < 0.04 * pred.max()
    # the amplitude convention itself: A = sqrt(2 P) is a mean square of P
    assert float(np.mean((amp_peak * np.cos(phase)) ** 2)) == pytest.approx(power, rel=0.05)


def test_unit_autocorrelation_reduces_to_the_deterministic_periodogram():
    """``R == 1`` (``D = 0``, no shaft term) must give exactly the folded
    finite-window periodogram of the deterministic atom."""
    n = 64
    w, wss = _window(n)
    atom, _, _ = _chirp_atom(n, w)
    pred = expected_periodogram_from_atoms(atom, np.ones(n), n_fft=n, window_sumsq=wss)
    spec = np.fft.fft(atom, n=2 * n)
    folded = np.array(
        [
            0.25 * (abs(spec[2 * j]) ** 2 + abs(spec[(2 * n - 2 * j) % (2 * n)]) ** 2) / wss
            for j in range(n // 2 + 1)
        ]
    )
    assert np.allclose(pred, folded, rtol=1e-12, atol=1e-12 * folded.max())


# ── 4-5: the exact integrated-OU algebra ────────────────────────────────────


def test_integrated_ou_transition_matches_the_sde_and_its_limits():
    """``F`` and ``Q`` against Euler-Maruyama simulation of the SDE, and against
    the ``lam dt -> 0`` limits, including ``lam dt ~ 1e-9`` where the naive
    difference of two nearly equal numbers loses every digit."""
    lam, sigma, dt = 6.0, 6.0, 0.002
    f_mat, q_mat = RP.ou_transition(lam, sigma, dt)
    e = math.exp(-lam * dt)
    assert f_mat[0, 0] == 1.0 and f_mat[1, 0] == 0.0
    assert f_mat[0, 1] == pytest.approx((1.0 - e) / lam, rel=1e-12)
    assert f_mat[1, 1] == pytest.approx(e, rel=1e-12)

    rng = np.random.default_rng(0)
    draws, sub = 8000, 800
    h = dt / sub
    nu = np.zeros(draws)
    theta = np.zeros(draws)
    noise = math.sqrt(2.0 * lam * sigma**2 * h)
    for _ in range(sub):
        theta = theta + nu * h
        nu = nu - lam * nu * h + noise * rng.standard_normal(draws)
    emp = np.cov(np.stack([theta, nu]))
    assert emp[1, 1] == pytest.approx(q_mat[1, 1], rel=0.07)
    assert emp[0, 0] == pytest.approx(q_mat[0, 0], rel=0.10)
    assert emp[0, 1] == pytest.approx(q_mat[0, 1], rel=0.10)

    for x, rel in ((1e-9, 1e-7), (1e-6, 1e-4)):
        dt_small = x / lam
        _, q_small = RP.ou_transition(lam, sigma, dt_small)
        assert np.all(np.isfinite(q_small))
        assert q_small[1, 1] == pytest.approx(2.0 * sigma**2 * lam * dt_small, rel=rel)
        assert q_small[0, 1] == pytest.approx(sigma**2 * lam * dt_small**2, rel=rel)
        assert q_small[0, 0] == pytest.approx(2.0 / 3.0 * sigma**2 * lam * dt_small**3, rel=rel)
        a, _, c = RP.ou_cholesky(lam, sigma, dt_small)
        assert a > 0.0 and c > 0.0


@pytest.mark.parametrize(
    "lam,sigma,tau",
    [(6.0, 6.0, 0.05), (6.0, 6.0, 0.5), (6.0, 6.0, 1e-4)],
)
def test_marginal_increment_variance_and_characteristic_function(
    lam: float, sigma: float, tau: float
):
    """``Var[k dtheta + deps]`` and ``R_k = E cos(k dtheta + deps)`` against
    Monte Carlo, including the small-``lam tau`` branch of the bracket."""
    draws, steps, k, d = 20000, 64, 2, 0.5
    dt = tau / steps
    rng = np.random.default_rng(3)
    innov = rng.standard_normal((draws, steps + 1, 2))
    theta, _ = RP.simulate_state(innov, lam=lam, sigma=sigma, dt=dt)
    d_theta = theta[:, steps] - theta[:, 0]
    closed = float(RP.integrated_ou_increment_var(np.asarray(tau), lam=lam, sigma=sigma))
    assert float(np.var(d_theta)) == pytest.approx(closed, rel=0.06)

    d_eps = math.sqrt(2.0 * d * tau) * rng.standard_normal(draws)
    total = k * d_theta + d_eps
    assert float(np.var(total)) == pytest.approx(k**2 * closed + 2.0 * d * tau, rel=0.06)
    r_closed = float(RP.prior_r_tau(np.asarray(tau), k, lam=lam, sigma=sigma, d=d))
    assert float(np.mean(np.cos(total))) == pytest.approx(r_closed, abs=0.02)


# ── 6: conditional vs marginal ──────────────────────────────────────────────


def test_conditioning_on_a_known_path_removes_the_ou_broadening():
    """Conditioned on a known ``theta`` path the residual kernel is
    ``exp(-D|tau|)`` and the line stays narrow; the marginal prediction still
    carries the OU broadening. Both must match their own Monte-Carlo mean, and
    phase modulation must conserve the band power."""
    n, draws = 4096, 1000
    w, wss = _window(n)
    lam, sigma, d, k = 6.0, 6.0, 0.5, 20
    t = np.arange(n, dtype=np.float64) / SR
    tau = np.arange(n, dtype=np.float64) / SR
    carrier = 2.0 * np.pi * 400.0 * t
    amp = math.sqrt(2.0 * 0.5)
    rng = np.random.default_rng(7)

    theta_known, _ = RP.simulate_state(
        rng.standard_normal((1, n, 2)), lam=lam, sigma=sigma, dt=1.0 / SR
    )
    theta_known = theta_known[0]
    eps = np.concatenate(
        (
            np.zeros((draws, 1)),
            np.cumsum(math.sqrt(2.0 * d / SR) * rng.standard_normal((draws, n - 1)), axis=1),
        ),
        axis=1,
    )
    phi0 = rng.uniform(0.0, 2.0 * np.pi, size=(draws, 1))

    pred_cond = expected_periodogram_from_atoms(
        w * amp * np.exp(1j * k * (carrier + theta_known)),
        RP.conditional_r_tau(tau, d=d),
        n_fft=n,
        window_sumsq=wss,
    )
    mc_cond = _mean_periodogram(
        amp * np.cos(k * (carrier + theta_known)[None, :] + eps + phi0), w, wss
    )
    assert np.max(np.abs(mc_cond - pred_cond)) < 0.08 * pred_cond.max()

    theta_fresh, _ = RP.simulate_state(
        rng.standard_normal((draws, n, 2)), lam=lam, sigma=sigma, dt=1.0 / SR
    )
    pred_marg = expected_periodogram_from_atoms(
        w * amp * np.exp(1j * k * carrier),
        RP.prior_r_tau(tau, k, lam=lam, sigma=sigma, d=d),
        n_fft=n,
        window_sumsq=wss,
    )
    mc_marg = _mean_periodogram(
        amp * np.cos(k * (carrier[None, :] + theta_fresh) + eps + phi0), w, wss
    )
    assert np.max(np.abs(mc_marg - pred_marg)) < 0.08 * pred_cond.max()

    assert pred_marg.max() < 0.7 * pred_cond.max()  # the OU broadening is real
    assert pred_marg.sum() == pytest.approx(pred_cond.sum(), rel=0.02)  # power conserved


# ── 7: overlap exposure ─────────────────────────────────────────────────────


def test_duplicate_windows_split_their_exposure():
    """A duplicated identical window must create no extra exposure and no extra
    information: the composite risk of the duplicated frame set equals the risk
    of the deduplicated set."""
    keys = [("a", 0), ("a", 128), ("a", 128), ("b", 0)]
    weights = RP.composite_weights(keys, hop=128, n_fft=256)
    assert weights.tolist() == [0.5, 0.25, 0.25, 0.5]

    rng = np.random.default_rng(11)
    power = rng.exponential(size=(2, 4, 5))
    model = rng.uniform(0.5, 2.0, size=(2, 4, 5))
    power[:, 2] = power[:, 1]  # the duplicated window is the SAME observation
    model[:, 2] = model[:, 1]
    dup = RP.composite_risk(power, model, weights)

    uniq = [("a", 0), ("a", 128), ("b", 0)]
    keep = [0, 1, 3]
    uniq_risk = RP.composite_risk(
        power[:, keep], model[:, keep], RP.composite_weights(uniq, hop=128, n_fft=256)
    )
    assert dup == pytest.approx(uniq_risk, rel=1e-12)


# ── 8: the renderer's labels and band content ───────────────────────────────


def _tiny_export(
    *,
    d_scalar: float = 0.5,
    n_fft: int = 256,
    hop: int = 128,
    sample_rate_work: int = RP.SAMPLE_RATE_WORK,
    profile_db: tuple[float, ...] = (-45.0, -51.0, -57.0, -60.0),
    n_mics: int = 2,
    sigma: float = 6.0,
    amp_exp: float = 0.0,
    floor_exp: float = 0.0,
    floor_mean_db: float = -60.0,
    floor_tilt_db_oct: float = 0.0,
    bias_mean_hz: float = 0.5,
) -> dict:
    """A hand-written export of the schema :func:`RP.fit_revised` emits."""
    return dict(
        schema_version=RP.SCHEMA_VERSION,
        model_family=RP.MODEL_FAMILY,
        rig_id="bench",
        parameters=dict(
            lam=6.0,
            sigma=sigma,
            d_scalar=d_scalar,
            lambda_source="fixed_reference",
            delay_s={"0": 0.0},
            bias_hz={},
            bias_mean_hz=[bias_mean_hz],
            bias_prior_std_hz=0.5,
            profile_db=[list(profile_db)],
            amp_exp=amp_exp,
            floor_mean_db=floor_mean_db,
            floor_shape_db=[0.0] * FLOOR_SHAPE_N_CTRL,
            floor_ctrl_hz=np.geomspace(30.0, 8000.0, FLOOR_SHAPE_N_CTRL).tolist(),
            floor_tilt_db_oct=floor_tilt_db_oct,
            floor_exp=floor_exp,
            floor_static_rel=0.0,
            mic_gain_db=[[0.0]] * n_mics,
            gain_all_db=[0.0] * n_mics,
            mic_floor_db=[0.0] * n_mics,
            k_cap=len(profile_db),
            n_mics=n_mics,
            n_rotors=1,
        ),
        training_provenance=dict(
            manifest_path=None,
            manifest_sha256=None,
            clips=[],
            front_end=dict(
                n_fft=n_fft,
                hop=hop,
                sr=SR,
                band_hz=[30.0, 7900.0],
                sample_rate_work=sample_rate_work,
            ),
            model_grid_hz=sample_rate_work,
            analysis_grid_hz=SR,
            state_rate_hz=1000.0,
            optimizer=dict(
                stage1_marginal=dict(iters=3, lr=0.05),
                stage2_carrier=dict(iters=3, lr=0.05),
                frame_chunk=2,
                frames_per_step=None,
                atom_dtype="float64",
            ),
            seed=0,
            composite_temperature=1.0,
            priors=dict(
                bias_std_hz=0.5,
                log_d_mean=0.0,
                log_d_std=2.0,
                log_sigma_mean=0.0,
                log_sigma_std=2.0,
            ),
        ),
        diagnostics=dict(shared_phase_evidence="not_identified_by_marginal_score", map_state={}),
    )


def _rps_clip(rps: np.ndarray, *, n_mics: int = 1) -> Clip:
    """A clip whose audio is never read: ``predict_spectrum`` uses its LENGTH
    and its raw telemetry, never its samples."""
    rps = np.atleast_2d(np.asarray(rps, dtype=np.float64))
    audio = np.zeros((n_mics, rps.shape[1]), dtype=np.float32)
    return Clip("grid", "synthetic", audio, rps, SR, rps.copy(), {})


def test_renderer_labels_exclude_eps_and_share_the_audio_timeline():
    """``physical_rps`` is the interval average of the EXACT shaft phase: it
    excludes ``eps`` (so it is unchanged when only the diffusion changes), it
    carries the audio's own length and timeline after resampling, and the
    rendered floor lands at the level the export names — the periodogram's
    ``sr/2`` normalization makes that a 6.0 dB trap on the 64 -> 16 kHz work
    grid."""
    rate = 2600.0
    rps = np.full((1, SR), rate)
    render = RP.render_revised(_tiny_export(), rps, n_mics=2, seed=5)

    assert render.audio.shape[0] == 2
    assert render.physical_rps.shape == (1, render.audio.shape[1])
    assert render.reference_rps.shape == (1, render.audio.shape[1])
    assert render.sample_rate == SR
    assert np.allclose(render.reference_rps, rate)
    # telemetry + the learned bias law, plus a zero-mean OU excursion
    assert float(render.physical_rps.mean()) == pytest.approx(rate + 0.5, abs=2.5)

    # eps never enters the physical shaft speed: only D changes, the path does not
    other = RP.render_revised(_tiny_export(d_scalar=50.0), rps, n_mics=2, seed=5)
    assert np.array_equal(render.physical_rps, other.physical_rps)
    assert not np.array_equal(render.audio, other.audio)

    # band content only (the AA filter's internals are the sibling's to change):
    # the order at 4 x 2600.5 Hz must not fold back to 16000 - 10402 Hz, and the
    # in-band line at 3 x 2600.5 Hz must survive
    pg = periodogram(Clip("r", "synthetic", render.audio, render.physical_rps, SR), 4096, 2048)
    level = pg.power.mean(axis=0).mean(axis=0)  # (F,) mic- and frame-mean
    median = float(np.median(level[(pg.freqs >= 30.0) & (pg.freqs <= 7900.0)]))

    def near(hz: float) -> float:
        bin_i = int(round(hz / pg.df))
        return float(level[bin_i - 3 : bin_i + 4].max())

    assert near(3.0 * (rate + 0.5)) > 100.0 * median  # passband line preserved
    assert near(SR - 4.0 * (rate + 0.5)) < 10.0 * median  # no alias of the 10.4 kHz line
    # the floor level itself: exponential cells have median ln2 x mean
    assert median == pytest.approx(math.log(2.0) * 1e-6, rel=0.5)


# ── 9-10: the objective's sign, scale and gradients ─────────────────────────


def _planted_clip(
    *,
    theta: np.ndarray,
    n_frames: int = 6,
    n_fft: int = 256,
    hop: int = 128,
    rate_hz: float = 1000.0,
    profile_db: tuple[float, ...] = (-6.0, -12.0),
    mic_scale: tuple[float, ...] = (1.0, 1.3),
    floor_std: float = 1e-3,
    d_per_sample: float = 0.0,
    seed: int = 0,
) -> Clip:
    """A planted clip: a constant-rate carrier times a known ``theta`` path."""
    rng = np.random.default_rng(seed)
    n = n_fft + (n_frames - 1) * hop
    assert theta.size == n
    t = np.arange(n, dtype=np.float64) / SR
    phase = 2.0 * np.pi * rate_hz * t + theta
    audio = rng.standard_normal((len(mic_scale), n)) * floor_std
    for k, level in enumerate(profile_db, start=1):
        amp = math.sqrt(2.0 * 10.0 ** (level / 10.0))
        eps = (
            np.cumsum(math.sqrt(2.0 * d_per_sample) * rng.standard_normal(n))
            if d_per_sample
            else 0.0
        )
        for m, scale in enumerate(mic_scale):
            audio[m] += scale * amp * np.cos(k * phase + eps + rng.uniform(0.0, 2.0 * np.pi))
    rps = np.full((1, n), rate_hz)
    return Clip("planted", "synthetic", audio.astype(np.float32), rps, SR, rps.copy(), {})


def _tiny_fit_config(**kw) -> RP.FitConfig:
    base = dict(
        n_fft=256,
        hop=128,
        sr=SR,
        band_hz=(30.0, 7900.0),
        state_rate_hz=1000.0,
        k_cap=2,
        delay_s=(0.0,),
        iters=200,
        lr=0.05,
        seed=0,
        frame_chunk=2,
        atom_dtype="float64",
    )
    base.update(kw)
    return RP.FitConfig(**base)  # type: ignore[arg-type]

def _line_clip(
    clip_id: str,
    *,
    rps: np.ndarray,
    profile_db: float,
    n_fft: int,
    hop: int,
    floor_std: float,
    seed: int,
) -> Clip:
    """One observed harmonic whose carrier follows ``rps`` exactly."""
    rng = np.random.default_rng(seed)
    n = int(rps.size)
    phase = 2.0 * np.pi * np.cumsum(rps.astype(np.float64)) / SR
    audio = rng.standard_normal((1, n)) * floor_std
    audio[0] += math.sqrt(2.0 * 10.0 ** (profile_db / 10.0)) * np.cos(phase + 0.17)
    return Clip(clip_id, "synthetic", audio.astype(np.float32), rps[None], SR, rps[None].copy(), {})


def test_initialization_keeps_quiet_and_loud_clip_floors_separate():
    """A quiet standby clip must not be initialized against a pooled cruise floor."""
    n_fft, hop, n_frames = 512, 256, 5
    n = n_fft + (n_frames - 1) * hop
    quiet = _line_clip(
        "quiet",
        rps=np.full(n, 80.0),
        profile_db=-48.0,
        n_fft=n_fft,
        hop=hop,
        floor_std=1e-7,
        seed=21,
    )
    loud = _line_clip(
        "loud",
        rps=np.full(n, 80.0),
        profile_db=-18.0,
        n_fft=n_fft,
        hop=hop,
        floor_std=1e-3,
        seed=22,
    )
    model = RP._RevisedModel(  # type: ignore[attr-defined]
        [(quiet.clip_id, quiet), (loud.clip_id, loud)],
        dynamics=RP.ShaftDynamics(lam=6.0, sigma=1.0, d_init=1.0, identified=False, diagnostics={}),
        config=_tiny_fit_config(n_fft=n_fft, hop=hop, k_cap=2, iters=1, training_recipe=RP.STAGE1_LADDER_RECIPE),
        observe=True,
    )

    clips = model.initialization_diagnostics["floor"]["clips"]
    q = clips["quiet"]["floor_db_p20_plus_6p5_band_median"]
    l = clips["loud"]["floor_db_p20_plus_6p5_band_median"]
    assert l - q > 50.0
    assert model.initialization_diagnostics["harmonic_seed"]["line_floor_db"][0][0] > 20.0


def test_initialization_integrates_fractional_moving_harmonic_energy_in_physical_units():
    """Fractional and moving line energy is integrated, not treated as floor."""
    n_fft, hop, n_frames = 512, 256, 5
    n = n_fft + (n_frames - 1) * hop
    t = np.arange(n, dtype=np.float64) / SR
    rps = 125.0 + 20.0 * np.sin(2.0 * np.pi * t / t[-1])
    clip = _line_clip(
        "moving",
        rps=rps,
        profile_db=-24.0,
        n_fft=n_fft,
        hop=hop,
        floor_std=1e-5,
        seed=23,
    )
    model = RP._RevisedModel(  # type: ignore[attr-defined]
        [(clip.clip_id, clip)],
        dynamics=RP.ShaftDynamics(lam=6.0, sigma=1.0, d_init=1.0, identified=False, diagnostics={}),
        config=_tiny_fit_config(n_fft=n_fft, hop=hop, k_cap=3, iters=1, training_recipe=RP.STAGE1_LADDER_RECIPE),
        observe=True,
    )

    profile = float(model.profile_db.detach().cpu().numpy()[0, 0])
    expected = -24.0 - RP.AMP_EXP_INIT * 10.0 * math.log10(float(np.mean(rps)) / RP.AMP_RPS_REF)
    assert profile == pytest.approx(expected, abs=6.0)
    assert model.initialization_diagnostics["harmonic_seed"]["line_floor_db"][0][0] > 20.0
    assert "P_meansquare = 2" in model.initialization_diagnostics["harmonic_seed"]["units"]


def test_band_energy_ladder_exports_full_k_and_rung_trace():
    """The temporary active-order ladder must end with a full-K export."""
    n_fft, hop, n_frames = 64, 32, 3
    n = n_fft + (n_frames - 1) * hop
    clip = _line_clip(
        "ladder",
        rps=np.full(n, 80.0),
        profile_db=-24.0,
        n_fft=n_fft,
        hop=hop,
        floor_std=1e-4,
        seed=24,
    )
    cfg = _tiny_fit_config(
        n_fft=n_fft,
        hop=hop,
        k_cap=17,
        iters=1,
        lr=1e-3,
        carrier_iters=1,
        carrier_lr=1e-3,
        frame_chunk=1,
        harmonic_chunk=None,
        training_recipe=RP.STAGE1_LADDER_RECIPE,
    )
    export = RP.fit_revised(
        [(clip.clip_id, clip)],
        rig_id="bench",
        dynamics=RP.ShaftDynamics(lam=6.0, sigma=1.0, d_init=1.0, identified=False, diagnostics={}),
        config=cfg,
    )

    assert export["training_recipe"] == "band_energy_ladder"
    assert len(export["parameters"]["profile_db"][0]) == 17
    trace = export["diagnostics"]["marginal_fit"]["rung_trace"]
    assert [r["active_k"] for r in trace] == [16, 17]
    assert export["training_provenance"]["optimizer"]["rung_schedule"] == [
        {"active_k": 16, "iterations": 40},
        {"active_k": 17, "iterations": 40},
    ]
    assert trace[-1]["newly_active_seed"]["orders"] == [17, 17]


def test_the_map_state_recovers_a_planted_shaft_path():
    """``D = 0`` and a planted ``theta`` drawn from the model's own prior: the
    MAP path must recover it up to the gauge freedoms the model declares (the
    ``theta_0 = 0`` offset and the rate the per-clip bias absorbs)."""
    lam, sigma = 100.0, 150.0
    n = 256 + 5 * 128
    rng = np.random.default_rng(2)
    theta, _ = RP.simulate_state(rng.standard_normal((1, n, 2)), lam=lam, sigma=sigma, dt=1.0 / SR)
    theta = theta[0]
    clip = _planted_clip(theta=theta, seed=4)
    cfg = _tiny_fit_config()
    dynamics = RP.ShaftDynamics(lam=lam, sigma=sigma, d_init=1e-6, identified=True, diagnostics={})
    export = RP.fit_revised([(clip.clip_id, clip)], rig_id="bench", dynamics=dynamics, config=cfg)

    trace = export["diagnostics"]["loss_trace"]
    assert trace[-1] < trace[0]
    state = export["diagnostics"]["map_state"][clip.clip_id]
    time_s = np.asarray(state["time_s"])
    got = np.asarray(state["theta_rad"])[0]
    want = np.interp(time_s, np.arange(n) / SR, theta)

    def detrend(x: np.ndarray) -> np.ndarray:
        return x - np.polyval(np.polyfit(time_s, x, 1), time_s)

    got_d, want_d = detrend(got), detrend(want)
    rms = float(np.sqrt(np.mean(want_d**2)))
    assert rms > 0.5  # the planted path is a real excursion, not noise
    assert float(np.sqrt(np.mean((got_d - want_d) ** 2))) < 0.6 * rms


def test_the_objective_locates_the_planted_phase_diffusion():
    """``theta = 0`` and a planted independent diffusion: the composite risk of
    the correct model, profiled over ``D``, is minimized within a factor two of
    the planted value and rises steeply on both sides. This is the objective's
    sign and scale, through the public kernel and risk."""
    n_fft, hop, n_frames, k = 512, 256, 12, 4
    rate, power, floor, d_true = 1000.0, 0.25, 4e-6, 50.0
    n = n_fft + (n_frames - 1) * hop
    w, wss = _window(n_fft)
    rng = np.random.default_rng(13)
    t = np.arange(n, dtype=np.float64) / SR
    eps = np.concatenate(
        ([0.0], np.cumsum(math.sqrt(2.0 * d_true / SR) * rng.standard_normal(n - 1)))
    )
    phase = 2.0 * np.pi * k * rate * t
    audio = (
        math.sqrt(2.0 * power) * np.cos(phase + eps + 1.234)
        + math.sqrt(floor) * rng.standard_normal(n)
    )[None, :]
    rps = np.full((1, n), rate)
    clip = Clip("planted_d", "synthetic", audio.astype(np.float32), rps, SR, rps.copy(), {})
    pg = periodogram(clip, n_fft, hop)

    starts = np.arange(n_frames) * hop
    atoms = np.stack(
        [w * math.sqrt(2.0 * power) * np.exp(1j * phase[s : s + n_fft]) for s in starts]
    )
    tau = np.arange(n_fft, dtype=np.float64) / SR
    band = (pg.freqs >= 30.0) & (pg.freqs <= 7900.0)
    weights = RP.composite_weights([(clip.clip_id, int(s)) for s in starts], hop=hop, n_fft=n_fft)



    def risk(d: float) -> float:
        shapes = expected_periodogram_from_atoms(
            atoms, RP.conditional_r_tau(tau, d=d), n_fft=n_fft, window_sumsq=wss
        )
        return float(RP.composite_risk(pg.power, (shapes + floor)[None], weights, band=band))

    grid = d_true * np.array([0.1, 0.25, 0.5, 0.8, 1.0, 1.25, 2.0, 4.0, 10.0])
    risks = np.array([risk(float(d)) for d in grid])
    best = float(grid[int(np.argmin(risks))])
    assert 0.5 * d_true <= best <= 2.0 * d_true
    assert risk(d_true) < risk(0.1 * d_true) and risk(d_true) < risk(10.0 * d_true)

def test_the_marginal_stage_recovers_sigma_and_d_on_exact_expected_power():
    """With exact marginal expected periodograms and fixed nuisance truth, the
    marginal objective must pull sigma and D toward their planted values. This
    fails if sigma is detached or if the stage-1 kernel accidentally uses the
    conditional residual law."""

    n = 256 + 6 * 128
    rps = np.vstack([
        np.linspace(96.0, 104.0, n, dtype=np.float64),
    ])
    clip = Clip("marginal_exact", "synthetic", np.zeros((1, n), dtype=np.float32), rps, SR, rps.copy(), {})
    cfg = _tiny_fit_config(iters=1, lr=0.05, k_cap=3, frame_chunk=2)
    truth = RP._RevisedModel(  # type: ignore[attr-defined]
        [(clip.clip_id, clip)],
        dynamics=RP.ShaftDynamics(lam=6.0, sigma=1.8, d_init=0.35, identified=False, diagnostics={}),
        config=cfg,
        observe=False,
    )
    with torch.no_grad():
        truth.profile_db.fill_(-25.0)
        truth.floor_mean_db.fill_(-80.0)
        truth.bias_mean_hz.fill_(0.7)
        truth.bias_hz.fill_(0.7)
        expected = torch.cat(
            [
                truth.frame_model(0, chunk, state=None, kernel="prior")
                for chunk in [np.arange(truth.clips[0].starts.size)]
            ],
            dim=1,
        ).detach().cpu().to(torch.float32)

    fit = RP._RevisedModel(  # type: ignore[attr-defined]
        [(clip.clip_id, clip)],
        dynamics=RP.ShaftDynamics(lam=6.0, sigma=0.5, d_init=0.05, identified=False, diagnostics={}),
        config=cfg,
        observe=False,
    )
    with torch.no_grad():
        fit.profile_db.copy_(truth.profile_db)
        fit.floor_mean_db.copy_(truth.floor_mean_db)
        fit.bias_mean_hz.copy_(truth.bias_mean_hz)
        fit.bias_hz.copy_(truth.bias_hz)
        fit.clips[0].power = expected

    opt = torch.optim.Adam([fit.log_sigma, fit.log_d], lr=0.08)
    start = np.array([fit.sigma_value(), float(torch.exp(fit.log_d).item())])
    target = np.array([truth.sigma_value(), float(torch.exp(truth.log_d).item())])
    start_err = float(np.linalg.norm(np.log(start / target)))
    for _ in range(80):
        opt.zero_grad(set_to_none=True)
        loss = fit.marginal_chunk_risk(0, np.arange(fit.clips[0].starts.size)) / cfg.temperature
        loss = loss + fit.prior(state=False, globals=True, bias_population=True)
        loss.backward()
        opt.step()
    got = np.array([fit.sigma_value(), float(torch.exp(fit.log_d).item())])
    got_err = float(np.linalg.norm(np.log(got / target)))
    assert got_err < 0.5 * start_err
    assert 0.5 * target[0] <= got[0] <= 2.0 * target[0]
    assert 0.5 * target[1] <= got[1] <= 2.0 * target[1]


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
        ),
    ],
)
def test_every_fitted_parameter_receives_a_finite_nonzero_gradient(device: str):
    """A fit whose gradient is silently zero for one block cannot move it: every
    fitted tensor, the state innovations included, must carry a finite non-zero
    gradient."""
    n = 256 + 5 * 128
    rng = np.random.default_rng(5)
    theta, _ = RP.simulate_state(
        rng.standard_normal((1, n, 2)), lam=100.0, sigma=150.0, dt=1.0 / SR
    )
    clip = _planted_clip(theta=theta[0], seed=6)
    cfg = _tiny_fit_config(iters=3)
    dynamics = RP.ShaftDynamics(lam=100.0, sigma=150.0, d_init=0.5, identified=True, diagnostics={})
    export = RP.fit_revised(
        [(clip.clip_id, clip)],
        rig_id="bench",
        dynamics=dynamics,
        config=cfg,
        device=device,
    )
    diag = export["diagnostics"]
    assert diag["shared_phase_evidence"] == "not_identified_by_marginal_score"
    for stage in ("marginal_fit", "carrier_fit"):
        assert diag[stage]["valid"], stage
        assert diag[stage]["finite_gradients"], stage

    marginal = diag["grad_norms"]["marginal"]["last_step"]
    for name in ("log_sigma", "log_d", "profile_db", "bias_mean_hz"):
        value = marginal[name]
        assert math.isfinite(value), f"{name} marginal gradient is not finite"
        assert value > 0.0, f"{name} received a zero marginal gradient"

    carrier = diag["grad_norms"]["carrier"]["last_step"]
    for name, value in carrier.items():
        assert math.isfinite(value), f"{name} carrier gradient is not finite"
        assert value > 0.0, f"{name} received a zero carrier gradient"


# ── 11: the state grid is fine enough, and the check can fail ───────────────


def test_state_grid_convergence_at_a_high_order():
    """The Hermite state interpolation omits the intra-interval OU bridge, and
    the atom carries ``k theta``, so the error scales with ``k^2``. At ``k =
    100`` with ``sigma = 15`` rad/s and ``lam = 15`` /s the predicted spectrum must
    be converged between 1000 Hz and 500 Hz — and must DIVERGE at a deliberately
    coarse 62.5 Hz grid, or the check proves nothing.

    THIS is the real two-sided check: a genuine prediction from a REFINED grid
    against the reference path. The export's ``coarsening_sensitivity`` block is
    not this and must not be read as a convergence result."""
    n_fft, k, lam, sigma, d = 1024, 100, 15.0, 15.0, 0.1
    w, wss = _window(n_fft)
    n_ref = n_fft + 257
    rng = np.random.default_rng(17)
    theta, nu = RP.simulate_state(
        rng.standard_normal((1, n_ref, 2)), lam=lam, sigma=sigma, dt=1.0 / SR
    )
    theta, nu = theta[0], nu[0]
    t = np.arange(n_fft, dtype=np.float64) / SR
    carrier = 2.0 * np.pi * 60.0 * t  # k = 100 lands at 6 kHz, inside the band
    tau = np.arange(n_fft, dtype=np.float64) / SR
    r_tau = RP.conditional_r_tau(tau, d=d)

    def spectrum(theta_path: np.ndarray) -> np.ndarray:
        atom = w * np.exp(1j * k * (carrier + theta_path))
        return expected_periodogram_from_atoms(atom, r_tau, n_fft=n_fft, window_sumsq=wss)

    def on_grid(rate_hz: float) -> np.ndarray:
        stride = int(round(SR / rate_hz))
        idx = np.arange(0, n_ref, stride)
        return RP.hermite_theta(theta[idx], nu[idx], dt_state=1.0 / rate_hz, n_out=n_fft, sr=SR)

    ref = spectrum(theta[:n_fft])
    fine = spectrum(on_grid(1000.0))
    half = spectrum(on_grid(500.0))
    coarse = spectrum(on_grid(62.5))

    def rel_l1(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.abs(a - b).sum() / np.abs(b).sum())

    assert rel_l1(fine, ref) < 0.15
    assert rel_l1(half, fine) < 0.15  # converged between the grid and twice it
    assert rel_l1(coarse, fine) > 0.5  # and the check can fail


# ── 12: the declared work grid and the known transfer ───────────────────────


def _transfer(n_fft: int) -> np.ndarray:
    """The export's own observation transfer on the analysis grid."""
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SR)
    return np.asarray(
        render_transfer_power(freqs, sample_rate_work=RP.SAMPLE_RATE_WORK, sample_rate_out=SR),
        dtype=np.float64,
    )


def test_a_band_edge_crossing_is_continuous_through_the_frame_model():
    """The whole-frame order mask zeroed an atom the instant its FRAME-MEAN
    carrier passed 7900 Hz, which is a discontinuous prediction exactly at a
    band-edge crossing. Sweeping a first-order carrier across the edge must now
    give a smoothly varying in-band power, with no all-or-none step."""
    n_fft, hop, n_frames = 256, 128, 9
    n = n_fft + (n_frames - 1) * hop
    rate = np.linspace(7870.0, 7930.0, n)[None, :]  # k = 1 crosses 7900 Hz mid-clip
    export = _tiny_export(
        profile_db=(0.0,),
        n_mics=1,
        floor_mean_db=-300.0,
        sigma=1e-9,
        d_scalar=1e-12,
        bias_mean_hz=0.0,
    )
    pred = RP.predict_spectrum(export, _rps_clip(rate), n_fft=n_fft, hop=hop)[0]
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SR)
    band = (freqs >= 30.0) & (freqs <= 7900.0)
    power = pred[:, band].sum(axis=1)

    frame_mean = np.array([rate[0, s : s + n_fft].mean() for s in np.arange(n_frames) * hop])
    assert frame_mean.min() < 7900.0 < frame_mean.max()  # the crossing is inside the sweep
    assert power.min() > 0.0
    ratios = power[1:] / power[:-1]
    # the old mask took a frame from its full line power to the bare floor
    # (here 1e-30) in one step; the continuous law moves by tens of percent
    assert ratios.min() > 0.25, f"in-band power collapses across the edge: {ratios.tolist()}"
    assert ratios.max() < 4.0, f"in-band power jumps across the edge: {ratios.tolist()}"


def test_an_order_reaching_the_work_nyquist_raises_in_both_paths():
    """Out of the declared domain is an ERROR, in the model and in the
    renderer alike — never a silently skipped order."""
    export = _tiny_export()  # k_cap 4
    rps = np.full((1, SR), 9000.0)  # 4 x 9000.5 Hz is past the 32 kHz work Nyquist
    with pytest.raises(ValueError, match=r"rotor 0 order 4"):
        RP.predict_spectrum(export, _rps_clip(rps, n_mics=2), n_fft=256, hop=128)
    with pytest.raises(ValueError, match=r"rotor 0 order 4"):
        RP.render_revised(export, rps, n_mics=2, seed=0)


@pytest.mark.parametrize("bin_index", [13, 126])
def test_the_work_grid_reads_back_on_the_analysis_bins_at_the_analysis_level(bin_index: int):
    """A planted tone pins BOTH halves of the grid convention: the work
    spectrum is read as a HEAD slice (a stride-4 read would move the line) and
    the ``Fs / Fs_work`` factor brings it to analysis units. Dividing by the
    transfer must recover the bare Hann identity ``A^2 n_fft / 6`` exactly —
    which is also the structural guard against applying the transfer twice."""
    n_fft, hop = 256, 128
    df = SR / n_fft
    power_ms = 0.7
    export = _tiny_export(
        profile_db=(10.0 * math.log10(power_ms),),
        n_mics=1,
        floor_mean_db=-300.0,
        sigma=1e-9,
        d_scalar=1e-12,
        bias_mean_hz=0.0,
    )
    rate = np.full((1, n_fft + 3 * hop), bin_index * df)
    pred = RP.predict_spectrum(export, _rps_clip(rate), n_fft=n_fft, hop=hop)[0, 0]
    gain = _transfer(n_fft)

    assert bin_index % 4 != 0  # a stride-4 read of the work spectrum cannot land here
    assert int(np.argmax(pred)) == bin_index
    source = pred / gain
    assert source[bin_index] == pytest.approx(2.0 * power_ms * n_fft / 6.0, rel=1e-6)
    if bin_index == 126:
        # 7875 Hz sits in the decimator's rolloff, so a doubled transfer would
        # show up here as a factor ~2 in dB; if this ever fails the test has
        # lost its teeth, not the implementation its correctness
        assert gain[bin_index] < 0.95


def _quadrature_reference(rate_track: np.ndarray, *, n_fft: int, hop: int) -> np.ndarray:
    """``(N, F)`` EXACT phase-ensemble mean periodogram of a deterministic
    unit-power carrier through the real chain.

    Two quadrature initial phases (0 and pi/2) average to the exact ensemble
    mean over a uniform initial phase for a deterministic signal through ANY
    real linear filter — no Monte-Carlo uncertainty is involved. The chain is
    the renderer's own: synthesize on the work grid, ``stage2.antialias``, then
    the unchanged ``clips.decimate``.
    """
    from experiments.stochastic_fit import clips as C

    work = RP.SAMPLE_RATE_WORK
    q = work // SR
    n16 = rate_track.size
    t16 = np.arange(n16) / SR
    t_work = np.arange(n16 * q) / work
    f_work = np.interp(t_work, t16, rate_track)
    phase = 2.0 * np.pi * np.cumsum(f_work) / work
    amp = math.sqrt(2.0)  # profile 0 dB -> mean-square power 1
    rps_work = np.ones((1, n16 * q))
    out = []
    for phi0 in (0.0, 0.5 * np.pi):
        x = antialias((amp * np.cos(phase + phi0))[None, :], work)
        dec = C.decimate(
            Clip("ref", "synthetic", np.asarray(x, dtype=np.float32), rps_work, work), SR
        )
        audio = np.asarray(dec.audio, dtype=np.float32)
        clip16 = Clip("ref16", "synthetic", audio, np.ones((1, audio.shape[1])), SR)
        out.append(periodogram(clip16, n_fft, hop).power[0])
    return 0.5 * (out[0] + out[1])


def _deterministic_export(n_fft: int, hop: int) -> dict:
    return _tiny_export(
        profile_db=(0.0,),
        n_mics=1,
        floor_mean_db=-300.0,
        sigma=1e-9,
        d_scalar=1e-12,
        bias_mean_hz=0.0,
        n_fft=n_fft,
        hop=hop,
    )


def test_a_planted_chirp_matches_the_exact_quadrature_reference():
    """End to end against an EXACT reference (no Monte Carlo): the predicted
    spectrum of a ramping first-order carrier against the phase-ensemble mean
    periodogram of the same carrier pushed through the real render AA and the
    real decimator. The gate is a normalized spectral L1 of 2% on the scored
    interior window support."""
    n_fft, hop, n16 = 256, 128, 4096
    t16 = np.arange(n16) / SR
    rate = 900.0 + 300.0 * t16 / t16[-1]
    pred = RP.predict_spectrum(
        _deterministic_export(n_fft, hop), _rps_clip(rate[None, :]), n_fft=n_fft, hop=hop
    )[0]
    ref = _quadrature_reference(rate, n_fft=n_fft, hop=hop)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SR)
    band = (freqs >= 30.0) & (freqs <= 7900.0)
    inner = slice(4, min(pred.shape[0], ref.shape[0]) - 4)  # skip the filters' edge transients
    p, r = pred[inner][:, band], ref[inner][:, band]
    l1 = float(np.abs(p - r).sum() / r.sum())
    assert l1 <= 0.02, f"normalized spectral L1 {l1:.4f} against the exact chirp reference"


def test_a_stationary_tone_integrates_to_the_exact_reference_within_a_tenth_of_a_db():
    """The in-band integrated power of a stationary tone, against the same
    exact quadrature reference: the gate is 0.1 dB, worst frame reported."""
    n_fft, hop, n16 = 256, 128, 4096
    rate = np.full(n16, 1005.0)  # deliberately off-bin
    pred = RP.predict_spectrum(
        _deterministic_export(n_fft, hop), _rps_clip(rate[None, :]), n_fft=n_fft, hop=hop
    )[0]
    ref = _quadrature_reference(rate, n_fft=n_fft, hop=hop)

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SR)
    band = (freqs >= 30.0) & (freqs <= 7900.0)
    inner = slice(4, min(pred.shape[0], ref.shape[0]) - 4)
    p = pred[inner][:, band].sum(axis=1)
    r = ref[inner][:, band].sum(axis=1)
    worst = float(np.max(np.abs(10.0 * np.log10(p / r))))
    assert worst <= 0.1, f"worst in-band integrated power error {worst:.4f} dB"


# ── 13: the numerics review's six defects ───────────────────────────────────


def test_the_kernel_dispatches_on_whichever_argument_is_a_tensor():
    """numpy atoms with a torch ``r_tau``: the result must stay on the TENSOR's
    device and keep its gradient, instead of dragging the only real tensor onto
    the atoms' (CPU) device."""
    n = 64
    w, wss = _window(n)
    atom, _, _ = _chirp_atom(n, w)
    d = torch.tensor(0.031, dtype=torch.float64, requires_grad=True)
    r_t = torch.exp(-d * torch.arange(n, dtype=torch.float64))

    out = expected_periodogram_from_atoms(atom, r_t, n_fft=n, window_sumsq=wss)
    assert isinstance(out, torch.Tensor)
    assert out.device == r_t.device
    out.sum().backward()
    assert d.grad is not None and torch.isfinite(d.grad) and float(d.grad) != 0.0

    ref = expected_periodogram_from_atoms(atom, r_t.detach().numpy(), n_fft=n, window_sumsq=wss)
    assert np.allclose(out.detach().numpy(), ref, rtol=1e-10, atol=1e-12 * ref.max())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA")
def test_the_kernel_keeps_a_cuda_r_tau_on_its_own_device():
    n = 64
    w, wss = _window(n)
    atom, _, _ = _chirp_atom(n, w)
    r_t = torch.exp(-0.031 * torch.arange(n, dtype=torch.float64, device="cuda"))
    out = expected_periodogram_from_atoms(atom, r_t, n_fft=n, window_sumsq=wss)
    assert out.device.type == "cuda"


@pytest.mark.parametrize("k_dtype", [torch.float64, torch.int64])
def test_prior_r_tau_dispatches_on_a_tensor_order(k_dtype: torch.dtype):
    """A torch ``k`` with numpy lags is a torch call: dispatching on the lags
    alone dropped the order's device and its autograd."""
    tau = (np.arange(16, dtype=np.float64) / SR)[None, :]
    k_t = torch.arange(1, 5, dtype=k_dtype)[:, None]
    got = RP.prior_r_tau(tau, k_t, lam=6.0, sigma=6.0, d=0.5)
    assert isinstance(got, torch.Tensor)
    want = RP.prior_r_tau(tau, k_t.numpy().astype(np.float64), lam=6.0, sigma=6.0, d=0.5)
    assert np.allclose(got.detach().numpy(), want)

    d = torch.tensor(0.5, dtype=torch.float64, requires_grad=True)
    RP.prior_r_tau(tau, k_t, lam=6.0, sigma=6.0, d=d).sum().backward()
    assert d.grad is not None and float(d.grad) != 0.0


def test_simulate_state_returns_one_state_on_a_one_sample_grid():
    """``S == 1``: there is no increment at all, and ``theta`` must still come
    back on the same grid as ``nu`` (length 1, the gauge value)."""
    innov = np.full((2, 1, 2), 0.5)
    theta, nu = RP.simulate_state(innov, lam=6.0, sigma=6.0, dt=1e-3)
    assert theta.shape == (2, 1) and nu.shape == (2, 1)
    assert np.all(theta == 0.0)

    t_t, n_t = RP.simulate_state(
        torch.full((2, 1, 2), 0.5, dtype=torch.float64), lam=6.0, sigma=6.0, dt=1e-3
    )
    assert tuple(t_t.shape) == (2, 1) and tuple(n_t.shape) == (2, 1)


def test_a_stopped_rotor_stays_finite_in_prediction_and_in_rendering():
    """Negative fitted speed exponents plus a stopped rotor: both paths clamp
    at the SAME positive speed floor, so neither produces a non-finite."""
    export = _tiny_export(
        profile_db=(0.0,), n_mics=1, amp_exp=-0.5, floor_exp=-0.5, bias_mean_hz=0.0
    )
    rps = np.zeros((1, 4096))
    pred = RP.predict_spectrum(export, _rps_clip(rps), n_fft=256, hop=128)
    assert np.isfinite(pred).all() and float(pred.max()) > 0.0
    render = RP.render_revised(export, rps, n_mics=1, seed=3)
    assert np.isfinite(render.audio).all()


def _floor_reference(
    export: dict, rate: np.ndarray, *, n_fft: int, hop: int, frame: int, bins: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """``(exact, flat_response)`` expected floor of one frame at ``bins``.

    ``exact`` is the EXPLICIT double sum
    ``sum_{t,s} w_t w_s A_t A_s R(t-s) e^{-2 pi i f (t-s)} / sum(w^2)`` on the
    work grid — the expected Hann periodogram of ``A(t) n(t)`` with ``n`` the
    coloured noise the renderer synthesizes from the SHARED power spectrum and
    ``A(t)`` the renderer's own sample-by-sample envelope — brought to analysis
    units and through the export's transfer. ``flat_response`` is the model
    that ignores the window's response on a coloured spectrum (the intrinsic
    curve times the ``w^2`` speed exposure), which a coloured floor must NOT
    equal.
    """
    params = export["parameters"]
    sr_work = int(export["training_provenance"]["front_end"]["sample_rate_work"])
    q = sr_work // SR
    n_work = n_fft * q
    w = hann_window(n_work)
    wss = float(np.sum(w**2))

    n = int(rate.size)
    rate_work = np.interp(np.arange(n * q) / sr_work, np.arange(n) / SR, rate)
    psd_len = RP.FLOOR_PSD_OVERSAMPLE * n_work
    shape_mat, tilt_oct = RP.floor_geometry(
        np.fft.rfftfreq(psd_len, d=1.0 / sr_work),
        np.asarray(params["floor_ctrl_hz"], dtype=np.float64),
    )
    cov = np.fft.irfft(
        RP.floor_power_spectrum(
            shape_mat,
            tilt_oct,
            mean_db=float(params["floor_mean_db"]),
            ctrl_db=np.asarray(params["floor_shape_db"], dtype=np.float64),
            tilt_db_oct=float(params["floor_tilt_db_oct"]),
            rate_factor=float(sr_work) / SR,
        ),
        n=psd_len,
    )

    start = frame * hop * q
    seg = rate_work[start : start + n_work]
    gain_t = (seg / AMP_RPS_REF) ** float(params["floor_exp"]) + float(params["floor_static_rel"])
    a = w * np.sqrt(gain_t)
    lag = np.arange(n_work)[:, None] - np.arange(n_work)[None, :]
    gram = a[:, None] * a[None, :] * cov[np.abs(lag)]
    exact = np.array(
        [float(np.sum(gram * np.exp(-2j * np.pi * j * lag / n_work)).real) / wss for j in bins]
    )

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SR)[bins]
    intrinsic = 10.0 ** (
        (
            float(params["floor_mean_db"])
            + float(params["floor_tilt_db_oct"])
            * np.log2(np.maximum(freqs, FLOOR_SHAPE_F_MIN) / FLOOR_TILT_REF_HZ)
        )
        / 10.0
    )
    exposure = float(np.sum(w**2 * gain_t) / wss)
    gain = _transfer(n_fft)[bins]
    return exact * (SR / sr_work) * gain, intrinsic * exposure * gain


def test_the_coloured_floor_is_the_exact_windowed_expectation():
    """A planted COLOURED floor on a ramp, against the explicit double sum.

    The floor goes through the same finite-window kernel as the lines (a real
    atom ``w A_floor`` with ``r_tau = 2 R_floor``), so its prediction must equal
    the exact quadratic form — level, factor of two, head slice, grid factor and
    the sample-by-sample envelope included. A FLAT floor is invariant under the
    window's response, which is why the old renderer regression could not see
    the defect; the second assertion is that on a COLOURED floor the exact
    answer is not the flat-response model."""
    n_fft, hop, n_frames, frame = 256, 128, 4, 2
    n = n_fft + (n_frames - 1) * hop
    rate = np.linspace(120.0, 900.0, n)
    export = _tiny_export(
        profile_db=(-300.0,),
        n_mics=1,
        floor_mean_db=-60.0,
        floor_tilt_db_oct=-6.0,
        floor_exp=2.0,
        bias_mean_hz=0.0,
    )
    pred = RP.predict_spectrum(export, _rps_clip(rate[None, :]), n_fft=n_fft, hop=hop)[0, frame]
    bins = np.array([1, 2, 3, 5, 8, 20, 40])
    exact, flat_response = _floor_reference(
        export, rate, n_fft=n_fft, hop=hop, frame=frame, bins=bins
    )

    assert np.allclose(pred[bins], exact, rtol=1e-6)
    assert float(np.max(np.abs(exact / flat_response - 1.0))) > 0.2


def test_the_predicted_floor_matches_the_rendered_floor_on_a_coloured_ramp():
    """Fit and render read ONE floor power spectrum (``floor_power_spectrum``),
    so the predicted floor must land on the rendered one. A wide gate on
    purpose: this catches a factor of two, a missing rate factor or a drifted
    convention, not the last per cent of the window/filter commutation."""
    n_fft, hop, n16 = 256, 128, 4096
    rate = np.linspace(200.0, 800.0, n16)
    export = _tiny_export(
        profile_db=(-300.0,),
        n_mics=1,
        floor_mean_db=-60.0,
        floor_tilt_db_oct=-3.0,
        floor_exp=2.0,
        bias_mean_hz=0.0,
    )
    pred = RP.predict_spectrum(export, _rps_clip(rate[None, :]), n_fft=n_fft, hop=hop)[0]

    freqs = np.fft.rfftfreq(n_fft, d=1.0 / SR)
    band = (freqs >= 1000.0) & (freqs <= 6000.0)
    rendered = []
    for seed in range(12):
        render = RP.render_revised(export, rate[None, :], n_mics=1, seed=seed)
        clip = Clip("r", "synthetic", render.audio, render.physical_rps, SR)
        rendered.append(periodogram(clip, n_fft, hop).power[0])
    n_frames = min(min(r.shape[0] for r in rendered), pred.shape[0])
    inner = slice(4, n_frames - 4)  # skip the filters' edge transients
    got = float(np.mean([r[inner][:, band] for r in rendered]))
    want = float(np.mean(pred[inner][:, band]))
    assert got == pytest.approx(want, rel=0.35), f"rendered {got:.4g} vs predicted {want:.4g}"


def test_the_floor_speed_law_uses_the_within_window_exposure():
    """On a ramp with a nonlinear ``floor_exp`` the observed floor is the
    ``w^2``-weighted average of ``rate(t)^p`` on the WORK grid — the renderer
    applies the exponent sample by sample there and the periodogram weights it
    by the window's own power envelope. One rectangular frame mean raised to
    ``p`` is a different, wrong number. (A FLAT floor has ``R(tau) = c(0)
    delta``, so the exact kernel route reduces to exactly this exposure.)"""
    n_fft, hop, n_frames = 256, 128, 5
    n = n_fft + (n_frames - 1) * hop
    rate = np.linspace(50.0, 1000.0, n)
    export = _tiny_export(
        profile_db=(-300.0,),
        n_mics=1,
        floor_mean_db=-60.0,
        floor_exp=2.0,
        bias_mean_hz=0.0,
    )
    pred = RP.predict_spectrum(export, _rps_clip(rate[None, :]), n_fft=n_fft, hop=hop)[0]

    q = RP.SAMPLE_RATE_WORK // SR
    n_work = n_fft * q
    w2 = hann_window(n_work) ** 2
    rate_work = np.interp(np.arange(n * q) / RP.SAMPLE_RATE_WORK, np.arange(n) / SR, rate)
    starts = np.arange(n_frames) * hop * q
    exposure = np.array(
        [
            float((w2 * (rate_work[s : s + n_work] / AMP_RPS_REF) ** 2).sum() / w2.sum())
            for s in starts
        ]
    )
    rectangular = np.array(
        [float((rate_work[s : s + n_work].mean() / AMP_RPS_REF) ** 2) for s in starts]
    )
    base = 10.0 ** (-60.0 / 10.0)
    got = (pred / _transfer(n_fft)[None, :])[:, 60]  # any in-band bin: the floor is flat

    assert np.allclose(got, base * exposure, rtol=1e-6)
    assert float(np.max(np.abs(rectangular - exposure) / exposure)) > 0.02


# ── 14: what the fit refuses, and the estimators it promises ────────────────


def test_an_inferred_rotor_track_key_is_refused():
    """``rps_refined`` is an inferred label and ``auto`` resolves to it first,
    so both must be refused by NAME, before a clip is loaded — never quietly
    replaced by a default."""
    for key in ("rps_refined", "auto", "", "refined"):
        with pytest.raises(ValueError, match=r"rigs\.michaels\.rps_key"):
            RP.check_rotor_track_key(key, rig="michaels", where="manifest.json")
    for key in RP.RAW_RPS_KEYS:
        assert RP.check_rotor_track_key(key, rig="michaels") == key


def _meta_clip(clip_id: str, *, start_s: float, seconds: float, recording: str = "REC") -> Clip:
    n = 256 + 2 * 128
    rps = np.full((1, n), 400.0)
    return Clip(
        clip_id,
        "synthetic",
        np.zeros((1, n), dtype=np.float32),
        rps,
        SR,
        rps.copy(),
        dict(dataset="ds", recording_id=recording, start_s=start_s, duration_s=seconds),
    )


def test_duplicate_and_overlapping_training_supports_are_refused():
    """Duplicate-split composite weights stop a repeated WINDOW from creating
    exposure, but each repeated ROW still gets its own state block and its own
    bias prior — two half-losses against two priors. The refusal must name both
    rows, and it must happen before any block is allocated."""
    a = _meta_clip("a", start_s=0.0, seconds=16.0)
    same = _meta_clip("same", start_s=0.0, seconds=16.0)
    overlap = _meta_clip("overlap", start_s=8.0, seconds=16.0)
    disjoint = _meta_clip("disjoint", start_s=16.0, seconds=16.0)
    other = _meta_clip("other", start_s=0.0, seconds=16.0, recording="REC2")

    RP.check_training_supports([("a", a), ("disjoint", disjoint), ("other", other)])
    for partner, name in ((same, "same"), (overlap, "overlap")):
        with pytest.raises(ValueError) as excinfo:
            RP.check_training_supports([("a", a), (name, partner)])
        assert "'a'" in str(excinfo.value) and repr(name) in str(excinfo.value)

    # the manifest-level guard names both ROWS, before a clip is decoded
    rows = [
        dict(recording="REC", regime="cruise", start_s=0.0, seconds=16.0),
        dict(recording="REC", regime="cruise", start_s=8.0, seconds=16.0),
    ]
    with pytest.raises(ValueError, match=r"clips\[0\].*clips\[1\]"):
        RP.check_manifest_supports("dregon", rows, where="m.json")

    # and the fit itself refuses, in the model constructor
    with pytest.raises(ValueError, match="overlapping TRAINING support"):
        RP.fit_revised(
            [("a", a), ("overlap", overlap)],
            rig_id="bench",
            dynamics=RP.ShaftDynamics(
                lam=100.0, sigma=150.0, d_init=0.5, identified=True, diagnostics={}
            ),
            config=_tiny_fit_config(iters=1),
        )


def test_the_minibatch_scale_is_unbiased_for_uniform_sampling():
    """With duplicate-split (unequal) weights the expectation of the scaled
    minibatch risk over uniform sampling WITHOUT replacement must equal the full
    risk. The self-normalized scale it replaced does not: it hands every draw
    the clip's whole exposure regardless of the drawn frames' weights."""
    keys = [("a", 0), ("a", 128), ("a", 128), ("a", 256), ("a", 384)]
    weights = RP.composite_weights(keys, hop=128, n_fft=256)
    cell = np.array([1.0, 4.0, 4.0, 9.0, 16.0])  # per-frame risk, deliberately uneven
    full = float(weights @ cell)

    for size in (1, 2, 3, 4):
        subsets = [list(s) for s in combinations(range(weights.size), size)]
        unbiased = float(
            np.mean([(weights.size / size) * float(weights[s] @ cell[s]) for s in subsets])
        )
        self_normalized = float(
            np.mean(
                [(weights.sum() / weights[s].sum()) * float(weights[s] @ cell[s]) for s in subsets]
            )
        )
        assert unbiased == pytest.approx(full, rel=1e-12)
        assert abs(self_normalized - full) > 1e-6 * full


def test_the_moment_gate_reads_every_microphone_and_every_frame():
    """The SNR gate is a (mic, frame) mask: a noise-only channel must fail it
    even when another channel carries the line loudly. A mic-average or a
    track median admitted exactly that channel."""
    n_fft, hop, n_frames = 512, 256, 24
    n = n_fft + (n_frames - 1) * hop
    rng = np.random.default_rng(7)
    audio = rng.standard_normal((2, n)) * 1e-2
    audio[0] += math.sqrt(2.0) * np.cos(2.0 * np.pi * 1000.0 * np.arange(n) / SR)
    rps = np.full((1, n), 1000.0)
    clip = Clip("gate", "synthetic", audio.astype(np.float32), rps, SR, rps.copy(), {})
    pg = periodogram(clip, n_fft, hop)

    snr = RP.line_snr_db(pg.power, np.full(pg.power.shape[1], 1000.0), pg.df)
    assert snr.shape == (2, pg.power.shape[1])
    assert float(snr[0].min()) > RP.GATE_MIN_LINE_SNR_DB
    assert float(np.mean(snr[1] >= RP.GATE_MIN_LINE_SNR_DB)) < 0.2


def test_lag_pairs_never_straddle_a_gated_frame():
    """A phase increment across a rejected frame is not an increment of the
    process being measured. Pairs come only from one continuous run of valid
    frames, per microphone."""
    valid = np.array([[True] * 5 + [False] * 2 + [True] * 5, [False] * 12])
    pairs = RP.valid_lag_pairs(valid, 1)

    assert {m for m, _, _ in pairs} == {0}  # the noise-only mic contributes nothing
    lo = np.concatenate([a for _, a, _ in pairs])
    hi = np.concatenate([b for _, _, b in pairs])
    assert lo.size == 8  # four pairs inside each run of five
    assert np.all(valid[0][lo]) and np.all(valid[0][hi])
    assert not np.any((lo < 5) & (hi >= 5)), "a pair crossed the gated frames"
    assert RP.valid_lag_pairs(valid, 5) == []  # no run is long enough for lag 5


def test_a_non_cohort_manifest_is_refused():
    """The frozen cohort is provenance, not a preference: a wrong DREGON count,
    a wrong window length or a missing Michael's regime is a different
    experiment and must be refused before the moment stage runs."""
    dregon = [
        dict(recording=f"updown_nosource_room2_{i}", regime="cruise", start_s=0.0, seconds=16.0)
        for i in range(5)
    ]
    RP.check_training_cohort("dregon", dregon)
    with pytest.raises(ValueError, match="not 5"):
        RP.check_training_cohort("dregon", dregon[:4])
    short = [dict(r) for r in dregon]
    short[2]["seconds"] = 8.0
    with pytest.raises(ValueError, match="window length"):
        RP.check_training_cohort("dregon", short)
    with pytest.raises(ValueError, match="cruise"):
        RP.check_training_cohort("dregon", [dict(r, regime="ramp") for r in dregon])

    michaels = [
        dict(recording="FLY125", regime=regime, start_s=start, seconds=16.0)
        for regime, start in (("standby", 0.0), ("ramp", 40.0), ("cruise", 80.0))
    ]
    RP.check_training_cohort("michaels", michaels)
    with pytest.raises(ValueError, match="regimes"):
        RP.check_training_cohort("michaels", michaels[:2])
    with pytest.raises(ValueError, match="FLY125"):
        RP.check_training_cohort("michaels", [dict(m, recording="FLY124") for m in michaels])


def test_a_bench_export_carries_its_diagnostic_only_markers():
    """``bench_diagnostic_only`` and ``scored_arm=False`` must survive into the
    SERIALIZED provenance, or no downstream arm selector can enforce the
    no-unreported-control-arm rule from the artifact itself."""
    n = 256 + 2 * 128
    clip = _planted_clip(theta=np.zeros(n), n_frames=3, seed=8)
    cfg = _tiny_fit_config(iters=1)
    cfg.provenance = dict(
        rig="bench", dataset="planted", bench_diagnostic_only=True, scored_arm=False
    )
    export = RP.fit_revised(
        [(clip.clip_id, clip)],
        rig_id="bench",
        dynamics=RP.ShaftDynamics(
            lam=100.0, sigma=150.0, d_init=0.5, identified=True, diagnostics={}
        ),
        config=cfg,
    )
    prov = export["training_provenance"]
    assert prov["bench_diagnostic_only"] is True
    assert prov["scored_arm"] is False
    assert prov["rig"] == "bench" and prov["dataset"] == "planted"

    # and the coarsening block says what it is, under its honest name
    diag = export["diagnostics"]
    assert "state_grid_convergence" not in diag
    assert diag["coarsening_sensitivity"]["is_convergence_result"] is False
    assert export["fit_method"] == "marginal_then_carrier"
    assert export["lambda_source"] == "fixed_reference"
    assert diag["shared_phase_evidence"] == "not_identified_by_marginal_score"
    assert "identified" not in diag
    assert diag["marginal_fit"]["valid"] and diag["carrier_fit"]["valid"]
    assert export["parameters"]["lambda_source"] == "fixed_reference"


def test_an_unidentified_moment_stage_yields_an_auditable_non_export():
    """When the dynamics are not identified the CLI writes THIS and stops
    before the MAP: it carries the reason and the moment diagnostics, and it is
    explicitly not an export (no parameters to mistake for a fit)."""
    dynamics = RP.ShaftDynamics(
        lam=1.0,
        sigma=1.0,
        d_init=1.0,
        identified=False,
        diagnostics=dict(unidentifiable_reason="longest lag does not reach 1/lam", n_cells=3),
    )
    cfg = _tiny_fit_config()
    cfg.provenance = dict(rig="dregon", scored_arm=True)
    diag = RP.unidentified_diagnostic("dregon", dynamics, cfg)

    assert diag["is_export"] is False and "parameters" not in diag
    assert diag["identified"] is False
    assert "1/lam" in diag["unidentifiable_reason"]
    assert diag["moments"]["n_cells"] == 3
    assert diag["training_provenance"]["rig"] == "dregon"


def _with_harmonic_chunk(export: dict, chunk: int | None) -> dict:
    prov = dict(export["training_provenance"])
    prov["optimizer"] = {**prov["optimizer"], "harmonic_chunk": chunk}
    return {**export, "training_provenance": prov}


def test_the_harmonic_chunk_is_a_memory_device_only():
    """Splitting the order axis into (checkpointed) blocks must not move the
    prediction: no order is dropped, ``k_cap`` is untouched, and only the peak
    memory of computing the same answer changes."""
    export = _tiny_export(profile_db=(0.0, -3.0, -6.0, -9.0), n_mics=1, bias_mean_hz=0.0)
    clip = _rps_clip(np.full((1, 256 + 3 * 128), 500.0))
    whole = RP.predict_spectrum(_with_harmonic_chunk(export, None), clip, n_fft=256, hop=128)
    chunked = RP.predict_spectrum(_with_harmonic_chunk(export, 1), clip, n_fft=256, hop=128)
    assert np.allclose(whole, chunked, rtol=1e-10)


def test_backtracking_accepts_only_a_full_objective_decrease():
    p = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    named = {"p": p}
    before = RP._snapshot_parameters(named)
    with torch.no_grad():
        p.fill_(4.0)  # full proposal makes (p-1)^2 worse than at p=0
    proposed = RP._snapshot_parameters(named)

    accepted, value, step, backtracks = RP._backtrack_segment(
        named,
        before,
        proposed,
        before_objective=1.0,
        objective=lambda: float((p - 1.0).square().item()),
        max_halvings=4,
    )

    assert accepted
    assert step == pytest.approx(0.25)
    assert backtracks == 2
    assert value < 1.0
    assert p.item() == pytest.approx(1.0)


def test_backtracking_restores_the_block_when_every_half_step_fails():
    p = torch.nn.Parameter(torch.tensor(0.0, dtype=torch.float64))
    named = {"p": p}
    before = RP._snapshot_parameters(named)
    with torch.no_grad():
        p.fill_(8.0)
    proposed = RP._snapshot_parameters(named)

    accepted, value, step, backtracks = RP._backtrack_segment(
        named,
        before,
        proposed,
        before_objective=0.0,
        objective=lambda: float((p - 1.0).square().item()),
        max_halvings=4,
    )

    assert not accepted
    assert value == 0.0
    assert step == 0.0 and backtracks == 4
    assert p.item() == pytest.approx(0.0)

def test_alternating_fit_exports_fixed_hyperparams_and_history():
    n = 256 + 2 * 128
    clip = _planted_clip(theta=np.zeros(n), n_frames=3, seed=11)
    cfg = _tiny_fit_config(
        fit_method="alternating_conditional_map",
        training_recipe="full",
        fixed_lambda=6.0,
        fixed_sigma=3.10117415072719,
        initial_d=2.0 * math.pi * 13.07,
        alternating_cycles=1,
        alternating_block_iters=1,
        alternating_lr=0.001,
        alternating_backtracks=2,
        frames_per_step=1,
        iters=1,
    )
    export = RP.fit_revised(
        [(clip.clip_id, clip)],
        rig_id="dregon",
        dynamics=RP.ShaftDynamics(
            lam=99.0, sigma=99.0, d_init=999.0, identified=True, diagnostics={}
        ),
        config=cfg,
    )

    assert export["fit_method"] == "alternating_conditional_map"
    assert export["parameters"]["lam"] == pytest.approx(6.0)
    assert export["parameters"]["sigma"] == pytest.approx(3.10117415072719)
    alt = export["diagnostics"]["alternating_conditional_map"]
    assert alt["valid"] and len(alt["objective_history"]) == 2
    assert alt["loss_trace"][0] == pytest.approx(
        export["diagnostics"]["initial_full_objective_after_reset"]
    )
    assert np.all(np.diff(np.asarray(alt["loss_trace"], dtype=float)) <= 1e-9)



def test_alternating_contract_history_is_full_objective_monotone(tmp_path: Path):
    export = _tiny_export(profile_db=(0.0, -3.0), n_mics=1, bias_mean_hz=0.0)
    export["fit_method"] = "alternating_conditional_map"
    export["lambda_source"] = "fixed_reference"
    export["shared_phase_evidence"] = "not_identified_by_marginal_score"
    export["diagnostics"]["alternating_conditional_map"] = {
        "valid": True,
        "loss_trace": [10.0, 9.0, 9.0],
        "objective_history": [
            {
                "cycle": 1,
                "block": "carrier",
                "full_objective_before": 10.0,
                "full_objective_after": 9.0,
                "accepted": True,
                "accepted_step": 0.5,
            },
            {
                "cycle": 1,
                "block": "spectral",
                "full_objective_before": 9.0,
                "full_objective_after": 9.0,
                "accepted": False,
                "accepted_step": 0.0,
            },
        ],
    }
    export["diagnostics"]["fixed_ou_hyperparameters"] = {
        "lambda_": 6.0,
        "sigma": 3.10117415072719,
        "provenance": "test",
    }
    path = tmp_path / "alt.json"
    path.write_text(json.dumps(export))

    from experiments.stochastic_fit import revised_eval as RE

    assert RE.read_candidate_export(path).fit_contract["fit_method"] == "alternating_conditional_map"

    export["diagnostics"]["alternating_conditional_map"]["loss_trace"] = [10.0, 11.0]
    path.write_text(json.dumps(export))
    with pytest.raises(ValueError, match="non-increasing full objective"):
        RE.read_candidate_export(path)


@pytest.mark.parametrize(
    ("rig_id", "carrier_source", "rps_key"),
    [
        ("dregon", "raw", "motors_command"),
        ("dregon", "refined", "rps_refined"),
        ("michaels", "raw", "rps"),
    ],
)
def test_fixed_carrier_fit_export_read_predict_and_render_smoke(
    tmp_path: Path, rig_id: str, carrier_source: str, rps_key: str
) -> None:
    n_fft, hop, n_frames = 64, 32, 2
    n = n_fft + (n_frames - 1) * hop
    clip = _line_clip(
        f"{rig_id}_{carrier_source}",
        rps=np.full(n, 70.0),
        profile_db=-18.0,
        n_fft=n_fft,
        hop=hop,
        floor_std=1e-3,
        seed=37,
    )
    clip.meta.update(
        dataset="toy",
        recording_id=f"{rig_id}_toy",
        start_s=0.0,
        duration_s=n / SR,
        channels=[0],
        rps_key=rps_key,
    )
    cfg = _tiny_fit_config(
        n_fft=n_fft,
        hop=hop,
        fit_method="fixed_carrier_marginal",
        training_recipe="full",
        iters=1,
        frame_chunk=1,
        harmonic_chunk=None,
        lbfgs_max_iter=2,
        lbfgs_max_eval=4,
        lbfgs_history_size=2,
        lbfgs_line_search="strong_wolfe",
        frames_per_step=None,
        carrier_source=carrier_source,
        provenance={"carrier_source": carrier_source, "regimes": {clip.clip_id: "toy"}, "rps_key": rps_key},
    )
    export = RP.fit_revised(
        [(clip.clip_id, clip)],
        rig_id=rig_id,
        dynamics=RP.ShaftDynamics(lam=6.0, sigma=1.0, d_init=1.0, identified=True, diagnostics={}),
        config=cfg,
    )
    assert export["fit_method"] == "fixed_carrier_marginal"
    assert export["carrier_source"] == carrier_source
    assert export["training_provenance"]["carrier_source"] == carrier_source
    assert export["training_provenance"]["clips"][0]["rps_key"] == rps_key
    assert "map_state" not in export["diagnostics"]
    assert "carrier_fit" not in export["diagnostics"]
    assert "coarsening_sensitivity" not in export["diagnostics"]
    assert "bias_vs_nu_mean_confounding" not in export["diagnostics"]
    assert np.all(np.asarray(export["parameters"]["bias_mean_hz"], dtype=float) == 0.0)
    assert np.all(np.asarray(export["parameters"]["bias_hz"][clip.clip_id], dtype=float) == 0.0)
    fit = export["diagnostics"]["marginal_fit"]
    assert fit["valid"] is True
    assert len(fit["closure_trace"]) == fit["eval_count"]

    path = tmp_path / "fixed.json"
    path.write_text(json.dumps(export))
    from experiments.stochastic_fit import revised_eval as RE

    read = RE.read_candidate_export(path)
    assert read.provenance()["carrier_source"] == carrier_source
    pred = RP.predict_spectrum(read.summary, clip, n_fft=n_fft, hop=hop)
    rendered = RP.render_revised(read.summary, clip.rps, n_mics=1, seed=3)
    assert pred.shape == (1, n_frames, n_fft // 2 + 1)
    assert np.isfinite(pred).all()
    assert rendered.audio.shape == (1, n)
    assert np.isfinite(rendered.audio).all()
