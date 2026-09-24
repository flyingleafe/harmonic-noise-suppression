"""The v3 wander measurement's load-bearing claims (``experiments.noise_model.wander``).

Every number the v3 fit holds fixed comes out of three pieces: the block-noise
variance subtracted from the observed spread, the centred-moment OU fit that
turns lag products into ``(sigma, tau)``, and the order-tracked block level
itself. Each is checked here on planted data -- no dataset, no network.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from experiments.noise_model import wander as W
from experiments.stochastic_fit.data import Clip, periodogram

SR = W.FLIGHT_SR


def _gamma_db(rng: np.random.Generator, n: float, size: tuple[int, ...]) -> np.ndarray:
    """dB of the mean of ``n`` unit exponentials: the explainer's block noise."""
    return 10.0 * np.log10(rng.gamma(n, 1.0 / n, size))


class TestBlockNoise:
    @pytest.mark.parametrize("n", [1.0, 4.0, 20.0])
    def test_gamma_formula_matches_simulation(self, n):
        rng = np.random.default_rng(0)
        sim = float(np.var(_gamma_db(rng, n, (200_000,))))
        assert float(W.noise_var_db(n)) == pytest.approx(sim, rel=0.02)

    def test_overlapping_frames_carry_fewer_independent_cells(self):
        # White noise on the flight front end, a 3-bin "line" read per 0.5 s
        # block: the spread of the block dB means is what the explainer's
        # count predicts only once the 75 % frame overlap and the Hann bin
        # correlation are counted; both the exact correlated-cell count and
        # the within-block measurement get it, the literal count does not.
        rng = np.random.default_rng(1)
        x = rng.normal(size=(1, SR * 90)).astype(np.float32)
        pg = periodogram(Clip("w", "w", x, np.zeros((1, x.shape[1])), SR, None, {}))
        power = pg.power[0].astype(np.float64)
        blocks = W.frame_blocks(pg.times, 0.5)
        cols = np.arange(40, 1000, 6)
        lines = np.stack([power[:, c - 1 : c + 2].mean(axis=1) for c in cols])  # (G, T)
        means = np.stack([lines[:, f].mean(axis=1) for f in blocks], axis=1)  # (G, B)
        y = 10.0 * np.log10(means)
        mc = float(np.var(y - y.mean(axis=1, keepdims=True), axis=1, ddof=1).mean())
        e = np.zeros_like(lines)
        for b, f in enumerate(blocks):
            e[:, f] = lines[:, f] / means[:, b : b + 1] - 1.0
        valid = np.ones(means.shape, dtype=bool)
        measured = float(np.nanmean(W.rel_var_to_db2(W.within_block_mean_var(e, blocks, valid))))
        n = np.array([f.size for f in blocks]) * 3
        exact = float(np.mean(W.noise_var_db(n / [W.overlap_inflation(int(k) // 3, 3) for k in n])))
        literal = float(np.mean(W.noise_var_db(n)))
        assert exact == pytest.approx(mc, rel=0.1)
        assert measured == pytest.approx(mc, rel=0.15)
        assert literal < 0.5 * mc


def _pooled(tracks: np.ndarray, s2: float, *, rig: bool) -> W.LagMoments:
    """``tracks`` ``(lines, windows, blocks)``: one moment set, centred per
    window (``rig=False``) or per line over all its windows (``rig=True``)."""
    mom = W.LagMoments()
    n_l, n_w, n_b = tracks.shape
    s = np.full((n_w, n_b), s2)
    for i in range(n_l):
        if rig:
            mom.add(tracks[i], None, s, n_blocks=n_b)
        else:
            for w in range(n_w):
                mom.add(tracks[i, w], None, s[w])
    return mom


class TestOUMoments:
    """y = d + eps: an OU at the block rate plus the known Gamma block noise."""

    NB = 20  # cells behind a block: s^2 = noise_var_db(20) ~ 1 dB^2

    @pytest.mark.parametrize("rig", [False, True])
    def test_recovers_sigma_and_tau_when_tau_is_near_the_window(self, rig):
        rng = np.random.default_rng(2)
        sigma, tau, bs = 2.0, 3.0, 0.5
        d = W.simulate_ou_blocks(rng, 200 * 10, 16, sigma, tau, bs).reshape(200, 10, 16)
        y = d + _gamma_db(rng, self.NB, d.shape)
        fit = W.fit_ou(_pooled(y, float(W.noise_var_db(self.NB)), rig=rig), bs)
        assert fit.sigma_db == pytest.approx(sigma, rel=0.08)
        assert fit.tau_s == pytest.approx(tau, rel=0.25)

    def test_rig_centring_sees_a_wander_slower_than_the_window(self):
        # tau = 20 s against 4 s windows: inside one window the wander is an
        # offset, so the window-centred fit cannot size it; centring on the
        # line's mean over all windows can, exactly as the v3 fit's shared
        # p_ik makes its latents carry the window offsets.
        rng = np.random.default_rng(3)
        sigma, tau, bs = 2.0, 20.0, 0.5
        d = W.simulate_ou_blocks(rng, 150 * 20, 8, sigma, tau, bs).reshape(150, 20, 8)
        y = d + _gamma_db(rng, self.NB, d.shape)
        s2 = float(W.noise_var_db(self.NB))
        rig = W.fit_ou(_pooled(y, s2, rig=True), bs)
        assert rig.sigma_db == pytest.approx(sigma, rel=0.08)
        assert rig.tau_s == pytest.approx(tau, rel=0.35)
        naive = _pooled(y, s2, rig=False).naive(bs)
        assert math.sqrt(naive["sigma2"]) < 0.6 * sigma

    def test_common_and_residual_split(self):
        # Four lines of one rotor: d (common) + v_k (own) + independent block
        # noise, plus a white block term SHARED by the lines of a block (a
        # broadband event). The pair moments at lags 1-4 give d without it;
        # the auto moments minus the d part give v.
        rng = np.random.default_rng(4)
        bs, n_rot, n_w, n_b, n_k = 0.5, 60, 10, 16, 4
        sd, td, sv, tv, shared = 1.0, 4.0, 2.0, 1.0, 1.0
        auto, same = W.LagMoments(), W.LagMoments()
        s2 = float(W.noise_var_db(self.NB))
        for _ in range(n_rot):
            d = W.simulate_ou_blocks(rng, n_w, n_b, sd, td, bs)
            common_noise = rng.normal(0.0, shared, (n_w, n_b))
            ys = [
                d
                + W.simulate_ou_blocks(rng, n_w, n_b, sv, tv, bs)
                + common_noise
                + _gamma_db(rng, self.NB, (n_w, n_b))
                for _ in range(n_k)
            ]
            s_blk = np.full((n_w, n_b), s2 + shared**2)
            for i, a in enumerate(ys):
                auto.add(a, None, s_blk, n_blocks=n_b)
                for b in ys[i + 1 :]:
                    same.add(a, b, n_blocks=n_b)
                    same.add(b, a, n_blocks=n_b)
        com = W.fit_ou(same, bs, lags=(1, 2, 3, 4), use_noise=False, skip_lag0=True)
        assert com.sigma_db == pytest.approx(sd, rel=0.15)
        assert com.tau_s == pytest.approx(td, rel=0.4)
        assert com.nugget == pytest.approx(shared**2, rel=0.2)
        off = com.sigma2 * auto.expected_unit(np.array([com.rho]))[0]
        res = W.fit_ou(auto, bs, offset=off)
        assert res.sigma_db == pytest.approx(sv, rel=0.1)
        assert res.tau_s == pytest.approx(tv, rel=0.3)

    def test_no_wander_clips_to_zero(self):
        rng = np.random.default_rng(5)
        y = _gamma_db(rng, self.NB, (100, 10, 8))
        fit = W.fit_ou(_pooled(y, float(W.noise_var_db(self.NB)) * 1.3, rig=True), 0.5)
        assert fit.clipped and fit.sigma2 == 0.0


class TestBlockLinePower:
    """The order-tracked block level from audio (the R4 accumulation, per block)."""

    @staticmethod
    def _rotor(seconds: float, f_lo: float, f_hi: float, k: int, amp: np.ndarray, seed: int):
        rng = np.random.default_rng(seed)
        n = int(seconds * SR)
        f0 = np.linspace(f_lo, f_hi, n)
        phase = 2.0 * np.pi * np.cumsum(k * f0) / SR
        x = amp * np.cos(phase) + 0.02 * rng.normal(size=n)
        return x[None, :].astype(np.float32), f0[None, :]

    def test_a_steady_line_on_a_moving_carrier_reads_flat(self):
        # Order 20 of a 78 -> 84 rev/s ramp crosses ~15 bins in 4 s; tracking
        # the carrier frame by frame keeps the block level flat to the noise.
        x, f0 = self._rotor(4.0, 78.0, 84.0, 20, np.full(4 * SR, 0.1), seed=6)
        lb = W.block_line_power_db(x, f0, 20, SR, 0.5)
        y = lb.micmean_db()[0, 0]
        assert np.isfinite(y).all() and lb.n_blocks >= 7
        assert float(np.ptp(y)) < 0.5

    def test_an_amplitude_step_is_read_in_the_right_blocks(self):
        n = 4 * SR
        amp = np.where(np.arange(n) < n // 2, 0.05, 0.1)  # +6.02 dB half-way
        x, f0 = self._rotor(4.0, 80.0, 82.0, 5, amp, seed=7)
        lb = W.block_line_power_db(x, f0, [5], SR, 0.5)
        y = lb.micmean_db()[0, 0]
        centre = (np.arange(y.size) + 0.5) * 0.5
        lo, hi = y[centre < 1.75], y[centre > 2.25]
        assert float(hi.mean() - lo.mean()) == pytest.approx(20 * math.log10(2.0), abs=0.3)
