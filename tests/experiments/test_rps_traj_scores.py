"""The campaign's forecast and sample-fidelity scores measure what they claim."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from scipy import stats
from scipy.signal import lfilter

from experiments.rps_traj.cruise_api import burn_samples
from experiments.rps_traj.scores import (
    crps,
    energy_score,
    forecast_scores,
    mmd2_rbf,
    pooled_acf,
    sample_scores,
    welch_modes,
)
from tracking.rotors import MIXER

RATE = 25.0


def _gauss_crps(y: float) -> float:
    """Closed-form CRPS of N(0, 1) at ``y`` (Gneiting and Raftery 2007, eq. 21)."""
    return float(y * (2.0 * stats.norm.cdf(y) - 1.0) + 2.0 * stats.norm.pdf(y) - 1 / np.sqrt(np.pi))


# --------------------------------------------------------------------------
# energy score / CRPS
# --------------------------------------------------------------------------


@pytest.mark.parametrize("y", [-2.0, -0.5, 0.0, 1.0, 3.0])
def test_fair_crps_is_unbiased_for_gaussian(y: float) -> None:
    # m = 5 makes the biased (1 / 2m^2) form off by E|X - X'| / (2m) = 0.11,
    # eight standard errors at 2500 replicates.
    rng = np.random.default_rng(10)
    draws = rng.normal(size=(2500, 5))
    c = np.array([crps(d, y) for d in draws])
    e = np.array([energy_score(d[:, None], np.array([y])) for d in draws[:200]])
    se = c.std() / np.sqrt(c.size)
    assert abs(c.mean() - _gauss_crps(y)) < 4.0 * se
    np.testing.assert_allclose(e, c[:200], rtol=1e-12, atol=1e-12)


def test_crps_matches_pairwise_definition() -> None:
    rng = np.random.default_rng(11)
    x = np.round(rng.normal(size=37), 1)  # ties included
    y = 0.3
    m = x.size
    brute = np.abs(x - y).mean() - np.abs(x[:, None] - x[None, :]).sum() / (2 * m * (m - 1))
    assert crps(x, y) == pytest.approx(brute, rel=1e-12)
    x4 = rng.normal(size=(23, 4))
    y4 = rng.normal(size=4)
    pair = np.linalg.norm(x4[:, None] - x4[None, :], axis=-1).sum()
    brute4 = np.linalg.norm(x4 - y4, axis=1).mean() - pair / (2 * 23 * 22)
    assert energy_score(x4, y4) == pytest.approx(brute4, rel=1e-12)


def test_miscalibrated_ensembles_score_worse() -> None:
    rng = np.random.default_rng(12)
    obs = rng.normal(size=(300, 4))
    ens = {
        "true": lambda: rng.normal(size=(50, 4)),
        "shift": lambda: rng.normal(1.0, 1.0, size=(50, 4)),
        "wide": lambda: rng.normal(0.0, 2.0, size=(50, 4)),
    }
    es = {k: np.mean([energy_score(f(), y) for y in obs]) for k, f in ens.items()}
    cr = {k: np.mean([crps(f()[:, 0], y[0]) for y in obs]) for k, f in ens.items()}
    for bad in ("shift", "wide"):
        assert es[bad] > 1.1 * es["true"]
        assert cr[bad] > 1.1 * cr["true"]


# --------------------------------------------------------------------------
# forecast_scores
# --------------------------------------------------------------------------


class _AR1:
    """Four independent AR(1) rotors; ``forecast`` is the exact conditional law."""

    name = "ar1"
    rate_hz = RATE
    n_params = 2

    def __init__(self, tau_s: float, sigma: float) -> None:
        self.a = float(np.exp(-1.0 / (tau_s * RATE)))
        self.sigma = sigma

    def logpdf(self, block: np.ndarray, burn: int) -> np.ndarray:
        raise NotImplementedError

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        e = rng.normal(0.0, self.sigma * np.sqrt(1 - self.a**2), size=(4, n + 300))
        return np.asarray(lfilter([1.0], [1.0, -self.a], e, axis=1))[:, 300:]

    def forecast(
        self, history: np.ndarray, horizon: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        out = np.empty((n_paths, 4, horizon))
        x = np.broadcast_to(history[:, -1], (n_paths, 4))
        s = self.sigma * np.sqrt(1 - self.a**2)
        for k in range(horizon):
            x = self.a * x + s * rng.normal(size=(n_paths, 4))
            out[:, :, k] = x
        return out

    def to_dict(self) -> dict[str, Any]:
        return {"a": self.a, "sigma": self.sigma}


class _Oracle:
    """Returns the true future of the block it was built on; records origins."""

    name = "oracle"
    rate_hz = RATE
    n_params = 0

    def __init__(self, blocks: list[np.ndarray]) -> None:
        self.blocks = blocks
        self.calls: list[tuple[int, int, int]] = []

    def logpdf(self, block: np.ndarray, burn: int) -> np.ndarray:
        raise NotImplementedError

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError

    def forecast(
        self, history: np.ndarray, horizon: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        o = history.shape[1]
        (b,) = [i for i, blk in enumerate(self.blocks) if np.array_equal(blk[:, :o], history)]
        self.calls.append((b, o, horizon))
        fut = self.blocks[b][:, o : o + horizon]
        return np.broadcast_to(fut, (n_paths, *fut.shape)).copy()

    def to_dict(self) -> dict[str, Any]:
        return {}


def test_true_conditional_law_beats_climatology_then_meets_it() -> None:
    model = _AR1(tau_s=0.5, sigma=3.0)
    rng = np.random.default_rng(20)
    blocks = [
        b - b.mean(axis=1, keepdims=True) for b in (model.sample(750, rng) for _ in range(30))
    ]
    res = forecast_scores(model, blocks, n_paths=100, rng=np.random.default_rng(21))
    assert res["n_origins"] == 30 * 14
    h = res["horizons"]
    # lead 0.2 s is 5 steps past the history: conditional std sqrt(1 - a^10) = 0.74.
    for key in ("energy", "crps"):
        assert h["0.2"][key] / h["0.2"][f"{key}_clim"] < 0.85
        assert h["2"][key] / h["2"][f"{key}_clim"] == pytest.approx(1.0, abs=0.07)
    # CRPS of a calibrated N(0, 3^2) forecast averages 3 / sqrt(pi) per rotor.
    assert h["2"]["crps"] == pytest.approx(3.0 / np.sqrt(np.pi), rel=0.07)


def test_origins_and_leads_follow_the_spec() -> None:
    rng = np.random.default_rng(22)
    blocks = [rng.normal(size=(4, 750)), rng.normal(size=(4, 420))]
    oracle = _Oracle(blocks)
    res = forecast_scores(oracle, blocks, n_paths=8, rng=np.random.default_rng(23))
    burn, horizon, stride = burn_samples(RATE), 50, 50
    expected = [
        (b, o)
        for b, blk in enumerate(blocks)
        for o in range(burn, blk.shape[1] - horizon + 1, stride)
    ]
    assert [(b, o) for b, o, _ in oracle.calls] == expected
    assert {hz for _, _, hz in oracle.calls} == {horizon}
    assert res["n_origins"] == len(expected) == 14 + 7
    for v in res["horizons"].values():
        assert abs(v["energy"]) < 1e-12 and abs(v["crps"]) < 1e-12
        assert v["energy_clim"] > 0.5 and v["crps_clim"] > 0.3

    oracle.calls.clear()
    res = forecast_scores(oracle, blocks, n_paths=8, rng=np.random.default_rng(23), max_origins=5)
    assert res["n_origins"] == 5
    picked = [(b, o) for b, o, _ in oracle.calls]
    assert picked == [expected[i] for i in (0, 5, 10, 15, 20)]


# --------------------------------------------------------------------------
# sample_scores and helpers
# --------------------------------------------------------------------------


def _proc(
    rng: np.random.Generator,
    *,
    tau: float = 0.5,
    sigma: float = 1.0,
    rho: float = 0.5,
    jumps: bool = False,
    n_blocks: int = 20,
    n: int = 750,
) -> list[np.ndarray]:
    """Equicorrelated (``rho``) four-rotor AR(1) blocks, demeaned."""
    a = np.exp(-1.0 / (tau * RATE))
    chol = np.linalg.cholesky((1 - rho) * np.eye(4) + rho * np.ones((4, 4)))
    out = []
    for _ in range(n_blocks):
        e = rng.normal(size=(4, n + 200))
        if jumps:  # 2 % of innovations x7, same variance
            e = np.where(rng.random(e.shape) < 0.02, 7.0 * e, e) / np.sqrt(0.02 * 49 + 0.98)
        x = np.asarray(lfilter([sigma * np.sqrt(1 - a * a)], [1.0, -a], chol @ e, axis=1))[:, 200:]
        out.append(x - x.mean(axis=1, keepdims=True))
    return out


@pytest.fixture(scope="module")
def real_and_null() -> tuple[list[np.ndarray], dict[str, float]]:
    rng = np.random.default_rng(30)
    real = _proc(rng)
    return real, sample_scores(real, _proc(rng), RATE, rng=np.random.default_rng(0))


def test_same_process_gives_null_distances(real_and_null) -> None:
    _, null = real_and_null
    for lag in ("0.2", "1"):
        assert null[f"inc_w1_{lag}"] < 0.05
        assert null[f"inc_kurt_diff_{lag}"] < 0.3
    assert null["acf_rmse"] < 0.08
    assert null["xcorr_mae"] < 0.08
    assert null["var_log_ratio"] < 0.2
    assert null["psd_lsd_db"] < 1.2
    assert abs(null["mmd2"]) < 0.01


@pytest.mark.parametrize(
    ("kind", "keys", "floor"),
    [
        ("var2", ("var_log_ratio", "inc_w1_0.2", "psd_lsd_db", "mmd2"), 0.02),
        ("tau", ("acf_rmse", "psd_lsd_db", "inc_w1_0.2"), 0.1),
        ("rho", ("xcorr_mae",), 0.3),
        ("jumps", ("inc_kurt_diff_0.2",), 2.0),
    ],
)
def test_each_alternative_moves_its_statistic(real_and_null, kind, keys, floor) -> None:
    real, null = real_and_null
    rng = np.random.default_rng(31)
    gen = {
        "var2": lambda: _proc(rng, sigma=np.sqrt(2.0)),
        "tau": lambda: _proc(rng, tau=1.5),
        "rho": lambda: _proc(rng, rho=0.0),
        "jumps": lambda: _proc(rng, jumps=True),
    }[kind]()
    alt = sample_scores(real, gen, RATE, rng=np.random.default_rng(0))
    for key in keys:
        assert alt[key] > floor
        assert alt[key] > 4.0 * abs(null[key])


def test_pure_scaling_moves_only_level_statistics(real_and_null) -> None:
    real, _ = real_and_null
    s = sample_scores(real, [np.sqrt(2.0) * b for b in real], RATE)
    assert s["psd_lsd_db"] == pytest.approx(10 * np.log10(2.0), rel=1e-9)
    assert s["var_log_ratio"] == pytest.approx(np.log(2.0), rel=1e-9)
    assert s["acf_rmse"] < 1e-12
    assert s["xcorr_mae"] < 1e-12
    assert s["inc_kurt_diff_0.2"] < 1e-9


def test_pooled_acf_matches_definition_on_unequal_blocks() -> None:
    rng = np.random.default_rng(40)
    blocks = [rng.normal(size=(4, 30)), rng.normal(size=(4, 100))]
    max_lag = 50
    num = np.zeros((4, max_lag + 1))
    cnt = np.zeros(max_lag + 1)
    for b in blocks:
        for k in range(min(max_lag + 1, b.shape[1])):
            num[:, k] += (b[:, : b.shape[1] - k] * b[:, k:]).sum(axis=1)
            cnt[k] += b.shape[1] - k
    want = (num / cnt) / (num[:, :1] / cnt[0])
    np.testing.assert_allclose(pooled_acf(blocks, max_lag), want, rtol=1e-10, atol=1e-12)


def test_pooled_acf_recovers_ar1() -> None:
    real = _proc(np.random.default_rng(41), n_blocks=40, tau=0.5)
    acf = pooled_acf(real, 25)
    a = np.exp(-1.0 / (0.5 * RATE))
    np.testing.assert_allclose(acf, np.broadcast_to(a ** np.arange(26), acf.shape), atol=0.05)


def test_welch_modes_puts_pitch_on_the_pitch_mode() -> None:
    t = np.arange(1000) / RATE
    modes = np.zeros((4, t.size))
    modes[2] = np.sin(2 * np.pi * 2.0 * t)
    freqs, psd = welch_modes([MIXER @ modes], RATE, 200)
    df = freqs[1] - freqs[0]
    assert freqs[np.argmax(psd[2])] == pytest.approx(2.0)
    assert psd[2].sum() * df == pytest.approx(0.5, rel=0.02)
    assert np.max(psd[[0, 1, 3]]) < 1e-20


# --------------------------------------------------------------------------
# MMD
# --------------------------------------------------------------------------


def test_mmd2_matches_definition() -> None:
    rng = np.random.default_rng(50)
    x, y = rng.normal(size=(7, 3)), rng.normal(size=(9, 3))
    bw = 1.3

    def k(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.exp(-np.sum((a - b) ** 2) / (2 * bw * bw)))

    kxx = sum(k(x[i], x[j]) for i in range(7) for j in range(7) if i != j) / 42
    kyy = sum(k(y[i], y[j]) for i in range(9) for j in range(9) if i != j) / 72
    kxy = sum(k(a, b) for a in x for b in y) / 63
    assert mmd2_rbf(x, y, bw) == pytest.approx(kxx + kyy - 2 * kxy, rel=1e-10)


def test_mmd2_separates_laws_and_is_unbiased() -> None:
    rng = np.random.default_rng(51)
    same = mmd2_rbf(rng.normal(size=(300, 5)), rng.normal(size=(300, 5)))
    diff = mmd2_rbf(rng.normal(size=(300, 5)), rng.normal(0.5, 1.0, size=(300, 5)))
    assert abs(same) < 0.01
    assert diff > 0.05
    # the biased V-statistic would sit ~2 (1 - E k) / n = 0.1 above zero at n = 10.
    reps = np.array(
        [mmd2_rbf(rng.normal(size=(10, 3)), rng.normal(size=(10, 3)), 1.5) for _ in range(800)]
    )
    assert abs(reps.mean()) < 4.0 * reps.std() / np.sqrt(reps.size)
