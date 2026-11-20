"""The cruise model of the flight-regime campaign: noise-level switching.

The fitted rotor-speed model (:mod:`data_processing.trajectory_model.params`)
is a linear Gaussian state space: per mixer mode a slow OU plus a damped
oscillator, driven by white noise, plus a per-rotor measurement OU.  Real
cruise is not Gaussian: short bursts of large corrections sit among long
stretches of small ones.  This module keeps the linear system and lets the
SIZE of its driving noise switch::

    x_{t+1} = Phi x_t + kappa_{s_t} w_shaft,t + w_meas,t      y_t = H x_t

with ``s_t`` a ``K``-level continuous-time Markov chain (generator ``G``) and
``sum_j pi_j kappa_j^2 = 1`` under its stationary law ``pi``.  The noise stays
white with the same mean power, so the stationary covariance, the spectrum,
the autocorrelation and the inter-rotor correlations of the Gaussian model are
unchanged; only WHEN the energy arrives changes.  ``K = 1`` is the Gaussian
model itself.

The fit has two stages, which that invariance makes natural:

1. :func:`fit_gaussian` — the second-order structure, by the campaign's exact
   Gaussian likelihood (:mod:`experiments.rps_traj.model`, unchanged), on the
   settled-cruise segments it is given;
2. :func:`fit_levels` — the ``K (K - 1)`` rates and ``K - 1`` level ratios, by
   the likelihood of the switching model with the dynamics held fixed.  That
   likelihood is evaluated by the interacting-multiple-model filter (Blom and
   Bar-Shalom 1988): ``K`` Kalman filters that differ only in process noise,
   merged at every step with the chain's transition matrix; Kim (1994) is the
   reference for estimating this class by maximum likelihood with a collapsed
   filter.

:class:`SwitchingCruise` implements
:class:`~experiments.rps_traj.cruise_api.CruiseModel`.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from numba import njit, prange
from scipy.linalg import expm
from scipy.optimize import minimize

from data_processing.trajectory_model import Params
from experiments.rps_traj import model as M
from experiments.rps_traj.cruise_api import NUM_ROTORS, burn_samples, cut_blocks
from tracking.rotors import MIXER

#: Shaft states of the 16-state model (4 slow OUs + 4 oscillator pairs); the
#: last four states are the per-rotor measurement process, which no flight
#: regime scales.
N_SHAFT = 12

#: Bounds of the level fit: rates between one switch per minute and the sample
#: rate (a stay may be as short as one sample, which is all a sampled track can
#: tell), consecutive level ratios between 1.02 and 30.
MIN_RATE_HZ = 1.0 / 60.0
MAX_RATIO = 30.0
MIN_RATIO = 1.02

#: Scored samples the starts of the level fit are compared on before the best
#: is polished on all blocks (the level law is a few numbers; ten minutes of
#: data rank its starts as well as an hour does, at a fraction of the cost).
EXPLORE_SAMPLES = 30_000

_LOG_2PI = float(np.log(2.0 * np.pi))


# ─── the state space, as numpy ────────────────────────────────────────────────


def state_space(
    params: Params, rate_hz: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """``(Phi, Q_shaft, Q_meas, P_stationary, H)`` of ``params`` at ``rate_hz``.

    The same exact discretisation the Gaussian likelihood uses
    (:func:`experiments.rps_traj.model._state_space_torch`); the process noise
    is split into the part a flight regime scales and the measurement part.
    """
    vec = M._fit_vector(params)
    phi, q, p_stat, h = (
        t.detach().numpy().copy() for t in M._state_space_torch(M._unpack(vec), 1.0 / rate_hz)
    )
    q = 0.5 * (q + q.T)
    q_shaft = np.zeros_like(q)
    q_shaft[:N_SHAFT, :N_SHAFT] = q[:N_SHAFT, :N_SHAFT]
    return phi, q_shaft, q - q_shaft, 0.5 * (p_stat + p_stat.T), h


def _psd_sqrt(m: np.ndarray) -> np.ndarray:
    """A square root ``L`` with ``L L' = m`` of a possibly singular PSD matrix."""
    vals, vecs = np.linalg.eigh(0.5 * (m + m.T))
    return vecs * np.sqrt(np.maximum(vals, 0.0))


# ─── the levels ───────────────────────────────────────────────────────────────


def stationary(gen: np.ndarray) -> np.ndarray:
    """The stationary law ``pi`` of a generator (``pi G = 0``, ``sum pi = 1``)."""
    k = gen.shape[0]
    if k == 1:
        return np.ones(1)
    a = np.vstack([gen.T, np.ones((1, k))])
    b = np.zeros(k + 1)
    b[-1] = 1.0
    pi = np.linalg.lstsq(a, b, rcond=None)[0]
    pi = np.clip(pi, 1e-12, None)
    return pi / pi.sum()


def levels_from_vector(theta: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """``(G, kappa^2)`` from the unconstrained level vector.

    ``theta`` holds the ``k (k - 1)`` log rates (row by row, off-diagonal) and
    then ``k - 1`` values whose ``1 + exp`` are the ratios of consecutive
    levels, so the levels are ordered quiet to loud and ``kappa^2`` is
    normalised to mean 1 under the stationary law.
    """
    if k == 1:
        return np.zeros((1, 1)), np.ones(1)
    n_rate = k * (k - 1)
    gen = np.zeros((k, k))
    gen[~np.eye(k, dtype=bool)] = np.exp(theta[:n_rate])
    np.fill_diagonal(gen, -gen.sum(axis=1))
    ratio = 1.0 + np.exp(theta[n_rate:])
    kappa = np.concatenate([[1.0], np.cumprod(ratio)])
    pi = stationary(gen)
    k2 = kappa * kappa
    return gen, k2 / float(pi @ k2)


def vector_from_levels(gen: np.ndarray, kappa2: np.ndarray) -> np.ndarray:
    """Inverse of :func:`levels_from_vector` (the normalisation is dropped)."""
    k = gen.shape[0]
    if k == 1:
        return np.zeros(0)
    rates = np.log(np.maximum(gen[~np.eye(k, dtype=bool)], 1e-12))
    ratio = np.sqrt(kappa2[1:] / kappa2[:-1])
    return np.concatenate([rates, np.log(np.maximum(ratio - 1.0, 1e-6))])


# ─── the interacting-multiple-model filter ────────────────────────────────────


@njit(cache=True)
def _rows(a):  # pragma: no cover - jitted
    """Row-sparse form of a small matrix: per row, the columns and values of
    its non-zeros.  ``Phi`` is block diagonal (two non-zeros a row at most) and
    ``H`` has nine a row, so the filter's products are a tenth of the dense
    ones; and no BLAS call is made, which matters inside ``prange``."""
    n, m = a.shape
    idx = np.zeros((n, m), dtype=np.int64)
    val = np.zeros((n, m))
    cnt = np.zeros(n, dtype=np.int64)
    for i in range(n):
        c = 0
        for j in range(m):
            if a[i, j] != 0.0:
                idx[i, c] = j
                val[i, c] = a[i, j]
                c += 1
        cnt[i] = c
    return idx, val, cnt


@njit(cache=True, fastmath=True)
def _imm_block(y, phi, q_shaft, q_meas, h, p0, trans, kappa2, pi0):  # pragma: no cover - jitted
    """One block ``y (n, 4)``: per-sample log predictive density ``(n,)``, the
    filtered level probabilities ``(n, k)`` and the final filtered moments.

    Step ``t`` mixes the ``k`` filtered Gaussians with the chain's transition
    matrix (the level in force over the step into ``t`` scales that step's
    shaft noise), predicts and updates each, and scores ``y_t`` under the
    mixture of the ``k`` predictions.  The filter starts from the stationary
    covariance and the stationary level law.  Every inner loop runs along a
    contiguous row, so it vectorises.
    """
    n = y.shape[0]
    k = kappa2.shape[0]
    ns = phi.shape[0]
    d = h.shape[0]
    pidx, pval, pcnt = _rows(phi)
    logp = np.empty(n)
    probs = np.empty((n, k))
    x = np.zeros((k, ns))
    cov = np.empty((k, ns, ns))
    for j in range(k):
        cov[j] = p0
    mu = pi0.copy()
    x0 = np.zeros((k, ns))
    cov0 = np.empty((k, ns, ns))
    like = np.empty(k)
    c = np.empty(k)
    xp = np.empty(ns)
    tmp = np.empty((ns, ns))
    pp = np.empty((ns, ns))
    pht = np.empty((d, ns))  # (P H')' : row r is P H'[:, r]
    s = np.empty((d, d))
    chol = np.zeros((d, d))
    e = np.empty(d)
    v = np.empty(d)
    z = np.empty(d)
    g = np.empty(d)
    const = d * _LOG_2PI
    for t in range(n):
        # mixing: the law of the previous level given the current one
        for j in range(k):
            acc = 0.0
            for i in range(k):
                acc += mu[i] * trans[i, j]
            c[j] = acc
        if k == 1:
            for a in range(ns):
                x0[0, a] = x[0, a]
                for b in range(ns):
                    cov0[0, a, b] = cov[0, a, b]
        else:
            for j in range(k):
                for a in range(ns):
                    x0[j, a] = 0.0
                for i in range(k):
                    w = mu[i] * trans[i, j] / c[j]
                    for a in range(ns):
                        x0[j, a] += w * x[i, a]
                for a in range(ns):
                    for b in range(ns):
                        cov0[j, a, b] = 0.0
                for i in range(k):
                    w = mu[i] * trans[i, j] / c[j]
                    for a in range(ns):
                        da = w * (x[i, a] - x0[j, a])
                        for b in range(ns):
                            cov0[j, a, b] += w * cov[i, a, b] + da * (x[i, b] - x0[j, b])
        # one Kalman step per level
        top = -np.inf
        for j in range(k):
            cj = cov0[j]
            pj = cov[j]
            for a in range(ns):
                acc = 0.0
                for m in range(pcnt[a]):
                    acc += pval[a, m] * x0[j, pidx[a, m]]
                xp[a] = acc
            # P- = Phi C Phi' + kappa^2 Q_shaft + Q_meas
            for a in range(ns):
                for b in range(ns):
                    tmp[a, b] = 0.0
                for m in range(pcnt[a]):
                    f = pval[a, m]
                    col = pidx[a, m]
                    for b in range(ns):
                        tmp[a, b] += f * cj[col, b]
            for a in range(ns):
                for b in range(ns):
                    pp[a, b] = kappa2[j] * q_shaft[a, b] + q_meas[a, b]
            for b in range(ns):
                for m in range(pcnt[b]):
                    f = pval[b, m]
                    col = pidx[b, m]
                    for a in range(ns):
                        pp[b, a] += f * tmp[a, col]
            for r in range(d):
                acc = y[t, r]
                for a in range(ns):
                    acc -= h[r, a] * xp[a]
                e[r] = acc
            for r in range(d):
                for b in range(ns):
                    acc = 0.0
                    for a in range(ns):
                        acc += pp[b, a] * h[r, a]
                    pht[r, b] = acc
            for r in range(d):
                for q in range(r, d):
                    acc = 0.0
                    for b in range(ns):
                        acc += h[r, b] * pht[q, b]
                    s[r, q] = acc
                    s[q, r] = acc
            # S = L L'; log det S; v = L^-1 e
            logdet = 0.0
            for r in range(d):
                for q in range(r + 1):
                    acc = s[r, q]
                    for m in range(q):
                        acc -= chol[r, m] * chol[q, m]
                    if r == q:
                        chol[r, r] = np.sqrt(max(acc, 1e-300))
                        logdet += 2.0 * np.log(chol[r, r])
                    else:
                        chol[r, q] = acc / chol[q, q]
            quad = 0.0
            for r in range(d):
                acc = e[r]
                for m in range(r):
                    acc -= chol[r, m] * v[m]
                v[r] = acc / chol[r, r]
                quad += v[r] * v[r]
            like[j] = np.log(c[j]) - 0.5 * (const + logdet + quad)
            # row a of the gain P H' S^-1 through the Cholesky factor, then
            # x = x- + K e and P = P- - K (P H')'
            for a in range(ns):
                for r in range(d):
                    acc = pht[r, a]
                    for m in range(r):
                        acc -= chol[r, m] * z[m]
                    z[r] = acc / chol[r, r]
                for r in range(d - 1, -1, -1):
                    acc = z[r]
                    for m in range(r + 1, d):
                        acc -= chol[m, r] * g[m]
                    g[r] = acc / chol[r, r]
                acc = xp[a]
                for r in range(d):
                    acc += g[r] * e[r]
                x[j, a] = acc
                for b in range(ns):
                    pj[a, b] = pp[a, b]
                for r in range(d):
                    f = g[r]
                    for b in range(ns):
                        pj[a, b] -= f * pht[r, b]
            if like[j] > top:
                top = like[j]
        tot = 0.0
        for j in range(k):
            tot += np.exp(like[j] - top)
        logp[t] = top + np.log(tot)
        for j in range(k):
            mu[j] = np.exp(like[j] - logp[t])
            probs[t, j] = mu[j]
    return logp, probs, x, cov, mu


@njit(cache=True, parallel=True)
def _imm_sum(
    ys, lengths, burn, phi, q_shaft, q_meas, h, p0, trans, kappa2, pi0
):  # pragma: no cover
    """Summed log predictive density of padded blocks ``ys (B, n_max, 4)``."""
    total = np.zeros(ys.shape[0])
    for b in prange(ys.shape[0]):
        logp = _imm_block(ys[b, : lengths[b]], phi, q_shaft, q_meas, h, p0, trans, kappa2, pi0)[0]
        total[b] = logp[burn:].sum()
    return total.sum()


@njit(cache=True)
def _simulate(phi, drive, x0):  # pragma: no cover - jitted
    """``x_{t+1} = Phi x_t + drive_t`` from ``x0``; returns ``(n, ns)`` with
    row ``t`` the state AFTER ``drive_t`` (row 0 is ``Phi x0 + drive_0``)."""
    n = drive.shape[0]
    ns = phi.shape[0]
    pidx, pval, pcnt = _rows(phi)
    out = np.empty((n, ns))
    x = x0.copy()
    nxt = np.empty(ns)
    for t in range(n):
        for a in range(ns):
            acc = drive[t, a]
            for m in range(pcnt[a]):
                acc += pval[a, m] * x[pidx[a, m]]
            nxt[a] = acc
        for a in range(ns):
            x[a] = nxt[a]
            out[t, a] = nxt[a]
    return out


def _level_path(
    trans: np.ndarray, start: int, n: int, rng: np.random.Generator, forced: int = 0
) -> np.ndarray:
    """``(n,)`` level indices: the chain from ``start``, its first ``forced``
    samples held at the loudest level."""
    k = trans.shape[0]
    path = np.empty(n, dtype=np.int64)
    if k == 1:
        path[:] = 0
        return path
    cum = np.cumsum(trans, axis=1)
    u = rng.random(n)
    s = start
    for t in range(n):
        s = k - 1 if t < forced else min(int(np.searchsorted(cum[s], u[t])), k - 1)
        path[t] = s
    return path


# ─── the model ────────────────────────────────────────────────────────────────


@dataclass
class SwitchingCruise:
    """The structured cruise model with ``K`` noise levels, at ``rate_hz``."""

    name: str
    rate_hz: float
    params: Params
    gen: np.ndarray = field(default_factory=lambda: np.zeros((1, 1)))
    kappa2: np.ndarray = field(default_factory=lambda: np.ones(1))
    #: Free parameters of the fluctuation law: the 23 dynamics plus the levels.
    n_params: int = M.N_FIT_PARAMS
    info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.gen = np.atleast_2d(np.asarray(self.gen, dtype=np.float64))
        self.kappa2 = np.asarray(self.kappa2, dtype=np.float64).reshape(-1)
        self._cache: dict[float, tuple[np.ndarray, ...]] = {}

    # ── derived ──
    @property
    def n_levels(self) -> int:
        return int(self.kappa2.size)

    @property
    def occupancy(self) -> np.ndarray:
        """Stationary share of time at each level."""
        return stationary(self.gen)

    @property
    def dwell_s(self) -> np.ndarray:
        """Mean stay at each level (s)."""
        if self.n_levels == 1:
            return np.array([np.inf])
        return 1.0 / np.maximum(-np.diag(self.gen), 1e-300)

    def matrices(self, rate_hz: float | None = None) -> tuple[np.ndarray, ...]:
        """``(Phi, Q_shaft, Q_meas, P_stat, H, transition)`` at a rate (cached)."""
        rate = float(self.rate_hz if rate_hz is None else rate_hz)
        if rate not in self._cache:
            self._cache[rate] = (*state_space(self.params, rate), expm(self.gen / rate))
        return self._cache[rate]

    # ── the CruiseModel contract ──
    def filter(self, block: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(log predictive (n,), filtered level probabilities (n, K))``."""
        phi, qs, qm, p0, h, trans = self.matrices()
        y = np.ascontiguousarray(np.asarray(block, dtype=np.float64).T)
        logp, probs, *_ = _imm_block(y, phi, qs, qm, h, p0, trans, self.kappa2, self.occupancy)
        return logp, probs

    def logpdf(self, block: np.ndarray, burn: int) -> np.ndarray:
        return self.filter(block)[0][int(burn) :]

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        *,
        rate_hz: float | None = None,
        forced: int = 0,
        shaft_only: bool = False,
        return_levels: bool = False,
    ) -> Any:
        """``(4, n)`` stationary zero-mean fluctuation.

        ``rate_hz`` samples the same continuous-time model on another grid;
        ``forced`` holds the first samples at the loudest level (the settling
        after a take-off); ``shaft_only`` leaves the measurement process out,
        which is what a renderer wants.
        """
        phi, qs, qm, p0, h, trans = self.matrices(rate_hz)
        k = self.n_levels
        start = k - 1 if forced else int(rng.choice(k, p=self.occupancy))
        path = _level_path(trans, start, int(n), rng, forced)
        ns = phi.shape[0]
        drive = (rng.standard_normal((int(n), ns)) @ _psd_sqrt(qs).T) * np.sqrt(self.kappa2)[
            path, None
        ]
        if not shaft_only:
            drive = drive + rng.standard_normal((int(n), ns)) @ _psd_sqrt(qm).T
        x0 = _psd_sqrt(p0) @ rng.standard_normal(ns)
        if shaft_only:
            x0[N_SHAFT:] = 0.0
        y = np.ascontiguousarray((_simulate(phi, drive, x0) @ h.T).T)
        return (y, path) if return_levels else y

    def forecast(
        self, history: np.ndarray, horizon: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        phi, qs, qm, p0, h, trans = self.matrices()
        y = np.ascontiguousarray(np.asarray(history, dtype=np.float64).T)
        _, _, x, cov, mu = _imm_block(y, phi, qs, qm, h, p0, trans, self.kappa2, self.occupancy)
        k, ns, m = self.n_levels, phi.shape[0], int(n_paths)
        # the filtered law is a mixture over the level of the last step
        level = rng.choice(k, size=m, p=mu / mu.sum())
        roots = np.stack([_psd_sqrt(cov[j]) for j in range(k)])
        state = x[level] + np.einsum("pab,pb->pa", roots[level], rng.standard_normal((m, ns)))
        ls, lm = _psd_sqrt(qs), _psd_sqrt(qm)
        cum = np.cumsum(trans, axis=1)
        scale = np.sqrt(self.kappa2)
        out = np.empty((m, NUM_ROTORS, int(horizon)))
        for t in range(int(horizon)):
            if k > 1:
                level = np.minimum((rng.random(m)[:, None] > cum[level]).sum(axis=1), k - 1)
            drive = (rng.standard_normal((m, ns)) @ ls.T) * scale[level, None]
            state = state @ phi.T + drive + rng.standard_normal((m, ns)) @ lm.T
            out[:, :, t] = state @ h.T
        return out

    # ── serialisation ──
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rate_hz": self.rate_hz,
            "n_params": self.n_params,
            "params": self.params.to_dict(),
            "generator_per_s": self.gen.tolist(),
            "kappa2": self.kappa2.tolist(),
            "occupancy": self.occupancy.tolist(),
            "dwell_s": [float(v) for v in self.dwell_s],
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> SwitchingCruise:
        return cls(
            name=str(payload["name"]),
            rate_hz=float(payload["rate_hz"]),
            params=Params.from_dict(payload["params"]),
            gen=np.asarray(payload["generator_per_s"], dtype=np.float64),
            kappa2=np.asarray(payload["kappa2"], dtype=np.float64),
            n_params=int(payload["n_params"]),
            info=dict(payload.get("info", {})),
        )


# ─── stage 1: the second-order structure ──────────────────────────────────────


def gaussian_batches(segments: Sequence[np.ndarray], rate_hz: float) -> list[M._Batch]:
    """The Gaussian likelihood's blocks from explicit segments at ``rate_hz``.

    The cut of :func:`experiments.rps_traj.model.likelihood_batches` —
    overlapping :data:`~experiments.rps_traj.model.BLOCK_S` blocks, each
    demeaned, the overlap filtered but not scored — applied to the segments it
    is given instead of to the frozen airborne rule's.
    """
    block = int(round(M.BLOCK_S * rate_hz))
    burn = int(round(M.BURN_S * rate_hz))
    least = int(round(M.MIN_SCORED_S * rate_hz))
    groups: dict[tuple[int, int], list[np.ndarray]] = {}
    for seg in segments:
        seg = np.asarray(seg, dtype=np.float64)
        if not np.isfinite(seg).all():
            continue
        n, start = seg.shape[1], 0
        while start < n:
            stop = min(start + block, n)
            this_burn = 0 if start == 0 else burn
            if stop - start - this_burn < least:
                break
            chunk = seg[:, start:stop]
            width = min(-(-chunk.shape[1] // M.PAD_QUANTUM) * M.PAD_QUANTUM, block)
            groups.setdefault((this_burn, width), []).append(
                chunk - chunk.mean(axis=1, keepdims=True)
            )
            if stop == n:
                break
            start += block - burn
    batches = []
    for (burn_n, width), blocks in sorted(groups.items()):
        y = np.zeros((len(blocks), NUM_ROTORS, width))
        mask = np.zeros((len(blocks), width))
        for i, b in enumerate(blocks):
            y[i, :, : b.shape[1]] = b
            mask[i, burn_n : b.shape[1]] = 1.0
        batches.append(
            M._Batch(
                y=torch.as_tensor(y, dtype=torch.float64).transpose(1, 2),
                mask=torch.as_tensor(mask, dtype=torch.float64),
                burn=burn_n,
            )
        )
    if not batches:
        raise ValueError("no usable likelihood blocks: every segment was too short or not finite")
    return batches


def level_stats(groups: Mapping[str, Sequence[np.ndarray]]) -> tuple[np.ndarray, float, np.ndarray]:
    """``(mu, s_c, s_r)`` from the settled cruise of each flight: the pooled
    rotor mean and the common / per-rotor spread of the per-flight means, by
    the moment rule of :func:`experiments.rps_traj.model.pooled_means`."""
    stacks = [np.concatenate(list(segs), axis=1) for segs in groups.values() if len(segs)]
    mu = np.concatenate(stacks, axis=1).mean(axis=1)
    if len(stacks) < M.S_MIN_FLIGHTS:
        return mu, 0.0, np.zeros(NUM_ROTORS)
    offsets = np.stack([s.mean(axis=1) - mu for s in stacks], axis=0)
    cov = offsets.T @ offsets / float(offsets.shape[0])
    var_c = max(float(cov[~np.eye(NUM_ROTORS, dtype=bool)].mean()), 0.0)
    return mu, float(np.sqrt(var_c)), np.sqrt(np.maximum(np.diag(cov) - var_c, 0.0))


def _data_start(
    segments: Sequence[np.ndarray], mu: np.ndarray, s_c: float, s_r: np.ndarray
) -> Params:
    """The data-driven start of :func:`experiments.rps_traj.model._start_params`."""
    centred = np.concatenate([s - s.mean(axis=1, keepdims=True) for s in segments], axis=1)
    mode_var = np.maximum(((MIXER.T @ centred) / NUM_ROTORS).var(axis=1), 1e-6)
    rms = float(np.sqrt(np.mean(centred.var(axis=1))))
    return Params(
        mu=mu,
        theta=0.0,
        tau_slow=np.full(NUM_ROTORS, 2.0),
        sigma_slow=np.sqrt(0.7 * mode_var),
        f0=np.full(NUM_ROTORS, 1.0),
        zeta=np.full(NUM_ROTORS, 1.0),
        sigma_osc=np.sqrt(0.3 * mode_var),
        tau_e=0.05,
        sigma_e=max(0.05 * rms, 1e-3),
        s_c=s_c,
        s_r=s_r,
    )


def fit_gaussian(
    groups: Mapping[str, Sequence[np.ndarray]],
    rate_hz: float,
    *,
    seed: int = 0,
    n_restarts: int = 4,
    start: Params | None = None,
    max_iter: int | None = None,
) -> tuple[Params, dict[str, Any]]:
    """Stage 1: the exact-likelihood MAP fit of the Gaussian model.

    ``groups`` maps a flight to its settled-cruise segments ``(4, n)`` at
    ``rate_hz``.  Same likelihood, priors, optimiser and restarts as
    :func:`experiments.rps_traj.model.fit_rig`; only the segments differ.
    """
    segments = [s for segs in groups.values() for s in segs]
    mu, s_c, s_r = level_stats(groups)
    batches = gaussian_batches(segments, rate_hz)
    start_vec = M._fit_vector(start or _data_start(segments, mu, s_c, s_r)).numpy()
    bounds = M._fit_bounds()
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])

    def objective(x: np.ndarray) -> tuple[float, np.ndarray]:
        vec = torch.as_tensor(x, dtype=torch.float64).requires_grad_(True)
        value = M._neg_log_post(vec, batches, rate_hz)
        (grad,) = torch.autograd.grad(value, vec)
        return float(value.item()), grad.numpy()

    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    best: tuple[float, np.ndarray, int] | None = None
    for restart in range(max(int(n_restarts), 1)):
        x0 = np.clip(start_vec if restart == 0 else M._jitter(start_vec, rng), lo, hi)
        res = minimize(
            objective,
            x0,
            jac=True,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": int(max_iter)} if max_iter else None,
        )
        if np.isfinite(res.fun) and (best is None or res.fun < best[0]):
            best = (float(res.fun), np.asarray(res.x, dtype=np.float64), int(res.nit))
    if best is None:
        raise RuntimeError("every restart of the Gaussian fit failed")
    params = M._params_from_fit_vector(best[1], mu, s_c, s_r).canonical()
    info = {
        "neg_log_post": best[0],
        "n_iter": best[2],
        "wall_s": time.perf_counter() - t0,
        "n_blocks": int(sum(b.n_blocks for b in batches)),
        "n_scored": int(sum(b.n_scored for b in batches)),
        "n_restarts": max(int(n_restarts), 1),
    }
    return params, info


# ─── stage 2: the noise levels ────────────────────────────────────────────────


def _pad(blocks: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    lengths = np.array([b.shape[1] for b in blocks], dtype=np.int64)
    ys = np.zeros((len(blocks), int(lengths.max()), NUM_ROTORS))
    for i, b in enumerate(blocks):
        ys[i, : b.shape[1]] = b.T
    return ys, lengths


def total_logpdf(model: SwitchingCruise, blocks: Sequence[np.ndarray], burn: int) -> float:
    """Summed log predictive density of ``blocks`` past their burn-in."""
    phi, qs, qm, p0, h, trans = model.matrices()
    ys, lengths = _pad(blocks)
    return float(
        _imm_sum(ys, lengths, int(burn), phi, qs, qm, h, p0, trans, model.kappa2, model.occupancy)
    )


def _level_starts(k: int, rate_hz: float) -> list[np.ndarray]:
    """Starting points: (entry rate, stay, ratio) of the loudest level."""
    out = []
    for enter_hz, stay_s, ratio in ((0.3, 0.5, 2.5), (1.5, 0.25, 2.0), (0.05, 1.0, 4.0)):
        if k == 2:
            gen = np.array([[-enter_hz, enter_hz], [1.0 / stay_s, -1.0 / stay_s]])
            k2 = np.array([1.0, ratio**2])
        else:
            gen = np.full((k, k), 0.05)
            for j in range(k - 1):
                gen[j, j + 1] = enter_hz * (j + 1)
                gen[j + 1, j] = 1.0 / (stay_s * (k - j - 1) ** 0.5)
            np.fill_diagonal(gen, 0.0)
            np.fill_diagonal(gen, -gen.sum(axis=1))
            k2 = ratio ** (2.0 * np.arange(k) / (k - 1) * 1.5)
        out.append(vector_from_levels(gen, k2))
    return out


def fit_levels(
    params: Params,
    blocks: Sequence[np.ndarray],
    rate_hz: float,
    n_levels: int,
    *,
    name: str | None = None,
    warm: SwitchingCruise | None = None,
) -> SwitchingCruise:
    """Stage 2: the rates and level ratios of a ``n_levels``-level switch, by
    the IMM likelihood of ``blocks`` (demeaned, scored past the burn-in) with
    the dynamics ``params`` fixed.  ``warm`` adds a start built from a fitted
    model with one level fewer (its loudest level split in two).
    """
    name = name or ("gauss" if n_levels == 1 else f"switch{n_levels}")
    burn = burn_samples(rate_hz)
    n_scored = int(sum(max(b.shape[1] - burn, 0) for b in blocks))
    if n_levels == 1:
        model = SwitchingCruise(name=name, rate_hz=rate_hz, params=params)
        model.info = {"train_logpdf": total_logpdf(model, blocks, burn) / n_scored}
        return model

    phi, qs, qm, p0, h = state_space(params, rate_hz)
    k = int(n_levels)
    n_rate = k * (k - 1)
    lo = np.concatenate(
        [np.full(n_rate, np.log(MIN_RATE_HZ)), np.full(k - 1, np.log(MIN_RATIO - 1.0))]
    )
    hi = np.concatenate([np.full(n_rate, np.log(rate_hz)), np.full(k - 1, np.log(MAX_RATIO - 1.0))])
    bounds = list(zip(lo, hi))

    def objective_on(subset: Sequence[np.ndarray]) -> Any:
        ys, lengths = _pad(subset)
        n = int(sum(max(b.shape[1] - burn, 0) for b in subset))

        def objective(theta: np.ndarray) -> float:
            gen, kappa2 = levels_from_vector(theta, k)
            value = _imm_sum(
                ys, lengths, burn, phi, qs, qm, h, p0, expm(gen / rate_hz), kappa2, stationary(gen)
            )
            return -float(value) / n

        return objective

    # every start is run on a fixed random subset of the blocks when the data
    # are many, and the best one is then polished on all of them
    subset: list[np.ndarray] = []
    count = 0
    for i in np.random.default_rng(0).permutation(len(blocks)):
        subset.append(blocks[int(i)])
        count += max(blocks[int(i)].shape[1] - burn, 0)
        if count >= EXPLORE_SAMPLES:
            break
    explore = objective_on(subset)

    starts = _level_starts(k, rate_hz)
    if warm is not None and warm.n_levels == k - 1 and k > 2:
        gen = np.full((k, k), 0.05)
        gen[: k - 1, : k - 1] = warm.gen
        gen[k - 2, k - 1], gen[k - 1, k - 2] = 0.5, 3.0
        np.fill_diagonal(gen, 0.0)
        np.fill_diagonal(gen, -gen.sum(axis=1))
        warm_start = vector_from_levels(gen, np.append(warm.kappa2, 3.0 * warm.kappa2[-1]))
        starts = [warm_start, starts[0]]
    t0 = time.perf_counter()
    best = None
    n_eval = 0
    for x0 in starts:
        res = minimize(
            explore,
            np.clip(x0, lo, hi),
            method="L-BFGS-B",
            bounds=bounds,
            options={"eps": 1e-4, "maxiter": 200},
        )
        n_eval += int(res.nfev)
        if np.isfinite(res.fun) and (best is None or res.fun < best.fun):
            best = res
    if best is None:
        raise RuntimeError("every start of the level fit failed")
    if len(subset) < len(blocks):
        res = minimize(
            objective_on(blocks),
            best.x,
            method="L-BFGS-B",
            bounds=bounds,
            options={"eps": 1e-4, "maxiter": 100},
        )
        n_eval += int(res.nfev)
        if np.isfinite(res.fun):
            best = res
    gen, kappa2 = levels_from_vector(best.x, k)
    model = SwitchingCruise(
        name=name,
        rate_hz=rate_hz,
        params=params,
        gen=gen,
        kappa2=kappa2,
        n_params=M.N_FIT_PARAMS + n_rate + k - 1,
    )
    model.info = {
        "train_logpdf": -float(best.fun),
        "n_eval": n_eval,
        "wall_s": time.perf_counter() - t0,
        "at_bound": bool(np.any(np.isclose(best.x, lo) | np.isclose(best.x, hi))),
    }
    return model


def training_blocks(groups: Mapping[str, Sequence[np.ndarray]], rate_hz: float) -> list[np.ndarray]:
    """Every demeaned block (:func:`~experiments.rps_traj.cruise_api.cut_blocks`)
    of the settled cruise in ``groups``."""
    return [b for segs in groups.values() for s in segs for b in cut_blocks(s, rate_hz)]


__all__ = [
    "N_SHAFT",
    "SwitchingCruise",
    "fit_gaussian",
    "fit_levels",
    "gaussian_batches",
    "level_stats",
    "levels_from_vector",
    "state_space",
    "stationary",
    "total_logpdf",
    "training_blocks",
    "vector_from_levels",
]
