"""Vector-autoregressive cruise baselines: plain VAR(p), Markov-switching VAR, VAR(p)-HSMM.

Three literature baselines of the flight-regime campaign, each a
:class:`~experiments.rps_traj.cruise_api.CruiseModel` over the campaign's
BLOCK (``(4, n)`` rotor speeds in rev/s):

* :class:`VarModel` / :func:`fit_var` — the plain Gaussian VAR(p)
  ``y_t = c + sum_k A_k y_{t-k} + e_t``, ``e_t ~ N(0, Sigma)`` (Lütkepohl,
  *New Introduction to Multiple Time Series Analysis*, 2005, ch. 3-4): pooled
  OLS over blocks (a block contributes only rows whose lags lie inside it),
  order chosen by BIC on a common set of rows, and an explicit stability
  guard (companion spectral radius pulled under :data:`STABLE_RADIUS`).
* :class:`SwitchingVarModel` with ``durations='geometric'`` — the
  Markov-switching VAR of Hamilton (1989, *Econometrica* 57:357), state
  dependent intercept, AR matrices and innovation covariance, a transition
  matrix with self-transitions (geometric dwell times).
* :class:`SwitchingVarModel` with ``durations='explicit'`` — the regularised
  VAR(p)-HSMM of Xu and Liu (2021, arXiv:1804.10308): initial law ``delta``,
  a transition matrix ``Q`` without self-transitions, a nonparametric dwell
  pmf ``r_j(n)``, ``n = 1..D`` (longer visits censored at ``D``), LASSO on the
  AR coefficients (Tibshirani 1996, coordinate descent after Friedman et al.
  2007) and the Ledoit-Wolf-type shrinkage ``(S + lam c I) / (1 + lam)`` of
  every state covariance (Ledoit and Wolf 2004); ``lam_a`` / ``lam_sigma``
  are picked by the paper's one-step MSFE criterion
  (:func:`select_regularization`).

Both switching variants are fitted by EM (:func:`fit_switching_var`).  The
E-step is one scaled forward-backward pass over the elapsed-time expanded state
``(j, n)`` ("in state j, n samples into the visit"), which costs ``O(T M D)``
instead of the paper's ``O(T M^2 D^2)`` segment recursion; the geometric model
is the same recursion with ``D = 1`` and self-transitions allowed, so one
numba kernel serves both.  Every block is an independent sequence that opens a
fresh visit drawn from ``delta`` at ``t = order`` (the first ``order`` samples
only supply lags), and the last visit is right-censored by the block end: its
likelihood is the survival ``P(dur >= n)``, and the M-step spreads its
expected count over ``m >= n`` with weights ``r_j(m) / P(dur >= n)``, which
keeps EM an exact EM (monotone without penalties) for the censored likelihood.

The M-step follows the paper: ``delta``, ``Q`` and ``r_j`` are normalised
expected counts; per state a ``gamma_t(j)``-weighted least squares, intercept
unpenalised, AR coefficients with ``lam_a * ||a||_1`` added to the plain
weighted sum of squares (summed over the four outputs, as the paper's
``vec`` form), and ``Sigma_j = (S_j + lam_sigma c_j I) / (1 + lam_sigma)``,
``c_j = tr(S_j) / 4``, then ``+ sigma_floor I``.  The fit runs on data
divided by one global scale (the pooled std of the training blocks), so
``lam_a``, ``lam_sigma`` and ``sigma_floor`` are scale-free: ``sigma_floor``
is a variance in units of the pooled training variance, and ``lam_a`` is on
the scale of the weighted sum of squares of unit-variance data (it grows
with the number of samples, as in the paper's unnormalised objective).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from functools import cached_property
from typing import Any, Literal

import numpy as np
from numba import njit
from scipy.linalg import solve_discrete_lyapunov, solve_triangular
from scipy.ndimage import uniform_filter1d

from experiments.rps_traj.cruise_api import NUM_ROTORS, burn_samples

#: A fitted VAR whose companion spectral radius reaches this is shrunk under it.
STABLE_RADIUS = 0.999

#: Length of the burn-in a ``start='stationary'`` switching sample discards (s);
#: at least ten mean dwells of the slowest state are discarded as well.
SAMPLE_BURN_S = 60.0

_LOG_2PI = float(np.log(2.0 * np.pi))
_DurationLaw = Literal["geometric", "explicit"]


# ─── shared linear-Gaussian helpers ──────────────────────────────────────────


def _check_block(block: np.ndarray) -> np.ndarray:
    y = np.asarray(block, dtype=np.float64)
    if y.ndim != 2 or y.shape[0] != NUM_ROTORS:
        raise ValueError(f"block must be (4, n), got {y.shape}")
    if not np.isfinite(y).all():
        raise ValueError("block has non-finite samples")
    return y


def _check_blocks(blocks: Any) -> list[np.ndarray]:
    out = [_check_block(b) for b in blocks]
    if not out:
        raise ValueError("no blocks")
    return out


def _check_burn(y: np.ndarray, burn: int, order: int) -> None:
    if burn < order:
        raise ValueError(f"burn={burn} must be >= the VAR order {order}")
    if burn > y.shape[1]:
        raise ValueError(f"burn={burn} exceeds the block length {y.shape[1]}")


def _design(
    y: np.ndarray, order: int, start: int, intercept: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Regressors ``(n - start, [1] + 4 order)`` and targets ``(n - start, 4)``.

    Columns are ``[1?, y_{t-1}, ..., y_{t-order}]``; with ``Y ~ X B`` the lag-k
    matrix is ``A_k = B[rows of lag k].T``.
    """
    n = y.shape[1]
    cols = [np.ones((n - start, 1))] if intercept else []
    cols += [y[:, start - k : n - k].T for k in range(1, order + 1)]
    return np.hstack(cols), np.ascontiguousarray(y[:, start:].T)


def _stack_design(
    blocks: list[np.ndarray], order: int, start: int, intercept: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pooled design over the blocks longer than ``start``, plus row offsets."""
    xs, ys, sizes = [], [], [0]
    for y in blocks:
        if y.shape[1] > start:
            x_b, y_b = _design(y, order, start, intercept)
            xs.append(x_b)
            ys.append(y_b)
            sizes.append(len(y_b))
    if not xs:
        raise ValueError(f"no block is longer than {start} samples")
    return np.vstack(xs), np.vstack(ys), np.cumsum(sizes).astype(np.int64)


def _coefs_from_b(b: np.ndarray, order: int, intercept: bool) -> tuple[np.ndarray, np.ndarray]:
    off = int(intercept)
    coefs = np.stack([b[off + 4 * k : off + 4 * (k + 1)].T for k in range(order)])  # (order, 4, 4)
    c = b[0].copy() if intercept else np.zeros(NUM_ROTORS)
    return coefs, c


def _cond_mean(y: np.ndarray, coefs: np.ndarray, c: np.ndarray, start: int) -> np.ndarray:
    """``(n - start, 4)`` VAR conditional means ``c + sum_k A_k y_{t-k}``."""
    n = y.shape[1]
    mean = np.broadcast_to(c, (n - start, NUM_ROTORS)).copy()
    for k in range(1, coefs.shape[0] + 1):
        mean += y[:, start - k : n - k].T @ coefs[k - 1].T
    return mean


def _safe_cholesky(s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lower Cholesky factor of a covariance, adding jitter only if it is singular."""
    s = 0.5 * (s + s.T)
    jitter = 0.0
    trace = float(np.trace(s)) / s.shape[0]
    base = trace if trace > 0.0 else 1.0
    for _ in range(30):
        try:
            mat = s + jitter * np.eye(s.shape[0])
            return np.linalg.cholesky(mat), mat
        except np.linalg.LinAlgError:
            jitter = base * 1e-12 if jitter == 0.0 else jitter * 10.0
    raise np.linalg.LinAlgError("covariance is not positive definite")


def _gauss_logpdf(resid: np.ndarray, chol: np.ndarray) -> np.ndarray:
    """``(T,)`` log N(resid; 0, L L^T) for residual rows ``(T, d)``."""
    z = solve_triangular(chol, resid.T, lower=True)
    d = resid.shape[1]
    return (
        -0.5 * np.einsum("ij,ij->j", z, z) - float(np.log(np.diag(chol)).sum()) - 0.5 * d * _LOG_2PI
    )


def _companion(coefs: np.ndarray) -> np.ndarray:
    p, d = coefs.shape[0], coefs.shape[1]
    f = np.zeros((d * p, d * p))
    f[:d] = np.hstack(list(coefs))
    if p > 1:
        f[d:, :-d] = np.eye(d * (p - 1))
    return f


def _spectral_radius(coefs: np.ndarray) -> float:
    return float(np.abs(np.linalg.eigvals(_companion(coefs))).max())


def _stationary_dist(p: np.ndarray) -> np.ndarray:
    """Stationary law of a row-stochastic matrix (least squares on ``pi P = pi``)."""
    m = p.shape[0]
    lhs = np.vstack([p.T - np.eye(m), np.ones((1, m))])
    rhs = np.zeros(m + 1)
    rhs[-1] = 1.0
    pi = np.linalg.lstsq(lhs, rhs, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    return pi / pi.sum()


# ─── numba kernels ───────────────────────────────────────────────────────────


@njit(cache=True)
def _simulate_kernel(states, mu, coefs, chol, eps, lags):  # pragma: no cover - jitted
    """Simulate ``(P, H, 4)`` switching-VAR paths along given state paths.

    ``lags[p, k]`` is ``y_{-1-k}`` of path ``p``; ``eps`` are standard normals.
    """
    n_paths, horizon = states.shape
    order = coefs.shape[1]
    d = mu.shape[1]
    out = np.empty((n_paths, horizon, d))
    for q in range(n_paths):
        for t in range(horizon):
            s = states[q, t]
            for i in range(d):
                acc = mu[s, i]
                for m in range(d):
                    acc += chol[s, i, m] * eps[q, t, m]
                for k in range(order):
                    idx = t - 1 - k
                    for m in range(d):
                        prev = out[q, idx, m] if idx >= 0 else lags[q, k - t, m]
                        acc += coefs[s, k, i, m] * prev
                out[q, t, i] = acc
    return out


@njit(cache=True)
def _draw(prob, u):  # pragma: no cover - jitted
    """Inverse-CDF draw from a pmf; rounding never lands on a zero-mass cell."""
    acc = 0.0
    last = 0
    for i in range(prob.shape[0]):
        if prob[i] > 0.0:
            last = i
            acc += prob[i]
            if u < acc:
                return i
    return last


@njit(cache=True)
def _simulate_states(j0, n0, fresh, delta, trans, haz, u_exit, u_next):  # pragma: no cover
    """State paths ``(P, H)`` of the elapsed-time chain.

    ``fresh``: the first step opens a visit drawn from ``delta``; otherwise the
    chain starts in ``(j0[p], n0[p])`` (the state of the last observed sample)
    and every step, the first included, is a transition.
    """
    n_paths, horizon = u_exit.shape
    dmax = haz.shape[1]
    states = np.empty((n_paths, horizon), dtype=np.int64)
    for q in range(n_paths):
        j = j0[q]
        e = n0[q]
        for h in range(horizon):
            if fresh and h == 0:
                j = _draw(delta, u_next[q, 0])
                e = 0
            elif u_exit[q, h] < haz[j, e]:
                j = _draw(trans[j], u_next[q, h])
                e = 0
            else:
                e = min(e + 1, dmax - 1)
            states[q, h] = j
    return states


#: Reassociation only (SIMD reductions); no finite-math assumptions, since a
#: zero predictive density must still propagate as ``log 0 = -inf``.
_FASTMATH = {"reassoc", "contract"}


@njit(cache=True, fastmath=_FASTMATH)
def _forward(b, delta, trans, haz, alpha, pred, cs):  # pragma: no cover - jitted
    """Scaled forward pass of one sequence over the expanded state ``(j, n)``.

    ``b[t, j]`` are emission densities (any per-time positive factor),
    ``alpha`` a buffer ``(A, M, D)`` indexed ``t % A`` (``A = T`` keeps every
    step, ``A = 2`` only the last).  On return ``alpha[t % A] / cs[t]`` is the
    filtered law at ``t`` (the normalisation is folded into the next step, so
    the stored rows are ``cs[t]`` times the filtered law), ``pred[t]`` is the
    one-step predicted state law and ``cs[t]`` the predictive density of
    ``b[t]`` (the scaling constant).
    """
    n_t, n_s = b.shape
    dmax = haz.shape[1]
    n_buf = alpha.shape[0]
    stay = 1.0 - haz
    exit_mass = np.zeros(n_s)
    inv_prev = 1.0
    for t in range(n_t):
        cur = alpha[t % n_buf]
        prev = alpha[(t - 1) % n_buf]
        nmax = min(dmax, t + 1)
        if t > 0:
            pm = min(dmax, t)
            for i in range(n_s):
                acc = 0.0
                for n in range(pm):
                    acc += prev[i, n] * haz[i, n]
                exit_mass[i] = acc * inv_prev
        total = 0.0
        for j in range(n_s):
            bj = b[t, j]
            if t == 0:
                p0 = delta[j]
            else:
                p0 = 0.0
                for i in range(n_s):
                    p0 += exit_mass[i] * trans[i, j]
            cur[j, 0] = p0 * bj
            pj = p0
            for n in range(1, nmax):
                v = prev[j, n - 1] * stay[j, n - 1] * inv_prev
                pj += v
                cur[j, n] = v * bj
            for n in range(nmax, dmax):
                cur[j, n] = 0.0
            pred[t, j] = pj
            total += pj * bj
        cs[t] = total
        inv_prev = 1.0 / total if total > 0.0 else 0.0


@njit(cache=True, fastmath=_FASTMATH)
def _e_step_kernel(
    b, offsets, delta, trans, haz, r, surv, alpha, pred, cs, gamma, xi, dur, first
):  # pragma: no cover - jitted
    """Forward-backward over every sequence; fills ``gamma`` and accumulates counts.

    ``xi[i, j]``: expected visit exits ``i -> j``; ``dur[j, n]``: expected
    visits of ``j`` lasting ``n + 1`` (completed visits, plus the censored last
    visit spread over ``m >= n`` by ``r_j(m) / surv_j(n)``); ``first[j]``:
    posterior of the first state, summed over sequences.
    """
    n_s = b.shape[1]
    dmax = haz.shape[1]
    stay = 1.0 - haz
    bcur = np.empty((n_s, dmax))
    bnext = np.empty((n_s, dmax))
    g = np.empty(n_s)
    ex = np.empty(n_s)
    for sq in range(offsets.shape[0] - 1):
        lo = offsets[sq]
        hi = offsets[sq + 1]
        n_t = hi - lo
        if n_t == 0:
            continue
        bs = b[lo:hi]
        al = alpha[:n_t]
        _forward(bs, delta, trans, haz, al, pred[lo:hi], cs[lo:hi])
        for t in range(n_t - 1, -1, -1):
            nmax = min(dmax, t + 1)
            a = al[t]
            inv_t = 1.0 / cs[lo + t] if cs[lo + t] > 0.0 else 0.0
            if t == n_t - 1:
                for j in range(n_s):
                    acc = 0.0
                    gacc = 0.0
                    for m in range(dmax):
                        if m < nmax:
                            bcur[j, m] = 1.0
                            gacc += a[j, m]
                            if surv[j, m] > 0.0:
                                acc += a[j, m] * inv_t / surv[j, m]
                        dur[j, m] += r[j, m] * acc
                    gamma[lo + t, j] = gacc * inv_t
            else:
                inv = 1.0 / cs[lo + t + 1] if cs[lo + t + 1] > 0.0 else 0.0
                b1 = bs[t + 1]
                for j in range(n_s):
                    acc = 0.0
                    for i in range(n_s):
                        acc += trans[j, i] * b1[i] * bnext[i, 0]
                    g[j] = acc * inv
                # elapsed index dmax - 1 has no successor: it can only exit
                n_inner = min(nmax, dmax - 1)
                for j in range(n_s):
                    keep = b1[j] * inv
                    gj = g[j]
                    e_sum = 0.0
                    gacc = 0.0
                    for n in range(n_inner):
                        h = haz[j, n]
                        v = stay[j, n] * keep * bnext[j, n + 1] + h * gj
                        bcur[j, n] = v
                        an = a[j, n] * inv_t
                        e = an * h
                        e_sum += e
                        dur[j, n] += e * gj
                        gacc += an * v
                    if nmax == dmax:
                        n = dmax - 1
                        h = haz[j, n]
                        v = h * gj
                        bcur[j, n] = v
                        an = a[j, n] * inv_t
                        e = an * h
                        e_sum += e
                        dur[j, n] += e * gj
                        gacc += an * v
                    ex[j] = e_sum
                    gamma[lo + t, j] = gacc
                for i in range(n_s):
                    for j in range(n_s):
                        xi[i, j] += ex[i] * trans[i, j] * b1[j] * bnext[j, 0] * inv
            if t == 0:
                for j in range(n_s):
                    first[j] += gamma[lo, j]
            tmp = bcur
            bcur = bnext
            bnext = tmp


@njit(cache=True)
def _lasso_cd(gram, cross, coef, lam, n_unpen, tol, max_sweeps):  # pragma: no cover
    """Coordinate descent on ``b' G b - 2 h' b + lam ||b[n_unpen:]||_1`` per column."""
    k, d = cross.shape
    for c in range(d):
        for _ in range(max_sweeps):
            worst = 0.0
            for i in range(k):
                gii = gram[i, i]
                if gii <= 0.0:
                    coef[i, c] = 0.0
                    continue
                z = cross[i, c]
                for m in range(k):
                    if m != i:
                        z -= gram[i, m] * coef[m, c]
                if i < n_unpen:
                    new = z / gii
                else:
                    mag = abs(z) - 0.5 * lam
                    new = 0.0 if mag <= 0.0 else np.sign(z) * mag / gii
                change = abs(new - coef[i, c]) * np.sqrt(gii)
                worst = max(worst, change)
                coef[i, c] = new
            if worst <= tol:
                break
    return coef


# ─── regularised estimators (public, the paper's M-step pieces) ──────────────


def _lasso_gram(gram: np.ndarray, cross: np.ndarray, lam_a: float, n_unpen: int) -> np.ndarray:
    coef = np.linalg.lstsq(gram, cross, rcond=None)[0]
    if lam_a <= 0.0:
        return coef
    tol = 1e-10 * float(np.sqrt(max(float(np.max(np.diag(gram))), 1e-300)))
    return _lasso_cd(
        np.ascontiguousarray(gram),
        np.ascontiguousarray(cross),
        np.ascontiguousarray(coef),
        float(lam_a),
        n_unpen,
        tol,
        100_000,
    )


def lasso_wls(
    x: np.ndarray, y: np.ndarray, w: np.ndarray, lam_a: float, *, n_unpenalized: int = 0
) -> np.ndarray:
    """``(k, d)`` minimiser of ``sum_t w_t ||y_t - B' x_t||^2 + lam_a ||B[n_unpenalized:]||_1``.

    Weighted LASSO on the Gram form (coordinate descent warm-started from the
    plain weighted least squares, which ``lam_a = 0`` returns unchanged); the
    first ``n_unpenalized`` regressors (the intercept) are not penalised.
    """
    xw = x * w[:, None]
    return _lasso_gram(xw.T @ x, xw.T @ y, lam_a, n_unpenalized)


def shrink_covariance(s: np.ndarray, lam_sigma: float, floor: float = 0.0) -> np.ndarray:
    """``(S + lam_sigma c I) / (1 + lam_sigma) + floor I`` with ``c = tr(S) / d``."""
    d = s.shape[0]
    c = float(np.trace(s)) / d
    return (s + lam_sigma * c * np.eye(d)) / (1.0 + lam_sigma) + floor * np.eye(d)


# ─── plain VAR(p) ────────────────────────────────────────────────────────────


@dataclass(eq=False)
class VarModel:
    """Gaussian VAR(p) ``y_t = c + sum_k A_k y_{t-k} + e_t``, ``e_t ~ N(0, Sigma)``.

    Attributes:
        coefs: ``(p, 4, 4)``, ``coefs[k-1] = A_k``.
        intercept: ``(4,)`` ``c`` (zeros when fitted without one).
        sigma: ``(4, 4)`` innovation covariance.
        has_intercept: whether ``c`` was estimated (it enters ``n_params``).
        shrink_factor: ``g`` of the stability guard ``A_k <- g^k A_k``; 1.0
            when the OLS fit was already stable.
    """

    name: str
    rate_hz: float
    coefs: np.ndarray
    intercept: np.ndarray
    sigma: np.ndarray
    has_intercept: bool = False
    shrink_factor: float = 1.0
    n_params: int = field(init=False)
    _chol: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.coefs = np.asarray(self.coefs, dtype=np.float64).reshape(-1, NUM_ROTORS, NUM_ROTORS)
        self.intercept = np.asarray(self.intercept, dtype=np.float64).reshape(NUM_ROTORS)
        self._chol, self.sigma = _safe_cholesky(np.asarray(self.sigma, dtype=np.float64))
        self.n_params = 16 * self.order + 10 + 4 * int(self.has_intercept)

    @property
    def order(self) -> int:
        return int(self.coefs.shape[0])

    @property
    def shrunk(self) -> bool:
        """Whether the stability guard rescaled the OLS coefficients."""
        return self.shrink_factor < 1.0

    @property
    def spectral_radius(self) -> float:
        """Largest modulus of the companion matrix's eigenvalues."""
        return _spectral_radius(self.coefs)

    def companion(self) -> np.ndarray:
        """``(4p, 4p)`` companion matrix of the state ``[y_{t-1}, ..., y_{t-p}]``."""
        return _companion(self.coefs)

    @property
    def stationary_mean(self) -> np.ndarray:
        return np.linalg.solve(np.eye(NUM_ROTORS) - self.coefs.sum(axis=0), self.intercept)

    def _companion_cov(self) -> np.ndarray:
        dim = NUM_ROTORS * self.order
        q = np.zeros((dim, dim))
        q[:NUM_ROTORS, :NUM_ROTORS] = self.sigma
        return solve_discrete_lyapunov(self.companion(), q)

    @property
    def stationary_cov(self) -> np.ndarray:
        """``(4, 4)`` stationary covariance (discrete Lyapunov equation)."""
        return self._companion_cov()[:NUM_ROTORS, :NUM_ROTORS]

    def logpdf(self, block: np.ndarray, burn: int) -> np.ndarray:
        y = _check_block(block)
        _check_burn(y, burn, self.order)
        resid = y[:, burn:].T - _cond_mean(y, self.coefs, self.intercept, burn)
        return _gauss_logpdf(resid, self._chol)

    def predict_mean(self, block: np.ndarray, burn: int) -> np.ndarray:
        """``(4, n - burn)`` one-step predictive means ``E[y_t | y_{<t}]``."""
        y = _check_block(block)
        _check_burn(y, burn, self.order)
        return _cond_mean(y, self.coefs, self.intercept, burn).T

    def _simulate(self, eps: np.ndarray, lags: np.ndarray) -> np.ndarray:
        states = np.zeros(eps.shape[:2], dtype=np.int64)
        return _simulate_kernel(
            states, self.intercept[None], self.coefs[None], self._chol[None], eps, lags
        )

    @cached_property
    def _stationary_root(self) -> tuple[np.ndarray, np.ndarray]:
        """Mean and a square root of the stationary law of ``[y_{t-1}, ..., y_{t-p}]``."""
        cov = self._companion_cov()
        vals, vecs = np.linalg.eigh(0.5 * (cov + cov.T))
        root = vecs * np.sqrt(np.clip(vals, 0.0, None))
        return np.tile(self.stationary_mean, self.order), root

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        """Stationary draw: the initial lags come from the exact stationary law
        (an infinitely long burn-in), so nothing needs discarding."""
        mean, root = self._stationary_root
        x0 = mean + root @ rng.standard_normal(root.shape[1])
        eps = rng.standard_normal((1, n, NUM_ROTORS))
        y = self._simulate(eps, x0.reshape(1, self.order, NUM_ROTORS))
        return np.ascontiguousarray(y[0].T)

    def forecast(
        self, history: np.ndarray, horizon: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        hist = _check_block(history)
        p = self.order
        if hist.shape[1] < p:
            raise ValueError(f"history has {hist.shape[1]} samples, the VAR needs {p}")
        lags = np.broadcast_to(hist[:, ::-1][:, :p].T, (n_paths, p, NUM_ROTORS)).copy()
        eps = rng.standard_normal((n_paths, horizon, NUM_ROTORS))
        return self._simulate(eps, lags).transpose(0, 2, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "var",
            "name": self.name,
            "rate_hz": float(self.rate_hz),
            "coefs": self.coefs.tolist(),
            "intercept": self.intercept.tolist(),
            "sigma": self.sigma.tolist(),
            "has_intercept": bool(self.has_intercept),
            "shrink_factor": float(self.shrink_factor),
            "n_params": int(self.n_params),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> VarModel:
        return cls(
            name=str(payload["name"]),
            rate_hz=float(payload["rate_hz"]),
            coefs=np.asarray(payload["coefs"], dtype=np.float64),
            intercept=np.asarray(payload["intercept"], dtype=np.float64),
            sigma=np.asarray(payload["sigma"], dtype=np.float64),
            has_intercept=bool(payload["has_intercept"]),
            shrink_factor=float(payload["shrink_factor"]),
        )


def _ols(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    b = np.linalg.lstsq(x, y, rcond=None)[0]
    resid = y - x @ b
    return b, resid.T @ resid / len(y)


def fit_var(
    blocks: Any,
    rate_hz: float,
    *,
    order: int | None = None,
    p_max: int = 12,
    intercept: bool = False,
    name: str = "var",
) -> VarModel:
    """Pooled-OLS Gaussian VAR(p) over blocks (MLE innovation covariance).

    ``order=None`` picks ``p`` in ``1..p_max`` by BIC
    ``N log det Sigma_p + n_params(p) log N``, every candidate scored on the
    same rows (``t >= p_max`` of every block); the chosen order is then refit
    on all rows ``t >= p``.  A fit whose companion spectral radius reaches
    :data:`STABLE_RADIUS` is shrunk ``A_k <- g^k A_k``, which scales every
    companion eigenvalue by ``g``, with ``g`` just under ``STABLE_RADIUS / rho``.
    """
    ys = _check_blocks(blocks)
    off = int(intercept)
    if order is None:
        if p_max < 1:
            raise ValueError("p_max must be >= 1")
        x_all, y_all, _ = _stack_design(ys, p_max, p_max, intercept)
        n_rows = len(y_all)
        best_bic, order = np.inf, 1
        for p in range(1, p_max + 1):
            _, sig = _ols(x_all[:, : off + 4 * p], y_all)
            n_par = 16 * p + 10 + 4 * off
            bic = n_rows * float(np.linalg.slogdet(sig)[1]) + n_par * np.log(n_rows)
            if bic < best_bic:
                best_bic, order = bic, p
    if order < 1:
        raise ValueError("order must be >= 1")
    x, y, _ = _stack_design(ys, order, order, intercept)
    b, sigma = _ols(x, y)
    coefs, c = _coefs_from_b(b, order, intercept)
    rho = _spectral_radius(coefs)
    g = 1.0
    if rho >= STABLE_RADIUS:
        g = STABLE_RADIUS / rho * (1.0 - 1e-6)
        coefs = coefs * (g ** np.arange(1, order + 1))[:, None, None]
    return VarModel(
        name=name,
        rate_hz=float(rate_hz),
        coefs=coefs,
        intercept=c,
        sigma=sigma,
        has_intercept=intercept,
        shrink_factor=g,
    )


# ─── regime-switching VAR(p): Markov (geometric) and semi-Markov (explicit) ──


def _surv_haz(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Survival ``S_j(n) = P(dur >= n + 1)`` and hazard ``r / S`` (1 where S = 0)."""
    surv = np.cumsum(r[:, ::-1], axis=1)[:, ::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        haz = np.where(surv > 0.0, r / np.where(surv > 0.0, surv, 1.0), 1.0)
    return np.ascontiguousarray(surv), np.ascontiguousarray(np.clip(haz, 0.0, 1.0))


@dataclass(eq=False)
class SwitchingVarModel:
    """M-state switching VAR(p) ``y_t = mu_j + sum_k A_kj y_{t-k} + e_t``, ``e_t ~ N(0, Sigma_j)``.

    Attributes:
        durations: ``'geometric'`` (Hamilton 1989: ``trans`` has
            self-transitions) or ``'explicit'`` (Xu and Liu 2021: ``trans`` is
            ``Q`` without self-transitions, ``dur_pmf[j, n-1] = r_j(n)``).
        mu: ``(M, 4)``; coefs ``(M, p, 4, 4)``; sigma ``(M, 4, 4)``, original units.
        delta: ``(M,)`` state law of a fresh visit at the start of a sequence.
        scale: the global scale the fit divided the data by.
        lam_a, lam_sigma, sigma_floor: the fit's regularisation (scale-free).
        loglik_: log-likelihood (original units) of the training blocks.
        loglik_trace_: log-likelihood after every E-step of the kept restart.
    """

    name: str
    rate_hz: float
    durations: _DurationLaw
    mu: np.ndarray
    coefs: np.ndarray
    sigma: np.ndarray
    delta: np.ndarray
    trans: np.ndarray
    dur_pmf: np.ndarray | None = None
    has_intercept: bool = True
    scale: float = 1.0
    lam_a: float = 0.0
    lam_sigma: float = 0.0
    sigma_floor: float = 0.0
    loglik_: float = float("nan")
    n_iter_: int = 0
    converged_: bool = False
    loglik_trace_: list[float] = field(default_factory=list)
    n_params: int = field(init=False)
    _chol: np.ndarray = field(init=False, repr=False)
    _haz: np.ndarray = field(init=False, repr=False)
    _r: np.ndarray = field(init=False, repr=False)
    _surv: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        m = int(np.asarray(self.delta).shape[0])
        self.mu = np.asarray(self.mu, dtype=np.float64).reshape(m, NUM_ROTORS)
        self.coefs = np.asarray(self.coefs, dtype=np.float64).reshape(m, -1, NUM_ROTORS, NUM_ROTORS)
        self.delta = np.asarray(self.delta, dtype=np.float64)
        self.trans = np.ascontiguousarray(self.trans, dtype=np.float64)
        sig = np.asarray(self.sigma, dtype=np.float64).reshape(m, NUM_ROTORS, NUM_ROTORS)
        chols = [_safe_cholesky(s) for s in sig]
        self._chol = np.stack([c for c, _ in chols])
        self.sigma = np.stack([s for _, s in chols])
        if self.durations == "geometric":
            self.dur_pmf = None
            self._r = np.ones((m, 1))
            self._surv, self._haz = np.ones((m, 1)), np.ones((m, 1))
        elif self.durations == "explicit":
            if m < 2:
                raise ValueError("explicit durations need >= 2 states (Q has no self-transitions)")
            if self.dur_pmf is None:
                raise ValueError("explicit durations need dur_pmf")
            self.dur_pmf = np.ascontiguousarray(self.dur_pmf, dtype=np.float64).reshape(m, -1)
            self._r = self.dur_pmf
            self._surv, self._haz = _surv_haz(self.dur_pmf)
        else:
            raise ValueError(f"unknown durations {self.durations!r}")
        p, d_max = self.order, self.max_dur
        common = (m - 1) + 4 * m * int(self.has_intercept) + 10 * m + 16 * p * m
        if self.durations == "explicit":
            self.n_params = common + m * (d_max - 1) + m * (m - 2)
        else:
            self.n_params = common + m * (m - 1)

    # ── structure ──

    @property
    def n_states(self) -> int:
        return int(self.delta.shape[0])

    @property
    def order(self) -> int:
        return int(self.coefs.shape[1])

    @property
    def max_dur(self) -> int:
        """``D`` (1 for geometric durations)."""
        return int(self._haz.shape[1])

    @property
    def mean_dwell_s(self) -> np.ndarray:
        """``(M,)`` mean visit length per state, seconds (inf for an absorbing state)."""
        if self.dur_pmf is not None:
            n = np.arange(1, self.max_dur + 1)
            return (self.dur_pmf @ n) / self.dur_pmf.sum(axis=1) / self.rate_hz
        stay = np.diag(self.trans)
        with np.errstate(divide="ignore"):
            return np.where(stay < 1.0, 1.0 / (1.0 - stay), np.inf) / self.rate_hz

    @property
    def stationary_occupancy(self) -> np.ndarray:
        """``(M,)`` long-run fraction of samples spent in each state."""
        pi = _stationary_dist(self.trans)
        if self.dur_pmf is None:
            return pi
        occ = pi * self.mean_dwell_s
        return occ / occ.sum()

    # ── likelihood ──

    def _log_emissions(self, y: np.ndarray) -> np.ndarray:
        """``(n - order, M)`` state-conditional log densities for ``t = order..n-1``."""
        p = self.order
        out = np.empty((y.shape[1] - p, self.n_states))
        for j in range(self.n_states):
            resid = y[:, p:].T - _cond_mean(y, self.coefs[j], self.mu[j], p)
            out[:, j] = _gauss_logpdf(resid, self._chol[j])
        return out

    def _filter(
        self, y: np.ndarray, keep_all: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Forward pass from ``delta`` at ``t = order``: log predictive densities
        ``(n - order,)``, predicted state laws ``(n - order, M)`` and the alpha buffer."""
        logb = self._log_emissions(y)
        n_t = len(logb)
        shift = logb.max(axis=1)
        b = np.exp(logb - shift[:, None])
        alpha = np.empty((n_t if keep_all else 2, self.n_states, self.max_dur))
        pred = np.empty((n_t, self.n_states))
        cs = np.empty(n_t)
        _forward(b, self.delta, self.trans, self._haz, alpha, pred, cs)
        with np.errstate(divide="ignore"):
            return np.log(cs) + shift, pred, alpha

    def logpdf(self, block: np.ndarray, burn: int) -> np.ndarray:
        y = _check_block(block)
        _check_burn(y, burn, self.order)
        if y.shape[1] == self.order:
            return np.empty(0)
        logc, _, _ = self._filter(y)
        return logc[burn - self.order :]

    def predict_mean(self, block: np.ndarray, burn: int) -> np.ndarray:
        """``(4, n - burn)`` one-step predictive means ``E[y_t | y_{<t}]``."""
        y = _check_block(block)
        _check_burn(y, burn, self.order)
        p = self.order
        if y.shape[1] == p:
            return np.empty((NUM_ROTORS, 0))
        _, pred, _ = self._filter(y)
        means = np.stack(
            [_cond_mean(y, self.coefs[j], self.mu[j], p) for j in range(self.n_states)], axis=1
        )
        return np.einsum("tj,tjd->dt", pred, means)[:, burn - p :]

    def posterior(self, block: np.ndarray) -> np.ndarray:
        """``(M, n - order)`` smoothed state probabilities ``P(S_t = j | y)``."""
        y = _check_block(block)
        e = _run_e_step(
            self._log_emissions(y), np.array([0, y.shape[1] - self.order], dtype=np.int64), self
        )
        return e.gamma.T

    # ── simulation ──

    def _simulate(
        self,
        rng: np.random.Generator,
        horizon: int,
        lags: np.ndarray,
        j0: np.ndarray | None = None,
        n0: np.ndarray | None = None,
    ) -> np.ndarray:
        n_paths = lags.shape[0]
        fresh = j0 is None
        if j0 is None or n0 is None:
            j0 = n0 = np.zeros(n_paths, dtype=np.int64)
        u_exit = rng.random((n_paths, horizon))
        u_next = rng.random((n_paths, horizon))
        states = _simulate_states(j0, n0, fresh, self.delta, self.trans, self._haz, u_exit, u_next)
        eps = rng.standard_normal((n_paths, horizon, NUM_ROTORS))
        return _simulate_kernel(states, self.mu, self.coefs, self._chol, eps, lags)

    def _burn_in(self) -> int:
        dwell = np.minimum(self.mean_dwell_s * self.rate_hz, 1e5)
        return int(max(round(SAMPLE_BURN_S * self.rate_hz), 10 * np.ceil(dwell.max())))

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        start: Literal["stationary", "initial"] = "stationary",
    ) -> np.ndarray:
        """``(4, n)`` realisation.

        ``'stationary'`` opens a visit from ``delta`` with zero lags, runs
        ``_burn_in()`` samples and returns the tail; ``'initial'`` returns the
        samples from that fresh start on (a whole flight starting at rest).
        """
        if start not in ("stationary", "initial"):
            raise ValueError(f"unknown start {start!r}")
        burn = self._burn_in() if start == "stationary" else 0
        lags = np.zeros((1, self.order, NUM_ROTORS))
        y = self._simulate(rng, n + burn, lags)
        return np.ascontiguousarray(y[0, burn:].T)

    def forecast(
        self, history: np.ndarray, horizon: int, n_paths: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Filter the history, draw ``(state, elapsed)`` from the filtered law of
        its last sample, then simulate forward (a history of exactly ``order``
        samples has no emission yet: the paths open a fresh visit from ``delta``)."""
        hist = _check_block(history)
        p, h = self.order, hist.shape[1]
        if h < p:
            raise ValueError(f"history has {h} samples, the model needs {p}")
        lags = np.broadcast_to(hist[:, ::-1][:, :p].T, (n_paths, p, NUM_ROTORS)).copy()
        if h == p:
            out = self._simulate(rng, horizon, lags)
        else:
            _, _, alpha = self._filter(hist)
            law = alpha[(h - p - 1) % alpha.shape[0]].ravel()
            idx = rng.choice(law.size, size=n_paths, p=law / law.sum())
            out = self._simulate(rng, horizon, lags, idx // self.max_dur, idx % self.max_dur)
        return out.transpose(0, 2, 1)

    # ── persistence ──

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": "switching_var",
            "name": self.name,
            "rate_hz": float(self.rate_hz),
            "durations": self.durations,
            "mu": self.mu.tolist(),
            "coefs": self.coefs.tolist(),
            "sigma": self.sigma.tolist(),
            "delta": self.delta.tolist(),
            "trans": self.trans.tolist(),
            "dur_pmf": None if self.dur_pmf is None else self.dur_pmf.tolist(),
            "has_intercept": bool(self.has_intercept),
            "scale": float(self.scale),
            "lam_a": float(self.lam_a),
            "lam_sigma": float(self.lam_sigma),
            "sigma_floor": float(self.sigma_floor),
            "loglik": float(self.loglik_),
            "n_iter": int(self.n_iter_),
            "converged": bool(self.converged_),
            "loglik_trace": [float(v) for v in self.loglik_trace_],
            "n_params": int(self.n_params),
            "stationary_occupancy": self.stationary_occupancy.tolist(),
            "mean_dwell_s": [float(v) for v in self.mean_dwell_s],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> SwitchingVarModel:
        pmf = payload.get("dur_pmf")
        durations = str(payload["durations"])
        if durations not in ("geometric", "explicit"):
            raise ValueError(f"unknown durations {durations!r}")
        return cls(
            name=str(payload["name"]),
            rate_hz=float(payload["rate_hz"]),
            durations="explicit" if durations == "explicit" else "geometric",
            mu=np.asarray(payload["mu"], dtype=np.float64),
            coefs=np.asarray(payload["coefs"], dtype=np.float64),
            sigma=np.asarray(payload["sigma"], dtype=np.float64),
            delta=np.asarray(payload["delta"], dtype=np.float64),
            trans=np.asarray(payload["trans"], dtype=np.float64),
            dur_pmf=None if pmf is None else np.asarray(pmf, dtype=np.float64),
            has_intercept=bool(payload["has_intercept"]),
            scale=float(payload["scale"]),
            lam_a=float(payload["lam_a"]),
            lam_sigma=float(payload["lam_sigma"]),
            sigma_floor=float(payload["sigma_floor"]),
            loglik_=float(payload["loglik"]),
            n_iter_=int(payload["n_iter"]),
            converged_=bool(payload["converged"]),
            loglik_trace_=[float(v) for v in payload.get("loglik_trace", [])],
        )


# ─── EM ──────────────────────────────────────────────────────────────────────


@dataclass
class _EStep:
    loglik: float
    gamma: np.ndarray  # (T, M)
    xi: np.ndarray  # (M, M)
    dur: np.ndarray  # (M, D)
    first: np.ndarray  # (M,)


def _run_e_step(
    logb: np.ndarray,
    offsets: np.ndarray,
    model: SwitchingVarModel,
    alpha: np.ndarray | None = None,
) -> _EStep:
    n_t, m = logb.shape
    d_max = model.max_dur
    longest = int(np.diff(offsets).max())
    if alpha is None or alpha.shape[0] < longest or alpha.shape[2] != d_max:
        alpha = np.empty((longest, m, d_max))
    shift = logb.max(axis=1)
    b = np.exp(logb - shift[:, None])
    pred = np.empty((n_t, m))
    cs = np.empty(n_t)
    gamma = np.empty((n_t, m))
    xi = np.zeros((m, m))
    dur = np.zeros((m, d_max))
    first = np.zeros(m)
    _e_step_kernel(
        b,
        offsets,
        model.delta,
        model.trans,
        model._haz,
        model._r,
        model._surv,
        alpha,
        pred,
        cs,
        gamma,
        xi,
        dur,
        first,
    )
    with np.errstate(divide="ignore"):
        loglik = float(np.log(cs).sum() + shift.sum())
    return _EStep(loglik, gamma, xi, dur, first)


def _emission_step(
    xy: np.ndarray,
    gamma: np.ndarray,
    lam_a: float,
    lam_sigma: float,
    sigma_floor: float,
    n_unpen: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-state penalised WLS on ``xy = [X, Y]``: ``B (M, k, 4)`` and ``Sigma (M, 4, 4)``.

    One weighted Gram ``[X, Y]' W [X, Y]`` per state gives the normal equations
    and the weighted residual covariance ``(Y'WY - B'H - H'B + B'GB) / sum w``.
    """
    k = xy.shape[1] - NUM_ROTORS
    bs, sigmas = [], []
    for j in range(gamma.shape[1]):
        w = gamma[:, j]
        full = (xy * w[:, None]).T @ xy
        gram, cross, yy = full[:k, :k], full[:k, k:], full[k:, k:]
        coef = _lasso_gram(gram, cross, lam_a, n_unpen)
        bh = coef.T @ cross
        s = (yy - bh - bh.T + coef.T @ gram @ coef) / max(float(w.sum()), 1e-300)
        bs.append(coef)
        sigmas.append(shrink_covariance(0.5 * (s + s.T), lam_sigma, sigma_floor))
    return np.stack(bs), np.stack(sigmas)


def _geometric_pmf(mean_run: float, d_max: int) -> np.ndarray:
    stay = 1.0 - 1.0 / mean_run
    r = (1.0 - stay) * stay ** np.arange(d_max)
    r[-1] = stay ** (d_max - 1)
    return r


def _init_labels(
    k: int,
    zs: list[np.ndarray],
    offsets: np.ndarray,
    order: int,
    n_states: int,
    rate_hz: float,
    run_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Restart 0: quantiles of the rolling RMS; later restarts: random
    piecewise-constant labelings with runs of up to ``2 run_len`` samples."""
    n_t = int(offsets[-1])
    if k == 0:
        window = max(3, int(round(0.5 * rate_hz)))
        rms = np.concatenate(
            [
                np.log(uniform_filter1d((z[:, order:] ** 2).sum(axis=0), window) + 1e-12)
                for z in zs
                if z.shape[1] > order
            ]
        )
        cuts = np.quantile(rms, np.arange(1, n_states) / n_states)
        return np.searchsorted(cuts, rms).astype(np.int64)
    labels = np.empty(n_t, dtype=np.int64)
    for lo, hi in itertools.pairwise(offsets):
        t = int(lo)
        while t < hi:
            run = int(rng.integers(1, 2 * run_len + 1))
            labels[t : min(t + run, int(hi))] = rng.integers(n_states)
            t += run
    return labels


def _mean_runs(labels: np.ndarray, offsets: np.ndarray, n_states: int) -> np.ndarray:
    counts, runs = np.zeros(n_states), np.zeros(n_states)
    for lo, hi in itertools.pairwise(offsets):
        lab = labels[lo:hi]
        if len(lab) == 0:
            continue
        starts = np.concatenate([[True], lab[1:] != lab[:-1]])
        np.add.at(runs, lab[starts], 1.0)
        np.add.at(counts, lab, 1.0)
    return np.where(runs > 0, counts / np.maximum(runs, 1.0), 1.0)


def _l1(model_b: np.ndarray, n_unpen: int) -> float:
    return float(np.abs(model_b[:, n_unpen:]).sum())


def fit_switching_var(
    blocks: Any,
    rate_hz: float,
    *,
    n_states: int = 2,
    order: int = 1,
    durations: _DurationLaw = "geometric",
    max_dur_s: float = 5.0,
    lam_a: float = 0.0,
    lam_sigma: float = 0.0,
    intercept: bool = True,
    sigma_floor: float = 0.0,
    n_init: int = 3,
    max_iter: int = 200,
    tol: float = 1e-6,
    seed: int = 0,
    name: str | None = None,
    init_labels: Any = None,
) -> SwitchingVarModel:
    """EM fit of the Markov-switching VAR / VAR(p)-HSMM (see the module docstring).

    Each restart iterates E-step, convergence test, M-step; it stops when the
    penalised log-likelihood ``loglik - lam_a ||a||_1`` (scaled data) rises by
    less than ``tol`` times its magnitude, and the restart with the best
    penalised log-likelihood is kept.  ``D = round(max_dur_s * rate_hz)``.

    ``init_labels`` (one integer array ``(n,)`` per block, values in
    ``0..n_states-1``) replaces the rolling-RMS labelling of restart 0: a
    known segmentation, such as the phases of a whole flight, to start EM at.
    """
    if durations not in ("geometric", "explicit"):
        raise ValueError(f"unknown durations {durations!r}")
    if n_states < 1 or order < 1 or n_init < 1:
        raise ValueError("n_states, order and n_init must be >= 1")
    explicit = durations == "explicit"
    if explicit and n_states < 2:
        raise ValueError("explicit durations need >= 2 states")
    ys = _check_blocks(blocks)
    scale = float(np.concatenate([y.ravel() for y in ys]).std())
    if not scale > 0.0:
        raise ValueError("training blocks are constant")
    zs = [y / scale for y in ys]
    x, y, offsets = _stack_design(zs, order, order, intercept)
    xy = np.hstack([x, y])
    n_unpen = int(intercept)
    d_max = max(1, int(round(max_dur_s * rate_hz))) if explicit else 1
    run_len = max(2, min(int(round(rate_hz)), d_max) if explicit else int(round(rate_hz)))
    m = n_states
    rng = np.random.default_rng(seed)
    alpha = np.empty((int(np.diff(offsets).max()), m, d_max))
    log_scale = float(offsets[-1]) * NUM_ROTORS * np.log(scale)

    def build(
        b: np.ndarray,
        sig: np.ndarray,
        delta: np.ndarray,
        trans: np.ndarray,
        pmf: np.ndarray | None,
    ) -> SwitchingVarModel:
        coefs_mu = [_coefs_from_b(b[j], order, intercept) for j in range(m)]
        return SwitchingVarModel(
            name="",
            rate_hz=float(rate_hz),
            durations=durations,
            mu=np.stack([c for _, c in coefs_mu]),
            coefs=np.stack([a for a, _ in coefs_mu]),
            sigma=sig,
            delta=delta,
            trans=trans,
            dur_pmf=pmf,
            has_intercept=intercept,
        )

    best: tuple[float, SwitchingVarModel, list[float], int, bool] | None = None
    for k in range(n_init):
        if k == 0 and init_labels is not None:
            labels = np.concatenate(
                [np.asarray(lab, dtype=np.int64)[order:] for lab in init_labels]
            )
            if labels.size != int(offsets[-1]) or labels.min() < 0 or labels.max() >= m:
                raise ValueError("init_labels must give one state in 0..n_states-1 per sample")
        else:
            labels = _init_labels(k, zs, offsets, order, m, rate_hz, run_len, rng)
        b, sig = _emission_step(xy, np.eye(m)[labels], lam_a, lam_sigma, sigma_floor, n_unpen)
        runs = np.clip(_mean_runs(labels, offsets, m), 2.0, d_max if explicit else np.inf)
        delta = np.full(m, 1.0 / m)
        if explicit:
            trans = (np.ones((m, m)) - np.eye(m)) / (m - 1)
            pmf: np.ndarray | None = np.stack([_geometric_pmf(r_, d_max) for r_ in runs])
        else:
            stay = 1.0 - 1.0 / runs
            trans = (
                np.diag(stay) + ((1.0 - stay) / (m - 1))[:, None] * (1.0 - np.eye(m))
                if m > 1
                else np.ones((1, 1))
            )
            pmf = None
        model = build(b, sig, delta, trans, pmf)
        trace: list[float] = []
        prev = -np.inf
        converged, n_iter = False, 0
        pll = -np.inf
        for it in range(max_iter + 1):
            logb = np.column_stack([_gauss_logpdf(y - x @ b[j], model._chol[j]) for j in range(m)])
            e = _run_e_step(logb, offsets, model, alpha)
            pll = e.loglik - lam_a * _l1(b, n_unpen)
            trace.append(e.loglik - log_scale)
            if it > 0 and pll - prev < tol * abs(prev):
                converged = True
                break
            if it == max_iter:
                break
            prev = pll
            # M-step
            b, sig = _emission_step(xy, e.gamma, lam_a, lam_sigma, sigma_floor, n_unpen)
            delta = e.first / e.first.sum()
            rows = e.xi.sum(axis=1, keepdims=True)
            trans = np.where(rows > 0.0, e.xi / np.where(rows > 0.0, rows, 1.0), model.trans)
            if explicit:
                tot = e.dur.sum(axis=1, keepdims=True)
                pmf = np.where(tot > 0.0, e.dur / np.where(tot > 0.0, tot, 1.0), model._r)
            model = build(b, sig, delta, trans, pmf)
            n_iter = it + 1
        if best is None or pll > best[0]:
            best = (pll, model, trace, n_iter, converged)

    assert best is not None
    _, model, trace, n_iter, converged = best
    return SwitchingVarModel(
        name=name if name is not None else ("var_hsmm" if explicit else "ms_var"),
        rate_hz=float(rate_hz),
        durations=durations,
        mu=model.mu * scale,
        coefs=model.coefs,
        sigma=model.sigma * scale**2,
        delta=model.delta,
        trans=model.trans,
        dur_pmf=model.dur_pmf,
        has_intercept=intercept,
        scale=scale,
        lam_a=float(lam_a),
        lam_sigma=float(lam_sigma),
        sigma_floor=float(sigma_floor),
        loglik_=trace[-1],
        n_iter_=n_iter,
        converged_=converged,
        loglik_trace_=trace,
    )


# ─── the paper's selection criterion ─────────────────────────────────────────


def msfe(model: VarModel | SwitchingVarModel, blocks: Any, burn: int) -> float:
    """One-step mean-square forecast error ``mean_t ||E[y_t | y_{<t}] - y_t||^2``
    over the samples ``t >= burn`` of every block (squared error summed over
    the four rotors, averaged over the scored samples; Xu and Liu 2021)."""
    total, count = 0.0, 0
    for y in _check_blocks(blocks):
        if y.shape[1] <= burn:
            continue
        err = model.predict_mean(y, burn) - y[:, burn:]
        total += float((err**2).sum())
        count += err.shape[1]
    if count == 0:
        raise ValueError(f"no block is longer than burn={burn}")
    return total / count


def select_regularization(
    train_blocks: Any,
    val_blocks: Any,
    rate_hz: float,
    *,
    grid_a: Any,
    grid_sigma: Any,
    burn: int | None = None,
    **fit_kwargs: Any,
) -> tuple[SwitchingVarModel, list[dict[str, float]]]:
    """Grid search of ``(lam_a, lam_sigma)`` minimising validation MSFE.

    Every grid point is fitted on ``train_blocks`` with ``fit_kwargs``; MSFE is
    scored on ``val_blocks`` past ``burn`` (default
    ``max(burn_samples(rate_hz), order)``).  Returns the best model and one
    record ``{lam_a, lam_sigma, msfe, loglik}`` per grid point.
    """
    records: list[dict[str, float]] = []
    best: tuple[float, SwitchingVarModel] | None = None
    for la in grid_a:
        for ls in grid_sigma:
            model = fit_switching_var(
                train_blocks, rate_hz, lam_a=float(la), lam_sigma=float(ls), **fit_kwargs
            )
            b = burn if burn is not None else max(burn_samples(rate_hz), model.order)
            score = msfe(model, val_blocks, b)
            records.append(
                {
                    "lam_a": float(la),
                    "lam_sigma": float(ls),
                    "msfe": score,
                    "loglik": model.loglik_,
                }
            )
            if best is None or score < best[0]:
                best = (score, model)
    if best is None:
        raise ValueError("empty regularisation grid")
    return best[1], records


__all__ = [
    "SAMPLE_BURN_S",
    "STABLE_RADIUS",
    "SwitchingVarModel",
    "VarModel",
    "fit_switching_var",
    "fit_var",
    "lasso_wls",
    "msfe",
    "select_regularization",
    "shrink_covariance",
]
