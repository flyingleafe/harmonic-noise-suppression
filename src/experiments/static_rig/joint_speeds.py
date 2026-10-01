"""Joint Bretthorst search for the shaft speeds of R rotors in a stationary
recording (campaign doc ``docs/experiments/static-rig-profiles.md``, stage A).

Per analysis window every channel gives a whitened Hann periodogram
``z = P / N`` (``N`` the running-median floor as a mean noise ordinate). A
cell (channel, bin) holds a line with prior probability ``pi`` and amplitude
``CN(0, g N)``, so its log evidence over noise is
``logaddexp(log(1 - pi), log pi - log(1 + g) + z g / (1 + g))``; cells add over
channels (Bretthorst eq. 7.2 with the per-channel noise integrated out), so a
channel that does not see a line costs ~``log(1 - pi)`` and one that does
counts in full. A comb at ``s`` scores ``Lambda(s) = sum_k L(k s)`` over the
``k_max`` orders below ``f_max``, each harmonic at ``f`` read over a band of
width ``max(1.5/T, f drift T)`` Hz (within-window fractional drift and line
width; the same band for every hypothesis reading that line) with the slab
prior spread over the band's bins. The fixed
order count and the order-dependent prior are what demote sub-multiple
ghosts: ``s/2`` puts the rotor's strongest lines at orders where the prior
expects weaker ones and lacks the true rotor's orders 21-40; ``2 s`` reads
the rotor's blade-pass line at a non-blade-pass order.

The joint evidence of R speeds counts a cell once (the orthogonal-projection
rule; two harmonics in one cell explain it once), so
``Lambda_joint = sum_i Lambda(s_i) - sum_{i<j} c_ij`` with ``c_ij`` the
evidence shared by colliding harmonics. Its modes are combinations of 1-D
peaks: the top ``n_peaks`` of ``Lambda`` (ghosts included) are combined
exhaustively, scored exactly, and the best distinct combinations refined by
coordinate ascent. Two rules carry the rotor physics: a candidate
speed's blade-pass line (order ``blades``) must carry at least ``bpf_frac``
of its strongest harmonic (``2 s`` reads the blade-pass line at its order 1
and a far weaker one at its order 2), and the R speeds of a rig lie within
``max_ratio`` (which removes ``s/2``: its profile is blade-pass-strongest
when the rotor's odd orders are weak). Lines neither a comb of the combination nor the
low-order intermodulation tones of its rotors (``a f_i +- b f_j``) explain
count against it (they are misfit of the ``R`` combs + noise model), so a ghost that
replaces a rotor and leaves that rotor's other lines unexplained loses to the
combination holding the rotor.

Window length is chosen by model comparison: a window is split into halves
while the halves' best modes beat the parent's by more than ``occam`` nats per
rotor (the Occam factor of the extra speed parameters); constant rigs stay at
the full span, drifting ones go down to ``min_win_s``. The leaves' modes are
linked by Viterbi with a random-walk prior on each rotor's speed (jump cost =
minimum over rotor permutations), which labels the rotors through time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, combinations_with_replacement
from typing import Any

import numpy as np
from scipy import fft as sfft
from scipy.ndimage import maximum_filter1d
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class JointParams:
    f_max: float = 3000.0
    #: Orders per comb (fixed, so a sub-multiple ghost loses the true rotor's
    #: high orders instead of gaining more harmonics).
    k_max: int = 40
    #: Line prior: presence probability per cell; prior line SNR at order k
    #: is ``g (B/k)**beta`` at multiples of the blade count ``B`` (the
    #: blade-pass lines, the strong ones) and ``g / k**beta`` elsewhere, so a
    #: strong line at a high order of a candidate -- a sub-multiple ghost --
    #: or at a non-blade-pass order -- a multiple ghost -- is discounted.
    pi: float = 0.3
    g: float = 100.0
    beta: float = 2.0
    blades: int = 2
    pad: int = 4
    floor_hz: float = 10.0
    #: Speed search range (rev/s) and grids.
    lo: float = 20.0
    hi: float = 260.0
    step: float = 0.002
    fine_step: float = 0.0002
    fine_half: float = 0.05
    #: Within-window fractional drift (per s): a line at f Hz is read over a
    #: band of width ``max(1.5/T, f * drift * T)`` Hz -- the same band for
    #: every hypothesis reading that line -- with the slab prior spread over
    #: the band's native bins (a wide band dilutes a line's evidence, which is
    #: what lets the window comparison prefer the length the drift allows).
    drift: float = 1e-3
    #: 1-D peaks kept for the joint enumeration, minimum separation (rev/s).
    n_peaks: int = 30
    peak_sep: float = 0.3
    n_modes: int = 8
    #: Line cells (order-1 evidence local maxima) above this count against a
    #: combination that explains none of them.
    line_min: float = 10.0
    #: Rotor physics as rules: a candidate speed's blade-pass line (order
    #: ``blades``) carries at least ``bpf_frac`` of its strongest harmonic's
    #: power (strictly strongest fails real rotors whose order-1 line is the
    #: louder: AVQ S1_seq3's 95.7 rev/s rotor); the R speeds of one rig lie
    #: within ``max_ratio``.
    bpf_frac: float = 0.3
    max_ratio: float = 1.7
    #: Intermodulation tones ``a f_i +- b f_j`` of two rotors' harmonics up to
    #: this order belong to the model (AVQ S1_seq2 shows 2 f_1 + f_2 to 0.1 Hz).
    im_order: int = 2
    #: Window hierarchy: split while gain > occam * R; smallest window (s).
    min_win_s: float = 4.0
    occam: float = 12.0
    #: Random-walk prior on each rotor's speed (rev/s per s) for the linking.
    sigma_rate: float = 0.1


def _floor_db(p_db: np.ndarray, half: int) -> np.ndarray:
    from experiments.static_rig.spectra import _floor

    return _floor(p_db, half)


class WindowEvidence:
    """Cell evidence ``L`` (padded bins up to ``f_max``) of one window."""

    def __init__(self, x_ct: np.ndarray, fs: int, p: JointParams) -> None:
        self.p = p
        n = x_ct.shape[1]
        self.T = n / fs
        nfft = 1 << int(math.ceil(math.log2(p.pad * n)))
        self.dfp = fs / nfft
        kp = int(p.f_max / self.dfp) + 2
        w = np.hanning(n).astype(np.float32)
        X = np.asarray(sfft.rfft(x_ct.astype(np.float32) * w, nfft, axis=1, workers=-1))
        P = np.abs(X[:, :kp]) ** 2
        fl = 10 ** (
            _floor_db(10 * np.log10(P + 1e-30), max(2, round(0.5 * p.floor_hz / self.dfp))) / 10
        )
        z = P / (fl / math.log(2))
        self.n_bins = z.shape[1]
        self._cz = np.concatenate([np.zeros((z.shape[0], 1)), np.cumsum(z, axis=1)], axis=1)
        self._lk: dict[int, np.ndarray] = {}

    def tol(self, f: np.ndarray | float) -> np.ndarray | float:
        """Half-width (Hz) of the band read at frequency ``f``."""
        return 0.5 * np.maximum(1.5 / self.T, np.asarray(f) * self.p.drift * self.T)

    def g_k(self, k: int) -> float:
        p = self.p
        return p.g * (p.blades / k) ** p.beta if k % p.blades == 0 else p.g / k**p.beta

    def lk(self, k: int) -> np.ndarray:
        """Evidence of order ``k`` for a band centred at each bin."""
        if k not in self._lk:
            p = self.p
            b = np.arange(self.n_bins)
            h = np.maximum(1, np.rint(self.tol(b * self.dfp) / self.dfp)).astype(int)
            n_nat = np.maximum(1.0, (2 * h + 1) * self.dfp * self.T)  # native bins in the band
            g = self.g_k(k) / n_nat
            a = g / (1 + g)
            cz = self._cz
            s_band = (
                cz[:, np.clip(b + h + 1, 0, self.n_bins)] - cz[:, np.clip(b - h, 0, self.n_bins)]
            ) / p.pad
            L = np.logaddexp(math.log(1 - p.pi), math.log(p.pi) - n_nat * np.log1p(g) + a * s_band)
            self._lk[k] = L.sum(axis=0)
        return self._lk[k]

    def k_max(self, s: float) -> int:
        return min(self.p.k_max, int(self.p.f_max // s))

    def comb(self, grid: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """``Lambda(s)`` on ``grid``; cells where ``mask`` is true count 0."""
        out = np.zeros(grid.size)
        for k in range(1, self.k_max(float(grid.min())) + 1):
            f = k * grid
            ok = f <= self.p.f_max
            idx = np.rint(f[ok] / self.dfp).astype(int)
            v = self.lk(k)[idx]
            if mask is not None:
                v = np.where(mask[idx], 0.0, v)
            out[ok] += v
        return out

    def harmonics(self, s: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(freq, tolerance, evidence) of every harmonic of ``s``."""
        k = np.arange(1, self.k_max(s) + 1)
        f = k * s
        tol = np.asarray(self.tol(f))
        idx = np.rint(f / self.dfp).astype(int)
        v = np.array([self.lk(int(kk))[i] for kk, i in zip(k, idx, strict=True)])
        return f, tol, v

    def mask_of(self, speeds: list[float]) -> np.ndarray:
        m = np.zeros(self.n_bins, dtype=bool)
        for s in speeds:
            f, tol, _ = self.harmonics(s)
            for fc, t in zip(f, tol, strict=True):
                m[max(0, int((fc - t) / self.dfp)) : int((fc + t) / self.dfp) + 1] = True
        return m

    def strength(self, s: float) -> np.ndarray:
        """Whitened band power (all channels) of every harmonic of ``s``."""
        f, tol, _ = self.harmonics(s)
        h = np.maximum(1, np.rint(tol / self.dfp)).astype(int)
        b = np.rint(f / self.dfp).astype(int)
        cz = self._cz.sum(axis=0)
        return (
            cz[np.clip(b + h + 1, 0, self.n_bins)] - cz[np.clip(b - h, 0, self.n_bins)]
        ) / self.p.pad

    def bpf_strongest(self, s: float) -> bool:
        st = self.strength(s)
        return bool(
            st.size >= self.p.blades and st[self.p.blades - 1] >= self.p.bpf_frac * st.max()
        )

    def peaks(self) -> tuple[np.ndarray, np.ndarray]:
        """Top 1-D peaks of ``Lambda`` whose blade-pass line is their
        strongest harmonic (the alias rule)."""
        p = self.p
        grid = np.arange(p.lo, p.hi, p.step)
        lam = self.comb(grid)
        r = int(round(p.peak_sep / p.step))
        loc = np.nonzero((lam == maximum_filter1d(lam, 2 * r + 1)) & (lam > 0))[0]
        loc = loc[np.argsort(lam[loc])[::-1]]
        keep = [i for i in loc[: 4 * p.n_peaks] if self.bpf_strongest(float(grid[i]))][: p.n_peaks]
        return grid[keep], lam[keep]

    def shared(self, a: float, b: float) -> float:
        """Evidence counted twice by combs ``a`` and ``b``: colliding harmonics."""
        fa, ta, va = self.harmonics(a)
        fb, tb, vb = self.harmonics(b)
        hit = np.abs(fa[:, None] - fb[None, :]) < ta[:, None] + tb[None, :]
        return float(np.sum(np.minimum(va[:, None], vb[None, :]) * hit))

    def lines(self) -> tuple[np.ndarray, np.ndarray]:
        """Line cells of the window: local maxima of the order-1 evidence map
        above ``line_min`` nats, (freq, evidence)."""
        if not hasattr(self, "_lines"):
            L = self.lk(1)
            r = max(1, int(round(float(self.tol(self.p.f_max)) / self.dfp)))
            loc = np.nonzero((maximum_filter1d(L, 2 * r + 1) == L) & (self.p.line_min < L))[0]
            self._lines = loc * self.dfp, L[loc]
        return self._lines

    def _near(self, fl: np.ndarray, f: np.ndarray, tol: np.ndarray) -> np.ndarray:
        """Which of ``fl`` lie within ``tol`` of some entry of sorted ``f``."""
        if f.size == 0:
            return np.zeros(fl.size, dtype=bool)
        j = np.clip(np.searchsorted(f, fl), 1, f.size - 1)
        d = np.minimum(np.abs(fl - f[j - 1]) - tol[j - 1], np.abs(fl - f[j]) - tol[j])
        return d <= 0

    def explains(self, s: float) -> np.ndarray:
        """Which line cells fall inside a harmonic band of comb ``s``."""
        fl, _ = self.lines()
        f, tol, _ = self.harmonics(s)
        return self._near(fl, f, tol)

    def intermod(self, speeds: list[float]) -> np.ndarray:
        """Sorted intermodulation frequencies ``a f_i +- b f_j`` of the combs'
        harmonics up to order ``im_order`` (the sum and difference tones the
        recording chain and the aerodynamics put between the rotors)."""
        k = np.arange(1, self.p.im_order + 1)
        out = []
        for i, j in combinations(range(len(speeds)), 2):
            fi, fj = k * speeds[i], k * speeds[j]
            out.append((fi[:, None] + fj[None, :]).ravel())
            out.append(np.abs(fi[:, None] - fj[None, :]).ravel())
        if not out:
            return np.array([])
        f = np.concatenate(out)
        return np.sort(f[(f > self.p.lo) & (f <= self.p.f_max)])

    def unexplained(self, speeds: list[float]) -> float:
        """Evidence of the line cells neither a comb of ``speeds`` nor their
        intermodulation explains."""
        fl, v = self.lines()
        ex = np.zeros(v.size, dtype=bool)
        for s in speeds:
            ex |= self.explains(s)
        im = self.intermod(speeds)
        ex |= self._near(fl, im, np.asarray(self.tol(im)))
        return float(v[~ex].sum())

    def joint(self, speeds: list[float]) -> float:
        """Joint evidence: explained cells once (a harmonic sitting on an
        intermodulation tone of the other combs counts 0), minus the lines
        left unexplained (misfit of the ``R`` combs + noise model)."""
        lam = 0.0
        for i, s in enumerate(speeds):
            f, tol, v = self.harmonics(s)
            im = self.intermod(speeds[:i] + speeds[i + 1 :])
            lam += float(v[~self._near(f, im, np.asarray(self.tol(im)))].sum())
        for i, j in combinations(range(len(speeds)), 2):
            lam -= self.shared(speeds[i], speeds[j])
        return lam - self.unexplained(speeds)

    def modes(self, R: int) -> list[dict[str, Any]]:
        """Best distinct R-combinations of the 1-D peaks, refined."""
        p = self.p
        sp, lam = self.peaks()
        M = sp.size
        if M == 0:
            return []
        c = np.zeros((M, M))
        for i in range(M):
            c[i, i] = lam[i]
            for j in range(i + 1, M):
                c[i, j] = c[j, i] = self.shared(float(sp[i]), float(sp[j]))
        combos = [
            cmb
            for cmb in combinations_with_replacement(range(M), R)
            if max(cmb.count(m) for m in set(cmb)) <= 2
            and sp[list(cmb)].max() <= p.max_ratio * sp[list(cmb)].min()
        ]
        if not combos:
            return []
        idx = np.array(combos)
        score = lam[idx].sum(axis=1)
        for a, b in combinations(range(R), 2):
            score -= c[idx[:, a], idx[:, b]]
        fl, lv = self.lines()
        ex = np.array([self.explains(float(s)) for s in sp])  # (M, n_lines)
        exi = np.zeros((M, M, fl.size), dtype=bool)  # pair intermodulation
        for i in range(M):
            for j in range(i + 1, M):
                im = self.intermod([float(sp[i]), float(sp[j])])
                exi[i, j] = exi[j, i] = self._near(fl, im, np.asarray(self.tol(im)))
        covered = ex[idx].any(axis=1)
        for a, b in combinations(range(R), 2):
            covered |= exi[idx[:, a], idx[:, b]]
        score -= (~covered) @ lv
        order = np.argsort(score)[::-1]
        out: list[dict[str, Any]] = []
        for o in order:
            s = sorted(float(sp[m]) for m in combos[o])
            if any(np.all(np.abs(np.array(s) - np.array(m["speeds"])) < p.peak_sep) for m in out):
                continue
            s, ev = self.refine(s)
            out.append({"speeds": s, "evidence": ev, "enum_evidence": float(score[o])})
            if len(out) >= p.n_modes:
                break
        out.sort(key=lambda m: -m["evidence"])
        return out

    def refine(self, speeds: list[float]) -> tuple[list[float], float]:
        p = self.p
        s = list(speeds)
        for _ in range(2):
            for i in range(len(s)):
                others = s[:i] + s[i + 1 :]
                mask = self.mask_of(others)
                grid = np.arange(s[i] - p.fine_half, s[i] + p.fine_half, p.fine_step)
                lam = self.comb(grid, mask)
                s[i] = float(grid[int(np.argmax(lam))])
        return sorted(s), self.joint(s)


def _tree(
    x: np.ndarray, fs: int, t0: int, t1: int, R: int, p: JointParams, depth: int
) -> tuple[list[dict[str, Any]], float]:
    """Leaves of the window tree under ``[t0, t1)`` and their total evidence
    (sum of the leaves' best modes)."""
    we = WindowEvidence(x[:, t0:t1], fs, p)
    modes = we.modes(R)
    best = modes[0]["evidence"] if modes else 0.0
    leaf = {"t0": t0 / fs, "t1": t1 / fs, "depth": depth, "modes": modes}
    if (t1 - t0) / fs >= 2 * p.min_win_s:
        mid = (t0 + t1) // 2
        left, e_left = _tree(x, fs, t0, mid, R, p, depth + 1)
        right, e_right = _tree(x, fs, mid, t1, R, p, depth + 1)
        leaf["split_gain"] = e_left + e_right - best
        if leaf["split_gain"] > p.occam * R:
            return left + right, e_left + e_right
    return [leaf], best


def _link(leaves: list[dict[str, Any]], R: int, p: JointParams) -> list[dict[str, Any]]:
    """Viterbi over each leaf's modes; transition = min over permutations of
    the Gaussian random-walk cost. Returns the chosen mode per leaf with
    rotor labels consistent through time."""
    n = len(leaves)
    cost = [np.array([-m["evidence"] for m in lf["modes"]]) for lf in leaves]
    back: list[np.ndarray] = []
    perms: list[list[list[np.ndarray]]] = []
    acc = cost[0]
    for i in range(1, n):
        dt = 0.5 * (
            (leaves[i]["t0"] + leaves[i]["t1"]) - (leaves[i - 1]["t0"] + leaves[i - 1]["t1"])
        )
        sig = p.sigma_rate * max(dt, 1.0)
        prev = leaves[i - 1]["modes"]
        cur = leaves[i]["modes"]
        trans = np.zeros((len(prev), len(cur)))
        pm: list[list[np.ndarray]] = []
        for a, ma in enumerate(prev):
            row = []
            for b, mb in enumerate(cur):
                d = (np.array(ma["speeds"])[:, None] - np.array(mb["speeds"])[None, :]) ** 2 / (
                    2 * sig**2
                )
                ri, ci = linear_sum_assignment(d)
                trans[a, b] = d[ri, ci].sum()
                row.append(ci)
            pm.append(row)
        perms.append(pm)
        tot = acc[:, None] + trans + cost[i][None, :]
        back.append(np.argmin(tot, axis=0))
        acc = tot.min(axis=0)
    path = [int(np.argmin(acc))]
    for i in range(n - 1, 0, -1):
        path.append(int(back[i - 1][path[-1]]))
    path = path[::-1]
    out = []
    label = np.arange(R)  # rotor label of each position in the current mode
    for i, (lf, m) in enumerate(zip(leaves, path, strict=True)):
        mode = lf["modes"][m]
        if i > 0:
            ci = perms[i - 1][path[i - 1]][m]
            new = np.empty(R, dtype=int)
            new[ci] = label
            label = new
        sp = np.array(mode["speeds"])
        out.append(
            {
                "t0": lf["t0"],
                "t1": lf["t1"],
                "depth": lf["depth"],
                "evidence": mode["evidence"],
                "speeds": [float(sp[np.nonzero(label == r)[0][0]]) for r in range(R)],
                "n_modes": len(lf["modes"]),
                "runner_up_gap": (mode["evidence"] - lf["modes"][1]["evidence"])
                if len(lf["modes"]) > 1
                else None,
            }
        )
    return out


def analyse(x_ct: np.ndarray, fs: int, R: int, p: JointParams = JointParams()) -> dict[str, Any]:
    """Window tree + mode linking of one motor-on span."""
    leaves, _ = _tree(x_ct, fs, 0, x_ct.shape[1], R, p, 0)
    leaves = [lf for lf in leaves if lf["modes"]]
    linked = _link(leaves, R, p)
    tracks = np.array([lf["speeds"] for lf in linked])
    return {
        "params": p.__dict__,
        "leaves": leaves,
        "linked": linked,
        "rotors": [
            {
                "mean": float(tracks[:, r].mean()),
                "min": float(tracks[:, r].min()),
                "max": float(tracks[:, r].max()),
                "hop_scatter": float(np.median(np.abs(np.diff(tracks[:, r]))))
                if len(tracks) > 1
                else 0.0,
            }
            for r in range(R)
        ],
    }
