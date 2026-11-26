"""Blind single-channel cMPDR (cyclic MPDR) filter, batched in torch.

Port of the cMPDR pre-processor of Bologni, Arrieta Larraza, Heusdens and
Hendriks, "A two-step approach for speech enhancement in low-SNR scenarios using
cyclostationary beamforming and DNNs", EUSIPCO 2026 (arXiv:2602.12986), which
runs the filter of Bologni, Heusdens and Hendriks, "MPDR beamforming for
almost-cyclostationary processes" (arXiv:2510.18391) on one microphone.

Upstream: https://github.com/Screeen/cmvdr at fe4386cda30bb622cc7c359b667026622da8cf0e
(MIT, "Copyright (c) 2021 All Licenses"), the ``cmvdr`` package that
https://github.com/narrietal/cMPDR_DNN (commit 753c04cbe986, MIT) pins as its
``cmpdr`` submodule and invokes as ``cmvdr -i noisy/ -o cmpdr/`` (blind: no
noise reference). The reference path is
``ExperimentManager.run_cmvdr_inference_file`` with the packaged
``presets/inference.yaml`` and ``config.assign_default_values``, beamformer
``cmvdr_blind``. This module re-expresses that NumPy/SciPy/numba pipeline as
batched torch so it runs on GPU inside ``forward``; every constant below is the
preset's value.

Algorithm (all per utterance, M = 1 microphone)
-----------------------------------------------
1. Resonance estimation (``SpectralEstimator.estimate_harmonics_periodogram``):
   periodogram ``|rfft(x_seg, 131072)|^2 / 131072`` of at most 2 s of the input,
   starting 0.05 s in (``start = max(0, min(T, 800 + 32000) - min(T, 32000))``);
   kept between 20 Hz and 2500 Hz and normalised to its maximum;
   ``scipy.signal.find_peaks(height=1e-4, distance=20 Hz, width=1,
   prominence=0)``; the 10 highest peaks are kept (``max_num_harmonics_peaks``).
   ONE set of peaks per utterance, from the whole 2 s segment.
2. Candidate cyclic frequencies (``compute_harmonic_and_modulation_sets_
   global_coherence``): all pairwise differences of ``{0} U peaks`` (up to
   11 x 11 = 121 values, signed), merged by ``unique_with_relative_tolerance_
   fast(tol=1e-3)`` (chained neighbour clusters of the sorted values, each
   represented by its first occurrence in the row-major, |d|-sorted list).
3. Spectral coherence (``CoherenceManager.compute_coherence``): for each
   candidate alpha, the two-sided STFT of ``x(n) exp(j 2 pi alpha n / fs)``
   (time-domain modulation, so the shift is not limited to the bin grid);
   ``rho(alpha, k) = |mean_l X_0 X_alpha^*|^2 / (psd_0 psd_alpha)``, time
   averages over ALL frames of the utterance, PSDs floored at
   ``max_k psd_0(k) / 1000``, zeroed where ``k df < alpha`` or
   ``k df > fs/2 + alpha``, ``rho(0, k) = 1``; bins ``k < ceil(3 + f_max/df)``.
4. Modulation sets (``calculate_harmonic_info_from_coherence``): bin k is
   cyclic iff some alpha != 0 has ``rho > 0.6``; its set keeps the (at most)
   ``P_max = 8`` most coherent alphas, ordered [most coherent, rest by
   increasing alpha]. Every other bin uses one channel: the first alpha of the
   first cyclic bin's set (always 0 in practice, since ``rho(0, k) = 1``).
5. Recursive cMPDR (``rank1_update_covariances`` +
   ``compute_cyclic_mvdr_beamformers``): per cyclic bin, ``S <- mean_l z z^H``
   over the whole utterance (the noisy signal is its own "noise covariance
   estimate" and that average seeds the recursion), then for every frame
   ``S <- 0.95 S + 0.05 z_l z_l^H`` (forgetting factor 0.05), diagonal loading
   ``max(1e-15, (l_max - 1000 max(l_min, 1e-15)) / 999)`` (condition number
   1000), ``w = (S + lam I)^-1 e1 / Re(e1^T (S + lam I)^-1 e1)``,
   ``y = w^H z_l`` (weights from the covariance that already contains frame
   l). Non-cyclic bins pass through (weight 1).
6. iSTFT with SciPy ``ShortTimeFFT.istft`` (canonical dual window), cropped
   to T.

STFT: periodic Hann 2048, hop 512 (75 % overlap), ``scale_to='magnitude'``
(window / sum(window)), SciPy frame layout (frame p centred on sample
``p * 512``, frames ``p_min .. p_max(T) - 1``, zero padding outside the
signal, phase referenced to the frame centre).

Assumptions of the released method, stated as facts from paper and code
-----------------------------------------------------------------------
* The cyclic frequencies are estimated ONCE per recording, from the
  periodogram of a single <= 2 s segment (paper: "we assume the noise resonant
  frequencies remain constant over the recording duration. Thus, M_k is
  estimated once per recording"). The coherence pruning uses time averages
  over the whole recording. Only the covariance is tracked over time (5.).
* Up to 10 periodogram peaks in 20-2500 Hz are used; peaks closer than 20 Hz
  are suppressed. Nothing groups peaks into harmonic series: several rotors
  simply contribute peaks to the same top-10 list, and their pairwise
  differences (within and across rotors) all become candidates; coherence
  keeps whatever is coherent with the unshifted bin.
* A time-varying fundamental is not tracked: its periodogram peaks smear over
  the 2 s segment (or split into several peaks), the fixed alphas then
  correlate only partly with the moving harmonics, coherence over the whole
  utterance drops, fewer bins pass the 0.6 threshold, and those bins fall
  back to the unprocessed STFT. Above 2500 Hz + 3 bins nothing is processed.
  (Paper: "adaptive tracking of time-varying resonances is reserved for
  future work.")

Settings the EUSIPCO paper leaves unstated
------------------------------------------
The paper names Q peaks, beta_x = 0.95 and the 0.6-coherence pruning but no
other value. The parent paper's Table II lists Q = 20 peaks and a square-root
Hann STFT window; the ``inference`` preset that cMPDR_DNN's README runs
(``cmvdr -i noisy/ -o cmpdr/``) uses Q = 10 and a plain Hann window with
magnitude scaling. This port follows the released preset (all values are
constructor arguments). The input is cast to float64 first, which is what
NumPy 1.26.4 (pinned in cmvdr's ``uv.lock``) does inside ``rfft`` of the
float32 audio that ``librosa.load`` returns.

Deviations and implementation choices
-------------------------------------
* Batched torch instead of per-file NumPy/numba; float64 by default (upstream
  is float64 throughout), ``dtype`` selectable.
* ``scipy.signal.find_peaks`` is re-implemented: strict local maxima, height
  filter, greedy highest-first distance suppression (same result as SciPy's
  priority loop, ``|Delta| < ceil(distance)``), then SciPy's prominence /
  half-prominence width (``wlen=None``, ``rel_height=0.5``) for the peaks the
  distance step keeps, stopping once 10 peaks pass the width test.
  Plateau maxima (two or more exactly equal neighbouring periodogram values)
  are not detected (SciPy takes their midpoint); they do not occur in a
  float64 periodogram of real audio. Exact height ties may be ordered
  differently from SciPy's unstable argsort.
* Coherence top-8 ordering with exactly equal ``rho`` may differ from NumPy's
  argsort (affects only exact ties).
* The modulated STFTs are computed frame-wise (window, modulate with the same
  absolute-sample phase ``alpha * (2 pi n / fs)``, roll, FFT) in alpha chunks
  instead of modulating the whole signal first; identical arithmetic per
  sample up to FFT rounding.
* Upstream ``scipy.linalg.solve(assume_a='pos')`` falls back to one channel on
  ``LinAlgError``; here a failed Cholesky does the same per (bin, frame).
* Upstream's CLI writes the result as 16-bit PCM WAV (``soundfile``
  default), which quantises and clips at +-1; this module returns the float
  signal.
* Input shorter than half a window (1024 samples) is rejected, as SciPy does.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import Tensor, nn

_EPS_LOADING = 1e-15  # cmvdr.beamforming.beamformer.eps
_ZERO_ATOL = 1e-8  # np.isclose(x, 0) in unique_with_relative_tolerance_fast


def _scipy_hann_periodic(n: int) -> np.ndarray:
    """``scipy.signal.get_window('hann', n, fftbins=True)``, same arithmetic."""
    fac = np.linspace(-np.pi, np.pi, n + 1)[:n]
    return 0.5 + 0.5 * np.cos(fac)


def _dual_canonical_window(win: np.ndarray, hop: int) -> np.ndarray:
    """``scipy.signal._short_time_fft._calc_dual_canonical_window``."""
    w2 = win**2
    dd = w2.copy()
    for k in range(hop, len(win), hop):
        dd[k:] += w2[:-k]
        dd[:-k] += w2[k:]
    return win / dd


class CMPDRFrontEnd(nn.Module):
    """Blind single-channel cMPDR filter: ``(B, T) -> (B, T)``, no parameters.

    Runs under ``torch.no_grad()``; see the module docstring for the algorithm.
    """

    window: Tensor
    dual_window: Tensor

    def __init__(
        self,
        sr: int = 16000,
        nfft: int = 2048,
        overlap_fraction: float = 0.75,
        forgetting_factor: float = 0.05,
        condition_number: float = 1000.0,
        p_max: int = 8,
        harmonic_threshold: float = 0.6,
        freq_range_cyclic: tuple[float, float] = (20.0, 2500.0),
        peaks_nfft: int = 131072,
        peaks_max_len_seconds: float = 2.0,
        peaks_wait_len_seconds: float = 0.05,
        peaks_min_dist_hz: float = 20.0,
        peaks_max_ratio_from_highest: float = 10000.0,
        peaks_min_width: float = 1.0,
        max_num_peaks: int = 10,
        coherence_min_relative_power: float = 1.0e3,
        alpha_rel_tol: float = 1.0e-3,
        dtype: str = "float64",
        chunk_elements: int = 1 << 22,
    ) -> None:
        super().__init__()
        self.sr = int(sr)
        self.nfft = int(nfft)
        noverlap = int(self.nfft * overlap_fraction)
        self.hop = self.nfft - noverlap
        self.n_bins = self.nfft // 2 + 1
        self.forgetting_factor = float(forgetting_factor)
        self.condition_number = float(condition_number)
        self.p_max = int(p_max)
        self.harmonic_threshold = float(harmonic_threshold)
        self.freq_range_cyclic = (float(freq_range_cyclic[0]), float(freq_range_cyclic[1]))
        self.peaks_nfft = int(peaks_nfft)
        self.peaks_max_len_seconds = float(peaks_max_len_seconds)
        self.peaks_wait_len_seconds = float(peaks_wait_len_seconds)
        self.max_num_peaks = int(max_num_peaks)
        self.peaks_min_width = float(peaks_min_width)
        self.coherence_min_relative_power = float(coherence_min_relative_power)
        self.alpha_rel_tol = float(alpha_rel_tol)
        if dtype not in ("float64", "float32"):
            raise ValueError(f"dtype must be 'float64' or 'float32', got {dtype!r}")
        self.dtype = getattr(torch, dtype)
        self.chunk_elements = int(chunk_elements)

        # Periodogram grid exactly as numpy.fft.rfftfreq(peaks_nfft, 1 / sr).
        self._res_hz = 1.0 / (self.peaks_nfft * (1.0 / self.sr))
        self._f_lo = int(self.freq_range_cyclic[0] / self._res_hz)
        self._f_hi = int(self.freq_range_cyclic[1] / self._res_hz) + 1
        self._max_ratio = float(peaks_max_ratio_from_highest)
        self._min_distance = math.ceil(max(1.0, float(peaks_min_dist_hz) / self._res_hz))
        # ShortTimeFFT.delta_f = 1 / (mfft * T), T = 1 / fs.
        self._delta_f = 1.0 / (self.nfft * (1.0 / self.sr))

        win = _scipy_hann_periodic(self.nfft)
        win = win * (1.0 / abs(win.sum()))  # scale_to('magnitude')
        dual = _dual_canonical_window(win, self.hop)
        w2 = win**2
        self._n_lead_zeros = int(np.argmax(w2 != 0))
        self._n_trail_zeros = int(np.argmax(w2[::-1] != 0))
        self._m_mid = self.nfft // 2
        self._p_min = self._pre_padding_p_min()
        self.register_buffer("window", torch.from_numpy(win), persistent=False)
        self.register_buffer("dual_window", torch.from_numpy(dual), persistent=False)

    # ------------------------------------------------------------------ STFT
    def _pre_padding_p_min(self) -> int:
        """``ShortTimeFFT.p_min`` (``_pre_padding``) for this window."""
        n0 = -self._m_mid
        for p_, n_ in enumerate(range(n0, n0 - self.nfft - 1, -self.hop)):
            n_next = n_ - self.hop
            if n_next + self.nfft <= 0 or -n_next <= self._n_trail_zeros:
                return -p_
        raise RuntimeError("unreachable")

    def _p_max(self, n: int) -> int:
        """``ShortTimeFFT.p_max(n)`` (``_post_padding``) for this window."""
        if n < self.nfft - self._m_mid:
            raise ValueError(f"cMPDR needs at least {self.nfft - self._m_mid} samples, got {n}")
        q1 = n // self.hop
        k1 = q1 * self.hop - self._m_mid
        for q_, k_ in enumerate(range(k1, n + self.nfft, self.hop), start=q1):
            n_next = k_ + self.hop
            if n_next >= n or n - n_next <= self._n_lead_zeros:
                return q_ + 1
        raise RuntimeError("unreachable")

    def _frames(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Windowed SciPy frames ``(B, L, nfft)`` and their absolute sample
        indices ``(L, nfft)``."""
        n = x.shape[-1]
        p_max = self._p_max(n)
        n_frames = p_max - self._p_min
        start0 = self._p_min * self.hop - self._m_mid
        total = (n_frames - 1) * self.hop + self.nfft
        x_pad = nn.functional.pad(x, (-start0, total + start0 - n))
        frames = x_pad.unfold(-1, self.nfft, self.hop) * self.window.to(x.dtype)
        starts = torch.arange(n_frames, device=x.device, dtype=x.dtype) * self.hop + start0
        idx = starts[:, None] + torch.arange(self.nfft, device=x.device, dtype=x.dtype)
        return frames, idx

    def _mod_stft(self, frames: Tensor, idx: Tensor, alpha: Tensor, n_bins: int) -> Tensor:
        """Two-sided STFT of ``x(n) exp(j alpha 2 pi n / fs)``.

        frames ``(B, L, N)``, alpha ``(B, A)`` Hz -> ``(B, A, n_bins, L)``.
        """
        base = (2.0 * math.pi * idx) / self.sr  # (L, N), as 2j*pi*arange(N)/fs
        theta = alpha[:, :, None, None] * base  # (B, A, L, N)
        frames = frames[:, None]
        mod = torch.complex(frames * torch.cos(theta), frames * torch.sin(theta))
        mod = torch.roll(mod, -self._m_mid, dims=-1)  # phase_shift=0 -> centre at m=0
        spec = torch.fft.fft(mod, n=self.nfft, dim=-1)[..., :n_bins]
        return spec.transpose(-1, -2)

    def _istft(self, spec: Tensor, n: int) -> Tensor:
        """``ShortTimeFFT.istft`` (onesided, canonical dual window), cropped to n."""
        b, _, n_frames = spec.shape
        seg = torch.fft.irfft(spec.transpose(-1, -2), n=self.nfft, dim=-1)
        seg = torch.roll(seg, self._m_mid, dims=-1) * self.dual_window.to(seg.dtype)
        total = (n_frames - 1) * self.hop + self.nfft
        out = nn.functional.fold(
            seg.transpose(-1, -2),
            output_size=(1, total),
            kernel_size=(1, self.nfft),
            stride=(1, self.hop),
        ).reshape(b, total)
        start0 = self._p_min * self.hop - self._m_mid
        return out[:, -start0 : -start0 + n]

    # ------------------------------------------------------------- peaks
    def _periodogram_peaks(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """Top peaks of the normalised periodogram, ascending Hz.

        Returns ``(B, Q)`` frequencies and a ``(B, Q)`` validity mask.
        """
        b, n = x.shape
        wait_len = int(self.peaks_wait_len_seconds * self.sr)
        target_len = int(min(n, self.peaks_max_len_seconds * self.sr))
        end = min(n, wait_len + target_len)
        start = max(0, end - target_len)
        seg = x[:, start:end]
        pxx = torch.fft.rfft(seg, n=self.peaks_nfft, dim=-1).abs() ** 2 / self.peaks_nfft
        xe = pxx[:, self._f_lo : self._f_hi]
        xe = xe / xe.amax(dim=-1, keepdim=True)
        n_x = xe.shape[-1]
        hmin = xe.amax(dim=-1, keepdim=True) / self._max_ratio
        cand = torch.zeros_like(xe, dtype=torch.bool)
        cand[:, 1:-1] = (xe[:, :-2] < xe[:, 1:-1]) & (xe[:, 2:] < xe[:, 1:-1])
        cand &= xe >= hmin
        pos = torch.arange(n_x, device=x.device)
        q = self.max_num_peaks
        out = torch.full((b, q), -1, dtype=torch.long, device=x.device)
        found = torch.zeros(b, dtype=torch.long, device=x.device)
        rows = torch.arange(b, device=x.device)
        neg_inf = torch.tensor(-math.inf, dtype=xe.dtype, device=x.device)
        for _ in range(n_x // self._min_distance + 2):
            hmax, p = torch.where(cand, xe, neg_inf).max(dim=-1)
            active = torch.isfinite(hmax) & (found < q)
            if not bool(active.any()):
                break
            near = (pos[None] - p[:, None]).abs() < self._min_distance
            cand &= ~(near & active[:, None])
            width = self._peak_width(xe, p, pos)
            ok = active & (width >= self.peaks_min_width)
            slot = found.clamp(max=q - 1)
            out[rows[ok], slot[ok]] = p[ok]
            found = found + ok.long()
        valid = out >= 0
        freqs = (out + self._f_lo).to(x.dtype) * self._res_hz
        freqs = torch.where(valid, freqs, torch.full_like(freqs, math.inf))
        freqs, order = freqs.sort(dim=-1)
        valid = valid.gather(-1, order)
        return torch.where(valid, freqs, torch.zeros_like(freqs)), valid

    @staticmethod
    def _peak_width(xe: Tensor, p: Tensor, pos: Tensor) -> Tensor:
        """SciPy ``peak_prominences`` + ``peak_widths(rel_height=0.5)`` of one
        peak per row (``wlen=None``)."""
        n_x = xe.shape[-1]
        big = n_x + 1
        xp = xe.gather(-1, p[:, None])
        higher = xe > xp
        posb = pos[None].expand_as(xe)
        left_lim = torch.where(higher & (posb < p[:, None]), posb, -1).amax(-1)
        right_lim = torch.where(higher & (posb > p[:, None]), posb, n_x).amin(-1)
        in_left = (posb > left_lim[:, None]) & (posb <= p[:, None])
        in_right = (posb >= p[:, None]) & (posb < right_lim[:, None])
        inf = torch.tensor(math.inf, dtype=xe.dtype, device=xe.device)
        left_min = torch.where(in_left, xe, inf).amin(-1, keepdim=True)
        right_min = torch.where(in_right, xe, inf).amin(-1, keepdim=True)
        # SciPy keeps the minimum closest to the peak on each side.
        left_base = torch.where(in_left & (xe == left_min), posb, -1).amax(-1)
        right_base = torch.where(in_right & (xe == right_min), posb, big).amin(-1)
        prom = xp - torch.maximum(left_min, right_min)
        height = xp - prom * 0.5
        # left: first i going left from p with x[i] <= height, stopping at left_base
        lmask = (posb > left_base[:, None]) & (posb <= p[:, None]) & (xe <= height)
        li = torch.where(lmask, posb, -1).amax(-1)
        li = torch.where(li >= 0, li, left_base)
        rmask = (posb >= p[:, None]) & (posb < right_base[:, None]) & (xe <= height)
        ri = torch.where(rmask, posb, big).amin(-1)
        ri = torch.where(ri < big, ri, right_base)
        h = height[:, 0]
        x_li = xe.gather(-1, li[:, None])[:, 0]
        x_li1 = xe.gather(-1, (li + 1).clamp(max=n_x - 1)[:, None])[:, 0]
        left_ip = li.to(xe.dtype) + torch.where(x_li < h, (h - x_li) / (x_li1 - x_li), 0.0)
        x_ri = xe.gather(-1, ri[:, None])[:, 0]
        x_ri1 = xe.gather(-1, (ri - 1).clamp(min=0)[:, None])[:, 0]
        right_ip = ri.to(xe.dtype) - torch.where(x_ri < h, (h - x_ri) / (x_ri1 - x_ri), 0.0)
        return right_ip - left_ip

    # ---------------------------------------------------------- alphas
    def _candidate_alphas(self, peaks: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        """Merged pairwise differences of ``{0} U peaks``, ascending.

        ``(B, Q)`` -> alphas ``(B, (Q+1)^2)`` and validity mask.
        """
        b, q = peaks.shape
        zero = torch.zeros(b, 1, dtype=peaks.dtype, device=peaks.device)
        f = torch.cat([zero, peaks], dim=1)
        fv = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1)
        d = f[:, :, None] - f[:, None, :]  # (B, Q+1, Q+1)
        dv = fv[:, :, None] & fv[:, None, :]
        # np.lexsort((d < 0, |d|)) per row: |d| first, positive before negative.
        key2 = torch.where(dv, (d < 0).to(d.dtype), torch.full_like(d, 2.0))
        o2 = key2.argsort(dim=-1, stable=True)
        key1 = torch.where(dv, d.abs(), torch.full_like(d, math.inf)).gather(-1, o2)
        o1 = key1.argsort(dim=-1, stable=True)
        order = o2.gather(-1, o1)
        d = d.gather(-1, order).reshape(b, -1)
        dv = dv.gather(-1, order).reshape(b, -1)
        n_all = d.shape[-1]
        orig = torch.arange(n_all, device=d.device).expand(b, n_all)
        # unique_with_relative_tolerance_fast
        v = torch.where(dv, d, torch.full_like(d, math.inf))
        srt = v.argsort(dim=-1, stable=True)
        vs = v.gather(-1, srt)
        vsv = dv.gather(-1, srt)
        os_ = orig.gather(-1, srt)
        a = vs.abs()
        zeros = a <= _ZERO_ATOL
        diffs = (vs[:, 1:] - vs[:, :-1]).abs() / torch.maximum(a[:, :-1], a[:, 1:])
        close = ((diffs < self.alpha_rel_tol) & ~zeros[:, :-1] & ~zeros[:, 1:]) | (
            zeros[:, :-1] & zeros[:, 1:]
        )
        close &= vsv[:, :-1] & vsv[:, 1:]
        new = torch.cat([torch.ones_like(close[:, :1]), ~close], dim=1)
        cid = new.long().cumsum(-1) - 1
        rep = torch.full((b, n_all), n_all, dtype=torch.long, device=d.device)
        rep.scatter_reduce_(1, cid, torch.where(vsv, os_, n_all), reduce="amin")
        cvalid = rep < n_all
        alphas = d.gather(-1, rep.clamp(max=n_all - 1))
        return torch.where(cvalid, alphas, torch.zeros_like(alphas)), cvalid

    # ----------------------------------------------------------- forward
    @torch.no_grad()
    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            if x.shape[1] != 1:
                raise ValueError(f"cMPDR front end is single-channel, got {tuple(x.shape)}")
            x = x[:, 0]
        in_dtype = x.dtype
        xd = x.to(self.dtype)
        b, n = xd.shape
        dev = xd.device

        peaks, pvalid = self._periodogram_peaks(xd)
        alphas, avalid = self._candidate_alphas(peaks, pvalid)
        a_tot = alphas.shape[1]
        idx0 = ((alphas == 0) & avalid).long().argmax(-1)  # the exact-zero representative

        fmax = torch.where(pvalid, peaks.abs(), torch.zeros_like(peaks)).amax(-1)
        df = self._delta_f
        max_bin = torch.ceil((3 * df + fmax) / df).long()  # (B,)
        k_max = int(max_bin.max())

        frames, idx = self._frames(xd)
        n_frames = frames.shape[1]
        chunk = max(1, self.chunk_elements // max(1, b * n_frames * self.nfft))
        zero_alpha = torch.zeros(b, 1, dtype=xd.dtype, device=dev)
        x0 = self._mod_stft(frames, idx, zero_alpha, k_max)[:, 0]  # (B, K, L)

        # --- coherence, chunked over alphas
        cross = torch.empty(b, a_tot, k_max, dtype=xd.dtype, device=dev)
        psd = torch.empty_like(cross)
        for a0 in range(0, a_tot, chunk):
            xa = self._mod_stft(frames, idx, alphas[:, a0 : a0 + chunk], k_max)
            psd[:, a0 : a0 + chunk] = (xa.abs() ** 2).mean(-1)
            cross[:, a0 : a0 + chunk] = (x0[:, None] * xa.conj()).mean(-1).abs() ** 2
        rows = torch.arange(b, device=dev)
        kk = torch.arange(k_max, device=dev)
        in_band = kk[None] < max_bin[:, None]  # (B, K)
        psd0 = psd[rows, idx0]  # (B, K)
        floor = torch.where(in_band, psd0, torch.zeros_like(psd0)).amax(-1)
        floor = floor / self.coherence_min_relative_power
        psd = torch.maximum(psd, floor[:, None, None])
        psd0 = psd[rows, idx0]
        rho = cross / (psd0[:, None] * psd)
        kk_hz = kk.to(xd.dtype) * df
        mask = (kk_hz[None, None] >= alphas[:, :, None]) & (
            kk_hz[None, None] <= (self.sr / 2 + alphas[:, :, None])
        )
        rho = torch.where(mask, rho, torch.zeros_like(rho))
        rho[rows, idx0] = 1.0
        rho = torch.where(avalid[:, :, None], rho, torch.zeros_like(rho))

        # --- modulation sets
        sel = rho > self.harmonic_threshold  # (B, A, K)
        count = sel.sum(1)  # (B, K)
        cyclic = (count > 1) & in_band
        score = torch.where(sel, rho, torch.full_like(rho, -math.inf)).transpose(1, 2)
        p_top = min(self.p_max, a_tot)
        top_val, top_idx = score.topk(p_top, dim=-1)  # descending, (B, K, P)
        top_ok = torch.isfinite(top_val)
        rest = torch.where(top_ok[..., 1:], top_idx[..., 1:], a_tot).sort(-1).values
        set_idx = torch.cat([top_idx[..., :1], rest], dim=-1)
        set_ok = set_idx < a_tot
        set_idx = torch.where(set_ok, set_idx, idx0[:, None, None].expand_as(set_idx))

        # reference channel for the non-cyclic bins: first alpha of the first cyclic set
        has_cyc = cyclic.any(-1)
        first_k = cyclic.long().argmax(-1)
        a_ref_idx = torch.where(has_cyc, set_idx[rows, first_k, 0], idx0)
        a_ref = alphas.gather(-1, a_ref_idx[:, None])
        spec = self._mod_stft(frames, idx, a_ref, self.n_bins)[:, 0]  # (B, F, L)

        cyc_bins = torch.nonzero(cyclic.any(0)).flatten()
        if cyc_bins.numel() > 0:
            y_cyc = self._cmpdr_bins(
                frames, idx, alphas, set_idx[:, cyc_bins], set_ok[:, cyc_bins], cyc_bins, chunk
            )
            cur = spec[:, cyc_bins]
            spec[:, cyc_bins] = torch.where(cyclic[:, cyc_bins, None], y_cyc, cur)
        y = self._istft(spec, n)
        return y.to(in_dtype)

    def _cmpdr_bins(
        self,
        frames: Tensor,
        idx: Tensor,
        alphas: Tensor,
        set_idx: Tensor,
        set_ok: Tensor,
        bins: Tensor,
        chunk: int,
    ) -> Tensor:
        """Recursive cMPDR on the given bins. set_idx/set_ok ``(B, Kc, P)``."""
        b, kc, p = set_idx.shape
        n_frames = frames.shape[1]
        a_tot = alphas.shape[1]
        k_need = int(bins.max()) + 1
        cdtype = torch.complex128 if alphas.dtype == torch.float64 else torch.complex64
        z = torch.zeros(b, kc, p, n_frames, dtype=cdtype, device=alphas.device)
        for a0 in range(0, a_tot, chunk):
            a1 = min(a_tot, a0 + chunk)
            xa = self._mod_stft(frames, idx, alphas[:, a0:a1], k_need)[:, :, bins]  # (B, a, Kc, L)
            xa = xa.permute(0, 2, 1, 3)  # (B, Kc, a, L)
            local = (set_idx - a0).clamp(0, a1 - a0 - 1)
            got = xa.gather(2, local[..., None].expand(b, kc, p, n_frames))
            hit = set_ok & (set_idx >= a0) & (set_idx < a1)
            z = torch.where(hit[..., None], got, z)

        ff = self.forgetting_factor
        cov = torch.einsum("bkpl,bkql->bkpq", z, z.conj()) / n_frames
        eye = torch.eye(p, dtype=z.dtype, device=z.device)
        pad = (~set_ok).to(z.real.dtype)  # padded slots, (B, Kc, P)
        e1 = torch.zeros(p, 1, dtype=z.dtype, device=z.device)
        e1[0, 0] = 1
        c = self.condition_number
        out = torch.empty(b, kc, n_frames, dtype=z.dtype, device=z.device)
        for t in range(n_frames):
            zl = z[..., t]
            cov = (1.0 - ff) * cov + ff * (zl[..., :, None] * zl[..., None, :].conj())
            # padded slots get the (real) first diagonal entry: it lies inside the
            # spectrum of the P x P block, so min/max eigenvalues are unchanged.
            diag_fill = cov[..., 0, 0].real[..., None] * pad
            a_mat = cov + torch.diag_embed(diag_fill.to(z.dtype))
            ev = torch.linalg.eigvalsh(a_mat)
            lo = ev[..., 0].clamp(min=_EPS_LOADING)
            load = ((ev[..., -1] - c * lo) / (c - 1)).clamp(min=_EPS_LOADING)
            a_mat = a_mat + load[..., None, None] * eye
            chol, info = torch.linalg.cholesky_ex(a_mat)
            v = torch.cholesky_solve(e1.expand(b, kc, p, 1), chol)[..., 0]
            w = v / v[..., :1].real
            y = (w.conj() * zl).sum(-1)
            out[..., t] = torch.where(info == 0, y, zl[..., 0])
        return out
