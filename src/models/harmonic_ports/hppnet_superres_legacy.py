"""HPPNet L2 with the RETIRED ``FreqSuperResHead`` — frozen, for old checkpoints only.

The head was deleted in the rp-protocol cutover (2026-10-14; last at commit
eb62e8d8) and replaced by `models.salience_rps.TranslationHead`. It is restored
here verbatim so the historical runs that trained through it (`hppnet_l2_r2_s0`,
the frozen real-split reference 2.27, and the `real_r4_hppnet_l2*_unified`
arms) can be built and re-scored on the rp_real panel
(``scripts/rp_panel_eval.py``). Nothing new trains with it.

WHAT IT DID, and why it matters for the post-mortem
(docs/experiments/paper-real-protocol.md): a fixed linear-interpolation matrix
``W`` maps the trunk's 352-bin log grid onto a linear 0-150 rev/s grid; output
bins BELOW the trunk's 27.5 rev/s are not masked but CLAMPED to the lowest log
bin (``searchsorted`` clipped to ``[1, F_in - 1]``, weight 1.0 on bin 0 of the
input), so bins 0..54 of the 300-bin grid are 55 copies of what the trunk says
at 27.5 rev/s. A ``Conv(5,1) -> BatchNorm -> ReLU`` stack (x2) then a 1x1,
with the four maps MIXED on the channel axis, sharpens on the linear grid.
That stack is what read "stopped" off the trunk's floor logit — a level read
with dataset statistics (BatchNorm), the one thing the instance-normalised
trunk cannot do by itself — and why the old HPPNet-L2 had a zero path the
minimal translation head lacks.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.harmonic_ports.hppnet_orig import HPPNetOrig
from models.multif0.utils import linear_freq_grid

__all__ = ["FreqSuperResHead", "HPPNetOrigSuperResLegacy"]


def _freq_interp_matrix(in_freqs: np.ndarray, out_freqs: np.ndarray) -> np.ndarray:
    """Linear-interpolation resampling matrix ``W`` of shape ``(F_out, F_in)``.

    Output frequencies outside the input range CLAMP to the nearest input bin
    (the retired behaviour; `models.salience_rps.translation_matrix` masks them).
    """
    in_freqs = np.asarray(in_freqs, dtype=np.float64)
    out_freqs = np.asarray(out_freqs, dtype=np.float64)
    f_in, f_out = len(in_freqs), len(out_freqs)
    hi = np.clip(np.searchsorted(in_freqs, out_freqs), 1, f_in - 1)
    lo = hi - 1
    f_lo, f_hi = in_freqs[lo], in_freqs[hi]
    w_hi = np.clip((out_freqs - f_lo) / (f_hi - f_lo + 1e-12), 0.0, 1.0)
    w = np.zeros((f_out, f_in), dtype=np.float32)
    rows = np.arange(f_out)
    w[rows, lo] = 1.0 - w_hi
    w[rows, hi] = w_hi
    return w


class FreqSuperResHead(nn.Module):
    """Resample salience logits onto a finer grid, then sharpen along frequency
    with a small ``(kernel, 1)`` conv stack whose first layer MIXES the maps."""

    def __init__(
        self,
        in_freqs: np.ndarray,
        out_freqs: np.ndarray,
        hidden: int = 32,
        kernel: int = 5,
        n_layers: int = 2,
        n_maps: int = 1,
    ):
        super().__init__()
        self.register_buffer("W", torch.from_numpy(_freq_interp_matrix(in_freqs, out_freqs)))
        self.n_maps = int(n_maps)
        pad = kernel // 2
        layers: list[nn.Module] = []
        ch_in = self.n_maps
        for _ in range(n_layers):
            layers += [
                nn.Conv2d(ch_in, hidden, (kernel, 1), padding=(pad, 0)),
                nn.BatchNorm2d(hidden),
                nn.ReLU(inplace=True),
            ]
            ch_in = hidden
        layers += [nn.Conv2d(ch_in, self.n_maps, (1, 1))]
        self.net = nn.Sequential(*layers)

    def forward(self, logits_in: torch.Tensor) -> torch.Tensor:
        """``(B, M, F_in, T)`` -> ``(B, M, F_out, T)``."""
        b, m, f_in, t = logits_in.shape
        x = torch.einsum("oi,bit->bot", self.W, logits_in.reshape(b * m, f_in, t))
        return self.net(x.reshape(b, m, -1, t))


class HPPNetOrigSuperResLegacy(HPPNetOrig):
    """`HPPNetOrig` whose per-rotor maps go through `FreqSuperResHead` onto a
    linear ``[out_fmin, out_fmax]`` grid of ``out_bins`` — the pre-cutover L2.
    No off logit, no translation head; state-dict keys ``trunk.*``, ``head.*``,
    ``frontend.*``, ``superres_head.*`` as the old checkpoints carry them.
    """

    def __init__(
        self,
        *,
        out_fmin: float = 0.0,
        out_fmax: float = 150.0,
        out_bins: int = 300,
        head_hidden: int = 32,
        head_kernel: int = 5,
        **kwargs,
    ):
        super().__init__(translation_head=False, **kwargs)
        if self.n_maps < 2:
            raise ValueError("the legacy super-res L2 needs per-rotor maps (n_maps > 1)")
        assert self.out_freqs is not None  # set by HPPNetOrig.__init__ (L1 grid)
        in_freqs = np.asarray(self.out_freqs[1:], dtype=np.float64)  # drop L1's off slot
        out_freqs = linear_freq_grid(float(out_fmin), float(out_fmax), int(out_bins))
        self.off_head = None  # the old model had no off logit
        self.translation = None
        self.out_freqs = out_freqs
        self.out_has_off = False
        self.superres_head = FreqSuperResHead(
            in_freqs, out_freqs, hidden=head_hidden, kernel=head_kernel, n_maps=self.n_maps
        )

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        x = self.frontend(audio).unsqueeze(1)  # (B, 1, T, F)
        n_time = x.shape[2]
        if self.time_pooling:
            x = F.max_pool2d(x, (2, 1))
        y = self.head(self.temporal(self.trunk(x)))  # (B, n_maps, T', F_out)
        if self.time_pooling:
            y = F.interpolate(y, size=(n_time, y.shape[-1]), mode="bilinear", align_corners=False)
        y = self.superres_head(y.transpose(2, 3))  # (B, n_maps, out_bins, T)
        b, m, f, t = y.shape
        return y.reshape(b, m * f, t)
