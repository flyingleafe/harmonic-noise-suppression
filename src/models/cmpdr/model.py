"""cMPDR + DNN two-stage enhancer (Bologni, Arrieta Larraza, Heusdens, Hendriks,
EUSIPCO 2026, arXiv:2602.12986): the blind cMPDR filter, then a CRNN or
ULCNet trained on its output.

Upstream trains the networks on cMPDR outputs computed offline by the
``cmvdr`` CLI; here the filter runs inside ``forward`` (no gradients, no
parameters), so the pair trains and evaluates as one model. Training recipe
of the paper (facts, not wired here): loss = mean absolute error in the
time-frequency domain; Adam, initial learning rate 1e-3; gradient clipping at
norm 3.0; learning rate x0.5 after 3 epochs without validation improvement
(min 1e-9, cooldown 1 in ``network_config.yml``); early stopping after 5
epochs; at most 300 epochs; batch size 64 (``network_config.yml``); 5 s
sequences; data at 16 kHz with SNR U(-20, 0) dB; no EMA; precision not stated
(TensorFlow default float32). Networks are retrained per pre-processor.

``network=None`` returns the cMPDR filter output alone (no parameters:
evaluation only). ``use_cmpdr=False`` gives the paper's no-pre-processor
baselines (network on the raw mixture).
"""

from __future__ import annotations

from typing import Any

from torch import Tensor, nn

from models.cmpdr.frontend import CMPDRFrontEnd
from models.cmpdr.networks import CRNN, ULCNet

_NETWORKS = {"crnn": CRNN, "ulcnet": ULCNet}


class CMPDRDNN(nn.Module):
    """``forward((B, 1, T) or (B, T)) -> (B, 1, T)``, 16 kHz mono."""

    def __init__(
        self,
        network: str | None = "crnn",
        use_cmpdr: bool = True,
        frontend: dict[str, Any] | None = None,
        network_params: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        if network is not None and network not in _NETWORKS:
            raise ValueError(f"network must be one of {sorted(_NETWORKS)} or None, got {network!r}")
        if network is None and not use_cmpdr:
            raise ValueError("network=None needs use_cmpdr=True (nothing to run otherwise)")
        self.frontend = CMPDRFrontEnd(**(frontend or {})) if use_cmpdr else None
        self.network = _NETWORKS[network](**(network_params or {})) if network else None

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() == 3:
            if x.shape[1] != 1:
                raise ValueError(f"expected mono (B, 1, T), got {tuple(x.shape)}")
            x = x[:, 0]
        y = self.frontend(x) if self.frontend is not None else x
        if self.network is not None:
            y = self.network(y)
        return y[:, None]


def build_cmpdr(**params: Any) -> nn.Module:
    """Hydra ``_target_`` factory; plain kwargs of :class:`CMPDRDNN`."""
    return CMPDRDNN(**params)
