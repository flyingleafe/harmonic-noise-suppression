"""BPF estimator from Marcoux and Grondin, ICRA 2026, arXiv:2602.09991.

Reimplementation of §II-A / Tables I–III, not the delivery change detector,
plus a four-track, activity-gated adaptation (:class:`MarcouxGrondinRPS4`).
"""

import torch
from torch import Tensor, nn

from .frontends import build_frontend


def _conv_block(kernel: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(2, 16, (3, kernel), padding=(1, kernel // 2)),
        nn.BatchNorm2d(16),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),
    )


class _Branch(nn.Module):
    def __init__(self, outputs: int, *, initial_bias: float | None = None) -> None:
        super().__init__()
        self.gru = nn.GRU(
            16384, 128, num_layers=3, batch_first=True, bidirectional=True, dropout=0.4
        )
        output = nn.Linear(128, outputs)
        if initial_bias is not None:
            nn.init.constant_(output.bias, initial_bias)
        self.head = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.4), output)

    def forward(self, features: Tensor) -> Tensor:
        return self.head(self.gru(features)[0])


class MarcouxGrondinRPS(nn.Module):
    """Paper network: the fastest shaft speeds, ascending per frame, in rev/s.

    Input is mono 16 kHz audio, ``(B, N)`` or ``(B, 1, N)``. The paper uses
    93-frame clips and two outputs; the network preserves the centered STFT
    grid for any clip length. Blade count converts BPF to shaft speed.
    """

    sort_outputs = True
    gate_by_activity = False
    activity_bias: float | None = None

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 2,
        num_blades: int = 2,
    ) -> None:
        super().__init__()
        if num_rotors < 1:
            raise ValueError("num_rotors must be positive")
        if num_blades < 1:
            raise ValueError("num_blades must be positive")
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.num_rotors = num_rotors
        self.num_blades = num_blades
        self.frontend = build_frontend("mel_cepstrum", n_fft=n_fft, hop_length=hop_length)
        self.blocks = nn.ModuleList(_conv_block(kernel) for kernel in (33, 21, 11, 3))
        # Keep the final ReLU active at initialization on quiet clips.
        self.bpf = _Branch(num_rotors, initial_bias=1.0)
        self.activity = _Branch(1, initial_bias=self.activity_bias)

    def _encode(self, audio: Tensor) -> Tensor:
        if audio.ndim == 3 and audio.shape[1] == 1:
            audio = audio[:, 0]
        if audio.ndim != 2:
            raise ValueError("expected mono audio shaped (batch, samples) or (batch, 1, samples)")
        features = self.frontend(audio).transpose(-1, -2)  # (B, 2, T, 128)
        branches = [block(features).transpose(1, 2) for block in self.blocks]
        return torch.cat(branches, dim=2).flatten(2)  # pyright: ignore[reportPrivateImportUsage]

    def forward(
        self, audio: Tensor, *, return_activity: bool = False
    ) -> Tensor | tuple[Tensor, Tensor]:
        """Return RPS ``(B, R, T)`` and optionally activity logits ``(B, T)``."""
        features = self._encode(audio)
        bpf = self.bpf(features).relu()
        if self.sort_outputs:
            bpf = bpf.sort(dim=-1).values
        rates = bpf.transpose(1, 2) / self.num_blades
        if not (return_activity or self.gate_by_activity):
            return rates
        logits = self.activity(features).squeeze(-1)
        if self.gate_by_activity:
            # Hard 0/1 gate: off frames are exactly zero. The decision has no
            # gradient; the activity BCE alone trains it.
            rates = rates * (logits > 0).unsqueeze(1)
        return (rates, logits) if return_activity else rates


class MarcouxGrondinRPS4(MarcouxGrondinRPS):
    """Adaptation, not in the paper: one unsorted track per rotor, for
    clip-level PIT, with speeds zeroed wherever activity is off (logit <= 0).
    """

    sort_outputs = False
    gate_by_activity = True
    # Open the gate at initialization. Otherwise a seed can close it on every
    # frame and the speed head gets no gradient until the activity loss
    # reopens it.
    activity_bias = 2.0

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        num_blades: int = 2,
    ) -> None:
        super().__init__(n_fft, hop_length, num_rotors, num_blades)
