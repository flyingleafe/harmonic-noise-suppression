"""Mel spectrogram and power cepstrum for the Marcoux–Grondin BPF estimator."""

import torch
from torch import Tensor

from . import SpectralFrontEnd, register_frontend


@register_frontend
class MelCepstrum(SpectralFrontEnd):
    """Stack 128 power-mel bins and the first 128 power-cepstrum bins.

    The paper does not specify the window, mel scale or log floor. Use a
    periodic Hann window, Slaney-normalized Slaney mel filters, and a power
    floor of 1e-10 before taking the natural log for the cepstrum. Both
    channels share the repository's centered, 16 kHz STFT grid.
    """

    key = "mel_cepstrum"
    out_channels = 2
    n_bins = 128

    def __init__(self, n_fft: int = 2048, hop_length: int = 512) -> None:
        super().__init__()
        from torchaudio.functional import melscale_fbanks

        self.n_fft = n_fft
        self.hop_length = hop_length
        self.window: Tensor
        self.mel_filters: Tensor
        window = torch.hann_window(n_fft)  # pyright: ignore[reportPrivateImportUsage]
        self.register_buffer("window", window, persistent=False)
        self.register_buffer(
            "mel_filters",
            melscale_fbanks(
                n_fft // 2 + 1, 0.0, 7000.0, self.n_bins, 16000, norm="slaney", mel_scale="slaney"
            ).T,
            persistent=False,
        )

    def num_frames(self, n_samples: int) -> int:
        return (n_samples + 2 * (self.n_fft // 2) - self.n_fft) // self.hop_length + 1

    def forward(self, audio: Tensor) -> Tensor:
        # fp32 always: training audio reaches 1e5 bin power, past fp16's 65504,
        # and an inf mel bin poisons the BatchNorm running stats with NaN.
        with torch.autocast(audio.device.type, enabled=False):
            spectrum = torch.stft(  # pyright: ignore[reportPrivateImportUsage]
                audio.float(),
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                window=self.window.float(),
                pad_mode="constant",
                return_complex=True,
            )
            real, imag = spectrum.real, spectrum.imag
            power = real.square().addcmul(imag, imag)
            mel = self.mel_filters.float() @ power
            cepstrum = torch.fft.irfft(power.clamp_min(1e-10).log(), n=self.n_fft, dim=1)
            return torch.stack((mel, cepstrum[:, : self.n_bins].square()), dim=1)
