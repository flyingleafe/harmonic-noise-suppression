"""Salience-map RPS predictors — multi-pitch models adapted as RPS baselines.

These models output a per-frequency-bin **salience map** (raw logits) instead of
RPS trajectories directly:

    forward(audio) -> (B, n_maps * G, T_grid)   logits on the model's output grid

- **Training**: BCE against targets derived on the fly from the batch's
  STFT-grid RPS target, so every dataloader stays on the common
  ``(audio, rps)`` task interface.
- **Inference / eval**: ``decode_logits()`` turns logits into ``(B, num_rotors,
  T_stft)`` rev/s on-device; ``predict_rps()`` is forward + that, and
  training-time validation calls it on the batch's logits, so the global-PIT
  metrics apply unchanged.

THE ADAPTATION LADDER (``docs/experiments/paper-real-protocol.md`` § "Models,
output grid and the adaptation ladder"), shared by the four multi-pitch
baselines — `LateDeepSalience`, `BasicPitchSalience` here and
`models.harmonic_ports.HarmoF0Orig` / `HPPNetOrig`:

    level  n_maps  translation_head  output grid                 readout
    L0     1       False             own log grid                threshold + Hungarian
    L1     R       False             [0.0] + own log grid        per-layer CRF
    L2     R       True              RP_LINEAR_GRID (0-159.2)    per-layer CRF

``n_maps > 1`` changes the OUTPUT only: the trunk's last map convolution
widens to ``n_maps`` and a per-map OFF logit (`OffLogitHead`, a linear read of
the trunk's frequency-pooled last features) is prepended as bin 0 — a log grid
has no zero rate, and a stopped rotor needs a state. L2 adds `TranslationHead`
(a fixed log -> linear resampling matrix plus one identity-initialised
depthwise 9-tap convolution per map), so the trunk must express the map itself:
the head adds no resolution. The maps stack along the codec's ``(batch, freq,
time)`` axis because a 4-D output does not type-check through
``tasks.codecs.SalienceRPSCodec``; `models.harmonic_ports.layer_readout`
splits them back and decodes one CRF path per layer. Pair per-rotor models with
`losses.LayerPITSalienceBCELoss` on the SAME grid spec (``conf/loss/
salience_layers_*.yaml``). ``n_maps=1`` is the published shared map exactly.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.harmonic_ports.layer_readout import LayerCRFReadout
from models.multif0.utils import (
    cqt_freq_grid,
    rps_to_salience,
    salience_target_from_resampled_rps,
    salience_to_rps_segmented,
)
from models.salience_crf import RP_LINEAR_GRID

#: Depthwise kernel of `TranslationHead` along the linear output axis (±4 bins).
TRANSLATION_KERNEL = 9
#: Fixed logit of an output bin the trunk's grid cannot reach (p ≈ 4.5e-5).
UNREACHABLE_LOGIT = -10.0


def stft_time_frames(audio_length: int, hop_length: int) -> int:
    """Number of STFT output frames for an audio length (center padding)."""
    return audio_length // hop_length + 1


def translation_matrix(
    in_freqs: np.ndarray, out_freqs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fixed resampling ``W (G_out, G_in)`` from an off+log grid onto an off+linear grid.

    Both grids hold the off state (0.0) at bin 0, and ``W[0, 0] = 1`` routes the
    off logit to output bin 0. For a rate bin ``o`` at ``f`` with output step
    ``Δ``: where the log grid is finer than ``Δ`` at ``f``, the MEAN of the log
    bins inside ``[f - Δ/2, f + Δ/2)`` (half-open windows tile the axis, so no
    log bin is skipped); where it is coarser, linear interpolation between the
    two bracketing log bins. A row whose window holds no log bin and that lies
    outside the log grid's span is ZERO — the trunk cannot express that rate
    (below the CQT's ``fmin``, e.g. 1-26 rev/s at fmin 27.5). Every other row
    sums to 1, so the logit scale is preserved.

    Returns ``(W float32, unreachable bool (G_out,))``; ``unreachable`` marks the
    zero rows.
    """
    fi = np.asarray(in_freqs, dtype=np.float64)
    fo = np.asarray(out_freqs, dtype=np.float64)
    if fi[0] != 0.0 or fo[0] != 0.0:
        raise ValueError("both grids carry the off state (0.0 rev/s) at bin 0")
    step = float(fo[2] - fo[1])
    if not np.allclose(np.diff(fo), step):
        raise ValueError("the output grid must be uniform")
    log = fi[1:]
    spacing = np.interp(fo, log[:-1], np.diff(log))
    w = np.zeros((fo.size, fi.size), dtype=np.float64)
    w[0, 0] = 1.0
    unreachable = np.zeros(fo.size, dtype=bool)
    for o in range(1, fo.size):
        f = fo[o]
        inside = np.flatnonzero((log >= f - step / 2) & (log < f + step / 2))
        in_span = log[0] <= f <= log[-1]
        if in_span and (spacing[o] >= step or inside.size == 0):
            hi = int(np.clip(np.searchsorted(log, f, side="right"), 1, log.size - 1))
            frac = (f - log[hi - 1]) / (log[hi] - log[hi - 1])
            w[o, hi] = 1.0 - frac  # column hi = log bin hi - 1
            w[o, hi + 1] = frac
        elif inside.size:
            w[o, 1 + inside] = 1.0 / inside.size
        else:
            unreachable[o] = True
    return w.astype(np.float32), unreachable


class OffLogitHead(nn.Module):
    """Per-map OFF logit: ``Linear(C -> n_maps)`` on frequency-pooled features.

    ``(B, C, F, T)`` -> ``(B, n_maps, 1, T)``, prepended as bin 0 of each map.
    """

    def __init__(self, channels: int, n_maps: int):
        super().__init__()
        self.proj = nn.Linear(int(channels), int(n_maps))

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        pooled = feats.mean(dim=2).transpose(1, 2)  # (B, T, C)
        return self.proj(pooled).transpose(1, 2).unsqueeze(2)


class TranslationHead(nn.Module):
    """L2's minimal head: ``(B, M, G_in, T)`` off+log logits -> ``(B, M, G_out, T)``.

    1. ``W`` (`translation_matrix`, fixed, 0 parameters) re-grids every map;
       the L1 off logit lands on output bin 0, which on the LINEAR grid is the
       0 rev/s RATE bin: the loss trains it as a Gaussian peak at 0 like any
       other rate (no off class on linear grids — `models.salience_crf`). A
       log trunk has no bins below its ``fmin`` (27.5 rev/s), so the pooled
       off logit is the only thing that can say "0 rev/s" there; bins 1-27
       stay `UNREACHABLE_LOGIT`.
    2. ONE depthwise linear ``Conv2d(M, M, (9, 1), groups=M)`` along the rate
       bins ``1..``, identity-initialised (centre tap 1, bias 0): no
       normalisation, no activation, no cross-map mixing, ``M * 10``
       parameters. At init the head IS the re-gridding; all it can learn is a
       9-tap sharpening kernel per rotor. Bin 0 bypasses it.
    3. ``unreachable`` output bins (zero rows of ``W``) are set to the fixed
       logit `UNREACHABLE_LOGIT` after the conv. The head works in the LOGIT
       domain, where a zero row means logit 0 = probability 0.5 — the floor a
       ReLU on logits would impose — which would cost an unlearnable BCE on
       those bins and let the CRF prefer them over a weak true peak. The rates
       the trunk cannot express are therefore "almost surely empty", not
       "undecided".
    """

    def __init__(
        self,
        in_freqs: np.ndarray,
        out_freqs: np.ndarray,
        n_maps: int,
        kernel: int = TRANSLATION_KERNEL,
    ):
        super().__init__()
        w, unreachable = translation_matrix(in_freqs, out_freqs)
        self.register_buffer("W", torch.from_numpy(w))
        self.register_buffer("unreachable", torch.from_numpy(unreachable))
        self.n_maps = int(n_maps)
        self.conv = nn.Conv2d(
            self.n_maps,
            self.n_maps,
            (int(kernel), 1),
            padding=(int(kernel) // 2, 0),
            groups=self.n_maps,
        )
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, 0, int(kernel) // 2, 0] = 1.0
            assert self.conv.bias is not None
            self.conv.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.W.to(x.dtype)  # type: ignore[operator]
        y = torch.einsum("oi,bmit->bmot", w, x)
        rates = self.conv(y[:, :, 1:])
        mask = self.unreachable[1:].view(1, 1, -1, 1)  # type: ignore[index]
        rates = rates.masked_fill(mask, UNREACHABLE_LOGIT)
        return torch.cat([y[:, :, :1], rates], dim=2)


class SalienceRPSPredictor(nn.Module):
    """Base class for salience-map RPS baselines.

    Subclasses set the grid descriptor attributes (``fmin``, ``n_octaves``,
    ``over_sample``, ``n_bins``, ``bins_per_octave``, ``spec_sr``, ``spec_hop``)
    and implement ``forward`` (returning ``(B, n_maps * G, T_grid)`` logits) and
    ``num_grid_frames``. The four ladder baselines build their output through
    `_setup_rotor_output` / `_prepend_off` / `_translate_and_flatten`.
    """

    outputs_salience = True

    # Set by subclasses
    fmin: float
    n_octaves: float
    over_sample: int
    n_bins: int
    bins_per_octave: int
    spec_sr: int
    spec_hop: int

    # Optional explicit OUTPUT salience grid (rev/s). When set, the salience
    # target and the readout run on this grid instead of ``grid_params()``.
    out_freqs: np.ndarray | None = None
    #: Whether ``out_freqs`` bin 0 is an OFF state (per-rotor maps on a log
    #: grid, L1) rather than a rate bin (every linear grid; L0's log map has
    #: no bin 0 state either — its readout is threshold + Hungarian).
    out_has_off: bool = False

    def __init__(self, n_fft: int, hop_length: int, num_rotors: int):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length  # STFT hop (target output time grid)
        self.num_rotors = num_rotors

    # ── per-rotor output (L1 / L2) ───────────────────────────────────────────

    def _setup_rotor_output(
        self, log_freqs: np.ndarray, feat_channels: int, n_maps: int, translation_head: bool
    ) -> None:
        """Wire the output for the ladder level that ``n_maps`` / ``translation_head`` name.

        ``n_maps == 1`` (L0) leaves the shared map and ``out_freqs`` alone.
        ``n_maps > 1`` adds the off logit and puts ``out_freqs`` on
        ``[0.0] + log_freqs`` with ``out_has_off`` True (L1);
        ``translation_head`` then re-grids onto `RP_LINEAR_GRID`, a plain rate
        grid, ``out_has_off`` False (L2).
        """
        self.n_maps = int(n_maps)
        self.off_head: OffLogitHead | None = None
        self.translation: TranslationHead | None = None
        if self.n_maps == 1:
            if translation_head:
                raise ValueError("translation_head maps per-rotor layers: it needs n_maps > 1")
            return
        l1_grid = np.concatenate([[0.0], np.asarray(log_freqs, dtype=np.float64)])
        self.off_head = OffLogitHead(feat_channels, self.n_maps)
        self.out_freqs = l1_grid
        self.out_has_off = True
        if translation_head:
            self.translation = TranslationHead(l1_grid, RP_LINEAR_GRID, self.n_maps)
            self.out_freqs = RP_LINEAR_GRID.copy()
            self.out_has_off = False

    def _prepend_off(self, maps: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        """``(B, M, F, T)`` maps + ``(B, C, F', T)`` features -> ``(B, M, [1 +] F, T)``."""
        if self.off_head is None:
            return maps
        return torch.cat([self.off_head(feats).to(maps.dtype), maps], dim=2)

    def _translate_and_flatten(self, maps: torch.Tensor) -> torch.Tensor:
        """``(B, M, G_in, T)`` -> the ``(B, M * G, T)`` wire format, through L2's head."""
        if self.translation is not None:
            maps = self.translation(maps)
        b, m, g, t = maps.shape
        return maps.reshape(b, m * g, t)

    # ── grid / target ────────────────────────────────────────────────────────

    def grid_params(self) -> dict:
        """CQT grid descriptor consumed by the rps<->salience helpers."""
        return dict(
            fmin=self.fmin,
            n_octaves=self.n_octaves,
            over_sample=self.over_sample,
            n_bins=self.n_bins,
            bins_per_octave=self.bins_per_octave,
        )

    def output_freqs(self) -> np.ndarray:
        """Frequency grid (rev/s) of the salience the model actually emits.

        The explicit ``out_freqs`` grid if set (per-rotor levels: bin 0 is the
        off state), else the CQT input grid from ``grid_params()``.
        """
        if self.out_freqs is not None:
            return np.asarray(self.out_freqs, dtype=np.float64)
        return cqt_freq_grid(**self.grid_params())

    def _default_max_jump_bins(self) -> int:
        """Frames-to-frame jump cap, scaled to the output grid (~1.5 Hz)."""
        if self.out_freqs is None:
            return 3
        spacing = float(np.median(np.diff(np.asarray(self.out_freqs)))) or 1.0
        return max(3, int(round(1.5 / abs(spacing))))

    def num_grid_frames(self, n_samples: int) -> int:
        """Number of salience time frames the front-end emits for this length."""
        raise NotImplementedError

    def salience_target(
        self,
        rps: torch.Tensor,
        n_samples: int,
        *,
        rps_sr: float = 1000.0,
        blur_bins: int = 0,
    ) -> torch.Tensor:
        """Binary/soft shared-map salience target on this model's grid.

        Args:
            rps: ``(4, T_rps)`` or ``(B, 4, T_rps)`` raw RPS (Hz) at ``rps_sr``.
            n_samples: audio length (to size the time grid).
            blur_bins: frequency-axis smoothing half-width (0 = strictly binary).

        Returns:
            ``(n_bins, T_grid)`` or ``(B, n_bins, T_grid)``.
        """
        n_grid = self.num_grid_frames(n_samples)
        if self.out_freqs is not None:
            return rps_to_salience(
                rps,
                n_grid,
                freqs=self.out_freqs,
                hcqt_sr=self.spec_sr,
                hcqt_hop=self.spec_hop,
                rps_sr=rps_sr,
                blur_bins=blur_bins,
            )
        return rps_to_salience(
            rps,
            n_grid,
            **self.grid_params(),
            hcqt_sr=self.spec_sr,
            hcqt_hop=self.spec_hop,
            rps_sr=rps_sr,
            blur_bins=blur_bins,
        )

    def salience_target_from_frame_rps(
        self,
        rps_frames: torch.Tensor,
        n_samples: int,
        *,
        blur_bins: int = 0,
    ) -> torch.Tensor:
        """Build a shared-map BCE target from RPS already on the STFT frame grid.

        Direct-regression models consume the STFT-grid RPS directly; salience
        models convert it to their own time/frequency grid here, inside the
        training loop.

        Args:
            rps_frames: ``(4, T_stft)`` or ``(B, 4, T_stft)`` RPS in Hz.
            n_samples: waveform length used to size this model's salience grid.
            blur_bins: frequency-axis smoothing half-width.

        Returns:
            ``(n_bins_out, T_grid)`` or ``(B, n_bins_out, T_grid)``.
        """
        was_batched = rps_frames.dim() == 3
        if not was_batched:
            rps_frames = rps_frames.unsqueeze(0)

        n_grid = self.num_grid_frames(n_samples)
        rps_grid = F.interpolate(
            rps_frames.float(), size=n_grid, mode="linear", align_corners=False
        )
        salience = salience_target_from_resampled_rps(
            rps_grid, self.output_freqs(), blur_bins=blur_bins
        )

        if not was_batched:
            salience = salience.squeeze(0)
        return salience

    # ── forward / inference ──────────────────────────────────────────────────

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        """Return salience **logits** ``(B, n_maps * G, T_grid)``."""
        raise NotImplementedError

    def decode_logits(
        self,
        logits: torch.Tensor,
        n_samples: int,
        *,
        threshold: float = 0.3,
        max_jump_bins: int | None = None,
    ) -> torch.Tensor:
        """``(B, n_bins, T_grid)`` logits -> tracked RPS on the STFT grid, on-device.

        This IS the deployed shared-map (L0) readout: ``sigmoid`` -> threshold
        -> per-frame peaks -> frame-to-frame assignment tracking
        (:mod:`models.salience_tracker`) -> resample to ``N // hop + 1`` STFT
        frames. Training-time validation and :meth:`predict_rps` share it, so
        the monitor scores exactly what evaluation reports.
        """
        salience = torch.sigmoid(logits.float())
        if max_jump_bins is None:
            max_jump_bins = self._default_max_jump_bins()
        rps_grid, _merge = salience_to_rps_segmented(
            salience,
            num_rotors=self.num_rotors,
            freqs=self.output_freqs(),
            threshold=threshold,
            max_jump_bins=max_jump_bins,
        )  # (B, num_rotors, T_grid)

        # Resample grid frames -> STFT frames. Endpoint-to-endpoint shape-stretch,
        # matching how the GT RPS target is built (both cover the same audio
        # span, so the time axes align).
        t_stft = stft_time_frames(int(n_samples), self.hop_length)
        if rps_grid.shape[-1] != t_stft:
            rps_grid = F.interpolate(rps_grid, size=t_stft, mode="linear", align_corners=False)
        return rps_grid

    @torch.no_grad()
    def predict_rps(
        self,
        audio: torch.Tensor,
        *,
        threshold: float = 0.3,
        max_jump_bins: int | None = None,
        chunk_size: int = 8,
    ) -> torch.Tensor:
        """Audio -> tracked RPS on the STFT frame grid, ``(B, num_rotors, T_stft)``.

        The CNN forward is run in row-chunks of ``chunk_size``: validation clips
        are much longer than training clips, and LateDeep's (360, 1)
        distribution conv has activation memory ~ ``B·T``. Rows are
        independent, so chunking does not change results. ``chunk_size <= 0``
        disables it.
        """
        if chunk_size and chunk_size > 0 and audio.shape[0] > chunk_size:
            logits = torch.cat(
                [
                    self.forward(audio[i : i + chunk_size])
                    for i in range(0, audio.shape[0], chunk_size)
                ],
                dim=0,
            )
        else:
            logits = self.forward(audio)  # (B, n_bins, T_grid)
        return self.decode_logits(
            logits, audio.shape[-1], threshold=threshold, max_jump_bins=max_jump_bins
        )


class LateDeepSalience(LayerCRFReadout, SalienceRPSPredictor):
    """LateDeep multi-F0 CNN (Cuesta et al. ISMIR 2020) over an HCQT front end.

    Native 16 kHz. The published grid is ``fmin`` 32.7 Hz (C1), 60 bins per
    octave, 360 bins (`conf/model/latedeep_l0.yaml`); the class default 27.5
    (A0) reaches lower. The grid descriptor is read back from the front end,
    so ``fmin`` reshapes the target and the readout with no other change.

    Ladder (module docstring): ``n_maps=R`` widens the final 1x1 convolution
    (``squishy.1``) and adds the off logit from the trunk's last 8 feature
    maps (`LateDeep.features`); ``translation_head=True`` re-grids onto
    `RP_LINEAR_GRID`. The INPUT, the trunk and the hop-256 time grid are
    untouched at every level.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        fmin: float = 27.5,  # A0; matches basic-pitch ANNOTATIONS_BASE_FREQUENCY
        fused_branches: bool = False,
        frontend: nn.Module | None = None,
        n_maps: int = 1,
        translation_head: bool = False,
        **frontend_kwargs,
    ):
        super().__init__(n_fft, hop_length, num_rotors)
        from typing import cast

        from models.frontends import build_frontend
        from models.frontends.hcqt import HCQTFrontEnd
        from models.multif0.model import LateDeep

        if frontend is None:
            frontend = build_frontend("hcqt", phase=True, fmin=fmin, **frontend_kwargs)
        self.frontend = frontend
        # Grid descriptor reads HCQT-specific attributes; cast for static typing
        # (the frontend must expose fmin/n_octaves/over_sample/n_bins/sr/hop_length).
        fe = cast(HCQTFrontEnd, frontend)

        self.n_harmonics = fe.out_channels // 2 if fe.use_phase else fe.out_channels
        self.cnn = LateDeep(
            n_harmonics=self.n_harmonics, fused_branches=fused_branches, n_maps=int(n_maps)
        )

        # Grid descriptor (HCQT input params)
        self.fmin = fe.fmin
        self.n_octaves = fe.n_octaves
        self.over_sample = fe.over_sample
        self.bins_per_octave = 12 * fe.over_sample
        self.n_bins = fe.n_bins
        self.spec_sr = fe.sr
        self.spec_hop = fe.hop_length

        self._setup_rotor_output(
            cqt_freq_grid(**self.grid_params()),
            self.cnn.dist_bn.num_features,
            int(n_maps),
            translation_head,
        )

    def num_grid_frames(self, n_samples: int) -> int:
        from typing import cast

        from models.frontends.hcqt import HCQTFrontEnd

        return cast(HCQTFrontEnd, self.frontend).num_frames(n_samples)

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        feats = self.frontend(audio)  # (B, 2H, F, T) or (B, H, F, T)
        H = self.n_harmonics
        if getattr(self.frontend, "use_phase", True):
            mag = feats[:, :H, :, :]
            dphase = feats[:, H:, :, :]
        else:
            mag = feats
            dphase = torch.zeros_like(mag)
        trunk = self.cnn.features(mag, dphase)  # (B, 8, F, T)
        maps = self.cnn.logits(trunk)  # (B, M, F, T)
        return self._translate_and_flatten(self._prepend_off(maps, trunk))

    def to(self, *args, **kwargs):
        # HCQT nnAudio modules are not plain submodules — move them explicitly.
        if hasattr(self.frontend, "to"):
            self.frontend = self.frontend.to(*args, **kwargs)
        return super().to(*args, **kwargs)


class BasicPitchSalience(LayerCRFReadout, SalienceRPSPredictor):
    """Basic Pitch contour branch (Bittner et al. ICASSP 2022) as a salience model.

    The published contour grid: ``fmin`` 27.5, 36 bins per octave, 264 bins,
    hop 256. Trained from scratch at native 16 kHz; the pretrained 22.05 kHz
    path is not implemented.

    Ladder (module docstring): ``n_maps=R`` widens ``BasicPitch.contour_out``
    and adds the off logit from the contour branch's last features
    (`BasicPitch.contour_features`); ``translation_head=True`` re-grids onto
    `RP_LINEAR_GRID`. The CQT, the harmonic stacking and the contour trunk are
    untouched at every level.
    """

    def __init__(
        self,
        n_fft: int = 2048,
        hop_length: int = 512,
        num_rotors: int = 4,
        sr: int = 16000,
        n_harmonics: int = 8,
        pretrained: bool = False,
        freeze: bool = False,
        # Input contour grid; the defaults are the published 27.5 Hz /
        # 88-semitone / 3-bins-per-semitone grid.
        bp_fmin: float = 27.5,
        bins_per_semitone: int = 3,
        n_contour_semitones: int = 88,
        n_maps: int = 1,
        translation_head: bool = False,
    ):
        super().__init__(n_fft, hop_length, num_rotors)
        from models.basic_pitch.cqt import FFT_HOP
        from models.basic_pitch.model import BasicPitch

        if pretrained:
            # Pretrained kernels assume 22.05 kHz CQT input.
            raise NotImplementedError(
                "Zero-shot pretrained Basic Pitch (with 16k->22.05k resampling) is not "
                "implemented; train from scratch at native 16 kHz instead."
            )

        self.net = BasicPitch(
            n_harmonics=n_harmonics,
            sr=sr,
            fmin=bp_fmin,
            bins_per_semitone=bins_per_semitone,
            n_contour_semitones=n_contour_semitones,
            n_maps=int(n_maps),
        )
        if freeze:
            for p in self.net.parameters():
                p.requires_grad_(False)

        # Grid descriptor of the contour (input) grid — sample-rate-invariant.
        self.spec_sr = sr
        self.spec_hop = FFT_HOP  # 256
        self.fmin = bp_fmin
        self.over_sample = bins_per_semitone
        self.bins_per_octave = 12 * bins_per_semitone
        self.n_bins = self.net.contour_bins
        self.n_octaves = self.n_bins / self.bins_per_octave  # (unused; n_bins explicit)

        self._setup_rotor_output(
            cqt_freq_grid(**self.grid_params()),
            self.net.contour_bn.num_features,
            int(n_maps),
            translation_head,
        )

    def num_grid_frames(self, n_samples: int) -> int:
        # nnAudio CQT2010v2 emits n_samples // hop + 1 frames (matches the
        # reference 43844-sample -> 172-frame mapping at hop 256).
        return n_samples // self.spec_hop + 1

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        trunk = self.net.contour_features(audio)  # (B, C, time, F)
        maps = self.net.contour_out(trunk).transpose(2, 3)  # (B, M, F, time)
        return self._translate_and_flatten(self._prepend_off(maps, trunk.transpose(2, 3)))
