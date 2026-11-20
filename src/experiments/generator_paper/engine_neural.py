"""Four-source mixture fits of Doerfler & Wyse's EUSIPCO 2026 PTR and HPN.

The official controller, synthesizer, loss, optimizer and scheduler are used
from a separate pinned checkout. This module only adapts the controls, lifts
mono synthesis to four sources and provides the calibration/checkpoint seam.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
import math
import os
import platform
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from functools import cache
from importlib.metadata import version
from pathlib import Path
from threading import Lock
from types import ModuleType, SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .cases import FlightClip


UPSTREAM_URL = "https://github.com/rdoerfler/ptr-model"
UPSTREAM_COMMIT = "9eed6b0d26e1d65cb6b892a125215dca32c97a37"
_SCHEMA = "generator-paper-engine-neural/1"
_N_ROTORS = 4
_SR = 16000
_TORQUE = 0.0
_GRAD_CLIP = 5.0
_BANK_BUFFERS = {f"synthesizer.noise_bank.{name}" for name in ("filterbank", "noisebank")}
_EMPTY_SHA256 = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
_SOURCE_SHA256 = {
    "configs/__init__.py": _EMPTY_SHA256,
    "configs/base_config.py": "fffec9b1ce63a1d06d155930c17ae7a3db03ec823a51590bb6da0da6460a8b6c",
    "src/__init__.py": _EMPTY_SHA256,
    "src/audio/__init__.py": _EMPTY_SHA256,
    "src/audio/noisebank.py": "dd72967536faa54378cec6a5caaa2f3c0f3dd346de5f64d78a53f43d20be85cf",
    "src/data/__init__.py": _EMPTY_SHA256,
    "src/data/dataloader.py": "08bb3c98278180067ea388f3e022c44f3120fb0e98786dbfdee877e82580e918",
    "src/models/__init__.py": _EMPTY_SHA256,
    "src/models/model.py": "18fdc635fd263c96bbba96a08df0ca7f860cc3abed3a0316b297b15fe106b646",
    "src/models/ptr_model.py": "7584e040a92a7456d4e0b7d83697ce62c0ab6761861e27a98509208cc8776a78",
    "src/models/ptr_synth.py": "ca64fe4dd2da46532ad5f84a8e5fb7b0e4f17d0a01688e0acd23ea5316d048ff",
    "src/models/hpn_model.py": "e2bbcb668526806ca088f28094455ff9ace17f379a66d1a56d9b74c1273e8060",
    "src/models/hpn_synth.py": "ee3cd27f90358d133d5b8eb8c5aeb09837dd4a60bf70bf3a6b9d45e84c179e8a",
    "src/training/__init__.py": _EMPTY_SHA256,
    "src/training/loss.py": "c085fe030d812c4057d97ec1e03e3d90532b32539abf439786c7a06f8d9fb2f8",
    "src/training/trainer.py": "7450014f650dac68425435a471d8967ad32298612f48772955d4e50226cd7cc6",
}
_IMPORT_LOCK = Lock()


@dataclass(frozen=True)
class EngineSettings:
    """Default production budget. Smaller epochs/batches are explicit smoke settings."""

    epochs: int = 100
    batch_size: int = 8
    chunk_size: int = 64000  # 4 s; preserve the exact calibration supports.


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _runtime(upstream: Path | None) -> SimpleNamespace:
    root = upstream if upstream is not None else os.environ.get("HNS_PTR_UPSTREAM")
    if root is None:
        raise ValueError(f"Set HNS_PTR_UPSTREAM to a checkout of {UPSTREAM_URL}@{UPSTREAM_COMMIT}")
    with _IMPORT_LOCK:
        return _import_runtime(Path(root).expanduser().resolve())


@cache
def _import_runtime(root: Path) -> SimpleNamespace:
    for name, digest in _SOURCE_SHA256.items():
        path = root / name
        if not path.is_file() or _sha256(path) != digest:
            raise ValueError(
                f"Expected unmodified PTR source {name} from {UPSTREAM_COMMIT} in {root}"
            )
    if version("torchlpc") != "0.6":
        raise ValueError("The pinned PTR implementation requires torchlpc==0.6")

    # The official files use private relative imports except for this one
    # absolute noisebank import. Don't put upstream src/configs on sys.path.
    prefix = f"_hns_ptr_{hashlib.sha256(os.fsencode(root)).hexdigest()}"
    module = ModuleType(prefix)
    module.__path__ = [str(root)]
    sys.modules[prefix] = module
    alias = "src.audio.noisebank"
    previous_alias = sys.modules.get(alias)
    previous_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        sys.modules[alias] = importlib.import_module(f"{prefix}.{alias}")
        return SimpleNamespace(
            torch=importlib.import_module("torch"),
            base=importlib.import_module(f"{prefix}.configs.base_config"),
            data=importlib.import_module(f"{prefix}.src.data.dataloader"),
            model=importlib.import_module(f"{prefix}.src.models.model"),
            loss=importlib.import_module(f"{prefix}.src.training.loss"),
            trainer=importlib.import_module(f"{prefix}.src.training.trainer"),
        )
    finally:
        sys.dont_write_bytecode = previous_bytecode
        if previous_alias is None:
            sys.modules.pop(alias, None)
        else:
            sys.modules[alias] = previous_alias


def _device(torch: Any, device: str) -> Any:
    selected = torch.device(device)
    if selected.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested for PTR/HPN but is not available")
        if selected.index is None:
            return torch.device("cuda", torch.cuda.current_device())
    elif selected.type != "cpu":
        raise ValueError(f"PTR/HPN supports cpu/cuda, not {selected}")
    return selected


@contextmanager
def _seeded(torch: Any, seed: int, device: Any = None):
    """Scope native CPU and selected CUDA RNGs without resetting other devices."""
    use_cuda = device is not None and device.type == "cuda"
    with torch.random.fork_rng(devices=[device.index] if use_cuda else []):
        torch.random.default_generator.manual_seed(seed)
        if use_cuda:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(seed)
        yield


def _config(api: SimpleNamespace, method: str, settings: EngineSettings) -> dict[str, Any]:
    if method not in ("ptr", "hpn"):
        raise ValueError(f"Unknown engine baseline {method!r}; expected 'ptr' or 'hpn'")
    cfg = copy.deepcopy(api.base.get_config())
    dataset = cfg["dataset"]
    dataset.pop("dataset_path")
    dataset.update(chunk_size=settings.chunk_size, control_channels=[0, 1], audio_channels=[])
    cfg["model"]["model_type"] = method
    cfg["trainer"].update(epochs=settings.epochs, batch_size=settings.batch_size)
    return {key: cfg[key] for key in ("dataset", "model", "synth", "trainer", "loss")}


def _rps_array(rps: np.ndarray) -> np.ndarray:
    tracks = np.asarray(rps)
    if (
        tracks.ndim != 2
        or tracks.shape[0] != _N_ROTORS
        or tracks.shape[1] == 0
        or not np.isfinite(tracks).all()
        or np.any(tracks <= 0)
    ):
        raise ValueError("Engine synthesis needs four finite, positive RPS tracks (4, samples)")
    return tracks


def _native_features(torch: Any, extractor: Any, rps: Any, rpm_reference: float) -> tuple[Any, Any]:
    # In both controllers u -> u*10000/60. PTR then integrates at half that
    # frequency and HPN's nominal lowest order is 0.5, as is the loss comb.
    # Thus the unmodulated native cycle (not the cylinder firing rate) is RPS.
    rpm_control = rps * (120.0 / rpm_reference)
    waveform = torch.stack((rpm_control, torch.full_like(rpm_control, _TORQUE)), dim=1)
    frames = torch.stack([extractor(controls)[0] for controls in waveform])
    return frames, waveform


def _prepare(api: SimpleNamespace, cfg: dict, clips: Sequence[FlightClip]):
    torch = api.torch
    if not clips:
        raise ValueError("Engine calibration needs at least one FlightClip")
    extractor = api.data.create_extractor(cfg)
    dataset = cfg["dataset"]
    length = dataset["chunk_size"]
    # The native STFT uses centered reflection padding. HarmonicLoss zero-pads
    # its unfolded windows separately and expects at least two per example.
    loss = cfg["loss"]
    harmonic = loss["harm_loss_params"]
    minimum = max(
        2 * extractor.shift + 1,
        max(loss["stft_loss_params"]["fft_sizes"]) // 2 + 1,
        harmonic["win_size"] + harmonic["hop_size"],
    )
    if length % extractor.frame_size or length < minimum:
        raise ValueError(
            f"Engine chunk_size must be a multiple of {extractor.frame_size} and at least {minimum}"
        )
    hop = int((1 - dataset["chunk_overlap"]) * length)
    n_mics = clips[0].audio.shape[0]
    if not n_mics:
        raise ValueError("Engine calibration needs at least one microphone")
    features, controls, targets, windows = [], [], [], []
    for clip in clips:
        tracks = _rps_array(clip.rps)
        n = tracks.shape[1]
        if clip.sr != _SR or clip.audio.shape != (n_mics, n) or n < length:
            raise ValueError(
                f"{clip.key}: expected at least {length} samples on a common {_SR} Hz grid"
            )
        if clip.provenance.get("split") != "calibration":
            raise ValueError(f"{clip.key}: engine fitting requires the calibration split")
        if not np.isfinite(clip.audio).all():
            raise ValueError(f"{clip.key}: nonfinite calibration audio")
        rps = torch.tensor(tracks, dtype=torch.float32)
        audio = torch.tensor(clip.audio, dtype=torch.float32)
        starts = list(range(0, n - length + 1, hop))
        for start in starts:
            sl = slice(start, start + length)
            x, u = _native_features(torch, extractor, rps[..., sl], dataset["rpm_range"][-1])
            features.append(x)
            controls.append(u)
            targets.append(audio[:, sl])
        windows.append(
            {
                "key": clip.key,
                "recording_id": clip.recording_id,
                "split": clip.provenance["split"],
                "rig": clip.rig,
                "regime": clip.regime,
                "dataset_pin": clip.provenance.get("dataset_pin"),
                "label_key": clip.provenance.get("label_key"),
                "start_s": clip.start_s,
                "samples": n,
                "chunk_starts_samples": starts,
            }
        )

    stacked = torch.stack(features)
    means, stds = [], []
    for rotor in range(_N_ROTORS):
        scaler = api.data.create_scaler(cfg)
        stacked[:, rotor] = scaler.fit_transform(stacked[:, rotor])
        means.append(scaler.mean.flatten().tolist())
        stds.append(scaler.std.flatten().tolist())
    return stacked, torch.stack(controls), torch.stack(targets), means, stds, windows


def _mixture_criterion(api: SimpleNamespace, cfg: dict, device: Any) -> Any:
    torch = api.torch
    criterion = api.loss.create_loss_fn(cfg, device)

    class LiftedHarmonicLoss(torch.nn.Module):
        def __init__(self, native: Any):
            super().__init__()
            self.native = native

        def forward(self, prediction: Any, target: Any, rpm: Any) -> Any:
            # Treat the four tracked combs as four examples for the *native*
            # harmonic loss/reduction, each on the same complete mixture.
            return self.native(
                prediction.repeat_interleave(_N_ROTORS, dim=0),
                target.repeat_interleave(_N_ROTORS, dim=0),
                rpm.reshape(-1, 1, rpm.shape[-1]),
            )

    criterion.harmonic_loss = LiftedHarmonicLoss(criterion.harmonic_loss)
    return criterion


def _new_source(api: SimpleNamespace, cfg: dict, device: Any, seed: int) -> Any:
    # The native builder initializes its noise basis first, on CPU. Its seed
    # reconstructs both length-dependent noise buffers; weights remain frozen.
    with _seeded(api.torch, seed):
        return api.model.create_model(cfg, device)


def _mixed_wave(sources: Any, features: Any, controls: Any) -> Any:
    total = sources[0](features[:, 0], controls[:, 0])
    for i in range(1, _N_ROTORS):
        total = total + sources[i](features[:, i], controls[:, i])
    return total


def _fit_mic(
    api: SimpleNamespace,
    cfg: dict,
    criterion: Any,
    x: Any,
    u: Any,
    target: Any,
    source_seeds: list[int],
    seed: int,
    device: Any,
    mic: int,
):
    torch = api.torch
    trainer = cfg["trainer"]
    batch_size = min(trainer["batch_size"], len(x))
    steps_per_epoch = len(x) // batch_size
    sources = torch.nn.ModuleList([_new_source(api, cfg, device, s) for s in source_seeds])
    optimizer = api.trainer.create_optimizer(cfg, sources)
    # Native create_scheduler only needs len(dataloader), which is the actual
    # number of optimizer updates here (upstream loader's len can be wrong).
    scheduler = api.trainer.create_scheduler(cfg, optimizer, range(steps_per_epoch))
    rng = np.random.RandomState(seed)
    indices = np.arange(len(x))
    parameters = tuple(sources.parameters())
    step_history, epoch_history = [], []
    for epoch in range(1, trainer["epochs"] + 1):
        rng.shuffle(indices)
        epoch_start = len(step_history)
        for start in range(0, steps_per_epoch * batch_size, batch_size):
            selection = indices[start : start + batch_size]
            inputs, controls, mixture = x[selection], u[selection], target[selection]
            optimizer.zero_grad()
            prediction = _mixed_wave(sources, inputs, controls)
            loss, terms = criterion(prediction, mixture, controls[:, :, 0])
            row = {
                "total_loss": loss.item(),
                "harmonic_loss": terms["harmonic"],
                "stft_loss": terms["stft"],
            }
            if not all(math.isfinite(value) for value in row.values()):
                raise RuntimeError(
                    f"{cfg['model']['model_type']} mic {mic}: nonfinite loss at epoch {epoch}: {row}"
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                parameters, _GRAD_CLIP, error_if_nonfinite=True
            )
            step_history.append(
                {
                    **row,
                    "epoch": epoch,
                    "step": len(step_history) + 1,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                    "gradient_norm": grad_norm.item(),
                }
            )
            optimizer.step()
            scheduler.step()
        recent = step_history[epoch_start:]
        summary = {name: math.fsum(row[name] for row in recent) / len(recent) for name in row}
        epoch_history.append({"epoch": epoch, **summary})
        print(
            f"{cfg['model']['model_type']} mic {mic} epoch {epoch}/{trainer['epochs']}: "
            f"loss={summary['total_loss']:.6f}",
            flush=True,
        )
    states = [
        {k: v.detach().cpu() for k, v in source.state_dict().items() if k not in _BANK_BUFFERS}
        for source in sources
    ]
    if any(not torch.isfinite(v).all() for state in states for v in state.values()):
        raise RuntimeError(f"{cfg['model']['model_type']} mic {mic}: nonfinite final state")
    return states, {
        "mic": mic,
        "source_initialization_seeds": source_seeds,
        "parameter_count": sum(param.numel() for param in parameters),
        "batch_size": batch_size,
        "steps_per_epoch": steps_per_epoch,
        "optimizer_steps": len(step_history),
        "steps": step_history,
        "epochs": epoch_history,
        "final_loss": epoch_history[-1]["total_loss"],
    }


def fit_engine(
    clips: Sequence[FlightClip],
    checkpoint: Path,
    *,
    method: str,
    seed: int = 0,
    device: str = "cuda",
    upstream: Path | None = None,
    settings: EngineSettings | None = None,
) -> dict[str, Any]:
    """Fit independently per microphone to mixed calibration audio, with no isolated targets."""
    seed = int(seed)
    settings = settings if settings is not None else EngineSettings()
    if settings.epochs < 1 or settings.batch_size < 1:
        raise ValueError("Engine epochs and batch_size must be positive")
    api = _runtime(upstream)
    torch = api.torch
    selected = _device(torch, device)
    cfg = _config(api, method, settings)
    inputs, controls, targets, means, stds, calibration = _prepare(api, cfg, clips)
    inputs, controls, targets = (item.to(selected) for item in (inputs, controls, targets))
    criterion = _mixture_criterion(api, cfg, selected)
    mic_fits, state_dicts = [], []
    n_mics = targets.shape[1]
    seed_stream = np.random.default_rng(seed)
    with _seeded(torch, seed, selected):
        for mic in range(n_mics):
            source_seeds = seed_stream.integers(0, np.iinfo(np.int64).max, size=_N_ROTORS).tolist()
            states, history = _fit_mic(
                api,
                cfg,
                criterion,
                inputs,
                controls,
                targets[:, mic : mic + 1],
                source_seeds,
                seed,
                selected,
                mic,
            )
            state_dicts.append(states)
            mic_fits.append(history)

    trainer = cfg["trainer"]
    settings_metadata = {
        "dataset": cfg["dataset"],
        "model": cfg["model"],
        "synth": cfg["synth"],
        "loss": cfg["loss"],
        "trainer": trainer,
        "native_synth_constructor_defaults": {"num_cylinders": 8}
        if method == "ptr"
        else {"num_modulators": 4},
    }
    report = {
        "schema": _SCHEMA,
        "status": "completed",
        "method": method,
        "seed": seed,
        "production_settings": seed == 0 and settings == EngineSettings(),
        "settings": asdict(settings),
        "native_settings": settings_metadata,
        "upstream": {
            "url": UPSTREAM_URL,
            "commit": UPSTREAM_COMMIT,
            "files_sha256": dict(_SOURCE_SHA256),
        },
        "sample_rate": _SR,
        "n_mics": n_mics,
        "n_rotors": _N_ROTORS,
        "parameter_count": sum(fit["parameter_count"] for fit in mic_fits),
        "epochs": settings.epochs,
        "steps_per_epoch": mic_fits[0]["steps_per_epoch"],
        "optimizer_steps_per_mic": mic_fits[0]["optimizer_steps"],
        "selected_epoch": settings.epochs,
        "selected_phase": "final_calibration_epoch",
        "optimizer": {
            "type": "AdamW",
            "lr": trainer["learning_rate"],
            "weight_decay": trainer["weight_decay"],
            "betas": trainer["betas"],
            "gradient_clip_norm": _GRAD_CLIP,
            "scheduler": {
                "type": "OneCycleLR",
                **{
                    name: trainer[name]
                    for name in (
                        "max_lr",
                        "pct_start",
                        "div_factor",
                        "final_div_factor",
                        "anneal_strategy",
                    )
                },
                "cycle_momentum": True,
                "base_momentum": 0.85,
                "max_momentum": 0.95,
            },
        },
        "controls": {
            "normalized_rpm": "120 * shaft_RPS / 10000; no clamping",
            "native_shaft_hz": "normalized_rpm * 10000 / 60",
            "native_fundamental_hz": "native_shaft_hz / 2 (PTR cycle and nominal HPN lattice)",
            "fixed_normalized_torque": _TORQUE,
            "torque_deltas": [0.0, 0.0],
            "propulsion_gate": 0.02**0.7,
            "dfco_gate": 0.02,
            "feature_mean_by_rotor": means,
            "feature_std_by_rotor": stds,
        },
        "initialization": {
            "models": "native random initialization, independent source models per microphone; no pretrained weights",
            "source_seed_stream": "NumPy default_rng(base seed), microphone-major then rotor-major",
            "shuffle": "NumPy RandomState(base seed), native shuffle per epoch, same ordering for every microphone",
            "training_stochasticity": "native stochastic forwards using a seeded Torch RNG; noise basis fixed per source",
            "render_rng": "one integer from supplied generator seeds Torch for native stochastic forwards; "
            "the native seeded noise basis is reconstructed for each synthesis length",
            "cuda_numerics": "TorchLPC CUDA uses floating-point atomics; RNG is replayed but native kernels "
            "do not guarantee bitwise CUDA output",
        },
        "native_pipeline_deviations": [
            "Four intact sources per mono microphone target, summed before scoring; native spectral term once per "
            "mixture, native harmonic term reduced over mixture/track pairs for all four tracks.",
            "Fit all supplied calibration clips without validation or model selection. Use observed clips without "
            "resampling, peak normalization or other waveform transformation; standardize controller features per rotor.",
            "Exact 4 s calibration supports use 64000 instead of 65536 samples in production; native 50% overlap, "
            "include the final window when it ends exactly at the support boundary.",
            "Keep complete batches (native loader drops a final full batch by mistake); use one short batch "
            "if the calibration subset has fewer windows than the configured batch size. This is not the paper's update budget.",
        ],
        "paper_code_differences": [
            "Pinned code AdamW weight decay 0.005 rather than paper text 0.01; use native OneCycleLR settings "
            "including a maximum scheduled learning rate of 0.0002, not the optimizer's configured 0.001.",
            *(
                [
                    "Pinned PTR passes learned offsets and jitter to the synthesizer but its native forward uses them "
                    "only for tensor dimensions. Preserve the fixed firing angles actually used by the source.",
                    "Pinned PTR uses the delay-robust L1 bound on direct-form coefficients and a 0.45 feedback gain "
                    "exponent, rather than the triangle-only constraint and 0.35 exponent described in the paper.",
                ]
                if method == "ptr"
                else []
            ),
        ],
        "render_length_policy": (
            "For inference on short/off-frame-grid requests, hold the final positive control to cover complete "
            "feature frames and the native reflection margin, and return the requested prefix of native synthesis.",
        ),
        "calibration": calibration,
        "calibration_keys": [item["key"] for item in calibration],
        "calibration_recordings": sorted({item["recording_id"] for item in calibration}),
        "calibration_splits": sorted({item["split"] for item in calibration}),
        "calibration_chunks": len(inputs),
        "mic_fits": mic_fits,
        "device": str(selected),
        "software": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cuda": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(selected)
            if selected.type == "cuda"
            else platform.machine(),
            "cudnn": torch.backends.cudnn.version() if selected.type == "cuda" else None,
            "packages": {
                name: version(name)
                for name in (
                    "torch",
                    "torchaudio",
                    "torchlpc",
                    "numba",
                    "llvmlite",
                    "numpy",
                    "tqdm",
                )
            },
        },
    }
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    weights = checkpoint.with_name(checkpoint.name + ".weights.pt")
    torch.save(state_dicts, weights)
    report["weights"] = {"file": weights.name, "sha256": _sha256(weights)}
    serialized = json.dumps(report, indent=2, allow_nan=False)
    checkpoint.write_text(serialized + "\n", encoding="utf-8")
    return json.loads(serialized)


@dataclass
class EngineFit:
    """A loaded calibration-only expert; caches native instances at the active render length."""

    report: dict[str, Any]
    _api: SimpleNamespace
    _device: Any
    _states: list
    _models: list = field(default_factory=list, init=False, repr=False)
    _samples: int = field(default=0, init=False, repr=False)

    def _models_for(self, cfg: dict[str, Any], samples: int) -> list:
        if samples != self._samples:
            self._models.clear()
            self._samples = 0
            cfg["dataset"]["chunk_size"] = samples
            for states, fit in zip(self._states, self.report["mic_fits"], strict=True):
                sources = []
                for state, seed in zip(states, fit["source_initialization_seeds"], strict=True):
                    source = _new_source(self._api, cfg, self._device, seed)
                    missing, unexpected = source.load_state_dict(state, strict=False)
                    if set(missing) != _BANK_BUFFERS or unexpected:
                        raise ValueError(
                            f"Incompatible engine weights: missing={missing}, unexpected={unexpected}"
                        )
                    sources.append(source.eval().requires_grad_(False))
                self._models.append(sources)
            self._samples = samples
        return self._models

    def render(self, rps: np.ndarray, *, sr: int, rng: np.random.Generator) -> np.ndarray:
        """Render from positive time-last RPS only, with no output scaling."""
        if sr != self.report["sample_rate"]:
            raise ValueError(
                f"Engine checkpoint requires {self.report['sample_rate']} Hz controls, got {sr}"
            )
        tracks = _rps_array(rps)
        torch = self._api.torch
        cfg = _config(self._api, self.report["method"], EngineSettings(**self.report["settings"]))
        extractor = self._api.data.create_extractor(cfg)
        length = tracks.shape[-1]
        frame_size = extractor.frame_size
        samples = max(length, 2 * extractor.shift + 1)
        samples = (samples + frame_size - 1) // frame_size * frame_size
        models = self._models_for(cfg, samples)
        seed = int(rng.integers(0, np.iinfo(np.int64).max))
        with torch.inference_mode(), _seeded(torch, seed, self._device):
            controls = torch.tensor(tracks, dtype=torch.float32, device=self._device)
            if samples > length:
                controls = torch.cat(
                    (controls, controls[:, -1:].expand(-1, samples - length)), dim=-1
                )
            features, controls = _native_features(
                torch, extractor, controls, cfg["dataset"]["rpm_range"][-1]
            )
            report_controls = self.report["controls"]
            mean = features.new_tensor(report_controls["feature_mean_by_rotor"]).unsqueeze(1)
            std = features.new_tensor(report_controls["feature_std_by_rotor"]).unsqueeze(1)
            features = self._api.data.FeatureScaler(mean, std).transform(features)
            out = torch.stack(
                [
                    _mixed_wave(sources, features[None], controls[None])[0, 0, :length]
                    for sources in models
                ]
            )
            if not torch.isfinite(out).all():
                raise RuntimeError(f"{self.report['method']} synthesis produced nonfinite audio")
            return out.cpu().numpy()


def load_engine(
    checkpoint: Path, *, device: str = "cpu", upstream: Path | None = None
) -> EngineFit:
    """Load a movable manifest/weights pair for RPS-only rendering on CPU or CUDA."""
    checkpoint = Path(checkpoint)
    report = json.loads(checkpoint.read_text(encoding="utf-8"))
    if report.get("schema") != _SCHEMA or report.get("status") != "completed":
        raise ValueError(f"Not a completed engine checkpoint: {checkpoint}")
    if (
        report["upstream"]["commit"] != UPSTREAM_COMMIT
        or report["upstream"]["files_sha256"] != _SOURCE_SHA256
    ):
        raise ValueError(
            f"Engine checkpoint does not match the pinned PTR source {UPSTREAM_COMMIT}"
        )
    name = report["weights"]["file"]
    if name != Path(name).name:
        raise ValueError("Engine checkpoint weights must be adjacent to the manifest")
    weights = checkpoint.with_name(name)
    if _sha256(weights) != report["weights"]["sha256"]:
        raise ValueError(f"Engine weights SHA-256 mismatch: {weights}")
    api = _runtime(upstream)
    selected = _device(api.torch, device)
    states = api.torch.load(weights, map_location="cpu", weights_only=True)
    if len(states) != report["n_mics"] or any(len(sources) != _N_ROTORS for sources in states):
        raise ValueError("Engine checkpoint source count does not match its manifest")
    return EngineFit(report, api, selected, states)
