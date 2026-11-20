"""Published DDSP-Piano (Renault, Mignot & Roebel, JAES 2023) on rotor RPS.

The native dafx22 graph sees four sustained voices and one recording-environment
ID per microphone. Its only training target is that microphone's mixed audio.
TensorFlow/DDSP and the piano checkout are needed only to fit or load this arm.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import platform
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from shutil import copyfile
from tempfile import TemporaryDirectory
from threading import RLock
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator

    from .cases import FlightClip

UPSTREAM_URL = "https://github.com/lrenault/ddsp-piano"
UPSTREAM_COMMIT = "e868b7ccd3fe31b39132048a72561d7fcf1b465f"
DDSP_VERSION = "3.7.0"
_CONFIG = "ddsp_piano/configs/dafx22.gin"
_SCHEMA = "generator-paper-ddsp-piano/1"
_SR, _FRAME_RATE, _VOICES = 16_000, 250, 4
_HOP = _SR // _FRAME_RATE
_SEGMENT_S, _OVERLAP = 3, 0.5
_SEGMENT_SAMPLES = _SEGMENT_S * _SR
_SEGMENT_HOP = int(_SEGMENT_SAMPLES * (1 - _OVERLAP))
_VELOCITY = 64 / 127
_LEARNING_RATES = (1e-3, 1e-5, 1e-3)
_PHASE_NAMES = ("amplitude_noise", "frequency_tuning", "amplitude_noise")
_GIN_LOCK = RLock()

# Validate the actual imported sources, including when staged without .git.
_PINNED_FILES = {
    "ddsp_piano/__init__.py": "810a309072a0bc45ac6bc4627884617b0886023e488945e15d778dbf0918d4e0",
    _CONFIG: "27e7761b82fbe3c760c7c1dd56e66f5d56bd05a976c43b60c974eead75a1493e",
    "ddsp_piano/data_pipeline.py": "a582a39e1b673b1d8fa9a1cf2e22e92d79360c41ea7d4e2a54007a9b95e37d96",
    "ddsp_piano/default_model.py": "bed4604258071353099d74efbc4d6017af9dc282cf5b992f4e20d5a678bebabf",
    "ddsp_piano/modules/__init__.py": "918ea1fe513fb3d70b1787d9ed7bc0adb8e544fb2cdd6f30c118bfcadd722173",
    "ddsp_piano/modules/fdn_reverb.py": "3649e723133ff3c8d4ca4cb7026cbd953f1965d9ea33eb0446ac273129aa92b2",
    "ddsp_piano/modules/filtered_noise_synth.py": "fbbfc726a9daccbfa4d768736ee1622aa26bd24a78cd324eff4494b731756735",
    "ddsp_piano/modules/inharm_synth.py": "2b125a740dc504319a353dec45f57ea40b194d74986cec9bc5f115aadab9817f",
    "ddsp_piano/modules/losses.py": "182f087d532a3c91dff3e645076516cbf891f6160ec83a4008aecfecceb86323",
    "ddsp_piano/modules/piano_model.py": "1d54ce03f191085db218ff21adec4156d0662142c617eb98627acac1fdc7f6ef",
    "ddsp_piano/modules/polyphonic_dag.py": "284431fd3e3c2d2b8a06a76be597810b2746c4bd1a23c2aaaece03cde6c6e7a4",
    "ddsp_piano/modules/sub_modules.py": "c6ed86c176538e3b3296b595cee6b7268cef49c08f5e86819ffcdcdecbdfc656",
    "ddsp_piano/modules/surrogate_synth.py": "ac507842d02a8e3138303c1f85da0ba5e27c43c8d415519074f5ab50e6dff113",
    "ddsp_piano/utils/__init__.py": "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
    "ddsp_piano/utils/io_utils.py": "0b6ebcaebd3fbb5b269da4a3b2c357ed6bf8b8a26fa65d7b155d64bcff3cba58",
    "ddsp_piano/utils/midi_encoders.py": "fc8da1caf9b4402fa0ae368921995019e3dd1f0e363836fc62bb3e6fb4f172fc",
}


@dataclass(frozen=True)
class PianoSettings:
    """The published phase order, with an explicit bounded update budget."""

    epochs: tuple[int, int, int] = (2, 1, 3)
    steps_per_epoch: int = 1000
    batch_sizes: tuple[int, int, int] = (6, 3, 6)

    def __post_init__(self) -> None:
        if len(self.epochs) != 3 or len(self.batch_sizes) != 3:
            raise ValueError("DDSP-Piano requires all three training phases")
        if min(*self.epochs, self.steps_per_epoch, *self.batch_sizes) < 1:
            raise ValueError("epochs, steps per epoch, and batch sizes must be positive")


@dataclass(frozen=True)
class _Native:
    tf: Any
    gin: Any
    ddsp: Any
    models: Any
    trainers: Any
    data: Any
    config: Path


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _native(upstream: Path | None) -> _Native:
    root = upstream or os.environ.get("HNS_PIANO_UPSTREAM")
    if root is None:
        raise ValueError(
            f"Set HNS_PIANO_UPSTREAM to a checkout of {UPSTREAM_URL}@{UPSTREAM_COMMIT}"
        )
    root = Path(root).resolve()
    for name, expected in _PINNED_FILES.items():
        path = root / name
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"{path} does not match DDSP-Piano {UPSTREAM_COMMIT}")

    with _GIN_LOCK:
        tf = importlib.import_module("tensorflow")
        gin = importlib.import_module("gin")
        ddsp = importlib.import_module("ddsp")
        if ddsp.__version__ != DDSP_VERSION:
            raise ValueError(f"Expected DDSP {DDSP_VERSION}, found {ddsp.__version__}")
        sys.path.insert(0, str(root))
        try:
            piano = importlib.import_module("ddsp_piano")
        finally:
            sys.path.pop(0)
        # A previous import may point to another copy of the same pinned source.
        if piano.__file__ is None:
            raise RuntimeError("Imported DDSP-Piano package has no source file")
        loaded_root = Path(piano.__file__).resolve().parents[1]
        if loaded_root != root and any(
            not (loaded_root / name).is_file() or _sha256(loaded_root / name) != expected
            for name, expected in _PINNED_FILES.items()
        ):
            raise RuntimeError(f"A different DDSP-Piano source is already imported: {loaded_root}")
        return _Native(
            tf,
            gin,
            ddsp,
            importlib.import_module("ddsp.training.models"),
            importlib.import_module("ddsp.training.trainers"),
            piano.data_pipeline,
            root / _CONFIG,
        )


def _device(tf: Any, device: str) -> str:
    if device == "cpu":
        return "/CPU:0"
    name, sep, index = device.partition(":")
    if name != "cuda" or (sep and not index.isdecimal()):
        raise ValueError(f"Invalid DDSP-Piano device {device!r}; use cpu or cuda[:index]")
    gpu = int(index) if sep else 0
    if gpu >= len(tf.config.list_physical_devices("GPU")):
        raise RuntimeError(f"TensorFlow cannot see requested device {device!r}")
    return f"/GPU:{gpu}"


def _conditioning(rps: np.ndarray) -> np.ndarray:
    """Native (frames, voices, pitch/velocity) control, sampled at 0, 4, ... ms."""
    rps = np.asarray(rps)
    if rps.ndim != 2 or rps.shape[0] != _VOICES or not rps.shape[1]:
        raise ValueError("DDSP-Piano requires four nonempty, audio-rate RPS tracks")
    if not np.isfinite(rps).all() or np.any(rps <= 0):
        raise ValueError("DDSP-Piano RPS must be finite and strictly positive")
    pitch = 69 + 12 * (np.log2(rps[:, ::_HOP].T, dtype=np.float64) - math.log2(440))
    result = np.zeros((*pitch.shape, 2), dtype=np.float32)
    result[..., 0] = pitch
    result[0, :, 1] = _VELOCITY
    return result


def _duration(n_frames: int) -> float:
    # The native Z-encoder computes int(duration * frame_rate).
    return math.nextafter(n_frames / _FRAME_RATE, math.inf)


def _model(native: _Native, n_mics: int, *, inference: bool) -> Any:
    gin = native.gin
    with _GIN_LOCK:
        gin.clear_config()
        try:
            gin.parse_config_file(str(native.config))
            gin.bind_parameter("%n_instruments", n_mics)
            gin.bind_parameter("%n_synths", _VOICES)
            gin.bind_parameter("%inference", inference)
            gin.bind_parameter("sub_modules.MultiInstrumentReverb.inference", inference)
            return native.models.get_model()
        finally:
            gin.clear_config()


def _build(tf: Any, model: Any, batch_size: int) -> None:
    # Native Parallelizer caches the batch size on its first call. The learned
    # shapes do not depend on duration; build with one frame before restoring.
    model.z_encoder.duration = _duration(1)
    model(
        {
            "conditioning": tf.broadcast_to(
                tf.constant([45.0, _VELOCITY], dtype=tf.float32), (batch_size, 1, _VOICES, 2)
            ),
            "pedal": tf.zeros((batch_size, 1, 4), dtype=tf.float32),
            "piano_model": tf.zeros((batch_size, 1), dtype=tf.int32),
        },
        training=False,
    )


def _prepare_tracks(native: _Native, clips: list[FlightClip], n_mics: int, path: Path) -> None:
    """Use upstream's preprocessed-dataset input, without MIDI conversion or MAESTRO."""
    tf = native.tf
    controls = [_conditioning(clip.rps) for clip in clips]

    def tracks() -> Iterator[dict[str, Any]]:
        for clip, conditioning in zip(clips, controls, strict=True):
            n_frames = len(conditioning)
            pedal = np.zeros((n_frames, 4), dtype=np.float32)
            polyphony = np.full(n_frames, _VOICES, dtype=np.int32)
            for mic in range(n_mics):
                yield {
                    "audio": np.asarray(clip.audio[mic], dtype=np.float32),
                    "conditioning": conditioning,
                    "pedal": pedal,
                    "polyphony": polyphony,
                    "piano_model": np.int32(mic),
                    "audio_filename": f"{clip.key}/mic{mic}".encode(),
                }

    signature = {
        "audio": tf.TensorSpec((None,), tf.float32),
        "conditioning": tf.TensorSpec((None, _VOICES, 2), tf.float32),
        "pedal": tf.TensorSpec((None, 4), tf.float32),
        "polyphony": tf.TensorSpec((None,), tf.int32),
        "piano_model": tf.TensorSpec((), tf.int32),
        "audio_filename": tf.TensorSpec((), tf.string),
    }
    dataset = tf.data.Dataset.from_generator(tracks, output_signature=signature)
    # Upstream uses len(dataset) for its shuffle buffer after Dataset.load.
    dataset.apply(tf.data.experimental.assert_cardinality(len(clips) * n_mics)).save(str(path))


def _training_data(native: _Native, path: Path, batch_size: int, *, shuffle: bool) -> Any:
    tf = native.tf
    dataset = native.data.get_training_dataset(
        str(path),
        batch_size=batch_size,
        max_polyphony=_VOICES,
        shuffle=shuffle,
        duration=_SEGMENT_S,
        overlap=_OVERLAP,
        sample_rate=_SR,
        frame_rate=_FRAME_RATE,
    )

    def onset(batch):
        # Each independent recurrent example starts with four already active
        # rotors. An overlapping crop must not remove the fixed first onset.
        conditioning = batch["conditioning"]
        velocity = tf.concat(
            [
                tf.fill((batch_size, 1, _VOICES), np.float32(_VELOCITY)),
                tf.zeros_like(conditioning[:, 1:, :, 1]),
            ],
            axis=1,
        )
        return {
            "audio": batch["audio"],
            "conditioning": tf.stack([conditioning[..., 0], velocity], axis=-1),
            "pedal": batch["pedal"],
            "piano_model": batch["piano_model"],
        }

    return dataset.map(onset)


def _fit_phase(
    native: _Native,
    tracks: Path,
    weights: Path,
    *,
    phase: int,
    settings: PianoSettings,
    n_mics: int,
    shuffle: bool,
    first_update: int,
) -> dict:
    tf = native.tf
    batch_size = settings.batch_sizes[phase - 1]
    epochs = settings.epochs[phase - 1]
    rate = _LEARNING_RATES[phase - 1]
    model = _model(native, n_mics, inference=False)
    model.alternate_training(first_phase=(phase != 2))
    # Build the native detuner in every checkpoint, including phase 1 (where
    # its initially zero correction is bypassed by the upstream switch).
    model.detuner.use_detune = True
    _build(tf, model, batch_size)
    model.z_encoder.duration = _SEGMENT_S
    state = tf.train.Checkpoint(model=model)
    if phase == 1:
        # Paper §3.3: zero initial detuning. The other instrument-specific
        # correction already starts at zero via the native multiplicative gain.
        detuning = model.z_encoder.detune_embedding.embeddings
        detuning.assign(tf.zeros_like(detuning))
        model.detuner.use_detune = False
    else:
        state.read(str(weights)).assert_consumed()
    # In the final amplitude phase the tuning weights are frozen *at their
    # phase-2 values*, not bypassed as in the legacy alternate_training switch.
    trainer = native.trainers.Trainer(
        model=model, strategy=tf.distribute.get_strategy(), learning_rate=rate
    )
    iterator = iter(_training_data(native, tracks, batch_size, shuffle=shuffle))
    history = []
    for epoch in range(1, epochs + 1):
        steps = []
        for step in range(1, settings.steps_per_epoch + 1):
            losses = {
                key: float(value.numpy()) for key, value in trainer.train_step(iterator).items()
            }
            update = first_update + (epoch - 1) * settings.steps_per_epoch + step
            if not all(map(math.isfinite, losses.values())):
                raise FloatingPointError(
                    f"Nonfinite DDSP-Piano phase {phase} update {update}: {losses}"
                )
            steps.append({"update": update, **losses})
        means = {key: sum(row[key] for row in steps) / len(steps) for key in losses}
        history.append({"phase": phase, "epoch": epoch, "mean_losses": means, "steps": steps})
        print(f"DDSP-Piano phase {phase} epoch {epoch}/{epochs}: {means}", flush=True)
    updates = int(trainer.step.numpy())
    if updates != epochs * settings.steps_per_epoch:
        raise RuntimeError(f"DDSP-Piano phase {phase} made {updates} optimizer updates")
    state.write(str(weights))
    return {
        "phase": phase,
        "kind": _PHASE_NAMES[phase - 1],
        "epochs": epochs,
        "batch_size": batch_size,
        "initial_learning_rate": rate,
        "final_learning_rate": float(trainer.optimizer.learning_rate.numpy()),
        "updates": updates,
        "parameter_count": model.count_params(),
        "trainable_parameter_count": sum(v.shape.num_elements() for v in model.trainable_weights),
        "history": history,
    }


def fit_piano(
    clips: Iterable[FlightClip],
    checkpoint: Path,
    *,
    seed: int = 0,
    device: str = "cuda",
    upstream: Path | None = None,
    settings: PianoSettings | None = None,
) -> dict:
    """Fit one expert on aligned mixed calibration audio and save native TF weights."""
    settings = settings if settings is not None else PianoSettings()
    clips = list(clips)
    if not clips or clips[0].audio.ndim != 2:
        raise ValueError("DDSP-Piano needs at least one multichannel calibration clip")
    n_mics = clips[0].audio.shape[0]
    if not n_mics:
        raise ValueError("DDSP-Piano needs at least one microphone")
    for clip in clips:
        if clip.sr != _SR or clip.audio.shape != (n_mics, clip.rps.shape[-1]):
            raise ValueError(f"{clip.key}: mismatched DDSP-Piano RPS/audio grid or mic count")
        if clip.audio.shape[1] < _SEGMENT_SAMPLES or not np.isfinite(clip.audio).all():
            raise ValueError(f"{clip.key}: expected at least 3 seconds of finite audio")

    native = _native(upstream)
    tf = native.tf
    tf_device = _device(tf, device)
    tf.keras.utils.set_random_seed(seed)
    checkpoint = Path(checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    prefix = checkpoint.with_name(checkpoint.name + ".weights")
    phases = []
    history = []
    total_updates = 0
    shuffle = len(clips) * n_mics > 1
    with TemporaryDirectory(prefix="hns-piano-") as tmp:
        tracks = Path(tmp) / "calibration.tfrecord"
        intermediate = Path(tmp) / "phase-weights"
        _prepare_tracks(native, clips, n_mics, tracks)
        with tf.device(tf_device):
            for phase in (1, 2, 3):
                report = _fit_phase(
                    native,
                    tracks,
                    intermediate,
                    phase=phase,
                    n_mics=n_mics,
                    settings=settings,
                    shuffle=shuffle,
                    first_update=total_updates,
                )
                total_updates += report["updates"]
                history.extend(report.pop("history"))
                phases.append(report)
        members = {}
        for src in sorted(intermediate.parent.glob(intermediate.name + ".*")):
            dst = prefix.with_name(prefix.name + src.name[len(intermediate.name) :])
            # The native checkpoint's index and data shards contain no absolute paths.
            copyfile(src, dst)
            members[dst.name] = _sha256(dst)
    report = {
        "schema": _SCHEMA,
        "method": "ddsp_piano",
        "status": "completed",
        "paper_doi": "10.17743/JAES.2022.0102",
        "upstream_url": UPSTREAM_URL,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_config": _CONFIG,
        "upstream_files": dict(_PINNED_FILES),
        "weight_prefix": prefix.name,
        "artifacts": members,
        "production_settings": settings == PianoSettings(),
        "settings": {
            "epochs": list(settings.epochs),
            "steps_per_epoch": settings.steps_per_epoch,
            "batch_sizes": list(settings.batch_sizes),
        },
        "sample_rate": _SR,
        "n_mics": n_mics,
        "n_rotors": _VOICES,
        "parameter_count": phases[-1]["parameter_count"],
        "seed": seed,
        "architecture": {
            "z_embedding_size": 16,
            "context_network": "Dense32/GRU64/Normalize/Dense32",
            "monophonic_network": "Dense128/GRU192/Dense192/Normalize",
            "n_substrings": 2,
            "n_partials": 96,
            "n_noise_coefficients": 64,
            "oscillator": "native DAFx22 config's cosine bank at the pinned source revision",
            "reverb_ir_samples": 24_000,
        },
        "control": {
            "frame_rate": _FRAME_RATE,
            "pitch": "69 + 12*log2(RPS/440), no rounding; strided samples at 0, 4, ... ms",
            "onset_velocity": _VELOCITY,
            "onset_rule": "first frame only, all four sustained voices",
            "pedal": [0, 0, 0, 0],
            "mic_environment_ids": list(range(n_mics)),
            "voice_order": "original rotor order",
            "inference": "native dynamic length, angular cumsum, learned detuning and decay-masked IR",
            "render_rng": "private TensorFlow Philox stream reset from one supplied NumPy RNG int64",
        },
        "initialization": {
            "keras_global_seed": seed,
            "checkpoint_initialization": "none; random/native published configuration defaults",
            "detuning": "zero detuner kernel, bias, and environment detuning",
            "inharmonicity": "native Rigaud asymptotes and zero environment modifier gain",
            "release_duration_s": 1.1,
            "reverb_duration_s": 1.5,
        },
        "losses": {
            "type": "native DDSP multiresolution spectral L1",
            "fft_sizes": [2048, 1024, 512, 256, 128, 64],
            "magnitude_weight": 1.0,
            "log_magnitude_weight": 1.0,
            "reverb_ir_l1_weight": 0.01,
            "inharmonicity_modifier_l1_weight": 0.1,
        },
        "optimizer": {
            "type": "ddsp.training.trainers.Trainer / TensorFlow Keras Adam",
            "learning_rates": list(_LEARNING_RATES),
            "lr_decay_steps": 10000,
            "lr_decay_rate": 0.98,
            "grad_clip_norm": 3.0,
            "phase_boundary": "fresh optimizer; transfer all model weights",
        },
        "pipeline": {
            "native_input": "get_training_dataset preprocessed Dataset.load path",
            "segment_seconds": _SEGMENT_S,
            "segment_overlap": _OVERLAP,
            "segment_tail": "native full segments only, independently within each support clip",
            "shuffle": shuffle,
            "shuffle_seed": 0,
            "repeat": True,
            "drop_remainder": True,
            "prefetch_batches": 4,
            "deviations": [
                "prealigned support clips split by microphone as native mono tracks, not MAESTRO",
                "fixed onset added at the start of each independent training segment",
                "reset Adam per phase and select final update, no validation/evaluation selection",
                "render filtered noise with a private RNG; native transfer function unchanged",
            ],
        },
        "paper_code_differences": {
            "phase_3_detuning": (
                "The pinned alternate_training(first_phase=True) helper disables use_detune "
                "in amplitude/noise mode. Phase 3 instead keeps use_detune=True with the "
                "phase-2 tuning weights frozen, so the amplitude/noise refinement uses "
                "the learned partial frequencies as described in JAES §3.4."
            ),
        },
        "calibration": [
            {
                "key": clip.key,
                "recording_id": clip.recording_id,
                "regime": clip.regime,
                "fit_regime": clip.provenance.get("fit_regime"),
                "split": clip.provenance.get("split"),
                "dataset_pin": clip.provenance.get("dataset_pin"),
                "label_key": clip.provenance.get("label_key"),
                "start_s": clip.start_s,
                "duration_s": clip.audio.shape[-1] / _SR,
                "segments_per_mic": 1 + (clip.audio.shape[-1] - _SEGMENT_SAMPLES) // _SEGMENT_HOP,
            }
            for clip in clips
        ],
        "selected_phase": 3,
        "final_phase": 3,
        "total_updates": total_updates,
        "phases": phases,
        "history": history,
        "device": {"requested": device, "tensorflow": tf_device},
        "software": {
            "tensorflow": tf.__version__,
            "keras_module": tf.keras.__name__,
            "tensorflow_build": dict(tf.sysconfig.get_build_info()),
            "tensorflow_probability": version("tensorflow-probability"),
            "ddsp": native.ddsp.__version__,
            "gin_config": version("gin-config"),
            "python": platform.python_version(),
            "architecture": platform.machine(),
            "numpy": np.__version__,
        },
    }
    checkpoint.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


class _PianoRenderer:
    def __init__(self, report: dict, native: _Native, device: str, prefix: Path):
        self.report = report
        self._tf = tf = native.tf
        self._device = device
        self._lock = RLock()
        self._n_frames = 0
        self._synthesize: Callable[[Any, Any], Any] | None = None
        with tf.device(device):
            self._model = _model(native, report["n_mics"], inference=True)
            _build(tf, self._model, batch_size=1)
            tf.train.Checkpoint(model=self._model).read(str(prefix)).assert_consumed()
            self._model.trainable = False
            self._noise = tf.random.Generator.from_seed(0, alg="philox")

        noise = self._model.processor_group.noise
        noise_rng = self._noise
        upsampling, window_size = noise.upsampling, noise.window_size
        frequency_filter = native.ddsp.core.frequency_filter

        def seeded_signal(magnitudes):
            # Exact DynamicSizeFilteredNoise.get_signal path, with the random
            # source isolated so retracing and TF's global seed cannot affect it.
            batch, frames, _ = magnitudes.shape
            white = noise_rng.uniform([batch, upsampling * frames], minval=-1.0, maxval=1.0)
            return frequency_filter(white, magnitudes, window_size=window_size)

        noise.get_signal = seeded_signal

    def _graph(self, n_frames: int) -> Callable[[Any, Any], Any]:
        tf = self._tf
        model = self._model
        model.z_encoder.duration = _duration(n_frames)

        @tf.function(
            input_signature=[
                tf.TensorSpec((1, n_frames, _VOICES, 2), tf.float32),
                tf.TensorSpec((1, 1), tf.int32),
            ]
        )
        def synthesize(conditioning, environment):
            return model(
                {
                    "conditioning": conditioning,
                    "pedal": tf.zeros((1, n_frames, 4), dtype=tf.float32),
                    "piano_model": environment,
                },
                training=False,
            )["audio_synth"]

        return synthesize

    def render(self, rps: np.ndarray, *, sr: int, rng: np.random.Generator) -> np.ndarray:
        if sr != _SR:
            raise ValueError(f"DDSP-Piano renders at {_SR} Hz, not {sr} Hz")
        conditioning = _conditioning(rps)
        n_frames = len(conditioning)
        n_samples = rps.shape[-1]
        n_mics = self.report["n_mics"]
        result = np.empty((n_mics, n_samples), dtype=np.float32)
        tf = self._tf
        with self._lock, tf.device(self._device):
            synthesize = self._synthesize
            if synthesize is None or self._n_frames != n_frames:
                synthesize = self._graph(n_frames)
                self._synthesize = synthesize
                self._n_frames = n_frames
            self._noise.reset_from_seed(int(rng.integers(np.iinfo(np.int64).max)))
            control = tf.convert_to_tensor(conditioning[None, ...])
            for mic in range(n_mics):
                signal = synthesize(control, tf.constant([[mic]], dtype=tf.int32))
                if signal.shape != (1, n_frames * _HOP):
                    raise RuntimeError(f"Unexpected native DDSP-Piano audio shape: {signal.shape}")
                # Render the whole last native control hop and keep only its
                # requested samples. No waveform padding or duration fallback.
                result[mic] = signal[0, :n_samples].numpy()
        if not np.isfinite(result).all():
            raise FloatingPointError("DDSP-Piano produced nonfinite audio")
        return result


def load_piano(
    checkpoint: Path, *, device: str = "cpu", upstream: Path | None = None
) -> _PianoRenderer:
    """Load a movable native model bundle without reading any fit or test audio."""
    checkpoint = Path(checkpoint)
    report = json.loads(checkpoint.read_text(encoding="utf-8"))
    if report["schema"] != _SCHEMA or report["upstream_commit"] != UPSTREAM_COMMIT:
        raise ValueError(f"Unsupported DDSP-Piano checkpoint: {checkpoint}")
    prefix = checkpoint.parent / report["weight_prefix"]
    for name, expected in report["artifacts"].items():
        path = checkpoint.parent / name
        if _sha256(path) != expected:
            raise ValueError(f"DDSP-Piano checkpoint weight checksum mismatch: {path}")
    native = _native(upstream)
    return _PianoRenderer(report, native, _device(native.tf, device), prefix)
