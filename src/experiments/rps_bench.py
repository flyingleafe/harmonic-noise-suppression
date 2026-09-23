"""The RPS benchmark parts and a browser over model outputs on them.

Three parts, the same for every model -- the sets ``scripts/rps_dump.py``
dumps and ``docs/experiments/rps-error-profile.md`` reads::

    comb    conf/data/salv2_comb_nomix.yaml  valid  (32 flights x 8 mics, 4 silent)
    stoch   conf/data/salv2_stoch_nomix.yaml valid  (32 flights, 8 silent)
    real    conf/data/m3cur_s2.yaml          valid  (dload:DREGON-LM-V4-michaels-valid-full, 37 clips)

plus the speech twins ``comb_speech`` / ``stoch_speech`` (same flights, a
LibriSpeech talker at -30 to 0 dB). A frame is one microphone of an 8 s clip:
frame ``i`` of a synthetic part is flight ``i // 8``, mic ``i % 8``.

Notebook use (``notebooks/rps_tracking.ipynb`` sections 3 and 6)::

    from experiments import rps_bench as rb
    rb.experiments()                                   # what the dump holds
    rb.worst("real", "hppnet_r4_l4")                   # the frames worth a look
    f = rb.compare(["hppnet_r4_l4", "r4hb_scv2"], "real", 71)
    rb.show(f, fmax=2000)                              # spectrogram + GT + one row per model
    dwym({p: rb.overlay("salv2_hppnet_stoch_nomix", p, 8) for p in ("comb", "stoch", "real")})

A model is either an experiment name (its ``best`` checkpoint) or a
``label -> (experiment, checkpoint)`` mapping, which is what a comparison
across training generations needs — the arms of the unified panel alias a
different file per monitored score::

    MODELS = {"legacy easy": ("rig_easy_scv2_unified", "best_real_overall")}
    rb.frames_table("real")                            # pick a frame by rig/regime
    rb.show(rb.compare(MODELS, "real", 176, source="live"))
    rb.mae_table(MODELS, "real", [216, 16], source="live")   # rows = models, cols = frames

Predictions come from the dump (``results/rps_dump/<part>/<exp>.npz``) when it
holds the experiment, else from the checkpoint through ``zoo.load`` on the
CPU (``source="live"`` forces that), memoised per (experiment, checkpoint,
readout, part, frame) under ``.cache/rps_bench/preds/``. A live salience
model is decoded by its OWN decoder (``decode_logits``, as
``training.validation`` calls it) unless ``readout="peak"`` asks for the
peak + parabola readout the dumps hold — see :data:`READOUTS`. A part is
built once per process and pickled under ``.cache/rps_bench/``: a synthetic
part is a minute of synthesis, the real part an R2 pull.

This module sits in ``experiments`` because it imports across the whole
stack (zoo, plots, metrics, training, data_processing); nothing in ``src``
imports it, scripts and notebooks do.
"""

from __future__ import annotations

import json
import pickle
from collections.abc import Mapping, Sequence
from itertools import permutations
from pathlib import Path
from typing import Any, cast

import matplotlib.figure
import numpy as np
import pandas as pd
import tdseries as td
import torch
import torch.nn.functional as F
import yaml
from omegaconf import OmegaConf

from data_processing.frames import meta_dict, rps_series
from losses.pit import align_rps_to_gt
from metrics import RPSMetric
from metrics._common import get_array
from metrics.salience_layers import LayerPeakRPSMetric, peak_readout
from models.multif0.utils import linear_freq_grid
from plots.timeframe import PlotTrack, plot_timeframe
from plots.timeframe.renderers import make_spectrogram_series
from training.config import build_dataset
from utils.audio import first_channel
from zoo.cache import REPO_ROOT

__all__ = [
    "PARTS",
    "READOUTS",
    "REGIME_THRESHOLDS",
    "ModelSpec",
    "Readout",
    "build_set",
    "compare",
    "experiments",
    "frame_label",
    "frame_regimes",
    "frames_table",
    "mae_table",
    "mean_speed_slew",
    "model_specs",
    "overlay",
    "parse_sets",
    "part",
    "pit_mae",
    "resample_like_metric",
    "show",
    "worst",
]

#: part name -> (data yaml whose ``valid`` block defines it, param overrides).
#: Paths are repo-root anchored so a notebook run from ``notebooks/`` works.
PARTS: dict[str, tuple[str, dict[str, Any]]] = {
    "comb": (str(REPO_ROOT / "conf/data/salv2_comb_nomix.yaml"), {}),
    "stoch": (str(REPO_ROOT / "conf/data/salv2_stoch_nomix.yaml"), {}),
    "real": (str(REPO_ROOT / "conf/data/m3cur_s2.yaml"), {}),
    "real_nospeech": (str(REPO_ROOT / "conf/data/m3cur_s2_nospeech.yaml"), {}),
    "comb_speech": (str(REPO_ROOT / "conf/data/salv2_comb_nomix.yaml"), {"speech": True}),
    "stoch_speech": (str(REPO_ROOT / "conf/data/salv2_stoch_nomix.yaml"), {"speech": True}),
}
RATE = (16000, 512)  # the label / prediction frame grid
DUMP_ROOT = REPO_ROOT / "results/rps_dump"
PROFILE = REPO_ROOT / "results/rps_profile/frames.csv"
CACHE_DIR = REPO_ROOT / ".cache/rps_bench"
PRED_CACHE = CACHE_DIR / "preds"
_PERMS = list(permutations(range(4)))


# ─── Parts ────────────────────────────────────────────────────────────────────


def parse_sets(spec: str) -> list[tuple[str, str, dict[str, Any]]]:
    """``name=path[:key=value...]`` items, comma separated -> (name, path, overrides).

    A bare name that is a key of :data:`PARTS` expands to that part.
    """
    out = []
    for item in filter(None, spec.split(",")):
        if "=" not in item and item.strip() in PARTS:
            path, kv = PARTS[item.strip()]
            out.append((item.strip(), path, dict(kv)))
            continue
        name, _, rest = item.partition("=")
        path, *overrides = rest.split(":")
        kv = {}
        for ov in overrides:
            k, _, v = ov.partition("=")
            kv[k] = yaml.safe_load(v)  # `true` -> True, `8` -> 8, text stays text
        out.append((name.strip(), path.strip(), kv))
    if not out:
        raise SystemExit("--sets is empty")
    return out


def build_set(path: str, overrides: dict[str, Any]) -> Any:
    """The ``valid`` block of a Hydra data yaml, built as training builds it."""
    cfg = OmegaConf.load(path)
    spec: dict[str, Any] = dict(cast(dict, OmegaConf.to_container(cfg.valid, resolve=True)))
    params: dict[str, Any] = dict(spec.get("params") or {})
    params.update(overrides)
    policy = params.get("path")
    if isinstance(policy, str) and not Path(policy).is_absolute():
        params["path"] = str(REPO_ROOT / policy)  # the synthetic parts' online-mix policy
    spec["params"] = params
    return build_dataset(spec)


_parts: dict[str, list[td.Frame]] = {}


def part(name: str, *, n: int | None = None, cache: bool = True) -> list[td.Frame]:
    """The frames of one benchmark part, built once and pickled.

    ``n`` overrides the part's frame count (a smoke); a sized part is never
    cached on disk.
    """
    key = name if n is None else f"{name}[{n}]"
    if key in _parts:
        return _parts[key]
    path, overrides = PARTS[name]
    disk = CACHE_DIR / f"{name}.pkl"
    if n is None and cache and disk.exists():
        frames = pickle.loads(disk.read_bytes())
    else:
        params = dict(overrides)
        if n is not None:
            params["n"] = int(n)
        ds = build_set(path, params)
        frames = [ds[i] for i in range(len(ds) if n is None else min(len(ds), n))]
        if n is None and cache:
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            disk.write_bytes(pickle.dumps(frames))
    _parts[key] = frames
    return frames


# ─── Which frame am I looking at: rig and flight regime ───────────────────────

#: The four-regime split the transfer campaign states (not fits) — the numbers
#: are the ones ``scripts/_regime_decomp.py`` writes verbatim into its JSON and
#: ``docs/experiments/noise-v2-transfer.md`` reports its per-regime tables on.
#: That script owns the decomposition; this module only needs the labels, so
#: the thresholds are restated here rather than importing across ``scripts/``
#: (``src`` may not import scripts, and ``experiments`` is a leaf by contract).
REGIME_THRESHOLDS = {
    "ramp_slew_rev_s2": 20.0,
    "zero_max_rev_s": 1.0,
    "standby_mean_rev_s": 45.0,
    "derivative_halfwidth_s": 0.25,
    "smoothing_width_s": 9 / (RATE[0] / RATE[1]),
    "frame_rate_hz": RATE[0] / RATE[1],
}
_FPS = RATE[0] / RATE[1]  # 31.25 Hz, the grid the rps label lives on
_SMOOTH_W = 9  # 0.288 s boxcar, odd so it is centred
_DERIV_HALF = int(round(REGIME_THRESHOLDS["derivative_halfwidth_s"] * _FPS))


def _smooth(x: np.ndarray, width: int = _SMOOTH_W) -> np.ndarray:
    """Centred boxcar moving average with edge padding, length preserved."""
    if width < 3:
        return x
    pad = width // 2
    return np.convolve(np.pad(x, pad, mode="edge"), np.ones(width) / width, mode="valid")


def mean_speed_slew(target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(mean_speed, |d mean_speed/dt|)``, both ``(F,)``, from ``(R, F)`` speeds.

    The derivative is a central difference over ``+-0.25 s`` of the smoothed
    rotor mean with the indices clamped at the clip edges.
    """
    mean = np.asarray(target, dtype=np.float64).mean(axis=0)
    smoothed = _smooth(mean)
    idx = np.arange(mean.size)
    hi = np.minimum(idx + _DERIV_HALF, mean.size - 1)
    lo = np.maximum(idx - _DERIV_HALF, 0)
    span = (hi - lo) / _FPS
    slew = np.divide(smoothed[hi] - smoothed[lo], span, out=np.zeros_like(mean), where=span > 0)
    return mean, np.abs(slew)


def frame_regimes(target: np.ndarray) -> np.ndarray:
    """``(F,)`` regime label per frame of an ``(R, F)`` label track.

    Slew first, then level — a spin-up through 45 rev/s is a ``ramp``, not half
    a ``standby`` and half a ``cruise``.
    """
    target = np.asarray(target, dtype=np.float64)
    mean, slew = mean_speed_slew(target)
    ramp = slew >= REGIME_THRESHOLDS["ramp_slew_rev_s2"]
    labels = np.full(target.shape[1], "cruise", dtype=object)
    labels[~ramp & (mean < REGIME_THRESHOLDS["standby_mean_rev_s"])] = "standby"
    labels[~ramp & (target.max(axis=0) < REGIME_THRESHOLDS["zero_max_rev_s"])] = "zero"
    labels[ramp] = "ramp"
    return labels


def _clip_recordings(part_name: str) -> dict[str, str]:
    """``sample id -> source recording`` for a part backed by a dload dataset.

    Empty for the synthetic parts, which are rendered rather than recorded.
    """
    cfg = OmegaConf.load(PARTS[part_name][0])
    data_dir = str(cast(Any, cfg).valid.get("params", {}).get("data_dir", ""))
    if not data_dir.startswith("dload:"):
        return {}
    from data_processing.streams import ensure_local

    root = Path(ensure_local(data_dir.removeprefix("dload:")))
    rows = json.loads((root / "metadata.json").read_text())
    if isinstance(rows, dict):
        rows = next(iter(rows.values()))
    return {str(r["id"]): str(r.get("recording_id", "")) for r in rows}


def _rig_of(recording: str) -> str:
    """The airframe a real recording came from — the rule ``clip_rigs`` uses."""
    michaels = "michael" in recording.lower() or recording.upper().startswith("FLY")
    return "michaels" if michaels else "dregon"


_tables: dict[str, pd.DataFrame] = {}


def frames_table(part_name: str, *, n: int | None = None) -> pd.DataFrame:
    """One row per frame of a part: what it is, and what the label does in it.

    Columns: ``frame`` (the index :func:`overlay` and :func:`compare` take),
    ``clip`` (the sample id), ``recording`` and ``dataset`` (the airframe:
    ``dregon`` / ``michaels`` for the real split, ``synthetic`` otherwise),
    ``mic``, ``mean_speed`` (rev/s, averaged over rotors and time),
    ``speed_range`` (peak-to-peak of the rotor mean), ``regime`` (the modal
    four-regime label, :data:`REGIME_THRESHOLDS`) and ``regime_share``.
    """
    if n is None and part_name in _tables:
        return _tables[part_name]
    frames = part(part_name, n=n)
    recordings = _clip_recordings(part_name)
    rows = []
    for i, fr in enumerate(frames):
        meta = meta_dict(fr)
        clip = str(meta.get("recording_id", meta.get("sample_id", i // 8)))
        rec = recordings.get(clip, "")
        gt = np.asarray(get_array(fr, "rps"), dtype=np.float64)
        mean, _ = mean_speed_slew(gt)
        labels = frame_regimes(gt)
        uniq, counts = np.unique(labels.astype(str), return_counts=True)
        top = int(counts.argmax())
        rows.append(
            {
                "frame": i,
                "clip": clip,
                "recording": rec,
                "dataset": _rig_of(rec) if rec else "synthetic",
                "mic": int(meta.get("channel", i % 8)),
                "mean_speed": float(mean.mean()),
                "speed_range": float(mean.max() - mean.min()),
                "regime": str(uniq[top]),
                "regime_share": float(counts[top] / counts.sum()),
            }
        )
    table = pd.DataFrame(rows)
    if n is None:
        _tables[part_name] = table
    return table


def frame_label(part_name: str, i: int) -> str:
    """``<dataset>/<clip> mic m`` — the column heading of :func:`mae_table`."""
    row = frames_table(part_name).iloc[i]
    return f"{row['dataset']}/{row['clip']} mic {row['mic']}"


# ─── Readout ──────────────────────────────────────────────────────────────────


def resample_like_metric(gt: np.ndarray, n_t: int) -> np.ndarray:
    """``(R, Tg)`` -> ``(R, n_t)``, torch's linear ``align_corners=False``."""
    tg = gt.shape[-1]
    if tg == n_t:
        return gt
    pos = np.clip((np.arange(n_t) + 0.5) * tg / n_t - 0.5, 0, tg - 1)
    return np.stack([np.interp(pos, np.arange(tg), row) for row in gt])


def pit_mae(pred: np.ndarray, gt: np.ndarray) -> float:
    """MAE-optimal assignment over all 4! permutations, on the prediction's grid."""
    gt = resample_like_metric(np.asarray(gt, dtype=np.float64), pred.shape[-1])
    cost = np.abs(pred[:, None] - gt[None, :]).mean(-1)
    return min(sum(cost[k, p[k]] for k in range(4)) for p in _PERMS) / 4


class Readout:
    """Turn a model's output Frame into ``(R, T)`` rev/s plus the monitored metric.

    A regressor's ``rps_pred`` is taken as is. A salience port's layers are
    read by the peak + parabola readout its ``rps_mae`` metric uses
    (``metrics.salience_layers.peak_readout``), not by the CRF.
    """

    def __init__(
        self,
        fmin: float = 0.0,
        fmax: float = 150.0,
        bins: int = 300,
        n_layers: int = 4,
        rate: tuple[int, int] = RATE,
    ):
        self.grid = np.asarray(linear_freq_grid(fmin, fmax, bins), dtype=np.float64)
        self.n_layers = n_layers
        self.reg_metric = RPSMetric("mae_frame", rate=rate)
        self.sal_metric = LayerPeakRPSMetric(
            out_fmin=fmin, out_fmax=fmax, out_bins=bins, n_layers=n_layers, rate=rate
        )

    def __call__(self, pred: td.Frame, frame: td.Frame) -> tuple[np.ndarray, float]:
        if "rps_pred" in pred:
            arr = np.asarray(get_array(pred, "rps_pred"), dtype=np.float32)
            return arr, float(self.reg_metric(pred, frame))
        if "salience" in pred:
            metric = float(self.sal_metric(pred, frame))
            logits = torch.as_tensor(get_array(pred, "salience")).unsqueeze(0)  # (1, R*G, T)
            _, fg, n_t = logits.shape
            if fg != self.n_layers * len(self.grid):
                raise ValueError(
                    f"model emits {fg} bins; expected "
                    f"{self.n_layers} layers x {len(self.grid)} bins"
                )
            layers = logits.reshape(1, self.n_layers, len(self.grid), n_t).double()
            speeds = peak_readout(F.logsigmoid(layers), self.grid)[0].numpy()
            return speeds.astype(np.float32), metric
        raise KeyError(f"no rps_pred or salience in {list(pred.keys())}")


# ─── Predictions: dump first, checkpoint second ───────────────────────────────

_dumps: dict[tuple[str, str], Any] = {}
_models: dict[tuple[str, str], Any] = {}
_readout: Readout | None = None


def experiments(part_name: str | None = None, *, dump_root: Path = DUMP_ROOT) -> list[str]:
    """The experiments the dump holds (for one part, or for every part)."""
    parts = [part_name] if part_name else [p.name for p in dump_root.iterdir() if p.is_dir()]
    names: set[str] | None = None
    for p in parts:
        here = {f.stem for f in (dump_root / p).glob("*.npz") if not f.stem.startswith("_")}
        names = here if names is None else names & here
    return sorted(names or [])


def _dump_pred(exp: str, part_name: str, i: int, dump_root: Path) -> np.ndarray | None:
    key = (part_name, exp)
    if key not in _dumps:
        f = dump_root / part_name / f"{exp}.npz"
        _dumps[key] = np.load(f) if f.exists() else None
    z = _dumps[key]
    if z is None:
        return None
    return z["pred"][i, :, : z["n_t"][i]].astype(np.float64)


#: How a live salience model's output becomes rev/s. ``"deployed"`` is the
#: model's OWN decoder — ``decode_logits``, shared-map tracking or per-layer
#: CRF — called exactly as ``training.validation._salience_readout`` calls it,
#: so a row is the number the campaign's checkpoint selection and
#: ``scripts/_regime_decomp.py`` were measured on. ``"peak"`` is
#: :class:`Readout`'s peak + parabola, which is what the ``rps_mae`` monitor
#: metric and the dumps under ``results/rps_dump/`` hold. A regressor emits
#: ``rps_pred`` and reads the same either way.
READOUTS = ("deployed", "peak")


def _deployed_speeds(model: Any, pred: td.Frame, frame: td.Frame) -> np.ndarray:
    """``(R, T)`` rev/s through the model's deployed decoder."""
    if "rps_pred" in pred:
        return np.asarray(get_array(pred, "rps_pred"), dtype=np.float64)
    if "salience" not in pred:
        raise KeyError(f"no rps_pred or salience in {list(pred.keys())}")
    decode = getattr(model.model, "decode_logits", None)
    if decode is None:
        raise AttributeError(
            f"{type(model.model).__name__} emits salience but has no decode_logits; "
            'use readout="peak"'
        )
    logits = torch.as_tensor(get_array(pred, "salience")).unsqueeze(0)  # (1, R*G, T)
    n_samples = int(np.asarray(get_array(frame, "mixture")).shape[-1])
    with torch.no_grad():
        speeds = decode(logits.to(model.device), n_samples)  # (1, R, T_stft)
    return np.asarray(speeds[0].detach().cpu(), dtype=np.float64)


def _live_pred(
    exp: str, frame: td.Frame, device: str, ckpt: str = "best", readout: str = "deployed"
) -> np.ndarray:
    """One model's raw ``(R, T)`` speeds for one frame, from its checkpoint."""
    global _readout
    import zoo  # heavy (Hydra + torch); only when a checkpoint is really needed

    if readout not in READOUTS:
        raise ValueError(f"readout must be one of {READOUTS}, got {readout!r}")
    key = (exp, ckpt)
    if key not in _models:
        _models[key] = zoo.load(exp, ckpt=ckpt, device=device)
    model = _models[key]
    pred = model(frame)
    if readout == "deployed":
        return _deployed_speeds(model, pred, frame)
    if _readout is None:
        _readout = Readout()
    return _readout(pred, frame)[0].astype(np.float64)


def _cached_live_pred(
    exp: str, part_name: str, i: int, frame: td.Frame, device: str, ckpt: str, readout: str
) -> np.ndarray:
    """:func:`_live_pred`, memoised per (experiment, checkpoint, readout, part, frame).

    A CPU forward pass over 8 s is seconds, the CRF decode behind
    ``readout="deployed"`` is tens of seconds; a notebook that changes one
    frame should not pay for the models it already ran.
    """
    disk = PRED_CACHE / part_name / f"{exp}__{ckpt}__{readout}__{i:05d}.npy"
    if disk.is_file():
        return np.load(disk).astype(np.float64)
    pred = _live_pred(exp, frame, device, ckpt, readout)
    disk.parent.mkdir(parents=True, exist_ok=True)
    np.save(disk, pred.astype(np.float32))
    return pred


def overlay(
    exp: str,
    part_name: str,
    i: int,
    *,
    source: str = "auto",
    device: str = "cpu",
    ckpt: str = "best",
    readout: str = "deployed",
    cache: bool = True,
    dump_root: Path = DUMP_ROOT,
) -> td.Frame:
    """One model on one frame: ``audio`` + ``rps`` (label) + ``rps_pred``, ready for ``dwym``.

    ``rps_pred`` is PIT-aligned to the label and ``meta`` carries the
    experiment, the checkpoint, the readout, the part, the frame index, the
    flight and mic, and the PIT MAE. ``source`` is ``"auto"`` (the dump when it
    holds ``exp``, else the checkpoint), ``"dump"`` or ``"live"``; ``ckpt``
    selects which checkpoint of ``exp`` runs live and ``readout`` how its
    output becomes rev/s (:data:`READOUTS`). The dump holds one peak-decoded
    array per experiment and ignores both.
    """
    frame = part(part_name)[i]
    pred = None
    if source in ("auto", "dump"):
        pred = _dump_pred(exp, part_name, i, dump_root)
        if pred is None and source == "dump":
            raise FileNotFoundError(f"{dump_root / part_name / exp}.npz")
    if pred is None:
        pred = (
            _cached_live_pred(exp, part_name, i, frame, device, ckpt, readout)
            if cache
            else _live_pred(exp, frame, device, ckpt, readout)
        )
    gt = np.asarray(get_array(frame, "rps"), dtype=np.float64)
    pred = align_rps_to_gt(pred, gt)
    meta = meta_dict(frame)
    meta.update(
        experiment=exp,
        ckpt=ckpt,
        readout=readout,
        part=part_name,
        index=i,
        flight=str(meta.get("recording_id", meta.get("sample_id", i // 8))),
        mae=pit_mae(pred, gt),
    )
    return td.Frame(
        {
            "audio": first_channel(frame["mixture"]),
            "rps": frame["rps"],
            "rps_pred": rps_series(
                pred.astype(np.float32), sample_rate=RATE[0], hop_length=RATE[1]
            ),
            "meta": td.Frame(meta),
        }
    )


#: What a caller may pass as "the models": a list of experiment names (each on
#: its ``best`` checkpoint), or ``label -> experiment`` / ``label -> (experiment,
#: checkpoint)``. The label is what the plot rows and table rows are called.
ModelSpec = Sequence[str] | Mapping[str, str | tuple[str, str]]


def model_specs(models: ModelSpec, ckpt: str = "best") -> list[tuple[str, str, str]]:
    """Normalise :data:`ModelSpec` to ``[(label, experiment, checkpoint)]``.

    ``ckpt`` is the fallback for entries that do not name one.
    """
    if isinstance(models, Mapping):
        out = []
        for label, spec in models.items():
            exp, exp_ckpt = spec if isinstance(spec, tuple) else (spec, ckpt)
            out.append((str(label), str(exp), str(exp_ckpt)))
        return out
    return [(str(exp), str(exp), ckpt) for exp in models]


def compare(models: ModelSpec, part_name: str, i: int, **kw: Any) -> td.Frame:
    """Several models on ONE frame: ``audio`` + ``rps`` + one aligned entry per model.

    Draw it with :func:`show`; ``meta.mae`` maps each label to its PIT MAE,
    ``meta.ckpt`` each label to the checkpoint it was read from, and
    ``meta.readout`` names the decoder (:data:`READOUTS`). ``kw`` goes to
    :func:`overlay`, so ``source="live"`` and ``readout="peak"`` land there.
    """
    specs = model_specs(models, str(kw.pop("ckpt", "best")))
    entries: dict[str, Any] = {}
    maes: dict[str, float] = {}
    ckpts: dict[str, str] = {}
    base_meta: dict[str, Any] = {}
    for label, exp, exp_ckpt in specs:
        one = overlay(exp, part_name, i, ckpt=exp_ckpt, **kw)
        entries.setdefault("audio", one["audio"])
        entries.setdefault("rps", one["rps"])
        entries[label] = one["rps_pred"]
        maes[label] = float(one["meta"]["mae"])
        ckpts[label] = exp_ckpt
        base_meta = meta_dict(one)
    entries["meta"] = td.Frame(
        {
            **base_meta,
            "experiment": [exp for _, exp, _ in specs],
            "ckpt": ckpts,
            "mae": maes,
        }
    )
    return td.Frame(entries)


def mae_table(models: ModelSpec, part_name: str, frames: Sequence[int], **kw: Any) -> pd.DataFrame:
    """PIT MAE (rev/s) of every model on every frame: rows = models, cols = frames.

    Columns are ``<dataset>/<clip> mic m`` (:func:`frame_label`) plus a
    ``mean`` over the frames given. ``kw`` goes to :func:`overlay`, so
    ``source="live"`` forces the checkpoints and ``readout=`` picks the
    decoder (:data:`READOUTS`; ``"deployed"`` by default).
    """
    specs = model_specs(models, str(kw.pop("ckpt", "best")))
    cols = {i: frame_label(part_name, i) for i in frames}
    rows: dict[str, dict[str, float]] = {}
    for label, exp, exp_ckpt in specs:
        rows[label] = {
            cols[i]: float(overlay(exp, part_name, i, ckpt=exp_ckpt, **kw)["meta"]["mae"])
            for i in frames
        }
    table = pd.DataFrame(rows).T.reindex(columns=list(cols.values()))
    table["mean"] = table.mean(axis=1)
    return table


def show(
    frame: td.Frame, *, fmax: float | None = 2000.0, row_height: float = 2.2
) -> matplotlib.figure.Figure:
    """Spectrogram, label, then one PIT-aligned row per model, on one time axis."""
    preds = [
        k
        for k, v in frame.items()
        if k not in ("audio", "rps", "meta") and isinstance(v, td.Series)
    ]
    maes = dict(frame["meta"]["mae"]) if "meta" in frame and "mae" in frame["meta"] else {}
    spec = make_spectrogram_series(first_channel(frame["audio"]), fmax=fmax)
    tracks: list[Any] = [
        PlotTrack(
            series=spec.series, renderer=spec.renderer, hints={**spec.hints, "title": "spectrogram"}
        ),
        PlotTrack(series=frame["rps"], hints={"title": "rps (label)"}),
    ]
    for k in preds:
        title = f"{k}   PIT MAE {maes[k]:.2f} rev/s" if k in maes else k
        tracks.append(PlotTrack(series=frame[k], hints={"title": title}))
    meta = meta_dict(frame)
    fig = plot_timeframe(frame, tracks=tracks, figsize=(16, row_height * len(tracks) + 1.5))
    fig.suptitle(f"{meta['part']}  flight {meta['flight']}  mic {meta.get('channel', '?')}", y=1.02)
    return fig


def worst(part_name: str, exp: str, k: int = 8, *, profile: Path = PROFILE) -> pd.DataFrame:
    """The ``k`` worst frames of ``exp`` on ``part_name``, from the error profile.

    Needs ``scripts/rps_error_profile.py`` to have run over the dump. Silence
    clips are excluded (a stopped-rotor clip is a trivial 0 for most models).
    """
    fr = pd.read_csv(profile, low_memory=False)
    sel = cast(
        pd.DataFrame, fr[(fr["set"] == part_name) & (fr["exp"] == exp) & (fr["n_stopped"] < 4)]
    )
    cols = ["frame", "flight", "channel", "mae", "cls", "gt_mean", "gt_range", "n_stopped"]
    top = sel.sort_values(by="mae", ascending=False).head(k)
    return cast(pd.DataFrame, top[cols]).reset_index(drop=True)
