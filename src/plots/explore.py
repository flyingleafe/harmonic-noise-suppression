"""Notebook data-exploration primitives: list, tabulate, thumbnail, pick.

Four calls cover the usual "what is in this dataset?" loop:

- :func:`datasets` — the dload catalog (``dload.lock`` pins) as a DataFrame.
- :func:`meta_table` — sample metadata rendered as a DataFrame.
- :func:`grid` — a sampled grid of spectrogram thumbnails with captions.
- :func:`pick` — one sample, coerced, ready for ``plots.dwym`` /
  ``zoo.FrameModel``.

Every function accepts the same dataset forms:

- a dload dataset **name** (``"DREGON-frames"`` — streamed through
  ``data_processing.streams.DloadFrameDataset``, so both ``tdframe-v1`` and
  DREGON-LM sample-dir layouts decode; needs R2 credentials in ``.env``);
- any **map-style dataset** (``__len__`` + ``__getitem__`` of ``td.Frame``);
- any **iterable of frames** (list, generator, ``DloadFrameDataset``).

Drone-noise recordings (``notebooks/noise_explorer.ipynb``) have three
more calls, over published ``tdframe-v1`` sets only:

- :func:`noise_datasets` — the drone-noise sets we hold (:data:`NOISE_DATASETS`
  frames + :data:`RAW_NOISE_DATASETS` raw trees), pins and sizes.
- :func:`noise_recordings` — one row per recording (id, shard, channels, sr,
  duration, dotted meta), built once per shard and cached under
  ``.cache/noise_explorer``.
- :func:`load_recording` — a recording by id; streams only its shard.

Heavy lifting stays where it already lives: streaming/decoding in
``data_processing.streams``, figure assembly in the ``plots.timeframe``
renderers, and entry-name coercion in ``data_processing.canonical``. Only the thumbnail
grid layout is new here.
"""

from __future__ import annotations

import itertools
import tomllib
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tdseries as td

from data_processing.canonical import CANONICAL_ENTRIES, _audio_candidates, coerce_frame
from plots.dwym import DwymResult, _in_ipython
from utils.audio import first_channel

__all__ = [
    "NOISE_DATASETS",
    "RAW_NOISE_DATASETS",
    "datasets",
    "grid",
    "load_recording",
    "meta_table",
    "noise_datasets",
    "noise_recordings",
    "pick",
]

#: Entry names probed (in order) for the thumbnail waveform of :func:`grid`.
_GRID_AUDIO_ENTRIES = ("audio", "mixture", "target", "enhanced", "generated")

#: Meta keys probed (in order) for thumbnail captions.
_CAPTION_KEYS = ("recording_id", "id", "split", "drone", "category", "snr_db", "snr")

_CAPTION_MAX_CHARS = 48
_CELL_MAX_CHARS = 80


# ---------------------------------------------------------------------------
# dataset catalog


def datasets(*, sizes: bool = False) -> pd.DataFrame:
    """The dload dataset catalog as a DataFrame (one row per ``dload.lock`` pin).

    Columns: ``name``, ``version`` (12-char prefix). With ``sizes=True`` each
    pinned manifest is fetched (network + credentials) and ``samples`` /
    ``size`` columns are added; a dataset whose manifest cannot be fetched
    gets null values instead of failing the whole table.
    """
    pins = _lock_pins()
    rows: list[dict[str, Any]] = [{"name": n, "version": v[:12]} for n, v in sorted(pins.items())]
    if sizes:
        import dload

        from data_processing.streams import open_repository

        repo = open_repository()
        for row in rows:
            try:
                manifest = repo.manifest(row["name"], pins[row["name"]])
                row["samples"] = int(manifest.num_samples)
                row["size"] = dload.format_size(int(manifest.total_bytes))
            except Exception:
                row["samples"] = None
                row["size"] = None
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# frame iteration over the accepted dataset forms


def _iter_frames(obj: Any, *, shuffle_seed: int | None = None) -> Iterator[td.Frame]:
    """Yield ``td.Frame``s from any accepted dataset form (module docstring).

    ``shuffle_seed`` only applies to dataset *names* (a seeded streaming
    shuffle, for spread without a full download); other forms keep their own
    order.
    """
    if isinstance(obj, str):
        from data_processing.streams import DloadFrameDataset

        shuffle: bool | int = False if shuffle_seed is None else int(shuffle_seed)
        yield from DloadFrameDataset(obj, shuffle=shuffle)
        return
    if isinstance(obj, td.Frame):
        yield obj
        return
    if isinstance(obj, Mapping):
        raise TypeError("explore: got a Mapping — pass a dataset name, dataset, or frame iterable")
    if hasattr(obj, "__getitem__") and hasattr(obj, "__len__"):
        for i in range(len(obj)):
            yield obj[i]
        return
    if isinstance(obj, Iterable):
        yield from obj
        return
    raise TypeError(
        f"explore: cannot iterate {type(obj).__name__} — pass a dload dataset name, "
        "a map-style dataset, or an iterable of td.Frame"
    )


def _split_remaps(hints: dict[str, Any]) -> dict[str, str]:
    """Pop the canonical-entry string remaps (``rps="motor_speed"``) out of hints."""
    return {
        k: hints.pop(k) for k in list(hints) if k in CANONICAL_ENTRIES and isinstance(hints[k], str)
    }


def _coerce(frame: td.Frame, remaps: Mapping[str, str]) -> td.Frame:
    present = {k: v for k, v in remaps.items() if v in dict(frame.items())}
    return coerce_frame(frame, **present)


# ---------------------------------------------------------------------------
# meta_table


def _cell(value: Any) -> Any:
    """One DataFrame cell: scalars pass through, anything else gets a short repr."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    text = repr(value)
    return text if len(text) <= _CELL_MAX_CHARS else text[: _CELL_MAX_CHARS - 1] + "…"


def _temporal_names(frame: td.Frame) -> list[str]:
    return [k for k, v in frame.items() if isinstance(v, td.Series) and v.has_time]


def _meta_row(frame: td.Frame, fields: Sequence[str] | None) -> dict[str, Any]:
    from data_processing.frames import meta_dict

    meta = meta_dict(frame)
    if fields is not None:
        row = {k: _cell(meta.get(k)) for k in fields}
    else:
        row = {k: _cell(v) for k, v in meta.items()}
    row["entries"] = ", ".join(k for k, _ in frame.items() if k != "meta")
    if _temporal_names(frame):
        row["duration_s"] = round(float(frame.duration), 3)
    return row


def meta_table(
    frames_or_dataset: Any,
    fields: Sequence[str] | None = None,
    limit: int = 32,
) -> pd.DataFrame:
    """Sample metadata as a DataFrame — one row per sample, up to ``limit``.

    Columns are the (flattened, scalar-repr'd) ``meta`` keys, restricted to
    ``fields`` when given, plus two computed columns: ``entries`` (the frame's
    non-meta entry names) and ``duration_s`` (for frames with temporal
    entries).
    """
    rows = [
        _meta_row(frame, fields)
        for frame in itertools.islice(_iter_frames(frames_or_dataset), int(limit))
    ]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# grid


def _grid_audio_series(frame: td.Frame) -> td.Series | None:
    """The entry to thumbnail: a canonical audio entry, else a sole waveform."""
    for name in _GRID_AUDIO_ENTRIES:
        if name in frame and isinstance(frame[name], td.Series):
            return frame[name]
    candidates = _audio_candidates(dict(frame.items()))
    if len(candidates) == 1:
        series = frame[candidates[0]]
        assert isinstance(series, td.Series)
        return series
    return None


def _caption(frame: td.Frame) -> str:
    from data_processing.frames import meta_dict

    meta = meta_dict(frame)
    parts = [f"{meta[k]}" for k in _CAPTION_KEYS if meta.get(k) is not None]
    text = " · ".join(parts) or ", ".join(k for k, _ in frame.items() if k != "meta")
    return text if len(text) <= _CAPTION_MAX_CHARS else text[: _CAPTION_MAX_CHARS - 1] + "…"


def _reservoir(stream: Iterator[td.Frame], n: int, rng: np.random.Generator) -> list[td.Frame]:
    kept: list[td.Frame] = []
    for i, frame in enumerate(stream):
        if len(kept) < n:
            kept.append(frame)
        else:
            j = int(rng.integers(0, i + 1))
            if j < n:
                kept[j] = frame
    return kept


def grid(
    dataset_or_frames: Any,
    n: int = 12,
    *,
    seed: int | None = None,
    scan: int | None = None,
    **dwym_hints: Any,
) -> DwymResult:
    """A sampled grid of compact spectrogram thumbnails with meta captions.

    ``n`` samples are reservoir-sampled from the first ``scan`` frames of the
    stream (default ``max(4 * n, 48)``; a dataset *name* additionally gets a
    seeded streaming shuffle so the scan spans shards). Hints: ``fmax``
    limits the thumbnail bandwidth; canonical-entry string remaps
    (``audio="waveform"``) pass to :func:`data_processing.canonical.coerce_frame`.

    Returns a :class:`plots.dwym.DwymResult` (``route="grid"``) — it displays
    itself in IPython and offers ``.figure`` / ``.save()`` outside.
    """
    remaps = _split_remaps(dwym_hints)
    fmax = dwym_hints.pop("fmax", None)
    if dwym_hints:
        raise TypeError(
            f"grid() got unsupported hints {sorted(dwym_hints)}; "
            "supported: fmax=<Hz> and canonical entry remaps (audio=..., rps=...)"
        )
    if n <= 0:
        raise ValueError(f"grid() needs n >= 1, got {n}")

    from plots.timeframe.renderers import make_spectrogram_series

    cap = int(scan) if scan is not None else max(4 * n, 48)
    rng = np.random.default_rng(seed)
    stream = itertools.islice(_iter_frames(dataset_or_frames, shuffle_seed=seed), cap)
    frames = [_coerce(f, remaps) for f in _reservoir(stream, int(n), rng)]
    if not frames:
        raise ValueError("grid() got no frames")

    ncols = min(4, len(frames))
    nrows = -(-len(frames) // ncols)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.2 * ncols, 2.2 * nrows), squeeze=False, constrained_layout=True
    )
    for ax, frame in zip(axes.flat, frames):
        series = _grid_audio_series(frame)
        if series is None:
            ax.text(0.5, 0.5, "no audio entry", ha="center", va="center", fontsize=8)
            ax.set_axis_off()
            ax.set_title(_caption(frame), fontsize=8)
            continue
        track = make_spectrogram_series(first_channel(series), fmax=fmax)
        spec = np.asarray(track.series.data)
        extent = (
            float(track.series.t_start),
            float(track.series.t_end),
            0.0,
            float(track.hints["freq_max_hz"]),
        )
        ax.imshow(spec, origin="lower", aspect="auto", extent=extent, cmap="magma")
        ax.set_title(_caption(frame), fontsize=8)
        ax.tick_params(labelsize=7)
    for ax in list(axes.flat)[len(frames) :]:
        ax.set_axis_off()

    result = DwymResult(figures=[fig], audio={}, route="grid")
    if _in_ipython():
        plt.close(fig)
    return result


# ---------------------------------------------------------------------------
# pick


def _matches(frame: td.Frame, query: str) -> bool:
    from data_processing.frames import get_meta

    for key in ("recording_id", "id"):
        value = get_meta(frame, key)
        if value is not None and query in str(value):
            return True
    return False


def pick(
    dataset: Any,
    index_or_query: int | str | Callable[[td.Frame], bool] = 0,
    **dwym_hints: str,
) -> td.Frame:
    """One sample, coerced (:func:`data_processing.canonical.coerce_frame`) and ready for
    ``plots.dwym`` / ``zoo.FrameModel``.

    ``index_or_query`` selects the sample:

    - ``int`` — the n-th sample (direct ``dataset[i]`` on map-style datasets,
      n-th of the stream otherwise; negative indices need a map-style
      dataset);
    - ``str`` — the first sample whose ``meta.recording_id`` / ``meta.id``
      contains the string (a stream scans — and downloads — until it hits);
    - callable — the first sample where ``fn(frame)`` is true.

    ``dwym_hints`` are canonical-entry remaps (``rps="motor_speed"``) applied
    silently at coercion.
    """
    remaps = _split_remaps(dict(dwym_hints))
    if set(dwym_hints) - set(remaps):
        raise TypeError(
            f"pick() got unsupported hints {sorted(set(dwym_hints) - set(remaps))}; "
            "supported: canonical entry remaps (audio=..., rps=...)"
        )

    if isinstance(index_or_query, (int, np.integer)):
        i = int(index_or_query)
        if hasattr(dataset, "__getitem__") and hasattr(dataset, "__len__"):
            return _coerce(dataset[i], remaps)
        if i < 0:
            raise IndexError("negative indices need a map-style (len + getitem) dataset")
        seen = 0
        for frame in _iter_frames(dataset):
            if seen == i:
                return _coerce(frame, remaps)
            seen += 1
        raise IndexError(f"index {i} is beyond the stream (exhausted after {seen} samples)")

    predicate: Callable[[td.Frame], bool]
    if isinstance(index_or_query, str):
        query = str(index_or_query)

        def _by_id(frame: td.Frame) -> bool:
            return _matches(frame, query)

        predicate = _by_id
    elif callable(index_or_query):
        predicate = index_or_query
    else:
        raise TypeError(
            f"index_or_query must be an int, str, or callable, got {type(index_or_query).__name__}"
        )

    scanned = 0
    for frame in _iter_frames(dataset):
        scanned += 1
        if predicate(frame):
            return _coerce(frame, remaps)
    raise ValueError(f"pick(): no sample matched {index_or_query!r} ({scanned} samples scanned)")


# ---------------------------------------------------------------------------
# drone-noise catalog, per-recording index, load by id

#: Drone / rotor ego-noise datasets published as recording Frames
#: (``tdframe-v1``): dload name -> what it holds. Display order.
NOISE_DATASETS: dict[str, str] = {
    "DREGON-frames": "DREGON (INRIA) quadrotor: 8-mic array, 44.1 kHz, motor telemetry "
    "(motors_command + motors_measured), rps_refined",
    "michaels-frames": "Michael's DJI M100 flights FLY124/FLY125: 8 mics, calibrated rps",
    "michaels-test-frames": "HELD-OUT TEST flights FLY103/FLY108 (mono, 48 kHz): look, do not train",
    "SPCUP19-frames": "IEEE SP Cup 2019 ego-noise: 10 team packages (1/3/8/16 ch), every recording "
    "tagged from the team reports (condition, active rotors, sources) with mic_pos/rotor_pos",
    "AVQ-egonoise": "AVQ quadrotor: the 5 pure rotor ego-noise sequences, channel 0, 16 kHz mono "
    "(meta from AVQ recipe 1, untagged)",
    "AVQ": "AVQ quadrotor on a tripod: 12 sequences, 8-ch; 5 ego-noise only, 3 speech only "
    "(motors muted), 4 mixtures; tagged from the publisher's spec table",
    "noise-v2-bench-points": "stationary bench/static windows (DREGON, SPCUP19, AVQ, "
    "DroneAudioSet, ChuMS) with measured shaft rates",
    "DroneAudioSet": "2 quads x 2 throttles x 3 rooms, 8-ch; drone-only / source-only / mixed "
    "(88 GiB: index a few shards)",
    "drone-detection-samples": "180k mono 16 kHz clips, drone / no-drone (13 GiB: index a few "
    "shards)",
}

#: Drone-noise dload pins that are raw file trees, not Frames: listed so the
#: catalog is complete; :func:`load_recording` cannot read them.
RAW_NOISE_DATASETS: dict[str, str] = {
    "drone_audio": "raw drone clip corpus (23k files, no frames builder)",
    "zenodo_drone_noises": "raw Zenodo drone-noise recordings (no frames builder)",
    "new-drone-noises": "raw tree of Michael's TEST flights (frames: michaels-test-frames)",
}

_ID_COLUMNS = ("recording_id", "key", "shard", "channels", "sr", "duration_s", "entries")
_META_ARRAY_MAX = 16  # longer array-valued meta is shown as its shape
#: Bump when :func:`_recording_row` changes: cached rows of an older schema are ignored.
_INDEX_SCHEMA = 2


def _lock_pins() -> dict[str, str]:
    from data_processing.streams import REPO_ROOT

    lock = tomllib.loads((REPO_ROOT / "dload.lock").read_text())
    return {str(k): str(v) for k, v in dict(lock.get("datasets", {})).items()}


def noise_datasets(*, sizes: bool = True) -> pd.DataFrame:
    """The drone-noise datasets we hold, one row each.

    Columns: ``name``, ``kind`` (``frames`` = loadable with
    :func:`load_recording`; ``raw`` = a file tree), ``pinned`` (12-char
    ``dload.lock`` version, ``None`` when not pinned yet), ``what``; with
    ``sizes=True`` (network) also ``recordings``, ``shards``, ``size``.
    """
    pins = _lock_pins()
    rows: list[dict[str, Any]] = [
        {"name": n, "kind": kind, "pinned": pins[n][:12] if n in pins else None, "what": what}
        for kind, table in (("frames", NOISE_DATASETS), ("raw", RAW_NOISE_DATASETS))
        for n, what in table.items()
    ]
    if sizes:
        import dload

        from data_processing.streams import open_repository

        repo = open_repository()
        for row in rows:
            row.update(recordings=None, shards=None, size=None)
            if row["name"] not in pins:
                continue
            try:
                m = repo.manifest(row["name"], pins[row["name"]])
            except Exception:
                continue
            row.update(
                recordings=int(m.num_samples),
                shards=len(m.shards),
                size=dload.format_size(int(m.total_bytes)),
            )
    columns = ["name", "kind", "pinned", "recordings", "shards", "size", "what"]
    keep = [c for c in columns if sizes or c not in ("recordings", "shards", "size")]
    df = pd.DataFrame(rows).reindex(columns=keep)
    for c in ("recordings", "shards"):
        if c in df.columns:
            df[c] = df[c].astype("Int64")
    return df


def _flat_meta(value: Any, prefix: str, out: dict[str, Any]) -> None:
    """Nested meta (Frames / dicts) -> dotted scalar columns, JSON-safe."""
    if isinstance(value, (td.Frame, Mapping)):
        items = value.items() if isinstance(value, Mapping) else ((k, value[k]) for k in value)
        for k, v in items:
            _flat_meta(v, f"{prefix}.{k}" if prefix else str(k), out)
        return
    if isinstance(value, td.Series):
        value = np.asarray(value.data)
    if isinstance(value, (np.generic,)):
        value = value.item()
    if isinstance(value, np.ndarray):
        value = value.tolist() if value.size <= _META_ARRAY_MAX else f"<array {value.shape}>"
    if isinstance(value, (list, tuple)):
        value = str(list(value)) if len(value) <= _META_ARRAY_MAX else f"<list of {len(value)}>"
    if value is not None and not isinstance(value, (str, int, float, bool)):
        value = str(value)
    out[prefix] = value


def _recording_row(key: str, frame: td.Frame, shard: int) -> dict[str, Any]:
    from data_processing.frames import meta_dict
    from plots.spectrum_viewer import _audio_entry

    meta = meta_dict(frame)
    row: dict[str, Any] = {
        "recording_id": str(meta.get("recording_id", key)),
        "key": key,
        "shard": shard,
    }
    try:
        _, audio = _audio_entry(frame, None)
    except ValueError:
        audio = None
    if audio is not None and isinstance(audio.tindex, td.GridIndex):
        n_t = int(audio.dim_size("time"))
        row.update(
            channels=int(np.prod(np.asarray(audio.data).shape) // max(n_t, 1)),
            sr=float(audio.tindex.sr),
            duration_s=round(n_t / float(audio.tindex.sr), 3),
        )
    row["entries"] = ", ".join(k for k, _ in frame.items() if k != "meta")
    flat: dict[str, Any] = {}
    _flat_meta({k: v for k, v in meta.items() if k != "recording_id"}, "", flat)
    row.update(flat)
    return row


def _index_path(name: str, version: str, shard: int, digest: str):
    from data_processing.streams import REPO_ROOT

    return (
        REPO_ROOT
        / ".cache"
        / "noise_explorer"
        / name
        / version[:12]
        / f"schema{_INDEX_SCHEMA}"
        / f"{shard:05d}_{digest[:16]}.json"
    )


def _index_shard(
    name: str,
    version: str,
    shard: int,
    digest: str,
    *,
    want: str | None = None,
    refresh: bool = False,
) -> tuple[list[dict[str, Any]], td.Frame | None]:
    """The shard's recording rows (cached on disk) and, when ``want`` is one of
    its recordings and the shard had to be streamed anyway, that Frame."""
    import json

    from data_processing.streams import iter_published_shard

    path = _index_path(name, version, shard, digest)
    if path.exists() and not refresh:
        return json.loads(path.read_text()), None
    rows: list[dict[str, Any]] = []
    found: td.Frame | None = None
    for key, frame in iter_published_shard(name, shard, version):
        row = _recording_row(key, frame, shard)
        rows.append(row)
        if want is not None and want in (key, row["recording_id"]):
            found = frame
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rows))
    tmp.replace(path)
    return rows, found


def _pinned_manifest(name: str, version: str | None):
    from data_processing.streams import open_repository

    if version is None:
        pins = _lock_pins()
        if name not in pins:
            raise KeyError(
                f"{name!r} is not pinned in dload.lock (pass version= to read it anyway)"
            )
        version = pins[name]
    return open_repository().manifest(name, version)


def noise_recordings(
    name: str,
    *,
    max_shards: int | None = None,
    version: str | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """One row per recording of a frames dataset: id, shard, channels, sample
    rate, duration, entry names and every ``meta`` field (nested keys dotted).

    dload has no per-key index, so each shard is streamed ONCE and its rows
    cached under ``.cache/noise_explorer/<name>/<version>/`` (keyed by shard
    digest; ``refresh=True`` rebuilds). ``max_shards`` bounds the scan for the
    very large sets (DroneAudioSet, drone-detection-samples); the table's
    ``attrs["shards"]`` says how many of how many were indexed.
    """
    manifest = _pinned_manifest(name, version)
    shards = manifest.shards if max_shards is None else manifest.shards[: int(max_shards)]
    rows: list[dict[str, Any]] = []
    for i, info in enumerate(shards):
        if not _index_path(name, manifest.version, i, info.digest).exists() or refresh:
            print(f"indexing {name}: shard {i + 1}/{len(shards)}", end="\r", flush=True)
        rows.extend(_index_shard(name, manifest.version, i, info.digest, refresh=refresh)[0])
    df = pd.DataFrame(rows)
    if not df.empty:
        first = [c for c in _ID_COLUMNS if c in df.columns]
        df = df.reindex(columns=first + [c for c in df.columns if c not in first])
    df.attrs["shards"] = f"{len(shards)}/{len(manifest.shards)}"
    df.attrs["version"] = manifest.version[:12]
    return df


def load_recording(name: str, recording_id: str, *, version: str | None = None) -> td.Frame:
    """One recording of a frames dataset as the published ``td.Frame`` (not coerced).

    Looks the id up in the :func:`noise_recordings` index (``recording_id`` or
    the dload key) and streams only its shard; shards not indexed yet are
    indexed on the way, so a cold lookup costs at most one pass.
    """
    manifest = _pinned_manifest(name, version)
    target = str(recording_id)
    for i, info in enumerate(manifest.shards):
        rows, found = _index_shard(name, manifest.version, i, info.digest, want=target)
        if found is not None:
            return found
        if any(target in (r["recording_id"], r["key"]) for r in rows):
            from data_processing.streams import iter_published_shard

            for key, frame in iter_published_shard(name, i, manifest.version):
                if target in (key, str(_recording_row(key, frame, i)["recording_id"])):
                    return frame
    raise KeyError(f"no recording {target!r} in {name}@{manifest.version[:12]}")
