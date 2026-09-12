"""The committed refined rotor-speed sidecars, as a publishable time series.

The raw telemetry is a tachometer (DREGON ``motors_measured``, Michael's
``rps``) or a command track (DREGON room2 publishes only ``motors_command``),
and it carries a scale-like error of a few tenths of a percent. The F_VK/L-BFGS
refiner corrects it against the recording's own comb, window by window, and
``scripts/refine_dregon_rps.py`` writes one sidecar per recording under
:data:`LABEL_DIR`.

This module is the seam between those sidecars and the published frame
datasets: :func:`refined_series` turns a sidecar into a ``td`` event series on
the recording's own clock, so ``sources.dregon`` / ``sources.michaels`` can
attach it as **another label track** (:data:`REFINED_KEY`) beside the raw one.
Consumers then choose which label to train on instead of re-deriving anything.

Two rules make the track safe to consume:

* **It is always defined where a raw label is.** A recording with no sidecar, or
  one whose windows were all rejected, publishes the reference track itself —
  so ``rps_refined`` never silently misses frames that ``motors_measured`` has.
* **Standby is never refined.** The sidecars are gated by
  :mod:`data_processing.rps_gating`: standby frames carry the telemetry exactly,
  settled cruise carries the refined trajectory, and the ramp blends the
  correction. A static shaft's comb is too weak to refine against, and
  correcting it made the refiner's own fitness worse.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

#: Where the committed sidecars live (repo-relative).
LABEL_DIR = Path(__file__).resolve().parent / "refined_labels"
#: The published track name, the same for both rigs.
REFINED_KEY = "rps_refined"
#: Raw label tracks, in the order they are preferred as the refinement's
#: reference. DREGON room1 carries the tachometer, room2 only the command
#: track, and Michael's rig calls its track ``rps`` - a chain pinned to
#: DREGON's two keys makes Michael's frames invisible.
REFERENCE_PREFERENCE: tuple[str, ...] = ("motors_measured", "motors_command", "rps")


def sidecar_path(recording_id: str, label_dir: Path | str | None = None) -> Path:
    return Path(label_dir or LABEL_DIR) / f"{recording_id}.npz"


def load_sidecar(recording_id: str, label_dir: Path | str | None = None) -> dict[str, Any] | None:
    """The sidecar of one recording, or ``None`` if it has not been refined."""
    path = sidecar_path(recording_id, label_dir)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def refined_series(
    recording_id: str,
    reference: td.Series,
    *,
    audio_t_start: float,
    label_dir: Path | str | None = None,
) -> td.Series:
    """``(rotor, time)`` refined label series for one recording.

    ``reference`` is the raw label track the frame already carries, and is
    returned unchanged when there is no sidecar — that is what keeps the
    published field defined for every frame that has a label at all.

    ``audio_t_start`` is the anchor and it must be the frame's AUDIO start.
    The sidecar's ``ft`` is seconds from the published recording's audio
    ``t_start`` with its own ``t0_offset_s`` already added in (the loader trims
    each frame to the audio-telemetry overlap, and the sidecar undoes that trim
    so a consumer can apply the labels to the untrimmed recording). Anchoring on
    the TELEMETRY track instead would add that offset twice — 0.01 s on
    Michael's rig, and an arbitrary amount on DREGON, whose clock is absolute
    unix seconds.
    """
    data = load_sidecar(recording_id, label_dir)
    if data is None:
        return reference
    ft = np.asarray(data["ft"], dtype=np.float64)
    r = np.asarray(data["r_refined"], dtype=np.float32)
    if r.ndim != 2 or r.shape[-1] != ft.size:
        raise ValueError(
            f"{recording_id}: sidecar r_refined {r.shape} does not match ft {ft.shape}"
        )
    base = float(audio_t_start)
    return td.events(ft + base, r, dims=("rotor", "time"), t_start=base)


def reference_track(frame_keys: Any, preference: tuple[str, ...] = ()) -> str | None:
    """The first label track present, from a preference chain."""
    for key in preference or REFERENCE_PREFERENCE:
        if key in frame_keys:
            return key
    return None


def attach_refined(
    frame: td.Frame,
    recording_id: str,
    *,
    preference: tuple[str, ...] = (),
    label_dir: Path | str | None = None,
) -> td.Frame:
    """Add :data:`REFINED_KEY` to a frame that carries a raw label track.

    A frame with no label track at all (the DREGON bench recordings have no
    telemetry) is returned unchanged: there is nothing to refine and nothing to
    fall back to. A frame with a label but no ``audio`` is an error, because the
    sidecar's clock is defined against the audio start.
    """
    keys = set(frame.keys())
    ref_key = reference_track(keys, preference)
    if ref_key is None:
        return frame
    if "audio" not in keys:
        raise ValueError(f"{recording_id}: cannot anchor refined labels without an audio track")
    series = refined_series(
        recording_id,
        frame[ref_key],
        audio_t_start=float(frame["audio"].t_start),
        label_dir=label_dir,
    )
    return frame.with_entry(REFINED_KEY, series)
