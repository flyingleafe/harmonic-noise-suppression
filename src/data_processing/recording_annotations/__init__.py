"""Recording-condition annotations of published drone-noise recordings.

Ten searchable facets per exact parent sample (``rig_mount``, ``speed_regime``,
``propulsion``, ``rotor_activity``, ``relative_motion``, ``speed_supplied``,
``sound_content``, ``environment``, ``blade_count``, ``wind``) with evidence,
role maps and acoustic shaft-rate windows, published as the read-only dload
companion release ``drone-recording-annotations``: one SQLite database per
parent ``dataset@version`` (no audio). Filter in SQLite, then fetch audio for
the selected keys only::

    from data_processing import recording_annotations as ra

    idx = ra.open_release().index("DREGON-frames")          # parent's dload.lock pin
    idx.facet_counts({"speed_regime": ["constant", "mixed"]})
    rows = idx.select({"speed_regime": "constant", "wind": "unknown",
                       "tach_status": ["resolved", "partial"]}, limit=50)
    frame = ra.load_frame("DREGON-frames", rows[0]["key"])
    frame["meta"]["recording_conditions"]["rig_mount"]
    frame["rps_acoustic"]                  # (rank|rotor, time) rev/s at every attempted window centre
    frame["rps_acoustic_uncertainty"]      # estimator-reported tolerance, NOT a calibrated 1-sigma
    frame["rps_acoustic_window"]           # (bound, time) absolute window start/end
    frame["rps_acoustic_accepted"]         # (time,) rejected windows are NaN gaps, never interpolate

Annotations bind to the exact parent version: a historical pin without its own
database raises :class:`NotAnnotatedError` instead of borrowing newer labels.
``rps_acoustic`` uses dim ``rank`` (ascending rate rank within each window, not
a tracked rotor) unless the estimator reported physical identity; it exists
only when some accepted window resolved a rate. Source series (native ``rps``,
measured/command/refined tracks) are returned unchanged. The SQLite schema is
documented on :mod:`.index` (``_DDL``) and in the catalog docs.

Layout: :mod:`.index` (database schema, :class:`IndexWriter`, :class:`Index`),
:mod:`.storage` (release packaging, ranged parent reads, annotated Frames);
``schema``/``rules`` + ``classify_*`` modules produce the records.
"""

from data_processing.recording_annotations.index import (
    FACET_NAMES,
    KNOWN,
    MAX_PAGE,
    NO_TACH,
    NULL_LABEL,
    TACH_STATUSES,
    Index,
    IndexWriter,
    NotAnnotatedError,
    SampleLocation,
    location_from_entry,
    open_index,
)
from data_processing.recording_annotations.storage import (
    DB_FIELD,
    MEDIA_AUDIO,
    RELEASE_LAYOUT,
    RELEASE_NAME,
    UNCERTAINTY_KIND,
    ParentInfo,
    Release,
    annotate_frame,
    decode_media_audio,
    decode_media_frame,
    decode_sample,
    iter_frames,
    load_frame,
    located_samples,
    media_field,
    media_input,
    open_release,
    pack_index,
    publish,
    read_spans,
    select,
    select_all,
)

__all__ = [
    "DB_FIELD",
    "MEDIA_AUDIO",
    "FACET_NAMES",
    "KNOWN",
    "MAX_PAGE",
    "NO_TACH",
    "NULL_LABEL",
    "RELEASE_LAYOUT",
    "RELEASE_NAME",
    "TACH_STATUSES",
    "UNCERTAINTY_KIND",
    "Index",
    "IndexWriter",
    "NotAnnotatedError",
    "ParentInfo",
    "Release",
    "SampleLocation",
    "annotate_frame",
    "decode_media_audio",
    "decode_media_frame",
    "decode_sample",
    "iter_frames",
    "load_frame",
    "located_samples",
    "location_from_entry",
    "media_field",
    "media_input",
    "open_index",
    "open_release",
    "pack_index",
    "publish",
    "read_spans",
    "select",
    "select_all",
]
