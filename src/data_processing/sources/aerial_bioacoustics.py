"""Drone-borne bioacoustics recordings (birds and bats) with rotor noise.

Four open deposits, each a separate registry source. The builders read the
publisher's files exactly as fetched: WAVs straight from the record, or
members streamed out of the published ZIPs. Nothing is extracted to disk.

- ``QMUL-insitu-aerial``: Zenodo 19925797, CC BY 4.0. Wang, Clayton &
  Rossberg (2026), MEE, doi:10.1111/2041-210X.70324. One field session
  (20 Nov 2023, Leigh-on-Sea, UK). A DJI Mavic 2 Pro hovers at 65/82/98 ft
  (20/25/30 m) above the canopy. A Rode NTG4+ shotgun microphone and a Tascam
  DR-10X recorder hang 2 m below it on a steel fishing line, facing down.
  Ground recordings run at the same time; each hover is one Frame holding the
  suspended WAV (``audio``) and the ground WAV (``ground_audio``) on their own
  native grids, placed by the authors' cue offset. Pre-flight ambient parts
  with no drone, the authors' time-montage merges and their denoised outputs
  are separate Frames (the last two as derivatives).
- ``QMUL-bird-drone``: Zenodo 8403622, CC BY 4.0. Wang, Clayton & Rossberg
  (2023), MEE, doi:10.1111/2041-210X.14234. Same suspended rig. A ground
  loudspeaker plays bird calls while the drone hovers at 5-30 m. The deposit
  also has ``clean.wav`` (bird-only recording) and ``original.wav`` (the
  source file that was played back). That bird audio is third-party; see
  ``rights_note``.
- ``Chirocopter``: Dryad doi:10.5061/dryad.cf472n3, CC0, fetched from its
  byte-identical Zenodo mirror 4993744. Fu, Kinniry & Kloepper (2018), MEE,
  doi:10.1111/2041-210X.12992, and Kloepper & Kinniry (2018), Sci. Rep.,
  doi:10.1038/s41598-018-26122-z. A DJI S900 hexacopter carries a
  Dodotronics Ultramic-250 (250 kHz). It climbs a 5-50 m hover ladder over a
  cave of Tadarida brasiliensis in Sierra County, NM (24 May-5 Jun 2017).
  There are 14 full flight recordings (one per battery) and 8 short noise
  tests of baffle/pole configurations. Each day folder also holds two iOSD
  ``FLY*.DAT`` flight logs; the original logs stay byte-exact in the raw ZIPs.
- ``SDU-DroneBatArray``: Zenodo 7329733, CC BY 4.0. Jespersen et al. (2022),
  Ecol. Evol., doi:10.1002/ece3.9577. A four-microphone vertical line array
  (Knowles FG, 1.30 m spacing) hangs 1/10/30 m below a DJI M210 V2. The
  deposit has noise tests at 375 kHz, the system noise floor, and four
  excerpts of bat recordings at 300 kHz (30 m tether).

Every Frame holds whole published WAVs, keyed by recording id, on the
recording's own sample clock (``t_start`` 0). The native PCM16 samples are
stored as exact float32 (``int16 / 32768``) at the native rate and channel
count; nothing is filtered, resampled, trimmed or joined. No two files in these
deposits share a sample clock; the only multi-file Frames are the QMUL in-situ
hover takes, whose two Series keep separate grids and a documented,
cue-level (not sample-accurate) offset.
"""

from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Iterator, Mapping
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import tdseries as td

from data_processing.sources._common import meta_frame, safe_key

if TYPE_CHECKING:
    from data_processing.sources.iosd_dat import IosdTelemetry


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 22):
            digest.update(block)
    return digest.hexdigest()


def _verify(root: Path, files: dict[str, tuple[int, str]], dataset: str) -> None:
    """Fail closed unless every pinned publisher file is present, intact."""
    for rel, (size, md5) in files.items():
        path = root / rel
        if not path.is_file() or path.stat().st_size != size:
            raise FileNotFoundError(f"{dataset}: missing or truncated {rel} under {root}")
        if _md5(path) != md5:
            raise ValueError(f"{dataset}: md5 mismatch for {rel}")


#: How the stored float32 relates to the published PCM16 (lossless).
AUDIO_ENCODING = "float32 = int16 PCM / 32768 (exact)"
#: Frames per ``readframes`` call; bounds the transient byte buffer only.
_READ_BLOCK = 1 << 22


@contextmanager
def _open_wav(root: Path, container: str, member: str | None):
    """``wave`` reader on a plain file or one ZIP member, read strictly forward
    (no seek to a deflated member's end)."""
    import wave

    with ExitStack() as stack:
        if member is None:
            src = stack.enter_context(open(root / container, "rb"))
        else:
            archive = stack.enter_context(zipfile.ZipFile(root / container))
            src = stack.enter_context(archive.open(member))
        yield stack.enter_context(wave.open(src, "rb"))


def _read_pcm16(
    root: Path, container: str, member: str | None, where: str
) -> tuple[np.ndarray, int]:
    """Every sample of one published PCM16 WAV as exact float32 ``(channel, time)``."""
    with _open_wav(root, container, member) as f:
        sr, ch, n = f.getframerate(), f.getnchannels(), f.getnframes()
        if f.getsampwidth() != 2 or f.getcomptype() != "NONE":
            raise ValueError(f"{where}: expected uncompressed PCM16")
        if n <= 0:
            raise ValueError(f"{where}: empty audio")
        audio = np.empty((ch, n), dtype=np.float32)
        pos = 0
        while pos < n:
            block = np.frombuffer(f.readframes(min(_READ_BLOCK, n - pos)), dtype="<i2")
            if not len(block):
                raise ValueError(f"{where}: data ends at frame {pos} of {n}")
            block = block.reshape(-1, ch)
            audio[:, pos : pos + len(block)] = block.T
            pos += len(block)
    audio *= np.float32(1 / 32768)
    return audio, sr


def _recording(
    root: Path,
    container: str,
    member: str | None,
    rid: str,
    meta_kw: dict[str, Any],
    *,
    mic_pos: np.ndarray | None = None,
    expect: tuple[int, int] | None = None,
    t_start: float = 0.0,
    other_series: dict[str, td.Series] | None = None,
) -> tuple[str, td.Frame]:
    """``(key, Frame)`` for one whole published WAV.

    Audio is float32 ``(time,)`` for mono or ``(mic, time)`` otherwise, at the
    native rate, starting at ``t_start`` s (0 unless it shares a Frame with
    another device's recording). ``meta_kw`` goes to :func:`meta_frame`; its
    ``extra`` gets the original file identity and exact sample count added.
    ``expect`` = ``(sample_rate, channels)`` the deposit documents.
    """
    where = f"{rid} ({container}:{member})"
    audio, sr = _read_pcm16(root, container, member, where)
    ch, n = audio.shape
    if expect is not None and (sr, ch) != expect:
        raise ValueError(f"{where}: {ch} ch at {sr} Hz, expected {expect[1]} ch at {expect[0]} Hz")
    extra = dict(meta_kw.get("extra") or {})
    extra.update(
        {
            "original_container": container,
            "original_member": member,
            "original_format": "WAV",
            "original_subtype": "PCM_16",
            "audio_encoding": AUDIO_ENCODING,
            "sample_rate_hz": sr,
            "channels": ch,
            "n_samples": n,
            "duration_s": n / sr,
        }
    )
    groups = {k: v for k, v in meta_kw.items() if k not in ("dataset", "extra")}
    entries: dict[str, Any] = {
        "audio": td.uniform(
            audio[0] if ch == 1 else audio,
            sr,
            dims=("time",) if ch == 1 else ("mic", "time"),
            t_start=t_start,
        )
    }
    if mic_pos is not None:
        entries["mic_pos"] = td.wrap(np.asarray(mic_pos, dtype=np.float64), dims=("mic", None))
    entries.update(other_series or {})
    entries["meta"] = meta_frame(rid, meta_kw["dataset"], **groups, extra=extra)
    return safe_key(rid), td.Frame(entries)


# ═══ QMUL in-situ aerial bioacoustics (Zenodo 19925797) ══════════════════════

QMUL_INSITU = "QMUL-insitu-aerial"
QMUL_INSITU_FILES = {"Data.zip": (1163385739, "a9c0ca5045e6c6b2ab0684f3f0a09908")}

#: Shared by both QMUL deposits. Source: Wang (2026, Quiet Drones), section 3,
#: Fig. 2, after Wang et al. (2023). Section 4 states the 2026 in-situ study
#: used the same rig.
QMUL_RIG = {
    "uav": "DJI Mavic 2 Pro (quadrotor)",
    "microphone": "Rode NTG4+ shotgun, pointing down at the ground; Rycote Super Softie windshield",
    "recorder": "Tascam DR-10X (PCM16, 48 kHz as published)",
    "suspension": "microphone and recorder hang 2 m below the drone on a steel fishing line",
    "mic_pos_frame": "metres, origin at the line's attachment point under the drone, z up; "
    "assumes the line hangs vertically (nominal, not tracked; pendulum/sway unknown)",
    "mic_pos_certainty": "nominal from the published 2 m line length; x/y assumed 0",
    "evidence": "L. Wang, 'Can drones hear birds from the sky?', Quiet Drones 2026 "
    "(https://edocs.tib.eu/files/e01mr26/1984112570.pdf) pp. 3-5, section 3/Fig. 2 "
    "(reproduced from Wang et al. 2023, CC BY) and section 4",
}
QMUL_SUSPENDED_MIC_POS = np.array([[0.0, 0.0, -2.0]])
QMUL_FREQUENCY_NOTE = (
    "PCM16 at 48 kHz (Nyquist 24 kHz); NTG4+ rated 20 Hz-20 kHz. Uncalibrated "
    "(no published sensitivity/gain); levels are relative. Derivatives at 22.05 kHz."
)
_FT_TO_M = 0.3048

#: Data.zip member -> (kind, altitude_ft or None). Directories and the
#: ``detection/`` MATLAB tables (manuscript analysis, not audio) are left out.
_INSITU_RAW = "raw recording 20th November 2023/"
QMUL_INSITU_MEMBERS: Mapping[str, tuple[str, int | None]] = (
    {
        f"{_INSITU_RAW}aerial recording/Mavic2Pro_Section1_Altitude({ft})_Flight_Audio.wav": (
            "aerial",
            int(ft[:-2]),
        )
        for ft in ("65ft", "82ft", "98ft")
    }
    | {
        f"{_INSITU_RAW}ground recording/Mavic2Pro_Section1_Altitude({ft})_Ground_Audio.WAV": (
            "ground",
            int(ft[:-2]),
        )
        for ft in ("65ft", "82ft", "98ft")
    }
    | {
        f"{_INSITU_RAW}ambient recording/Section1_Ground_Audio_Separate_Part{p}.WAV": (
            "ambient",
            None,
        )
        for p in (1, 2, 3)
    }
    | {
        "align and merge/1120_flight_merge.wav": ("merge_aerial", None),
        "align and merge/1120_ground_merge.wav": ("merge_ground", None),
        "align and merge/1120_ambient_merge.wav": ("merge_ambient", None),
        "denoising/1120_flight_merge_y.wav": ("denoised_aerial", None),
        "denoising/1120_ground_merge_y.wav": ("denoised_ground", None),
    }
)

_INSITU_KIND = {
    # kind: (mic_placement, condition, subcategory, source_kind). Hover takes get
    # the nominal altitude appended ("_20m"); ground WAVs ride on their take.
    "aerial": ("suspended", "flight", "hover_suspended_and_ground", "original_recording"),
    "ambient": ("external", "rotors_off", "ambient_raw_no_drone", "original_recording"),
    "merge_aerial": ("suspended", "flight", "derivative_aligned_merge", "derivative_aligned_merge"),
    "merge_ground": ("external", "flight", "derivative_aligned_merge", "derivative_aligned_merge"),
    "merge_ambient": (
        "external",
        "rotors_off",
        "derivative_aligned_merge",
        "derivative_aligned_merge",
    ),
    "denoised_aerial": ("suspended", "flight", "derivative_denoised", "derivative_denoised"),
    "denoised_ground": ("external", "flight", "derivative_denoised", "derivative_denoised"),
}
#: File-name altitude (ft) -> the paper's nominal hover altitude (m).
_INSITU_NOMINAL_M = {65: 20, 82: 25, 98: 30}

#: Where each raw WAV sits in the authors' 22.05 kHz merge, measured from the
#: deposit itself: FFT cross-correlation of 20 s pieces, every 90 s, of
#: ``resample_poly(raw, 147, 320)`` against the whole merge (Apocrita Slurm
#: 37570711, 2026-11-26). Every matching piece of a file gave the same integer
#: lag. raw stem -> (merge stem, merge start, merge end, lag, level ratio
#: merge/raw, waveform correlation range); merge positions and lag are 22.05 kHz
#: samples, lag = merge sample - raw seconds * 22050. The 600 s flight/ground
#: segment ends follow from the 1800 s merges and the loss of correlation past
#: them; the ambient merge is the three parts end to end, untrimmed.
_MERGE_RATE = 22050
QMUL_INSITU_MERGE_MAP: dict[str, tuple[str, int, int, int, float, tuple[float, float]]] = {
    "Mavic2Pro_Section1_Altitude(65ft)_Flight_Audio": (
        "1120_flight_merge",
        0,
        13230000,
        -2395533,
        1.006,
        (0.998, 1.0),
    ),
    "Mavic2Pro_Section1_Altitude(82ft)_Flight_Audio": (
        "1120_flight_merge",
        13230000,
        26460000,
        7555746,
        0.316,
        (0.993, 0.996),
    ),
    "Mavic2Pro_Section1_Altitude(98ft)_Flight_Audio": (
        "1120_flight_merge",
        26460000,
        39690000,
        20488002,
        1.001,
        (0.74, 0.90),
    ),
    "Mavic2Pro_Section1_Altitude(65ft)_Ground_Audio": (
        "1120_ground_merge",
        0,
        13230000,
        -3149489,
        0.992,
        (0.987, 0.996),
    ),
    "Mavic2Pro_Section1_Altitude(82ft)_Ground_Audio": (
        "1120_ground_merge",
        13230000,
        26460000,
        6786469,
        0.986,
        (0.975, 0.989),
    ),
    "Mavic2Pro_Section1_Altitude(98ft)_Ground_Audio": (
        "1120_ground_merge",
        26460000,
        39690000,
        20724631,
        1.002,
        (0.9996, 0.9998),
    ),
    "Section1_Ground_Audio_Separate_Part1": (
        "1120_ambient_merge",
        0,
        19883103,
        0,
        1.004,
        (0.9997, 0.9999),
    ),
    "Section1_Ground_Audio_Separate_Part2": (
        "1120_ambient_merge",
        19883103,
        33126686,
        19883103,
        1.003,
        (0.9996, 0.9999),
    ),
    "Section1_Ground_Audio_Separate_Part3": (
        "1120_ambient_merge",
        33126686,
        48048010,
        33126686,
        1.003,
        (0.9996, 0.9999),
    ),
}
_MERGE_NOTE = (
    "measured from the deposit (QMUL_INSITU_MERGE_MAP): the authors' alignment on a "
    "pre-flight acoustic cue, resolved to 1/22050 s after resampling. The aerial "
    "(Tascam DR-10X) and ground recorders are separate devices with independent sample "
    "clocks; the raw files are not sample-synchronous (QMUL_INSITU_PAIR_LAG)"
)
#: Hover altitude (ft) -> (aerial-clock window s, ground-minus-aerial lag ms) at the
#: authors' offset: 8 s Hann windows every 20 s, 300 Hz-3 kHz and 1-10 kHz GCC-PHAT
#: (Apocrita Slurm 37572358). During hover the drone noise gives a stable lag that
#: rises overall by 5-12 ms in all three pairs (about 8.5-18 ppm if read as clock drift,
#: 370-790 samples at 48 kHz over a 15 min file); propagation (drone to ground vs 2 m to the
#: suspended mic) and drone drift are not separable from it. Climb/landing windows
#: give unrelated lags; peak-to-sidelobe ratios are only 1.3-3.5.
QMUL_INSITU_PAIR_LAG = {
    65: ((165.0, 745.0), (49.5, 58.4)),
    82: ((205.0, 825.0), (88.9, 94.2)),
    98: ((185.0, 825.0), (88.6, 100.3)),
}


def _merge_alignment(stem: str) -> dict[str, Any]:
    """The span of one raw WAV that the authors put into their merge."""
    merge, start, end, lag, ratio, corr = QMUL_INSITU_MERGE_MAP[stem]
    raw = [(start - lag) / _MERGE_RATE, (end - lag) / _MERGE_RATE]
    return {
        "merge": merge,
        "merge_interval_s": [start / _MERGE_RATE, end / _MERGE_RATE],
        "raw_interval_s": raw,
        "raw_interval_samples": [round(t * 48000) for t in raw],
        "resolution_s": 1 / _MERGE_RATE,
        "level_ratio_merge_over_raw": ratio,
        "waveform_correlation": list(corr),
        "note": _MERGE_NOTE,
    }


def _ground_stem(aerial_stem: str) -> str:
    """The ground recorder's raw WAV of the same hover."""
    return aerial_stem.removesuffix("_Flight_Audio") + "_Ground_Audio"


def _insitu_frame_id(stem: str) -> str:
    """Recording id of the Frame that carries one raw WAV."""
    for suffix in ("_Flight_Audio", "_Ground_Audio"):
        if stem.endswith(suffix):
            return stem.removesuffix(suffix)
    return stem


def _insitu_parents(kind: str) -> list[str]:
    """Raw recording ids a derivative is built from (deposit readme)."""
    raw_kind = kind.split("_")[-1]
    return [Path(m).stem for m, (k, _) in QMUL_INSITU_MEMBERS.items() if k == raw_kind]


def _take_alignment(aerial: str, alt_ft: int) -> tuple[float, float, dict[str, Any]]:
    """``(audio t_start, ground_audio t_start, meta)`` on the take's shared axis.

    The ground recorder's native time ``t_g`` equals the suspended recorder's
    ``t_a + producer_offset_s`` at equal time in the authors' montages. The
    axis origin is the earlier of the two native starts, so neither series is
    cropped or padded and both keep their native 48 kHz grids.
    """
    ground = _ground_stem(aerial)
    a_span, g_span = _merge_alignment(aerial), _merge_alignment(ground)
    offset = g_span["raw_interval_s"][0] - a_span["raw_interval_s"][0]
    a0, g0 = max(offset, 0.0), max(-offset, 0.0)
    (t0, t1), (l0, l1) = QMUL_INSITU_PAIR_LAG[alt_ft]
    return (
        a0,
        g0,
        {
            "time_origin": "earlier native recording start: the ground recorder's first sample"
            if offset > 0
            else "earlier native recording start: the suspended recorder's first sample",
            "audio_t_start_s": a0,
            "ground_audio_t_start_s": g0,
            "producer_offset_s": offset,
            "definition": "ground_audio native time = audio native time + producer_offset_s at "
            "equal time in the authors' flight/ground montages",
            "resolution_s": 1 / _MERGE_RATE,
            "uncertainty": "producer montage/cue only; not sample-accurate or phase-safe. After "
            f"this offset the hover drone noise reaches the ground recorder {l0}-{l1} ms after "
            f"the suspended mic (suspended-clock {t0:g}-{t1:g} s, QMUL_INSITU_PAIR_LAG): the "
            "residual can include propagation distance, drone motion and 8.5-18 ppm-equivalent "
            "clock drift, none of them corrected",
            "not_applied": "no resampling, drift correction, propagation removal, cropping or "
            "padding; every pre- and post-roll sample of both files is kept",
            "producer_spans": {"audio": a_span, "ground_audio": g_span},
        },
    )


def build_qmul_insitu(root: Path) -> Iterator[tuple[str, td.Frame]]:
    """One Frame per hover take (suspended ``audio`` + ``ground_audio``), per
    ambient part, and per author derivative."""
    root = Path(root)
    _verify(root, QMUL_INSITU_FILES, QMUL_INSITU)
    for member, (kind, alt_ft) in QMUL_INSITU_MEMBERS.items():
        if kind == "ground":
            continue  # carried as ``ground_audio`` by the same hover's take Frame
        placement, condition, subcat, source_kind = _INSITU_KIND[kind]
        nominal_m = None if alt_ft is None else _INSITU_NOMINAL_M[alt_ft]
        stem = Path(member).stem
        rid = _insitu_frame_id(stem)
        derivative = source_kind != "original_recording"
        operating = {
            "condition": condition,
            "flight_mode": "hover" if condition == "flight" else None,
            "hover_altitude_m": nominal_m,
            "hover_altitude_ft_filename": alt_ft,
            "altitude_source": "file name in feet (65/82/98 ft); the paper's nominal 20/25/30 m"
            if alt_ft is not None
            else None,
            "contains_rotor_noise": condition == "flight",
            "external_source": "other",
            "external_source_detail": "natural in-situ bird vocalisations, wind and woodland background (unannotated)",
            "rotor_telemetry": "none published",
        }
        observation = {
            "mic_placement": placement,
            "type": {
                "suspended": "uav_suspended_microphone",
                "external": "ground_recorder",
            }[placement],
            "detail": {
                "aerial": "audio: recorder hanging 2 m below the hovering drone; ground_audio: "
                "ground recorder running during the same hover (its offset from the drone and "
                "its hardware are not given in the deposit)",
                "ground": "ground recorder running while the drone hovered above the canopy",
                "ambient": "ground recorder, pre-flight ambient with no drone",
            }[kind.split("_")[-1]],
            "environment": "woodland hillside, Leigh-on-Sea, UK (outdoor field)",
            "recording_date": "2023-11-20",
            "source_motion": "hovering" if condition == "flight" else "none",
        }
        extra: dict[str, Any] = {
            "source_kind": source_kind,
            "doi": "10.5281/zenodo.19925797",
            "paper_doi": "10.1111/2041-210X.70324",
            "frequency_note": QMUL_FREQUENCY_NOTE,
            "clock": "audio sample clock only; the deposit publishes no wall-clock "
            "timestamps or telemetry",
        }
        if placement == "suspended":
            extra["geometry"] = QMUL_RIG
        t_start, other_series = 0.0, None
        if kind == "aerial":
            if alt_ft is None:
                raise ValueError(f"{member}: hover take without a file-name altitude")
            ground = _ground_stem(stem)
            ground_member = f"{_INSITU_RAW}ground recording/{ground}.WAV"
            a0, g0, alignment = _take_alignment(stem, alt_ft)
            pcm, sr = _read_pcm16(root, "Data.zip", ground_member, f"{rid} ({ground_member})")
            if (sr, pcm.shape[0]) != (48000, 1):
                raise ValueError(
                    f"{ground_member}: {pcm.shape[0]} ch at {sr} Hz, expected mono 48 kHz"
                )
            t_start = a0
            other_series = {"ground_audio": td.uniform(pcm[0], sr, dims=("time",), t_start=g0)}
            subcat = f"{subcat}_{nominal_m}m"
            extra["take_alignment"] = alignment
            extra["series"] = {
                "audio": {
                    "role": "suspended NTG4+/DR-10X, 2 m below the drone",
                    "original_id": stem,
                    "t_start_s": a0,
                },
                "ground_audio": {
                    "role": "ground recorder during the same hover (hardware and position unpublished)",
                    "original_id": ground,
                    "original_container": "Data.zip",
                    "original_member": ground_member,
                    "sample_rate_hz": sr,
                    "n_samples": pcm.shape[1],
                    "duration_s": pcm.shape[1] / sr,
                    "t_start_s": g0,
                },
            }
            del pcm
        elif kind == "ambient":
            extra["producer_alignment"] = _merge_alignment(stem)
            extra["producer_alignment"]["note"] = (
                "measured from the deposit (QMUL_INSITU_MERGE_MAP): the ambient merge is "
                "the three parts end to end, untrimmed. The parts are NOT joined here: "
                "each ends in about 1 ms (48-50 samples) of digital zeros, the same "
                "file-close signature as every ground-recorder file, and no timestamps "
                "establish that one part starts where the previous stopped"
            )
        if derivative:
            parents = _insitu_parents(kind)
            extra["derivative_of"] = {
                "merge": "a time montage, not a mix: the three raw files of the same type "
                "resampled to 22.05 kHz and placed end to end. Flight and ground merges "
                "keep one 600 s span per hover, chosen so that equal merge times are "
                "simultaneous per the authors' pre-flight cue (deposit readme; spans measured)",
                "denoised": "the matching aligned merge, after the authors' deep-learning "
                "noise reduction (MEE 2026); model output, not a recording",
            }[kind.split("_")[0]]
            extra["parent_recordings"] = parents
            extra["parent_frames"] = [safe_key(_insitu_frame_id(p)) for p in parents]
            extra["merge_segments"] = [{"raw": p, **_merge_alignment(p)} for p in parents]
            if kind.startswith("denoised"):
                extra["parent_derivative"] = (
                    "1120_"
                    + {"aerial": "flight", "ground": "ground"}[kind.split("_")[-1]]
                    + "_merge"
                )
            extra["duplicate_note"] = (
                "derivative of this source's raw recordings (parent_recordings); not an independent recording"
            )
        yield _recording(
            root,
            "Data.zip",
            member,
            rid,
            {
                "dataset": QMUL_INSITU,
                "system": {
                    "category": "multirotor_uav",
                    "platform": "DJI Mavic 2 Pro",
                    "n_rotors": 4,
                    "rig": "qmul_mavic2pro_suspended_ntg4",
                },
                "observation": observation,
                "operating": operating,
                "label": {
                    "subcategory": subcat,
                    "rig": "Mavic 2 Pro + 2 m suspended NTG4+/DR-10X",
                    "health": "unknown",
                    "mixed": "acoustic (natural birds + rotor noise)"
                    if condition == "flight"
                    else "acoustic (natural birds, no drone)",
                    "session": "20 Nov 2023 (1 of the study's 8 sessions)",
                },
                "extra": extra,
            },
            mic_pos=QMUL_SUSPENDED_MIC_POS if placement == "suspended" else None,
            expect=(22050 if derivative else 48000, 1),
            t_start=t_start,
            other_series=other_series,
        )


QMUL_INSITU_PROVENANCE = {
    "source_url": "https://zenodo.org/records/19925797",
    "doi": "10.5281/zenodo.19925797",
    "version": "Zenodo record 19925797 (concept 19925796), published 2026-04-30, single version",
    "license": "CC BY 4.0",
    "rights_note": "Deposit licence CC BY 4.0 (Zenodo metadata). Attribution required: "
    "cite the deposit and the paper. No third-party audio: natural field recordings.",
    "citation": "Wang L., Clayton M., Rossberg A. G. (2026). In situ aerial bioacoustic "
    "monitoring from extremely noisy drone recordings. Methods Ecol. Evol., "
    "doi:10.1111/2041-210X.70324; data: In situ aerial bioacoustic recordings, "
    "Zenodo doi:10.5281/zenodo.19925797.",
    "collection_method": "drone hovering above a woodland canopy with a suspended recorder, plus "
    "simultaneous ground and pre-flight ambient recordings",
    "equipment": "DJI Mavic 2 Pro; Rode NTG4+ + Tascam DR-10X hanging 2 m below the drone",
    "observation_type": "uav_suspended_microphone",
    "sample_rates": [48000, 22050],  # raw 48 kHz; authors' merges/denoised 22.05 kHz
    "channels": 1,  # every WAV header: mono PCM16
    "release_bytes": 1163385739,
    "description": "11 whole-file Frames. 3 hover takes (20/25/30 m): the suspended WAV "
    "as 'audio' and the simultaneous ground WAV as 'ground_audio', each mono 48 kHz on its "
    "own native grid, placed on one axis by the authors' cue offset (take_alignment; "
    "849.8-977.4 s per file, nothing cropped or resampled). 3 ambient parts "
    "(901.7/600.6/676.7 s), kept separate. Derivatives: 3 time-montage merges + 2 "
    "denoised merges at 22.05 kHz (1800 s; ambient 2179.0 s). Not published: detection/ "
    "MATLAB tables (Tables 4-6, S2). They are "
    "aggregate species-count and Jaccard/Bray-Curtis tables with no event times "
    "(checked: MCOS 'table' objects holding name/count/accuracy). readme.txt is kept "
    "as provenance.",
    "geometry": QMUL_RIG,
}


# ═══ QMUL bird + drone playback (Zenodo 8403622) ═════════════════════════════

QMUL_BIRD_DRONE = "QMUL-bird-drone"
QMUL_BIRD_DRONE_FILES = {"bird_drone.zip": (323479480, "96c313dbcc881567797c975fe8c92fad")}
QMUL_BIRD_DRONE_MEMBERS = (
    "5m.wav",
    "10m.wav",
    "15m.wav",
    "20m.wav",
    "25m.wav",
    "30m.wav",
    "clean.wav",
    "original.wav",
)
_THIRD_PARTY_BIRDS = (
    "The bird calls are third-party recordings. Wang et al. (2023) trained on Xeno-canto "
    "and tested on the British Birdsong dataset. The deposit does not say which "
    "recordings were played or their per-recording licences (Xeno-canto items are "
    "commonly CC BY-NC(-SA)). Treat the bird content as restricted: internal research "
    "only, no redistribution."
)


def build_qmul_bird_drone(root: Path) -> Iterator[tuple[str, td.Frame]]:
    root = Path(root)
    _verify(root, QMUL_BIRD_DRONE_FILES, QMUL_BIRD_DRONE)
    for member in QMUL_BIRD_DRONE_MEMBERS:
        stem = Path(member).stem
        extra: dict[str, Any] = {
            "doi": "10.5281/zenodo.8403622",
            "paper_doi": "10.1111/2041-210X.14234",
            "frequency_note": QMUL_FREQUENCY_NOTE,
            "rights_note": _THIRD_PARTY_BIRDS,
            "clock": "audio sample clock only; no timestamps, telemetry or playback "
            "sync marks are published",
            "deposit_readme": "Real recording with drones at various hovering attitudes: "
            "5m-30m; clean.wav: bird-only recording; original.wav: original bird call",
        }
        if stem.endswith("m") and stem[:-1].isdigit():
            alt = float(stem[:-1])
            placement, condition, subcat, source_kind = (
                "suspended",
                "flight",
                f"playback_hover_{int(alt)}m",
                "original_recording",
            )
            observation = {
                "mic_placement": placement,
                "type": "uav_suspended_microphone",
                "detail": "recorder 2 m below the hovering drone; bird calls played from a ground loudspeaker",
                "environment": "outdoor field (Wang et al. 2023, Fig. 2b)",
                "source_motion": "hovering",
            }
            operating = {
                "condition": condition,
                "flight_mode": "hover",
                "hover_altitude_m": alt,
                "altitude_source": "file name (deposit readme: hovering altitudes 5-30 m)",
                "contains_rotor_noise": True,
                "external_source": "other",
                "external_source_detail": "bird calls from a loudspeaker on the ground",
                "rotor_telemetry": "none published",
            }
            extra["geometry"] = QMUL_RIG
            extra["loudspeaker_position"] = (
                "on the ground below the drone (exact offset not published)"
            )
            mic_pos = QMUL_SUSPENDED_MIC_POS
        elif stem == "clean":
            placement, condition, subcat, source_kind = (
                "external",
                "rotors_off",
                "playback_no_drone_control",
                "original_recording",
            )
            observation = {
                "mic_placement": placement,
                "type": "playback_reference_recording",
                "detail": "drone off; microphone 1 m vertically above and aimed down "
                "toward the upward-facing ground loudspeaker (Wang et al. 2023 §4.2.2)",
                "source_motion": "none",
            }
            operating = {
                "condition": condition,
                "contains_rotor_noise": False,
                "external_source": "other",
                "external_source_detail": "bird calls from the loudspeaker",
            }
            extra["geometry"] = {
                "mic_pos_frame": "metres, origin at the loudspeaker, +z vertically up",
                "mic_pos_certainty": "nominal experimental separation; not a surveyed XYZ",
                "orientation": "microphone points down towards the loudspeaker",
                "evidence": "Wang et al. 2023, doi:10.1111/2041-210X.14234 §4.2.2",
            }
            mic_pos = np.array([[0.0, 0.0, 1.0]])
        else:  # original.wav
            placement, condition, subcat, source_kind = (
                "other",
                "rotors_off",
                "playback_source_reference",
                "playback_source_file",
            )
            observation = {
                "mic_placement": placement,
                "type": "playback_source_file",
                "detail": "the bird-call file fed to the loudspeaker (deposit readme 'original bird call'), "
                "not a recording made in this study; 20 kHz as published",
                "source_motion": "none",
            }
            operating = {
                "condition": condition,
                "contains_rotor_noise": False,
                "external_source": "other",
                "external_source_detail": "third-party bird recordings (no microphone of this study)",
            }
            extra["frequency_note"] = (
                "PCM16 at 20 kHz (Nyquist 10 kHz), the third-party source as published"
            )
            mic_pos = None
        extra["source_kind"] = source_kind
        yield _recording(
            root,
            "bird_drone.zip",
            member,
            stem,
            {
                "dataset": QMUL_BIRD_DRONE,
                "system": {
                    "category": "multirotor_uav",
                    "platform": "DJI Mavic 2 Pro",
                    "n_rotors": 4,
                    "rig": "qmul_mavic2pro_suspended_ntg4",
                },
                "observation": observation,
                "operating": operating,
                "label": {
                    "subcategory": subcat,
                    "rig": "drone-off loudspeaker control"
                    if stem == "clean"
                    else "Mavic 2 Pro + 2 m suspended NTG4+/DR-10X",
                    "health": "unknown",
                    "mixed": "acoustic (loudspeaker playback + rotor noise)"
                    if condition == "flight"
                    else "none",
                },
                "extra": extra,
            },
            mic_pos=mic_pos,
            expect=(20000 if stem == "original" else 48000, 1),
        )


QMUL_BIRD_DRONE_PROVENANCE = {
    "source_url": "https://zenodo.org/records/8403622",
    "doi": "10.5281/zenodo.8403622",
    "version": "Zenodo record 8403622 (concept 8403621), published 2023-10-03, single version",
    "license": "CC BY 4.0 (deposit); third-party bird content, see rights_note",
    "rights_note": "Deposit licence CC BY 4.0 (Zenodo metadata). Attribution required. "
    "Playback material: 96 selected 5 s clips (480 s) from the British Birdsong "
    "test set, a Xeno-canto subset (Wang et al. 2023 §§3.2.1, 4.2.1); "
    "the release gives no original recording IDs or per-clip licences, "
    "so treat third-party bird content as internal-research only, no redistribution.",
    "citation": "Wang L., Clayton M., Rossberg A. G. (2023). Drone audition for bioacoustic "
    "monitoring. Methods Ecol. Evol. 14(12):3068-3082, doi:10.1111/2041-210X.14234; "
    "data: Bird and drone dataset, Zenodo doi:10.5281/zenodo.8403622.",
    "collection_method": "drone hovering at 5-30 m with a suspended recorder above an "
    "upward-facing Behringer EPA40 on the ground (Wang et al. 2023 §4.2.1, "
    "field photo Fig. 1b; the hardware diagram also appears as Fig. 2 in Wang's "
    "2026 Quiet Drones paper). Separate drone-off bird-only control (§4.2.2) and "
    "the playback source, not a study capture.",
    "equipment": "DJI Mavic 2 Pro; Rode NTG4+ + Tascam DR-10X hanging 2 m below the drone; "
    "clean control: microphone 1 m above the ground loudspeaker, pointing towards it",
    "observation_type": "uav_suspended_microphone",
    "sample_rates": [48000, 20000],  # hover/clean 48 kHz; original.wav (playback source) 20 kHz
    "channels": 1,  # every WAV header: mono PCM16
    "release_bytes": 323479480,
    "description": "8 whole-file Frames: 6 hover recordings (5/10/15/20/25/30 m, 48 kHz mono, "
    "650.6-801.3 s), clean.wav (bird-only control, 48 kHz, 509.7 s) and original.wav "
    "(playback source, 20 kHz, 490.0 s). Each hover is a separate flight; no file shares "
    "a sample clock or published sync mark with another.",
    "geometry": QMUL_RIG,
}


# ═══ Chirocopter (Dryad doi:10.5061/dryad.cf472n3 = Zenodo 4993744) ══════════

CHIROCOPTER = "Chirocopter"
CHIRO_CLOCK_SHA256 = "de8d92fe4ea7d1122edbc2f21698c461bf4bf8b0ea9da54ccb6ff03d3e33dd08"
#: Zenodo 4993744 keys -> (bytes, md5). Dryad version 1 (2019-02-22) lists the
#: same files with the same md5s. The five README_*.txt are byte-identical, so
#: only one is pinned.
CHIROCOPTER_FILES = {
    "README_for_UAV_Bat_Recordings1.txt": (806, "907603992ce650755beab0ef219bc199"),
    "UAV_Noise_Testing.zip": (128287, "9ede19c6471d121c4ffb49860129d084"),
    "UAV_Bat_Recordings1.zip": (848795660, "fdb0e55e8fd12d8ca9282c7dc7b2c2b0"),
    "UAV_Bat_Recordings2.zip": (837484894, "f7af4702288c7daeef9682e8ced92264"),
    "UAV_Bat_Recordings3.zip": (859379482, "4a2cced0700b678f5a6e651103a890ae"),
    "UAV_Bat_Recordings4.zip": (413070122, "2b608fd293958ea61d4d0cbd40ab014b"),
}

#: (zip, day folder) -> (WAV basenames in recording order, iOSD flight logs
#: with their uncompressed bytes). The deposit pairs logs with WAVs only by
#: day folder. FLY112 is absent from Drone_VA_0605.
CHIROCOPTER_DAYS: dict[tuple[str, str], tuple[tuple[str, ...], dict[str, int]]] = {
    ("UAV_Bat_Recordings1.zip", "Drone_VA_0524"): (
        ("Record_2017_May_24_050412.508.wav", "Record_2017_May_24_051823.299.wav"),
        {"FLY095.DAT": 101711872, "FLY096.DAT": 45744128},
    ),
    ("UAV_Bat_Recordings1.zip", "Drone_VA_0527"): (
        ("Record_2017_May_27_050031.280.wav", "Record_2017_May_27_051243.934.wav"),
        {"FLY097.DAT": 72548352, "FLY098.DAT": 41549824},
    ),
    ("UAV_Bat_Recordings2.zip", "Drone_VA_0529"): (
        ("Record_2017_May_29_050253.297.wav", "Record_2017_May_29_051419.878.wav"),
        {"FLY099.DAT": 76939264, "FLY100.DAT": 42106880},
    ),
    ("UAV_Bat_Recordings2.zip", "Drone_VA_0530"): (
        ("Record_2017_May_30_045846.040.wav", "Record_2017_May_30_051504.216.wav"),
        {"FLY101.DAT": 88670208, "FLY102.DAT": 45907968},
    ),
    ("UAV_Bat_Recordings3.zip", "Drone_VA_0531"): (
        ("Record_2017_May_31_045909.278.wav", "Record_2017_May_31_051034.760.wav"),
        {"FLY103.DAT": 84869120, "FLY104.DAT": 46563328},
    ),
    ("UAV_Bat_Recordings3.zip", "Drone_VA_0601"): (
        ("Record_2017_Jun_01_050144.089.wav", "Record_2017_Jun_01_051303.805.wav"),
        {"FLY106.DAT": 99581952, "FLY107.DAT": 49283072},
    ),
    ("UAV_Bat_Recordings4.zip", "Drone_VA_0605"): (
        ("Record_2017_Jun_05_042247.703.wav", "Record_2017_Jun_05_043922.901.wav"),
        {"FLY111.DAT": 49315840, "FLY113.DAT": 36700160},
    ),
}
#: Noise-test clip -> ZIP 'UX' (0x5855) mtime, UTC (1 s resolution). The clips
#: were written on 2017-10-22, months after the bat flights. Order = time order.
CHIROCOPTER_NOISE_TESTS = {
    "no_baffle": "2017-10-22T16:51:24Z",
    "no_baffle_1m_pole": "2017-10-22T16:52:11Z",
    "10cm_cone_1m_pole": "2017-10-22T16:52:48Z",
    "10cm_cup_1m_pole": "2017-10-22T16:53:21Z",
    "threeinch_ball_1m_pole": "2017-10-22T16:54:24Z",
    "fiveinch_ball_1m_pole": "2017-10-22T16:54:59Z",
    "sixinch_ball_1m_pole": "2017-10-22T16:55:26Z",
    "bakingdish_1m_pole": "2017-10-22T16:55:55Z",
}
_MONTHS = {"May": 5, "Jun": 6}

CHIROCOPTER_RIG = {
    "uav": "DJI Spreading Wings S900 hexacopter (6 rotors), DJI A2 controller",
    "microphone": "Dodotronic Ultramic-250K, omnidirectional, 250 kHz / 16 bit",
    "recorder": "Motorola MotoG with USB AudioRecorder PRO (Fu et al. 2018, Table 1)",
    "flight_logs": "every day folder holds two DJI iOSD Mark II binary flight logs "
    "(IOSD/FLY*.DAT). The original members remain in the pinned raw ZIPs.",
    "evidence": "Fu et al. 2018 MEE, doi:10.1111/2041-210X.12992 §§2–3, pp.1532–1533, "
    "Table 1 and Figs.1–3; Kloepper & Kinniry 2018 Sci. Rep. 8:7779 (PMC5958051) "
    "Methods; Dryad README and ZIP member names",
}
_CHIRO_FIELD_GEOMETRY = {
    **CHIROCOPTER_RIG,
    "mounting": "rigid 0.50 m fibreglass pole above the UAV with a 0.1524 m diameter "
    "Styrofoam sphere; the microphone is just above the sphere. Fu et al. §2 and "
    "Fig.1 show the assembled system; §3.2 reports the May–June flights, and "
    "Fig.3(b) shows the baffle in the thermal view during an actual flight.",
    "pole_length_m": 0.50,
    "baffle_diameter_m": 0.1524,
    "mic_pos_frame": "metres, origin at the pole base on the UAV and +z along the "
    "pole, upward in the nominal mounting. x=y=0 is the nominal pole line; "
    "its offset relative to the rotor plane, UAV centre or world is not measured.",
    "mic_pos_certainty": "approximate nominal pole-end position (~0.5 m), from the "
    "published 0.50 m pole length and Fig.1, where the capsule appears at its upper "
    "end just above the baffle. The sphere attachment and capsule-to-end offset "
    "are not dimensioned; the pole length alone does not give an exact height.",
}
_CHIRO_FIELD_MIC_POS = np.array([[0.0, 0.0, 0.5]])
CHIROCOPTER_FREQUENCY_NOTE = (
    "PCM16 at 250 kHz (Nyquist 125 kHz), mono; Fu et al. (2018) state the Ultramic "
    "'recorded full spectrum (250 kHz sampling rate)'. The manufacturer Ultramic User Guide "
    "(Dodotronic, rev. 3.4, 2016-04-07, p. 7) lists a MEMS sensor with a nominal upper "
    "range of 125 kHz for the 250K, but specifies no lower cutoff, response curve or "
    "in-situ calibration. The archived WAVs were not subjected to the stated 16 kHz "
    "call-analysis filter: Kloepper & Kinniry (2018, Sci. Rep. 8:7779, Methods and "
    "Fig. 1 caption) applied a 16 kHz eighth-order Butterworth high-pass to the "
    "extracted calls for analysis and figures only. Neither that nor the 250 kHz rate "
    "establishes a flat native response; any hardware roll-off is unknown. In "
    "Record_2017_Jun_05_042247.703 at 300 s (2 s Welch, 10 Hz bins) the 125 Hz-8 kHz "
    "octaves sit at -42 to -51 dBFS, level with 16-64 kHz (-55 dBFS), with BPF-like "
    "peaks near 210/380/570 Hz; that filter would put 4 kHz about 96 dB down. The "
    "released PCM levels are relative to full scale, not calibrated SPL. "
    "Welch estimates from 4 s excerpts at 60 s in each of the 14 released flight WAVs "
    "(uni-cpu Slurm 37562551, 7.63 Hz bin spacing) show 20 Hz-20 kHz power of "
    "-85 to -79 dBFS in first-in-day takes and -54 to -38 dBFS in second takes. "
    "The louder excerpts have strong narrow-band peaks around 170-590 Hz, with "
    "possible rotor components (not calibrated RPM). They are not uniformly audible-band "
    "quiet; the quieter excerpts are often dominated by energy above 20 kHz. "
    "The recorder's effective band is narrower than Nyquist: in flight and noise-test "
    "files alike (2,500-point Welch, third octaves) levels reach a flat analog floor of "
    "about -130 dB re FS^2/Hz above 65-80 kHz (a 30 dB fall from 50 to 80 kHz in the "
    "louder files), drop between 10 and 13 kHz, and carry the same +0.0084 FS DC "
    "offset. The eight noise tests are genuine 250 kHz Ultramic captures "
    "of 12,500 samples (0.05 s) each, not mislabelled lower-rate audio; they have "
    "20 Hz-20 kHz power of -57 to -40 dBFS and include sub-kHz tones. "
    "A single excerpt does not establish consistent flight phase or gain, and these "
    "recording-level measurements are not the mic response."
)


def _chiro_start(wav: str) -> tuple[str, str]:
    """``(local, utc)`` ISO start times from a recorder file name.

    ``Record_2017_May_24_050412.508.wav`` is the recorder PC's local start time
    in UTC-6 (MDT). Evidence: for all 14 WAVs, the ZIP's Info-ZIP 'UX' (0x5855)
    mtime, which is UTC, equals name start + header duration + 6 h within
    +-1.02 s (the mtime has 1 s resolution).
    """
    import datetime as dt

    _, year, mon, day, hms = Path(wav).stem.split("_")
    hh, mm, ss = hms[0:2], hms[2:4], hms[4:]
    local = f"{year}-{_MONTHS[mon]:02d}-{int(day):02d}T{hh}:{mm}:{ss}"
    utc = dt.datetime.fromisoformat(local) + dt.timedelta(hours=6)
    return local + "-06:00", utc.isoformat(timespec="milliseconds") + "Z"


_IOSD_CLOCK_NOTE = {
    "osd": "0x1002; time from the unwrapped OSD sample counter and GPS rate fit",
    "gps": "0x1003; original calendar fields retained (date YYYYMMDD, time HHMMSS); "
    "subsecond timing interpolated by CAN record order between adjacent OSD packets",
    "controller": "0x4a44/0xa249; native six-channel motor output codes, voltages and "
    "original payload; timing interpolated by CAN record order between adjacent OSD packets",
    "unstamped": "rows with no calibrated clock or outside the bracketing OSD packets "
    "retain CAN record ordinals but have no timestamp; osd_unplaced also retains "
    "CRC-valid OSD packets with duplicate/backward counters without inventing a sample index",
    "units": {
        "lon_rad/lat_rad": "radians",
        "lon_e7/lat_e7": "1e-7 degrees",
        "gps_height_m/baro_height_m": "metres (native GPS and barometric references)",
        "height_mm": "millimetres (native GPS reference)",
        "accel_g": "g, native axis order",
        "gyro_rad_s": "rad/s, native axis order",
        "velocity_ned_m_s": "m/s, North/East/Down",
        "velocity_ned_cm_s": "cm/s, North/East/Down",
        "quaternion": "unitless, native component order",
        "magnetometer_raw": "native counts",
        "sample": "unwrapped OSD packet count since the first decoded OSD packet",
        "record_index": "CAN record ordinal among retained slots, including the sparse prelude",
        "motor_output_code": "native command codes, controller channel order 1–6; "
        "physical rotor assignment and conversion to RPM or PWM are unknown",
        "voltage_mv": "millivolts, battery/CAN/BEC order",
    },
}


@dataclass(frozen=True)
class _IosdTrack:
    rows: np.ndarray
    # Timestamps in ticks since the first OSD packet, for the contiguous part
    # of rows bracketed by OSD packets. Missing stamps are not extrapolated.
    ticks: np.ndarray
    first_timed: int

    def window(self, start: int, end: int, offset: int) -> td.Series | None:
        lo, hi = self.ticks.searchsorted([start - offset, end - offset])
        if lo == hi:
            return None
        return td.events(
            self.ticks[lo:hi] + offset,
            self.rows[self.first_timed + lo : self.first_timed + hi],
            t_start=start,
            t_end=end,
            dims=("time",),
        )

    def untimed(self) -> Iterator[tuple[str, np.ndarray]]:
        """Rows without a calibrated stamp, labelled by their place in the log."""
        if not len(self.ticks):
            if len(self.rows):
                yield "all", self.rows
            return
        if self.first_timed:
            yield "before_first_osd", self.rows[: self.first_timed]
        tail = self.rows[self.first_timed + len(self.ticks) :]
        if len(tail):
            yield "after_last_osd", tail


def _iosd_tracks(log: IosdTelemetry, seconds_per_sample: float | None) -> dict[str, _IosdTrack]:
    groups = {"osd": log.osd, "gps": log.gps, "controller": log.controller}
    if len(log.osd_unplaced) and seconds_per_sample is None:
        groups["osd_unplaced"] = log.osd_unplaced
    if seconds_per_sample is None:
        return {name: _IosdTrack(rows, np.empty(0, np.int64), 0) for name, rows in groups.items()}
    scale = seconds_per_sample * td.TICKS_PER_SECOND
    index, sample = log.osd["record_index"], log.osd["sample"]
    tracks = {}
    for name, rows in groups.items():
        if name == "osd":
            lo, ordinal = 0, sample
        else:
            packet_index = rows["record_index"]
            lo = int(packet_index.searchsorted(index[0]))
            hi = int(packet_index.searchsorted(index[-1], side="right"))
            ordinal = np.interp(packet_index[lo:hi], index, sample)
        tracks[name] = _IosdTrack(rows, np.rint(ordinal * scale).astype(np.int64), lo)
    if len(log.osd_unplaced):
        tracks["osd_unplaced"] = _IosdTrack(log.osd_unplaced, np.empty(0, np.int64), 0)
    return tracks


def _iosd_window(
    tracks: dict[str, _IosdTrack], start: int, end: int, offset: int = 0
) -> dict[str, td.Series]:
    return {
        name: window
        for name, track in tracks.items()
        if (window := track.window(start, end, offset)) is not None
    }


def _read_chiro_tracks(
    root: Path, zname: str, member: str, clock: dict[str, Any]
) -> tuple[dict[str, _IosdTrack], dict[str, int]]:
    from data_processing.sources.iosd_dat import read_iosd_dat

    with zipfile.ZipFile(root / zname) as zf, zf.open(member) as f:
        log = read_iosd_dat(f)
    return _iosd_tracks(log, clock["seconds_per_sample"]), {
        "irregular_blocks": log.irregular_blocks,
        "unframed_bytes": log.unframed_bytes,
        "osd_unplaced_rows": len(log.osd_unplaced),
    }


def _chiro_log_frame(
    tracks: dict[str, _IosdTrack],
    clock: dict[str, Any],
    recovery: dict[str, int],
    audio_bounds: tuple[int, int] | None,
    member: str,
    nbytes: int,
    zname: str,
    day: str,
    wavs: tuple[str, ...],
    system: dict[str, Any],
) -> tuple[str, td.Frame] | None:
    """One log-only Frame per DAT: every row not carried by its audio Frame.

    ``audio_bounds`` is the audio Frame's validated span in log ticks (seconds
    since the first OSD packet). Stamped rows before/after it keep that native
    log clock as separate sections; unstamped rows keep CAN record order only.
    ``None`` when the audio Frame already holds every row.
    """
    native_end = max((int(t.ticks[-1]) + 1 for t in tracks.values() if len(t.ticks)), default=0)
    if audio_bounds is None:
        spans = [("native_log", 0, native_end)]
    else:
        start, end = audio_bounds
        spans = [
            ("before_verified_overlap", 0, min(start, native_end)),
            ("after_verified_overlap", max(end, 0), native_end),
        ]
    sections: dict[str, td.Frame] = {}
    native_spans: dict[str, list[float]] = {}
    for section, start, end in spans:
        if start < end and (entries := _iosd_window(tracks, start, end)):
            sections[section] = td.Frame(entries)
            native_spans[section] = [start / td.TICKS_PER_SECOND, end / td.TICKS_PER_SECOND]
    unstamped = {
        name if where == "all" else f"{name}_{where}": td.wrap(rows)
        for name, track in tracks.items()
        for where, rows in track.untimed()
    }
    if unstamped:
        sections["unstamped"] = td.Frame(unstamped)
    if not sections:
        return None
    stem = Path(member).stem
    rid = f"{day}/{stem}"
    wav = clock["recording"]
    meta = meta_frame(
        rid,
        CHIROCOPTER,
        system=system,
        observation={
            "mic_placement": "onboard",
            "type": "uav_flight_telemetry",
            "detail": "DJI iOSD flight-controller telemetry rows outside the associated audio; no audio",
        },
        operating={"condition": "unknown"},
        label={"subcategory": "iosd_native_log"},
        extra={
            "source_kind": "original_telemetry_log",
            "doi": "10.5061/dryad.cf472n3",
            "day_folder": day,
            "original_container": zname,
            "original_member": member,
            "original_bytes": nbytes,
            "iosd_sections": list(sections),
            "iosd_native_spans_s": native_spans,
            "associated_audio": None if wav is None else f"{day}/{Path(wav).stem}",
            "rows_in_audio_frame": "stamped rows inside the validated audio interval ride on the "
            "associated audio Frame (audio clock); every other row is here, exactly once"
            if audio_bounds is not None
            else None,
            "same_day_audio": [f"{day}/{Path(w).stem}" for w in wavs],
            "flight_clock": clock,
            "iosd_decode_recovery": recovery,
            "iosd_clock": "stamped sections: seconds since the first OSD packet, not audio "
            "seconds; 'unstamped': CAN record order only",
            "iosd_fields": _IOSD_CLOCK_NOTE,
        },
    )
    return safe_key(rid), td.Frame({"iosd": td.Frame(sections), "meta": meta})


def build_chirocopter(root: Path) -> Iterator[tuple[str, td.Frame]]:
    import json

    root = Path(root)
    _verify(root, CHIROCOPTER_FILES, CHIROCOPTER)
    clock_bytes = Path(__file__).with_name("chirocopter_clocks.json").read_bytes()
    if hashlib.sha256(clock_bytes).hexdigest() != CHIRO_CLOCK_SHA256:
        raise ValueError("Chirocopter clock table SHA-256 does not match the pinned calibration")
    clocks = json.loads(clock_bytes)
    expected = {Path(name).stem for _, files in CHIROCOPTER_DAYS.values() for name in files}
    if set(clocks) != expected:
        raise ValueError("Chirocopter clock table does not cover the pinned iOSD members")
    system = {
        "category": "multirotor_uav",
        "platform": "DJI Spreading Wings S900",
        "n_rotors": 6,
        "rig": "chirocopter_s900_ultramic250",
    }
    for (zname, day), (wavs, logs) in CHIROCOPTER_DAYS.items():
        top = zname.removesuffix(".zip")
        day_clocks = {name: clocks[Path(name).stem] for name in logs}
        pairs = {
            clock["recording"]: name
            for name, clock in day_clocks.items()
            if clock["recording"] is not None
        }
        if not set(pairs).issubset(wavs) or len(pairs) != sum(
            clock["recording"] is not None for clock in day_clocks.values()
        ):
            raise ValueError(f"{day}: invalid Chirocopter WAV to DAT pairing")
        for order, wav in enumerate(wavs, start=1):
            member = f"{top}/{day}/{wav}"
            start_local, start_utc = _chiro_start(wav)
            log_name = pairs.get(wav)
            log_member = clock = tracks = recovery = None
            # (name, member, clock, tracks, recovery) of a log with an audio-clock map
            aligned = None
            if log_name is not None:
                log_member = f"{top}/{day}/IOSD/{log_name}"
                clock = day_clocks[log_name]
                if clock["audio_offset_s"] is not None:
                    if clock["seconds_per_sample"] is None:
                        raise ValueError(f"{day}/{wav}: aligned iOSD log has no calibrated clock")
                    tracks, recovery = _read_chiro_tracks(root, zname, log_member, clock)
                    aligned = (log_name, log_member, clock, tracks, recovery)
            key, frame = _recording(
                root,
                zname,
                member,
                f"{day}/{Path(wav).stem}",
                {
                    "dataset": CHIROCOPTER,
                    "system": system,
                    "observation": {
                        "mic_placement": "onboard",
                        "type": "uav_mounted_ultrasonic_microphone",
                        "detail": "Ultramic-250K above the S900 on a rigid pole, above the Styrofoam baffle",
                        "environment": "outdoors above a natural cave entrance, Sierra County, New Mexico, USA",
                        "source_motion": "uav_ladder_climb",
                    },
                    "operating": {
                        "condition": "flight",
                        "flight_mode": "mixed",
                        "protocol": "take off, hover 1 min each at 5, 10, 20, 30, 40, 50 m above "
                        "the cave edge, land, change battery (Sci. Rep. 2018 Methods); segment "
                        "times are not published",
                        "hover_ladder_m": [5, 10, 20, 30, 40, 50],
                        "contains_rotor_noise": True,
                        "external_source": "other",
                        "external_source_detail": "Tadarida brasiliensis echolocation during morning "
                        "roost re-entry (unannotated; the authors extracted 3,847 calls from 84 min)",
                        "rotor_telemetry": "iOSD includes six native motor output codes; "
                        "no per-rotor speed calibration or physical rotor assignment",
                    },
                    "label": {
                        "subcategory": "flight_hover_ladder_bats",
                        "rig": "S900 + Ultramic-250",
                        "health": "unknown",
                        "mixed": "acoustic (wild bats + rotor noise)",
                        "species": "Tadarida brasiliensis",
                    },
                    "extra": {
                        "source_kind": "original_recording",
                        "doi": "10.5061/dryad.cf472n3",
                        "day_folder": day,
                        "recording_in_day": order,
                        "recording_start_local": start_local,
                        "recording_start_recorder_utc": start_utc,
                        "clock": "audio t=0 is the recorder PC start in the file name. "
                        "The inferred UTC-6 (MDT) offset accounts for the ZIP Info-ZIP UX "
                        "mtime minus WAV start and duration across all 14 files (residual "
                        "at most 1.02 s). These timestamps may come from the same PC; the "
                        "evidence establishes timezone, not accuracy; this is the recorder "
                        "clock expressed in UTC, not a GPS-disciplined absolute start.",
                        "flight_logs": [f"{top}/{day}/IOSD/{name}" for name in logs],
                        "flight_log_bytes": dict(logs),
                        "associated_flight_log": log_member,
                        "associated_flight_log_id": None
                        if log_name is None
                        else f"{day}/{Path(log_name).stem}",
                        "flight_clock": clock,
                        "iosd_decode_recovery": recovery,
                        "flight_log_alignment": "audio-clock map within the validated interval only"
                        if tracks is not None
                        else "paired, separate clocks"
                        if log_member is not None
                        else "day association only; log-to-WAV pairing unverified",
                        "iosd_clock": "audio seconds where present" if tracks is not None else None,
                        "iosd_fields": _IOSD_CLOCK_NOTE if tracks is not None else None,
                        "frequency_note": CHIROCOPTER_FREQUENCY_NOTE,
                        "geometry": _CHIRO_FIELD_GEOMETRY,
                    },
                },
                mic_pos=_CHIRO_FIELD_MIC_POS,
                expect=(250000, 1),
            )
            if aligned is None:
                yield key, frame
                continue
            log_name, log_member, clock, tracks, recovery = aligned
            offset = round(clock["audio_offset_s"] * td.TICKS_PER_SECOND)
            valid_start, valid_end = (
                round(t * td.TICKS_PER_SECOND) for t in clock["validated_audio_interval_s"]
            )
            valid_end = min(valid_end, frame.t_end_ticks)
            if not 0 <= valid_start < valid_end:
                raise ValueError(f"{day}/{wav}: invalid shared-clock interval")
            entries = _iosd_window(tracks, valid_start, valid_end, offset)
            yield (
                key,
                frame.with_entry("iosd", td.Frame(entries), expand=False) if entries else frame,
            )
            del frame
            log_frame = _chiro_log_frame(
                tracks,
                clock,
                recovery,
                (valid_start - offset, valid_end - offset),
                log_member,
                logs[log_name],
                zname,
                day,
                wavs,
                system,
            )
            if log_frame is not None:
                yield log_frame
        for log_name, nbytes in logs.items():
            clock = day_clocks[log_name]
            if clock["audio_offset_s"] is not None:
                if clock["recording"] is None:
                    raise ValueError(f"{log_name}: iOSD audio-clock map without associated WAV")
                continue
            log_member = f"{top}/{day}/IOSD/{log_name}"
            tracks, recovery = _read_chiro_tracks(root, zname, log_member, clock)
            log_frame = _chiro_log_frame(
                tracks, clock, recovery, None, log_member, nbytes, zname, day, wavs, system
            )
            if log_frame is not None:
                yield log_frame
    for test in CHIROCOPTER_NOISE_TESTS:
        pole = "1m_pole" in test
        yield _recording(
            root,
            "UAV_Noise_Testing.zip",
            f"UAV_Noise_Testing/{test}.wav",
            f"noise_test/{test}",
            {
                "dataset": CHIROCOPTER,
                "system": system,
                "observation": {
                    "mic_placement": "onboard",
                    "type": "uav_mounted_ultrasonic_microphone",
                    "detail": f"noise test: baffle '{test.removesuffix('_1m_pole')}'"
                    + (" on a 1 m pole" if pole else " (no 1 m pole indicated)")
                    + " (from the file name)",
                    "source_motion": "unknown",
                },
                "operating": {
                    "condition": "unknown",
                    "condition_detail": "Dryad: 'Noise testing files of UAV'; rotor state, "
                    "throttle and whether airborne are not documented in the deposit",
                    "contains_rotor_noise": True,
                    "external_source": "unknown",
                },
                "label": {
                    "subcategory": f"noise_test_{test}",
                    "rig": "S900 + Ultramic-250",
                    "health": "unknown",
                    "mixed": "none",
                    "baffle": test.removesuffix("_1m_pole"),
                    "pole_1m": pole,
                },
                "extra": {
                    "source_kind": "original_recording",
                    "doi": "10.5061/dryad.cf472n3",
                    "clock": "complete published clip of 12,500 samples at 250 kHz (0.05 s); "
                    "no start timestamp. File mtime (ZIP UX extra, UTC) "
                    f"{CHIROCOPTER_NOISE_TESTS[test]}, 22 Oct 2017, not a bat-survey day. The "
                    "eight clips were written 27-63 s apart; that they are excerpts saved from "
                    "longer takes (cf. Fu et al. 2018 Fig. 2) is an inference, not documented",
                    "file_mtime_utc": CHIROCOPTER_NOISE_TESTS[test],
                    "frequency_note": CHIROCOPTER_FREQUENCY_NOTE,
                    "geometry": {
                        **CHIROCOPTER_RIG,
                        "mounting": "Oct 2017 UAV-noise test configurations differ "
                        "from the May–June field rig. Only the baffle identity and nominal "
                        "1 m pole length (where named) are given in the ZIP names/README. "
                        "Fu et al. Fig.2 describes the microphone pointing up for its "
                        "tests, but the capsule XYZ for each released clip is not surveyed.",
                        "pole_length_m": 1.0 if pole else None,
                        "mic_pos_certainty": "no coordinates inferred for the noise tests",
                    },
                },
            },
            expect=(250000, 1),
        )


CHIROCOPTER_PROVENANCE = {
    "source_url": "https://datadryad.org/dataset/doi:10.5061/dryad.cf472n3",
    "mirror_url": "https://zenodo.org/records/4993744",
    "doi": "10.5061/dryad.cf472n3",
    "version": "Dryad version 1 (2019-02-22); fetched from Zenodo 4993744, byte-identical (md5)",
    "license": "CC0 1.0",
    "rights_note": "Dryad data are CC0 (public-domain dedication); the Zenodo mirror is also "
    "CC0. The MEE article itself is CC BY-NC, which covers only the text, not the data. "
    "Cite as a scholarly courtesy.",
    "citation": "Fu Y., Kinniry M., Kloepper L. N. (2018). The Chirocopter: a UAV for recording "
    "sound and video of bats at altitude. Methods Ecol. Evol. 9(6):1531-1535, "
    "doi:10.1111/2041-210X.12992; Kloepper L. N., Kinniry M. (2018). Recording animal "
    "vocalizations from a UAV: bat echolocation during roost re-entry. Sci. Rep. 8:7779, "
    "doi:10.1038/s41598-018-26122-z; data: Dryad doi:10.5061/dryad.cf472n3.",
    "collection_method": "UAV-mounted ultrasonic microphone climbing a 5-50 m hover ladder "
    "over a bat cave at dawn (14 battery flights, 24 May-5 Jun 2017), plus 0.05 s noise "
    "tests of baffle/pole configurations",
    "equipment": "DJI S900 hexacopter; Dodotronics Ultramic-250 (250 kHz, 16-bit); DJI iOSD Mark II",
    "observation_type": "uav_mounted_microphone",
    "sample_rate": 250000,
    "channels": 1,
    "release_bytes": 2958859251,  # pinned files; the full record (5 identical READMEs) is 2958862475
    "description": "14 mono 250 kHz flight recordings (462.1-673.7 s each, 2.16 h total), "
    "one whole Frame each with the iOSD rows that fall inside its validated audio-clock "
    "interval, plus one log-only Frame per DAT holding every other decoded row once "
    "(native-log-clock sections before/after the audio and CAN-order unstamped rows), "
    "plus 8 noise-test clips (12,500 samples = 0.05 s each, as published). CRC-checked "
    "iOSD inputs supply IMU, pose, GPS and controller series. Original DAT logs and "
    "READMEs remain in the pinned raw archives.",
    "related_studies": "The same Dryad deposit backs both papers (the dates and the 84 min "
    "of hovering match).",
    "geometry": _CHIRO_FIELD_GEOMETRY,
}


# ═══ SDU drone-borne bat array (Zenodo 7329733) ══════════════════════════════

SDU_BAT_ARRAY = "SDU-DroneBatArray"
#: Zenodo 7329733 file key -> (bytes, md5); single published version.
SDU_FILES = {
    "DroneNoise1meter.WAV": (7865314, "9ef4719065f1a5fcb973d4ee604204e2"),
    "DroneNoise10meter.wav": (30000044, "837201f716beb629de2d8fa7603a1788"),
    "DroneNoise10meter_includingPinger.wav": (30000044, "6695f65bfcb785c72ca487009e409b19"),
    "DroneNoise30meter.wav": (30000044, "12a9bbffa61a64603d022c46a7b77c37"),
    "Noisefloor.wav": (15000044, "0fb217d72252ff604a24495e163ac9d7"),
    "Noctule.wav": (48000044, "528c6e24f6afc3043f688b8dfbfed154"),
    "Serotine.wav": (36000044, "d69102556e332fb2957223acfb26b914"),
    "Daubentone.wav": (24000044, "d6468dd526bfd9e3cfc48300a2107151"),
    "Pipistrel.wav": (24000044, "02d3939eff90b580bf86b70b07a01ef4"),
}
#: file -> (kind, tether_m, foam_pad_on_sonar, species)
SDU_RECORDINGS: dict[str, tuple[str, float | None, bool | None, str | None]] = {
    "DroneNoise1meter.WAV": ("noise", 1.0, True, None),
    "DroneNoise10meter.wav": ("noise", 10.0, True, None),
    "DroneNoise10meter_includingPinger.wav": ("noise", 10.0, False, None),
    "DroneNoise30meter.wav": ("noise", 30.0, True, None),
    "Noisefloor.wav": ("floor", None, None, None),
    "Noctule.wav": ("bats", 30.0, True, "Nyctalus noctula"),
    "Serotine.wav": ("bats", 30.0, True, "Eptesicus serotinus"),
    "Daubentone.wav": ("bats", 30.0, True, "Myotis daubentonii"),
    "Pipistrel.wav": ("bats", 30.0, True, "Pipistrellus pygmaeus"),
}
#: Mic heights on the 4 m vertical line array: metres from the array centre
#: (the A/D board), z up. Mics are 1.30 m apart, symmetric about the board.
SDU_MIC_Z_TOP_DOWN = (1.95, 0.65, -0.65, -1.95)
#: Array half-length: the tether ties to the top of the two 2 m tubes.
SDU_ARRAY_HALF_LENGTH_M = 2.0


def sdu_mic_pos(channel_rows: tuple[int, ...]) -> np.ndarray:
    """``(4, 3)`` mic positions in channel order. ``channel_rows[c]`` is the
    top-down position index of WAV channel ``c``."""
    return np.array([[0.0, 0.0, SDU_MIC_Z_TOP_DOWN[r]] for r in channel_rows])


#: WAV channel c -> top-down position index. Not documented; inferred from
#: data (see SDU_GEOMETRY["channel_order_evidence"]).
SDU_CHANNEL_ROWS = (0, 1, 2, 3)
SDU_MIC_POS = sdu_mic_pos(SDU_CHANNEL_ROWS)
#: Published single-channel (WAV header); every other SDU file is 4-channel.
SDU_MONO_FILES = frozenset({"DroneNoise1meter.WAV"})
SDU_GEOMETRY = {
    "array": "4 Knowles FG-series mics on two linked 2 m carbon tubes (4 m), A/D board "
    "in the middle, mics 1.30 m apart; the array hangs vertically from a nylon tether "
    "tied to its top (ece3.9577 section 2.1, Fig. 1; the vertical orientation follows "
    "from 'tether between the UAV and the top of the array' and the up/down vs side-"
    "to-side scanning argument in sections 3-4)",
    "mic_pos_frame": "metres, origin at the array centre (A/D board), z up along the "
    "array, x/y = 0 (nominal: assumes a straight, vertical array)",
    "mic_pos_certainty": "spacing nominal (paper: 1.30 m; symmetric about the board "
    "assumed). The authors do not document channel order; it was measured from the "
    "data (channel_order_evidence)",
    "channel_order_evidence": "high: band-limited GCC-PHAT (2-9, 9-20, 20-60 kHz; three "
    "0.5 s windows each; lag 0 masked against common-mode pickup), uni-cpu Slurm "
    "37560801. In DroneNoise10meter, DroneNoise30meter and 10meter_includingPinger, the "
    "overhead UAV's noise reaches ch0, ch1, ch2, ch3 in that order in every band and "
    "window, with lags behind ch0 of 3.54/8.00-8.02/11.65-11.73 ms. The Pipistrel, "
    "Serotine and Daubentone excerpts agree (3.61-3.66/7.93-8.01/11.53-11.62 ms). So "
    "ch0 is the top mic and ch3 the bottom",
    "measured_spacing": "the same end-fire lags at c=343 m/s give steps of ≈1.21-1.24, "
    "1.49-1.53 and 1.25-1.28 m (top to bottom), ≈4.0 m in total, against the paper's "
    "uniform 1.30 m (3.90 m). Plausibly the middle gap spans the A/D box, but c and "
    "tether tilt are not measured. mic_pos keeps the paper's nominal spacing",
    "evidence": "Jespersen et al. 2022 Ecol. Evol. 12:e9577 (PMC9719081) sections 2.1-2.4, "
    "3.1-3.2; Zenodo 7329733 description",
}


SDU_FREQUENCY_NOTE = (
    "PCM16; 375 kHz (noise tests, noise floor) or 300 kHz (bat excerpts) as published. "
    "Hardware (ece3.9577 section 2.1): 20 dB preamp + AD8336 PGA per mic, a 151 kHz "
    "2nd-order anti-alias filter, and EITHER a 16 kHz 1st-order OR a 9 kHz 2nd-order "
    "high-pass. The paper does not assign the option per file, and the spectra below "
    "do not identify it. The authors calibrated against a GRAS 40BF with a 140-5 kHz "
    "sweep and filtered 'the recordings' with the derived impulse response; the "
    "deposit does not say which files that applies to. "
    "Measured (4 s Welch per channel, 23 Hz bins, third octaves; Hetzner probe "
    "2026-11-26): the four-channel 375 kHz files keep a +0.034 to +0.044 FS DC offset "
    "per channel and an analog floor of about -123 to -129 dB re FS^2/Hz below 1 kHz "
    "(16-bit quantization would be -154). Relative to the unfiltered-looking mono "
    "DroneNoise1meter, the 10 m noise rises about 10-12 dB per octave from 0.5 to 8 kHz, "
    "a high-pass-like shape, but a common recorder chain and gain have not been "
    "established (see mono_file); this is not a transfer-function measurement. The 300 kHz bat "
    "excerpts differ in kind: DC below 1e-4 FS, every band below about 5 kHz at "
    "exactly the 16-bit quantization floor (-153 dB re FS^2/Hz), about 30 dB per octave "
    "rise from 5 to 10 kHz, and a steep fall from 100 kHz to that floor by about 140 kHz. "
    "This suggests post-acquisition processing consistent with the calibration filter; "
    "the per-file processing is undocumented. No rotor signal is resolved below about "
    "5 kHz in those excerpts; this does not establish its absence up to 20 kHz. "
    "The noise floor file shows a broad "
    "12-20 kHz hump of about 8 dB above its low-frequency floor. DroneNoise1meter "
    "reaches full scale (|x| >= 32767 on 5.8e-5 of samples over the file). Earlier 2 s "
    "probes (uni-cpu Slurm 37560801): UAV noise exceeds the system floor by about 30 dB "
    "at 2-10 kHz and 20 dB "
    "at 10-16 kHz at 10 m, and by about 20 and 8 dB at 30 m; the 40 kHz sonar pinger "
    "is detectable in the unshielded 'includingPinger' file (+15 dB above the system "
    "floor in the 30-50 kHz band)."
)


def build_sdu_bat_array(root: Path) -> Iterator[tuple[str, td.Frame]]:
    root = Path(root)
    _verify(root, SDU_FILES, SDU_BAT_ARRAY)
    for fname, (kind, tether, foam, species) in SDU_RECORDINGS.items():
        stem = Path(fname).stem
        geometry: dict[str, str | list[float]] = dict(SDU_GEOMETRY)
        mono = fname in SDU_MONO_FILES
        if mono:
            geometry["mono_file"] = (
                "this file is single-channel as published (header: 1 ch, 375 kHz, "
                "3,932,160 frames = 10.49 s; a lag-1..8 autocorrelation probe shows no "
                "4-channel interleave). Unlike the array WAVs it carries trailing TIME "
                "and BWF bext chunks naming Avisoft-RECORDER (session clock 12/13/80, not a "
                "real date), DC near 0 (0.0012 FS) instead of +0.04, full low-frequency "
                "content and a roll-off from about 30 kHz. It is therefore probably not "
                "the RPi/LTC2205 array chain the paper describes [inference]; the paper's "
                "Methods name no other recorder. Which microphone it holds is not stated, "
                "so no mic_pos is attached"
            )
        if tether is not None:
            # UAV relative to the array centre: tether + top half of the array.
            geometry["uav_position_m"] = [0.0, 0.0, tether + SDU_ARRAY_HALF_LENGTH_M]
            geometry["uav_position_certainty"] = (
                "nominal: tether length + 2 m half-array, assumes a vertical, taut "
                "tether (sway and sag unknown)"
            )
        if kind == "noise":
            if tether is None:
                raise ValueError(f"{fname}: drone-noise test without a tether length")
            subcat = f"noise_test_tether_{int(tether)}m" + ("" if foam else "_sonar_unshielded")
            placement, condition = "suspended", "flight"
            detail = (
                f"line array hung {tether:g} m below the hovering M210 V2, sonar foam pad "
                + ("on" if foam else "off (pinger audible)")
                + "; open field, University of Southern Denmark, Odense"
            )
            external = {"external_source": "none"}
            mixed = "none"
        elif kind == "floor":
            subcat = "system_noise_floor"
            placement, condition = "other", "rotors_off"
            detail = "microphones embedded in foam, recording-system self-noise (no drone)"
            external = {"external_source": "none"}
            mixed = "none"
        else:
            subcat = "field_bats_tether_30m"
            placement, condition = "suspended", "flight"
            detail = (
                "line array 30 m below the hovering M210 V2 over a foraging site near "
                "Odense, Jun-Sep 2021; deposit excerpt of a 10 min field recording"
            )
            external = {
                "external_source": "other",
                "external_source_detail": f"{species} echolocation incl. feeding buzzes (unannotated)",
            }
            mixed = (
                "acoustic (wild bats over a hovering drone); spectrum suggests processing, "
                "with no resolved signal above the quantization floor below about 5 kHz"
            )
        excerpt_note = {
            "noise": "complete published noise-test file; the paper recorded 10 s noise "
            "segments at 375 kHz (the 10/30 m files hold exactly 3,750,000 samples)",
            "floor": "complete published 5 s noise-floor file",
            "bats": "published 10-20 s excerpt of a 10 min field recording; its position "
            "in that recording is not given",
        }[kind]
        yield _recording(
            root,
            fname,
            None,
            stem,
            {
                "dataset": SDU_BAT_ARRAY,
                "system": {
                    "category": "multirotor_uav",
                    "platform": "DJI Matrice 210 V2",
                    "n_rotors": 4,
                    "rig": "sdu_m210v2_tethered_4mic_line",
                },
                "observation": {
                    "mic_placement": placement,
                    "type": "uav_suspended_line_array"
                    if placement == "suspended"
                    else "system_self_noise",
                    "detail": detail,
                    "source_motion": "hovering" if condition == "flight" else "none",
                },
                "operating": {
                    "condition": condition,
                    "flight_mode": "hover" if condition == "flight" else None,
                    "tether_m": tether,
                    "sonar_foam_pad": foam,
                    "contains_rotor_noise": condition == "flight",
                    "rotor_telemetry": "none published",
                    **external,
                },
                "label": {
                    "subcategory": subcat,
                    "rig": "M210 V2 + 4-mic 4 m line array (Knowles FG)",
                    "health": "unknown",
                    "mixed": mixed,
                    "species": species,
                },
                "extra": {
                    "source_kind": "published_field_excerpt"
                    if kind == "bats"
                    else "original_recording",
                    "doi": "10.5281/zenodo.7329733",
                    "paper_doi": "10.1002/ece3.9577",
                    "excerpt_note": excerpt_note,
                    "clock": "audio sample clock only; no timestamps or telemetry published",
                    "frequency_note": SDU_FREQUENCY_NOTE,
                    "geometry": geometry,
                },
            },
            mic_pos=None if mono else SDU_MIC_POS,
            expect=(300000 if kind == "bats" else 375000, 1 if mono else 4),
        )


SDU_PROVENANCE = {
    "source_url": "https://zenodo.org/records/7329733",
    "doi": "10.5281/zenodo.7329733",
    "version": "Zenodo record 7329733 (concept 7329732), published 2022-11-17, single version",
    "license": "CC BY 4.0",
    "rights_note": "Deposit licence CC BY 4.0 (Zenodo metadata); the paper is also CC BY "
    "4.0. Attribution required. Wild bat recordings, no third-party content.",
    "citation": "Jespersen C., Docherty D., Hallam J., Albertsen C., Jakobsen L. (2022). Drone "
    "exploration of bat echolocation: a UAV-borne multimicrophone array to study bat "
    "echolocation. Ecol. Evol. 12:e9577, doi:10.1002/ece3.9577; data: Zenodo "
    "doi:10.5281/zenodo.7329733.",
    "collection_method": "4-mic vertical line array hung 1/10/30 m below a hovering DJI M210 "
    "V2 (noise tests, open field), the system noise floor, and 30 m tether field "
    "recordings of four Danish bat species",
    "equipment": "DJI M210 V2; custom RPi recorder, LTC2205 ADC, 4 Knowles FG mics on a 4 m carbon line",
    "observation_type": "uav_suspended_line_array",
    "sample_rates": [375000, 300000],
    "channels": [4, 1],  # 4 everywhere except DroneNoise1meter.WAV (mono as published)
    "release_bytes": 244865666,
    "description": "9 WAVs, one whole Frame each (5-20 s): noise at 1 m (mono, 10.49 s), "
    "10 m (sonar padded and unpadded) and 30 m tethers (10 s each), system noise floor "
    "(5 s), and digitally processed Noctule/Serotine/Daubenton's/Pipistrelle excerpts "
    "(20/15/10/10 s at 300 kHz).",
    "geometry": SDU_GEOMETRY,
}
