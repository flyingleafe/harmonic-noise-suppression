"""D3SET — IntRoLab drone package-delivery detection dataset (onboard + ground array).

Marcoux & Grondin, "Acoustic Drone Package Delivery Detection",
arXiv:2602.09991 (2026). Data: https://github.com/introlab/d3set (GPL-3.0),
mirrored at https://www.kaggle.com/datasets/francoismarcoux1212/d3set-introlab.

Raw identity: git commit :data:`GIT_COMMIT` of ``introlab/d3set``; every file
is a Git-LFS object whose sha256 (the LFS oid) and size are pinned in
``d3set_files.tsv`` beside this module. :func:`download_d3set` fetches each
object from GitHub's LFS media endpoint at that commit and verifies it; the
Kaggle v1 mirror re-encodes the ``26 mai vol 1-3`` CSV/JSON (CRLF), so it is
not the pin.

Rig (paper §III-A, Fig. 4): a modified Holybro X500 V2 quadrotor with a servo
drop mechanism and an onboard Raspberry Pi logging MAVSDK telemetry and one
microphone on a landing-gear leg (``drone.wav``, captured at 48 kHz and
resampled to 16 kHz by the authors); a ground 16-MEMS-microphone array on a
16SoundsUSB card recorded by a separate laptop (``matrix.wav``, 16 kHz). The
authors aligned the array to ``drone.wav`` by hand claps at the start of each
flight and padded it to the same length; ``StartTimestamp.csv`` is the Unix ms
of the first ``drone.wav`` sample on the RPi clock, and every telemetry CSV row
carries the RPi *receive* time in Unix ms (README).

One Frame per release folder, whole recordings (no builder chunking), keyed by
the folder path:

- ``audio``: ``drone.wav`` (mono ``(time,)``), plus ``mic_pos``;
  ``array_audio``: ``matrix.wav`` ``(array_mic, time)`` on the same time axis
  where the published clap alignment is consistent with the measured lag
  evidence (:data:`ARRAY_SYNC_EVIDENCE`). An array whose alignment is refuted
  (:data:`ARRAY_MISALIGNED`) is its own Frame ``<folder>/matrix.wav`` on its
  own clock, never shifted.
- ``telemetry``: every timestamped CSV row of a folder whose StartTimestamp
  is the logged audio start, time-last on a ``StampIndex`` (seconds) with the
  original column names and units; rows outside the audio stay. Rows whose
  TIMESTAMP is N/A are kept in ``telemetry_unaligned/<track>_untimed`` with
  their CSV row numbers.
- ``telemetry_unaligned``: ``Other`` flights (audio start never logged) keep
  every row on its own row axis with ``logger_seconds`` from the nominal
  StartTimestamp; :data:`OTHER_CLOCK_EVIDENCE` and the publisher's excerpt
  timestamps are reported in ``meta.audio_start_evidence``, not applied. CRNN
  prediction exports have no time column: ordered rows only.
- ``Delivery/*`` and ``Flights/SNR test/*`` excerpts are cut from originals
  at whole seconds and re-quantised (every sample within 1 LSB of the parent,
  not bit-identical). Each stays its own derivative Frame with its own files,
  annotated with the measured parent span (``meta.derivative_of``); the parent
  lists them in ``meta.excerpts``. Nothing is glued or de-duplicated.
- ``Noise/ManuallySelectedNoise`` (72 Pixabay downloads) is third-party
  content outside the GPL grant and is excluded (:data:`EXCLUDED_PREFIXES`).

Audio is exact float32 (int16 / 32768), native rate and channels.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from urllib.parse import quote

import numpy as np
import soundfile as sf
import tdseries as td

from data_processing.sources._common import meta_frame, validate_operating

DATASET = "D3SET"
GIT_REPO = "introlab/d3set"
#: ``main`` HEAD on 2026-03-14 ("Added link to kaggle in readme"); unchanged
#: on 2026-11-26.
GIT_COMMIT = "b7f80cfbb1393ed490a8677cb1550ed95b076718"
MEDIA_URL = f"https://media.githubusercontent.com/media/{GIT_REPO}/{GIT_COMMIT}/"
RAW_URL = f"https://raw.githubusercontent.com/{GIT_REPO}/{GIT_COMMIT}/"
#: Plain git blobs (the media endpoint serves only LFS objects).
NON_LFS = frozenset({"README.md", "LICENSE"})
#: ``path<TAB>bytes<TAB>sha256`` for every tracked file at :data:`GIT_COMMIT`
#: (LFS oid for LFS files, content sha256 for README.md/LICENSE).
FILES_TSV = Path(__file__).with_name("d3set_files.tsv")
VERIFIED_MARKER = ".d3set-verified"
_CHUNK = 1 << 20

PROVENANCE: dict[str, Any] = {
    "source_url": f"https://github.com/{GIT_REPO}",
    "git_commit": GIT_COMMIT,
    "mirror_url": "https://www.kaggle.com/datasets/francoismarcoux1212/d3set-introlab",
    "mirror_version": "Kaggle v1 (datasetVersionId 15183685); not byte-identical (26 mai CSV/JSON)",
    "license": "GPL-3.0 (repository LICENSE; Kaggle lists 'GPL 3')",
    "license_scope": (
        "Recordings, telemetry and predictions authored by IntRoLab. Noise/ManuallySelectedNoise "
        "is third-party Pixabay content (README) outside that grant and is excluded."
    ),
    "citation": (
        "F. Marcoux and F. Grondin, 'Acoustic Drone Package Delivery Detection', "
        "arXiv:2602.09991 (2026)."
    ),
    "attribution": "D3SET, IntRoLab, Universite de Sherbrooke (F. Marcoux, F. Grondin).",
    "paper": "arXiv:2602.09991",
    "files_manifest": "data_processing/sources/d3set_files.tsv (path, bytes, sha256 per file)",
    "description": (
        "Holybro X500 V2 quadrotor flights with a servo package-drop mechanism: onboard mono "
        "microphone + ground 16-mic MEMS array (16 kHz), MAVSDK telemetry (acceleration, body "
        "velocity, Euler angles, GPS, in-air flag, drop-servo PWM, motor PWM or BPF labels) and "
        "the authors' CRNN BPF estimates. One whole-recording Frame per release folder: 17 "
        "onboard+array flights (paper split 8/3/6), 4 distance-test hover flights, 11 early "
        "onboard-only flights, and 27 delivery/SNR-test excerpts kept as annotated derivatives "
        "of their parent flights."
    ),
}

# ─── Rig facts (evidence in strings) ─────────────────────────────────────────

N_ROTORS = 4
SYSTEM: dict[str, Any] = {
    "platform": "Holybro X500 V2 (modified: servo drop mechanism, onboard Raspberry Pi)",
    "platform_evidence": "arXiv:2602.09991 §III-A and Fig. 4",
    "n_rotors": N_ROTORS,
    "blades_per_rotor": 2,
    "blades_per_rotor_evidence": (
        "estimated: the X500 V2 kit ships 2-blade 10x4.5 in (1045) propellers (Holybro product "
        "page); the paper defines BPF = N_p * rev/s without stating N_p"
    ),
    "rotor_geometry": (
        "estimated: 500 mm wheelbase (Holybro spec) -> hubs ~0.25 m from centre in an X; the "
        "actuator.csv MOTOR_1..4 -> airframe position mapping is undocumented (depends on the "
        "autopilot output configuration); no rotor_pos published"
    ),
    "motors": "Holybro 2216 KV920 (X500 V2 kit spec; not restated in the paper)",
    "autopilot_link": "MAVLink telemetry read through MAVSDK on the RPi (README links)",
    "payload": "500-600 g package for delivery flights (paper §III-A)",
}
ARRAY_GEOMETRY: dict[str, Any] = {
    "kind": "two coaxial horizontal rings; per-channel coordinates not published, none emitted",
    "device": "16 MEMS microphones on a 16SoundsUSB card (paper ref. [9], Lagace et al. 2023)",
    "n_microphones": 16,
    "ring_axis": "vertical (rings horizontal, radial axes in the ring planes)",
    "ring_radius_lower_m": 0.10,
    "ring_radius_upper_m": 0.061,
    "ring_separation_m": 0.25,
    "slot_step_deg": 30.0,
    "slots_per_ring_nominal": 12,
    "populated_slots_observed": (
        "photo: lower ring ~8 of 12 drilled slots populated, upper ring ~8 populated (partly "
        "occluded); slot indices and their WAV-channel correspondence unknown"
    ),
    "evidence": {
        "ring_radius_lower_m": "label '10 cm', double arrow outer slot -> ring centre (Fig. 4)",
        "ring_radius_upper_m": (
            "label '6.1cm', same arrow style, interpreted as a radius (estimated/interpretative)"
        ),
        "ring_separation_m": "label '25 cm' between the ring plates (Fig. 4)",
        "slot_step_deg": "label '30°' between adjacent lower-ring slots; nominal",
        "layout_intent": "'positioned to minimize symmetry' (paper §III-A)",
        "source": "arXiv:2602.09991 Fig. 4 (PDF p.4) and §III-A",
    },
    "orientation": "azimuth not aligned to true north or any marker (README 'Warning about GPS')",
}
#: Onboard microphone, top-level ``mic_pos`` (metres, body frame: origin = centre of
#: the lower body plate, z up; x/y signed axes unresolved -> NaN).
ONBOARD_MIC_POS = [[math.nan, math.nan, -0.22]]
ONBOARD_MIC: dict[str, Any] = {
    "status": "estimated from the photo: z only; x/y unresolved (NaN)",
    "placement": (
        "single microphone in a foam windscreen on a flexible gooseneck beside the pictured "
        "right landing skid, below the airframe and the rotor plane; not rigidly tied to a "
        "motor (Fig. 4 left)"
    ),
    "mic_pos_frame": (
        "metres; origin = centre of the lower body plate, z up; signed x/y body axes not "
        "resolved from the photo"
    ),
    "mic_pos_z_m": -0.22,
    "mic_pos_z_uncertainty_m": 0.05,
    "lateral_offset_magnitude_m": 0.10,
    "lateral_offset_range_m": [0.09, 0.11],
    "lateral_offset_note": (
        "horizontal displacement toward the pictured skid; direction (front/back, left/right) "
        "unresolved"
    ),
    "mic_pos_evidence": (
        "Fig. 4 left (PDF p.4 image 1463x655): plate/gear attachment y~260 px, skid junction "
        "y~580 px, windscreen centre y~605 px, scaled by the 0.215 m landing-gear height and "
        "144x144 mm body plates (Holybro X500 V2 spec, "
        "https://docs.holybro.com/drone-development-kit/px4-development-kit-x500v2); "
        "±0.05 m allowance for attachment, perspective and gooseneck flex; not tracked in flight"
    ),
    "mic_pos_scope": (
        "assumed for every session: the photo is undated and no other placement is documented"
    ),
    "capture": "recorded at 48 kHz on the RPi, resampled to 16 kHz by the authors (README)",
    "evidence": "arXiv:2602.09991 Fig. 4; README 'Audio files'",
}

# ─── Layout of the release ───────────────────────────────────────────────────

#: Release files deliberately not converted, with the reason.
EXCLUDED_PREFIXES: dict[str, str] = {
    "Noise/": (
        "72 third-party sound effects (aircraft, helicopter, insect) downloaded from Pixabay "
        "(README 'Noise'); Pixabay Content License forbids standalone redistribution and is not "
        "relicensable under GPL-3.0 by D3SET, so no standalone copy is published"
    ),
}
#: Group (folder prefix) -> (subcategory, native split, role).
GROUPS: dict[str, tuple[str, str, str]] = {
    "Flights/Training/": ("bpf_train_flight", "train", "original"),
    "Flights/Validation/": ("bpf_validation_flight", "validation", "original"),
    "Flights/Test/": ("bpf_test_flight", "test", "original"),
    "Flights/Distance test flights/": ("distance_test_hover", "distance_test", "original"),
    "Other/": ("early_flight_unsynced", "other", "original"),
    "Delivery/": ("delivery_excerpt", "delivery", "excerpt"),
    "Flights/SNR test/": ("snr_test_excerpt", "snr_test", "excerpt"),
}
#: Excerpt folder -> parent folder. Evidence: folder names + StartTimestamp
#: differences; the builder re-verifies every pair by exact PCM match.
EXCERPT_PARENTS: dict[str, str] = {
    **{f"Delivery/23_oct_vol4_{i}": "Flights/Test/23_oct_vol4" for i in range(1, 5)},
    **{f"Delivery/26 oct vol 5 {i}": "Flights/Test/26 oct vol 5" for i in (1, 2)},
    **{f"Delivery/26 oct vol 6 {i}": "Flights/Test/26 oct vol 6" for i in (1, 2)},
    **{f"Delivery/23_sept_drp_charge_{i}": "Other/23_sept_drp_charge" for i in range(6)},
    **{f"Delivery/29_aout_grp_05_charge_{i}": "Other/29_aout_grp_05_charge" for i in (1, 2, 3)},
    **{f"Delivery/29_aout_grp_05_charge2_{i}": "Other/29_aout_grp_05_charge2" for i in range(1, 9)},
    "Flights/SNR test/drone": "Flights/Training/31 oct vol 1",
    "Flights/SNR test/background noise": "Flights/Training/6 nov vol 2",
}
#: 8 + 3 + 6 synchronised flights, 4 distance-test flights, 11 ``Other`` flights
#: (paper §III-A: 17 + 11 = 28 flights, plus distance flights).
N_ORIGINALS = 32
#: wav -> (Frame entry, mic_placement). Both are 16 kHz and the same length per folder.
STREAMS: dict[str, tuple[str, str]] = {
    "drone.wav": ("audio", "onboard"),
    "matrix.wav": ("array_audio", "external"),
}
#: Channel dimension of ``array_audio`` (distinct from the onboard ``mic`` of ``mic_pos``).
ARRAY_DIM = "array_mic"
#: Excerpt WAVs are re-quantised copies of their parent span: tolerated |difference| (int16 LSB).
EXCERPT_TOLERANCE_LSB = 1
#: Search radius around the StartTimestamp-implied parent offset, seconds.
EXCERPT_SEARCH_S = 3
#: Samples of the excerpt probe used to find candidate parent offsets.
EXCERPT_PROBE = 64
#: CRNN analysis hop (paper Table I: 512 samples at 16 kHz). Delivery predictions
#: carry no time column or documented inference-window/export clock.
PRED_HOP = 512
UNIX_MS_FLOOR = 1e11  # TIMESTAMP above this is Unix ms; below, ms since audio start

# ─── Clock evidence (measured on the pinned raw files, 2026-11-26) ───────────

#: How the onboard/array lags were measured (raw read-only probes on Apocrita).
ARRAY_SYNC_METHOD = (
    "(1) clap check: 2-7.9 kHz onset strength (5 ms hops, dB above a 0.5 s mean) in drone.wav "
    "and the loudest array channel over the first 90 s of array audio; transients >= 15 dB "
    "paired within 10 ms for every candidate lag within ±3 s. (2) GCC-PHAT, 300-6000 Hz, "
    "4 s drone.wav windows every 2 s against all array channels, lag search ±15 s; 'strong' "
    "= peak > 8x the median |GCC| and > 1.3x the next peak beyond ±10 ms. Lag > 0: the array "
    "hears it later. Expected lag = drone-to-array distance / c (~2.9 ms/m) + residual sync "
    "error, so ground epochs near the array bound the sync error and in-flight lags must be "
    "positive. GPS (README: unreliable altitude) only bounds distance."
)
#: Folders whose published ``matrix.wav`` alignment to ``drone.wav`` the evidence refutes.
#: The array then stays a separate, unshifted Frame; no correction is applied.
ARRAY_MISALIGNED: dict[str, dict[str, Any]] = {
    "Flights/Training/26 oct session2 vol2": {
        "measured_array_minus_drone_s": -0.92,
        "measured_uncertainty_s": 0.01,
        "method": ARRAY_SYNC_METHOD,
        "clap_evidence": (
            "6 onboard claps (13.635-18.200 s, 0.88-0.93 s apart) pair with 6 array claps "
            "0.915-0.920 s EARLIER (12.715-17.280 s); at the published zero lag only 3 pair "
            "and the loud first array clap has no onboard partner: the alignment is off by one "
            "clap period"
        ),
        "gcc_evidence": (
            "strong peaks at -0.921/-0.910 s near the array (20/40 s, ~2.5 m) and -0.734/"
            "-0.734/-0.760 s in flight (93/95/119 s, ~50-63 m GPS distance): the array would "
            "hear the drone before its own microphone, impossible if aligned"
        ),
    },
}


def _sync(claps: str, gcc: str) -> dict[str, Any]:
    return {"clap_evidence": claps, "gcc_evidence": gcc, "method": ARRAY_SYNC_METHOD}


#: Folders whose published onboard/array alignment the evidence supports (array kept
#: on the drone.wav time axis). Lags in ms at ``t`` s of the recording.
ARRAY_SYNC_EVIDENCE: dict[str, dict[str, Any]] = {
    "Flights/Distance test flights/24 avril vol 1": _sync(
        "9 claps pair at 0..+5 ms (4.3-9.0 s); best competing lag (+0.395 s) pairs 3",
        "-0.9 ms at 17 s, -7.1/-7.6 ms at 29-31 s, +9.2..+10.6 ms at 39-45 s; isolated -15.0 s "
        "and -12.6 s peaks at 25/79 s",
    ),
    "Flights/Distance test flights/24 avril vol 2": _sync(
        "7 claps pair at 0..+5 ms (6.1-10.7 s); best competing lag (-0.87 s) pairs 1",
        "+2.6..+4.2 ms at 23-29 s near the array, +14.3..+21.3 ms at 45-53 s, +29.8..+297.6 ms at "
        "16 hover epochs 263-449 s (lag falls as the drone returns); isolated -7.78 s at 33 s",
    ),
    "Flights/Distance test flights/24 avril vol 3": _sync(
        "8/8 claps pair at 0..+5 ms (5.6-9.7 s); best competing lag (-0.97 s) pairs 2",
        "+6.4/+6.8 ms at 19-29 s, +13.7..+21.4 ms at 51-57 s",
    ),
    "Flights/Distance test flights/24 avril vol 4": _sync(
        "7 claps pair at +5..+10 ms (2.9-17.9 s); best competing lag (-0.745 s) pairs 3",
        "+5.3..+8.9 ms at 17-35 s near the array; +66.8..+237.6 ms at 13 hover epochs 97-497 s; "
        "one isolated -5.80 s peak at 41 s",
    ),
    "Flights/Test/23_oct_vol4": _sync(
        "3 claps pair at 0..+5 ms (26.0-27.1 s); competing lags pair 1",
        "+22.0..+39.3 ms at 20 epochs 40-302 s, +8.9 ms at 148 s; no negative peak",
    ),
    "Flights/Test/26 mai vol 1": _sync(
        "6 claps pair at 0..+5 ms (13.7-17.8 s); best competing lag (+0.83 s) pairs 3",
        "+0.6..+2.1 ms at 24-38 s near the array, +8.4..+18.6 ms at 62-82 s, +40.7..+42.9 ms at "
        "214-218 s, +120.5 ms at 126 s",
    ),
    "Flights/Test/26 mai vol 3": _sync(
        "7 claps pair at 0..+5 ms (10.7-16.2 s); best competing lag (+1.745 s) pairs 3",
        "+3.9/+5.4 ms at 33-35 s, +17.6..+23.9 ms at 59-73 s, +50.8..+53.3 ms at 361-483 s; "
        "isolated -10.2 s, -14.8 s, +6.9 s peaks at 43/93/377 s",
    ),
    "Flights/Test/26 oct vol 5": _sync(
        "not testable: the 4 onboard claps (3.4-6.5 s) precede the array start (7.76 s)",
        "-20.4/-19.8 ms at 25-27 s, 0.0..+6.2 ms at 31-47 s near the array, +5.6/+11.2 ms at "
        "71-73 s, +23.6..+86.7 ms at 13 in-flight epochs 129-463 s; isolated +3.7 s, -13.0 s, "
        "-5.07 s, +10.3 s peaks at 295/401/465/519 s",
    ),
    "Flights/Training/23_oct_vol1": _sync(
        "2 claps pair at -5 ms (20.4, 22.9 s); no competing lag pairs more than 1",
        "-4.9..-1.4 ms at 25-39 s near the array, +5.1..+9.6 ms at 61-81 s, +24.2..+29.8 ms at "
        "125-153 s; one isolated -14.9 s peak at 45 s",
    ),
    "Flights/Training/25 oct": _sync(
        "3/3 claps pair at 0..+5 ms (11.1-12.4 s)",
        "+11.4 ms at 31-33 s, +20.9..+26.5 ms in flight (81-97 s), -9.9 ms at 105-107 s",
    ),
    "Flights/Training/26 oct session2 vol1": _sync(
        "no clap >= 15 dB in both recordings",
        "-4.6..-0.9 ms at 6 epochs 26-36 s near the array; +13.1..+59.4 ms at 14 in-flight epochs "
        "214-510 s; isolated +14.2 s, +2.3 s, -12.2 s peaks at 164/260/276 s",
    ),
    "Flights/Training/26 oct session2 vol3": _sync(
        "1 clap pairs at -10 ms (2.3 s)",
        "+9.4..+14.4 ms at 35-43 s, +42.4..+126.3 ms at 9 in-flight epochs 125-323 s",
    ),
    "Flights/Training/26 oct vol 2": _sync(
        "best clap lag -10 ms (4 pairs); no competing lag",
        "+0.1..+3.0 ms at 17 epochs 17-53 s (near the array), +118.1 ms at 267 s in flight",
    ),
    "Flights/Training/31 oct vol 1": _sync(
        "2 claps pair at 0 ms (7.2, 8.2 s); competing lags pair 1",
        "+7.6/+11.1 ms at 24-28 s, +29.4..+122.2 ms at 18 in-flight epochs 112-498 s; isolated "
        "-15.0 s and -8.2 s peaks at 258/482 s",
    ),
    "Flights/Training/6 nov vol 2": _sync(
        "5 claps pair at -5..0 ms (5.9-9.0 s); best competing lag (-0.805 s) pairs 2",
        "+18.6 ms at 38 s",
    ),
    "Flights/Validation/23_oct_vol2": _sync(
        "6/6 claps pair at 0..+5 ms (16.2-20.2 s); best competing lag (+0.795 s) pairs 4",
        "+13.3 ms at 39 s, +27.1..+28.0 ms at 367-373 s; one isolated -556 ms peak at 301 s",
    ),
    "Flights/Validation/23_oct_vol3": _sync(
        "4 claps pair at 0 ms (21.9-24.5 s); competing lags pair 1",
        "136 of 138 strong epochs at -1.1..+42.3 ms over 46-650 s (16 of them -1.1..+7.8 ms, "
        "59-364 s); isolated -14.9 s and -4.9 s peaks at 80/248 s",
    ),
    "Flights/Validation/26 oct vol 4": _sync(
        "5 claps pair at -30/-35 ms (6.1-9.6 s; clapper nearer the array); competing lag pairs 2",
        "+1.6..+2.9 ms at 27-35 s near the array, +39.6/+39.9 ms at 127-131 s; isolated -9.1 s "
        "and +11.2 s peaks at 43/83 s",
    ),
    "Flights/Test/26 mai vol 2": _sync(
        "7 claps pair at 0..+5 ms (5.1-9.8 s); best competing lag (-0.78 s) pairs 3",
        "all 19 strong epochs positive: +5.7 ms at 25 s and +3.6..+5.8 ms at 541-595 s (on the "
        "ground near the array at both ends: drift < ~2 ms over 570 s), +14.7..+66.8 ms between",
    ),
    "Flights/Test/26 oct vol 6": _sync(
        "ambiguous: 2 claps pair at 0 ms (6.2, 7.2 s) and 2 at ±0.97 s",
        "-16.8..-7.5 ms at 23-29 s, +7.9..+12.4 ms at 9 epochs 33-73 s, +28.4..+42.8 ms at "
        "385-459 s: no ±0.97 s peak; onset envelopes peak at 0 ±10 ms at 31/46/61/451 s; "
        "isolated +14.9 s peak at 255 s",
    ),
}


def _other(vib: float, onset: float, thrust: float, verdict: str) -> dict[str, Any]:
    return {
        "acoustic_vs_imu_lag_s": {
            "vibration_envelope": vib,
            "vibration_onsets": onset,
            "thrust_fluctuation": thrust,
        },
        "verdict": verdict,
        "method": OTHER_CLOCK_METHOD,
    }


OTHER_CLOCK_METHOD = (
    "50 Hz drone.wav log-RMS envelope cross-correlated (±5 s) with the accelerometer rows on "
    "the nominal StartTimestamp clock: vibration envelope (log rolling std of high-passed "
    "|a|), its onsets, and high-passed -ACCEL_DOWN (thrust). Lag > 0: telemetry stamps later "
    "than the audio. Calibration on the 21 folders with a logged audio start: vibration-"
    "envelope lag -0.12..+0.31 s, median +0.09 s (RPi receive latency + estimator bias), "
    "with flat correlation peaks: resolution ~±0.3 s. Not applied to the Frames."
)
#: Audio-start evidence for the ``Other`` flights (StartTimestamp = first telemetry row).
OTHER_CLOCK_EVIDENCE: dict[str, dict[str, Any]] = {
    "Other/14_aout": _other(
        0.185, -0.025, 0.251, "within calibration: start ~= StartTimestamp ±0.3 s"
    ),
    "Other/23_sept_01": _other(0.334, 0.286, -3.845, "within calibration (thrust lag unreliable)"),
    "Other/23_sept_02": _other(
        -2.74, -2.714, -2.761, "audio began ~2.8 s BEFORE StartTimestamp (3 estimators agree)"
    ),
    "Other/23_sept_03": _other(
        -5.0, 0.728, 3.05, "undetermined: estimators disagree; audio 21.7 s longer than the log"
    ),
    "Other/23_sept_drp_charge": _other(
        -1.34, -1.371, -1.265, "audio began ~1.4 s BEFORE StartTimestamp (3 estimators agree)"
    ),
    "Other/24_sept_02": _other(0.312, 0.267, -4.2, "within calibration (thrust lag unreliable)"),
    "Other/24_sept_03": _other(
        0.633, 0.775, 0.64, "audio began ~0.5-0.7 s AFTER StartTimestamp (3 estimators agree)"
    ),
    "Other/24_sept_04": _other(
        0.331, 0.298, 0.512, "within calibration: start ~= StartTimestamp ±0.3 s"
    ),
    "Other/29_aout_grp_05_charge": _other(
        0.294, 0.247, 0.286, "within calibration: start ~= StartTimestamp ±0.3 s"
    ),
    "Other/29_aout_grp_05_charge2": _other(
        0.294, 0.266, 0.339, "within calibration: start ~= StartTimestamp ±0.3 s"
    ),
    "Other/8_aout": _other(0.259, 0.274, -4.18, "within calibration (thrust lag unreliable)"),
}

TRACKS: dict[str, dict[str, Any]] = {
    "acceleration.csv": {
        "name": "acceleration_frd",
        "unit": "m/s^2",
        "quantity": "MAVSDK Telemetry::AccelerationFrd (body forward/right/down)",
        "provenance": "measured (IMU via autopilot)",
    },
    "velocity.csv": {
        "name": "velocity_body",
        "unit": "m/s",
        "quantity": "MAVSDK Telemetry::VelocityBody (body frame)",
        "provenance": "measured/estimated by autopilot state estimator",
    },
    "orientation.csv": {
        "name": "orientation_euler",
        "unit": "deg",
        "quantity": "MAVSDK Telemetry::EulerAngle roll/pitch/yaw",
        "provenance": "autopilot attitude estimate",
    },
    "position.csv": {
        "name": "position_gps",
        "unit": "deg, deg, m",
        "quantity": "MAVSDK Telemetry::Position latitude/longitude/altitude",
        "provenance": "GPS; altitude unreliable, jumps mid-flight (README)",
    },
    "isflying.csv": {
        "name": "is_flying",
        "unit": "flag (1 flying, 0 not)",
        "quantity": "MAVSDK Telemetry::in_air",
        "provenance": "autopilot land detector",
    },
    "pwm.csv": {
        "name": "servo_pwm",
        "unit": "us",
        "quantity": "drop-mechanism servo PWM (README: ~1900 closed, ~1300 open; approximate)",
        "provenance": "commanded servo output; openings without delivery occur (README)",
    },
    "drone_predictions.csv": {
        "name": "bpf_estimate_onboard",
        "unit": "Hz",
        "quantity": "CRNN BPF estimate of the two fastest motors from drone.wav",
        "provenance": "model prediction (arXiv:2602.09991 §II-A), NOT ground truth",
    },
    "matrix_predictions.csv": {
        "name": "bpf_estimate_array",
        "unit": "Hz",
        "quantity": "CRNN BPF estimate of the two fastest motors from matrix.wav",
        "provenance": "model prediction (arXiv:2602.09991 §II-A), NOT ground truth",
    },
    "audio_predictions.csv": {
        "name": "bpf_estimate_audio",
        "unit": "Hz",
        "quantity": "CRNN BPF estimate of the two fastest motors (drone.wav: only audio present)",
        "provenance": "model prediction (arXiv:2602.09991 §II-A), NOT ground truth",
    },
}
ACTUATOR_PWM = {
    "name": "motor_pwm",
    "unit": "us",
    "quantity": "MAVSDK ActuatorOutputStatus: PWM sent to the 4 motors (raw)",
    "provenance": "commanded by flight controller; not rotor speed",
}
ACTUATOR_BPF = {
    "name": "bpf_label",
    "unit": "Hz",
    "quantity": "per-motor blade-passing frequency label (README: BPF instead of raw PWM)",
    "provenance": (
        "derived by the authors: PWM -> BPF calibration curve (optical tachometer, averaged over "
        "motors) plus a manual per-flight offset ramp (typically <= 20 Hz) to match the audible "
        "BPF (paper §III-A); estimated, not measured"
    ),
}
FLIGHT_OPERATING: dict[str, Any] = {
    "condition": "flight",
    "n_active_rotors": N_ROTORS,
    "contains_rotor_noise": True,
    "external_source": "unknown",
    "confidence": "medium",
}


def pinned_files() -> list[tuple[str, int, str]]:
    """``(relative path, bytes, sha256)`` of every pinned raw file."""
    with FILES_TSV.open(encoding="utf-8", newline="") as f:
        return [
            (r["path"], int(r["bytes"]), r["sha256"]) for r in csv.DictReader(f, delimiter="\t")
        ]


def _sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def download_d3set(dest: Path) -> Path:
    """Fetch every pinned, converted file of the D3SET commit into ``dest`` (idempotent).

    :data:`EXCLUDED_PREFIXES` (third-party Pixabay noise) are not fetched.
    Each file streams to ``<file>.part`` with a running sha256 and is renamed
    only when size and hash match the pin. A completed tree is marked with
    :data:`VERIFIED_MARKER` (commit + manifest hash); re-runs then only
    re-check sizes.
    """
    import requests  # deferred: transitively present, not a hard import-time dep

    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    files = [f for f in pinned_files() if not any(f[0].startswith(p) for p in EXCLUDED_PREFIXES)]
    stamp = f"{GIT_COMMIT} {_sha256(FILES_TSV)}\n"
    marker = dest / VERIFIED_MARKER
    if (
        marker.exists()
        and marker.read_text(encoding="utf-8") == stamp
        and all((dest / p).is_file() and (dest / p).stat().st_size == n for p, n, _ in files)
    ):
        return dest
    marker.unlink(missing_ok=True)
    with requests.Session() as session:
        for i, (rel, size, digest) in enumerate(files):
            target = dest / rel
            if target.is_file() and target.stat().st_size == size and _sha256(target) == digest:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            h = hashlib.sha256()
            got = 0
            url = (RAW_URL if rel in NON_LFS else MEDIA_URL) + quote(rel)
            with session.get(url, stream=True, timeout=120) as resp:
                resp.raise_for_status()
                with part.open("wb") as fh:
                    for block in resp.iter_content(chunk_size=_CHUNK):
                        fh.write(block)
                        h.update(block)
                        got += len(block)
            if got != size or h.hexdigest() != digest:
                part.unlink(missing_ok=True)
                raise OSError(
                    f"D3SET {rel}: got {got} B sha256 {h.hexdigest()}, want {size} B {digest}"
                )
            part.replace(target)
            print(f"  [{i + 1}/{len(files)}] {rel} ({size} B)", flush=True)
    marker.write_text(stamp, encoding="utf-8")
    return dest


# ─── Raw readers ──────────────────────────────────────────────────────────────


@dataclass
class Table:
    """One CSV: values ``(channel, row)`` float64 (N/A -> NaN) + its clock.

    Rows whose TIMESTAMP cell is N/A are not dropped: their values and
    0-based data-row numbers are kept in ``untimed`` / ``untimed_rows``.
    """

    file: str
    columns: list[str]
    values: np.ndarray
    t: np.ndarray | None  # seconds on the recording/nominal clock, or absent
    clock: str
    info: dict[str, Any] = field(default_factory=dict)
    untimed: np.ndarray | None = None  # (channel, row) values of rows without a timestamp
    untimed_rows: np.ndarray | None = None  # their 0-based data-row numbers in the CSV


@dataclass
class Recording:
    """One release folder: everything but the CSV payloads (loaded per use)."""

    rel: str
    group: str
    subcategory: str
    split: str
    role: str
    start_ms: float | None
    additional: dict[str, Any]
    wavs: dict[str, Any]  # wav name -> soundfile info
    csvs: list[str]
    files: list[dict[str, Any]]

    @property
    def ref(self) -> Any:
        return self.wavs.get("drone.wav") or next(iter(self.wavs.values()))


def _group(rel_dir: str) -> tuple[str, str, str, str]:
    for prefix, (sub, split, role) in GROUPS.items():
        if rel_dir.startswith(prefix):
            return prefix.rstrip("/"), sub, split, role
    raise KeyError(f"D3SET: {rel_dir!r} is in no known group")


def _cells(line: str) -> list[str]:
    return [c.strip().strip('"').strip() for c in line.split(";")]


def _read_start(path: Path) -> float:
    rows = [r for r in path.read_text(encoding="utf-8").splitlines() if r.strip()]
    if len(rows) != 2 or _cells(rows[0]) != ["Start Time"]:
        raise ValueError(f"{path}: unexpected StartTimestamp layout {rows!r}")
    return float(_cells(rows[1])[0])


def _same_rows(excerpt: Path, parent: Path) -> bool:
    """Check that the parent already contains the excerpt's rows in file order."""
    with excerpt.open(encoding="utf-8") as ex, parent.open(encoding="utf-8") as src:
        if _cells(ex.readline()) != _cells(src.readline()):
            return False
        parent_rows = (line.strip() for line in src)
        return all(
            any(row == existing for existing in parent_rows) for line in ex if (row := line.strip())
        )


def read_table(root: Path, rec: Recording, name: str) -> Table:
    """One CSV of ``rec`` on its recording clock, clock units resolved per file."""
    path = root / rec.rel / name
    first, _, rest = path.read_text(encoding="utf-8").partition("\n")
    header = _cells(first.strip())
    rows = [_cells(ln) for ln in (s.strip() for s in rest.splitlines()) if ln]
    del rest
    if any(len(r) != len(header) for r in rows):
        raise ValueError(f"{path}: ragged rows")
    arr = np.array(
        [[math.nan if c in ("N/A", "") else float(c) for c in r] for r in rows], dtype=np.float64
    ).reshape(len(rows), len(header))
    del rows
    info: dict[str, Any] = {"rows": int(arr.shape[0])}
    untimed = untimed_rows = None
    if "TIMESTAMP" in header:
        k = header.index("TIMESTAMP")
        ts = arr[:, k]
        keep = np.isfinite(ts)
        info["rows_without_timestamp"] = int((~keep).sum())
        if not keep.all():
            untimed_rows = np.flatnonzero(~keep)
            untimed = np.ascontiguousarray(np.delete(arr[~keep], k, axis=1).T)
            info["rows_without_timestamp_all_values_na"] = bool(not np.isfinite(untimed).any())
        ts, vals = ts[keep], np.delete(arr[keep], k, axis=1).T
        cols = [h for h in header if h != "TIMESTAMP"]
        if ts.size and ts.max() > UNIX_MS_FLOOR:
            if rec.start_ms is None:
                raise ValueError(f"{path}: Unix-ms rows but no StartTimestamp")
            t = (ts - rec.start_ms) / 1000.0
            clock = "unix_ms (RPi receive time) minus StartTimestamp, / 1000"
        else:
            t = ts / 1000.0
            clock = "relative: file TIMESTAMP is ms since the first drone.wav sample, / 1000"
        info["timestamp_resolution_s"] = 0.001
        order = np.argsort(t, kind="stable")
        info["rows_reordered_by_time"] = int((order != np.arange(order.size)).sum())
        t, vals = t[order], vals[:, order]
    else:
        if name not in {"audio_predictions.csv", "drone_predictions.csv", "matrix_predictions.csv"}:
            raise ValueError(f"{path}: expected a TIMESTAMP column, got {header!r}")
        cols, vals = header, arr.T
        t = None
        wav = "matrix.wav" if name == "matrix_predictions.csv" else "drone.wav"
        audio = rec.wavs[wav]
        nominal_rows = audio.frames / audio.samplerate * (16000 / PRED_HOP)
        # The paper specifies the input analysis hop, not which CRNN output row
        # corresponds to each audio sample. An approximate count is not a clock.
        clock = "unclocked: original CSV row order; analysis hop is not an export timestamp"
        info.update(
            related_audio=f"{rec.rel}/{wav}",
            expected_rows_at_analysis_hop=round(nominal_rows, 3),
            rows_x_hop_over_audio_samples=round(arr.shape[0] / nominal_rows, 4),
        )
    finite = vals[np.isfinite(vals)]
    if finite.size:
        info["value_range"] = [float(finite.min()), float(finite.max())]
    info["nan_values"] = int((~np.isfinite(vals)).sum())
    return Table(name, cols, np.ascontiguousarray(vals), t, clock, info, untimed, untimed_rows)


def describe(table: Table) -> dict[str, Any]:
    """Track name + semantics, resolving the per-flight actuator.csv / pwm.csv meanings."""
    v = table.values
    desc: dict[str, Any]
    if table.file == "actuator.csv":
        nz = v[np.isfinite(v) & (v != 0)]
        bpf = table.clock.startswith("relative") and (nz.size == 0 or float(nz.max()) < 900.0)
        desc = dict(ACTUATOR_BPF if bpf else ACTUATOR_PWM)
    else:
        desc = dict(TRACKS[table.file])
    if table.file == "pwm.csv":
        fin = v[np.isfinite(v)]
        if fin.size and float(np.median(fin)) < 500.0:
            desc["unit"] = "unknown"
            desc["quantity"] += (
                "; this flight's values lie far below the servo us range: units/meaning undocumented"
            )
    if table.file == "position.csv":
        alt = v[2][np.isfinite(v[2])]
        desc["altitude_reference"] = (
            "absolute (inferred from values ~200 m)"
            if alt.size and float(np.median(alt)) > 50.0
            else "relative to home (inferred from values ~0 m)"
        )
    desc.update(file=table.file, columns=table.columns, clock=table.clock, **table.info)
    return desc


def _scan(root: Path) -> dict[str, Recording]:
    """Every converted folder, sizes checked against the pin, in release (sorted) order."""
    by_dir: dict[str, list[tuple[str, int, str]]] = {}
    for rel, size, digest in pinned_files():
        if "/" not in rel or any(rel.startswith(p) for p in EXCLUDED_PREFIXES):
            continue  # README.md / LICENSE: documentation; Noise/: excluded
        by_dir.setdefault(rel.rsplit("/", 1)[0], []).append((rel, size, digest))
    known = {*STREAMS, *TRACKS, "actuator.csv", "StartTimestamp.csv", "additional_data.json"}
    recs: dict[str, Recording] = {}
    for rel_dir in sorted(by_dir):
        d = root / rel_dir
        files = []
        for rel, size, digest in by_dir[rel_dir]:
            p = root / rel
            if not p.is_file() or p.stat().st_size != size:
                raise FileNotFoundError(f"D3SET {rel}: missing or size != {size} (re-run fetch)")
            files.append({"relpath": rel, "size": size, "sha256": digest})
        names = {Path(f["relpath"]).name for f in files}
        if names - known:
            raise ValueError(f"D3SET {rel_dir}: unhandled files {sorted(names - known)}")
        wavs = {w: sf.info(str(d / w)) for w in STREAMS if w in names}
        if not wavs:
            raise ValueError(f"D3SET {rel_dir}: no audio")
        for w, info in wavs.items():
            if info.subtype != "PCM_16":
                raise ValueError(f"D3SET {rel_dir}/{w}: {info.subtype} is not PCM_16")
        group, sub, split, role = _group(rel_dir + "/")
        recs[rel_dir] = Recording(
            rel=rel_dir,
            group=group,
            subcategory=sub,
            split=split,
            role=role,
            start_ms=_read_start(d / "StartTimestamp.csv")
            if "StartTimestamp.csv" in names
            else None,
            additional=json.loads((d / "additional_data.json").read_text(encoding="utf-8"))
            if "additional_data.json" in names
            else {},
            wavs=wavs,
            csvs=sorted(n for n in names if n.endswith(".csv") and n != "StartTimestamp.csv"),
            files=files,
        )
    return recs


def _verify_hashes(root: Path) -> None:
    """Full sha256 check of every converted file unless the fetch marker vouches."""
    marker = root / VERIFIED_MARKER
    if (
        marker.exists()
        and marker.read_text(encoding="utf-8") == f"{GIT_COMMIT} {_sha256(FILES_TSV)}\n"
    ):
        return
    for rel, _, digest in pinned_files():
        if any(rel.startswith(p) for p in EXCLUDED_PREFIXES):
            continue
        got = _sha256(root / rel)
        if got != digest:
            raise ValueError(f"D3SET {rel}: sha256 {got} != pinned {digest}")


# ─── Excerpts ────────────────────────────────────────────────────────────────


def _pcm(path: Path, info: Any) -> np.ndarray:
    """Memory-mapped ``(frames, channels)`` int16 view of a canonical PCM WAV."""
    with path.open("rb") as f:
        head = f.read(1 << 16)
    i = 12
    while i + 8 <= len(head):
        cid, size = head[i : i + 4], int.from_bytes(head[i + 4 : i + 8], "little")
        if cid == b"data":
            return np.memmap(
                path, dtype="<i2", mode="r", offset=i + 8, shape=(info.frames, info.channels)
            )
        i += 8 + size + (size & 1)
    raise ValueError(f"{path}: no data chunk in the first 64 KiB")


def _span_stats(excerpt: np.ndarray, parent: np.ndarray, offset: int) -> dict[str, int]:
    """PCM differences of ``excerpt`` against ``parent[offset : offset + len(excerpt)]``."""
    d = np.abs(
        np.asarray(parent[offset : offset + len(excerpt)], dtype=np.int32)
        - np.asarray(excerpt, dtype=np.int32)
    )
    return {
        "max_abs_diff_lsb": int(d.max()) if d.size else 0,
        "samples_differing": int(np.count_nonzero(d)),
        "samples_compared": int(d.size),
    }


def _locate(excerpt: np.ndarray, parent: np.ndarray, lo: int, hi: int, sr: int) -> int | None:
    """First offset in ``[lo, hi]`` where every excerpt sample is within
    :data:`EXCERPT_TOLERANCE_LSB` of the parent span, on every channel.

    Candidates come from a short probe at the loudest frame of the excerpt's
    first second (its loudest channel), so digital silence does not multiply
    them; each candidate is then compared in full.
    """
    n = len(excerpt)
    lo, hi = max(lo, 0), min(hi, len(parent) - n)
    if n == 0 or hi < lo:
        return None
    head = np.abs(np.asarray(excerpt[: min(n, sr)], dtype=np.int32))
    c = int(head.sum(axis=0).argmax())
    j = int(head[:, c].argmax())
    m = min(EXCERPT_PROBE, n - j)
    width = hi - lo + 1
    seg = np.asarray(parent[lo + j : hi + j + m, c], dtype=np.int32)
    probe = np.asarray(excerpt[j : j + m, c], dtype=np.int32)
    ok = np.ones(width, dtype=bool)
    for k in range(m):
        ok &= np.abs(seg[k : k + width] - probe[k]) <= EXCERPT_TOLERANCE_LSB
    for cand in np.flatnonzero(ok) + lo:
        if _span_stats(excerpt, parent, int(cand))["max_abs_diff_lsb"] <= EXCERPT_TOLERANCE_LSB:
            return int(cand)
    return None


@dataclass
class ExcerptMatch:
    """Where an excerpt folder's audio sits in its parent recording, measured per WAV."""

    rec: Recording
    parent: Recording
    ts_offset_ms: float | None  # excerpt StartTimestamp - parent StartTimestamp
    streams: dict[str, dict[str, Any]]  # wav -> location + PCM comparison
    offset_samples: int | None  # common offset of every WAV; None if unmatched/inconsistent
    tables_rows_in_parent: dict[
        str, bool
    ]  # excerpt CSV -> its rows appear in the parent CSV, in order

    @property
    def start_residual_ms(self) -> float | None:
        """StartTimestamp offset minus the measured audio offset (0 on a shared clock)."""
        if self.offset_samples is None or self.ts_offset_ms is None:
            return None
        return self.ts_offset_ms - 1000.0 * self.offset_samples / self.rec.ref.samplerate


def match_excerpt(root: Path, ex: Recording, parent: Recording) -> ExcerptMatch:
    """Locate each excerpt WAV in the parent's WAV of the same name (±3 s around
    the StartTimestamp difference, the whole parent if either has none)."""
    ts_offset = (
        ex.start_ms - parent.start_ms
        if ex.start_ms is not None and parent.start_ms is not None
        else None
    )
    streams: dict[str, dict[str, Any]] = {}
    for w, info in ex.wavs.items():
        pinfo = parent.wavs.get(w)
        if pinfo is None or (pinfo.channels, pinfo.samplerate) != (info.channels, info.samplerate):
            streams[w] = {"matched": False, "reason": "parent has no WAV with these channels/rate"}
            continue
        sr = info.samplerate
        if ts_offset is None:
            lo, hi = 0, pinfo.frames
        else:
            guess = round(ts_offset / 1000.0 * sr)
            lo, hi = guess - EXCERPT_SEARCH_S * sr, guess + EXCERPT_SEARCH_S * sr
        e, p = _pcm(root / ex.rel / w, info), _pcm(root / parent.rel / w, pinfo)
        off = _locate(e, p, lo, hi, sr)
        streams[w] = {
            "matched": off is not None,
            "parent_offset_samples": off,
            "search_window_samples": [max(lo, 0), min(hi, pinfo.frames - info.frames)],
            **(_span_stats(e, p, off) if off is not None else {}),
        }
    offsets = {s.get("parent_offset_samples") for s in streams.values()}
    matched = all(s["matched"] for s in streams.values())
    offset = next(iter(offsets)) if matched and len(offsets) == 1 else None
    rows = {
        n: _same_rows(root / ex.rel / n, root / parent.rel / n) for n in ex.csvs if n in parent.csvs
    }
    return ExcerptMatch(ex, parent, ts_offset, streams, offset, rows)


def _excerpt_meta(m: ExcerptMatch) -> dict[str, Any]:
    sr, n, off = m.rec.ref.samplerate, m.rec.ref.frames, m.offset_samples
    worst = max((s.get("max_abs_diff_lsb", 0) for s in m.streams.values()), default=0)
    return {
        "excerpt_id": m.rec.rel,
        "subcategory": m.rec.subcategory,
        "parent_id": m.parent.rel,
        "audio_entries": sorted(STREAMS[w][0] for w in m.rec.wavs),
        "parent_offset_samples": off,
        "num_samples": n,
        "sample_rate": sr,
        "parent_start_s": None if off is None else off / sr,
        "parent_end_s": None if off is None else (off + n) / sr,
        "pcm_comparison": m.streams,
        "pcm_relation": (
            "not located in the parent"
            if off is None
            else "bit-identical to the parent span"
            if worst == 0
            else f"re-quantised copy of the parent span: every sample within {worst} LSB, "
            "not bit-identical"
        ),
        "start_timestamp_unix_ms": m.rec.start_ms,
        "start_timestamp_offset_ms": m.ts_offset_ms,
        "start_timestamp_minus_audio_offset_ms": m.start_residual_ms,
        "tables_rows_in_parent": m.tables_rows_in_parent or None,
        "note": (
            "delivery excerpt: the delivery moment is not always synced with the real drop "
            "(motor-speed change); offset it or accept a window (README 'Delivery')"
            if m.rec.group == "Delivery"
            else "SNR-evaluation sample extracted from another flight (README 'SNR test')"
        ),
    }


# ─── Frame assembly ──────────────────────────────────────────────────────────


def _telemetry_on_audio_clock(rec: Recording, match: ExcerptMatch | None) -> bool:
    """Logger rows share the audio clock: the folder's StartTimestamp is the
    logged first drone.wav sample (not ``Other``), and an excerpt's
    StartTimestamp agrees with its measured parent offset within 1 ms."""
    if match is None:
        return rec.group != "Other"
    if match.parent.group == "Other":
        return False
    resid = match.start_residual_ms
    return resid is None or abs(resid) <= 1.0


def _audio(root: Path, rec: Recording, wav: str) -> tuple[td.Series, dict[str, Any]]:
    """Exact float32 audio of one WAV (int16 / 32768) + its stream description."""
    info = rec.wavs[wav]
    entry, placement = STREAMS[wav]
    pcm = _pcm(root / rec.rel / wav, info)
    x = np.asarray(pcm.T, dtype=np.float32, order="C")
    x *= np.float32(1.0 / 32768.0)
    sounding = np.flatnonzero(np.any(pcm != 0, axis=1))
    desc: dict[str, Any] = {
        "file": f"{rec.rel}/{wav}",
        "mic_placement": placement,
        "n_channels": info.channels,
        "sample_rate": info.samplerate,
        "num_samples": info.frames,
        "duration_s": info.frames / info.samplerate,
        "encoding": "float32 = int16 PCM / 32768 (exact)",
        "leading_digital_zero_samples": int(sounding[0]) if sounding.size else info.frames,
        "trailing_digital_zero_samples": (
            info.frames - 1 - int(sounding[-1]) if sounding.size else info.frames
        ),
    }
    if entry == "audio":
        desc["mic_pos"] = "top-level mic_pos (1, 3) m, estimated z only (meta.geometry.audio)"
        return td.uniform(x[0], info.samplerate, dims=("time",)), desc
    desc.update(
        dims=[ARRAY_DIM, "time"],
        mic_pos="not emitted: ring constraints only (meta.geometry.array_audio)",
        digital_zero_note=(
            "all-channel digital zeros at the start/end are publisher padding to the drone.wav "
            "length (README: the array is synced to the onboard audio, hence leading silence)"
        ),
        **_channel_flags(rec, info.channels),
    )
    return td.uniform(x, info.samplerate, dims=(ARRAY_DIM, "time")), desc


def _operating(rec: Recording) -> dict[str, Any]:
    if rec.rel == "Flights/SNR test/background noise":
        op: dict[str, Any] = {
            "condition": "rotors_off",
            "n_active_rotors": 0,
            "contains_rotor_noise": False,
            "external_source": "unknown",
            "confidence": "medium",
            "note": "array background excerpt; its actuator.csv holds 1000 us (disarmed) throughout",
        }
    elif rec.subcategory == "distance_test_hover":
        op = {
            **FLIGHT_OPERATING,
            "flight_mode": "hover",
            "note": "hover at several array distances up to 180 m (paper §III-A)",
        }
    else:
        op = {
            **FLIGHT_OPERATING,
            "flight_mode": "mixed",
            "note": "ground idle, take-off, manoeuvres and landing within one recording",
        }
    validate_operating(op, N_ROTORS, rec.rel)
    return op


def _channel_flags(rec: Recording, n_channels: int) -> dict[str, Any]:
    key = "index" if "index" in rec.additional else "mic_to_ignore"
    flagged = rec.additional.get(key)
    if flagged is None:
        return {}
    if n_channels == 16:
        how = "listed microphones are present in the file; ignore them (README)"
    elif n_channels == 16 - len(flagged):
        how = (
            "listed microphones were removed from the file (README); the remaining channel "
            "order and the list's index base (0/1) are undocumented"
        )
    else:
        how = f"inconsistent: {n_channels} channels vs {len(flagged)} listed of 16"
    return {
        "array_channels_flagged": list(flagged),
        "array_flag_key": key,
        "array_flag_meaning": how,
    }


def _clocked(t: Table, entry: str, t0: float, t1: float) -> td.Series | None:
    """Every timestamped row on the time axis; the axis spans at least ``[t0, t1)``."""
    assert t.t is not None
    if t.t.size == 0:
        return None
    channel = f"{entry}_channel"
    ev = td.events(
        t.t,
        t.values,
        dims=(channel, "time"),
        t_start=min(t0, float(t.t[0])),
        t_end=max(t1, float(t.t[-1]) + 1e-6),
    )
    return td.Series(
        ev.data, ev.dims, {channel: td.LabelIndex(tuple(t.columns)), "time": ev.tindex}
    )


def _rows(t: Table, entry: str) -> td.Series | td.Frame:
    """Values on their own row axis (no audio time); logger seconds beside them if stamped."""
    # td.Frame shares dimension names even across nested entries. Neither the
    # CSV columns nor the row counts are shared between tracks.
    channel, row = f"{entry}_channel", f"{entry}_row"
    rows = td.RangeIndex(t.values.shape[1])
    values = td.Series(
        t.values, (channel, row), {channel: td.LabelIndex(tuple(t.columns)), row: rows}
    )
    if t.t is None:
        return values
    return td.Frame({"values": values, "logger_seconds": td.Series(t.t, (row,), {row: rows})})


def _untimed(t: Table, entry: str) -> td.Frame:
    """Rows whose TIMESTAMP cell is N/A, with their 0-based CSV data-row numbers."""
    assert t.untimed is not None and t.untimed_rows is not None
    channel, row = f"{entry}_untimed_channel", f"{entry}_untimed_row"
    rows = td.RangeIndex(t.untimed_rows.size)
    return td.Frame(
        {
            "values": td.Series(
                t.untimed, (channel, row), {channel: td.LabelIndex(tuple(t.columns)), row: rows}
            ),
            "source_row": td.Series(t.untimed_rows.astype(np.int64), (row,), {row: rows}),
        }
    )


def _clock_note(on_clock: bool, match: ExcerptMatch | None) -> str:
    if on_clock:
        origin = (
            "the excerpt StartTimestamp (equal to the parent StartTimestamp plus the measured "
            "parent audio offset within 1 ms)"
            if match is not None
            else "StartTimestamp (the logged first drone.wav sample on the RPi clock)"
        )
        return (
            "t = 0 is the first audio sample. Telemetry rows: RPi receive time (Unix ms) minus "
            f"{origin}, / 1000; BPF-label actuator.csv files carry their own ms since audio start"
        )
    if match is not None:
        return (
            "UNALIGNED: excerpt of an 'Other' flight whose audio start was never logged; its "
            "StartTimestamp is "
            f"{match.start_residual_ms} ms off the measured parent audio offset "
            "(meta.derivative_of). Logger rows stay on row axes with logger_seconds relative to "
            "the excerpt StartTimestamp: not an audio clock."
        )
    return (
        "UNALIGNED: the audio start was not logged (README 'Other'); StartTimestamp is the "
        "first telemetry row. Logger rows stay on row axes with logger_seconds relative to it: "
        "not an audio clock (evidence: meta.audio_start_evidence)."
    )


def _audio_start_evidence(rec: Recording, excerpts: list[ExcerptMatch]) -> dict[str, Any] | None:
    """Evidence on where an ``Other`` flight's audio starts relative to its StartTimestamp."""
    if rec.group != "Other":
        return None
    implied = sorted(
        {round(m.start_residual_ms, 3) for m in excerpts if m.start_residual_ms is not None}
    )
    return {
        **OTHER_CLOCK_EVIDENCE[rec.rel],
        "publisher_excerpts_imply_audio_start_minus_start_timestamp_ms": implied or None,
        "publisher_excerpts_note": (
            "every excerpt StartTimestamp of this flight minus the parent StartTimestamp, minus "
            "the excerpt's measured audio offset in the parent: the publisher's own audio-start "
            "estimate, if their excerpt StartTimestamps mark the excerpt audio start"
            if implied
            else None
        ),
    }


def _frame(
    root: Path,
    rec: Recording,
    wavs: list[str],
    match: ExcerptMatch | None,
    excerpts: list[ExcerptMatch],
) -> td.Frame:
    """One Frame of the listed WAVs of ``rec`` (whole recordings); the logger
    tables ride with drone.wav (or with the array when the folder has no
    drone.wav)."""
    parent_rel = match.parent.rel if match is not None else rec.rel
    with_tables = "drone.wav" in wavs or "drone.wav" not in rec.wavs
    entries: dict[str, Any] = {}
    streams: dict[str, dict[str, Any]] = {}
    for wav in wavs:
        series, desc = _audio(root, rec, wav)
        entries[STREAMS[wav][0]] = series
        streams[STREAMS[wav][0]] = desc
    if "drone.wav" in wavs:
        entries["mic_pos"] = td.wrap(
            np.asarray(ONBOARD_MIC_POS, dtype=np.float64), dims=("mic", None)
        )
    duration = max(rec.wavs[w].frames / rec.wavs[w].samplerate for w in wavs)
    on_clock = _telemetry_on_audio_clock(rec, match)
    descs: dict[str, dict[str, Any]] = {}
    if with_tables:
        tel: dict[str, Any] = {}
        unal: dict[str, Any] = {}
        for n in rec.csvs:
            t = read_table(root, rec, n)
            d = describe(t)
            entry = d["name"]
            clocked = t.t is not None and on_clock
            s = _clocked(t, entry, 0.0, duration) if clocked else _rows(t, entry)
            if s is not None:
                (tel if clocked else unal)[entry] = s
            if t.untimed is not None:
                unal[f"{entry}_untimed"] = _untimed(t, entry)
            descs[entry] = {**d, "on_audio_clock": clocked}
        if tel:
            entries["telemetry"] = td.Frame(tel)
        if unal:
            entries["telemetry_unaligned"] = td.Frame(unal)
    placements = sorted({STREAMS[w][1] for w in wavs})
    observation: dict[str, Any] = {
        "mic_placement": "onboard_and_external" if len(placements) == 2 else placements[0],
        "audio_entries": [STREAMS[w][0] for w in wavs],
        "sample_rate": rec.wavs[wavs[0]].samplerate,
    }
    if "matrix.wav" in wavs:
        observation.update(
            array_position_wgs84=rec.additional.get("reference_position"),
            array_position_meaning="ground microphone-array GPS [lat deg, lon deg, alt m] "
            "(additional_data.json 'reference_position'); position reference only, array "
            "azimuth unaligned",
        )
    elif parent_rel.startswith("Other/"):
        observation.update(
            drone_first_gps_wgs84=rec.additional.get("reference_position"),
            drone_first_gps_meaning="additional_data.json 'reference_position' of an array-less "
            "flight = the drone's first GPS point, altitude relative (README 'Other'); NOT a "
            "recorder position",
        )
    label = {
        "subcategory": rec.subcategory,
        "split": rec.split,
        "role": "derivative" if match is not None else rec.role,
        "split_source": (
            "folder name (paper §III-B: 8 train / 3 validation / 6 test flights)"
            if rec.group in ("Flights/Training", "Flights/Validation", "Flights/Test")
            else "folder name"
        ),
    }
    if match is not None:
        label["parent_split"] = match.parent.split
    coregistered = "drone.wav" in rec.wavs and "matrix.wav" in rec.wavs
    misaligned = ARRAY_MISALIGNED.get(parent_rel)
    extra: dict[str, Any] = {
        "source_kind": "derivative" if match is not None else "source_recording",
        "group": rec.group,
        "audio_streams": streams,
        "recording_duration_s": duration,
        "start_timestamp_unix_ms": rec.start_ms,
        "start_timestamp_note": "RPi clock without internet: the Unix date need not match the "
        "flight date (README)",
        "clock": (
            _clock_note(on_clock, match)
            if with_tables
            else "the array WAV's own sample clock as published; NOT the drone.wav/telemetry "
            "clock (meta.array_sync)"
        ),
        "array_sync": (
            None
            if not coregistered
            else {"verdict": "published alignment refuted", **misaligned}
            if misaligned
            else {"verdict": "published alignment consistent", **ARRAY_SYNC_EVIDENCE[parent_rel]}
        ),
        "audio_start_evidence": _audio_start_evidence(rec, excerpts) if with_tables else None,
        "telemetry_tracks": descs or None,
        "excerpts": [_excerpt_meta(e) for e in excerpts] or None,
        "derivative_of": _excerpt_meta(match) if match is not None else None,
        "geometry": {
            STREAMS[w][0]: ONBOARD_MIC if w == "drone.wav" else ARRAY_GEOMETRY for w in wavs
        },
        "additional_data": rec.additional or None,
        "files": rec.files,
        "provenance": {
            "builder": "data_processing.sources.d3set.build",
            "git_commit": GIT_COMMIT,
            "license": PROVENANCE["license"],
            "citation": PROVENANCE["citation"],
        },
    }
    json.dumps(extra, allow_nan=False)
    entries["meta"] = meta_frame(
        rec.rel,
        DATASET,
        system=SYSTEM,
        observation=observation,
        operating=_operating(rec),
        label=label,
        extra=extra,
    )
    return td.Frame(entries)


def _frames_of(
    root: Path, rec: Recording, match: ExcerptMatch | None, excerpts: list[ExcerptMatch]
) -> Iterator[tuple[str, td.Frame]]:
    """``<folder>``: drone.wav, a co-registered array and every logger table;
    ``<folder>/matrix.wav``: an array whose published alignment is refuted."""
    parent_rel = match.parent.rel if match is not None else rec.rel
    wavs = [w for w in STREAMS if w in rec.wavs]
    apart = [
        w
        for w in wavs
        if w == "matrix.wav" and "drone.wav" in wavs and parent_rel in ARRAY_MISALIGNED
    ]
    yield rec.rel, _frame(root, rec, [w for w in wavs if w not in apart], match, excerpts)
    for w in apart:
        yield f"{rec.rel}/{w}", _frame(root, rec, [w], match, excerpts)


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """Yield ``(key, frame)``: one Frame per release folder, whole recordings.

    Originals first in release (sorted folder) order, each listing the
    excerpts located in it; then every ``Delivery``/``SNR test`` excerpt as its
    own derivative Frame (they are re-quantised, not bit-identical, copies, so
    the published files are kept and annotated with the measured parent span).
    """
    root = Path(raw_dir)
    _verify_hashes(root)
    recs = _scan(root)
    excerpt_ids = sorted(r for r, rec in recs.items() if rec.role == "excerpt")
    if excerpt_ids != sorted(EXCERPT_PARENTS):
        raise ValueError(
            "D3SET excerpt folders differ from EXCERPT_PARENTS: "
            f"{sorted(set(excerpt_ids) ^ set(EXCERPT_PARENTS))}"
        )
    originals = [r for r, rec in recs.items() if rec.role == "original"]
    if len(originals) != N_ORIGINALS:
        raise ValueError(f"D3SET: {len(originals)} original folders, expected {N_ORIGINALS}")
    pairs = {r for r in originals if {"drone.wav", "matrix.wav"} <= set(recs[r].wavs)}
    if pairs != set(ARRAY_SYNC_EVIDENCE) | set(ARRAY_MISALIGNED):
        raise ValueError(f"D3SET: onboard/array sync verdicts do not cover {sorted(pairs)}")
    if missing := sorted(
        {r for r in originals if r.startswith("Other/")} - set(OTHER_CLOCK_EVIDENCE)
    ):
        raise ValueError(f"D3SET: no audio-start evidence for {missing}")
    for rel in originals + excerpt_ids:
        if {"drone.wav", "matrix.wav"} <= set(recs[rel].wavs):
            a, b = recs[rel].wavs["drone.wav"], recs[rel].wavs["matrix.wav"]
            if (a.frames, a.samplerate) != (b.frames, b.samplerate):
                raise ValueError(f"D3SET {rel}: drone.wav and matrix.wav differ in length/rate")
    matches = {
        rel: match_excerpt(root, recs[rel], recs[EXCERPT_PARENTS[rel]]) for rel in excerpt_ids
    }
    by_parent: dict[str, list[ExcerptMatch]] = {}
    for m in matches.values():
        by_parent.setdefault(m.parent.rel, []).append(m)
    for rel in originals:
        yield from _frames_of(root, recs[rel], None, by_parent.get(rel, []))
    for rel in excerpt_ids:
        yield from _frames_of(root, recs[rel], matches[rel], [])
