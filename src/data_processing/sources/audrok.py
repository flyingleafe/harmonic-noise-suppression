"""AuDroK drone sound data (H2Think gGmbH, mFUND; Mobilithek offer 605778370199691264).

Built from the existing raw pin ``AuDroK`` (``raw-files``, all seven
Mobilithek distributions), never from a second download. Two disjoint
recording sets are in the public offer; the AuDroK SQL tables that label
each file (drone type, origin, distance, quality, ...) are NOT — the offer
says a database download must be requested from info@h2think.org — so
every label here comes from the file names plus the offer description and
the paper (Kümmritz & Paul, Forum Acusticum 2023, doi:10.61782/fa.2023.0049):

- ``Measurements_{1-19,20-39,40-59,60-79}/NN-PP-MiK.xml`` — 78 measurements
  x 7 flight phases x 6 channel files (3276). Each XML is a BASt
  ``TrafficDataService`` MDM container whose ``<ns3:binary>`` body is the
  base64 of one complete mono WAV (offer description: original file names
  kept, ``.wav`` replaced by ``.xml``); the container header (identifier,
  created/expires) is packaging metadata, never audio.
- ``Dresden-II-S-{A,B}/NN-DD-V-Dm-{Arr1,Arr2,Arr3,Mic1}.wav`` — the Dresden
  outdoor campaign (Fraunhofer IVI test field): per flight the sum signals
  of three 24-mic circular arrays (publisher derivative) and one
  single-microphone track.

``Angebotsbeschreibung.xml`` (the offer description, a Word XML package
whose content is five slide images) has no per-file data.

One Frame per simultaneous take, whole and unwindowed, native 44.1 kHz
float32, ``t_start`` 0:

- chamber ``NN-PP`` (546): ``audio`` ``(mic, time)`` = the six ``Mi1..Mi6``
  files of one measurement phase in label order. Exception ``04-LD``: the
  offset between ``Mi1..Mi3`` and ``Mi4..Mi6`` is unverified, so ``audio`` holds
  ``Mi1..Mi3`` and ``additional_audio`` ``(additional_mic, time)`` holds
  ``Mi4..Mi6`` on its own grid (``meta.clock_blocks``); both start at a
  file-local t = 0 that is NOT a shared clock;
- Dresden ``NN-DD-V[-Dm]`` (56): ``audio`` ``(time,)`` = ``Mic1``, the only
  physical microphone; ``array_sum_audio`` ``(array, time)`` = the three
  publisher array sums ``Arr1..Arr3`` on the same sample grid. A sum of 24
  capsules is not a microphone, so it is never in ``audio``/``mic_pos``.

A row is NaN outside its own file's samples: ``meta.channel_start_samples`` /
``channel_valid_samples`` (``additional_audio_*`` / ``array_sum_audio_*`` for
the other Series) give each row's ``[start, valid)`` span in samples from its
Series' t = 0. Only five chamber takes have unequal files; their placement,
its confidence (``channel_alignment_confidence``) and evidence are in
:data:`CHAMBER_UNEQUAL`. Nothing is joined across files: no adjacent chamber
phases and no consecutive Dresden flights pass the seam evidence
(``ANECHOIC["phase_files"]``, ``DRESDEN["flight_files"]``).
"""

from __future__ import annotations

import base64
import re
import struct
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.sources._common import meta_frame

DATASET = "AuDroK-frames"
RAW_DATASET = "AuDroK"
RAW_VERSION = "ea0e533e6ce4160cb67d9b57c252d569ae6c4865e2f2371e6a5e19975140c08f"

_MEAS = re.compile(r"^(?P<num>\d{2})-(?P<phase>TO|H1|FF|H2|FB|HF|LD)-Mi(?P<mic>[1-6])$")
_DRESDEN = re.compile(
    r"^(?P<take>(?P<num>\d{2})-(?P<drone>\d+)-(?P<speed>S|F|var)(?:-(?P<dist>\d+)m)?)"
    r"-(?P<chan>Arr[1-3]|Mic1)$"
)
_BINARY = re.compile(rb"<ns3:binary\b[^>]*>")
_HEADER = re.compile(r"<(?:ns\d+:)?(Identifier|Created|Expires)>([^<]*)<")

CHAMBER_CHANNELS = tuple(f"Mi{k}" for k in range(1, 7))
DRESDEN_MICS = ("Mic1",)
DRESDEN_SUMS = ("Arr1", "Arr2", "Arr3")

#: Phase code -> (meaning, flight_mode, subcategory part). Meanings are
#: inferred from the code letters + the paper's Fig. 4 procedure (the SQL
#: tables that would state them are not in the public offer).
PHASES: dict[str, tuple[str, str, str]] = {
    "TO": ("take-off from the launch platform", "manoeuvre", "takeoff"),
    "H1": ("hover near microphones 1 and 2", "hover", "hover"),
    "FF": ("fly forth across the room past microphone 3", "manoeuvre", "transit"),
    "H2": ("hover near microphones 4 and 5", "hover", "hover"),
    "FB": ("fly back past microphone 3", "manoeuvre", "transit"),
    "HF": ("final hover near microphones 1 and 2", "hover", "hover"),
    "LD": ("landing on the platform", "manoeuvre", "landing"),
}
_SPEED = {"S": "slow", "F": "fast", "var": "var"}

ANECHOIC = {
    "campaign": "AuDroK indoor campaign, anechoic chamber of the Institute of Acoustics and "
    "Speech Communication, TU Dresden (1000 m3) - paper Sec. 3.1.1",
    "attribution_evidence": "INFERRED: the 7-phase file naming matches the paper's anechoic "
    "procedure (Fig. 4); the offer does not label the Measurements zips by campaign",
    "microphones": "paper: 5 tripod microphones at ~1.5 m height - 1 weles WA101 PU "
    "intensity probe (pressure + particle velocity), 2 and 3 AKG C451B, 4 Behringer "
    "ECM8000, 5 BSWA MP253/MA231; 1&2 near one hover spot, 4&5 near the other (~20 m "
    "apart), 3 in between",
    "channel_map": "UNKNOWN: files Mi1..Mi6 = 6 channels vs 5 microphones (the PU probe "
    "gives two signals); which file is which transducer is not published. Rows of audio "
    "are Mi1..Mi6 in that order",
    "mic_pos": "see GEOMETRY_ANECHOIC: only the ~1.5 m height is published numerically",
    "take": "PROBE (uni Slurm 37570395/37571681/37571947/37572431, all 3276 files): every "
    "file is mono IEEE-float32 44.1 kHz with only fmt+data chunks, no timestamp/BWF "
    "chunk; all 546 measurement-phase groups have 6 channel files, and every file length "
    "is a multiple of 256 samples except the truncated 14-FB-Mi2. 541 groups have equal "
    "sample counts; 04-LD, 29-FF, 40-H2, 47-LD differ by 256-512 samples and 14-FB-Mi2 "
    "is a truncated prefix: their per-channel placement and its evidence are in "
    "CHAMBER_UNEQUAL (meta channel_alignment/_evidence). A group is one simultaneous "
    "multichannel take: whole-file GCC of every channel pair shows a common electrical "
    "zero-lag component and stable acoustic lags (e.g. Mi1-Mi2 ~0, Mi5-Mi6 -3..-7, "
    "Mi4-Mi5 ~-1190 samples)",
    "phase_files": "NOT JOINED: the seven phase files of a measurement are separately "
    "trimmed excerpts with no chronology beyond the phase name (the container 'Created' "
    "stamps, 2023-06-12/13, increase in alphabetical file-name order: packaging time), and "
    "their edges are not lossless consecutive slices. "
    "All 78 x 42 ordered same-measurement phase pairs were tested on all six channels, "
    "with the 2808 non-adjacent pairs as a control: (1) AR(32) one-step residual at the "
    "seam, normalised by the in-file scale; all six channels < 6 for 5/468 (1.1 %) "
    "adjacent vs 61/2808 (2.2 %) control pairs; median worst channel 43 vs 53. (2) "
    "data_processing.sources._common.clean_audio_seam on the six channels together: 153/468 "
    "(33 %) vs 837/2808 (30 %). (3) 64-sample AR free-run across the seam / in-file: worst "
    "channel < 2 for 18/468 (3.8 %) vs 215/2808 (7.7 %). All three together: 4/468 vs "
    "34/2808. Adjacent phases pass no more often than impossible pairs, so no seam is "
    "evidenced; no verbatim overlap in any pair",
    "coverage": "78 measurements (01..79, no 32 in the offer) x 7 phases x 6 files = 3276 "
    "XML; 14-FB-Mi2's base64 body holds 318895 of the 715008 samples its WAV header "
    "declares (truncated upstream): the valid prefix is kept and the rest of that row is NaN",
}
DRESDEN = {
    "campaign": "AuDroK outdoor campaign, Fraunhofer IVI test field, Dresden - paper Sec. 3.1.2",
    "arrays": "3 circular arrays of 24 Behringer microphones, 34 cm diameter; the database "
    "holds only the per-array SUM signal (paper: 'the database only includes sum signals of "
    "all microphones per array'). Published as array_sum_audio, never as a microphone",
    "mic1": "single microphone track; type and position not published",
    "name_tokens": "NN = measurement number; DD = drone index (matches the database's "
    "tb_dronetypes D-ID usage; 7 values = the 7 outdoor-measured UAVs of paper Tab. 3, but "
    "the ID->model mapping is not published); S/F = slow (~1-2 m/s) / fast (~4-5 m/s) "
    "flight (paper); var = undocumented variant (paper mentions extra payload and swapped "
    "propellers); Dm = the database Distance value in metres (its reference - lateral or "
    "altitude - is not published)",
    "take": "PROBE (uni Slurm 37570395): 224 mono IEEE-float32 44.1 kHz WAVs (fmt+data "
    "chunks only), 56 flights x {Arr1, Arr2, Arr3, Mic1}, equal sample counts within every "
    "flight (81.9-322.1 s, 2.86 h in total); Mic1 is acoustically coherent with Arr3 with "
    "time-varying lags (Slurm 37560802). One flight = one take",
    "flight_files": "NOT JOINED: no flight continues another. The same AR(32) seam test "
    "over all 3080 ordered flight pairs (4 channels each) passed none: consecutive "
    "numbers (52 pairs) have a worst-channel seam score of 13-324 (median 150), others "
    "20-389+. Measurement numbers 11, 30, 54, 55 are absent from the offer",
    "mic_pos": "see GEOMETRY_DRESDEN: array positions shown only on a satellite sketch "
    "(paper Fig. 5)",
}

#: Numeric geometry constraints the publications state. Unknown axes are NaN
#: rather than guessed.
GEOMETRY_ANECHOIC: dict[str, Any] = {
    "frame": "chamber frame: z up from the floor (m); x/y origin and axes unpublished (NaN)",
    "mic_pos": [None, None, 1.5],
    "status": "z = approximate tripod height ('approximate height of 1.5 meters', paper "
    "Sec. 3.1.1), assumed for all six channels; x, y unknown",
    "constraints": {
        "mic_height_m": 1.5,
        "hover_sites_separation_m": 20.0,
        "hover_sites": "hover site A (launch-platform side) near microphones 1 and 2, hover "
        "site B near microphones 4 and 5, microphone 3 between them on the flight path "
        "(paper Fig. 4, schematic, not to scale)",
        "along_track_x_m": {"mics_1_2": 0.0, "mic_3": "in (0, 20), drawn ~10", "mics_4_5": 20.0},
        "along_track_note": "x measured from hover site A along the A->B flight line; "
        "+-several m (schematic); cross-track position unknown",
        "flight_altitude_m": 3.0,
        "flight_altitude_note": "average altitude of the traversal (paper Sec. 3.1.1)",
        "pairs_in_close_proximity": [[1, 2], [4, 5]],
        "chamber": "TU Dresden IAS anechoic room, V = 1000 m3, lower cut-off 60 Hz "
        "(tu-dresden.de .../Spezialraeume/reflexionsarmer-raum); dimensions not published",
    },
    "channel_map": "UNKNOWN (Mi1..Mi6 vs the paper's microphones 1..5), so the along-track "
    "constraints cannot be attached to a channel",
    "signal_observations": "data probe, uni Slurm 37560802 (measurements 05/20/45/70 x "
    "H1/H2/FF/FB): GCC-PHAT peaks sit at 0 ms for nearly all channel pairs, even ones the "
    "paper puts ~20 m apart (common-mode electrical coupling), so TDOA cannot measure "
    "positions. Observations only, not geometry: Mi5-Mi6 constant -0.07..-0.11 ms lag in "
    "every phase (co-located or fixed-delay pair); Mi1-Mi2 0 ms; Mi1/Mi2-Mi3 lag "
    "0.27-0.57 ms varying with drone position (Mi3 >= 0.2 m from Mi1/Mi2); Mi4 gains "
    "30-60 dB below 150 Hz only in FF/FB (downwash at the fly-past point - possibly the "
    "paper's microphone 3 or the PU velocity channel, unresolved); H1 vs H2 levels do not "
    "separate the two hover-site clusters. Geometry frozen: no channel is assigned x/y.",
}
GEOMETRY_DRESDEN: dict[str, Any] = {
    "frame": "Mic1 position unidentified (all axes NaN); the Arr1/Arr2/Arr3 -> station "
    "mapping is not published; station estimates below are an unordered set",
    "mic_pos": [None, None, None],
    "array_sum": {
        "n_mics": 24,
        "layout": "circular (photo suggests two tiers of ~12 capsules, inference)",
        "diameter_m": 0.34,
        "effective_position": "ring centre (phase centre of an unweighted ring sum)",
        "sum": "publisher-side sum of the 24 microphone signals; individual channels "
        "are not distributed",
    },
    "single_mic": "type, position and height not published (a separate thin stand ~1 m from "
    "one array in the Fig. 5 photo is not identifiable as Mic1)",
    "station_estimates": {
        "method": "paper Fig. 5 (lower satellite panel) georeferenced to OpenStreetMap "
        "(ODbL) geometry of the Fraunhofer IVI double-loop test track (way 263666017 "
        "centreline; meadow islands ways 433168693/433168694): similarity transform from "
        "the two island centroids, cross-checked by loop radius and along-axis extent; "
        "scale 0.365 m/px (0.356-0.377), north-up within ~1 deg",
        "frame": "local ENU metres; origin = left meadow-island centroid, WGS84 51.026723 N, "
        "13.738363 E (OSM); z not determinable",
        "stations": {
            "N (north of track, by pavilion)": {
                "enu_m": [31.3, 23.6, None],
                "wgs84": [51.026936, 13.738810],
            },
            "L (left island)": {"enu_m": [1.3, -0.2, None], "wgs84": [51.026721, 13.738380]},
            "R (right island)": {"enu_m": [45.9, -5.6, None], "wgs84": [51.026672, 13.739018]},
        },
        "pavilion_enu_m": [17.6, 21.7],
        "pairwise_m": {"L-R": 45.0, "N-L": 38.4, "N-R": 32.6, "scale_uncertainty_m": 1.5},
        "uncertainty": "1-sigma ~4 m horizontal per station (hand-drawn circles ~6 m radius, "
        "scale +-3 %, OSM tracing ~2 m); figure-based estimate, not a survey",
        "assignment": "UNKNOWN: which of Arr1/Arr2/Arr3 sits at N/L/R is not published",
    },
    "signal_observations": "data probe, uni Slurm 37560802 (01-36-S-15m, 03-36-S-30m, "
    "12-43-S-15m): Arr1/2/3 pairs are zero-lag (electrical coupling), so no Arr->station "
    "mapping is identifiable. Mic1 is acoustically coherent with Arr3 only, with "
    "time-varying lags -0.2..-0.98 ms, so Mic1 stands next to Arr3 (|offset| >= 0.34 m, "
    "plausibly ~1 m; possibly the stand beside the pavilion-side array, i.e. Arr3 = N - "
    "low-medium confidence, not encoded). Arr3 carries ~6-9 dB more 150 Hz-8 kHz energy than "
    "Arr1/Arr2. Geometry frozen: Mic1 mic_pos stays NaN; array sums get no position.",
    "site": "three arrays and a base pavilion on the Fraunhofer IVI test field, Dresden",
}

PROVENANCE: dict[str, Any] = {
    "source_url": "https://mobilithek.info/offers/605778370199691264",
    "publisher": "H2Think gGmbH (mFUND project AuDroK), with Fraunhofer IVI",
    "raw_pin": f"dload:{RAW_DATASET}@{RAW_VERSION}",
    "license": "none granted",
    "rights": "Mobilithek offer metadata (fetched 2026-11-26): rights (Mobilithek field "
    "'Type of Terms of Use') = https://w3id.org/mdp/schema/license#NO_LICENSE, whose "
    "Mobilithek vocabulary label (/mdp-api/mdp-msa-metadata/v1/vocabs) is 'lizenzfrei' / "
    "'royalty-free'; standardLicense, customLicenseAsResource, conditionForUse and "
    "sourceNote are all null; accessRights PUBLIC; providerApprovalRequired false; "
    "contract 2023-10-26..2033-12-31. The Mobilithek General Terms of Use say data use is "
    "governed only by the terms the provider attaches to the offer - here none beyond the "
    "royalty-free category. So no stated condition restricts downloading or internally "
    "transforming the lawfully obtained files, but no licence to redistribute is granted "
    "either (rights in the recordings stay with H2Think). The paper's CC BY 3.0 covers the "
    "article only. Keep the frames internal; do not redistribute.",
    "attribution": "AUDROK Drohnen Sounddaten, H2Think gGmbH (mFUND project AuDroK, funded by "
    "the German Federal Ministry for Digital and Transport), Mobilithek offer 605778370199691264",
    "citation": "S. Kümmritz, L. Paul, Comprehensive Database of UAV Sounds for Machine "
    "Learning, Forum Acusticum 2023, doi:10.61782/fa.2023.0049",
    "excluded": "AuDroK SQL label tables (not in the public offer); the third-party/"
    "YouTube part of the database (not in the public offer)",
    "overlap": "geronimobasso/drone-audio-detection-samples (dload drone-detection-samples) "
    "lists AUDROK as its source #5, so it may hold 16 kHz mono trimmed clips derived from "
    "these files: keep both on the same side of any split.",
    "description": "AuDroK drone sound data as published on Mobilithek, one Frame per "
    "simultaneous take: 546 anechoic-chamber measurement phases (78 measurements x 7 "
    "phases, 6 synchronous channel files each, base64 WAV inside MDM XML containers) and "
    "56 Dresden outdoor flights (1 microphone + 3 publisher array-sum signals each); native "
    "44.1 kHz float32, whole files, no labels beyond the file names, since the AuDroK SQL "
    "tables are not in the public offer.",
    "anechoic": ANECHOIC,
    "dresden": DRESDEN,
}


#: Channel blocks of a chamber take: Series name -> channels on one sample
#: grid. A take whose cross-block offset is unknown keeps its channels in one
#: Frame but on separate grids, each starting at its own file-local t = 0.
SINGLE_BLOCK = {"audio": CHAMBER_CHANNELS}
#: Row dimension and geometry entry of each block Series.
BLOCK_DIMS = {
    "audio": ("mic", "mic_pos"),
    "additional_audio": ("additional_mic", "additional_mic_pos"),
}

#: The five chamber takes whose channel files differ in length: per channel
#: sample counts in the pinned raw (guard), the channel blocks, the start
#: sample of each channel on its block grid (others start at 0), and how that
#: placement was decided. Whole-file GCC-PHAT (uni Slurm 37571947/37572431) of
#: the full-band signals is compared with the same channel pair in
#: equal-length takes of the same phase; "prominence" = peak / median |GCC|,
#: noise floor ~6. The 150 Hz-8 kHz band carries a zero-lag electrical
#: coupling component that is largely periodic in 256 samples, so it cannot
#: decide whole-block shifts.
CHAMBER_UNEQUAL: dict[str, dict[str, Any]] = {
    "04-LD": {
        "samples": {
            "Mi1": 687616,
            "Mi2": 687616,
            "Mi3": 687872,
            "Mi4": 688128,
            "Mi5": 688128,
            "Mi6": 688128,
        },
        "blocks": {"audio": ("Mi1", "Mi2", "Mi3"), "additional_audio": ("Mi4", "Mi5", "Mi6")},
        "starts": {},
        "confidence": "within-block verified; cross-block offset unverified",
        "status": "two clock blocks: audio = Mi1-Mi3, additional_audio = Mi4-Mi6. Both "
        "Series start at their own file-local t = 0; the equal zeros are NOT phase sync. "
        "Mi4-Mi6 vs Mi1/Mi2 offset unknown among 0, 256 and 512 samples",
        "evidence": "Mi3 (+256 samples) starts with Mi1/Mi2: Mi1-Mi3 / Mi2-Mi3 lag 0/+3 "
        "(prominence 53/66) with a common start vs +256/+259 otherwise; 1 s windows give "
        "+20/+18 vs the 76-take LD reference median +20.5/+19, so Mi1/Mi2 are NaN over Mi3's "
        "last 256 samples. Mi4-Mi6 (equal length, +512 vs Mi1/Mi2) vs Mi1-Mi3 is NOT "
        "established: every cross pair peaks near the noise floor (prominence 6-9); the "
        "strongest, Mi2-Mi6/Mi2-Mi5 -1466/-1462 with a common start, fall inside the LD "
        "references (median -1484/-1562, 13/4 takes) and the 256/512-sample alternatives "
        "(-1210/-1206, -954/-950) outside, which is too weak to place the block",
    },
    "14-FB": {
        "samples": {
            "Mi1": 715008,
            "Mi2": 318895,
            "Mi3": 715008,
            "Mi4": 715008,
            "Mi5": 715008,
            "Mi6": 715008,
        },
        "blocks": SINGLE_BLOCK,
        "starts": {},
        "confidence": "verified",
        "status": "Mi2 truncated upstream; NaN past its 318895 present samples",
        "evidence": "Mi2 is the truncated prefix of a 715008-sample file (header-declared): "
        "Mi1-Mi2 / Mi2-Mi3 peak at -3/0 (prominence 68/58) with a common start, no peak "
        "(<= 6) when Mi2 is placed at the end",
    },
    "29-FF": {
        "samples": {
            "Mi1": 517632,
            "Mi2": 517632,
            "Mi3": 517632,
            "Mi4": 517888,
            "Mi5": 517888,
            "Mi6": 517888,
        },
        "starts": {"Mi1": 256, "Mi2": 256, "Mi3": 256},
        "blocks": SINGLE_BLOCK,
        "confidence": "inferred",
        "status": "Mi1-Mi3 start 256 samples after Mi4-Mi6 (common end)",
        "evidence": "Mi4-Mi6 carry 256 extra LEADING samples (common end): in the last 1 s "
        "Mi1-Mi4 / Mi3-Mi4 lag +1251/+1239 (coherence 0.06) with a common end vs +995/+983 "
        "with a common start; equal-length references Mi1-Mi4 +1263 (FF), +1255 (FB/HF, 53 "
        "takes), Mi3-Mi4 +1243..+1252. Whole-file full-band peaks are at the noise floor",
    },
    "40-H2": {
        "samples": {
            "Mi1": 343808,
            "Mi2": 343808,
            "Mi3": 343808,
            "Mi4": 343808,
            "Mi5": 344064,
            "Mi6": 344064,
        },
        "starts": {"Mi1": 256, "Mi2": 256, "Mi3": 256, "Mi4": 256},
        "blocks": SINGLE_BLOCK,
        "confidence": "verified",
        "status": "Mi1-Mi4 start 256 samples after Mi5/Mi6 (common end); Mi1-Mi3 follow Mi4 "
        "(equal length)",
        "evidence": "Mi5/Mi6 carry 256 extra LEADING samples (common end): Mi4-Mi5 / Mi4-Mi6 "
        "whole-file peaks (prominence 107/150) at -1181/-1184 with a common end vs "
        "-1437/-1440 with a common start; equal-length H2 references -1189 (median of 66 "
        "takes, IQR -1197..-1107), 39-H2 -1187/-1190",
    },
    "47-LD": {
        "samples": {
            "Mi1": 595456,
            "Mi2": 595456,
            "Mi3": 595456,
            "Mi4": 595712,
            "Mi5": 595712,
            "Mi6": 595712,
        },
        "starts": {"Mi1": 256, "Mi2": 256, "Mi3": 256},
        "blocks": SINGLE_BLOCK,
        "confidence": "inferred",
        "status": "Mi1-Mi3 start 256 samples after Mi4-Mi6 (common end)",
        "evidence": "Mi4-Mi6 carry 256 extra LEADING samples (common end): all six "
        "Mi{1,2,3} x Mi{4,6} whole-file peaks (prominence 15-21) move 256 samples towards the "
        "equal-length LD references, e.g. Mi1-Mi4 -446 vs -702 (references median -251, "
        "10-90 % -455..-37, 20 takes), Mi1-Mi6 -1642 vs -1898 (median -1438, 10-90 % "
        "-1621..-1280, 13 takes); still at the edge of those distributions",
    },
}


def chamber_alignment(take: str, samples: list[int]) -> dict[str, Any]:
    """Channel placement of one chamber take (``samples`` in ``Mi1..Mi6`` order)."""
    if take in CHAMBER_UNEQUAL:
        entry = CHAMBER_UNEQUAL[take]
        if [entry["samples"][c] for c in CHAMBER_CHANNELS] != samples:
            raise ValueError(
                f"AuDroK {take}: channel samples {samples} != audited {entry['samples']}"
            )
        return entry
    if len(set(samples)) != 1:
        raise ValueError(f"AuDroK {take}: unaudited unequal channel samples {samples}")
    return {
        "blocks": SINGLE_BLOCK,
        "starts": {},
        "confidence": "verified",
        "status": "equal sample counts",
        "evidence": "all six files hold the same number of samples; common start and end",
    }


def wav_layout(head: bytes, total_len: int | None = None) -> dict[str, Any]:
    """RIFF chunk walk over a WAV's leading bytes: format, data offset, declared
    vs present ``data`` bytes (``total_len`` = full file length when ``head``
    is only a prefix)."""
    total = len(head) if total_len is None else int(total_len)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE file")
    pos, chunks, fmt = 12, [], None
    while pos + 8 <= len(head):
        cid, size = (
            head[pos : pos + 4].decode("latin1"),
            struct.unpack("<I", head[pos + 4 : pos + 8])[0],
        )
        chunks.append(cid)
        if cid == "fmt ":
            tag, ch, sr, _, align, bits = struct.unpack("<HHIIHH", head[pos + 8 : pos + 24])
            fmt = {
                "format_tag": tag,
                "channels": ch,
                "sample_rate": sr,
                "block_align": align,
                "bits": bits,
            }
        if cid == "data":
            if fmt is None:
                raise ValueError("WAV data chunk precedes fmt")
            present = total - (pos + 8)
            return {
                "chunks": chunks,
                **fmt,
                "data_offset": pos + 8,
                "declared_data_bytes": size,
                "present_data_bytes": min(size, present),
                "truncated": present < size,
            }
        pos += 8 + size + (size & 1)
    raise ValueError("WAV has no data chunk in its leading bytes")


def _samples(layout: dict[str, Any], where: str) -> int:
    """Whole mono IEEE-float32 frames present (every AuDroK file is this format)."""
    if (layout["format_tag"], layout["channels"], layout["bits"], layout["block_align"]) != (
        3,
        1,
        32,
        4,
    ):
        raise ValueError(f"{where}: expected mono IEEE float32 WAV, got {layout}")
    return layout["present_data_bytes"] // 4


def read_container(path: Path) -> tuple[np.ndarray, dict[str, str], dict[str, Any]]:
    """MDM XML container -> (present float32 samples, container header, WAV layout)."""
    raw = Path(path).read_bytes()
    m = _BINARY.search(raw)
    end = raw.rfind(b"</ns3:binary>")
    if m is None or end < m.end():
        raise ValueError(f"{path}: no <ns3:binary> body")
    header = dict(_HEADER.findall(raw[: m.start()].decode("utf-8")))
    wav = base64.b64decode(raw[m.end() : end])
    del raw
    layout = wav_layout(wav)
    n = _samples(layout, str(path))
    return np.frombuffer(wav, dtype="<f4", count=n, offset=layout["data_offset"]), header, layout


def _read_wav(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with open(path, "rb") as fh:
        head = fh.read(4096)
    layout = wav_layout(head, path.stat().st_size)
    n = _samples(layout, str(path))
    return np.fromfile(path, dtype="<f4", count=n, offset=layout["data_offset"]), layout


def _stack(rows: list[np.ndarray], starts: list[int]) -> tuple[np.ndarray, list[int]]:
    """``(C, T)`` float32 on the take grid, row ``c`` placed at ``starts[c]``;
    NaN outside a row's own samples (never zero-filled). Returns the data and
    each row's exclusive valid end (``start + samples``) on that grid."""
    ends = [s + int(r.size) for s, r in zip(starts, rows, strict=True)]
    out = np.full((len(rows), max(ends)), np.nan, dtype=np.float32)
    for c, (s, r) in enumerate(zip(starts, rows, strict=True)):
        out[c, s : s + r.size] = r
    return out, ends


def _source_file(relpath: str, layout: dict[str, Any], **extra: Any) -> dict[str, Any]:
    return {
        "relpath": relpath,
        "wav_chunks": layout["chunks"],
        "declared_samples": layout["declared_data_bytes"] // 4,
        "present_samples": layout["present_data_bytes"] // 4,
        "truncated": layout["truncated"],
        **extra,
    }


def _take_meta(starts: list[int], ends: list[int], prefix: str = "channel") -> dict[str, Any]:
    """Valid span ``[start, valid)`` of every row on the take grid (samples from t = 0)."""
    complete = set(starts) == {0} and len(set(ends)) == 1
    return {
        f"{prefix}_start_samples": starts,
        f"{prefix}_valid_samples": ends,
        f"{prefix}_complete": complete,
        f"{prefix}_missing_samples": None
        if complete
        else "NaN outside each row's [start, valid) span",
    }


def chamber_blocks(
    take: str, rows: dict[str, np.ndarray], sr: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Mi1..Mi6 samples of one chamber take -> (Frame entries, meta fields).

    One ``(row, time)`` Series per clock block of :func:`chamber_alignment`
    (``audio`` = ``mic`` rows; an unplaceable block becomes ``additional_audio``
    = ``additional_mic`` rows), each with its NaN geometry rows and its own
    ``<prefix>_labels/_start_samples/_valid_samples`` meta (``channel`` for
    ``audio``, the Series name otherwise)."""
    alignment = chamber_alignment(take, [int(rows[c].size) for c in CHAMBER_CHANNELS])
    entries: dict[str, Any] = {}
    fields: dict[str, Any] = {
        "channel_alignment": alignment["status"],
        "channel_alignment_confidence": alignment["confidence"],
        "channel_alignment_evidence": alignment["evidence"],
        "clock_blocks": list(alignment["blocks"]),
    }
    z = GEOMETRY_ANECHOIC["mic_pos"]
    for name, chans in alignment["blocks"].items():
        dim, pos_name = BLOCK_DIMS[name]
        prefix = "channel" if name == "audio" else name
        starts = [alignment["starts"].get(c, 0) for c in chans]
        data, ends = _stack([rows[c] for c in chans], starts)
        entries[name] = td.uniform(data, sr, dims=(dim, "time"), t_start=0.0)
        pos = np.array([[np.nan if v is None else v for v in z]] * len(chans))
        entries[pos_name] = td.wrap(pos, dims=(dim, None))
        fields[f"{prefix}_labels"] = list(chans)
        if name == "audio":
            fields["take_samples"] = int(data.shape[1])
            fields["duration_s"] = data.shape[1] / sr
        else:
            fields[f"{name}_samples"] = int(data.shape[1])
        fields.update(_take_meta(starts, ends, prefix))
    return entries, fields


def _chamber_take(root: Path, take: str, paths: list[Path]) -> tuple[str, td.Frame]:
    num, phase = take.split("-")
    meaning, mode, sub = PHASES[phase]
    rows, files, rates = {}, [], set()
    for chan, path in zip(CHAMBER_CHANNELS, paths, strict=True):
        x, header, layout = read_container(path)
        rates.add(layout["sample_rate"])
        rows[chan] = x
        files.append(
            _source_file(
                str(path.relative_to(root)),
                layout,
                container={**header, "format": "BASt TrafficDataService MDM XML, base64 WAV body"},
            )
        )
    if len(rates) != 1:
        raise ValueError(f"AuDroK {take}: channel sample rates {sorted(rates)}")
    sr = rates.pop()
    entries, fields = chamber_blocks(take, rows, sr)
    del rows
    meta = meta_frame(
        take,
        DATASET,
        system={"category": "drone", "model": "unknown (AuDroK SQL tables not public)"},
        observation={
            "mic_placement": "external",
            "type": "tripod_microphones_static",
            "environment": "anechoic chamber (inferred, see provenance.anechoic)",
            "n_channels": int(entries["audio"].data.shape[0]),
            "sample_rate": sr,
            "channel_map": ANECHOIC["channel_map"],
        },
        operating={
            "condition": "flight",
            "flight_mode": mode,
            "phase_code": phase,
            "phase": meaning,
            "phase_meaning_source": "inferred from code + paper Fig. 4",
            "contains_rotor_noise": True,
            "external_source": "none",
            "confidence": "medium",
        },
        label={
            "subcategory": f"anechoic_{sub}",
            "content_class": "A (single UAV free-field), inferred",
            "recording_kind": "source_recording",
        },
        extra={
            "campaign": "anechoic",
            "measurement": int(num),
            "raw_relpaths": [f["relpath"] for f in files],
            **fields,
            "source": {
                "distribution": sorted({p.parent.name for p in paths}),
                "files": files,
                "native_format": "WAV IEEE float32 mono per channel; stored as float32",
                "take": ANECHOIC["take"],
                "phase_files": ANECHOIC["phase_files"],
            },
            "geometry": GEOMETRY_ANECHOIC,
            "attribution": PROVENANCE["attribution"],
            "license": PROVENANCE["license"],
        },
    )
    return take, td.Frame({**entries, "meta": meta})


def _dresden_take(root: Path, take: str, paths: dict[str, Path]) -> tuple[str, td.Frame]:
    m = _DRESDEN.match(f"{take}-Mic1")
    assert m is not None
    order = DRESDEN_MICS + DRESDEN_SUMS
    rows, files, rates = [], [], set()
    for chan in order:
        x, layout = _read_wav(paths[chan])
        rates.add(layout["sample_rate"])
        rows.append(x)
        files.append(_source_file(str(paths[chan].relative_to(root)), layout, channel=chan))
    if len(rates) != 1:
        raise ValueError(f"AuDroK {take}: channel sample rates {sorted(rates)}")
    sr = rates.pop()
    if len({int(r.size) for r in rows}) != 1:
        raise ValueError(f"AuDroK {take}: unequal channel lengths {[r.size for r in rows]}")
    data, ends = _stack(rows, [0] * len(rows))
    del rows
    n_mic = len(DRESDEN_MICS)
    speed = _SPEED[m["speed"]]
    meta = meta_frame(
        take,
        DATASET,
        system={"category": "drone", "drone_index": int(m["drone"]), "model": "unknown"},
        observation={
            "mic_placement": "external",
            "type": "ground_microphone_and_array_sums_static",
            "environment": "outdoor test field, Fraunhofer IVI Dresden",
            "n_channels": len(DRESDEN_MICS),
            "sample_rate": sr,
            "array_sum": "array_sum_audio rows = publisher sums of 24 microphones per array; "
            "not microphones",
            "distance_token_m": None if m["dist"] is None else int(m["dist"]),
        },
        operating={
            "condition": "flight",
            "flight_mode": "mixed",
            "speed": speed,
            "contains_rotor_noise": True,
            "external_source": "unknown",
            "confidence": "medium",
        },
        label={
            "subcategory": f"outdoor_{speed}",
            "content_class": "B/C (single UAV outdoor), inferred",
            "recording_kind": "source_recording",
            "array_sum_audio": "derivative: publisher sum of the 24 array microphones",
        },
        extra={
            "campaign": "dresden",
            "measurement": int(m["num"]),
            "raw_relpaths": [f["relpath"] for f in files],
            "take_samples": int(data.shape[1]),
            "duration_s": data.shape[1] / sr,
            "channel_labels": list(DRESDEN_MICS),
            **_take_meta([0] * n_mic, ends[:n_mic]),
            "array_sum_audio_labels": list(DRESDEN_SUMS),
            **_take_meta([0] * len(DRESDEN_SUMS), ends[n_mic:], prefix="array_sum_audio"),
            "source": {
                "distribution": sorted({p.parent.name for p in paths.values()}),
                "files": files,
                "native_format": "WAV IEEE float32 mono per channel; stored as float32",
                "take": DRESDEN["take"],
                "flight_files": DRESDEN["flight_files"],
            },
            "geometry": GEOMETRY_DRESDEN,
            "attribution": PROVENANCE["attribution"],
            "license": PROVENANCE["license"],
        },
    )
    pos = np.full((n_mic, 3), np.nan)
    return take, td.Frame(
        {
            "audio": td.uniform(data[0], sr, dims=("time",), t_start=0.0),
            "array_sum_audio": td.uniform(data[n_mic:], sr, dims=("array", "time"), t_start=0.0),
            "mic_pos": td.wrap(pos, dims=("mic", None)),
            "meta": meta,
        }
    )


def build(root: Path) -> Iterator[tuple[str, td.Frame]]:
    """Every simultaneous take of the pinned raw tree: 546 chamber phases, 56
    Dresden flights."""
    root = Path(root)
    chamber: dict[str, dict[str, Path]] = {}
    for path in sorted(root.glob("Measurements_*/*.xml")):
        m = _MEAS.match(path.stem)
        if m is None:
            raise ValueError(f"AuDroK: unparsed measurement file {path.name!r}")
        take = chamber.setdefault(f"{m['num']}-{m['phase']}", {})
        if f"Mi{m['mic']}" in take:
            raise ValueError(f"AuDroK: duplicate {path.stem}")
        take[f"Mi{m['mic']}"] = path
    dresden: dict[str, dict[str, Path]] = {}
    for path in sorted(root.glob("Dresden-II-S-*/*.wav")):
        m = _DRESDEN.match(path.stem)
        if m is None:
            raise ValueError(f"AuDroK: unparsed Dresden file {path.name!r}")
        take = dresden.setdefault(m["take"], {})
        if m["chan"] in take:
            raise ValueError(f"AuDroK: duplicate {path.stem}")
        take[m["chan"]] = path
    if not chamber or not dresden:
        raise FileNotFoundError(f"AuDroK: Measurements_*/ or Dresden-II-S-*/ missing under {root}")
    if set(chamber) & set(dresden):
        raise ValueError("AuDroK: take ids collide across campaigns")
    for take, files in chamber.items():
        if set(files) != set(CHAMBER_CHANNELS):
            raise ValueError(f"AuDroK {take}: channel files {sorted(files)}")
    for take, files in dresden.items():
        if set(files) != set(DRESDEN_MICS + DRESDEN_SUMS):
            raise ValueError(f"AuDroK {take}: channel files {sorted(files)}")
    for take in sorted(chamber):
        yield _chamber_take(root, take, [chamber[take][c] for c in CHAMBER_CHANNELS])
    for take in sorted(dresden):
        yield _dresden_take(root, take, dresden[take])
