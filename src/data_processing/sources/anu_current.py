"""ANU motor current-specific drone noise (Zenodo 16196947).

Manamperi & Abhayapala's deposit for "Drone Audition: Sound Source
Localization Using On-Board Microphones" (IEEE/ACM TASLP 30, 2022,
doi:10.1109/TASLP.2022.3140550): eleven recordings of the drone noise picked
up by microphones on the drone body, each at one constant motor current
(``Recording_<I>mA.mat``, 110-1020 mA), plus ``Room_noise.mat`` with the
motors off. Each ``.mat`` (MATLAB 5, zlib) holds exactly one variable,
``Recording``: a float64 ``(samples, 32)`` matrix; no rate, no channel map
and no current trace are stored. The channel map and positions come from the
authors' GitHub readme (:data:`MIC_TABLE`, ``mic_pos`` rows 7-8 NaN).

The motor current is constant per recording (bench power supply in
constant-current mode, the authors' APSIPA 2020 protocol), so it is a
recording annotation (``meta.operating.motor_current_mA``), not a time series.
Each recording is one Frame keyed by its file stem, the whole deposited
``(32, samples)`` matrix on the recording's own clock (``t_start`` = 0). The
float64 values are int32 codes / 2**31 with <= 24 significant bits, so they
are held as float32 exactly; the builder refuses a file where that cast loses
a bit.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import tdseries as td

DATASET = "ANU-drone-motor-current"
DOI = "10.5281/zenodo.16196947"
SAMPLE_RATE = 48000

#: Zenodo 16196947 (published 2022-01-05): file -> (bytes, md5).
FILES: dict[str, tuple[int, str]] = {
    "Room_noise.mat": (39276137, "27bed1a3c74936f451f789fba5d67683"),
    "Recording_110mA.mat": (53284720, "8ed829bd14d338e3ba7a5f4357eda1ff"),
    "Recording_200mA.mat": (54531441, "54e7178db36799f9eff1b94d51f5b782"),
    "Recording_320mA.mat": (55110831, "7b50121ccb0fbefb07a397e06ef6d719"),
    "Recording_410mA.mat": (55387513, "edd43db61b28a2abfcfa73906bd2e839"),
    "Recording_500mA.mat": (55554338, "e191096d612e8771cb298f82fd5e9ff1"),
    "Recording_600mA.mat": (55651252, "465c13fe332355ae4a27dc71bc967202"),
    "Recording_720mA.mat": (55737417, "d5f5a3efa321d6b6b6c4ca061d275918"),
    "Recording_810mA.mat": (55890757, "fe96bcc417ffcfdb6e12153d25126e2b"),
    "Recording_900mA.mat": (55968944, "65e8430ba6227e7ac4c6ab18fd60e679"),
    "Recording_990mA.mat": (56130400, "b3f161d49e15fa57ed93b512278aea73"),
    "Recording_1020mA.mat": (56149240, "79ccc05b251a38cbfdc5601477758ce1"),
}

SAMPLE_RATE_EVIDENCE = (
    "inferred, not stored in the .mat files: the authors' earlier APSIPA 2020 protocol "
    "used 48 kHz for a 10 s recording, consistent with 480000 samples here; "
    "no independent rate measurement for this deposit"
)

#: Authors' readme (github.com/wnilmini/DroneAcousticDataset ``Readme.pdf``,
#: "Readme file for drone noise recordings taken in 08/08", W. Manamperi),
#: Table 1: MEMS microphone positions on the drone, Cartesian (x, y, z) in
#: metres at 0.01 m resolution, indexed by audio-interface channel (1-based).
#: Channels 7 and 8 are absent from the table (no microphone) -> NaN rows.
MIC_TABLE: dict[int, tuple[float, float, float]] = {
    1: (-0.12, 0.0, -0.02),
    2: (-0.11, 0.0, -0.02),
    3: (-0.10, 0.0, -0.01),
    4: (-0.09, 0.0, -0.01),
    5: (-0.07, 0.0, -0.09),
    6: (-0.07, 0.0, -0.10),
    9: (0.12, 0.0, -0.02),
    10: (0.11, 0.0, -0.02),
    11: (0.10, 0.0, -0.01),
    12: (0.09, 0.0, -0.01),
    13: (0.07, 0.0, -0.09),
    14: (0.07, 0.0, -0.10),
    15: (0.0, -0.12, -0.02),
    16: (0.0, -0.11, -0.02),
    17: (0.0, -0.10, -0.01),
    18: (0.0, -0.09, -0.01),
    19: (0.0, -0.07, -0.09),
    20: (0.0, -0.07, -0.10),
    21: (0.0, 0.12, -0.02),
    22: (0.0, 0.11, -0.02),
    23: (0.0, 0.10, -0.01),
    24: (0.0, 0.09, -0.01),
    25: (0.0, 0.07, -0.09),
    26: (0.0, 0.07, -0.10),
    27: (0.04, -0.04, 0.0),
    28: (0.04, -0.04, -0.01),
    29: (-0.04, 0.04, 0.0),
    30: (-0.04, 0.04, -0.01),
    31: (0.0, 0.0, 0.0),
    32: (0.0, 0.0, -0.01),
}

#: Readme channel groups (BB/FB/BA/FA = the four motor arms as the authors name them).
CHANNEL_ROLES = {
    "under_motor_BB": [1, 2],
    "side_of_motor_BB": [3, 4],
    "landing_gear_BB": [5, 6],
    "no_microphone": [7, 8],
    "under_motor_FB": [9, 10],
    "side_of_motor_FB": [11, 12],
    "landing_gear_FB": [13, 14],
    "under_motor_BA": [15, 16],
    "side_of_motor_BA": [17, 18],
    "landing_gear_BA": [19, 20],
    "under_motor_FA": [21, 22],
    "side_of_motor_FA": [23, 24],
    "landing_gear_FA": [25, 26],
    "back": [27, 28],
    "front": [29, 30],
    "top": [31, 32],
}


def mic_positions(n_channels: int) -> np.ndarray:
    """``(n_channels, 3)`` metres from :data:`MIC_TABLE`; NaN where no mic is listed."""
    pos = np.full((n_channels, 3), np.nan)
    for ch, xyz in MIC_TABLE.items():
        pos[ch - 1] = xyz
    return pos


GEOMETRY = {
    "mic_pos": "positions as tabulated by the authors (Readme.pdf Table 1; measured vs "
    "nominal not stated), 0.01 m resolution; rows 7-8 NaN (channels absent from the table)",
    "coordinate_frame": "authors' Table 1 frame as-is, metres: origin at the 'Top' mic "
    "(ch31), z up (landing-gear mics at z -0.09/-0.10), arm mics on -x (BB), +x (FB), -y "
    "(BA), +y (FA); heading (which arm is the drone front) and handedness unresolved "
    "(the 'Back'/'Front' modules sit at (+0.04,-0.04)/(-0.04,+0.04)); 0.01 m quantisation",
    "channel_roles": CHANNEL_ROLES,
    "evidence": "https://raw.githubusercontent.com/wnilmini/DroneAcousticDataset/HEAD/Readme.pdf "
    "(repo README: 'the readme file here contains the microphone array configuration on "
    "the drone' for the TASLP 2022 study, listing both Zenodo 16197912 and 16196947); "
    "interface 'STWaves Audio Interface 8x32', MEMS dual-mic modules; consistency: the "
    "group's 2025 TASLP DRTF paper (doi:10.1109/TASLPRO.2025.3562006) uses 15 dual-mic "
    "modules 'similar to our previous experiment' (= 30 populated channels), arm radius 0.12 m",
    "channel_map_check": {
        "files": [
            "Room_noise.mat",
            "Recording_110mA.mat",
            "Recording_500mA.mat",
            "Recording_1020mA.mat",
        ],
        "unused_inputs": "channels 7 and 8 are not zero-filled: both measure -186.6 dBFS "
        "RMS and correlate at 1.00 (rounded) in all four checked files. No acoustic signal "
        "is evident; the electrical/digital origin of the residual is unknown. Audio columns "
        "are kept unchanged in order, with NaN position rows.",
        "room_noise_mutual_best_pairs": [
            [1, 2],
            [3, 4],
            [5, 6],
            [9, 10],
            [11, 12],
            [13, 14],
            [15, 16],
            [19, 20],
            [25, 26],
            [27, 28],
            [29, 30],
            [31, 32],
        ],
        "method": "mutual strongest absolute zero-lag correlation over the first 240000 "
        "samples, bandpass 500-4000 Hz at assumed 48 kHz; uni-cpu job 37560802, "
        "replicated on Hetzner",
        "nearby_modules": "the remaining active pairs (17-18, 21-22, 23-24) are not "
        "mutual best partners in the room recording. In powered runs close cross-module "
        "channels correlate strongly, e.g. 2/3 (r=0.71 at 1020 mA), 15/17 (r=0.88 at "
        "500 mA), and 22/24 (r=0.78 at 1020 mA), consistent with the tabulated layout.",
    },
    "caveat": "the readme is titled for the '08/08' session. The channel checks support "
    "transferring its numbered map to the motor-current deposit but do not independently "
    "establish absolute positions, within-module polarity/orientation or heading.",
}

PROVENANCE = {
    "source_url": "https://zenodo.org/records/16196947",
    "doi": DOI,
    "version": "single version (published 2022-01-05, record created 2025-07-20)",
    "license": "CC BY 4.0",
    "rights_note": "Zenodo record licence CC-BY-4.0; attribute Manamperi & Abhayapala.",
    "citation": "W. N. Manamperi, T. D. Abhayapala, J. Zhang, P. N. Samarasinghe, 'Drone "
    "Audition: Sound Source Localization Using On-Board Microphones', IEEE/ACM TASLP 30 "
    "(2022) 508-519, doi:10.1109/TASLP.2022.3140550; data: Motor Current-Specific Drone "
    "Noise Dataset, Zenodo doi:10.5281/zenodo.16196947.",
    "collection_method": "drone fixed on a stand indoors, motors driven at a constant current "
    "from a bench supply; on-board microphones",
    "observation_type": "onboard_bench",
    "sample_rate": SAMPLE_RATE,
    "sample_rate_evidence": SAMPLE_RATE_EVIDENCE,
    "description": "11 constant-current recordings (110-1020 mA) + 1 motors-off room-noise "
    "recording, 10 s each, all deposited channels, one whole-recording Frame each.",
    "geometry": GEOMETRY,
}

_CURRENT = re.compile(r"^Recording_(\d+)mA$")


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


def build(root: Path) -> Iterator[tuple[str, td.Frame]]:
    from scipy.io import loadmat

    from data_processing.sources._common import meta_frame

    root = Path(root)
    for fname, (size, md5) in sorted(FILES.items(), key=lambda kv: (len(kv[0]), kv[0])):
        path = root / fname
        if path.stat().st_size != size or _md5(path) != md5:
            raise ValueError(f"{DATASET}: {fname} differs from the pinned deposit file")
        mat = loadmat(str(path), variable_names=["Recording"])
        rec = mat.pop("Recording")  # (samples, channels) float64
        if rec.ndim != 2 or rec.shape[1] != 32:
            raise ValueError(f"{path}: expected 32 audio columns for Table 1, got {rec.shape}")
        n_total, n_ch = rec.shape
        stem = fname[: -len(".mat")]
        m = _CURRENT.match(stem)
        current = int(m.group(1)) if m else None
        motors_on = current is not None
        audio = np.ascontiguousarray(rec.T, dtype=np.float32)
        if not np.array_equal(audio, rec.T):
            raise ValueError(f"{DATASET}: {fname} is not exactly representable as float32")
        native_dtype = str(rec.dtype)
        del rec, mat
        nonzero = np.flatnonzero(np.any(audio != 0, axis=0))
        trailing_zeros = int(n_total - 1 - nonzero[-1]) if nonzero.size else n_total
        meta = meta_frame(
            stem,
            DATASET,
            system={
                "category": "multirotor_uav",
                "rig": "anu_drone_onboard_32ch",
                "n_rotors": 4,
                "motor_note": "APSIPA 2020 (same group): brushed DC motors, 150 mm "
                "two-blade propellers; identity with this rig assumed, not stated",
            },
            observation={
                "mic_placement": "onboard",
                "type": "onboard_bench",
                "environment": "indoor, drone on a stand; Readme Fig. 1 shows a "
                "foam-lined booth (08/08 session), room for this deposit not stated",
                "source_motion": "static",
                "relative_trajectory": "none",
                "channels_deposited": n_ch,
            },
            operating={
                "condition": "bench" if motors_on else "rotors_off",
                "motor_current_mA": current,
                "motor_current_kind": "constant per recording (filename); constant-"
                "current bench supply per the authors' APSIPA 2020 protocol"
                if motors_on
                else None,
                "n_active_rotors": None if motors_on else 0,
                "n_active_rotors_note": "not stated for this deposit (APSIPA 2020 drove "
                "a single motor; the readme's all-motor 'DroneNoise' folder is a "
                "different session)"
                if motors_on
                else None,
                "contains_rotor_noise": motors_on,
                "external_source": "none",
            },
            label={
                "subcategory": f"stand_{current}mA" if motors_on else "room-noise_control",
                "rig": "ANU on-board array",
                "health": "normal" if motors_on else None,
            },
            extra={
                "source_kind": "original_recording",
                "doi": DOI,
                "original_file": fname,
                "original_variable": "Recording",
                "original_dtype": native_dtype,
                "stored_dtype": "float32 (exact: every deposited value round-trips)",
                "sample_rate_evidence": SAMPLE_RATE_EVIDENCE,
                "recording_n_samples": n_total,
                "recording_duration_s": n_total / SAMPLE_RATE,
                "trailing_digital_zero_samples": trailing_zeros,
                "trailing_zero_note": "every deposited file ends with 768 samples (16 ms) "
                "that are exactly 0 on all 32 channels: recorder padding, kept as deposited",
                "geometry": GEOMETRY,
            },
        )
        frame = td.Frame(
            {
                "audio": td.uniform(audio, SAMPLE_RATE, dims=("mic", "time"), t_start=0.0),
                "mic_pos": td.wrap(mic_positions(n_ch), dims=("mic", None)),
                "meta": meta,
            }
        )
        yield stem, frame
        del audio, frame
