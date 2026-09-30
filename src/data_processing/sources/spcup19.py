"""SPCUP19 egonoise source: 10 heterogeneous student-team drone rigs (DREGON/INRIA).

The "the-spcup19-egonoise-dataset" bonus task: 10 teams each recorded their own
drone's ego-noise with their own mic array (1/4/8/16 ch). Packages are wildly
heterogeneous: most ship loose .wav, ChuMS ships one .mat of a static
PROPELLER RIG (parsed per run, see :func:`_chums_recordings`), KumamoTech ships
only ROS bags (not read). The builder bakes the per-team drone model / channel
count / condition into meta; ChuMS's mic positions go into meta too (not a
shared-dim geometry Series, matching the other teams' meta-only convention).
"""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import tdseries as td

from data_processing.sources._common import (
    audio_frame,
    meta_frame,
    read_audio_file,
    safe_key,
)

_TEAMS: dict[str, dict[str, Any]] = {
    "Idea_ssu": {"id": 393, "drone": "DJI Phantom 4 (GL300C)", "channels": 1},
    "Maverick": {"id": 394, "drone": "YH-19HW quadcopter", "channels": 4},
    "Diagonal_Unloading": {"id": 395, "drone": "DJI Phantom 4 PRO", "channels": 16},
    "ChuMS": {"id": 396, "drone": "Skylark M4-680 (Dotterel)", "channels": 8},
    "LEADS_UAV": {"id": 397, "drone": "DJI Phantom 3 Advanced", "channels": 1},
    "NSS_Chellamma": {"id": 398, "drone": "self-assembled UAV", "channels": 1},
    "KumamoTech": {"id": 399, "drone": "enRoute Zion PG560", "channels": 16},
    "AGH": {"id": 400, "drone": "quadrotor (unspecified)", "channels": 8},
    "KU_Leuven": {"id": 401, "drone": "MikroKopter MK EASY Quadro V3", "channels": 8},
    "Shout_COOEE": {"id": 405, "drone": "Intel Aero RTF", "channels": 8},
}
URLS = {
    f"{team}.zip": f"http://dregon.inria.fr/?smd_process_download=1&download_id={info['id']}"
    for team, info in _TEAMS.items()
}
_CHUMS_DETAILS = re.compile(r"\s*(\d+)\s*propellers?\b.*?Repeat\s*:\s*(\d+)", re.IGNORECASE)


def _condition(name: str) -> str | None:
    low = name.lower()
    table = (
        ("free_flight", "free_flight"),
        ("free-flight", "free_flight"),
        ("stationary", "stationary"),
        ("static", "stationary"),
        ("hover", "hover"),
        ("spinn", "spinning"),
        ("spin", "spinning"),
        ("up_down", "up_down"),
        ("updown", "up_down"),
        ("takeoff", "takeoff"),
        ("landing", "landing"),
        ("single rotor", "single_rotor"),
        ("single_rotor", "single_rotor"),
        ("calibration", "calibration"),
    )
    return next((cond for key, cond in table if key in low), None)


def _frame(
    team: str,
    info: dict,
    rid_suffix: str,
    audio_ct: np.ndarray,
    sr: int,
    *,
    condition: str | None,
    relpath: str,
    mic_positions: list | None = None,
    observation: dict[str, Any] | None = None,
    operating: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> tuple[str, td.Frame]:
    rid = f"{team}__{safe_key(rid_suffix)}"
    extras: dict[str, Any] = {
        "raw_relpath": relpath,
        "competition": "IEEE SP Cup 2019 ego-noise (bonus task)",
    }
    if mic_positions is not None:
        extras["mic_positions"] = mic_positions
    extras.update(extra or {})
    meta = meta_frame(
        rid,
        "SPCUP19-egonoise",
        system={
            "category": "drone",
            "make_model": info["drone"],
            "team": team,
            "n_channels_expected": info["channels"],
        },
        observation={
            "type": "onboard_array",
            "source_motion": "onboard",
            "relative_trajectory": "none",
            "mic_array": f"{info['channels']}ch",
            **(observation or {}),
        },
        operating={"condition": condition, **(operating or {})},
        label={"team": team, "drone": info["drone"]},
        extra=extras,
    )
    return safe_key(rid), audio_frame(audio_ct, int(sr), meta)


def _chums_recordings(
    mat_path: Path, team: str, info: dict, rel_stem: str
) -> Iterator[tuple[str, td.Frame]]:
    """ChuMS ``UAV_rotor_recordings.mat``: a static PROPELLER RIG, not a flight.

    One ``TestResults`` struct: ``MicPositions`` (8×2, x/y in mm) and
    ``Test(1..9)``, each a run described by ``Details`` ("N propellers.
    Repeat:R") whose ``Data(1..8)`` are the eight microphones —
    ``RawTruncatedCalibrated`` (Pa at ``Fs``) plus a per-mic ``OASPL`` and a
    precomputed ``Freq``/``SPL`` spectrum, which are not audio. The mics are
    truncated to slightly different lengths, so they are stacked start-aligned
    to the common minimum (original lengths kept in ``meta.mic_n_samples``).
    One Frame per run; a layout mismatch raises rather than publishing arrays
    of unknown meaning.
    """
    from scipy.io import loadmat

    d = loadmat(str(mat_path), squeeze_me=True, struct_as_record=False)
    if "TestResults" not in d:
        raise ValueError(f"SPCUP19 {team}: {mat_path.name} has no TestResults struct")
    root = np.atleast_1d(d["TestResults"])[0]
    mic_mm = np.asarray(root.MicPositions, dtype=np.float64)
    for run, test in enumerate(np.atleast_1d(root.Test), start=1):
        details = str(test.Details).strip()
        match = _CHUMS_DETAILS.match(details)
        if match is None:
            raise ValueError(
                f"SPCUP19 {team}: Test({run}) Details {details!r} is not 'N propellers. Repeat:R'"
            )
        n_props, repeat = int(match.group(1)), int(match.group(2))
        mics = np.atleast_1d(test.Data)
        rates = {int(round(float(mic.Fs))) for mic in mics}
        if len(rates) != 1 or len(mics) != mic_mm.shape[0]:
            raise ValueError(
                f"SPCUP19 {team}: Test({run}) has {len(mics)} mics at Fs {sorted(rates)} "
                f"for {mic_mm.shape[0]} MicPositions"
            )
        sigs = [
            np.asarray(mic.RawTruncatedCalibrated, dtype=np.float32).reshape(-1) for mic in mics
        ]
        n = min(s.size for s in sigs)
        yield _frame(
            team,
            info,
            f"{rel_stem}/{n_props}prop_repeat{repeat}",
            np.stack([s[:n] for s in sigs]),
            rates.pop(),
            condition="propeller_rig",
            relpath=f"{team}/{rel_stem}.mat:TestResults/Test({run})",
            mic_positions=mic_mm.tolist(),
            observation={"type": "fixed_array_bench", "source_motion": "static"},
            operating={"n_propellers": n_props, "repeat": repeat, "details": details},
            extra={
                "mic_positions_unit": "mm",
                "mic_positions_axes": "xy",
                "audio_unit": "Pa",
                "oaspl_db": [float(mic.OASPL) for mic in mics],
                "mic_n_samples": [int(s.size) for s in sigs],
            },
        )


def _dedup_key(seen: dict[str, int], key: str) -> str:
    """Return ``key`` the first time, then ``key_2``/``key_3``/… on collision."""
    n = seen.get(key, 0)
    seen[key] = n + 1
    return key if n == 0 else f"{key}_{n + 1}"


def build(raw_dir: Path) -> Iterator[tuple[str, td.Frame]]:
    """10 team packages under ``raw_dir/<team>/``. Loose .wav → one recording
    each; the ChuMS .mat → one recording per propeller-rig run.

    Teams lay files out in *scenario subdirs* (``static clean/1.wav``,
    ``ego-noise/single rotors/1.wav``, …) that reuse bare-integer stems, so keys
    are derived from the **team-relative path** (not the stem) — and the subdir
    tokens feed condition detection. A final per-team dedup guard suffixes any
    residual collision so the publish never aborts on a duplicate key. A .mat
    outside ChuMS raises: guessing audio from an unknown struct is what once
    published ChuMS's per-mic spectra as 216 unlabelled "recordings"."""
    for team, info in _TEAMS.items():
        team_dir = Path(raw_dir) / team
        if not team_dir.exists():
            continue
        seen: dict[str, int] = {}
        for wav in sorted(team_dir.rglob("*.wav")):
            try:
                audio, sr = read_audio_file(wav)
            except Exception:  # noqa: BLE001
                continue
            rel = wav.relative_to(team_dir).with_suffix("")  # e.g. "static clean/1"
            key, frame = _frame(
                team,
                info,
                str(rel),
                audio,
                sr,
                condition=_condition(str(rel)),
                relpath=str(wav.relative_to(raw_dir)),
            )
            yield _dedup_key(seen, key), frame
        for mat in sorted(team_dir.rglob("*.mat")):
            if team != "ChuMS":
                raise ValueError(f"SPCUP19 {team}: no parser for {mat.relative_to(raw_dir)}")
            rel_stem = str(mat.relative_to(team_dir).with_suffix(""))
            for key, frame in _chums_recordings(mat, team, info, rel_stem):
                yield _dedup_key(seen, key), frame


PROVENANCE = {
    "source_url": "https://dregon.inria.fr/datasets/the-spcup19-egonoise-dataset/",
    "doi": "10.48550/arXiv.1907.04655",
    "license": "free for personal, educational and academic use only",
    "citation": "Deleforge et al., Audio-Based Search and Rescue With a Drone: IEEE SP Cup 2019 (IEEE SPM 36(5), 2019).",
    "collection_method": "10 student teams recorded their own drone's ego-noise with their own on-board mic array (bonus task)",
    "equipment": "heterogeneous: 10 different drones + mic arrays (1/4/8/16 ch); see per-sample system.make_model + team",
    "observation_type": "onboard_array (ChuMS: fixed_array_bench, a static propeller rig)",
    "sample_rate": "per recording: 16 / 44.1 / 48 kHz (ChuMS: the .mat's Fs, 44.1 kHz)",
    "channels": "per recording: 1 / 3 / 8 (KumamoTech's 16-ch ROS bags are not read)",
    "description": "Drone-variety ego-noise: heterogeneous team rigs (Phantom 3/4, Skylark, Intel Aero, MikroKopter, ...). Loose .wav from 8 teams + the ChuMS propeller-rig .mat (one 8-mic Frame per run, calibrated Pa, n_propellers/repeat in meta.operating, mic positions in mm); KumamoTech ships only ROS bags and is absent. Rich per-team drone/condition meta.",
}
