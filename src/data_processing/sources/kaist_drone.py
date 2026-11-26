"""KAIST drone fault sounds: the ICSV29 study (Zenodo 7779574) and the ICSV31
AI Challenge set derived from the same recordings.

Both releases are **manufactured mixtures**, not original recordings. Yi, Choi
& Lee (ICSV29 2023, arXiv:2304.11708 §2) recorded three toy quadcopters (A:
Holy Stone HS720, B: MJX Bugs 12 EIS, C: ZLRC SG906 Pro2) in an anechoic
chamber, loosely tethered by elastic ropes (1.5 m to the ceiling, 1 m to the
floor, rotatable rings), healthy or with one dented motor cap (MF1-4) or one
~10 %-cut propeller (PC1-4), with the two transmitters of a RØDE Wireless GO II
on the top of the body. The action labels are six *maneuvering directions*
(paper §1/§2.3, Zenodo README ``F(Front), B(Back), R(Right), L(Left),
C(Clockwise), CC(Counter-clockwise)``; challenge README "moving direction":
``Front/Back/Right/Left/Clockwise/CounterClockwise``). No source defines them
further; Clockwise/CounterClockwise are rotation commands, read as yaw
[inference: free rotation through the rotatable rings]. The clean 48 kHz
recordings were never released.

Waveform audit of the released files (Uni Slurm 37570802, 37571032-37572495):

- ``build`` (Zenodo 7779574 v1, CC BY 4.0): per drone, direction, fault and
  microphone, one recording cut into 0.5 s windows at a 0.25 s hop (window
  ``s`` starts at ``0.25 s * s``; ``s``/``s+1`` agree at a 4000-sample lag,
  median r 0.93), each window mixed with a campus background excerpt at SNR
  10-15 dB. The ``mic1``/``mic2`` files of one mixing draw (same split,
  background, background index, exact SNR string) are *not* sample-aligned
  channels: lag-0 r median 0.05; averaged over ~700 windows each condition's
  mic2 drone audio sits -2.0..+2.0 s from mic1's, and shifting only the drone
  component would not align the independently mixed scenes. So one Frame per
  released waveform, never ``(mic, time)``. In some drone-C propeller
  conditions mic2 decodes to the same PCM as mic1: those draws are one Frame with both
  file ids. The authors' random 6:2:2 split remixes the same window into
  other splits (79,660 window keys, 159,320 files). For 93 members libsndfile
  cannot read, strict FFmpeg returns all 8000 samples, checked against the PCM
  audit and the FLAC STREAMINFO MD5.
- ``build_icsv31`` (ICSV31AIchallenge/ICSV31-AI-Challenge, CC BY-SA 4.0): 2 s
  windows of the same per-microphone recordings (mic1 or mic2, about half
  each by waveform match; the test ``orig_file_name`` index is roughly a start
  second on a mic1-then-mic2 timeline [inference]), mixed at SNR -5..5 dB with three campus
  backgrounds plus DEMAND park/town-square/traffic noise; train (normal
  only), eval and test (test adds MF4/PC4). About 40 % of eval/test clips
  overlap a train clip in time. The challenge forbids training on
  ``eval``/``test`` for challenge submissions; ``meta.label.split`` keeps it.

All ICSV31 drone audio is contained in Zenodo windows: neither release is a
holdout for the other.
"""

from __future__ import annotations

import hashlib
import io
import re
import tarfile
import zipfile
from collections.abc import Iterator
from pathlib import Path
from typing import NamedTuple, Protocol, runtime_checkable

import numpy as np
import tdseries as td

ZENODO_DATASET = "KAIST-drone-fault"
ICSV31_DATASET = "ICSV31-drone-anomaly"

DRONES = {
    "A": "Holy Stone HS720",
    "B": "MJX Bugs 12 EIS",
    "C": "ZLRC SG906 Pro2",
}
DIRECTIONS = {
    "F": "front",
    "B": "back",
    "R": "right",
    "L": "left",
    "C": "clockwise",
    "CC": "counterclockwise",
}
ICSV31_DIRECTIONS = {
    "Front": "front",
    "Back": "back",
    "Right": "right",
    "Left": "left",
    "Clockwise": "clockwise",
    "CounterClockwise": "counterclockwise",
}

#: Zenodo 7779574 v1 (2023-03-29): file -> (bytes, md5).
ZENODO_FILES = {
    "drone_A.tar": (2554408960, "14a6fd11add6aedb6e53404df7811147"),
    "drone_B.tar": (2535034880, "b3d19705fdc1f31f618c1f4d2682f2a4"),
    "drone_C.tar": (2533406720, "54b1e10b92d572551942a437130ecb36"),
}

#: Basename forms seen in the tars (UniPublish full inventory, 2026-11-26):
#: train names carry a ``_mic1_``/``_mic2_`` token, valid/test names do not;
#: 26 drone-B names lack the fault token (``B_R_110_ConstructionSite_…``).
#: The microphone is always the ``mic1``/``mic2`` directory; a token must agree.
_ZENODO_NAME = re.compile(
    r"^(?P<drone>[ABC])_(?P<dir>CC|F|B|R|L|C)(?:_(?P<fault>N|MF[1-4]|PC[1-4]))?(?:_(?P<mic>mic[12]))?_"
    r"(?P<seg>\d+)_(?P<bg>[A-Za-z]+)_(?P<bgidx>\d+)_snr=(?P<snr>[-+0-9.eE]+)\.wav$"
)

RECORDING = {
    "microphone": "RØDE Wireless GO II (two transmitters on the top side of the drone body)",
    "environment": "anechoic chamber (KAIST); drone tethered by elastic ropes to ceiling "
    "(1.5 m) and floor (1 m) through rotatable rings",
    "original_rate_hz": 48000,
    "faults": "MFk: motor cap of motor k dented in a vice (friction); PCk: ~10 % of "
    "propeller k cut off",
    "mic_pos": "not published: both transmitters on the top of the body (paper Sec. 2.2, "
    "Fig. 3); no coordinates, mic1/mic2 placement order not documented",
    "transmitter_envelope_mm": [44.0, 45.3, 18.3],
    "transmitter_envelope_source": "RØDE Wireless GO II datasheet "
    "(https://edge.rode.com/pdf/page/88/modules/425/WirelessGOII_Datasheet_2.pdf)",
    "unresolved": "per-rig transmitter positions, spacing, height and mic1/mic2 mapping "
    "(paper Fig. 3(b) is a blurred frontal photo of drone B)",
    "maneuver_labels": "six maneuvering directions F/B/R/L/C/CC = front, back, right, left, "
    "clockwise, counter-clockwise (paper Sec. 1 and 2.3; Zenodo README; challenge README "
    "'moving direction'). Not defined further; C/CC read as yaw rotation [inference].",
}

ZENODO_PROVENANCE = {
    "source_url": "https://zenodo.org/records/7779574",
    "doi": "10.5281/zenodo.7779574",
    "version": "1 (2023-03-29)",
    "license": "CC BY 4.0",
    "rights_note": "Zenodo record licence CC-BY-4.0 covers the deposited data; attribute "
    "Yi, Choi & Lee.",
    "citation": "W. Yi, J.-W. Choi, J.-W. Lee, 'Sound-based drone fault classification using "
    "multitask learning', Proc. ICSV29, Prague 2023 (arXiv:2304.11708); data: Zenodo "
    "doi:10.5281/zenodo.7779574.",
    "collection_method": "tethered quadcopters in an anechoic chamber, 6 commanded "
    "maneuvering directions, 9 health states; one recording per drone, direction, health "
    "state and microphone, cut into windows mixed with separately recorded campus noise",
    "observation_type": "onboard_mic_tethered",
    "sample_rate": 16000,
    "channels": 1,
    "derivative": "artificial mixtures only: 0.5 s windows at a 0.25 s hop, 16 kHz, drone + "
    "campus background at SNR 10-15 dB (the clean anechoic drone recordings are not released)",
    "description": "324,000 original mono members (train PCM24 WAV, valid/test PCM24 FLAC) "
    "-> one mono Frame per distinct released waveform; a draw whose mic1 and mic2 files hold "
    "identical decoded PCM under the same name identity is one Frame listing both files. Keys "
    "<drone>/<split>/<mic1|mic2|mic1=mic2>/<original basename>. extra.mixture_partner_file "
    "names the other microphone's file of the same mixing draw (same split, background, "
    "background index, exact SNR string): a different recording, never a synchronous channel.",
    "mic_alignment_evidence": "Uni Slurm 37570802 (540 sampled draws): mic1 vs mic2 lag-0 r "
    "median 0.05. Slurm 37571920: per drone/direction/fault, cross-spectra of mic1 window s "
    "and mic2 window s+k averaged over ~700 windows, |lag| <= 30 s: one clear peak in 157 of "
    "162 conditions at -2.0..+2.0 s (peak / 99.9th percentile median 12, min 2.8) and r "
    "1.000 at 0 s where mic2 copies mic1. Slurm 37572495 (train files only, every mic1 "
    "fault x mic2 fault): the "
    "same-named mic2 recording is the aligned one in 157 of 162 conditions (peak median "
    "0.61 vs <= 0.06 off-label); drone B Right N/MF1-4 mic2 align with no mic1 recording "
    "(peaks <= 0.16), so those mic2 labels are not cross-verified. Offsets relate the drone "
    "components only; the mixed scenes are independent.",
    "counts": "Uni Slurm 37572729 dry run: 324,000 released files -> 318,743 Frames "
    "(156,743 mic1 + 156,743 mic2 + 5,257 mic1=mic2 draws with identical decoded PCM, drone "
    "C PC3/PC4 conditions, holding 10,514 file ids); no other PCM duplicate. Normal by file "
    "name and not disputed: 35,622 (train 21,600 / valid 7,022 / test 7,000); disputed: "
    "1,978 (176 named N, 1,776 MF, 26 without fault token).",
    "window_evidence": "Slurm 37570802: same microphone, window s vs s+1 r 0.93 at exactly "
    "4000 samples; equal window index in another split r 0.94 at lag 0 (remixed duplicate). "
    "extra.window_offset_s_inferred = 0.25 s * index, relative to window 0 of that "
    "drone/direction/fault/microphone recording as released; the authors' initial trim is "
    "unknown, so it is not a recorder timestamp and not a clock shared across microphones.",
    "split_note": "authors' random 6:2:2 split of mixtures (paper Sec. 2.3): 79,660 "
    "(drone, direction, fault, mic, window) keys occur in two splits (159,320 files) and "
    "neighbouring windows overlap by 0.25 s. Group by extra.parent_window / window offset, "
    "not by file, for leakage-free evaluation.",
    "label_note": "label.* = released file name. label.label_disputed marks the 1,978 drone-B "
    "Right valid/test mic2 files whose names contradict their waveform identity (Slurm "
    "37571929); 26 of them lack a fault token (health 'unlabelled').",
    "also_redistributed_in": "geronimobasso/drone-audio-detection-samples (DADS; card source "
    "#6) = project dload drone-detection-samples@77b8a98db065 as 16-bit mono 16 kHz label-1 "
    "clips; not an independent source, do not split across the two",
    "decoder_evidence": "Uni Slurm 37561172: libsndfile rejected 93 of 324,000 members. "
    "Slurm 37562399: strict FFmpeg 7.0.2 decoded all 93 in full. Each alternate decode "
    "checks the independently recorded PCM SHA-256 and the publisher's FLAC PCM MD5.",
    "recording": RECORDING,
}

ICSV31_PROVENANCE = {
    "source_url": "https://github.com/ICSV31AIchallenge/ICSV31-AI-Challenge",
    "license": "CC BY-SA 4.0",
    "rights_note": "Official challenge repository is licensed CC BY-SA 4.0 and distributes "
    "the dataset links; underlying recordings are the CC BY 4.0 Zenodo 7779574 study; "
    "DEMAND noise (zenodo 1227121) is CC BY-SA 4.0. Challenge rule: eval/test must not be "
    "used as training data for challenge submissions.",
    "citation": "ICSV31 AI Challenge (2nd KSNVE AI Challenge, 2025), organisers J.-W. Choi, "
    "J. Choi, S. Park, Y. Kim (KAIST); based on Yi, Choi & Lee, ICSV29 2023.",
    "collection_method": "ICSV29 anechoic tethered per-microphone drone recordings cut into "
    "2 s windows (1 s hop) and re-mixed with campus + DEMAND noise",
    "observation_type": "onboard_mic_tethered",
    "sample_rate": 16000,
    "derivative": "artificial mixtures only: 2 s, 16 kHz, SNR -5..5 dB",
    "related_release": "doi:10.5281/zenodo.7779574 (same per-microphone drone recordings)",
    "overlap_evidence": "Uni Slurm 37570802: every clip matched against all Zenodo windows of "
    "its drone and direction (both mics; normal, test-labelled or eval-anomaly faults). 7,622 "
    "of 7,920 best matches fall exactly on the 0.25 s window grid (median r 0.75); 7,017 are "
    "confident (r >= 0.55, >= 2 consistent hits), mic1 3,599 / mic2 3,418. Test fault labels "
    "agree with every best match. Test orig index k is approximately a start second on a "
    "mic1-then-mic2 timeline (mic1: start = k within 0.25 s for 69 %; mic2: k - start "
    "~190-260 s per condition) [inference]. Of confident clips 398/951 eval and 503/1,270 "
    "test overlap a train clip of the same recording in time; none start at the same "
    "sample. Per-clip mapping is not exhaustive and not published.",
    "description": "train 5400 normal, eval 1080 normal+anomaly (eval anomaly fault type not "
    "released), test 1440 (labels from the organisers' test-label archive). "
    "extra.parent_group_2s / parent_segment_index_2s copy the test label's orig_file_name: "
    "a start second, so neighbouring indices overlap by 1 s.",
    "recording": RECORDING,
}

#: Google Drive file ids of the official challenge links: archive -> (id, bytes, md5).
ICSV31_FILES: dict[str, tuple[str, int, str]] = {
    "ICSV31AIChallengeDataset.zip": (
        "1BK-dpXG68gTmwA68JbbDW10AlaVRMYJU",
        607788578,
        "c590af1b8a7224bb839436465d762403",
    ),
    "ICSV31AIChallengeDataset_test.zip": (
        "1veguvvaMPhcwwNr1vKE9QPiZkmK-i_5N",
        121787379,
        "a8a7c5d8b4dc93907b74f7f372565236",
    ),
    "ICSV31AIChallengeDataset_test_labels.zip": (
        "1jsOcfb_k4XnDPfegJha_j-5TJC4dws4N",
        10855,
        "033cebe7692d3fc9ed837db28cf79e92",
    ),
}


def _health(fault: str | None) -> tuple[str, int | None]:
    if fault is None:  # name without a fault token (26 drone-B mic2 test files)
        return "unlabelled", None
    if fault == "N":
        return "normal", None
    return ("motor_fault" if fault.startswith("MF") else "propeller_fault"), int(fault[2:])


def _identity(m: re.Match[str]) -> tuple[str, ...]:
    """What a file name claims besides the mixing draw: drone, direction, fault, segment."""
    return (m["drone"], m["dir"], m["fault"] or "nofault", m["seg"])


class _Member(NamedTuple):
    split: str
    mic: str
    name: str  # TAR member path
    offset: int
    size: int
    match: re.Match[str]


@runtime_checkable
class _MemberCache(Protocol):
    """CPython ``TarFile`` read cache: every TarInfo ``next()`` has returned so far."""

    members: list[tarfile.TarInfo]


#: Drone B "Right" valid/test: 1,978 draws whose mic1 and mic2 files name
#: different windows (their mic2 files have no name-matching mic1 file). Waveform
#: identity against the same microphone's train files (Uni Slurm 37571929):
#: 3,284/3,285 confidently identified mic1 valid/test files and 1,480/1,483
#: name-matched mic2 files are what their names say; 0/1,805 confidently
#: identified mic2 files of these draws are (segment always differs; fault in
#: 251). Their mic2 file names stay as released, flagged ``label_disputed``.
_DISPUTED_DRAWS = 1978
_DISPUTED_SCOPE = {("B", "R", "valid"), ("B", "R", "test")}


def _tar_pairs(path: Path, drone: str) -> list[tuple[_Member, _Member]]:
    """``(mic1, mic2)`` members sharing one mixing draw, sorted by mic1 member.

    Header-only scan of the uncompressed tar (tarfile seeks over payloads); the
    TarInfo cache is dropped per member so RSS stays at the small index. Every
    exact SNR string occurs once under ``mic1/`` and once under ``mic2/`` of
    one split with the same background and background index."""
    by_draw: dict[tuple[str, str], dict[str, _Member]] = {}
    with tarfile.open(path, "r:") as tar:
        if not isinstance(tar, _MemberCache):
            raise TypeError("tarfile.TarFile has no TarInfo members cache to drop")
        while (member := tar.next()) is not None:
            tar.members.clear()
            if not member.isfile():
                continue
            parts = member.name.split("/")
            m = _ZENODO_NAME.match(parts[-1])
            if (
                m is None
                or len(parts) != 4
                or parts[0] != drone
                or m["drone"] != drone
                or parts[1] not in ("train", "valid", "test")
                or parts[2] not in ("mic1", "mic2")
                or (m["mic"] is not None and m["mic"] != parts[2])
            ):
                raise ValueError(f"{path.name}: unexpected member {member.name!r}")
            mics = by_draw.setdefault((parts[1], m["snr"]), {})
            if parts[2] in mics:
                raise ValueError(f"{path.name}: two {parts[2]} files with SNR {m['snr']}")
            mics[parts[2]] = _Member(
                parts[1], parts[2], member.name, member.offset_data, member.size, m
            )
    pairs = []
    for (split, snr), mics in by_draw.items():
        if set(mics) != {"mic1", "mic2"}:
            raise ValueError(f"{path.name}: {split} SNR {snr} lacks a microphone: {sorted(mics)}")
        a, b = mics["mic1"].match, mics["mic2"].match
        if (a["dir"], a["bg"], a["bgidx"]) != (b["dir"], b["bg"], b["bgidx"]):
            raise ValueError(f"{path.name}: {split} SNR {snr} pairs different draws")
        pairs.append((mics["mic1"], mics["mic2"]))
    pairs.sort(key=lambda p: p[0].name)
    return pairs


class _FlacAudit(NamedTuple):
    offset: int
    size: int
    pcm_s32le_sha256: str


# Exact libsndfile exceptions from the Zenodo 7779574 v1 TARs. Uni Slurm
# 37562399, FFmpeg 7.0.2 strict PCM decode; every row has 8000 samples.
# Columns: TAR member | byte offset | byte length | SHA-256 of PCM S32LE.
_FLAC_AUDIT_ROWS = """\
A/test/mic1/A_B_MF3_104_ConstructionSite_136_snr=10.747008676682475.wav 2094462976 21965 a1900d08334ee3d7687266ce60b0cc4b7a855918b35a2d0412714b78ff614ab5
A/test/mic1/A_B_MF3_836_ConstructionSite_796_snr=10.666921227077394.wav 2210985984 21767 c6149ba51b99752decfec61f3509f122ab411fc9bb326b1893d1c9e30fca5f74
A/test/mic1/A_B_MF3_95_ConstructionSite_454_snr=11.49897333705066.wav 2237916160 21793 f0541fb451404107a22466d6dac940625585ffdc051598ed233298b86aa4ee15
A/test/mic1/A_B_N_8_SportsComplex_647_snr=10.792405050043577.wav 2289036800 21772 07c1d20962c7e03ea248acd0d5602e42fd4b2e29d088dd2e19233642d01da545
A/test/mic1/A_B_PC4_774_WestDoor_387_snr=10.374676577401553.wav 2084507648 21786 d925113de57828faf47f6d8de9e5bddf2fb7799f2a044e8ffa6bbdaccbcfc44f
A/test/mic1/A_CC_MF2_145_ConstructionSite_947_snr=10.971519450887751.wav 2205165056 21852 7f60093b592041b6cfe2ad78fb36507442a98810910d534ea741b9c9d462fb73
A/test/mic1/A_CC_MF2_178_EoeunHill_30_snr=11.00177736609973.wav 2146929664 21783 7cca5fa7fb279845eee8eb63ce7d545532ed8acbe6c799057001c38356f8ff26
A/test/mic1/A_CC_MF2_188_SportsComplex_135_snr=12.150880071777213.wav 2177725952 21808 d532b8ebd49a1f04005c32e9111fe0c2f042edde579722a6cd5d2162a1de066c
A/test/mic1/A_CC_MF2_251_DuckPond_402_snr=11.338514552933438.wav 2234864128 21770 4d474112e02ad6a55152ae00ca460352af8afad8cf187caae7489ea9c277160f
A/test/mic1/A_CC_PC1_131_WestDoor_132_snr=11.249427852321885.wav 2257658880 21583 5eb145b1aea3ccfacaf394b9490d62180519cbf6549067ed4d76d771dad998a2
A/test/mic1/A_CC_PC1_149_SportsComplex_800_snr=11.383932792065142.wav 2295031296 21705 6dbbb9d71ef48d04ccf9c11c0451c82b31a97f4ca9c84955ca2e0ac8b867b740
A/test/mic1/A_CC_PC1_157_EoeunHill_892_snr=10.129591348299106.wav 2109464064 21742 3b154034cb9c2c90150f7a0da4ab7dad7d80de95fca3cfc5c0624dfc713e48b5
A/test/mic1/A_CC_PC1_158_EoeunHill_54_snr=14.88381655336415.wav 2290707968 21643 06ccb7a995b1da6c906aee562566c272c615e45a23bf7ac5ae1bbf12b5a4bfe7
A/test/mic1/A_CC_PC1_15_EoeunHill_317_snr=12.331173811472036.wav 2305812992 21708 0e891d9063be1c10250f8337a969acc036eaa036385d288ef778b015b051b04e
A/test/mic1/A_CC_PC1_160_EoeunHill_7_snr=14.902329852464412.wav 2254652416 21660 74bfff1efc8f32641d103ef62544f9e84901d1312ca352ed6aa035b0ed4c9378
A/test/mic1/A_CC_PC1_16_WestDoor_373_snr=11.105140972655008.wav 2199519744 21753 92cb4d18c5cb0639684e73eb205e0d1fae7f8086a601d40eb841ea512f605f1a
A/test/mic1/A_CC_PC1_20_EoeunHill_761_snr=13.63161387139151.wav 2294897152 21671 199a1ad0d92064fa697a6adce17fc110fb8c8468ec95d646e43da8f39e1d7bcb
A/test/mic1/A_CC_PC1_256_SportsComplex_238_snr=11.436663507615506.wav 2203088896 21734 8386888957046a5c6fb8f2a347c2d44c95d11485c269710f1609c318a81b3713
A/test/mic1/A_CC_PC1_27_EoeunHill_892_snr=13.356294858680403.wav 2304967168 21617 19646826327e5c8a202716782cc11c5700949b5a6a827b428c4356bc7c953809
A/test/mic1/A_C_PC2_107_SportsComplex_929_snr=11.151275174758629.wav 2282837504 21519 d01976353ea730a0ce73622dfbef3720e25cfeaf096a106e9e7a82d6c85036d8
A/test/mic1/A_L_MF2_883_EoeunHill_573_snr=10.71156114831791.wav 2258681856 21816 5b165af4269be2488d446599ee7aa1bc9130ac33439e887faf404f2b94ef95f2
A/test/mic1/A_L_MF4_954_ConstructionSite_990_snr=10.035145106145581.wav 2310985728 21214 6598732dc4d5ca3a5df92f31185d9a20ff7e969ff5cfa51bd59c4387169baa51
A/test/mic1/A_L_PC1_131_SportsComplex_218_snr=10.55016472256248.wav 2254117888 21768 88f092b5f9899f6da66b7a8b73930c8e9ca5d59da2bb313a84ae42ee6c6c4db9
A/test/mic1/A_L_PC1_150_SportsComplex_17_snr=10.146051754673604.wav 2175587328 21808 3e2b690ff71fdc60a1179a476a3c903c7bbfc23d4337c395e71e35d677c92790
A/test/mic1/A_L_PC1_846_ConstructionSite_490_snr=11.903251873198295.wav 2261199872 21693 4473668801f627439074c1c49746271703d86208f1d8adbdff8ed05a7605684d
A/test/mic1/A_L_PC1_871_SportsComplex_1033_snr=12.115762100499706.wav 2080901632 21788 8662c26e7fc3e4882a87622faae7b0754c0cf8ce1f64fe50bcb6d86981fc27f9
A/test/mic1/A_L_PC2_794_SportsComplex_773_snr=14.990022745318736.wav 2164625920 21416 293daab686030e7592ef3379d5b32260e0cc25f29093d271cd1610b774d44e77
A/test/mic2/A_B_MF3_112_WestDoor_220_snr=10.912026148791462.wav 2490951680 21801 d88954f08754fab4c4d28f6ec816ad612206a576c7e9224724f85a12f88c5fb8
A/test/mic2/A_CC_MF2_114_DuckPond_619_snr=10.826581238160536.wav 2327770624 22059 88b763a802e8bf4151859da01317f613d7ed903f9429864d4874a0a5615bf36c
A/test/mic2/A_CC_MF2_117_WestDoor_114_snr=12.62493952249661.wav 2355258368 21947 4b2bfc35c78f251c40c24368423799bbfcfd79221ce1e48ccbbdfa0a4675b5e8
A/test/mic2/A_R_PC2_190_SportsComplex_1144_snr=10.712522078948865.wav 2508448768 21738 74a982250d36702bfc5d49fdb33a6aadcaee516af84fe7e7ae4df497ec6f7f1b
A/valid/mic1/A_B_MF2_788_WestDoor_728_snr=10.737970939546484.wav 1619084800 21785 79f821ca80d9a7ad03c6f543a7d57b4b10c2edea427c58c007453268ce2fa0c6
A/valid/mic1/A_B_MF3_719_SportsComplex_361_snr=12.020708415650434.wav 1593776640 21583 f972eebd44c66fe98af82c557d3d2d377a6acabedcf9155396dab13a02d3cbb1
A/valid/mic1/A_B_MF3_768_ConstructionSite_286_snr=10.842793445321204.wav 1764857856 21669 60a2602016c34220aff520e2664ff29cebed4b8766a35d434f9f7039039558f9
A/valid/mic1/A_B_PC2_731_ConstructionSite_253_snr=11.166238417763065.wav 1766107648 21479 83b080f39cdd09d27dd1e68fc631d342c5f7d56248abe6dcd0018446a1f34b58
A/valid/mic1/A_B_PC4_651_WestDoor_535_snr=11.978523582907995.wav 1636704256 21870 0ff9fc0977d5cce929231b60017cb5264ff03118f77e79164e85ae3326ce2768
A/valid/mic1/A_B_PC4_687_SportsComplex_362_snr=10.545608462464598.wav 1702078464 21899 084ab6befd3df110041b1cd25260b4aa75fa751cbedd56cdd96d3ff31cc61491
A/valid/mic1/A_CC_MF2_728_EoeunHill_792_snr=10.809060421670585.wav 1764590080 21797 552fd7230d4403b092720fba25de99200a654e18928fea97e201423ac5b7e89b
A/valid/mic1/A_CC_MF3_66_DuckPond_232_snr=10.546508494453443.wav 1799981568 21731 d9da72c951dca11635dbea3723ea5d9dfd9d4f328b7b266d7a04201a97a42e26
A/valid/mic1/A_CC_MF3_76_EoeunHill_173_snr=10.226057672367379.wav 1711259648 21633 4a61e469d6cfe91e8dd2e00c3e255b5e342f8626a7470b492212d497660290d4
A/valid/mic1/A_CC_MF3_77_SportsComplex_464_snr=11.171916846019249.wav 1643744256 21751 67038b3bfcb97e94e7a2edbe94bd78d4acbe11380338634e41f31e356af9367c
A/valid/mic1/A_CC_PC1_102_SportsComplex_1076_snr=10.29092376563029.wav 1625952768 21817 2614c54c55098a9023b8656e65f815c4fed073c8d2432ea3315d1bc442440075
A/valid/mic1/A_CC_PC1_672_SportsComplex_128_snr=10.570303021585936.wav 1811018240 21859 2eca7115b45392f1103cd7d5985e0446d1923682a1c6d15321e54e5dabca14aa
A/valid/mic1/A_CC_PC1_70_ConstructionSite_66_snr=14.564640034459234.wav 1748588544 21727 74fa7763bf4fb99f3981b4feeff92f1dfc6a472063b8ec3739ccc0b2e14fbdee
A/valid/mic1/A_F_MF4_706_DuckPond_308_snr=10.485196301315627.wav 1726289408 21719 1aed227b46b1d4a80c5166cd951b8c3de29e16df2ab06d2d67a25e426e42ef3a
A/valid/mic1/A_L_MF1_686_ConstructionSite_636_snr=12.547337586869915.wav 1679852544 21824 46e88b10671205a71d8ee664134b168f5adfa1cb0828ed330248c887e573e120
A/valid/mic1/A_L_MF1_687_EoeunHill_289_snr=14.31099110651123.wav 1619641344 21783 8fa23350dce6f357cf9a1ae8df1e14b444646412c4c5af9e5b68e04b7b83a092
A/valid/mic1/A_L_MF1_696_WestDoor_180_snr=10.657685814149664.wav 1724354048 21878 4547871e8f40accc2d7af2992cc1deeb88dec2daeb13c5c7d79a98efc322cf78
A/valid/mic1/A_L_MF1_697_EoeunHill_603_snr=10.162178867911486.wav 1802049536 21826 d7474da2ca9768cbbb01697e72174555b6ccfe12c74ca791d69a3db95ba68031
A/valid/mic1/A_L_MF1_704_SportsComplex_203_snr=13.590019813903815.wav 1748187136 21830 6c197c2f1dfa9f902ee393b01448503514980e2bfed1281bdb3e714ec2b44b4e
A/valid/mic1/A_L_MF1_739_DuckPond_701_snr=10.252335166583055.wav 1711970304 21751 669ce29c45f9a70d7794f1792f52bf0fcb1a2aa9a3370477bda61dd026b70de1
A/valid/mic1/A_L_MF3_7_WestDoor_675_snr=10.601150221835892.wav 1825289216 22025 f1955bcfa18d03696f57579b7d3f66e92c1694fafb7efb8489d6900262e2de04
A/valid/mic1/A_L_N_602_EoeunHill_819_snr=12.059474608285596.wav 1813534208 21550 cbadd4ba63b95e9250eb054d9a275da6478ed96c496466d761c605dcf2f55be7
A/valid/mic1/A_L_N_681_EoeunHill_466_snr=10.912358262441122.wav 1710040064 21714 dad41536a1838a389164619a0dd2a8b2188b1d7e744d422e0f57cab59a34043e
A/valid/mic1/A_L_PC1_796_SportsComplex_1166_snr=12.231206993656485.wav 1751482368 21716 3cb6795384ec28eb825ef798256736097236d9d20e3919fd561bf86e3eb0b6f8
B/test/mic1/B_B_N_88_DuckPond_302_snr=12.953648798419794.wav 2146741760 21265 a4b458cfae182d13fa22c10224b02c2733189863977c7a6311e71444710d5440
B/test/mic1/B_CC_MF3_0_EoeunHill_481_snr=12.348400413125297.wav 2084655616 21079 ed93a3443104ed6e1527eb1a6fc5f3dbf403935053248dc2c36230bcea252b47
B/test/mic1/B_CC_MF3_90_WestDoor_534_snr=13.320098017547451.wav 2299462656 21052 b098702d3587d7dcc3eb9e2278bd173c9a10ebc3bd863727d0aa6cdd41a5d1dc
B/test/mic1/B_F_MF4_94_SportsComplex_48_snr=11.371765926894234.wav 2290643968 21284 319947c1de7168ba1037303d7a6059680dc234ef6c112f3d98deba1280304e4f
B/test/mic1/B_F_MF4_95_SportsComplex_1129_snr=14.537953919359866.wav 2182126080 21370 7a8f13ec44e85ee206bb2c310c2610a8f162faa89f20077174b01977a57fa072
B/test/mic1/B_F_MF4_96_SportsComplex_552_snr=10.76881656005117.wav 2080023040 21076 5b7ef59e4739361f8e5e71050733da6708b15aa09df0621491a2dd304dd64e90
B/test/mic1/B_L_PC3_93_ConstructionSite_509_snr=11.275140614405098.wav 2161352704 21450 5f8c89d5265be3972353988f1e5609b9dba913baf0509a2252b94c5ae0d078a4
B/test/mic2/B_F_N_0_WestDoor_553_snr=12.608286066546675.wav 2347087872 21448 bee13eb7aa9f15a5bc1dacfd6451111ca356446366fa9f7161dffc0a29c4bfa1
B/test/mic2/B_F_N_85_SportsComplex_1092_snr=13.580591827204204.wav 2370116096 21266 1faf3791e7346bcb187cf2476897c6feaf731c2f14ed3475c65c9c2eb31e1968
B/test/mic2/B_L_MF4_854_ConstructionSite_773_snr=10.584083795674003.wav 2518704640 21100 670920d67f9011dc6fcc38449e795ff8ac784ec0a0c59cae59b7133b1cb4917e
B/valid/mic1/B_B_MF3_684_DuckPond_305_snr=10.831063094535708.wav 1780271616 21541 4425b5d980d3ca308efcfe41eefc461208fbbdd896b3bcbdb38db320e67ff794
B/valid/mic1/B_B_MF3_766_SportsComplex_378_snr=11.235221352480293.wav 1659434496 21366 9aecadea9fce120a73e54301f2d10c9fd1d69619404e1545e59573a29b9aa208
B/valid/mic1/B_CC_MF3_8_SportsComplex_779_snr=14.498331680152027.wav 1733172224 21280 b88ba111bc3c5c962bb51b937c6e1f19c8146b1d1b5fa100a7d2851aa1ad8c03
B/valid/mic1/B_CC_MF4_747_EoeunHill_55_snr=11.429662153802784.wav 1712268800 21250 35d157cca6f5eae1b19958da493698da4e4e2909d4bc9ed3e5c54c0ef77ee695
B/valid/mic1/B_C_N_707_WestDoor_240_snr=10.152649511479442.wav 1729824768 21127 388b2a824ae0f409d91258c96eb63bb4b730f10286887550efab0b196a07b858
B/valid/mic1/B_L_MF4_641_DuckPond_870_snr=14.432219550856429.wav 1718943232 21225 f79b45290ad78cda041a663489cffe66433291cdcac8d96c6354936e1bd09eb5
B/valid/mic1/B_L_MF4_645_SportsComplex_1076_snr=11.449415448699328.wav 1725736960 21165 56a45bbdda89e78c8a2b31966402b0aa5491d60671831439308284752ad4681a
B/valid/mic1/B_L_MF4_664_WestDoor_481_snr=11.343854436612466.wav 1630829568 21198 de71476a40caf67ef5f6741a3e49d4ce559fd68aa26fee1c4a2e640d351eca56
B/valid/mic1/B_L_MF4_714_WestDoor_264_snr=11.507609740672756.wav 1793024512 21150 22a216d4fe4201658ffdf1fffa9f18b16311721ca8088114d2171cb4ba3105ac
B/valid/mic1/B_L_MF4_740_EoeunHill_883_snr=10.060224618088368.wav 1645663232 21144 390b5af242ffeb2466d28d2d7c44ebd876abd8a9a04d0e00e24bf936656a5224
B/valid/mic1/B_L_MF4_760_SportsComplex_239_snr=12.928131494906005.wav 1618432000 21338 9a187ad4004559308e114e539bc257f0a761da1478f5be6b7cb16e85d25220a6
B/valid/mic1/B_R_MF4_644_WestDoor_704_snr=13.82880280459193.wav 1693471232 21028 3bf3425ce800fa45fce62c63c98edfff2e0916466687c83cb34d4f0229d71058
B/valid/mic1/B_R_MF4_645_EoeunHill_790_snr=10.499972117931476.wav 1665485312 20964 373d9821d208b81894f9d0f1d2a11de5f498e297c3ae66f6dfecafe023000438
B/valid/mic1/B_R_MF4_688_EoeunHill_413_snr=11.85417067233217.wav 1794008576 20924 59adefd93a50fa2fe464b6b5fa935478bad5cc62591bfb7b502b3efadf2b5739
B/valid/mic2/B_L_MF4_657_WestDoor_336_snr=12.59525032745589.wav 2042579456 21346 385427214855fc2115ed894c5112cebcc8d82e2121fafaa9b5494229069d82be
B/valid/mic2/B_L_MF4_664_WestDoor_481_snr=11.343854436612466.wav 1867120640 21429 d78f450fdcd5c84854fa9ccafb50bfe78d966c359ca541793608137f278dae48
B/valid/mic2/B_L_MF4_670_EoeunHill_173_snr=10.52267651357632.wav 2014330880 21385 95b2adcd80ff7ee9f65c9d042cbb027d6ed85f372cbc1a2cabac7328603d9e26
B/valid/mic2/B_L_MF4_681_DuckPond_465_snr=11.13281171252146.wav 1953555456 21523 8c1edfea04615db4fe9e37fd2faf6137dfe9df43be5d170bf933fcbefec131ff
B/valid/mic2/B_L_MF4_682_WestDoor_289_snr=10.003786865094904.wav 2004204544 21470 32dfd12966be756f1bddfd96a96c02ca2c60fee8029b9ee13428d9db71072b58
B/valid/mic2/B_L_MF4_714_WestDoor_264_snr=11.507609740672756.wav 2028817920 21319 ef0a2d98d11fffe086fc906810c81728dc1f2a4e69b3139a950dc45e08df4066
B/valid/mic2/B_L_MF4_740_EoeunHill_883_snr=10.060224618088368.wav 1881880576 21236 cd85dc248784d906c61ebbf1c4324d374a994b49cc4c37c957fd1048a1a257a7
B/valid/mic2/B_L_MF4_778_ConstructionSite_230_snr=11.490373153467367.wav 1983537152 21507 afa3e241470ba6695809d106cb7ab231dba5dbe0072f7fc003987d73cc5f158f
B/valid/mic2/B_L_MF4_779_ConstructionSite_509_snr=12.32934378407155.wav 1856589312 21459 8ba3e2927baf2e6d57202304fe8389767af149a1b5d30188683c914fbc8e060c
B/valid/mic2/B_L_MF4_795_EoeunHill_448_snr=11.171024771909586.wav 1871812096 21202 5689ff02d97b5c2690b9b87e70dbd624f64be774cfbb576564253c32218d9040
B/valid/mic2/B_L_N_660_SportsComplex_715_snr=12.725392634150655.wav 1978906624 21357 bc5b01ce5bcf0eab047f82a263192cf4cbd688799cba2fcf11cc2aa644f63307
B/valid/mic2/B_L_N_678_SportsComplex_716_snr=10.550663156056654.wav 1911398400 21264 b994b51a0d091a4a6760d83da5d6a4ed6e064a64de3362e07396d7c58e66ca88
B/valid/mic2/B_L_N_679_EoeunHill_400_snr=11.669135601188689.wav 2031390720 21242 84723a08808baf11a8e4b550578497cf0c5319635528c3faab427153c66d239c
B/valid/mic2/B_L_N_745_ConstructionSite_144_snr=12.874887825668104.wav 1863270400 21323 aa5315ffee3861de0729ba5da5737dadfaf7dd2396b5c4e29cad5868e3260f33
"""
_ALTERNATE_FLAC = {
    name: _FlacAudit(int(offset), int(size), digest)
    for name, offset, size, digest in (line.split() for line in _FLAC_AUDIT_ROWS.splitlines())
}


def _read_verified_flac(raw: bytes, known: _FlacAudit, member: str) -> tuple[np.ndarray, str]:
    """Decode one audited libsndfile exception, never conceal or pad bad data."""
    import subprocess

    from imageio_ffmpeg import get_ffmpeg_exe

    if len(raw) < 42 or raw[:4] != b"fLaC" or raw[4] & 0x7F or raw[5:8] != b"\x00\x00\x22":
        raise ValueError(f"{member}: missing FLAC STREAMINFO")
    info = int.from_bytes(raw[18:26], "big")
    rate = info >> 44
    channels = ((info >> 41) & 7) + 1
    bits = ((info >> 36) & 31) + 1
    count = info & ((1 << 36) - 1)
    streaminfo_md5 = raw[26:42]
    if (rate, channels, bits, count) != (16000, 1, 24, 8000):
        raise ValueError(
            f"{member}: unexpected FLAC STREAMINFO {rate=}, {channels=}, {bits=}, {count=}"
        )

    result = subprocess.run(
        [
            get_ffmpeg_exe(),
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-err_detect",
            "crccheck+explode",
            "-f",
            "flac",
            "-i",
            "pipe:0",
            "-f",
            "s32le",
            "-acodec",
            "pcm_s32le",
            "pipe:1",
        ],
        input=raw,
        capture_output=True,
        timeout=15,
        check=False,
    )
    if result.returncode or result.stderr:
        raise ValueError(
            f"{member}: strict FLAC decode failed: {result.stderr.decode(errors='replace')}"
        )
    pcm = result.stdout
    if len(pcm) != 4 * count or hashlib.sha256(pcm).hexdigest() != known.pcm_s32le_sha256:
        raise ValueError(f"{member}: FFmpeg PCM differs from the full-length reference decode")
    # FFmpeg returns left-aligned 24-bit samples in signed 32-bit words. FLAC's
    # own checksum covers signed little-endian samples at natural width.
    bytes_per_sample = np.frombuffer(pcm, dtype=np.uint8).reshape(-1, 4)
    if (
        bytes_per_sample[:, 0].any()
        or hashlib.md5(bytes_per_sample[:, 1:].tobytes()).digest() != streaminfo_md5
    ):
        raise ValueError(f"{member}: FFmpeg PCM fails the publisher's FLAC audio MD5")
    audio = np.frombuffer(pcm, dtype="<i4").astype(np.float32)
    audio *= 2.0**-31
    return audio, streaminfo_md5.hex()


def _decode(fh, tar_name: str, mem: _Member) -> tuple[np.ndarray, dict | None]:
    """One released member → (8000 float32 samples, alternate-decoder record or None)."""
    import soundfile as sf

    fh.seek(mem.offset)
    raw = fh.read(mem.size)
    member_path = f"{tar_name}:{mem.name}"
    if len(raw) != mem.size:
        raise ValueError(f"{member_path}: short TAR member read")
    container = "WAV" if mem.split == "train" else "FLAC"
    known = _ALTERNATE_FLAC.get(mem.name)
    if known is None:
        with sf.SoundFile(io.BytesIO(raw)) as snd:
            observed = (snd.format, snd.subtype, snd.samplerate, snd.channels, snd.frames)
            if observed != (container, "PCM_24", 16000, 1, 8000):
                raise ValueError(f"{member_path}: unexpected audio format {observed}")
            samples = snd.read(dtype="float32")
        if samples.shape != (8000,):
            raise ValueError(f"{member_path}: decoded {samples.size}, expected 8000 samples")
        return samples, None
    if container != "FLAC" or (mem.offset, mem.size) != (known.offset, known.size):
        raise ValueError(f"{member_path}: TAR member differs from the FFmpeg audit")
    samples, streaminfo_md5 = _read_verified_flac(raw, known, member_path)
    return samples, {
        "source_member": mem.name,
        "source_sha256": hashlib.sha256(raw).hexdigest(),
        "decode_status": "complete_verified_by_ffmpeg",
        "pcm_s32le_sha256": known.pcm_s32le_sha256,
        "streaminfo_pcm24le_md5": streaminfo_md5,
    }


def build(root: Path) -> Iterator[tuple[str, td.Frame]]:
    """Zenodo 7779574 → one mono Frame per distinct released waveform.

    The ``mic1``/``mic2`` files of one mixing draw (same split, background,
    background index and exact SNR string) are separately recorded and
    separately cut, so they are never stacked as synchronous channels; each
    Frame names the other file of its draw. When both files of a draw carry
    identical decoded PCM and the same name identity, one Frame holds the single
    waveform and both released file ids. Keys:
    ``<drone>/<split>/<mic1|mic2>/<original basename>`` or
    ``<drone>/<split>/mic1=mic2/<mic1 basename>`` (``.wav`` dropped)."""
    from data_processing.sources._common import meta_frame

    root = Path(root)
    disputed = 0
    for drone in DRONES:
        path = root / f"drone_{drone}.tar"
        size, _ = ZENODO_FILES[path.name]
        if path.stat().st_size != size:
            raise ValueError(f"{path} is not the pinned {size}-byte archive")
        with open(path, "rb") as fh:
            for pair in _tar_pairs(path, drone):
                conflict = _identity(pair[0].match) != _identity(pair[1].match)
                if conflict:
                    if (drone, pair[0].match["dir"], pair[0].split) not in _DISPUTED_SCOPE:
                        raise ValueError(
                            f"{pair[0].name} vs {pair[1].name}: unexpected name conflict"
                        )
                    disputed += 1
                decoded = [_decode(fh, path.name, mem) for mem in pair]
                if not conflict and np.array_equal(decoded[0][0], decoded[1][0]):
                    groups = [(pair, decoded[0][0])]
                else:
                    groups = [((mem,), decoded[i][0]) for i, mem in enumerate(pair)]
                alternate = {mem.mic: rec for mem, (_, rec) in zip(pair, decoded) if rec}
                for members, samples in groups:
                    key, meta = _zenodo_meta(
                        meta_frame, path.name, members, pair, conflict, alternate
                    )
                    audio = td.uniform(samples, 16000, dims=("time",))
                    yield key, td.Frame({"audio": audio, "meta": meta})
    if disputed != _DISPUTED_DRAWS:
        raise ValueError(f"{disputed} name-conflicting draws, expected {_DISPUTED_DRAWS}")


def _zenodo_meta(meta_frame, tar_name, members, pair, conflict, alternate):
    m = members[0].match
    drone, direction, seg = m["drone"], m["dir"], int(m["seg"])
    health, unit = _health(m["fault"])
    mics = [mem.mic for mem in members]
    role = "=".join(mics)
    split = members[0].split
    base = members[0].name.rsplit("/", 1)[-1][: -len(".wav")]
    partner = None if len(members) == 2 else pair[1] if members[0] is pair[0] else pair[0]
    disputed = conflict and members[0].mic == "mic2"
    partner_match = None
    if disputed:
        if partner is None:
            raise ValueError(f"{members[0].name}: disputed file without a mixture partner")
        partner_match = partner.match
    meta = meta_frame(
        f"{split}/{role}/{base}",
        ZENODO_DATASET,
        system={
            "category": "multirotor_uav",
            "platform": DRONES[drone],
            "rig": f"kaist_drone_{drone}",
        },
        observation={"mic_placement": "onboard", "type": "onboard_mic_tethered", "mics": mics},
        operating={
            "condition": "tethered",
            "maneuver_direction": DIRECTIONS[direction],
            "contains_rotor_noise": True,
            "external_source": "other",
            "external_source_detail": m["bg"],
        },
        label={
            "subcategory": f"drone-{drone}_{health}_artificial-mix",
            "split": split,
            "health": health,
            "fault_code": m["fault"],
            "fault_unit": unit,
            "maneuver_direction_code": direction,
            "label_disputed": disputed or None,
            "mixture_partner_fault_code": None if partner_match is None else partner_match["fault"],
            "mixture_partner_segment_index": None
            if partner_match is None
            else int(partner_match["seg"]),
        },
        extra={
            "original_files": [f"{tar_name}:{mem.name}" for mem in members],
            "mixture_partner_file": None if partner is None else f"{tar_name}:{partner.name}",
            "parent_window": None
            if disputed
            else f"{drone}_{direction}_{m['fault'] or 'nofault'}_{role}_{seg}",
            "window_offset_s_inferred": None if disputed else seg * 0.25,
            "background_index": int(m["bgidx"]),
            "snr_db": float(m["snr"]),
            "alternate_decode_members": {mic: alternate[mic] for mic in mics if mic in alternate}
            or None,
        },
    )
    return f"{drone}/{split}/{role}/{base}", meta


def download_icsv31(dest: Path) -> Path:
    """Fetch the three official challenge archives (skips files already at size)."""
    from data_processing.downloaders import http_fetch

    dest = Path(dest)
    for fname, (file_id, size, _) in ICSV31_FILES.items():
        http_fetch(
            f"https://drive.usercontent.google.com/download?id={file_id}&export=download&confirm=t",
            dest / fname,
            expected_size=size,
        )
    return dest


def _md5(path: Path) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 20):
            digest.update(block)
    return digest.hexdigest()


_ICSV31_NAME = re.compile(
    r"^(?P<split>train|eval|test)_(?P<drone>[ABC])_(?P<dir>Front|Back|Right|Left|Clockwise|"
    r"CounterClockwise)(?:_(?P<flag>normal|anomaly))?_(?P<idx>\d+)\.wav$"
)
_ICSV31_ORIG = re.compile(
    r"^(?P<drone>[ABC])_(?P<dir>CC|F|B|R|L|C)_(?P<fault>N|MF\d|PC\d)_(?P<seg>\d+)\.wav$"
)


def _icsv31_test_labels(root: Path) -> dict[str, dict[str, str]]:
    import csv

    with zipfile.ZipFile(root / "ICSV31AIChallengeDataset_test_labels.zip") as zf:
        rows = csv.DictReader(io.TextIOWrapper(zf.open("test_labels_sorted.csv"), "utf-8"))
        return {row["file_name"]: row for row in rows}


def build_icsv31(root: Path) -> Iterator[tuple[str, td.Frame]]:
    """ICSV31 challenge archives → one mono Frame per clip (train PCM24 WAV;
    eval/test are 24-bit FLAC streams despite the ``.wav`` names)."""
    import soundfile as sf

    from data_processing.sources._common import meta_frame

    root = Path(root)
    for fname, (_, size, md5) in ICSV31_FILES.items():
        path = root / fname
        if path.stat().st_size != size or _md5(path) != md5:
            raise ValueError(f"{ICSV31_DATASET}: {fname} differs from the pinned archive")
    labels = _icsv31_test_labels(root)
    for fname in ("ICSV31AIChallengeDataset.zip", "ICSV31AIChallengeDataset_test.zip"):
        with zipfile.ZipFile(root / fname) as zf:
            for info in sorted(
                (i for i in zf.infolist() if not i.is_dir()), key=lambda i: i.filename
            ):
                base = info.filename.rsplit("/", 1)[-1]
                m = _ICSV31_NAME.match(base)
                if m is None:
                    raise ValueError(f"{fname}: unexpected member {info.filename!r}")
                raw = zf.read(info)
                container = "FLAC" if raw[:4] == b"fLaC" else "WAV"
                with sf.SoundFile(io.BytesIO(raw)) as snd:
                    sr, subtype = snd.samplerate, snd.subtype
                    audio = snd.read(dtype="float32", always_2d=True)
                if audio.shape[1] != 1:
                    raise ValueError(f"{info.filename}: expected mono, got {audio.shape[1]} ch")
                split, drone = m["split"], m["drone"]
                fault, seg, orig = None, None, None
                if split == "test":
                    row = labels[base]
                    flag = row["label"]
                    orig = row["orig_file_name"]
                    om = _ICSV31_ORIG.match(orig)
                    if om is None or om["drone"] != drone or om["fault"] != row["fault_type"]:
                        raise ValueError(f"{base}: inconsistent test label row {row}")
                    fault, seg = om["fault"], int(om["seg"])
                else:
                    flag = m["flag"]
                    fault = "N" if flag == "normal" else None
                if fault is not None:
                    health, unit = _health(fault)
                else:
                    health, unit = "anomaly_unspecified", None
                if (flag == "normal") != (health == "normal"):
                    raise ValueError(f"{base}: label {flag!r} vs fault {fault!r}")
                stem = base[: -len(".wav")]
                meta = meta_frame(
                    stem,
                    ICSV31_DATASET,
                    system={
                        "category": "multirotor_uav",
                        "platform": DRONES[drone],
                        "rig": f"kaist_drone_{drone}",
                        "n_rotors": 4,
                    },
                    observation={
                        "mic_placement": "onboard",
                        "type": "onboard_mic_tethered",
                        "microphone": "RØDE Wireless GO II (one channel; which of the two "
                        "not documented)",
                        "environment": "anechoic chamber + mixed-in campus/DEMAND background",
                    },
                    operating={
                        "condition": "tethered",
                        "maneuver_direction": ICSV31_DIRECTIONS[m["dir"]],
                        "contains_rotor_noise": True,
                        "external_source": "other",
                        "external_source_detail": "artificially added background (campus "
                        "pond/hill/gate or DEMAND park/square/traffic; per-clip not given)",
                    },
                    label={
                        "subcategory": f"drone-{drone}_{health}_artificial-mix",
                        "split": split,
                        "rig": f"drone_{drone}",
                        "health": health,
                        "anomaly_flag": flag,
                        "fault_code": fault,
                        "fault_unit": unit,
                        "mixed": "artificial",
                        "challenge_rule": "eval/test must not be used as training data for "
                        "challenge submissions"
                        if split != "train"
                        else None,
                    },
                    extra={
                        "source_kind": "artificial_mixture",
                        "original_files": [f"{fname}:{info.filename}"],
                        "original_container": container,
                        "original_subtype": subtype,
                        "test_label_orig_file_name": orig,
                        "parent_group_2s": f"{drone}_{_ICSV31_DIRS_SHORT[m['dir']]}_{fault}_{seg}"
                        if seg is not None
                        else None,
                        "parent_segment_index_2s": seg,
                        "segment_s": audio.shape[0] / sr,
                        "processing": "ICSV29 anechoic 48 kHz recording -> 16 kHz -> 2 s "
                        "segment -> mixed with background at SNR -5..5 dB (organisers)",
                        "related_release": "doi:10.5281/zenodo.7779574",
                    },
                )
                series = td.uniform(np.ascontiguousarray(audio[:, 0]), sr, dims=("time",))
                yield f"{split}/{stem}", td.Frame({"audio": series, "meta": meta})


_ICSV31_DIRS_SHORT = {
    "Front": "F",
    "Back": "B",
    "Right": "R",
    "Left": "L",
    "Clockwise": "C",
    "CounterClockwise": "CC",
}
