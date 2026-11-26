"""Hear-and-Avoid source: TU Delft ground recordings of general-aviation overflights.

Publisher: D. Wijnker, T. van Dijk, M. Snellen, G. de Croon, C. De Wagter (TU Delft),
DataverseNL doi:10.34894/ARULTD (handle 10411/VXZ1AQ, dataset id 21082), version 1.0
(versionId 2718) released 2020-02-14, licence CC0-1.0 with no terms of use or
access restrictions. Paper: "Hear-and-avoid for UAVs using convolutional neural
networks", IMAV 2019 (journal version IJMAV 2021, doi 10.1177/1756829321992137).

The deposit is EXTERNAL conventional-aircraft audio: general-aviation propeller
aircraft and helicopters flying over a microphone array on the ground at
Lelystad Airport (NL) on 2018-06-29. It does NOT contain the paper's Parrot
Bebop onboard (UAV ego-noise) microphone recordings, nor its per-second flyover
labels, ADS-B data or microphone coordinates.

Two recorders, two record families (``meta.label.subcategory``):

- ``array_channel`` — ``wav/flight_01..82.wav``: mono 50 kHz PCM_16, one
  microphone of TU Delft's 64-microphone "acoustic camera" (8 spiral bundles of
  8 PUI Audio POM-2735P-R capsules, foam-covered, DAQ at 50 kHz; IMAV 2019
  §2.1.2). A DERIVATIVE of the 64-channel capture: one channel (index not
  published), amplitude rescaled by the publisher (e.g. flight_01 holds only 62
  distinct sample values, 1047-1048 LSB apart, so not exact multiples of one
  step — inferred from the samples; what the gain was, e.g. peak normalisation, is not
  documented). ``recordings.csv`` gives each flight's local start time and the
  number of the phone video it belongs to.
- ``phone_audio`` — the AAC-LC stereo 48 kHz audio track of
  ``Videos/VID_1..11.mp4``, recorded by the phone placed at the array centre
  (IMAV 2019 §2.1.2). Native lossy audio of a different recorder; the video
  pixels are never fetched into the raw tree (the fetcher stream-copies the
  audio track to ``phone/VID_n.m4a`` and keeps the container's ffprobe record
  as ``phone/VID_n.ffprobe.json``).

Clocks: the array files carry only ``recordings.csv`` local start times
(second resolution, recording-PC clock, zone unrecorded: Lelystad local time =
CEST); the phone files carry an Android ``creation_time`` (UTC, second
resolution). Read as the END of the video (CEST = UTC+2), 57 of the 80
video-linked flights start inside [creation_time - duration, creation_time]
(as the START: 13 of 80), and for 10 of 11 videos the first listed flight
starts 11-183 s after that inferred video start; VID_3 contains none of its
flights under either reading, and the later flights of VID_2/4/7/9/11 start
after the inferred video end. The builder
therefore publishes, per array file, an ESTIMATED offset into its companion
video (low confidence: clock-tag inference between two devices, no acoustic
alignment) only when the flight start falls inside that span; nothing is
resampled or time-shifted.

Each Frame is one whole file on its own clock (``t_start`` = 0): ``audio``
``(time,)`` (array WAV, PCM16 held as exact float32) or ``(mic, time)``
(phone, 2 AAC channels decoded by ffmpeg) at the native rate, and ``meta``.
The two recorders are never merged: their link is a clock-tag estimate, not a
shared sample clock. No ``mic_pos``:
``meta.geometry`` records the known constraints and flags the unknowns.

Requires the ``ffmpeg`` and ``ffprobe`` executables on ``PATH`` (no Python
binding): the fetcher probes each mp4 and stream-copies its audio (no decode),
the builder decodes ``phone/*.m4a`` to float32 through an ffmpeg pipe. The
array WAVs need only soundfile. The published pin decodes with Apocrita's
``module load ffmpeg`` (6.1.1); Nix 9.0.x trims the AAC tail differently.
"""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import tdseries as td

from data_processing.downloaders import http_fetch
from data_processing.sources._common import meta_frame

DATASET = "Hear-and-Avoid"
DOI = "10.34894/ARULTD"
PAGE_URL = "https://dataverse.nl/dataset.xhtml?persistentId=doi:10.34894/ARULTD"
FILE_URL = "https://dataverse.nl/api/access/datafile/{file_id}"

ARRAY_SR = 50_000
PHONE_SR = 48_000
PHONE_CHANNELS = 2
#: Lelystad local time on 2018-06-29 (CEST) relative to UTC.
LOCAL_UTC_OFFSET = timedelta(hours=2)

#: (Dataverse file id, file name, bytes, SHA-1) of every array file in dataset
#: version 1.0 (``/api/datasets/:persistentId`` file listing).
ARRAY_FILES: tuple[tuple[int, str, int, str], ...] = (
    (21114, "flight_01.wav", 377544, "f6f3a4c665326869c73105eb2355154f306b3a58"),
    (21115, "flight_02.wav", 9822544, "dbed9eec4b7072b04e2c9ec3db406fef08d9d511"),
    (21116, "flight_03.wav", 8107544, "a8b14210bd74e7be7e156893832811fa61e1eec0"),
    (21117, "flight_04.wav", 9360044, "2e09a2724581a6f9c2c6b764fe480839887f3bf9"),
    (21118, "flight_05.wav", 41482, "2d67bb1dd6fc6bb73f7aedfa3d9f644aff1dd718"),
    (21119, "flight_06.wav", 17405044, "c16d27492c5abfb6eeef1858525e804f71e275bb"),
    (21120, "flight_07.wav", 9972544, "482ca00c02edbe49fe0eff8236e67dfb0481db1d"),
    (21121, "flight_08.wav", 2035044, "60b0d8d8c78b8b6b8798128446b0f1936ea94b86"),
    (21122, "flight_09.wav", 1762544, "238c7e695bf313b593f54d3ef47d8fff91957d52"),
    (21123, "flight_10.wav", 2887544, "10331c256727b54142d5692d5b63136513d73198"),
    (21124, "flight_11.wav", 3725044, "28b91125a5601d3791b9e68d45e9bb9fe0575e1a"),
    (21125, "flight_12.wav", 3052544, "9542ef25f4bacf7b62f74d6a9c5c5a86fd55c474"),
    (21126, "flight_13.wav", 3922544, "211f153fb5598585b84067d89f240475864e25be"),
    (21127, "flight_14.wav", 11812544, "0a3e087b5ecb5fa0775b4b8be29f3adc9b85eb40"),
    (21128, "flight_15.wav", 3652544, "7345aa1cd655bc312f7a1177a1e4ad0c95cb838a"),
    (21129, "flight_16.wav", 1757544, "7f9ce74d19ec4ad17190abded999e303c2073fb3"),
    (21130, "flight_17.wav", 1817544, "dceb0ca741d4b7e35d3256b1d7e765f107e603f6"),
    (21131, "flight_18.wav", 10752544, "f329b2ed12f08b3452958cdae10a4626f3f20afb"),
    (21132, "flight_19.wav", 4967544, "2fcb50e756de5a925beec2aec1f7e5103461d6ab"),
    (21133, "flight_20.wav", 3297544, "49a87e90e107271dd3211f9457f37c95b711e23a"),
    (21134, "flight_21.wav", 5832544, "18fcc7d6e4d08644ab12f012d23b068a8c1c0e61"),
    (21135, "flight_22.wav", 3257544, "feb1bac49a4e85287c15e7836d05bbdc6e6e6b5d"),
    (21136, "flight_23.wav", 1465044, "73ce4932ea61cf581771b26ae49be942b66598e0"),
    (21137, "flight_24.wav", 385044, "652e0f34eb834f7a5cc6317e28372694f8bca7ad"),
    (21138, "flight_25.wav", 3252544, "9ec978aa548c72d1531d343c6386aa41dade868e"),
    (21139, "flight_26.wav", 9220044, "2853ccf97b3f4fd927f509db6c40007c22dafcf5"),
    (21140, "flight_27.wav", 3760044, "f183a3c7c2c78dfa2f3eeca3a88f1d234008c406"),
    (21141, "flight_28.wav", 6647544, "10302ce5a35ef59cb9c6ee24fc98b9e2da610336"),
    (21142, "flight_29.wav", 2935044, "aeff35ecd17ab9a2f0d6cf49c1a233a626ee4b5a"),
    (21143, "flight_30.wav", 3730044, "fe3f65e5bbb96e2b4a9fccfacf449c0f85f9a7a8"),
    (21144, "flight_31.wav", 5520044, "3ee7038db8dac63edc5ad26ff17a72b1b44e2c38"),
    (21145, "flight_32.wav", 7140044, "19c58ed40a6e2688f99a9dbbb332e773b8fdab16"),
    (21146, "flight_33.wav", 3270044, "afb7b3953011e26e86b6b8fd0dc08bfacd375070"),
    (21147, "flight_34.wav", 8290044, "2d93369f6e573d212c52a26c9dff8642f039d4e1"),
    (21148, "flight_35.wav", 8642544, "7cc816f186b931814bd016fe3c479ebadfb0730e"),
    (21149, "flight_36.wav", 22410044, "93a4e7b7161b49a705eb3ea14719f27ab4fd9104"),
    (21150, "flight_37.wav", 18982, "f5f825a8ffde6d056b1b3cc66884498d388ad94c"),
    (21151, "flight_38.wav", 2740044, "88bd9a47f8f4b91ec112f0929d54c8283c781862"),
    (21152, "flight_39.wav", 4650044, "9e1762798c2acb03623ba79b8c04f55193109ee1"),
    (21153, "flight_40.wav", 4752544, "45ff289d971199a41fcf94c56079936d8de2842a"),
    (21154, "flight_41.wav", 11627544, "2d50a22aff949c704098bc75867f01f3a87556fb"),
    (21155, "flight_42.wav", 5727544, "e0ec7c03fab1497c911417bf9b1cecee04537e23"),
    (21156, "flight_43.wav", 301482, "9bc914c9f59185cf4848c63310879be67722af8e"),
    (21157, "flight_44.wav", 1425044, "f3e5a18873ccc02660b2972c8aba704abfc3aca4"),
    (21158, "flight_45.wav", 5080044, "9147d7226c99f38dfee84a8ba25149eac4a842c4"),
    (21159, "flight_46.wav", 5937544, "3f148219e9d80a235843ee088c5eff127980dfff"),
    (21160, "flight_47.wav", 13167544, "a5d57d5e050e0697a582cd8768fe14fc022c4bed"),
    (21161, "flight_48.wav", 11217544, "f1753382a846e6fa027f13edef0627c9902b95bd"),
    (21162, "flight_49.wav", 1780044, "bdd637e5bb6bc8d7698eba446157b7aa72a10129"),
    (21163, "flight_50.wav", 3450044, "a952c23cfe2d5200b7a468d3c5149f7d5fb63c4c"),
    (21164, "flight_51.wav", 14295044, "a50ab0d2b6e2f2928e433e153d172f51dd3aa530"),
    (21165, "flight_52.wav", 15440044, "0dcde39f00b1958ca6e93bacc310ebeb3b943915"),
    (21166, "flight_53.wav", 10935044, "914b67bc985e5a49f7f86274fad3c9eceaa254f2"),
    (21167, "flight_54.wav", 14092544, "a0c9ccacad09c0da4288d516ff1cbf0a1d7c14e8"),
    (21168, "flight_55.wav", 4372544, "b7b42aa3b2fd9da6a552fcc9e859a78e4204846c"),
    (21169, "flight_56.wav", 7467544, "41222331ce3266800b79764c87e79dffd98d121b"),
    (21170, "flight_57.wav", 13277544, "78fcb25cbb8c5467a974647559fa5354a7b24721"),
    (21171, "flight_58.wav", 13977544, "e52b6c1743cbb359329da705b5de7597e8943e2e"),
    (21172, "flight_59.wav", 6600044, "6562ea39f8b7ab18e85eb999026b35378882d962"),
    (21173, "flight_60.wav", 10632544, "ef824a56882ef2e6c2e96223d764ebb6e5cfb129"),
    (21174, "flight_61.wav", 2465044, "755a4e6e362e9228e0208d6d736bfe2f4047804e"),
    (21175, "flight_62.wav", 5060044, "bfee6717ff9a6d56e1910be7360018bbb168659c"),
    (21176, "flight_63.wav", 12177544, "08820858a890a4d42cd75b1f9dcb816584332dbc"),
    (21177, "flight_64.wav", 23087544, "f01e6ad8d213161388d7f5cff9e9605538eb9467"),
    (21178, "flight_65.wav", 16947544, "ed5373387ec597fbf5e9e9a22e18a40c8b101fb3"),
    (21179, "flight_66.wav", 11605044, "8bf2a2e142a876240013222bfde09cebac376424"),
    (21180, "flight_67.wav", 21670044, "960a7d589f69271a15478dffd2151570693c0079"),
    (21181, "flight_68.wav", 9500044, "2500b77ac3e837bb294a30d513e72c4a64029e05"),
    (21182, "flight_69.wav", 434966, "975214c36b35e7b7595ba84b6821540ffddfe676"),
    (21183, "flight_70.wav", 8287544, "00e484c396933a72db2c9b7e5c6664fa5d6ce90a"),
    (21184, "flight_71.wav", 35475044, "32d81b47906913b5173d77ce062c6d385cdcca45"),
    (21185, "flight_72.wav", 14760044, "27ed9f0f087daf9e8eb49b7a30d4af48d5c919dc"),
    (21186, "flight_73.wav", 5400044, "0c3fbfe6ef20e8aa555f4219a7709629901db5c1"),
    (21187, "flight_74.wav", 5652544, "54d83ee91cd4aad964d0aff24bfe76559fa51163"),
    (21188, "flight_75.wav", 7800044, "4a1961818726cc52bbe1e8103e6b91974deaf91c"),
    (21189, "flight_76.wav", 6047544, "ea7aa3ff9f448cdd7d88d24fda0ff3b24d108462"),
    (21190, "flight_77.wav", 5470044, "da490d4e099fcbf7c485faaf325e7ae829ed0c70"),
    (21191, "flight_78.wav", 5332544, "550a483febe0a0e23a1c8f4d6f77adf13c28f008"),
    (21192, "flight_79.wav", 5147544, "908ffa2060aa80ccd1638f9fb19fc4420cfd39be"),
    (21193, "flight_80.wav", 6730044, "26b8b7192f7753d31b52d0bd062ea76e84489ef8"),
    (21194, "flight_81.wav", 6220044, "415b0775fb2f812d5f8ba86328f98efc44c81d8c"),
    (21195, "flight_82.wav", 5435044, "c932774d8d0681b09035d09fc68da8d005b5f8f7"),
    (21200, "recordings.csv", 2583, "8de7d04f307df88c6790daddcf5d87fb0a116fa3"),
)
#: (Dataverse file id, file name, bytes, SHA-1) of the phone videos (``Videos/``).
VIDEO_FILES: tuple[tuple[int, str, int, str], ...] = (
    (21204, "VID_1.mp4", 169430825, "9465a0fe44f68a8eccb038013daff2c31411e680"),
    (21207, "VID_2.mp4", 99992827, "68459a1874f5f652c7801cdc67f83d9ce1df9cfa"),
    (21198, "VID_3.mp4", 102768911, "f7a5d582bcc66e142d45c831b776a9528ba86e68"),
    (21199, "VID_4.mp4", 229411886, "17e221f4e2671d2ded9978a438841d4256ca187f"),
    (21205, "VID_5.mp4", 185543293, "e5d96fb62ef6654f0c083635f3a1673b3d944081"),
    (21209, "VID_6.mp4", 275793293, "d34c5e8e140984179f1a3d83c50184c70c463ad3"),
    (21203, "VID_7.mp4", 428499353, "edb3690365ad9c452298fe6e9b8e54d295ed35e2"),
    (21197, "VID_8.mp4", 437344331, "1a4240502d3f17d0563a7ac0851a7baaa376d922"),
    (21202, "VID_9.mp4", 470077608, "a8b9257f2a4b40c82d92c967d52cc4047e5e3f13"),
    (21201, "VID_10.mp4", 917174557, "8703e66ed9bd892543f6d5dd37630157cc92e300"),
    (21206, "VID_11.mp4", 369152564, "1b6c3bbdf7717c8782fa1066a37131d4866836a7"),
)
INDEX = "recordings.csv"
PHONE_DONE = ".done"

_SITE = "Lelystad Airport (NL), unused runway extension under the approach path"

ARRAY_GEOMETRY: dict[str, Any] = {
    "mic_pos_quality": "unknown",
    "placement": "external: ground array (TU Delft acoustic camera, 64 mics in 8 spiral bundles "
    "of 8, foam-covered against wind and ground reflections) at " + _SITE,
    "channel_map": "one microphone of the 64; which one is not published (IMAV 2019: one of the 64 "
    "was faulty and excluded)",
    "unknown": "microphone index and coordinates, array layout coordinates, aircraft altitude/track",
    "evidence": "IMAV 2019 §2.1.2 and Fig. 1; IJMAV 2021 p.3 and Fig. 1 (eight bundles of eight PUI "
    "AUDIO 665-POM-2735P-R in a spiral; 'only the recording of one microphone is necessary'); "
    "Dataverse deposit description",
}
PHONE_GEOMETRY: dict[str, Any] = {
    "mic_pos_quality": "unknown",
    "placement": "external: mobile phone (Android 8.0.0) placed in the centre of the ground array, "
    "camera filming the overflights, at " + _SITE,
    "channel_map": "audio rows = AAC stereo channels (left, right) of the phone's built-in microphones",
    "unknown": "phone model, microphone positions/orientation, gain/AGC settings",
    "evidence": "IMAV 2019 §2.1.2 ('a mobile phone camera is placed in the center of the array'); "
    "mp4 tags (com.android.version 8.0.0, ISO-6709 location)",
}

PROVENANCE: dict[str, Any] = {
    "source_url": PAGE_URL,
    "doi": DOI,
    "handle": "hdl:10411/VXZ1AQ",
    "dataset_version": "1.0 (versionId 2718, released 2020-02-14, state RELEASED)",
    "download_url": FILE_URL,
    "license": "CC0-1.0 (DataverseNL dataset licence field; termsOfUse/termsOfAccess empty, "
    "no file restricted)",
    "attribution": "cite the dataset DOI and Wijnker et al. (scholarly norm; CC0 requires none)",
    "citation": (
        "D. Wijnker, T. van Dijk, M. Snellen, G. de Croon, C. De Wagter, 'Hear-and-Avoid: "
        "Acoustic Detection of General Aviation Aircraft for UAV', DataverseNL, V1, 2020, "
        "doi 10.34894/ARULTD; paper: 'Hear-and-avoid for UAVs using convolutional neural "
        "networks', IMAV 2019, pp. 144-156; journal: IJMAV 13 (2021), doi 10.1177/1756829321992137."
    ),
    "collection_method": (
        "2018-06-29, " + _SITE + ": general-aviation propeller aircraft and helicopters fly over a "
        "TU Delft 64-microphone ground acoustic camera (50 kHz DAQ; one channel deposited per "
        "recording) while a phone at the array centre films them (AAC stereo 48 kHz audio). "
        "External conventional-aircraft audio, not UAV onboard ego-noise."
    ),
    "equipment": "TU Delft acoustic camera (64 x PUI Audio POM-2735P-R, 8 spiral bundles), 50 kHz "
    "DAQ; Android 8.0.0 phone (AAC-LC 192 kb/s stereo 48 kHz)",
    "observation_type": "external_ground",
    "recordings": {"array_channel": len(ARRAY_FILES) - 1, "phone_audio": len(VIDEO_FILES)},
    "frames": "one whole file per Frame (no chunking)",
    "file_separation": "no file is joined to another. Array: each recordings.csv start is "
    ">= 3.35 s after the previous file's start + duration (second-resolution clock). Phone: "
    "consecutive videos are separate MP4 captures, 2.5-1431 s apart with creation_time read "
    "as the video end",
    "source_kind": {
        "array_channel": "derivative: one channel of the 64-channel capture, publisher-rescaled",
        "phone_audio": "original recording (lossy AAC as recorded, audio track stream-copied)",
    },
    "telemetry": "none (the ADS-B receiver logged nothing useful per the paper; not deposited)",
    "labels": "none deposited (the paper's per-second flyover labels are not published)",
    "splits": "none published",
    "not_converted": "video pixels of Videos/VID_*.mp4; camera_calibration.zip (camera intrinsics, "
    "calibration screenshots)",
    "description": (
        "Hear-and-Avoid: ground recordings of general-aviation propeller aircraft and helicopter "
        "overflights (and background) at Lelystad Airport: 82 mono 50 kHz single-microphone "
        "array files + the stereo 48 kHz AAC audio of 11 phone videos at the array centre, "
        "one Frame per file; flight-to-video map from recordings.csv, clock offsets estimated (low "
        "confidence)."
    ),
}


# ─── Fetch ────────────────────────────────────────────────────────────────────


def _sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _fetch_verified(file_id: int, name: str, size: int, sha1: str, path: Path) -> None:
    http_fetch(FILE_URL.format(file_id=file_id), path, expected_size=size)
    got_size, got_sha1 = path.stat().st_size, _sha1(path)
    if (got_size, got_sha1) != (size, sha1):
        path.unlink()
        raise OSError(
            f"{name} (file {file_id}): got {got_size} B sha1 {got_sha1}, want {size} B {sha1}"
        )


def _tool(name: str) -> str:
    exe = shutil.which(name)
    if exe is None:
        raise FileNotFoundError(f"{name} not on PATH (needed for the {DATASET} phone audio)")
    return exe


def download_hear_avoid(dest: Path) -> Path:
    """Fetch + verify the array WAVs and ``recordings.csv``; for each phone video
    (one at a time) fetch + verify the mp4, keep its ffprobe record, stream-copy
    its audio track to ``phone/VID_n.m4a`` and delete the mp4. Idempotent."""
    dest = Path(dest)
    for file_id, name, size, sha1 in ARRAY_FILES:
        path = dest / "wav" / name if name.endswith(".wav") else dest / name
        _fetch_verified(file_id, name, size, sha1, path)
    phone = dest / "phone"
    phone.mkdir(parents=True, exist_ok=True)
    for file_id, name, size, sha1 in VIDEO_FILES:
        stem = Path(name).stem
        done = phone / f"{stem}{PHONE_DONE}"
        if done.exists() and done.read_text(encoding="utf-8").strip() == sha1:
            continue
        mp4 = phone / name
        _fetch_verified(file_id, name, size, sha1, mp4)
        probe = subprocess.run(
            [
                _tool("ffprobe"),
                "-v",
                "error",
                "-show_format",
                "-show_streams",
                "-of",
                "json",
                str(mp4),
            ],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        info = json.loads(probe)
        info["format"].pop("filename", None)
        info["source_file"] = {
            "dataverse_file_id": file_id,
            "name": f"Videos/{name}",
            "bytes": size,
            "sha1": sha1,
        }
        (phone / f"{stem}.ffprobe.json").write_text(
            json.dumps(info, indent=1, sort_keys=True), encoding="utf-8"
        )
        audio = phone / f"{stem}.m4a"
        subprocess.run(
            [
                _tool("ffmpeg"),
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(mp4),
                "-map",
                "0:a:0",
                "-c:a",
                "copy",
                "-map_metadata",
                "0",
                "-f",
                "mp4",
                str(audio),
            ],
            check=True,
        )
        mp4.unlink()
        done.write_text(sha1 + "\n", encoding="utf-8")
    return dest


# ─── Index + clocks ───────────────────────────────────────────────────────────


def load_index(root: Path) -> dict[int, dict[str, Any]]:
    """``{flight #: {start_native, start_local, video}}`` from ``recordings.csv``."""
    out: dict[int, dict[str, Any]] = {}
    with open(Path(root) / INDEX, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter=";"):
            flight = (row.get("Flight") or "").strip()
            if not flight:
                continue
            n = int(flight)
            if n in out:
                raise ValueError(f"{INDEX}: flight {n} listed twice")
            native = row["Time"].strip()
            video = (row.get("Video") or "").strip()
            out[n] = {
                "start_native": native,
                "start_local": datetime.strptime(native, "%d-%b-%y_%H-%M-%S"),
                "video": int(video) if video else None,
            }
    return out


def load_phone_info(root: Path) -> dict[int, dict[str, Any]]:
    """``{video #: ffprobe-derived facts}`` from ``phone/VID_n.ffprobe.json``."""
    out: dict[int, dict[str, Any]] = {}
    for _, name, _, _ in VIDEO_FILES:
        stem = Path(name).stem
        info = json.loads(
            (Path(root) / "phone" / f"{stem}.ffprobe.json").read_text(encoding="utf-8")
        )
        audio = [s for s in info["streams"] if s.get("codec_type") == "audio"]
        if len(audio) != 1:
            raise ValueError(f"{stem}: {len(audio)} audio streams")
        a, tags = audio[0], info["format"].get("tags", {})
        if (int(a["sample_rate"]), int(a["channels"]), a["codec_name"]) != (
            PHONE_SR,
            PHONE_CHANNELS,
            "aac",
        ):
            raise ValueError(
                f"{stem}: audio {a['codec_name']} {a['sample_rate']} Hz x {a['channels']}"
            )
        created = datetime.strptime(tags["creation_time"][:19], "%Y-%m-%dT%H:%M:%S").replace(
            tzinfo=UTC
        )
        out[int(stem.removeprefix("VID_"))] = {
            "stem": stem,
            "source_file": info["source_file"],
            "creation_time_utc": created,
            "duration_s": float(info["format"]["duration"]),
            "audio_codec": f"{a['codec_name']} {a.get('profile', '')}".strip(),
            "audio_bit_rate": int(a["bit_rate"]) if a.get("bit_rate") else None,
            "location_iso6709": tags.get("location"),
            "android_version": tags.get("com.android.version"),
        }
    return out


def _video_window_local(v: dict[str, Any]) -> tuple[datetime, datetime]:
    """Phone video span on the local (CEST) wall clock, ASSUMING creation_time = its end."""
    end = (v["creation_time_utc"] + LOCAL_UTC_OFFSET).replace(tzinfo=None)
    return end - timedelta(seconds=v["duration_s"]), end


# ─── Build ────────────────────────────────────────────────────────────────────


_SYSTEM = {
    "category": "aircraft",
    "aircraft_types": "general-aviation propeller aircraft and helicopters (per-file type not published)",
}
_OPERATING = {"condition": "unknown", "flight_mode": "unknown", "contains_rotor_noise": "unknown"}


def _build_array(root: Path, index: dict[int, dict[str, Any]], phone: dict[int, dict[str, Any]]):
    import numpy as np
    import soundfile as sf

    wavs = {int(p.stem.removeprefix("flight_")): p for p in (root / "wav").glob("flight_*.wav")}
    want = {int(name[7:9]) for _, name, _, _ in ARRAY_FILES if name.endswith(".wav")}
    if set(wavs) != want or set(index) != want:
        raise ValueError(
            f"{DATASET}: wavs {sorted(wavs)} / index {sorted(index)} != deposit {sorted(want)}"
        )
    for n in sorted(want):
        path, row = wavs[n], index[n]
        video = row["video"]
        link: dict[str, Any] = {
            "companion_video": None if video is None else f"Videos/VID_{video}.mp4"
        }
        if video is not None:
            lo, hi = _video_window_local(phone[video])
            offset = (row["start_local"] - lo).total_seconds()
            if lo <= row["start_local"] <= hi:
                link["companion_video_offset_s_est"] = offset
                link["companion_video_offset_confidence"] = (
                    "low: phone creation_time read as the video end, CEST = UTC+2, second-resolution "
                    "clocks of two devices, no acoustic alignment"
                )
            else:
                link["companion_video_offset_confidence"] = (
                    "none: the flight start lies outside the clock-tag span of its listed video"
                )
        with sf.SoundFile(str(path)) as fh:
            if (fh.samplerate, fh.channels, fh.subtype) != (ARRAY_SR, 1, "PCM_16"):
                raise ValueError(f"{path}: {fh.samplerate} Hz x {fh.channels} ch {fh.subtype}")
            n_frames = fh.frames
            block = np.ascontiguousarray(fh.read(n_frames, dtype="float32", always_2d=False))
        if block.shape != (n_frames,):
            raise ValueError(f"{path}: read {block.shape}, header says {n_frames} frames")
        rec_id = f"wav/{path.name}"
        meta = meta_frame(
            rec_id,
            DATASET,
            system=_SYSTEM,
            observation={
                "mic_placement": "external",
                "type": "ground_array_single_channel",
                "recorder": "TU Delft acoustic camera DAQ (50 kHz)",
                "environment": "outdoor airfield",
            },
            operating=_OPERATING,
            label={"subcategory": "array_channel", "flyover_labels": "not published"},
            extra={
                "geometry": td.Frame(dict(ARRAY_GEOMETRY)),
                "flight_number": n,
                "source_kind": "derivative",
                "derivation": "one channel of the 64-channel acoustic-camera capture; amplitude "
                "rescaled by the publisher (inferred from coarse, evenly spaced sample values)",
                "raw_path": rec_id,
                "native_format": "WAV PCM_16",
                "start_time_local": row["start_local"].isoformat(),
                "start_time_native": row["start_native"],
                "start_time_zone": "local wall clock (Lelystad, CEST), zone not recorded",
                **link,
                "recording_frames": n_frames,
                "recording_duration_s": n_frames / ARRAY_SR,
            },
        )
        audio = td.uniform(block, ARRAY_SR, dims=("time",), t_start=0.0)
        yield f"array/flight_{n:02d}", td.Frame({"audio": audio, "meta": meta})
        del block


def _decode_phone(path: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            _tool("ffmpeg"),
            "-nostdin",
            "-v",
            "error",
            "-i",
            str(path),
            "-map",
            "0:a:0",
            "-f",
            "f32le",
            "-acodec",
            "pcm_f32le",
            "-",
        ],
        stdout=subprocess.PIPE,
    )


def _decode_phone_audio(path: Path) -> Any:
    """The whole decoded AAC track as a ``(2, n)`` float32 array."""
    import numpy as np

    proc = _decode_phone(path)
    stdout = proc.stdout
    if stdout is None:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"{path}: decoder started without its stdout pipe")
    try:
        buf = stdout.read()
    finally:
        stdout.close()
        code = proc.wait()
    if code != 0:
        raise subprocess.CalledProcessError(code, proc.args)
    if not buf or len(buf) % (PHONE_CHANNELS * 4):
        raise ValueError(f"{path}: empty or truncated decoder output ({len(buf)} B)")
    audio = np.frombuffer(buf, dtype="<f4").reshape(-1, PHONE_CHANNELS).T.copy()
    del buf
    return audio


def _build_phone(root: Path, index: dict[int, dict[str, Any]], phone: dict[int, dict[str, Any]]):
    decoder = subprocess.run(
        [_tool("ffmpeg"), "-version"], check=True, capture_output=True, text=True
    ).stdout.splitlines()[0]
    for v in sorted(phone):
        info = phone[v]
        path = root / "phone" / f"{info['stem']}.m4a"
        block = _decode_phone_audio(path)
        n_frames = int(block.shape[1])
        lo, hi = _video_window_local(info)
        flights = sorted(n for n, r in index.items() if r["video"] == v)
        rec_id = f"Videos/{info['stem']}.mp4#audio"
        meta = meta_frame(
            rec_id,
            DATASET,
            system=_SYSTEM,
            observation={
                "mic_placement": "external",
                "type": "phone_at_array_centre",
                "recorder": f"Android {info['android_version']} phone video, {info['audio_codec']} "
                f"{info['audio_bit_rate']} b/s",
                "environment": "outdoor airfield",
            },
            operating=_OPERATING,
            label={"subcategory": "phone_audio", "flyover_labels": "not published"},
            extra={
                "geometry": td.Frame(dict(PHONE_GEOMETRY)),
                "video_number": v,
                "source_kind": "original_recording",
                "decoder": decoder,
                "decoder_note": "decoded length depends on the ffmpeg build: 6.1.1 (Apocrita module "
                "ffmpeg/6.1.1-gcc-12.2.0, used for the 2026-11-26 pin) emits every AAC frame "
                "(VID_2: 11,065,344 = 10,806 x 1024); Nix 9.0 and 9.0.1 trim VID_2 to 11,065,327",
                "native_format": "AAC-LC stereo 48 kHz (lossy) in MP4; decoded to float32 by ffmpeg",
                "raw_path": f"phone/{info['stem']}.m4a",
                "source_file": info["source_file"]["name"],
                "source_file_sha1": info["source_file"]["sha1"],
                "creation_time_utc": info["creation_time_utc"].isoformat(),
                "location_iso6709": info["location_iso6709"],
                "est_span_local": f"{lo.isoformat()}/{hi.isoformat()}",
                "est_span_confidence": "low: creation_time read as the video end (57 of 80 "
                "linked flights start inside it; VID_3 none), CEST = UTC+2",
                "listed_flights": ";".join(str(n) for n in flights),
                "recording_frames": n_frames,
                "recording_duration_s": n_frames / PHONE_SR,
            },
        )
        audio = td.uniform(block, PHONE_SR, dims=("mic", "time"), t_start=0.0)
        yield f"phone/VID_{v:02d}", td.Frame({"audio": audio, "meta": meta})
        del block


def build(root: Path) -> Iterator[tuple[str, td.Frame]]:
    """Fetched raw dir → one Frame per array WAV, then one per phone audio track."""
    root = Path(root)
    index = load_index(root)
    phone = load_phone_info(root)
    unknown_videos = {r["video"] for r in index.values() if r["video"] is not None} - set(phone)
    if unknown_videos:
        raise ValueError(
            f"{DATASET}: recordings.csv lists videos {sorted(unknown_videos)} absent from the deposit"
        )
    yield from _build_array(root, index, phone)
    yield from _build_phone(root, index, phone)
