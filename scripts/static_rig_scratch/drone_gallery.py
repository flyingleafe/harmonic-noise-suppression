"""Rough harmonic profile + broadband floor shape for every drone we have
some fit of: DREGON bench singles (OU profiles, floors from the inter-
harmonic gaps, rotor-off from the pre-spin-up segment), DREGON flight and
Michael's cruise/standby (noise-model-v3 round-4 records: profile_db at 80
rev/s, floor spline), AVQ (round-1 full-record line powers, gap floor from
the cached audio), AGH single rotors and KU Leuven (round-1 line powers only).
Each source keeps its own dB scale (noted in the panel title).

Writes results/static_rig/gallery/gallery.png."""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.signal import welch  # noqa: E402

sys.path.insert(0, "notebooks")
import single_rotor_lab as L  # noqa: E402
from data_processing.noise_model.floor import floor_geometry  # noqa: E402
from data_processing.noise_model.spectrum import floor_ctrl_hz, floor_shape_db  # noqa: E402

OUT = Path("results/static_rig/gallery")
OUT.mkdir(parents=True, exist_ok=True)
REP = json.loads(Path("results/static_rig/report/profiles_full.json").read_text())
OU = Path("results/static_rig/single_rotor/ou")


def mic_mean_db(e):
    """Mic-mean dB of a round-1 entry's power_db (None = unread line → nan)."""
    v = np.array(
        [[np.nan if x is None else x for x in np.atleast_1d(row)] for row in e["power_db"]], float
    )
    return np.nanmean(v, axis=1)


def gap_floor(x, fs, speeds, k_max=60, rel=0.03, min_hz=8.0, nperseg=8192):
    """Median over mics of the Welch spectrum with every rotor's harmonics
    masked (±max(rel·k·s, min_hz)); returns (f, dB) on the unmasked bins."""
    f, P = welch(x.astype(np.float64), fs, nperseg=nperseg, axis=1)
    mask = np.ones(f.size, bool)
    for s in speeds:
        for k in range(1, k_max + 1):
            mask &= np.abs(f - k * s) > max(rel * k * s, min_hz)
    sel = mask & (f > 30) & (f < 0.45 * fs)
    return f[sel], 10 * np.log10(np.median(P[:, sel], axis=0))


def v3_floor(rec):
    p = rec["params"]["floor"]
    sr = rec["sr"]
    f = np.linspace(30, sr / 2, 600)
    shape, _ = floor_geometry(f, floor_ctrl_hz(sr))
    ctrl = floor_shape_db(p["floor_shape_z"], sr=sr, scale_db=p["floor_shape_sd_db"])
    return f, p["floor_mean_db"] + shape @ ctrl


rows = []  # (title, [(label, k, dB)], floor list [(label, f, dB)], note)

# DREGON bench singles at 70 %
prof, floors = [], []
for r in ["Motor1", "Motor2", "Motor3", "Motor4"]:
    key = f"motor_{r}_70"
    d = json.loads((OU / f"{key}.json").read_text())
    good = [c for c in range(8) if c not in L.WIND[r]]
    a = np.array(d["amp2"])[good]
    prof.append(
        (
            f"{r} @ {d['s']:.1f} rev/s",
            np.arange(1, a.shape[1] + 1),
            10 * np.log10(np.maximum(a, 1e-30)).mean(0),
        )
    )
    x, fs, model = L._analysed(key)
    a0, b0 = model.span
    f, db_on = gap_floor(x[good, a0:b0], fs, [d["s"]], k_max=150)
    floors.append((f"{r} on", f, db_on))
    if r == "Motor1":
        f, db_off = gap_floor(x[good, : max(a0 - fs // 2, fs)], fs, [d["s"]], k_max=150)
        floors.append(("rotor off (pre-spin-up)", f, db_off))
rows.append(
    (
        "DREGON bench, single rotors at 70 % (OU profiles; 0 dB = unit amplitude)",
        prof,
        floors,
        "floor: gap median, mics w/o wind",
    )
)

# noise-model-v3 round-4 records
for name, title in (
    (
        "dregon_room2_floor",
        "DREGON flight (v3 r4 fit, room2 floor; profile at 80 rev/s, fit's STFT units)",
    ),
    ("michaels_fly125_cruise", "Michael's FLY125 cruise (v3 r4 fit; profile at 80 rev/s)"),
    ("michaels_fly125_standby", "Michael's FLY125 standby (v3 r4 fit; profile at 80 rev/s)"),
):
    rec = json.loads(Path(f"results/noise_v3/fits_r4/{name}__flight_v3.json").read_text())
    pdb = np.array(rec["params"]["profile"]["profile_db"])
    prof = [(f"rotor {i}", np.arange(1, pdb.shape[1] + 1), pdb[i]) for i in range(pdb.shape[0])]
    f, db = v3_floor(rec)
    rows.append(
        (
            title,
            prof,
            [("floor spline", f, db)],
            f"amp_exp {rec['params']['profile']['amp_exp']:.1f}, floor_exp {rec['params']['floor']['floor_exp']:.2f}",
        )
    )

# AVQ: round-1 line powers + gap floor from the cached audio
units = {u["key"]: u for u in json.loads(Path("/tmp/sr_avq/units.json").read_text())}
for key in ["S1_seq2", "S1_seq3"]:
    ents = [e for e in REP["avq"] if e["recording"] == key]
    prof = [
        (f"{key} rotor @ {e['speed_mean']:.0f} rev/s", np.array(e["orders"]), mic_mean_db(e))
        for e in ents
    ]
    u = units[key]
    x = np.load(u["audio"], mmap_mode="r")
    n = min(x.shape[1], 60 * u["fs"])
    f, db = gap_floor(
        np.asarray(x[:, :n]),
        u["fs"],
        [e["speed_mean"] for e in ents],
        k_max=40,
        rel=0.02,
        min_hz=12.0,
    )
    rows.append(
        (
            f"AVQ {key} (round-1 full-record line powers, drifting rotors; own dB scale)",
            prof,
            [("gap floor (rough mask)", f, db)],
            "floor: all rotors' lines masked ±max(2 %, 12 Hz)",
        )
    )

# AGH single rotors (reference mic) and KU Leuven: profiles only
prof = [
    (
        f"{e['recording'].split('__')[-1]} @ {e['speed_mean']:.0f} rev/s",
        np.array(e["orders"]),
        mic_mean_db(e),
    )
    for e in REP["spcup_agh"]
]
rows.append(
    (
        "AGH single rotors, anechoic, reference mic (round-1 line powers)",
        prof,
        [],
        "no audio on the laptop → no floor",
    )
)
prof = [
    (
        f"{e['recording'].split('_')[-2]} rotor @ {e['speed_mean']:.0f} rev/s",
        np.array(e["orders"]),
        mic_mean_db(e),
    )
    for e in REP["spcup_ku_leuven"]
]
rows.append(
    (
        "KU Leuven MK quad (round-1 line powers, drifting)",
        prof,
        [],
        "no audio on the laptop → no floor",
    )
)

fig, axs = plt.subplots(len(rows), 2, figsize=(18, 4.2 * len(rows)))
for i, (title, prof, floors, note) in enumerate(rows):
    ax = axs[i, 0]
    for lab, k, db in prof:
        ax.plot(k, db, lw=1, label=lab)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("order k")
    ax.set_ylabel("line power (dB)")
    ax.legend(fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    ax = axs[i, 1]
    for lab, f, db in floors:
        ax.semilogx(f, db, lw=1, label=lab)
    ax.set_title(f"floor — {note}", fontsize=10)
    ax.set_xlabel("Hz")
    ax.set_ylabel("dB")
    if floors:
        ax.legend(fontsize=7)
    ax.grid(alpha=0.3, which="both")
fig.tight_layout()
fig.savefig(OUT / "gallery.png", dpi=70)
print("wrote", OUT / "gallery.png")
