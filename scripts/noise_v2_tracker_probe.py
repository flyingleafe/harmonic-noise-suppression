#!/usr/bin/env python
"""R5 probe (b): WHAT do the two RPS trackers respond to on DREGON-like data?

R4 left one thing unexplained. HPPNet tracks the REAL DREGON free-flight clip
(1.07 rev/s) and the LEGACY render (1.96), but not the v2 render fitted to the
real clip (69.6) — even though that render is within +2..+4 dB of the real
periodogram in every 2048-frame cell class. This runner asks two questions the
periodogram cannot answer, reusing the frozen render + score machinery:

* ``tracks`` — score BOTH trackers (frozen HPPNet `hppnet_l2_r2_s0/best` and
  the SCv2 regressor `real_r4_scv2_unified/best_real_r2.ckpt`) on the five R4
  arms, keep the per-rotor TRACKS, and decompose the error into a MEAN-track
  part (all four predictions move together) and a SPREAD part (the comb of
  four tracks). A tracker that emits an equally spaced comb around the right
  mean is reading a different thing from one that emits four flat lines.
* ``cqt`` — the MODEL'S OWN front end (`CQTLogSpecgram`: nnAudio CQT2010v2,
  fmin 27.5 Hz, 48 bins/octave, 352 bins, hop 512, `AmplitudeToDB(top_db=80)`)
  on every arm, and the per-order log-magnitude contrast of the label's own
  comb against the local floor. The constant-Q window is 859 ms at k=1 and
  54 ms at k=16 — a resolution the 2048-sample (128 ms) periodogram cells of
  round 3 never had.

    python scripts/noise_v2_tracker_probe.py tracks --out DIR --figures
    python scripts/noise_v2_tracker_probe.py cqt --out DIR --figures
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, NoReturn

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from experiments.noise_model import render as RD  # noqa: E402
from experiments.stochastic_fit import revised_eval as RE  # noqa: E402

SCHEMA = "noise-v2-tracker-probe/1"
OUT_DEFAULT = Path("results/noise_v2/rounds/round5/tracker_probe")
R3_FITS = Path("results/noise_v2/rounds/round3/fits")
R4_FITS = Path("results/noise_v2/rounds/round4/legacy_truth/fits")

#: The five arms of R4 §B, on the three frozen cruise score windows.
FIT_PATHS: dict[str, Path] = {
    "v2_real": R3_FITS / "dregon_room2_floor__flight_floor_lowk.json",
    "v2_legacy_lowk": R4_FITS / "dregon_legacy_render__flight_floor_lowk.json",
    "v2_legacy_free": R4_FITS / "dregon_legacy_render__flight.json",
}
ARM_ORDER = ("real", "legacy", "v2_real", "v2_legacy_lowk", "v2_legacy_free")
ARM_LABEL = {
    "real": "real DREGON room-2 clip",
    "legacy": "legacy stage-2 render, identity-matched",
    "v2_real": "v2 fitted to the REAL clip (R3, flight_floor_lowk)",
    "v2_legacy_lowk": "v2 fitted to the LEGACY render (flight_floor_lowk)",
    "v2_legacy_free": "v2 fitted to the LEGACY render (flight, free profile)",
}
RECORDINGS = (
    "free-flight_nosource_room2",
    "hovering_nosource_room2",
    "updown_nosource_room2",
)
SEED = 2001
N_MICS = 8
#: A prediction under this rate is the trackers' OFF verdict, not a mistracked
#: rotor: every cruise label here sits at 74-90 rev/s and both training streams
#: carry a zero-labelled `kind: silence` arm (conf/online_mix/hb_*_dload.yaml).
OFF_RPS = 5.0

#: The two trackers. The first is the frozen scorer every round reports.
TRACKERS: dict[str, dict[str, str] | None] = {
    "hppnet": None,  # -> noise_v2_round_score.Probe.load(), digest-checked
    "scv2": dict(experiment="real_r4_scv2_unified", ckpt="best_real_r2.ckpt"),
}

# ── the CQT read ────────────────────────────────────────────────────────────

#: Orders the contrast ladder is read at, and the harmonic-sum cut-offs.
CQT_ORDERS = tuple(range(1, 17))
SUM_ORDERS = (4, 8, 16)
#: dB a per-order contrast must clear to count as "visible" in the frame count.
VISIBLE_DB = 3.0
#: Local-floor annulus, in SEMITONES of log-distance from the read frequency.
#: The inner radius also excludes every OTHER comb line (all rotors, all
#: orders up to the grid top), so the floor is never another order's skirt.
ANNULUS_INNER_ST = 0.5
ANNULUS_OUTER_ST = 2.0
#: The literal R5-brief band: +-1/2 semitone, comb bins removed. Kept as a
#: cross-check because at 48 bins/octave it is only +-2 bins wide.
TIGHT_BAND_ST = 0.5
#: Orders masked out of every floor estimate (k * f_r up to the grid top).
EXCLUDE_ORDERS = 56
#: Half-width, in CQT bins, of the window the "where is the line really"
#: readout searches. One bin is 25 cents; at any k the four rotors span 1.7
#: bins, so +-2 bins covers the whole four-rotor spread and nothing more.
PEAK_SEARCH_BINS = 2
#: Mis-tuned carrier hypotheses the harmonic sum is evaluated at. +-10 % in
#: 2 % steps (2 % = 0.8 CQT bins at any k) plus the two octave confusions.
ALPHA_GRID = (0.5, 0.90, 0.92, 0.94, 0.96, 0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.10, 2.0)


def die(m: str) -> NoReturn:
    raise SystemExit(f"error: {m}")


def _module(name: str) -> Any:
    path = Path("scripts") / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, str(path))
    if spec is None or spec.loader is None:
        die(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def git_rev() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return "unknown"


# ── the frozen renders (R4's own route, bit-identical) ──────────────────────


def window_material(recording: str) -> dict[str, Any]:
    """The clip, the raw label and the scored support of one score window."""
    lt = _module("noise_v2_legacy_truth")
    support = lt.scored_support(recording)
    clip = lt.load_clip(support)
    sr = int(clip.sr)
    real = np.asarray(clip.audio, dtype=np.float64)[:N_MICS]
    rps = np.atleast_2d(np.asarray(clip.rps, dtype=np.float64))
    rsupport = RE.regime_support(
        support.window,
        rps,
        regime=support.regime,
        min_rps=support.min_rps,
        max_rps=support.max_rps,
        sr=sr,
    )
    return dict(lt=lt, support=support, sr=sr, real=real, rps=rps, rsupport=rsupport)


def arm_audio(name: str, mat: dict[str, Any], fits: dict[str, dict[str, Any]]) -> np.ndarray:
    """One arm's 8-microphone render on this window, at the frozen seed."""
    if name == "real":
        return np.asarray(mat["real"], dtype=np.float64)
    if name == "legacy":
        return np.asarray(
            mat["lt"].legacy_render(mat["support"].recording, mat["rps"]), dtype=np.float64
        )[:N_MICS]
    return np.asarray(
        RD.render_noise(fits[name], mat["rps"], n_mics=N_MICS, seed=SEED), dtype=np.float64
    )[:N_MICS]


def load_trackers(which: tuple[str, ...]) -> dict[str, Any]:
    rs = _module("noise_v2_round_score")
    wd = _module("noise_v2_widen_dregon")
    out: dict[str, Any] = {}
    for name in which:
        spec = TRACKERS[name]
        out[name] = rs.Probe.load() if spec is None else wd._probe(spec["experiment"], spec["ckpt"])
    return out


# ── (1) track shapes: mean-track error against comb-spread error ────────────


def comb_stats(pred: np.ndarray, truth: np.ndarray) -> dict[str, Any]:
    """Decompose one microphone's ``(R, F)`` prediction against its label.

    Sorted per frame, four rotor tracks are a comb: a CENTRE (their mean) and
    three GAPS. The tracker can get the centre right and the comb wrong, or
    the other way round, and the PIT MAE adds the two. ``mean_gap`` is the
    comb's pitch, ``gap_cv`` how equally spaced it is within a frame, and
    ``mean_gap_std_t`` how much the pitch moves over time. ``frac_below`` is
    the SILENCE verdict: the share of predicted values under
    ``OFF_RPS``, the rate below which the training stream's zero-labelled
    silence arm lives — a tracker that emits it is not mistracking the comb,
    it is answering "the rotors are off".
    """
    ps = np.sort(np.asarray(pred, dtype=np.float64), axis=0)
    ls = np.sort(np.asarray(truth, dtype=np.float64), axis=0)
    gp, gl = np.diff(ps, axis=0), np.diff(ls, axis=0)
    cp, cl = ps.mean(axis=0), ls.mean(axis=0)
    dev_p, dev_l = ps - cp[None, :], ls - cl[None, :]
    centre_err = cp - cl
    spread_err = dev_p - dev_l
    return dict(
        mae_pit=float(np.abs(pred - truth).mean()),
        mae_sorted=float(np.abs(ps - ls).mean()),
        mean_gap_pred=float(gp.mean()),
        mean_gap_label=float(gl.mean()),
        mean_gap_std_t_pred=float(gp.mean(axis=0).std()),
        mean_gap_std_t_label=float(gl.mean(axis=0).std()),
        gap_cv_pred=float(np.mean(gp.std(axis=0) / np.maximum(np.abs(gp.mean(axis=0)), 1e-9))),
        gap_cv_label=float(np.mean(gl.std(axis=0) / np.maximum(np.abs(gl.mean(axis=0)), 1e-9))),
        centre_pred=float(cp.mean()),
        centre_label=float(cl.mean()),
        centre_err_mean=float(centre_err.mean()),
        centre_err_abs=float(np.abs(centre_err).mean()),
        centre_err_std_t=float(centre_err.std()),
        spread_err_abs=float(np.abs(spread_err).mean()),
        spread_rms_pred=float(np.sqrt((dev_p**2).mean())),
        spread_rms_label=float(np.sqrt((dev_l**2).mean())),
        pred_median=float(np.median(ps)),
        frac_pred_below_off=float(np.mean(ps < OFF_RPS)),
        frac_frames_all_off=float(np.mean((ps < OFF_RPS).all(axis=0))),
    )


def _mean_of(rows: list[dict[str, Any]], key: str) -> float:
    return float(np.mean([r[key] for r in rows]))


def run_tracks(*, out: Path, recordings: tuple[str, ...], trackers: tuple[str, ...]) -> dict:
    tk = load_trackers(trackers)
    fits = {k: json.loads(p.read_text()) for k, p in FIT_PATHS.items()}
    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        study="tracks",
        git=git_rev(),
        protocol=dict(
            seed=SEED,
            n_mics=N_MICS,
            arms={k: ARM_LABEL[k] for k in ARM_ORDER},
            fits={k: str(v) for k, v in FIT_PATHS.items()},
            trackers={k: tk[k].record for k in trackers},
            note=(
                "PIT MAE is scripts/_synthetic_probe.score through "
                "revised_eval.pit_mae (the frozen path); the per-mic tracks "
                "here are the same call's return value, so the recomputed "
                "mean over mics is asserted equal to pit()'s"
            ),
            decomposition=(
                "per frame the four tracks are SORTED; centre = mean over "
                "rotors, gaps = the three adjacent differences. centre_err = "
                "centre(pred) - centre(label); spread_err = the deviation "
                "from the centre, predicted minus label"
            ),
        ),
        windows={},
    )
    figures: dict[str, Any] = {}
    for recording in recordings:
        mat = window_material(recording)
        support = mat["support"]
        row: dict[str, Any] = dict(support=support.as_dict(), arms={})
        fig_row: dict[str, Any] = dict(sr=mat["sr"], arms={})
        for name in ARM_ORDER:
            audio = arm_audio(name, mat, fits)
            entry: dict[str, Any] = dict(
                level_dbrms_per_mic=[
                    float(v) for v in 20.0 * np.log10(np.sqrt((audio**2).mean(axis=1)) + 1e-20)
                ]
            )
            for tname in trackers:
                probe = tk[tname]
                pit = probe.tracker.pit(
                    audio,
                    mat["rps"],
                    mics=list(range(N_MICS)),
                    expected_samples=int(mat["real"].shape[-1]),
                    support=mat["rsupport"],
                )
                keep = None
                per_mic: list[dict[str, Any]] = []
                tracks0 = None
                for mic in range(N_MICS):
                    pred, truth, _ = probe.tracker.score(
                        probe.tracker.fm,
                        probe.tracker.metric,
                        audio,
                        mat["rps"],
                        mat["sr"],
                        mic,
                    )
                    if keep is None:
                        keep = RE.frames_in_support(
                            int(pred.shape[-1]),
                            mat["rsupport"],
                            n_samples=int(audio.shape[-1]),
                            sr=mat["sr"],
                        )
                        tracks0 = (pred[:, keep].copy(), truth[:, keep].copy())
                    per_mic.append(comb_stats(pred[:, keep], truth[:, keep]))
                assert tracks0 is not None and keep is not None
                mae_recomputed = _mean_of(per_mic, "mae_pit")
                if abs(mae_recomputed - float(pit["mae"])) > 1e-9:
                    die(
                        f"{recording}/{name}/{tname}: recomputed MAE "
                        f"{mae_recomputed} != frozen path {pit['mae']}"
                    )
                entry[tname] = dict(
                    pit_mae=float(pit["mae"]),
                    per_mic_mae=[float(v) for v in pit["per_mic"]],
                    n_scored_frames=int(pit["n_scored_frames"]),
                    stats=dict(
                        mic0=per_mic[0],
                        mic_mean={
                            k: _mean_of(per_mic, k)
                            for k in per_mic[0]
                            if isinstance(per_mic[0][k], float)
                        },
                    ),
                )
                fig_row["arms"].setdefault(name, {})[tname] = dict(
                    pred=tracks0[0], truth=tracks0[1], pit_mae=float(pit["mae"])
                )
                print(
                    f"[{recording}] {name}/{tname}: pit={pit['mae']:.3f} "
                    f"gap={per_mic[0]['mean_gap_pred']:+.3f} "
                    f"(label {per_mic[0]['mean_gap_label']:+.3f}) "
                    f"centre_err={per_mic[0]['centre_err_mean']:+.3f}",
                    flush=True,
                )
            row["arms"][name] = entry
        payload["windows"][support.key] = row
        figures[recording] = fig_row
    payload["summary"] = tracks_summary(payload, trackers)
    Path(out).mkdir(parents=True, exist_ok=True)
    return payload | {"_figures": figures}


def tracks_summary(payload: dict[str, Any], trackers: tuple[str, ...]) -> dict[str, Any]:
    """3-window means of every scalar, per arm and tracker (mic-mean stats)."""
    keys = list(payload["windows"])
    out: dict[str, Any] = {}
    for tname in trackers:
        per_arm: dict[str, Any] = {}
        for arm in ARM_ORDER:
            rows = [payload["windows"][k]["arms"][arm][tname] for k in keys]
            stat_keys = list(rows[0]["stats"]["mic_mean"])
            per_arm[arm] = dict(
                pit_mae=float(np.mean([r["pit_mae"] for r in rows])),
                per_window_pit=[float(r["pit_mae"]) for r in rows],
            ) | {k: float(np.mean([r["stats"]["mic_mean"][k] for r in rows])) for k in stat_keys}
        out[tname] = per_arm
    return out


def write_track_figures(figures: dict[str, Any], out: Path, trackers: tuple[str, ...]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recs = list(figures)
    written: list[str] = []
    cols = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e"]
    for tname in trackers:
        fig, axes = plt.subplots(
            len(ARM_ORDER), len(recs), figsize=(4.6 * len(recs), 2.5 * len(ARM_ORDER)), sharex="col"
        )
        axes = np.atleast_2d(axes)
        for i, arm in enumerate(ARM_ORDER):
            for j, rec in enumerate(recs):
                ax = axes[i, j]
                d = figures[rec]["arms"][arm][tname]
                pred, truth = d["pred"], d["truth"]
                t = np.arange(pred.shape[-1])
                for r in range(truth.shape[0]):
                    ax.plot(t, truth[r], color=cols[r % 4], lw=2.6, alpha=0.30)
                for r in range(pred.shape[0]):
                    ax.plot(t, pred[r], color=cols[r % 4], lw=0.9, marker=".", ms=2.2)
                ax.grid(alpha=0.2)
                ax.set_title(
                    f"{arm} | {rec.replace('_nosource_room2', '')} | MAE {d['pit_mae']:.2f} rev/s",
                    fontsize=8,
                )
                if j == 0:
                    ax.set_ylabel("rev/s", fontsize=8)
                if i == len(ARM_ORDER) - 1:
                    ax.set_xlabel("scored output frame", fontsize=8)
                ax.tick_params(labelsize=7)
        fig.suptitle(
            f"{tname}: four predicted rotor tracks (thin, PIT-matched) against the four "
            "labels (thick, faded), mic 0, scored frames only",
            fontsize=11,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.975))
        path = out / f"tracks_{tname}.png"
        fig.savefig(path, dpi=110)
        plt.close(fig)
        written.append(str(path))
    return written


# ── (2) HPPNet's own CQT statistic ──────────────────────────────────────────


def cqt_frontend() -> tuple[Any, dict[str, Any]]:
    """The frozen HPPNet's OWN front-end module and its stated parameters."""
    rs = _module("noise_v2_round_score")
    probe = rs.Probe.load()
    net = probe.tracker.fm.model
    fe = None
    for _, mod in net.named_modules():
        if type(mod).__name__ == "CQTLogSpecgram":
            fe = mod
            break
    if fe is None:
        die("the frozen scorer carries no CQTLogSpecgram front end")
    c = fe.cqt
    bpo = int(net.bins_per_octave)
    q = 1.0 / (2.0 ** (1.0 / bpo) - 1.0)
    params = dict(
        module="models.harmonic_ports.hppnet_orig.CQTLogSpecgram",
        implementation="nnAudio.features.cqt.CQT2010v2 + torchaudio AmplitudeToDB(top_db=80)",
        sr=int(net.spec_sr),
        hop_length=int(net.spec_hop),
        fmin_hz=float(net.fmin),
        n_bins=int(net.n_bins),
        bins_per_octave=bpo,
        filter_scale=1.0,
        q=float(q),
        cents_per_bin=float(1200.0 / bpo),
        frame_rate_hz=float(net.spec_sr) / float(net.spec_hop),
        top_bin_hz=float(net.fmin) * 2.0 ** ((int(net.n_bins) - 1) / bpo),
        n_octaves=int(getattr(c, "n_octaves", 0)),
        downsample_factor=float(getattr(c, "downsample_factor", 1.0)),
        note=(
            "constant-Q: the analysis window of the bin at f is Q/f seconds "
            "(Q = 1/(2^(1/B)-1)); the -3 dB bandwidth is f/Q Hz. Both are "
            "tabulated per order in protocol.window_ms_at_fbar"
        ),
        scorer=probe.record,
    )
    return fe, params


def cqt_db(fe: Any, audio: np.ndarray) -> np.ndarray:
    """``(M, T) -> (M, frames, bins)`` dB, one ``top_db`` floor per microphone."""
    import torch

    with torch.no_grad():
        x = torch.as_tensor(np.asarray(audio, dtype=np.float32))
        return fe(x).numpy().astype(np.float64)


def label_on_frames(rps: np.ndarray, n_frames: int, hop: int) -> np.ndarray:
    """``(R, T)`` raw label sampled at the CQT's own frame centres ``j*hop``."""
    idx = np.clip(np.arange(n_frames) * int(hop), 0, int(rps.shape[-1]) - 1)
    return np.asarray(rps, dtype=np.float64)[:, idx]


def comb_masks(fr: np.ndarray, *, freqs: np.ndarray, bpo: int, inner_st: float) -> np.ndarray:
    """``(frames, bins)`` True where a bin is within ``inner_st`` of ANY line.

    Every rotor and every order ``k <= EXCLUDE_ORDERS`` whose line falls on the
    grid. Depends only on the label, so one mask serves every arm.
    """
    logb = np.log2(np.asarray(freqs, dtype=np.float64) / float(freqs[0])) * bpo
    ks = np.arange(1, EXCLUDE_ORDERS + 1, dtype=np.float64)
    tol = float(inner_st) * (bpo / 12.0)
    out = np.zeros((fr.shape[-1], freqs.size), dtype=bool)
    for t in range(fr.shape[-1]):
        lines = np.outer(ks, fr[:, t]).ravel()
        lines = lines[(lines > freqs[0]) & (lines < freqs[-1])]
        if lines.size == 0:
            continue
        lb = np.log2(lines / float(freqs[0])) * bpo
        out[t] = (np.abs(logb[:, None] - lb[None, :]) <= tol).any(axis=1)
    return out


def comb_geometry(fr: np.ndarray, *, freqs: np.ndarray, bpo: int) -> dict[str, Any]:
    """Every bin index the ladder reads, precomputed from the LABEL alone.

    The label is the same for all five arms and all eight microphones, so the
    geometry is built once per window and only the gathers are per-arm. Three
    reference sets per ``(k, rotor, frame)``:

    * ``bidx`` — the bin nearest ``k * f_r``, and ``off`` the +-2 bin window
      around it, whose argmax says where the line ACTUALLY sits;
    * ``mid`` — the INTER-TOOTH floor: the bins nearest ``(k -+ 1/2) * f_r``,
      +-1 bin each, i.e. the six bins furthest from any tooth of this rotor's
      comb. Defined at every ``k``, which the annulus is not;
    * ``ann`` — the strict local annulus, ``(ANNULUS_INNER_ST,
      ANNULUS_OUTER_ST]`` semitones from the read bin with every comb line of
      every rotor masked out. At 48 bins/octave the four rotors span 1.7 bins
      and order ``k``'s neighbours sit ``48 log2(1 + 1/k)`` bins away, so this
      set EMPTIES above ``k ~ 11``: the comb stops being resolvable on the
      model's own grid. ``n_annulus`` records where.
    """
    nb = int(freqs.size)
    logb = np.log2(np.asarray(freqs, dtype=np.float64) / float(freqs[0])) * bpo
    st = bpo / 12.0
    ks = np.asarray(CQT_ORDERS, dtype=np.float64)[:, None, None]
    f = ks * np.asarray(fr, dtype=np.float64)[None, :, :]  # (nk, nr, nt)
    pos = np.log2(np.maximum(f, 1e-9) / float(freqs[0])) * bpo
    valid = (f > freqs[0]) & (f < freqs[-1])
    bidx = np.clip(np.round(pos).astype(int), 0, nb - 1)
    off = np.clip(bidx[..., None] + np.arange(-PEAK_SEARCH_BINS, PEAK_SEARCH_BINS + 1), 0, nb - 1)
    mid = np.concatenate(
        [
            np.clip(
                np.round(
                    np.log2(np.maximum((ks + s) * fr[None, :, :], 1e-9) / float(freqs[0])) * bpo
                ).astype(int)[..., None]
                + np.arange(-1, 2),
                0,
                nb - 1,
            )
            for s in (-0.5, 0.5)
        ],
        axis=-1,
    )
    masked = comb_masks(fr, freqs=freqs, bpo=bpo, inner_st=ANNULUS_INNER_ST)
    d = np.abs(logb[None, None, None, :] - pos[..., None]) / st  # (nk, nr, nt, nb)
    free = ~masked[None, None, :, :]  # (1, 1, nt, nb)
    ann = (d > ANNULUS_INNER_ST) & (d <= ANNULUS_OUTER_ST) & free & valid[..., None]
    tight = (d <= TIGHT_BAND_ST) & free & valid[..., None]
    return dict(
        bidx=bidx,
        off=off,
        mid=mid,
        ann=ann,
        tight=tight,
        valid=valid,
        n_annulus=ann.sum(axis=-1).astype(np.float64),
        frac_mid_on_comb=float(masked[np.arange(mid.shape[2])[None, None, :, None], mid].mean()),
    )


def contrast_ladder(L: np.ndarray, geo: dict[str, Any]) -> dict[str, np.ndarray]:
    """Per-order contrast of the label's comb over the floor, one microphone.

    ``L`` is the ``(frames, bins)`` dB CQT. Headline ``c`` is the read bin over
    the INTER-TOOTH floor (defined at every order); ``c_local`` is the same
    read over the strict annulus (NaN where the comb is unresolved);
    ``c_tight`` is the brief's literal +-1/2 semitone band. ``c_peak1`` /
    ``c_peak2`` read the strongest bin within +-1 / +-2 bins instead of the
    label's own bin, and ``peak_off`` is where that maximum sat: together they
    separate a line that is WEAK from a line that is MISPLACED.
    """
    nb = L.shape[-1]
    nt = L.shape[0]
    tt = np.arange(nt)[None, None, :]
    at = L[tt, geo["bidx"]]  # (nk, nr, nt)
    offv = L[tt[..., None], geo["off"]]
    midv = L[tt[..., None], geo["mid"]]
    grid = L[None, None, :, :]  # (1, 1, nt, nb)
    floor_mid = np.median(midv, axis=-1)
    band = np.where(geo["ann"], grid, np.nan)
    with np.errstate(invalid="ignore"):
        floor_ann = np.nanmedian(band, axis=-1)
    tband = np.where(geo["tight"], grid, np.nan)
    with np.errstate(invalid="ignore"):
        floor_tight = np.nanmedian(tband, axis=-1)
    valid = geo["valid"]
    m = PEAK_SEARCH_BINS
    peak1 = offv[..., m - 1 : m + 2].max(axis=-1)
    peak2 = offv.max(axis=-1)
    peak_off = offv.argmax(axis=-1).astype(np.float64) - m
    assert nb == geo["ann"].shape[-1]
    return dict(
        c=np.where(valid, at - floor_mid, np.nan),
        c_local=np.where(valid, at - floor_ann, np.nan),
        c_tight=np.where(valid, at - floor_tight, np.nan),
        c_peak1=np.where(valid, peak1 - floor_mid, np.nan),
        c_peak2=np.where(valid, peak2 - floor_mid, np.nan),
        peak_off=np.where(valid, peak_off, np.nan),
        floor=np.where(valid, floor_mid, np.nan),
        n_annulus=geo["n_annulus"],
    )


def alpha_curve(L: np.ndarray, fr: np.ndarray, *, freqs: np.ndarray, bpo: int) -> dict[str, Any]:
    """Harmonic sum as a function of a MIS-TUNED carrier — the decisive test.

    A magnitude-domain comb tracker can only prefer the true carrier if the
    harmonic sum at ``alpha = 1`` beats the sum at nearby wrong carriers. This
    evaluates ``S_N`` exactly as :func:`ladder_stats` does — read bin over the
    inter-tooth floor of the SAME hypothesis, so the spectral tilt cancels —
    on the comb of ``alpha * f_r`` for every ``alpha`` in `ALPHA_GRID`, plus
    the two octave confusions. ``carrier_gain`` is ``S_N(1)`` minus the median
    over the mis-tuned alphas: the dB of evidence the label's carrier has over
    a wrong one in this front end.
    """
    nb, nt = int(freqs.size), int(L.shape[0])
    tt = np.arange(nt)[None, None, :]
    ks = np.asarray(CQT_ORDERS, dtype=np.float64)[:, None, None]
    curve: dict[float, dict[int, float]] = {}
    for a in ALPHA_GRID:
        fa = ks * (float(a) * np.asarray(fr, dtype=np.float64))[None, :, :]
        valid = (fa > freqs[0]) & (fa < freqs[-1])
        b = np.clip(
            np.round(np.log2(np.maximum(fa, 1e-9) / float(freqs[0])) * bpo).astype(int), 0, nb - 1
        )
        mid = np.concatenate(
            [
                np.clip(
                    np.round(
                        np.log2(
                            np.maximum((ks + s) * (float(a) * fr)[None, :, :], 1e-9)
                            / float(freqs[0])
                        )
                        * bpo
                    ).astype(int)[..., None]
                    + np.arange(-1, 2),
                    0,
                    nb - 1,
                )
                for s in (-0.5, 0.5)
            ],
            axis=-1,
        )
        c = np.where(valid, L[tt, b] - np.median(L[tt[..., None], mid], axis=-1), np.nan)
        pos = np.where(np.isfinite(c), np.maximum(c, 0.0), 0.0)
        curve[float(a)] = {
            n: float(np.median(pos[: CQT_ORDERS.index(n) + 1].sum(axis=0))) for n in SUM_ORDERS
        }
    wrong = [a for a in ALPHA_GRID if abs(a - 1.0) > 1e-9]
    return dict(
        curve={f"{a:.2f}": {str(n): curve[a][n] for n in SUM_ORDERS} for a in ALPHA_GRID},
        carrier_gain_db={
            str(n): curve[1.0][n] - float(np.median([curve[a][n] for a in wrong]))
            for n in SUM_ORDERS
        },
        argmax_alpha={
            str(n): float(max(ALPHA_GRID, key=lambda a: curve[a][n])) for n in SUM_ORDERS
        },
    )


def _med(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(np.median(a)) if a.size else float("nan")


def ladder_stats(lad: dict[str, np.ndarray]) -> dict[str, Any]:
    """Medians over frames/rotors per order, the harmonic sums, the fractions."""
    c = lad["c"]

    def per_k(key: str) -> dict[str, float]:
        return {str(k): _med(lad[key][i]) for i, k in enumerate(CQT_ORDERS)}

    out: dict[str, Any] = dict(
        c_k_median_db=per_k("c"),
        c_k_local_median_db=per_k("c_local"),
        c_k_tight_median_db=per_k("c_tight"),
        c_k_peak1_median_db=per_k("c_peak1"),
        c_k_peak2_median_db=per_k("c_peak2"),
        floor_median_db=per_k("floor"),
        peak_offset_bins_median=per_k("peak_off"),
        peak_offset_bins_absmean={
            str(k): float(np.nanmean(np.abs(lad["peak_off"][i]))) for i, k in enumerate(CQT_ORDERS)
        },
        n_annulus_bins={
            str(k): float(np.nanmean(lad["n_annulus"][i])) for i, k in enumerate(CQT_ORDERS)
        },
        frac_c_gt_3db={
            str(k): float(np.nanmean((c[i] > VISIBLE_DB).astype(float)))
            for i, k in enumerate(CQT_ORDERS)
        },
    )
    pos = np.where(np.isfinite(c), np.maximum(c, 0.0), 0.0)
    for n in SUM_ORDERS:
        s = pos[: CQT_ORDERS.index(n) + 1].sum(axis=0)  # (R, frames)
        out[f"S{n}_median_db"] = float(np.median(s))
        out[f"S{n}_mean_db"] = float(np.mean(s))
    return out


def run_cqt(*, out: Path, recordings: tuple[str, ...]) -> dict[str, Any]:
    fe, params = cqt_frontend()
    fits = {k: json.loads(p.read_text()) for k, p in FIT_PATHS.items()}
    bpo = int(params["bins_per_octave"])
    freqs = float(params["fmin_hz"]) * 2.0 ** (np.arange(params["n_bins"]) / bpo)
    payload: dict[str, Any] = dict(
        schema=SCHEMA,
        study="cqt",
        git=git_rev(),
        protocol=dict(
            seed=SEED,
            n_mics=N_MICS,
            arms={k: ARM_LABEL[k] for k in ARM_ORDER},
            fits={k: str(v) for k, v in FIT_PATHS.items()},
            frontend=params,
            orders=list(CQT_ORDERS),
            sum_orders=list(SUM_ORDERS),
            visible_db=VISIBLE_DB,
            annulus_semitones=[ANNULUS_INNER_ST, ANNULUS_OUTER_ST],
            tight_band_semitones=TIGHT_BAND_ST,
            exclude_orders=EXCLUDE_ORDERS,
            peak_search_bins=PEAK_SEARCH_BINS,
            floor=(
                "headline c_k reads the label's own bin over the INTER-TOOTH "
                "floor: the median of the six bins nearest (k -+ 1/2) * f_r. "
                "c_k_local is the strict annulus and is NaN above the order "
                "where the four-rotor comb stops being resolvable at 48 "
                "bins/octave; n_annulus_bins records it"
            ),
        ),
        windows={},
    )
    for recording in recordings:
        mat = window_material(recording)
        support = mat["support"]
        n_frames = int(np.asarray(mat["real"].shape[-1]) // params["hop_length"]) + 1
        fr = label_on_frames(mat["rps"], n_frames, int(params["hop_length"]))
        fbar = float(fr.mean())
        geo = comb_geometry(fr, freqs=freqs, bpo=bpo)
        row: dict[str, Any] = dict(
            support=support.as_dict(),
            fbar_rev_s=fbar,
            n_frames=int(n_frames),
            window_ms_at_fbar={
                str(k): 1000.0 * float(params["q"]) / (k * fbar) for k in CQT_ORDERS
            },
            bandwidth_hz_at_fbar={str(k): (k * fbar) / float(params["q"]) for k in CQT_ORDERS},
            bin_spacing_hz_at_fbar={
                str(k): (k * fbar) * (2.0 ** (1.0 / bpo) - 1.0) for k in CQT_ORDERS
            },
            frac_mid_bins_on_comb=float(geo["frac_mid_on_comb"]),
            arms={},
        )
        for name in ARM_ORDER:
            audio = arm_audio(name, mat, fits)
            L = cqt_db(fe, audio)
            if L.shape[1] != n_frames:
                die(f"{recording}/{name}: CQT gave {L.shape[1]} frames, expected {n_frames}")
            per_mic = [ladder_stats(contrast_ladder(L[m], geo)) for m in range(N_MICS)]
            agg: dict[str, Any] = {}
            for key in per_mic[0]:
                if isinstance(per_mic[0][key], dict):
                    agg[key] = {
                        kk: float(np.nanmean([p[key][kk] for p in per_mic]))
                        for kk in per_mic[0][key]
                    }
                else:
                    agg[key] = float(np.nanmean([p[key] for p in per_mic]))
            clamped = float(np.mean((L.max(axis=(1, 2), keepdims=True) - 80.0 + 1e-6) >= L))
            sweeps = [alpha_curve(L[m], fr, freqs=freqs, bpo=bpo) for m in range(N_MICS)]
            agg["carrier_gain_db"] = {
                str(n): float(np.mean([s["carrier_gain_db"][str(n)] for s in sweeps]))
                for n in SUM_ORDERS
            }
            agg["alpha_curve_S8"] = {
                a: float(np.mean([s["curve"][a]["8"] for s in sweeps])) for a in sweeps[0]["curve"]
            }
            row["arms"][name] = dict(
                mic_mean=agg,
                mic0=per_mic[0] | dict(alpha=sweeps[0]),
                frac_at_top_db_floor=clamped,
            )
            lad = agg["c_k_median_db"]
            print(
                f"[{recording}] {name}: c1={lad['1']:+.2f} c2={lad['2']:+.2f} "
                f"c4={lad['4']:+.2f} c8={lad['8']:+.2f} c16={lad['16']:+.2f} dB, "
                f"S8={agg['S8_median_db']:.2f}, "
                f"carrier gain S8={agg['carrier_gain_db']['8']:+.2f} dB",
                flush=True,
            )
        payload["windows"][support.key] = row
    payload["summary"] = cqt_summary(payload)
    Path(out).mkdir(parents=True, exist_ok=True)
    return payload


def cqt_summary(payload: dict[str, Any]) -> dict[str, Any]:
    keys = list(payload["windows"])
    out: dict[str, Any] = {}
    for arm in ARM_ORDER:
        rows = [payload["windows"][k]["arms"][arm]["mic_mean"] for k in keys]
        entry: dict[str, Any] = {}
        for key in rows[0]:
            if isinstance(rows[0][key], dict):
                entry[key] = {
                    kk: float(np.nanmean([r[key][kk] for r in rows])) for kk in rows[0][key]
                }
            else:
                entry[key] = float(np.nanmean([r[key] for r in rows]))
        out[arm] = entry
    return out


def write_cqt_figure(payload: dict[str, Any], out: Path) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    s = payload["summary"]
    ks = [int(k) for k in CQT_ORDERS]
    w0 = next(iter(payload["windows"].values()))
    cols = {
        "real": "#000000",
        "legacy": "#d62728",
        "v2_real": "#1f77b4",
        "v2_legacy_lowk": "#2ca02c",
        "v2_legacy_free": "#ff7f0e",
    }
    fig, axes = plt.subplots(1, 4, figsize=(21.0, 5.2))
    ax = axes[0]
    for arm in ARM_ORDER:
        y = [s[arm]["c_k_median_db"][str(k)] for k in ks]
        ax.plot(ks, y, marker="o", ms=4, color=cols[arm], label=arm, lw=1.8)
    ax.axhline(VISIBLE_DB, color="0.4", ls=":", lw=1)
    ax.set_xlabel("order k")
    ax.set_ylabel("median contrast $c_k$ over inter-tooth floor (dB)")
    ax.set_title("(a) per-order CQT contrast at the label's own comb", fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks([1, 4, 8, 12, 16])
    ax2.set_xticklabels(
        [f"{w0['window_ms_at_fbar'][str(k)]:.0f} ms" for k in (1, 4, 8, 12, 16)], fontsize=7
    )
    ax2.set_xlabel("constant-Q analysis window at the window's mean carrier", fontsize=8)

    ax = axes[1]
    for arm in ARM_ORDER:
        y = [s[arm]["c_k_peak2_median_db"][str(k)] for k in ks]
        ax.plot(ks, y, marker="s", ms=4, color=cols[arm], lw=1.8, label=arm)
    ax.axhline(VISIBLE_DB, color="0.4", ls=":", lw=1)
    ax.set_xlabel("order k")
    ax.set_ylabel("median contrast, best bin within $\\pm$2 bins (dB)")
    ax.set_title(
        "(b) same, free to find the line within $\\pm$2 bins ($\\pm$50 cents)", fontsize=10
    )
    ax.grid(alpha=0.25)

    ax = axes[3]
    alphas = sorted(float(a) for a in s["real"]["alpha_curve_S8"] if 0.85 <= float(a) <= 1.15)
    for arm in ARM_ORDER:
        y = [s[arm]["alpha_curve_S8"][f"{a:.2f}"] for a in alphas]
        ax.plot(alphas, y, marker="^", ms=4, color=cols[arm], lw=1.8, label=arm)
    ax.axvline(1.0, color="0.4", ls=":", lw=1)
    ax.set_xlabel(r"carrier hypothesis $\alpha$ ($f = \alpha\,f_r^{label}$)")
    ax.set_ylabel("$S_8$ at the mis-tuned comb (dB)")
    ax.set_title(
        "(d) does the TRUE carrier win in the magnitude CQT?\n"
        "carrier gain $S_8(1)-\\mathrm{med}_{\\alpha\\neq1}S_8(\\alpha)$: "
        + ", ".join(f"{a} {s[a]['carrier_gain_db']['8']:+.2f}" for a in ("real", "legacy"))
        + " dB",
        fontsize=9,
    )
    ax.grid(alpha=0.25)

    ax = axes[2]
    x = np.arange(len(ARM_ORDER))
    for i, n in enumerate(SUM_ORDERS):
        ax.bar(
            x + (i - 1) * 0.26,
            [s[a][f"S{n}_median_db"] for a in ARM_ORDER],
            width=0.25,
            label=f"$S_{{{n}}}$",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(ARM_ORDER, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("median harmonic sum (dB)")
    ax.set_title("(c) $S_N=\\sum_{k\\leq N}\\max(c_k,0)$", fontsize=10)
    ax.grid(alpha=0.25, axis="y")
    ax.legend(fontsize=8)
    fig.suptitle(
        "HPPNet's OWN front end (CQT2010v2, fmin 27.5 Hz, 48 bins/oct, 352 bins, hop 512, "
        "AmplitudeToDB top_db=80) — 3-window, 8-mic, 4-rotor medians",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    path = out / "cqt_contrast.png"
    fig.savefig(path, dpi=115)
    plt.close(fig)
    return [str(path)]


# ── CLI ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = ap.add_subparsers(dest="cmd", required=True)
    for cmd in ("tracks", "cqt"):
        p = sub.add_parser(cmd)
        p.add_argument("--out", default=str(OUT_DEFAULT))
        p.add_argument("--recordings", default=",".join(RECORDINGS))
        p.add_argument("--figures", action="store_true")
        if cmd == "tracks":
            p.add_argument("--trackers", default=",".join(TRACKERS))
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = tuple(r for r in str(args.recordings).split(",") if r)

    if args.cmd == "tracks":
        trackers = tuple(t for t in str(args.trackers).split(",") if t)
        payload = run_tracks(out=out, recordings=recs, trackers=trackers)
        figs = payload.pop("_figures")
        written = write_track_figures(figs, out, trackers) if args.figures else []
        payload["figures"] = written
        (out / "tracks.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(f"# wrote {out / 'tracks.json'}")
    else:
        payload = run_cqt(out=out, recordings=recs)
        written = write_cqt_figure(payload, out) if args.figures else []
        payload["figures"] = written
        (out / "cqt.json").write_text(json.dumps(payload, indent=1) + "\n")
        print(f"# wrote {out / 'cqt.json'}")
    for p in written:
        print(f"# wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
