"""Generate the generative-model figures for the supervisor deck and the paper.

Writes ``gen_*.pdf`` and ``gen_*.png`` into the paper submodule's ``src/figures/``
and ``gen_*.tex`` (booktabs) into ``src/tables/``; the PNGs are what the deck
(``writing/slides/2026-09-22_generative-model-supervisor/slides.qmd``) embeds.

Figures
-------
``gen_traj_real_vs_sampled``
    30 s of real airborne telemetry (4 rotors, 100 Hz grid) beside a 30 s draw
    from that rig's fitted trajectory model, dregon and michaels, shared y-axis
    per rig.
``gen_traj_hyperprior``
    Four flights drawn from the rig hyperprior (``traj_rig="posterior"``),
    30 s each, hover level in the title.
``gen_noise_real_vs_sampled``
    Real DREGON room-2 flight and Michael's FLY125 cruise windows against the
    fitted model rendered on the SAME telemetry label, percentile-clipped dB.
``gen_noise_hyperprior``
    Four hard-bank entries rendered on hyperprior trajectories -- the noise the
    ``nv2_hard`` arm trains on.

Run: ``systemd-run --user --scope -p MemoryMax=10G python scripts/gen_model_figures.py``
"""

from __future__ import annotations

import contextlib
import sys
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "notebooks"))

PAPER = ROOT / "writing/papers/2026-08_wrapup/src"
FIGS = PAPER / "figures"
TABLES = PAPER / "tables"
SUPPORTS = ROOT / "results/noise_v2/rounds/round2/supports"

SR = 16000
NFFT = 2048
HOP = 512
ROTOR_LABELS = ("RFront", "LFront", "LBack", "RBack")
COLOURS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd")

warnings.filterwarnings("ignore", category=RuntimeWarning)
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.25})


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(FIGS / f"{name}.png", bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"wrote {name}.pdf/.png", flush=True)


# ── trajectories ────────────────────────────────────────────────────────────


def _real_window(rig: str, seconds: float = 30.0):
    from experiments.rps_traj.data import RATE_HZ, airborne_segments, load_rig

    n = int(seconds * RATE_HZ)
    best = None
    for fl in load_rig(rig):
        for sl in airborne_segments(fl.rps, RATE_HZ):
            seg = fl.rps[:, sl]
            if seg.shape[1] >= n and np.isfinite(seg[:, :n]).all():
                span = float(np.nanmax(seg[:, :n]) - np.nanmin(seg[:, :n]))
                if best is None or span > best[0]:
                    best = (span, seg[:, :n])
    if best is None:
        raise RuntimeError(f"no {seconds} s airborne window on {rig}")
    return best[1]


def _plot_tracks(ax, rps, fs, title, ylim=None):
    t = np.arange(rps.shape[1]) / fs
    for r in range(rps.shape[0]):
        ax.plot(t, rps[r], lw=0.7, color=COLOURS[r % 4], label=ROTOR_LABELS[r % 4])
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("rotor speed (rev/s)")
    ax.set_xlim(0, t[-1])
    if ylim is not None:
        ax.set_ylim(*ylim)


def fig_traj_real_vs_sampled(
    rigs=("dregon", "michaels"), name: str = "gen_traj_real_vs_sampled", row_h: float = 3.1
) -> None:
    import noise_lab as NL

    from experiments.rps_traj.data import RATE_HZ

    fig, axes = plt.subplots(len(rigs), 2, figsize=(11, row_h * len(rigs)))
    axes = np.atleast_2d(axes)
    for row, rig in enumerate(rigs):
        real = _real_window(rig, 30.0)
        traj = NL.trajectory("fitted", rig=rig, seed=11 + row, duration_s=30.0)
        # ``trajectory`` already carries the plot track at RPS_PLOT_SR = RATE_HZ
        samp = np.asarray(traj["rps"].data)
        lo = min(np.nanmin(real), samp.min())
        hi = max(np.nanmax(real), samp.max())
        pad = 0.06 * (hi - lo)
        ylim = (lo - pad, hi + pad)
        _plot_tracks(axes[row, 0], real, RATE_HZ, f"{rig} — real telemetry, 30 s airborne", ylim)
        _plot_tracks(axes[row, 1], samp, RATE_HZ, f"{rig} — fitted model, 30 s sample", ylim)
    axes[0, 1].legend(ncol=4, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, name)


def fig_traj_hyperprior(n: int = 4) -> None:
    import noise_lab as NL

    from experiments.rps_traj.data import RATE_HZ

    fig, axes = plt.subplots(2, 2, figsize=(11, 6))
    for i, ax in enumerate(axes.ravel()[:n]):
        traj = NL.trajectory("fitted", rig="posterior", seed=101 + i, duration_s=30.0)
        rps = np.asarray(traj["rps"].data)
        hover = float(traj["meta"]["hover_rev_s"])
        _plot_tracks(ax, rps, RATE_HZ, f"hyperprior drone {i + 1} — hover {hover:.0f} rev/s")
    axes[0, 1].legend(ncol=4, fontsize=7, loc="upper right")
    fig.tight_layout()
    save(fig, "gen_traj_hyperprior")


# ── noise ───────────────────────────────────────────────────────────────────


def _spec_db(audio: np.ndarray) -> np.ndarray:
    w = np.hanning(NFFT + 1)[:NFFT]
    n = (len(audio) - NFFT) // HOP + 1
    frames = np.stack([audio[i * HOP : i * HOP + NFFT] * w for i in range(n)])
    p = np.abs(np.fft.rfft(frames, axis=-1)) ** 2 / (w**2).sum()
    return 10.0 * np.log10(np.maximum(p, 1e-30))


def _show_spec(ax, db, title, dyn_range=45.0, fmax=8000.0, dur=None):
    freqs = np.linspace(0, SR / 2, db.shape[1])
    keep = freqs <= fmax
    d = db[:, keep]
    top = float(np.percentile(d, 99.5))
    dur = dur if dur is not None else d.shape[0] * HOP / SR
    ax.imshow(
        np.clip(d, top - dyn_range, top).T,
        origin="lower",
        aspect="auto",
        extent=(0.0, dur, 0.0, freqs[keep][-1]),
        cmap="magma",
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("frequency (Hz)")
    ax.grid(False)


def _support(name_contains: str, prefer: str = "+4_"):
    cands = sorted(SUPPORTS.glob("*.npz"))
    hit = [p for p in cands if name_contains in p.name and prefer in p.name]
    if not hit:
        hit = [p for p in cands if name_contains in p.name]
    if not hit:
        raise RuntimeError(f"no support matching {name_contains!r}")
    return np.load(hit[0], allow_pickle=True), hit[0].name


def _render_on_label(rig_name: str, carrier: np.ndarray, seed: int) -> np.ndarray:
    import noise_lab as NL
    import tdseries as td

    carrier = np.ascontiguousarray(carrier, dtype=np.float64)
    step = max(int(round(SR / NL.RPS_PLOT_SR)), 1)
    traj = td.Frame(
        {
            "rps": td.uniform(
                np.ascontiguousarray(carrier[:, ::step]),
                NL.RPS_PLOT_SR,
                dims=("rotor", "time"),
                t_start=0.0,
            ),
            "rps_render": td.uniform(carrier, SR, dims=("rotor", "time"), t_start=0.0),
            "meta": td.Frame({"traj_kind": "real-label", "seed": int(seed), "rig": rig_name}),
        }
    )
    out = NL.render(NL.V2Fit(rig_name), traj, seed=seed, n_mics=1)
    return np.asarray(out["audio"].data)[0].astype(np.float64)


def fig_noise_real_vs_sampled() -> None:
    panels = []
    for tag, contains, rig_name in (
        ("DREGON room 2, hovering", "hovering_nosource_room2", "dregon"),
        ("Michael's FLY125, cruise", "michaels", "michaels"),
    ):
        z, fname = _support(contains)
        power = np.asarray(z["power"])[0]
        real_db = 10.0 * np.log10(np.maximum(power, 1e-30))
        carrier = np.asarray(z["carrier_rev_s_audio"], dtype=np.float64)
        dur = carrier.shape[1] / SR
        audio = _render_on_label(rig_name, carrier, seed=7)
        panels.append((tag, real_db, _spec_db(audio), carrier, dur, fname))

    fig, axes = plt.subplots(len(panels), 2, figsize=(11, 3.4 * len(panels)))
    axes = np.atleast_2d(axes)
    for row, (tag, real_db, samp_db, carrier, dur, fname) in enumerate(panels):
        mean_rps = float(np.mean(carrier))
        _show_spec(axes[row, 0], real_db, f"{tag} — real, {mean_rps:.0f} rev/s", dur=dur)
        _show_spec(axes[row, 1], samp_db, f"{tag} — fitted model on the same label", dur=dur)
        print(f"  {tag}: support {fname}", flush=True)
    fig.tight_layout()
    save(fig, "gen_noise_real_vs_sampled")


def _hard_pool(n_mics: int = 1):
    import yaml

    from data_processing.noise_v2_pool import NoiseV2Pool

    cfg = yaml.safe_load((ROOT / "conf/online_mix/noise_v2_hard_5050.yaml").read_text())
    src = dict(cfg["sources"]["noise"][0])
    src["n_mics"] = n_mics
    src["render_reuse"] = 1
    src["render_pool"] = 1
    src["normalize_rms_range"] = None
    return NoiseV2Pool.from_config(src, duration_s=4.0, sample_rate=SR)


def fig_noise_hyperprior(n: int = 4) -> None:
    pool = _hard_pool()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.4))
    rng = np.random.default_rng(2026)
    for i, ax in enumerate(axes.ravel()[:n]):
        frame = pool.sample_timeframe(rng, 4.0)
        audio = np.asarray(frame["audio"].data)[0].astype(np.float64)
        rps = np.asarray(frame["rps"].data)
        hover = float(np.mean(rps))
        name = ""
        with contextlib.suppress(Exception):
            name = str(frame["meta"]["entry"])
        label = name or f"entry {i + 1}"
        title = f"hard bank {label} — hyperprior flight, {hover:.0f} rev/s"
        _show_spec(ax, _spec_db(audio), title, dur=len(audio) / SR)
    fig.tight_layout()
    save(fig, "gen_noise_hyperprior")


# ── tables ──────────────────────────────────────────────────────────────────


def _booktabs(colspec: str, header: list[str], rows: list[list[str]], caption: str, label: str):
    esc = lambda s: s  # noqa: E731 - rows are authored already-escaped
    out = [
        "\\begin{table}[tb]",
        "  \\centering",
        "  \\small",
        f"  \\begin{{tabular}}{{{colspec}}}",
        "    \\toprule",
        "    " + " & ".join(esc(h) for h in header) + " \\\\",
        "    \\midrule",
    ]
    out += ["    " + " & ".join(esc(c) for c in r) + " \\\\" for r in rows]
    out += [
        "    \\bottomrule",
        "  \\end{tabular}",
        f"  \\caption{{{caption}}}",
        f"  \\label{{{label}}}",
        "\\end{table}",
        "",
    ]
    return "\n".join(out)


def write_table(name: str, colspec, header, rows, caption, label) -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / f"{name}.tex").write_text(_booktabs(colspec, header, rows, caption, label))
    print(f"wrote {name}.tex", flush=True)


def tables() -> None:
    write_table(
        "gen_traj_model_params",
        "llll",
        ["symbol", "meaning", "shape", "range / prior"],
        [
            ["$\\mu$", "per-rotor hover level (rev/s)", "4", "estimated from airborne samples"],
            [
                "$\\theta$",
                "rotation of the (collective, yaw) plane (rad)",
                "1",
                "$(-\\pi/4,\\pi/4]$ canonical",
            ],
            [
                "$\\tau_{\\mathrm{slow}}$",
                "slow OU time constant per mode (s)",
                "4",
                "$(\\tau_c,10]$, LN(2.0\\,s, 1.5)",
            ],
            [
                "$\\sigma_{\\mathrm{slow}}$",
                "slow OU stationary sd per mode (rev/s)",
                "4",
                "LN(1.0, 3.0)",
            ],
            ["$f_0$", "oscillator centre frequency per mode (Hz)", "4", "$(0.05,20)$"],
            ["$\\zeta$", "oscillator damping ratio per mode", "4", "$(0.2,3.0)$"],
            [
                "$\\sigma_{\\mathrm{osc}}$",
                "oscillator stationary sd per mode (rev/s)",
                "4",
                "LN(1.0, 3.0)",
            ],
            [
                "$\\tau_e$",
                "measurement OU time constant (s)",
                "1",
                "$(10^{-3},1)$, LN(0.05\\,s, 2.0)",
            ],
            ["$\\sigma_e$", "measurement OU sd, per rotor (rev/s)", "1", "LN(0.1, 3.0)"],
            [
                "$s_c$",
                "sd of the per-flight level common to the rotors (rev/s)",
                "1",
                "moment estimate",
            ],
            ["$s_r$", "sd of the per-flight per-rotor level (rev/s)", "4", "moment estimate"],
        ],
        "Parameters of the rotor-speed trajectory model (32 in total; 23 enter the "
        "likelihood, $\\mu$, $s_c$ and $s_r$ are estimated from the airborne samples). "
        "$\\tau_c = 1/(2\\pi f_0)$ orders the two shaft components without a label-switching gauge. "
        "LN$(m,s)$: log-normal with median $m$ and log-sd $s$.",
        "tab:gen-traj-model-params",
    )

    write_table(
        "gen_traj_data",
        "llrrll",
        ["rig", "vehicle", "flights", "airborne (s)", "native rate", "speed source"],
        [
            [
                "\\texttt{michaels}",
                "DJI Matrice 100",
                "4",
                "395",
                "29.41\\,Hz",
                "DatCon ESC feedback",
            ],
            [
                "\\texttt{dregon}",
                "MikroKopter quad",
                "10",
                "429",
                "$\\sim$1\\,kHz grid",
                "\\texttt{motor.measured}",
            ],
            [
                "\\texttt{neurobem\\_quad}",
                "custom quad",
                "247",
                "2479",
                "400\\,Hz",
                "Betaflight ESC feedback",
            ],
            [
                "\\texttt{pitcn\\_quad}",
                "micro quad, 0.25\\,kg",
                "68",
                "3345",
                "100\\,Hz",
                "ESC telemetry",
            ],
            [
                "\\texttt{nanobench\\_cf21b}",
                "Crazyflie 2.1 Brushless",
                "15",
                "668",
                "100\\,Hz",
                "bidirectional DSHOT eRPM",
            ],
            [
                "\\texttt{vid\\_m100}",
                "DJI M100 derivative",
                "4",
                "397",
                "$\\sim$1\\,kHz CAN",
                "M3508/C620 CAN feedback",
            ],
            [
                "\\texttt{blackbird\\_quad}",
                "custom quad",
                "1",
                "199",
                "187\\,Hz",
                "optical motor encoders",
            ],
            ["\\textbf{total}", "", "\\textbf{349}", "\\textbf{7912}", "", ""],
        ],
        "The seven ingested rigs, on the frozen airborne rule and the common 100\\,Hz "
        "analysis grid. Hover level spans 78--278\\,rev/s and per-rotor variance a factor of 600.",
        "tab:gen-traj-data",
    )

    write_table(
        "gen_noise_model_params",
        "lllp{0.30\\linewidth}",
        ["symbol", "meaning", "shape", "prior"],
        [
            [
                "$\\sigma_\\nu$",
                "shaft OU speed-error sd (rad/s)",
                "per rotor",
                "bench LN(0.3, 0.7); flight LN(0.3, 0.5)",
            ],
            [
                "$\\lambda$",
                "shaft OU rate (s$^{-1}$)",
                "per rotor",
                "LN(2, 1.0) on the bench; pinned in flight",
            ],
            [
                "$\\gamma_{rk}$",
                "Lorentzian half-width of line $(r,k)$ (Hz)",
                "rotor $\\times$ order",
                "LN($0.01k$\\,Hz, 1.0), $\\gamma\\ge0$",
            ],
            [
                "$p_{rk}$",
                "line power of order $k$ (dB), the comb profile",
                "rotor $\\times$ order",
                "N(measured, 10) if SNR $\\ge$ 2\\,dB, else N($\\mu_F-15$, 8)",
            ],
            ["$c$", "level of a transplanted comb (dB)", "1", "N(0, 20)"],
            ["$a$", "comb speed exponent", "1", "N(2, 1); pinned at 2 on a short span"],
            ["$b$", "floor speed exponent", "1", "LN(2, 0.5); same span pin"],
            ["$s$", "static fraction of the floor", "1", "LN($2.5\\times10^{-3}$, 1.0)"],
            ["$\\mu_F$", "floor level (dB)", "1", "N(measured band median, 10)"],
            ["$z$", "GP control points of the floor shape", "14", "N(0, 1) each"],
            ["$t_F$", "floor tilt (dB/octave)", "1", "N(0, 5)"],
            [
                "$g_{mr}$",
                "per-microphone line gain (dB), mean-pinned",
                "mic $\\times$ rotor",
                "N(0, 6)",
            ],
            ["$h_m$", "per-microphone floor gain (dB)", "mic", "N(0, 6)"],
            ["$G_m$", "per-microphone overall gain (dB), mean-pinned", "mic", "N(0, 6)"],
            ["$f_r$", "carrier (rev/s)", "per rotor", "given: window-refined or telemetry label"],
        ],
        "Parameters of the rotor-noise model and their priors. LN$(m,s)$: log-normal with "
        "median $m$ and log-sd $s$; N$(\\mu,\\sigma)$: normal.",
        "tab:gen-noise-model-params",
    )

    write_table(
        "gen_noise_data",
        "lllrl",
        ["recording set", "class", "rotors / mics", "duration (s)", "role in the fit"],
        [
            [
                "DREGON single-motor bench, 50--90\\,\\%",
                "bench",
                "1 / 8",
                "12--34 per window",
                "identifies $\\sigma_\\nu$, $\\lambda$, $\\gamma_{rk}$",
            ],
            [
                "DREGON \\texttt{motor\\_allMotors\\_70} bench",
                "bench",
                "4 / 8",
                "$\\sim$30",
                "four-rotor control",
            ],
            [
                "DREGON room 2 free flight / hover / rectangle / spinning",
                "flight",
                "4 / 8",
                "4 per window, pooled",
                "floor and flight profile (R5 fit)",
            ],
            [
                "Michael's FLY125 cruise",
                "flight",
                "4 / 1",
                "4 per window, pooled",
                "cruise comb and floor (R3 fit)",
            ],
            [
                "Michael's FLY125 standby",
                "flight",
                "4 / 1",
                "4 per window, pooled",
                "standby regime of the same rig",
            ],
        ],
        "Recordings behind the shipped noise fits. Flight supports are periodic-Hann frames, "
        "$N=2048$, hop 512 at 16\\,kHz, pooled over windows; bench supports are one stationary "
        "periodogram each. The fit band is $30\\,\\mathrm{Hz} \\le f \\le 7900\\,\\mathrm{Hz}$.",
        "tab:gen-noise-data",
    )

    ref = "\\textit{real reference}"
    write_table(
        "gen_results_by_dataset_regime",
        "lllrrrrrr",
        [
            "trunk",
            "arm",
            "status",
            "real\\_overall",
            "overall",
            "zero",
            "standby",
            "ramp",
            "cruise",
        ],
        [
            [
                "SCv2",
                f"\\texttt{{real\\_r4\\_scv2\\_unified}} ({ref})",
                "done",
                "\\textbf{2.99}",
                "3.11",
                "2.44",
                "3.20",
                "8.17",
                "2.95",
            ],
            [
                "SCv2",
                "\\texttt{nv2\\_easy\\_scv2} (synthetic only)",
                "done",
                "7.94",
                "7.99",
                "10.64",
                "11.05",
                "11.12",
                "6.87",
            ],
            [
                "SCv2",
                "\\texttt{nv2\\_hard\\_scv2} (synthetic only)",
                "done",
                "7.06",
                "7.05",
                "1.25",
                "6.16",
                "15.81",
                "7.76",
            ],
            [
                "SCv2",
                "\\texttt{nv2\\_mixed\\_scv2} (mixed)",
                "done",
                "2.53",
                "---",
                "---",
                "---",
                "---",
                "---",
            ],
            [
                "SCv2",
                "\\texttt{nv2\\_easy\\_ft\\_scv2} (easy $\\to$ real)",
                "done",
                "\\textbf{2.73}",
                "2.73",
                "2.23",
                "2.36",
                "8.16",
                "2.60",
            ],
            [
                "SCv2",
                "\\texttt{nv2\\_hard\\_ft\\_scv2} (hard $\\to$ real)",
                "running, r32",
                "2.23",
                "---",
                "---",
                "---",
                "---",
                "---",
            ],
            [
                "HPPNet L2",
                f"\\texttt{{hppnet\\_l2\\_r2\\_s0}} ({ref})",
                "done",
                "\\textbf{2.27}",
                "---",
                "---",
                "---",
                "---",
                "---",
            ],
            [
                "HPPNet L2",
                "\\texttt{nv2\\_easy\\_hppnet\\_l2} (synthetic only)",
                "done",
                "6.46",
                "6.45",
                "3.58",
                "5.04",
                "26.50",
                "6.14",
            ],
            [
                "HPPNet L2",
                "\\texttt{nv2\\_hard\\_hppnet\\_l2} (synthetic only)",
                "running, r93",
                "6.80",
                "---",
                "---",
                "---",
                "---",
                "---",
            ],
            [
                "HPPNet L2",
                "\\texttt{nv2\\_mixed\\_hppnet\\_l2} (mixed)",
                "running, r54",
                "2.48",
                "---",
                "---",
                "---",
                "---",
                "---",
            ],
            [
                "HPPNet L2",
                "\\texttt{nv2\\_easy\\_ft\\_hppnet\\_l2} (easy $\\to$ real)",
                "queued",
                "---",
                "---",
                "---",
                "---",
                "---",
                "---",
            ],
        ],
        "Rotor-speed MAE (rev/s, lower is better) on the frozen real validation split. "
        "\\texttt{real\\_overall} is \\texttt{val/real\\_r3} at the selected checkpoint; the four "
        "regime columns are the per-frame PIT MAE of \\texttt{scripts/\\_regime\\_decomp.py} on the "
        "same checkpoint. Running arms quote the value at the round stated. The regime "
        "decomposition is pooled over DREGON and Michael's: the committed decomposition "
        "carries no per-dataset split.",
        "tab:gen-results-by-regime",
    )


def main() -> None:
    which = sys.argv[1:] or ["tables", "traj", "traj_other", "hyper", "noise", "bank"]
    if "tables" in which:
        tables()
    if "traj" in which:
        fig_traj_real_vs_sampled()
    if "traj_other" in which:
        fig_traj_real_vs_sampled(
            ("pitcn_quad", "nanobench_cf21b", "vid_m100"),
            name="gen_traj_real_vs_sampled_other",
            row_h=2.9,
        )
    if "hyper" in which:
        fig_traj_hyperprior()
    if "noise" in which:
        fig_noise_real_vs_sampled()
    if "bank" in which:
        fig_noise_hyperprior()


if __name__ == "__main__":
    main()
