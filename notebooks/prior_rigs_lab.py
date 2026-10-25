"""Prior-rig lab: drones drawn from the generic prior
(:mod:`experiments.noise_model.drone_prior`), gated for identifiability on
trajectory-hyperprior windows, each rendered into 8-channel recordings on
fresh hyperprior trajectories. Driven by ``prior_rigs_explainer.ipynb``
(kernel cwd ``notebooks/``).

Artefacts: ``results/prior_rigs/<name>/rig_<i>.json`` (the payloads, with
``_prior`` provenance and the gate counts) and
``.cache/prior_rigs/<name>/rig_<i>_rec_<j>.npz`` (audio ``(8, T)`` at 16 kHz,
the trajectory ``(R, T)`` rev/s at 100 Hz).
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from experiments.noise_model import drone_prior as PR
from experiments.noise_model.identifiability import GateConfig, frame_pool, gate

SR = 16000
TRAJ_FS = 100.0
RESULTS = Path("../results/prior_rigs")
CACHE = Path("../.cache/prior_rigs")


@dataclass
class Lab:
    name: str
    rigs: list[dict[str, Any]]
    stats: dict[str, Any]
    duration_s: float
    n_rec: int
    audio: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def recording(self, i: int, j: int) -> tuple[np.ndarray, np.ndarray]:
        """``(audio (8, T) at 16 kHz, rps (R, T100) at 100 Hz)``."""
        return self.audio[(i, j)]


def build(
    name: str = "prior_v1_seed0",
    *,
    seed: int = 0,
    n_rigs: int = 10,
    n_rec: int = 3,
    duration_s: float = 8.0,
    threshold_db: float = 3.0,
    min_orders: int = 2,
    spec: PR.PriorSpec = PR.PriorSpec(),
    n_gate_windows: int = 8,
    force: bool = False,
) -> Lab:
    """Draw ``n_rigs`` gated drones and render ``n_rec`` recordings each;
    everything cached under ``name`` (``force`` redraws)."""
    rdir, cdir = RESULTS / name, CACHE / name
    rdir.mkdir(parents=True, exist_ok=True)
    cdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    meta_path = rdir / "stats.json"
    if meta_path.exists() and not force:
        stats = json.loads(meta_path.read_text())
        rigs = [json.loads((rdir / f"rig_{i}.json").read_text()) for i in range(stats["accepted"])]
    else:
        t0 = time.time()
        wins = PR.trajectory_windows(rng, n_gate_windows, duration_s)
        pool = frame_pool(wins, n_frames=80, seed=seed)
        rigs, stats = PR.gated_draws(
            rng,
            n_rigs,
            spec=spec,
            pool=pool,
            threshold_db=threshold_db,
            min_orders=min_orders,
            max_attempts=20 * n_rigs,
        )
        # frames with >= 2 orders at 6 dB per rotor (information, not a gate)
        strict = GateConfig(threshold_db=6.0, min_orders=2, pass_frac=0.0, min_eligible_frames=1)
        for r in rigs:
            res = gate(r, pool, strict)
            r["_prior"]["gate_6db_frames_ok"] = res.frames_ok.tolist()
        stats["gate_s"] = time.time() - t0
        for i, r in enumerate(rigs):
            (rdir / f"rig_{i}.json").write_text(json.dumps(r))
        meta_path.write_text(json.dumps(stats))
    lab = Lab(name=name, rigs=rigs, stats=stats, duration_s=duration_s, n_rec=n_rec)
    from data_processing.noise_model.render import render_noise

    for i, rig in enumerate(rigs):
        for j in range(n_rec):
            path = cdir / f"rig_{i}_rec_{j}.npz"
            if path.exists() and not force:
                z = np.load(path)
                lab.audio[(i, j)] = (z["audio"], z["rps"])
                continue
            t0 = time.time()
            rrng = np.random.default_rng([seed, i, j])
            rps16 = PR.trajectory_windows(rrng, 1, duration_s, full_flight=False)[0]
            rps100 = rps16[:, :: int(SR / TRAJ_FS)]
            audio = render_noise(rig, rps16, sr=SR, n_mics=8, seed=int(rrng.integers(2**31)))
            audio = (audio / max(float(np.abs(audio).max()), 1e-9) * 0.5).astype(np.float32)
            np.savez(path, audio=audio, rps=rps100.astype(np.float32))
            lab.audio[(i, j)] = (audio, rps100)
            print(
                f"rig {i} rec {j}: rendered {duration_s:.0f} s x 8 ch in {time.time() - t0:.0f} s",
                flush=True,
            )
    return lab


def summary(lab: Lab, i: int) -> str:
    p = lab.rigs[i]["params"]
    d = lab.rigs[i]["_prior"]["drawn"]
    g = lab.rigs[i]["_prior"].get("gate", {})
    g6 = lab.rigs[i]["_prior"].get("gate_6db_frames_ok")
    am = np.array(p["am"]["sigma2"])[0]
    k = np.arange(1, am.size + 1)
    return (
        f"rig {i}: shaft σ_ν {p['sigma_nu']:.2f} rad/s ({p['sigma_nu'] / 6.283:.3f} rev/s), λ {p['lam']:.1f} 1/s; "
        f"amp_exp {p['profile']['amp_exp']:.1f}, floor_exp {p['floor']['floor_exp']:.1f}; comb over the floor: k=2 {d['k2_over_floor_db']:.0f} dB, {d['slope_db_dec']:.0f} dB/dec down to a {d['tail_over_floor_db']:+.0f} dB tail; "
        f"motor family every {d['motor_period']} orders (+{d['motor_boost_db']:.0f} dB); floor {p['floor']['floor_mean_db']:.0f} dB, {d['floor_template']}, σ_B {p['floor']['floor_shape_sd_db']:.1f}; "
        f"AM σ² median k≤8 {np.median(am[k <= 8]):.3f}, 9–30 {np.median(am[(k > 8) & (k <= 30)]):.3f}, >30 {np.median(am[k > 30]):.2f} (σ_ψ {d['sigma_psi']:.3f} rad); "
        f"mic scatter {p['profile']['mic_dev_sd_db']} dB; gate {g.get('threshold_db')} dB: frames ok {g.get('frames_ok')} of {g.get('eligible')}; 6 dB/2 orders: {g6}"
    )


def plot_rig(lab: Lab, i: int, speeds: Any = None, fmax: float = 7900.0) -> Any:
    """The parameter view of rig ``i`` as ``noise_lab.rig_view_plotly``: a
    dropdown picks ``all rotors`` / ``rotor mean`` / one rotor (its stems at
    ``k f_r`` and red caps of +/- the line's half width over the grey floor),
    a slider sets the common rotor speed (``speeds``, default 20-120 rev/s by
    5). The per-order AM pedestals are not in the expectation drawn here."""
    import noise_lab as NL

    return NL.rig_view_plotly(lab.rigs[i], speeds=speeds, fmax=fmax, title=f"rig {i}")


def rig_browser(lab: Lab, **kw: Any):
    """Dropdown (rig) → :func:`plot_rig` (rotor dropdown + speed slider inside)."""
    import ipywidgets as W
    from IPython.display import display

    w_rig = W.Dropdown(options=list(range(len(lab.rigs))), description="rig")
    out = W.Output()

    def refresh(*_):
        with out:
            out.clear_output(wait=True)
            print(summary(lab, w_rig.value))
            display(plot_rig(lab, w_rig.value, **kw))

    w_rig.observe(refresh, "value")
    refresh()
    return W.VBox([w_rig, out])


def plot_trajectory(lab: Lab, i: int, j: int) -> None:
    _, rps = lab.recording(i, j)
    t = np.arange(rps.shape[1]) / TRAJ_FS
    fig, ax = plt.subplots(figsize=(18, 2.2))
    for r in range(rps.shape[0]):
        ax.plot(t, rps[r], lw=1, label=f"rotor {r}")
    ax.set_xlabel("s")
    ax.set_ylabel("rev/s")
    ax.set_title(f"rig {i} recording {j}: hyperprior trajectory")
    ax.legend(ncol=4, fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()


def view(lab: Lab, i: int, j: int, **kw: Any):
    """A :func:`plots.spectrum_viewer` over recording ``j`` of rig ``i``, with an
    audio player of the viewed channel (follows the channel dropdown; the clip
    is peak-normalised, so no absolute level is implied)."""
    import ipywidgets as W
    from IPython.display import Audio, display

    from plots.spectrum_viewer import spectrum_viewer

    audio, _ = lab.recording(i, j)
    viewer = spectrum_viewer(audio, sr=SR, title=f"rig {i} rec {j}", **kw)
    player = W.Output()

    def play(*_):
        with player:
            player.clear_output(wait=True)
            ch = int(viewer.w_channel.value)
            display(Audio(audio[ch], rate=SR, normalize=False))

    viewer.w_channel.observe(play, "value")
    play()
    viewer.widget = W.VBox([viewer.widget, player])
    return viewer


def browser(lab: Lab):
    """Dropdowns (rig, recording) → trajectory plot + spectrum viewer + player."""
    import ipywidgets as W
    from IPython.display import display

    w_rig = W.Dropdown(options=list(range(len(lab.rigs))), description="rig")
    w_rec = W.Dropdown(options=list(range(lab.n_rec)), description="recording")
    out = W.Output()

    def refresh(*_):
        with out:
            out.clear_output(wait=True)
            print(summary(lab, w_rig.value))
            plot_trajectory(lab, w_rig.value, w_rec.value)
            display(view(lab, w_rig.value, w_rec.value).widget)

    w_rig.observe(refresh, "value")
    w_rec.observe(refresh, "value")
    refresh()
    return W.VBox([W.HBox([w_rig, w_rec]), out])


# ── a prior rig on a REAL clip's trajectory ─────────────────────────────────

#: The frozen real validation split of the unified protocol
#: (``conf/validation/rps_unified.yaml``): 37 clips of 8 s, eight channels.
VALID_SET = "dload:DREGON-LM-V4-michaels-valid-full"
VALID_GROUPS = ((0, 14, "DREGON, speech present"), (14, 22, "DREGON, no source"), (22, 37, "MD2"))
_VALID: Any = None


def valid_clip_label(j: int) -> str:
    for lo, hi, name in VALID_GROUPS:
        if lo <= j < hi:
            return f"clip {j}: {name}"
    return f"clip {j}"


def valid_clip(j: int) -> tuple[np.ndarray, np.ndarray]:
    """``(audio (8, T) at 16 kHz, rps (R, T) at 16 kHz)`` of validation clip ``j``:
    the mixture and its rotor track (the dataset's STFT-frame track, linearly
    interpolated onto the audio grid)."""
    from data_processing.frame_datasets import DregonLMFrameDataset

    global _VALID
    if _VALID is None:
        _VALID = DregonLMFrameDataset(VALID_SET, flatten_channels=False)
    fr = _VALID[int(j)]
    audio = np.asarray(fr["mixture"].data, dtype=np.float64)
    rps = np.asarray(fr["rps"].data, dtype=np.float64)  # (R, n_frames)
    t_frames = np.linspace(0.0, audio.shape[-1] / SR, rps.shape[-1])
    t = np.arange(audio.shape[-1]) / SR
    return audio, np.stack([np.interp(t, t_frames, row) for row in rps])


def render_on_clip(
    lab: Lab, i: int, j: int, *, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(real (8, T), rendered (8, T), rps (R, T))``: rig ``i`` rendered on
    validation clip ``j``'s own rotor track, scaled so its mic-mean RMS equals
    the clip's (a prior rig's absolute level is in the fits' units, the clip
    in the recorder's; the comparison is of spectral SHAPE and structure)."""
    from data_processing.noise_model.render import render_noise

    real, rps = valid_clip(j)
    rig = lab.rigs[i]
    n_rot = int(np.asarray(rig["params"]["profile"]["profile_db"]).shape[0])
    if rps.shape[0] != n_rot:
        raise ValueError(f"clip {j} has {rps.shape[0]} rotors, rig {i} has {n_rot}")
    rendered = np.asarray(render_noise(rig, rps, sr=SR, n_mics=real.shape[0], seed=int(seed)))
    rendered = rendered * (np.sqrt(np.mean(real**2)) / max(np.sqrt(np.mean(rendered**2)), 1e-12))
    return real, rendered, rps


def compare(lab: Lab, i: int, j: int, *, seed: int = 0, **kw: Any):
    """Spectrum viewer of validation clip ``j`` with rig ``i``'s render on the
    same trajectory as the overlay (channel dropdown, spectrogram toggle),
    the trajectory, and a player for each (the viewed channel)."""
    import ipywidgets as W
    from IPython.display import Audio, display

    from plots.spectrum_viewer import spectrum_viewer

    real, rendered, rps = render_on_clip(lab, i, j, seed=seed)
    viewer = spectrum_viewer(
        real,
        sr=SR,
        overlay=rendered,
        labels=(valid_clip_label(j), f"rig {i} on its trajectory"),
        title=f"{valid_clip_label(j)} vs rig {i}",
        **kw,
    )
    players = W.Output()
    peak = max(float(np.abs(real).max()), float(np.abs(rendered).max()), 1e-12)

    def play(*_):
        with players:
            players.clear_output(wait=True)
            ch = int(viewer.w_channel.value)
            for name, x in ((f"real, ch {ch}", real[ch]), (f"rig {i}, ch {ch}", rendered[ch])):
                print(name)
                display(Audio(x / peak * 0.9, rate=SR, normalize=False))

    viewer.w_channel.observe(play, "value")
    play()
    traj = W.Output()
    with traj:
        t = np.arange(rps.shape[1]) / SR
        fig, ax = plt.subplots(figsize=(18, 2.2))
        for r in range(rps.shape[0]):
            ax.plot(t, rps[r], lw=0.8, label=f"rotor {r}")
        ax.set_ylabel("rev/s")
        ax.set_xlabel("s")
        ax.legend(loc="upper right", fontsize=8, ncol=4)
        ax.grid(alpha=0.3)
        fig.tight_layout()
        plt.show()
    viewer.widget = W.VBox([traj, viewer.widget, players])
    return viewer


def compare_browser(lab: Lab, **kw: Any):
    """Dropdowns (rig, validation clip, seed) → :func:`compare`."""
    import ipywidgets as W
    from IPython.display import display

    w_rig = W.Dropdown(options=list(range(len(lab.rigs))), description="rig")
    w_clip = W.Dropdown(
        options=[(valid_clip_label(j), j) for j in range(VALID_GROUPS[-1][1])],
        value=VALID_GROUPS[1][0],
        description="clip",
    )
    w_seed = W.IntText(value=0, description="seed")
    out = W.Output()

    def refresh(*_):
        with out:
            out.clear_output(wait=True)
            print(summary(lab, w_rig.value))
            display(compare(lab, w_rig.value, w_clip.value, seed=int(w_seed.value), **kw).widget)

    for w in (w_rig, w_clip, w_seed):
        w.observe(refresh, "value")
    refresh()
    return W.VBox([W.HBox([w_rig, w_clip, w_seed]), out])


# ── v1 prior rigs vs the v4 bank with the proposed phase pedestal ─────────────

#: The three renders of one bank rig, in display order.
PED_VARIANTS = ("v1 rig (as trained)", "v4 converted", "v4 + phase pedestal (no guard)")
#: Whether the third variant applies the floor guard
#: (:func:`experiments.noise_model.prior_v4.guard_phase_pedestal`); off since
#: 2026-10-25 (Dmitrii: show the pedestal without it).
PED_GUARD = False
#: Bank rigs offered by :func:`pedestal_browser`: the median rig of the
#: sector survey, then ``default_rng(20261025).choice(16384, 5)`` (any index
#: can be typed in).
PED_RIGS = (13504, 2332, 2624, 2913, 10209, 10390)
#: The ``steady`` trajectory: constant, distinct speeds per rotor (rev/s), so
#: the line SHAPES are visible in a long spectrum.
STEADY_RPS = (72.0, 76.0, 80.0, 84.0)
_V1_BANK: Any = None


def _v1_bank() -> Any:
    from data_processing.noise_model.prior_bank import PriorBank
    from data_processing.streams import resolve_source
    from experiments.noise_model import prior_v4 as PV4

    global _V1_BANK
    if _V1_BANK is None:
        _V1_BANK = PriorBank(resolve_source(PV4.SOURCE_BANK))
    return _V1_BANK


def pedestal_payloads(i: int) -> dict[str, dict[str, Any]]:
    """Bank rig ``i`` three ways (:data:`PED_VARIANTS`): the v1 payload the
    prior arms trained on, its v4 conversion under the CURRENT rules
    (:func:`experiments.noise_model.prior_v4.convert_prior_v1`, so the
    published ``prior-v4-banks`` may lag them), and the conversion with the
    proposed phase pedestal (:func:`experiments.noise_model.prior_v4.with_phase_pedestal`
    on the rig's third stream ``SeedSequence([SEED, i]).spawn(3)[2]``, floor
    guard per :data:`PED_GUARD`)."""
    from experiments.noise_model import prior_v4 as PV4

    bank = _v1_bank()
    v1 = bank.payload(int(i))
    v4 = PV4.convert_prior_v1(bank, int(i), seed=PV4.SEED)
    rng = np.random.default_rng(np.random.SeedSequence([PV4.SEED, int(i)]).spawn(3)[2])
    ped = PV4.with_phase_pedestal(v4, rng, guard=PED_GUARD)
    return dict(zip(PED_VARIANTS, (v1, v4, ped), strict=True))


def pedestal_trajectory(i: int, traj: str | int, duration_s: float = 8.0) -> np.ndarray:
    """``(R, T)`` rev/s at 16 kHz: ``"steady"`` (:data:`STEADY_RPS`) or a
    hyperprior airborne window drawn from ``default_rng([traj, i])``."""
    n = int(round(duration_s * SR))
    if traj == "steady":
        return np.repeat(np.asarray(STEADY_RPS, dtype=np.float64)[:, None], n, axis=1)
    return PR.trajectory_windows(
        np.random.default_rng([int(traj), int(i)]), 1, duration_s, full_flight=False
    )[0][:, :n]


def pedestal_renders(i: int, traj: str | int = "steady", *, seed: int = 0, duration_s: float = 8.0):
    """``(audio {variant: (8, T)}, rps (R, T), payloads)``: the three variants
    on ONE trajectory with one render seed, the v4 renders with v1's per-clip
    line scatter (``line_scatter_db = mic_dev_sd_db``) and per-clip wind (the
    render option ``wind_sc`` = the v1 rig's ``wind_sc`` law), all scaled by ONE
    factor (their joint peak to 0.9), so loudness differences are real.
    Cached under ``.cache/prior_rigs/phase_pedestal/``, per :data:`PED_GUARD`
    and conversion rules version."""
    from data_processing.noise_model.render import render_noise
    from experiments.noise_model import prior_v4 as PV4

    pays = pedestal_payloads(i)
    tag = ("guard" if PED_GUARD else "noguard") + "_" + PV4.RULES_VERSION.replace("/", "")
    path = CACHE / "phase_pedestal" / f"rig_{int(i)}_traj_{traj}_seed_{int(seed)}_{tag}.npz"
    rps = pedestal_trajectory(i, traj, duration_s)
    if path.exists():
        z = np.load(path)
        return {v: z[f"a{j}"] for j, v in enumerate(PED_VARIANTS)}, rps, pays
    p1 = pays[PED_VARIANTS[0]]["params"]
    v4_kw = {"line_scatter_db": float(p1["profile"]["mic_dev_sd_db"]), "wind_sc": p1["wind_sc"]}
    audio = {}
    for v, pay in pays.items():
        kw = {} if v == PED_VARIANTS[0] else v4_kw
        audio[v] = np.asarray(render_noise(pay, rps, sr=SR, n_mics=8, seed=int(seed), **kw))
    peak = max(float(np.abs(a).max()) for a in audio.values())
    audio = {v: (a / max(peak, 1e-12) * 0.9).astype(np.float32) for v, a in audio.items()}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **{f"a{j}": audio[v] for j, v in enumerate(PED_VARIANTS)})
    return audio, rps, pays


def _db(x: np.ndarray) -> np.ndarray:
    return 10.0 * np.log10(np.maximum(x, 1e-20))


def plot_pedestal_params(pays: dict[str, dict[str, Any]]) -> None:
    """Phase-pedestal ``sigma2`` and mean half-width ``W`` per order: the v4
    conversion (v1's leftover, slow) against the proposal, with the bench
    group medians it is drawn around and the caps."""
    from experiments.noise_model import prior_v4 as PV4
    from experiments.noise_model import v4_priors as VP

    fig, axs = plt.subplots(1, 2, figsize=(18, 2.8))
    for v, c in ((PED_VARIANTS[1], "tab:blue"), (PED_VARIANTS[2], "tab:red")):
        ped = pays[v]["params"]["pedestal"]
        s2 = np.asarray(ped["sigma2_phase"], dtype=np.float64)
        g = np.asarray(ped["gamma_phase_hz"], dtype=np.float64)
        w = np.asarray(ped.get("width_phase_hz") or VP.pedestal_width(g, np.maximum(s2, 1e-12)))
        k = np.arange(1, s2.size + 1)
        axs[0].plot(k, s2, ".", ms=4, color=c, label=v)
        axs[1].plot(k, np.where(s2 > 1e-6, w, np.nan), ".", ms=4, color=c, label=v)
    for lo, hi, s2m, _, wm, _ in PV4.PHASE_GROUPS:
        hi = min(hi, k.size)
        axs[0].hlines(s2m, lo, hi, color="k", lw=1)
        axs[1].hlines(wm, lo, hi, color="k", lw=1)
    axs[0].axhline(VP.PHASE_SIGMA2_MAX, color="grey", ls=":")
    axs[1].axhline(VP.W_MAX_HZ, color="grey", ls=":")
    axs[0].set(
        yscale="log",
        xlabel="order k",
        ylabel="phase σ² (rad²)",
        title="phase pedestal variance (black: bench medians; dotted: cap)",
    )
    axs[1].set(yscale="log", xlabel="order k", ylabel="W (Hz)", title="mean half-width")
    for ax in axs:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    shrunk = pays[PED_VARIANTS[2]].get("phase_pedestal", {}).get("guard_shrunk")
    fig.suptitle(
        "no floor guard" if shrunk is None else f"floor guard shrank {shrunk} orders", fontsize=9
    )
    fig.tight_layout()
    plt.show()


# ── the rig profile: needles, pedestals, motor lines, floor (interactive) ─────

#: Line families of the profile view and their colours.
FAMILY_COLORS = {
    "aero": "royalblue",
    "motor": "darkorange",
    "aero + v1 motor boost": "purple",
}
_FAMILY_CODE = {name: j for j, name in enumerate(FAMILY_COLORS)}
#: Where the profile grid is refined around each line, in units of its widths.
_REFINE = np.array([-4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0])


def _shaft_hwhm_hz(k: Any, sigma_nu: float, lam: float) -> np.ndarray:
    """Approximate half width (Hz) of order ``k``'s needle from the shaft FM
    alone: the smaller of the diffusion Lorentzian ``k^2 sigma^2 / (2 pi
    lam)`` (valid for ``k sigma << lam``) and the frozen-rate Gaussian HWHM
    ``sqrt(2 ln 2) k sigma / (2 pi)`` (valid for ``k sigma >> lam``),
    ``sigma`` the shaft OU's rad/s scale (:func:`.ou.integrated_ou_increment_var`)."""
    k = np.asarray(k, dtype=np.float64)
    lor = k**2 * sigma_nu**2 / (2.0 * np.pi * lam)
    gau = np.sqrt(2.0 * np.log(2.0)) * k * sigma_nu / (2.0 * np.pi)
    return np.minimum(lor, gau)


def profile_lines(pay: dict[str, Any], f0: float) -> dict[str, np.ndarray]:
    """Every line of ``pay`` with all rotors at ``f0`` rev/s, one entry per
    (rotor, line): ``rotor``, ``k``, ``family`` (code into
    :data:`FAMILY_COLORS`), ``f`` (Hz), ``power`` (the line's total variance,
    mic mean, in the render's units), the pedestal ``s2a, ga, s2p, gp`` (the
    needle keeps ``e^-(s2a + s2p)`` of the power) and ``shaft_hw`` (Hz,
    :func:`_shaft_hwhm_hz`). A v4 payload's motor line has no pedestal; a v1
    payload (``noise-v3-fit/1``) has one log-OU amplitude pedestal per (rotor,
    order), its core ``10^{profile_db/10} s^amp_exp`` is the needle, and its
    motor family is a boost of the aerodynamic line at ``k = 3 p n``."""
    from data_processing.noise_model import v4 as V4
    from data_processing.noise_model.constants import AMP_RPS_REF

    cols: dict[str, list[np.ndarray]] = {
        c: [] for c in ("rotor", "k", "family", "power", "s2a", "ga", "s2p", "gp", "shaft_hw")
    }

    def add(r: int, k: np.ndarray, fam: Any, power: np.ndarray, ped: tuple, hw: np.ndarray) -> None:
        for c, a in zip(
            cols,
            (np.full(k.size, r), k, np.broadcast_to(fam, k.shape), power, *ped, hw),
            strict=True,
        ):
            cols[c].append(np.asarray(a, dtype=np.float64))

    if "profile_aero_db" in pay["params"]:
        par = V4.V4Params.from_payload(pay)
        s = par.speed(np.full(par.n_rotors, float(f0)))
        p_aero, p_motor = par.line_power(s)
        ped = par.pedestal
        k = np.arange(1, par.k_max + 1, dtype=np.float64)
        km = np.asarray(par.motor_orders, dtype=np.float64)
        zero = np.zeros(km.size)
        for r in range(par.n_rotors):
            sig, lam = float(par.shaft.sigma_nu(s[r])), float(par.shaft.lam(s[r]))
            g_a = (10.0 ** (par.mic.db(k * f0)[r] / 10.0)).mean(axis=0)
            g_m = (10.0 ** (par.mic.db(km * f0)[r] / 10.0)).mean(axis=0)
            pa = (ped.sigma2_amp, ped.gamma_amp_hz, ped.sigma2_phase, ped.gamma_phase_hz)
            add(r, k, _FAMILY_CODE["aero"], g_a * p_aero[r], pa, _shaft_hwhm_hz(k, sig, lam))
            add(
                r,
                km,
                _FAMILY_CODE["motor"],
                g_m * p_motor[r],
                (zero,) * 4,
                _shaft_hwhm_hz(km, sig, lam),
            )
    else:
        p, d = pay["params"], pay["_prior"]["drawn"]
        prof = np.asarray(p["profile"]["profile_db"], dtype=np.float64)
        R, K = prof.shape
        s = float(f0) / float(AMP_RPS_REF)
        core = 10.0 ** (prof / 10.0) * s ** float(p["profile"]["amp_exp"])
        s2 = np.asarray(p["am"]["sigma2"], dtype=np.float64)
        gam = np.broadcast_to(np.asarray(p["am"]["gamma_hz"], dtype=np.float64), (R, K))
        k = np.arange(1, K + 1, dtype=np.float64)
        period = int(d["motor_period"])
        n = np.arange(1, K // period + 1)
        boost = float(d["motor_boost_db"]) * float(
            pay["_prior"]["spec"]["motor_decay_per_multiple"]
        ) ** (n - 1)
        fam = np.full(K, _FAMILY_CODE["aero"], dtype=np.float64)
        fam[period * n[boost > 0.0] - 1] = _FAMILY_CODE["aero + v1 motor boost"]
        hw = _shaft_hwhm_hz(k, float(p["sigma_nu"]), float(p["lam"]))
        zero = np.zeros(K)
        for r in range(R):
            add(r, k, fam, core[r] * np.exp(s2[r]), (s2[r], gam[r], zero, zero), hw)
    out = {c: np.concatenate(a) for c, a in cols.items()}
    out["f"] = out["k"] * float(f0)
    return out


def _pedestal_mean_hw(
    s2a: np.ndarray, ga: np.ndarray, s2p: np.ndarray, gp: np.ndarray
) -> np.ndarray:
    """Power-weighted mean half-width (Hz) of the skirt of a line with both
    pedestals: ``(s2a ga + s2p gp) / (1 - e^-(s2a + s2p))`` (0 without one)."""
    tot = s2a + s2p
    return np.where(tot > 1e-12, (s2a * ga + s2p * gp) / np.maximum(-np.expm1(-tot), 1e-300), 0.0)


def _profile_row(pay: dict[str, Any], f0: float, rotor: int | None, fmax: float) -> dict[str, Any]:
    """What one row of :func:`plot_rig_profiles` draws (dB = 10 log10 of
    power, densities per 1 Hz): the summed pedestal skirts on a grid refined
    around every line, and per (order, family) the needle, its shaft half
    width, the skirt's peak and mean half width, with the hover text."""
    from experiments.noise_model import prior_v4 as PV4

    L = profile_lines(pay, f0)
    keep = (L["f"] < fmax) & (L["power"] > 1e-25)
    if rotor is not None:
        keep &= L["rotor"] == rotor
    L = {c: a[keep] for c, a in L.items()}
    ped = (L["s2a"], L["ga"], L["s2p"], L["gp"])
    tot = L["s2a"] + L["s2p"]
    p_ped = L["power"] * -np.expm1(-tot)
    w_ped = _pedestal_mean_hw(*ped)
    width = np.maximum(np.maximum(w_ped, L["ga"]), 0.05)
    grid = np.unique(
        np.concatenate(
            [np.arange(1.0, fmax, 1.0), (L["f"][:, None] + width[:, None] * _REFINE).ravel()]
        )
    )
    grid = grid[(grid > 0.0) & (grid < fmax)]
    live = p_ped > 0.0
    # lines at one frequency with one pedestal (v4: every rotor's line k) share
    # a skirt shape: evaluate each shape once, with the summed power
    shapes, s_inv = np.unique(np.stack([L["f"], *ped], axis=1)[live], axis=0, return_inverse=True)
    s_pow = np.bincount(s_inv.ravel(), weights=p_ped[live], minlength=shapes.shape[0])
    skirt = np.zeros(grid.size)
    for a in range(0, grid.size, 2048):
        g = grid[a : a + 2048]
        dens = PV4.pedestal_skirt_density(
            *shapes[:, 1:].T, g[:, None] - shapes[:, 0][None, :], tol=1e-6
        )
        skirt[a : a + 2048] = dens @ s_pow
    peak = np.zeros(L["f"].size)
    peak[live] = p_ped[live] * PV4.pedestal_skirt_density(*(x[live] for x in ped), 0.0, tol=1e-6)
    # merge the rotors: every rotor's line k lands on the same frequency
    key = L["k"] * 10 + L["family"]
    uk, inv = np.unique(key, return_inverse=True)

    def total(x: np.ndarray) -> np.ndarray:
        return np.bincount(inv, weights=x, minlength=uk.size)

    def wmean(x: np.ndarray, w: np.ndarray) -> np.ndarray:
        return total(x * w) / np.maximum(total(w), 1e-300)

    power, needle = total(L["power"]), total(L["power"] * np.exp(-tot))
    m = dict(
        k=np.round(uk // 10).astype(int),
        family=np.round(uk % 10).astype(int),
        f=wmean(L["f"], np.ones_like(L["f"])),
        power=power,
        needle=needle,
        peak=total(peak),
        w_ped=wmean(w_ped, np.maximum(p_ped, 1e-300)),
        shaft_hw=wmean(L["shaft_hw"], L["power"]),
        n_rot=total(np.ones_like(L["f"])).astype(int),
        **{c: wmean(L[c], L["power"]) for c in ("s2a", "ga", "s2p", "gp")},
    )
    names = list(FAMILY_COLORS)
    m["hover"] = [
        f"k {m['k'][j]} · {names[m['family'][j]]} · {m['f'][j]:.1f} Hz"
        + ("" if rotor is not None else f" · {m['n_rot'][j]} rotors summed")
        + f"<br>line {10.0 * np.log10(m['power'][j]):.1f} dB, needle {100 * m['needle'][j] / m['power'][j]:.0f} %"
        + f"<br>needle half-width (shaft FM) ≈ {m['shaft_hw'][j]:.3f} Hz"
        + f"<br>amp pedestal σ² {m['s2a'][j]:.3f}, γ {m['ga'][j]:.2f} Hz"
        + f"<br>phase pedestal σ² {m['s2p'][j]:.3f}, γ {m['gp'][j]:.2f} Hz"
        + f"<br>pedestal mean half-width {m['w_ped'][j]:.2f} Hz"
        for j in range(uk.size)
    ]
    return dict(grid=grid, skirt=skirt, lines=m)


def plot_rig_profiles(
    pays: dict[str, dict[str, Any]],
    f0: float = 80.0,
    *,
    rotor: int | None = None,
    fmax: float = 7900.0,
    title: str = "",
) -> Any:
    """Plotly figure, one row per variant (shared, linked frequency axis),
    every rotor at ``f0`` rev/s (``rotor``: that rotor's lines only), in dB of
    power (densities per 1 Hz, so a needle reads its power in one 1 Hz bin):

    * stems: the NEEDLE of each line (power ``e^-(s2a+s2p)`` of the line),
      blue aerodynamic, orange v4 motor line, purple a v1 aerodynamic line
      carrying the motor boost; a short cap at the top spans the needle's
      shaft-FM half width; hover: the line's parameters;
    * red curve: the summed pedestal skirts (amplitude + phase) of all lines;
      red caps: each line's skirt peak, spanning its mean half width;
    * grey: the floor (the rig floor whatever ``rotor``). No wind: every
      variant draws it per clip (v1's ``wind_sc`` law; the v4 render option).

    The v1 row uses the converted payload's floor: the conversion keeps v1's
    floor law, equal at equal rotor speeds. Mic mean; v1's per-clip line
    scatter (a render option) is not drawn."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from data_processing.noise_model import v4 as V4

    conv = V4.V4Params.from_payload(pays[PED_VARIANTS[1]])
    s = conv.speed(np.full(conv.n_rotors, float(f0)))
    fig = make_subplots(
        rows=len(pays),
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.035,
        subplot_titles=list(pays),
    )
    rows = {v: _profile_row(pay, f0, rotor, fmax) for v, pay in pays.items()}
    in_legend: set[str] = set()
    for j, row in enumerate(rows.values(), start=1):
        grid, m = row["grid"], row["lines"]
        floor = 2.0 * V4.broadband_psd(conv, grid, s).mean(axis=0) / conv.sr
        floor_db = _db(floor)
        base = float(np.percentile(floor_db, 1)) - 25.0

        def tr(j: int = j, **kw: Any) -> None:
            # SVG, not Scattergl: renders where WebGL is unavailable; one
            # legend entry per name, toggling it in every row
            name = kw["name"]
            kw.setdefault("mode", "lines")
            fig.add_trace(
                go.Scatter(**kw, showlegend=name not in in_legend, legendgroup=name), row=j, col=1
            )
            in_legend.add(name)

        tr(x=grid, y=floor_db, name="floor", line=dict(color="grey", width=1.5), hoverinfo="skip")
        tr(
            x=grid,
            y=np.maximum(_db(row["skirt"]), base),
            name="pedestal skirts (sum)",
            line=dict(color="red", width=1),
            hoverinfo="skip",
        )
        live = m["peak"] > 0.0
        cx = np.stack([m["f"] - m["w_ped"], m["f"] + m["w_ped"], np.full(m["f"].size, np.nan)], 1)[
            live
        ].ravel()
        cy = np.repeat(_db(m["peak"][live]), 3)
        tr(
            x=cx,
            y=cy,
            name="pedestal peak ± mean half-width",
            line=dict(color="red", width=3),
            hoverinfo="skip",
        )
        for fam, color in FAMILY_COLORS.items():
            sel = m["family"] == _FAMILY_CODE[fam]
            if not sel.any():
                continue
            f, top = m["f"][sel], _db(m["needle"][sel])
            nan = np.full(f.size, np.nan)
            tr(
                x=np.stack([f, f, nan], 1).ravel(),
                y=np.stack([np.full(f.size, base), top, nan], 1).ravel(),
                name=f"needle: {fam}",
                line=dict(color=color, width=1.5),
                hoverinfo="skip",
            )
            hw = m["shaft_hw"][sel]
            tr(
                x=np.stack([f - hw, f + hw, nan], 1).ravel(),
                y=np.repeat(top, 3),
                name=f"needle ± shaft half-width: {fam}",
                line=dict(color=color, width=4),
                hoverinfo="skip",
            )
            tr(
                x=f,
                y=top,
                mode="markers",
                name=f"line: {fam}",
                marker=dict(color=color, size=5),
                hovertext=[h for h, ok in zip(m["hover"], sel, strict=True) if ok],
                hoverinfo="text",
            )
        fig.update_yaxes(
            title_text="dB", range=[base, float(_db(m["needle"]).max()) + 5.0], row=j, col=1
        )
    who = "all rotors (lines summed)" if rotor is None else f"rotor {rotor}"
    fig.update_xaxes(title_text="Hz", row=len(pays), col=1)
    fig.update_layout(
        height=330 * len(pays),
        title=f"{title} at {f0:g} rev/s, {who}: dB of power, densities per 1 Hz",
        hovermode="closest",
        legend=dict(orientation="h", y=-0.08),
        margin=dict(t=80, b=40, l=60, r=20),
    )
    return fig


def profile_viewer(pays: dict[str, dict[str, Any]], *, title: str = "") -> Any:
    """Speed slider and rotor dropdown → :func:`plot_rig_profiles` (zoom with
    the mouse; the three rows share the frequency axis)."""
    import ipywidgets as W
    from IPython.display import display

    w_f0 = W.IntSlider(
        value=80, min=30, max=120, step=5, description="rev/s", continuous_update=False
    )
    n_rot = int(pays[PED_VARIANTS[1]]["n_rotors"])
    w_rot = W.Dropdown(
        options=[("all rotors", -1)] + [(f"rotor {r}", r) for r in range(n_rot)],
        description="lines",
    )
    out = W.Output()

    def refresh(*_):
        with out:
            out.clear_output(wait=True)
            rot = None if w_rot.value < 0 else int(w_rot.value)
            display(plot_rig_profiles(pays, float(w_f0.value), rotor=rot, title=title))

    for w in (w_f0, w_rot):
        w.observe(refresh, "value")
    refresh()
    return W.VBox([W.HBox([w_f0, w_rot]), out])


#: One colour per variant, shared by every pedestal figure.
PED_COLORS = dict(zip(PED_VARIANTS, ("black", "tab:blue", "tab:red"), strict=True))


def _steadiest_second(rps: np.ndarray) -> int:
    """Start sample of the 1 s window (0.25 s steps) whose rotors move least
    (summed peak-to-peak speed)."""
    starts = np.arange(0, rps.shape[1] - SR + 1, SR // 4)
    spread = [float(np.ptp(rps[:, a : a + SR], axis=1).sum()) for a in starts]
    return int(starts[int(np.argmin(spread))])


def plot_pedestal_compare(
    audio: dict[str, np.ndarray],
    rps: np.ndarray,
    *,
    ch: int = 0,
    zoom_hz: tuple[float, float] = (2200.0, 2700.0),
) -> None:
    """Trajectory; per variant the full-band spectrogram and a fine one over
    ``zoom_hz`` (one dB scale per column); then the three spectra overlaid in
    three bands (mic-mean, 1 Hz bins): over the whole clip when the speeds
    are constant, else over its steadiest second."""
    from scipy.signal import spectrogram, welch

    steady = bool(np.all(np.ptp(rps, axis=1) < 1e-9))
    t = np.arange(rps.shape[1]) / SR
    fig, ax = plt.subplots(figsize=(18, 1.8))
    for r in range(rps.shape[0]):
        ax.plot(t, rps[r], lw=0.8, label=f"rotor {r}")
    a0 = 0 if steady else _steadiest_second(rps)
    if not steady:
        ax.axvspan(a0 / SR, a0 / SR + 1.0, color="grey", alpha=0.2, label="spectra window")
    ax.set(ylabel="rev/s", xlabel="s")
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    plt.show()

    n_zoom = 8192 if steady else 4096  # moving lines: shorter frames follow them
    specs = {}
    for v, a in audio.items():
        f1, t1, s1 = spectrogram(a[ch], fs=SR, nperseg=1024, noverlap=768)
        f2, t2, s2 = spectrogram(a[ch], fs=SR, nperseg=n_zoom, noverlap=n_zoom * 7 // 8)
        m = (f2 >= zoom_hz[0]) & (f2 <= zoom_hz[1])
        specs[v] = ((f1, t1, _db(s1)), (f2[m], t2, _db(s2[m])))
    top = max(float(np.max(s[0][2])) for s in specs.values())
    # zoom: one scale from the joint 99.5th percentile down 35 dB (a few hot
    # noise bins would otherwise set it and wash the lines out)
    top2 = float(np.percentile(np.concatenate([s[1][2].ravel() for s in specs.values()]), 99.5))
    fig, axs = plt.subplots(len(audio), 2, figsize=(18, 3.0 * len(audio)), sharex=True)
    for row, (v, ((f1, t1, d1), (f2, t2, d2))) in zip(axs, specs.items(), strict=True):
        row[0].pcolormesh(t1, f1, d1, vmin=top - 70, vmax=top, shading="auto", cmap="magma")
        row[1].pcolormesh(t2, f2, d2, vmin=top2 - 35, vmax=top2, shading="auto", cmap="magma")
        row[0].set(ylabel="Hz", title=f"{v}: ch {ch}, 1024-pt, 70 dB")
        row[1].set(
            ylabel="Hz",
            title=f"{v}: {zoom_hz[0]:.0f}-{zoom_hz[1]:.0f} Hz, {n_zoom}-pt "
            f"({SR / n_zoom:.0f} Hz bins), 35 dB",
        )
    for ax in axs[-1]:
        ax.set_xlabel("s")
    fig.tight_layout()
    plt.show()

    bands = ((250.0, 650.0), (2200.0, 2700.0), (5400.0, 5900.0))
    seg = slice(None) if steady else slice(a0, a0 + SR)
    fig, axs = plt.subplots(1, len(bands), figsize=(18, 3.2))
    for v, a in audio.items():
        f, p = welch(a[:, seg], fs=SR, nperseg=16384 if steady else SR, axis=-1)
        for ax, (lo, hi) in zip(axs, bands, strict=True):
            m = (f >= lo) & (f <= hi)
            ax.plot(f[m], _db(p.mean(axis=0)[m]), lw=0.9, color=PED_COLORS[v], label=v)
    span = "whole clip" if steady else f"{a0 / SR:.2f}-{a0 / SR + 1:.2f} s"
    for ax, (lo, hi) in zip(axs, bands, strict=True):
        ax.set(xlabel="Hz", ylabel="dB", title=f"{lo:.0f}-{hi:.0f} Hz, mic mean, 1 Hz bins, {span}")
        ax.grid(alpha=0.3)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    plt.show()


def pedestal_compare(i: int, traj: str | int = "steady", *, ch: int = 0, seed: int = 0):
    """Bank rig ``i`` as v1 / v4 converted / v4 + phase pedestal on one
    trajectory: pedestal parameters, spectrograms, line shapes and a player
    per variant (channel ``ch``, one shared scale)."""
    from IPython.display import Audio, display

    audio, rps, pays = pedestal_renders(i, traj, seed=seed)
    plot_pedestal_params(pays)
    display(profile_viewer(pays, title=f"rig {int(i)}"))
    plot_pedestal_compare(audio, rps, ch=ch)
    for v, a in audio.items():
        print(v)
        display(Audio(a[ch], rate=SR, normalize=False))


def pedestal_browser():
    """Dropdowns (bank rig, trajectory, channel) and a free rig index →
    :func:`pedestal_compare`."""
    import ipywidgets as W

    w_rig = W.Dropdown(options=list(PED_RIGS), description="rig")
    w_any = W.BoundedIntText(value=PED_RIGS[0], min=0, max=16383, description="or index")
    w_traj = W.Dropdown(options=["steady", 0, 1, 2], description="trajectory")
    w_ch = W.Dropdown(options=list(range(8)), description="channel")
    out = W.Output()

    def refresh(*_):
        with out:
            out.clear_output(wait=True)
            pedestal_compare(int(w_any.value), w_traj.value, ch=int(w_ch.value))

    w_rig.observe(lambda c: setattr(w_any, "value", c["new"]), "value")
    for w in (w_any, w_traj, w_ch):
        w.observe(refresh, "value")
    refresh()
    return W.VBox([W.HBox([w_rig, w_any, w_traj, w_ch]), out])


__all__ = [
    "Lab",
    "browser",
    "build",
    "compare",
    "compare_browser",
    "pedestal_browser",
    "pedestal_compare",
    "pedestal_payloads",
    "pedestal_renders",
    "plot_pedestal_compare",
    "plot_pedestal_params",
    "plot_rig",
    "plot_trajectory",
    "render_on_clip",
    "rig_browser",
    "summary",
    "valid_clip",
    "view",
]
