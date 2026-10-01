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


__all__ = [
    "Lab",
    "browser",
    "build",
    "plot_rig",
    "plot_trajectory",
    "rig_browser",
    "summary",
    "view",
]
