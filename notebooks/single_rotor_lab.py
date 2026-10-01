"""Single-rotor harmonic profile lab: DREGON single-motor recording vs its
reconstruction from ``experiments.static_rig.single_rotor`` (shaft rate,
phase diffusion, harmonic amplitudes — nothing else), on the same spectrum
pane of one ``plots.spectrum_viewer`` (two colours, spectrogram toggle), both
listenable, with the profile drawn as stems with caps. Driver notebook: ``single_rotor_explainer.ipynb``.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from experiments.static_rig import single_rotor as SR

DATASET = "DREGON-frames"
RECORDINGS = [f"motor_Motor{m}_{t}" for m in (1, 2, 3, 4) for t in (50, 60, 70, 80, 90)]
#: Wind-corrupted mics per motor (the ones in the downwash; Dmitrii's reading):
#: excluded from the span/speed/jitter estimates, still read and rebuilt.
WIND = {"Motor1": [5, 6], "Motor2": [0], "Motor3": [1, 2], "Motor4": [4]}
#: Measured once (docs/experiments/static-rig-profiles.md, DREGON singles):
#: shaft rate s̄ (rev/s; core peaks of orders 2-7) and the rms shaft-phase
#: drift from s̄·t after 1 s, r(1 s) (rev; BPF phase, median over good mics).
#: Phase diffusion D = r(1 s)². Taken as given here, never re-estimated.
TABLE = {
    #            s̄       r(1 s)
    "motor_Motor1_50": (49.011, 0.0273),
    "motor_Motor1_60": (58.771, 0.0379),
    "motor_Motor1_70": (68.463, 0.0332),
    "motor_Motor1_80": (78.234, 0.0339),
    "motor_Motor1_90": (87.979, 0.0266),
    "motor_Motor2_50": (48.440, 0.0175),
    "motor_Motor2_60": (58.192, 0.0155),
    "motor_Motor2_70": (67.679, 0.0320),
    "motor_Motor2_80": (77.307, 0.0361),
    "motor_Motor2_90": (86.930, 0.0167),
    "motor_Motor3_50": (49.150, 0.0218),
    "motor_Motor3_60": (59.048, 0.0162),
    "motor_Motor3_70": (68.723, 0.0379),
    "motor_Motor3_80": (78.652, 0.0300),
    "motor_Motor3_90": (88.457, 0.0198),
    "motor_Motor4_50": (49.777, 0.0252),
    "motor_Motor4_60": (59.810, 0.0214),
    "motor_Motor4_70": (69.515, 0.0176),
    "motor_Motor4_80": (79.611, 0.0151),
    "motor_Motor4_90": (89.405, 0.0155),
}
CACHE = Path(".cache/single_rotor")


def load(key: str) -> tuple[np.ndarray, int]:
    """``(audio (C, T) float32, fs)`` of one recording, cached as ``.npy``."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / f"{key}.npy"
    if not path.exists():
        from plots import explore

        frame = explore.load_recording(DATASET, key)
        audio = frame["audio"]
        np.save(path, np.ascontiguousarray(np.asarray(audio.data, dtype=np.float32)))
        (CACHE / f"{key}.fs").write_text(str(int(round(float(audio.tindex.sr)))))
    return np.load(path), int((CACHE / f"{key}.fs").read_text())


@dataclass
class Case:
    key: str
    fs: int
    real: np.ndarray  #: (C, n) the strict rotor-on span
    synth: np.ndarray  #: (C, n) the reconstruction
    model: SR.RotorModel
    wind: list[int]  #: windy mics (excluded from speed/jitter)


_cases: dict[str, Case] = {}


def case(key: str, seed: int = 0) -> Case:
    if key not in _cases:
        x, fs = load(key)
        wind = WIND[key.split("_")[1]]
        s_bar, r1 = TABLE[key]
        model = SR.analyse(
            x, fs, s_bar, r1**2, mics=[c for c in range(x.shape[0]) if c not in wind]
        )
        a, b = model.span
        real = np.asarray(x[:, a:b], dtype=np.float32)
        synth = SR.synthesise(model.fit, fs, b - a, seed=seed).astype(np.float32)
        _cases[key] = Case(key=key, fs=fs, real=real, synth=synth, model=model, wind=wind)
    return _cases[key]


def _wav_bytes(x: np.ndarray, fs: int, gain: float) -> bytes:
    from scipy.io import wavfile

    y = np.clip(x * gain, -1.0, 1.0)
    buf = io.BytesIO()
    wavfile.write(buf, fs, (y * 32767).astype(np.int16))
    return buf.getvalue()


def summary_html(c: Case) -> str:
    fit = c.model.fit
    a, b = c.model.span
    g = fit.gamma
    rows = [
        f"<b>{c.key}</b> — strict rotor-on span {a / c.fs:.1f}–{b / c.fs:.1f} s (T = {fit.T:.1f} s), "
        f"fs {c.fs} Hz, windy mics (not used for the span): {c.wind}",
        f"given: s̄ = <b>{fit.s:.3f} rev/s</b>, r(1 s) = {TABLE[c.key][1]:.4f} rev → D = {fit.D:.2e} rev²/s",
        f"line HWHM γ_k = π k² D: k=2 {g[1]:.3f} Hz, k=10 {g[9]:.2f}, k=20 {g[19]:.2f}, "
        f"k=40 {g[min(39, g.size - 1)]:.1f} Hz (resolution 1/T = {1 / fit.T:.3f} Hz); "
        f"orders read: 1–{fit.orders[-1]}",
    ]
    return "<div style='font-size:13px;line-height:1.5'>" + "<br>".join(rows) + "</div>"


class Lab:
    """Dropdown → one spectrum viewer with the recording and its
    reconstruction on the same spectrum pane (two colours; spectrogram
    toggle), audio for both, and the harmonic profile of the selected mic."""

    def __init__(self, width: int = 1100, height: int = 680) -> None:
        import ipywidgets as W
        import plotly.graph_objects as go

        from plots.spectrum_viewer import SpectrumViewer

        self.w_rec = W.Dropdown(options=RECORDINGS, value=RECORDINGS[2], description="recording:")
        self.w_info = W.HTML()
        c = case(self.w_rec.value)
        self.viewer = SpectrumViewer(
            c.real,
            c.fs,
            overlay=c.synth,
            labels=("recording", "reconstruction"),
            title=c.key,
            height=height,
            width=width,
            fmax=8000.0,
        )
        self.a_real, self.a_synth = W.Output(), W.Output()
        self.profile = go.FigureWidget(
            data=[
                go.Scatter(
                    x=[],
                    y=[],
                    mode="lines",
                    line={"color": "#1f77b4", "width": 1.5},
                    hoverinfo="skip",
                    name="stem",
                ),
                go.Scatter(
                    x=[],
                    y=[],
                    mode="markers",
                    marker={
                        "symbol": "line-ew-open",
                        "size": 9,
                        "color": "#1f77b4",
                        "line": {"width": 2},
                    },
                    name="harmonic",
                    hovertemplate="k %{customdata}<br>f %{x:.1f} Hz<br>%{y:.1f} dB<extra></extra>",
                ),
            ],
            layout=go.Layout(
                height=320,
                width=width,
                margin={"l": 60, "r": 20, "t": 40, "b": 40},
                xaxis={"title": "Hz"},
                yaxis={"title": "dB re 1"},
                showlegend=False,
            ),
        )
        self.viewer.w_channel.observe(lambda _c: self._redraw_profile_audio(), "value")
        self.viewer.w_frange.observe(lambda _c: self._sync_profile_axis(), "value")
        self.viewer.w_faxis.observe(lambda _c: self._sync_profile_axis(), "value")
        self.w_rec.observe(lambda _c: self._on_rec(), "value")
        self.widget = W.VBox(
            [
                self.w_rec,
                self.w_info,
                W.HBox(
                    [
                        W.VBox([W.HTML("<b>recording</b>"), self.a_real]),
                        W.VBox([W.HTML("<b>reconstruction</b>"), self.a_synth]),
                    ]
                ),
                self.viewer.widget,
                self.profile,
            ]
        )
        self._refresh()

    def _on_rec(self) -> None:
        c = case(self.w_rec.value)
        v = self.viewer
        v.audio, v.overlay, v.sr = c.real, c.synth, float(c.fs)
        v._cache.clear()
        dur = c.real.shape[1] / v.sr
        v.w_time.max = dur
        v.w_time.value = dur / 2
        v.figure.layout.title.text = c.key
        v._redraw_all()
        self._refresh()

    def _refresh(self) -> None:
        c = case(self.w_rec.value)
        self.w_info.value = summary_html(c)
        self._redraw_profile_audio()

    def _sync_profile_axis(self) -> None:
        lo, hi = self.viewer._band()
        log = self.viewer.w_faxis.value == "log"
        self.profile.layout.xaxis.update(
            type="log" if log else "linear",
            range=[np.log10(max(lo, 1.0)), np.log10(hi)] if log else [lo, hi],
        )

    def _redraw_profile_audio(self) -> None:
        from IPython.display import Audio, display

        c = case(self.w_rec.value)
        ch = int(self.viewer.w_channel.value)
        gain = 0.9 / float(np.max(np.abs(c.real[ch])) + 1e-9)
        for out, x in ((self.a_real, c.real[ch]), (self.a_synth, c.synth[ch])):
            out.clear_output(wait=True)
            with out:
                display(Audio(data=_wav_bytes(x, c.fs, gain), rate=c.fs, autoplay=False))
        fit = c.model.fit
        f = fit.orders * fit.s
        db = 10 * np.log10(np.maximum(fit.amp2[ch], 1e-20))
        base = float(db.min()) - 6.0
        xs: list[float | None] = []
        ys: list[float | None] = []
        for fk, v in zip(f, db, strict=True):
            xs += [float(fk), float(fk), None]
            ys += [base, float(v), None]
        with self.profile.batch_update():
            self.profile.data[0].x, self.profile.data[0].y = xs, ys
            self.profile.data[1].x, self.profile.data[1].y = f, db
            self.profile.data[1].customdata = fit.orders
            self.profile.layout.title.update(
                text=f"harmonic profile — {c.key} ch {ch} — amplitude dB re 1, stems with caps, no floor",
                font={"size": 13},
            )
            self.profile.layout.yaxis.range = [base, float(db.max()) + 3]
        self._sync_profile_axis()

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.widget)


def lab(**kw) -> Lab:
    return Lab(**kw)


def precompute(keys: list[str] | None = None) -> None:
    """Warm the audio cache and the analyses for every recording."""
    for k in keys or RECORDINGS:
        c = case(k)
        m = c.model
        print(
            f"{k}: s {m.fit.s:.3f} D {m.fit.D:.2e} span {m.span[0] / c.fs:.1f}-{m.span[1] / c.fs:.1f}"
        )


__all__ = ["RECORDINGS", "Case", "Lab", "case", "lab", "load", "precompute", "summary_html"]
