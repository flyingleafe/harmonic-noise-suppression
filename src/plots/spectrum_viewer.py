"""Interactive Plotly spectrogram + spectrum viewer for one audio recording.

Plotly port of the ``Compare`` viewer of the deleted
``notebooks/drone_embedding_explorer.ipynb`` (commit ``719eb38b``): a
spectrogram on top, a spectrum under it, and one control row. Entry point::

    from plots.spectrum_viewer import spectrum_viewer
    spectrum_viewer(frame)                 # a td.Frame (its audio entry)
    spectrum_viewer(series)                # a td.Series with a GridIndex
    spectrum_viewer(array, sr=16000)       # (T,) or (C, T)

Controls:

- **spectrum**: ``overall`` = one FFT of the whole (cropped) channel;
  ``mean`` = the RMS of the STFT columns (the mean power spectrum, shown as an
  amplitude); ``frame`` = one STFT column. Clicking the spectrogram pins the
  frame at the clicked time and switches to ``frame``; the time slider moves
  it.
- **frequency axis** ``linear`` / ``log`` — both panels (the spectrogram's y,
  the spectrum's x).
- **amplitude** ``dB`` / ``linear`` — both panels: the spectrum's y and the
  spectrogram's colour. In dB the colour range is ``dyn range`` below the
  99.5th percentile; in linear it is 0 to the 99.5th percentile.
- **frequency range** — the band shown on both panels. The overall FFT is
  re-decimated for the band, so zooming in reveals more bins.

Amplitude convention: every spectrum is an AMPLITUDE spectrum normalised so
that a sinusoid of amplitude ``A`` centred on a bin reads ``A`` (0 dB = unit
amplitude) — Hann window, ``|X| * 2 / sum(w)``. Tonal peaks therefore read the
same in all three modes; a broadband floor does not (its level scales with the
window's noise bandwidth, i.e. with the analysis length), which is the
physics, not a bug.

Display decimation (exact data, reduced drawing): spectrogram columns are
power-averaged down to ``max_cols``; spectrum curves keep the MAXIMUM of each
of ``max_points`` frequency buckets (log-spaced buckets on a log axis), so no
narrow line disappears.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import tdseries as td

__all__ = [
    "SPECTRUM_MODES",
    "SpectrumViewer",
    "Stft",
    "frame_spectrum",
    "mean_spectrum",
    "overall_spectrum",
    "peak_decimate",
    "spectrum_viewer",
    "stft_of",
]

#: Spectrum modes, in control order.
SPECTRUM_MODES = ("overall", "mean", "frame")

_TINY = 1e-12
#: Audio entry names probed (in order) on a Frame.
_AUDIO_ENTRIES = ("audio", "mixture", "target", "enhanced", "generated")


# ─── numerics ─────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Stft:
    """Amplitude-normalised STFT power of one channel."""

    freqs: np.ndarray  # (F,) Hz
    times: np.ndarray  # (N,) frame centres, s (the signal's own time)
    power: np.ndarray  # (F, N) float32, (|X| * 2 / sum(w))**2


def stft_of(
    x: np.ndarray, sr: float, *, t_start: float = 0.0, n_fft: int = 2048, hop: int = 512
) -> Stft:
    """One channel through the project's single spectrogram implementation,
    :func:`plots.timeframe.renderers.make_spectrogram_series` (Hann, centred
    frames), rescaled by ``2 / sum(w)`` to the amplitude convention above."""
    from fractions import Fraction

    from plots.timeframe.renderers import make_spectrogram_series

    rate = int(sr) if float(sr).is_integer() else Fraction(sr).limit_denominator(1000)
    audio = td.uniform(np.asarray(x, dtype=np.float32), rate, dims=("time",), t_start=t_start)
    series = make_spectrogram_series(audio, n_fft=n_fft, hop_length=hop, log=False).series
    mag = np.asarray(series.data, dtype=np.float64) * (2.0 / (n_fft / 2.0))  # sum(hann) = n_fft/2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / float(sr))[: mag.shape[0]]
    tindex = series.tindex
    assert isinstance(tindex, td.GridIndex)  # td.uniform always builds a GridIndex
    times = np.asarray(tindex.sample_times(), dtype=np.float64)
    return Stft(freqs=freqs, times=times, power=(mag**2).astype(np.float32))


def overall_spectrum(x: np.ndarray, sr: float) -> tuple[np.ndarray, np.ndarray]:
    """``(freqs, amplitude)``: one Hann-window FFT over the whole signal."""
    x = np.asarray(x, dtype=np.float64)
    win = np.hanning(x.size + 1)[:-1]
    amp = np.abs(np.fft.rfft(x * win)) * (2.0 / max(win.sum(), _TINY))
    return np.fft.rfftfreq(x.size, 1.0 / sr), amp


def mean_spectrum(s: Stft) -> np.ndarray:
    """RMS amplitude over the STFT columns (the mean power spectrum)."""
    return np.sqrt(s.power.mean(axis=1, dtype=np.float64))


def frame_spectrum(s: Stft, t: float) -> np.ndarray:
    """Amplitude of the STFT column whose centre is nearest ``t`` (signal time)."""
    i = int(np.argmin(np.abs(s.times - float(t))))
    return np.sqrt(s.power[:, i].astype(np.float64))


def peak_decimate(
    freqs: np.ndarray,
    values: np.ndarray,
    fmin: float,
    fmax: float,
    max_points: int,
    *,
    log: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """``values`` inside ``[fmin, fmax]`` reduced to at most ``max_points``.

    Each bucket keeps its maximum (and that maximum's frequency), so a line
    narrower than a bucket survives. ``log=True`` spaces the buckets
    geometrically, for a log frequency axis (``fmin`` must then be > 0).
    """
    m = (freqs >= fmin) & (freqs <= fmax)
    f, v = freqs[m], values[m]
    if f.size <= max_points:
        return f, v
    edges = (
        np.geomspace(max(fmin, f[f > 0][0]), fmax, max_points + 1)
        if log
        else np.linspace(fmin, fmax, max_points + 1)
    )
    idx = np.clip(np.searchsorted(edges, f, side="right") - 1, 0, max_points - 1)
    # argmax per bucket: sort by (bucket, value) and take each bucket's last.
    order = np.lexsort((v, idx))
    last = np.r_[np.nonzero(np.diff(idx[order]))[0], order.size - 1]
    keep = np.sort(order[last])
    return f[keep], v[keep]


def _pool_columns(power: np.ndarray, times: np.ndarray, max_cols: int):
    """Power-average adjacent STFT columns down to at most ``max_cols``
    (the last group may be shorter; no column is dropped)."""
    n = power.shape[1]
    if n <= max_cols:
        return power, times
    k = int(np.ceil(n / max_cols))
    starts = np.arange(0, n, k)
    counts = np.diff(np.r_[starts, n])
    pooled = np.add.reduceat(power, starts, axis=1) / counts
    return pooled.astype(np.float32), np.add.reduceat(times, starts) / counts


def _db(amp: np.ndarray) -> np.ndarray:
    return 20.0 * np.log10(np.maximum(amp, _TINY))


# ─── input normalisation ─────────────────────────────────────────────────────


def _audio_entry(frame: td.Frame, entry: str | None) -> tuple[str, td.Series]:
    entries = dict(frame.items())
    if entry is not None:
        return entry, entries[entry]
    for name in _AUDIO_ENTRIES:
        if isinstance(entries.get(name), td.Series):
            return name, entries[name]
    from data_processing.canonical import _audio_candidates

    candidates = _audio_candidates(entries)
    if len(candidates) != 1:
        raise ValueError(
            f"cannot tell which entry is the audio (candidates {candidates}); pass entry="
        )
    return candidates[0], entries[candidates[0]]


def _as_channels(source: Any, entry: str | None, sr: float | None):
    """``(audio (C, T) float32, sr, t_start, label)`` from any accepted form."""
    label = ""
    if isinstance(source, td.Frame):
        label, source = _audio_entry(source, entry)
    if isinstance(source, td.Series):
        tindex = source.tindex
        if not isinstance(tindex, td.GridIndex):
            raise TypeError(f"audio needs a GridIndex time axis, got {type(tindex).__name__}")
        data = np.asarray(source.data, dtype=np.float32)
        t_axis = list(source.dims).index("time")
        data = np.moveaxis(data, t_axis, -1).reshape(-1, data.shape[t_axis])
        return data, float(tindex.sr), float(tindex.t_start), label
    if sr is None:
        raise ValueError("a raw array needs sr=")
    data = np.asarray(source, dtype=np.float32)
    if data.ndim == 1:
        data = data[None]
    if data.ndim != 2:
        raise ValueError(f"expected (T,) or (C, T), got shape {data.shape}")
    return data, float(sr), 0.0, label


# ─── the widget ──────────────────────────────────────────────────────────────


class SpectrumViewer:
    """Spectrogram + spectrum ``plotly.graph_objects.FigureWidget`` with controls.

    Build through :func:`spectrum_viewer`. ``.widget`` is the ipywidgets box
    (displayed automatically in a notebook), ``.figure`` the FigureWidget.
    Programmatic control mirrors the widgets: :meth:`set`.
    """

    def __init__(
        self,
        audio: np.ndarray,
        sr: float,
        *,
        t_start: float = 0.0,
        title: str = "",
        channel: int = 0,
        n_fft: int = 2048,
        hop: int | None = None,
        fmin: float = 0.0,
        fmax: float | None = None,
        dyn_range_db: float = 80.0,
        max_cols: int = 1500,
        max_points: int = 6000,
        height: int = 720,
        width: int = 1100,
    ) -> None:
        import ipywidgets as W
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots

        self.audio, self.sr, self.t_start = audio, float(sr), float(t_start)
        self.n_fft, self.hop = int(n_fft), int(hop or n_fft // 4)
        self.max_cols, self.max_points = int(max_cols), int(max_points)
        self.nyquist = self.sr / 2.0
        self._cache: dict[int, tuple[Stft, tuple[np.ndarray, np.ndarray]]] = {}

        n_ch = audio.shape[0]
        self.w_channel = W.Dropdown(
            options=[(f"ch {c}", c) for c in range(n_ch)],
            value=int(np.clip(channel, 0, n_ch - 1)),
            description="channel:",
            layout=W.Layout(width="160px", display="" if n_ch > 1 else "none"),
        )
        self.w_mode = W.ToggleButtons(
            options=[("overall (FFT)", "overall"), ("mean (STFT)", "mean"), ("frame", "frame")],
            value="overall",
            description="spectrum:",
            style={"button_width": "112px"},
        )
        self.w_faxis = W.ToggleButtons(
            options=["linear", "log"],
            value="linear",
            description="freq axis:",
            style={"button_width": "64px"},
        )
        self.w_amp = W.ToggleButtons(
            options=["dB", "linear"],
            value="dB",
            description="amplitude:",
            style={"button_width": "64px"},
        )
        hi = self.nyquist if fmax is None else float(min(fmax, self.nyquist))
        self.w_frange = W.FloatRangeSlider(
            value=(float(max(fmin, 0.0)), hi),
            min=0.0,
            max=self.nyquist,
            step=max(self.sr / self.n_fft, 1.0),
            description="freq (Hz):",
            continuous_update=False,
            readout_format=".0f",
            layout=W.Layout(width="520px"),
        )
        self.w_dyn = W.IntSlider(
            value=int(dyn_range_db),
            min=20,
            max=140,
            step=5,
            description="dyn range:",
            continuous_update=False,
            layout=W.Layout(width="300px"),
        )
        dur = audio.shape[1] / self.sr
        self.w_time = W.FloatSlider(
            value=self.t_start + dur / 2,
            min=self.t_start,
            max=self.t_start + dur,
            step=self.hop / self.sr,
            description="frame t (s):",
            continuous_update=False,
            readout_format=".3f",
            layout=W.Layout(width="520px"),
        )

        fig = make_subplots(
            rows=2,
            cols=1,
            row_heights=[0.58, 0.42],
            vertical_spacing=0.1,
            subplot_titles=("spectrogram", "spectrum"),
        )
        fig.add_trace(
            go.Heatmap(
                z=[[0.0]],
                x=[0.0],
                y=[0.0],
                colorscale="Magma",
                zsmooth=False,
                colorbar={"title": "dB", "len": 0.55, "y": 0.73},
                hovertemplate="t %{x:.3f} s<br>f %{y:.1f} Hz<br>%{z:.1f} dB<extra></extra>",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scattergl(
                x=[0.0],
                y=[0.0],
                mode="lines",
                line={"width": 1.1},
                hovertemplate="f %{x:.2f} Hz<br>%{y:.3g}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=1,
        )
        fig.update_layout(
            height=height,
            width=width,
            margin={"l": 60, "r": 20, "t": 60, "b": 40},
            title={"text": title, "x": 0.01, "font": {"size": 13}},
            shapes=[
                {
                    "type": "line",
                    "xref": "x",
                    "yref": "y domain",
                    "x0": 0,
                    "x1": 0,
                    "y0": 0,
                    "y1": 1,
                    "line": {"color": "cyan", "width": 1.5},
                    "visible": False,
                }
            ],
        )
        fig.update_xaxes(title_text="time (s)", row=1, col=1)
        fig.update_yaxes(title_text="Hz", row=1, col=1)
        fig.update_xaxes(title_text="Hz", row=2, col=1)
        self.figure = go.FigureWidget(fig)
        self.figure.data[0].on_click(self._on_click)

        for w in (self.w_channel, self.w_dyn):
            w.observe(lambda _c: self._redraw_all(), "value")
        self.w_mode.observe(lambda _c: self._redraw_spectrum(), "value")
        self.w_amp.observe(lambda _c: self._redraw_all(), "value")
        self.w_faxis.observe(lambda _c: self._redraw_all(), "value")
        self.w_frange.observe(lambda _c: self._redraw_all(), "value")
        self.w_time.observe(lambda _c: self._on_time(), "value")

        self.widget = W.VBox(
            [
                W.HBox([self.w_mode, self.w_channel]),
                W.HBox([self.w_faxis, self.w_amp, self.w_dyn]),
                W.HBox([self.w_frange, self.w_time]),
                self.figure,
            ]
        )
        self._redraw_all()

    # ── state ────────────────────────────────────────────────────────────────

    def _channel_data(self) -> tuple[Stft, tuple[np.ndarray, np.ndarray]]:
        c = int(self.w_channel.value)
        if c not in self._cache:
            x = self.audio[c]
            s = stft_of(x, self.sr, t_start=self.t_start, n_fft=self.n_fft, hop=self.hop)
            self._cache[c] = (s, overall_spectrum(x, self.sr))
        return self._cache[c]

    def _band(self) -> tuple[float, float]:
        lo, hi = (float(v) for v in self.w_frange.value)
        if self.w_faxis.value == "log":
            lo = max(lo, self.sr / self.n_fft)  # first non-DC STFT bin
        return lo, max(hi, lo * 1.001)

    def spectrum(self) -> tuple[np.ndarray, np.ndarray]:
        """The CURRENT spectrum curve as drawn: ``(freqs, values)`` in the
        chosen amplitude unit, after band selection and peak decimation."""
        s, (f_all, a_all) = self._channel_data()
        mode = self.w_mode.value
        if mode == "overall":
            f, a = f_all, a_all
        elif mode == "mean":
            f, a = s.freqs, mean_spectrum(s)
        else:
            f, a = s.freqs, frame_spectrum(s, float(self.w_time.value))
        lo, hi = self._band()
        f, a = peak_decimate(f, a, lo, hi, self.max_points, log=self.w_faxis.value == "log")
        return f, (_db(a) if self.w_amp.value == "dB" else a)

    # ── drawing ──────────────────────────────────────────────────────────────

    def _redraw_all(self) -> None:
        s, _ = self._channel_data()
        lo, hi = self._band()
        rows = (s.freqs >= lo) & (s.freqs <= hi)
        power, times = _pool_columns(s.power[rows], s.times, self.max_cols)
        db = self.w_amp.value == "dB"
        if db:
            z = (10.0 * np.log10(np.maximum(power, _TINY))).astype(np.float32)
            top = float(np.percentile(z, 99.5)) if z.size else 0.0
            zlim, unit = (top - float(self.w_dyn.value), top), "dB re 1"
        else:
            z = np.sqrt(power).astype(np.float32)
            zlim, unit = (0.0, float(np.percentile(z, 99.5)) if z.size else 1.0), "amplitude"
        log = self.w_faxis.value == "log"
        with self.figure.batch_update():
            hm = self.figure.data[0]
            hm.z, hm.x, hm.y = z, times, s.freqs[rows]
            hm.zmin, hm.zmax = zlim
            hm.colorbar.title.text = unit
            hm.hovertemplate = (
                f"t %{{x:.3f}} s<br>f %{{y:.1f}} Hz<br>%{{z:.3g}} {unit}<extra></extra>"
            )
            self.w_dyn.disabled = not db
            ytype = "log" if log else "linear"
            yrange = [np.log10(lo), np.log10(hi)] if log else [lo, hi]
            self.figure.layout.yaxis.update(type=ytype, range=yrange)
            self.figure.layout.xaxis2.update(type=ytype, range=yrange)
            self._redraw_spectrum(inside_batch=True)

    def _redraw_spectrum(self, inside_batch: bool = False) -> None:
        if not inside_batch:
            with self.figure.batch_update():
                self._redraw_spectrum(inside_batch=True)
            return
        f, v = self.spectrum()
        mode, db = self.w_mode.value, self.w_amp.value == "dB"
        line = self.figure.data[1]
        line.x, line.y = f, v
        tag = {"overall": "overall FFT", "mean": "mean over STFT frames"}.get(
            mode, f"frame @ {float(self.w_time.value):.3f} s"
        )
        ch = f"ch {int(self.w_channel.value)}"
        self.figure.layout.annotations[1].text = f"spectrum — {tag} — {ch}"
        self.figure.layout.yaxis2.update(title_text="dB re 1" if db else "amplitude")
        cursor = self.figure.layout.shapes[0]
        cursor.update(
            x0=float(self.w_time.value), x1=float(self.w_time.value), visible=mode == "frame"
        )

    def _on_time(self) -> None:
        if self.w_mode.value != "frame":
            self.w_mode.value = "frame"  # observer redraws
        else:
            self._redraw_spectrum()

    def _on_click(self, _trace: Any, points: Any, _selector: Any) -> None:
        if points.xs:
            self.w_time.value = float(np.clip(points.xs[0], self.w_time.min, self.w_time.max))

    # ── public ───────────────────────────────────────────────────────────────

    def set(
        self,
        *,
        mode: str | None = None,
        freq_axis: str | None = None,
        amplitude: str | None = None,
        frange: tuple[float, float] | None = None,
        t: float | None = None,
        channel: int | None = None,
        dyn_range_db: float | None = None,
    ) -> SpectrumViewer:
        """Drive the controls from code (each change redraws like a click)."""
        if mode is not None:
            if mode not in SPECTRUM_MODES:
                raise ValueError(f"mode must be one of {SPECTRUM_MODES}, got {mode!r}")
            self.w_mode.value = mode
        if freq_axis is not None:
            self.w_faxis.value = freq_axis
        if amplitude is not None:
            self.w_amp.value = amplitude
        if frange is not None:
            self.w_frange.value = (float(frange[0]), float(frange[1]))
        if channel is not None:
            self.w_channel.value = int(channel)
        if dyn_range_db is not None:
            self.w_dyn.value = int(dyn_range_db)
        if t is not None:  # like a click: pin the frame AND show it
            self.w_time.value = float(t)
            self.w_mode.value = "frame"
        return self

    def _ipython_display_(self) -> None:
        from IPython.display import display

        display(self.widget)


def spectrum_viewer(
    source: td.Frame | td.Series | np.ndarray,
    *,
    entry: str | None = None,
    sr: float | None = None,
    start_s: float | None = None,
    duration_s: float | None = None,
    **kw: Any,
) -> SpectrumViewer:
    """A :class:`SpectrumViewer` over ``source`` (module docstring).

    ``source`` is a ``td.Frame`` (its ``audio``/``mixture``/… entry, or
    ``entry=``), a ``td.Series`` on a ``GridIndex``, or a ``(T,)``/``(C, T)``
    array with ``sr=``. Time on every axis is SECONDS FROM THE START OF THE
    SOURCE (a recording's absolute start, e.g. a DREGON Unix timestamp, goes
    into the title); ``start_s``/``duration_s`` crop on that same clock.
    ``kw`` go to :class:`SpectrumViewer` (``channel``, ``n_fft``, ``hop``,
    ``fmin``, ``fmax``, ``dyn_range_db``, ``max_cols``, ``max_points``,
    ``height``, ``width``, ``title``).
    """
    title = kw.pop("title", None)
    frame_title = ""
    if isinstance(source, td.Frame):
        from data_processing.frames import get_meta

        rid = get_meta(source, "recording_id")
        frame_title = str(rid) if rid is not None else ""
    audio, rate, origin, label = _as_channels(source, entry, sr)
    t0 = 0.0
    if start_s is not None or duration_s is not None:
        a = 0 if start_s is None else int(np.clip(round(float(start_s) * rate), 0, audio.shape[1]))
        b = audio.shape[1] if duration_s is None else a + int(round(float(duration_s) * rate))
        audio, t0 = audio[:, a : min(b, audio.shape[1])], a / rate
    if audio.shape[1] == 0:
        raise ValueError("the selected span holds no samples")
    if title is None:
        parts = [p for p in (frame_title, label) if p]
        parts.append(f"{audio.shape[0]} ch · {rate:g} Hz · {audio.shape[1] / rate:.2f} s")
        if origin:
            parts.append(f"t = 0 at {origin:.3f} s")
        title = " · ".join(parts)
    return SpectrumViewer(audio, rate, t_start=t0, title=title, **kw)
