"""Print-sized spectral and observable-control figures for the generator paper."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy.signal import welch

from plots.paper import TEXT_IN, paper_style

LABELS = {
    "reference": "Recorded reference",
    "stochastic": "Proposed stochastic",
    "stochastic_coherent": "Coherent-line ablation",
    "yang2025": "Yang, integer periods",
    "yang2025_phase": "Yang, phase carry",
    "lee2026_gp": "Coefficient GP, white",
    "lee2026_gp_colored": "Coefficient GP, coloured",
    "doerfler2026_eone": "EONE-RPS curve bank",
    # Same families, fitted separately on the standby and cruise supports of our
    # accepted fits and blended with our regime power crossfade.
    "yang2025_regimes": "Yang, integer, 2-expert",
    "yang2025_phase_regimes": "Yang, phase carry, 2-expert",
    "lee2026_gp_regimes": "GP, white, 2-expert",
    "lee2026_gp_colored_regimes": "GP, coloured, 2-expert",
    "doerfler2026_eone_regimes": "EONE-RPS, 2-expert",
    "doerfler2026_ptr_regimes": "PTR, 2-expert",
    "doerfler2026_hpn_regimes": "HPN, 2-expert",
    "renault2023_piano_regimes": "DDSP-Piano, 2-expert",
}
#: Historical methods may still be present in the evidence, not in current comparisons.
ARCHIVED_METHODS = frozenset({"heutschi2020", "heutschi2020_regimes"})
#: Published neural source stages tested only as two-expert fits.
MATCHED_ONLY_METHODS = (
    "doerfler2026_ptr_regimes",
    "doerfler2026_hpn_regimes",
    "renault2023_piano_regimes",
)
#: Suffix of the standby/cruise-matched external arms.
REGIME_SUFFIX = "_regimes"
#: Line style of every two-expert external arm in overlay plots.
MATCHED_STYLE = "-."
COLOURS = {
    "reference": "black",
    "stochastic": "C0",
    "stochastic_coherent": "C3",
    "yang2025": "C4",
    "yang2025_phase": "C2",
    "lee2026_gp": "C5",
    "lee2026_gp_colored": "C4",
    "doerfler2026_eone": "C6",
    "yang2025_regimes": "C4",
    "yang2025_phase_regimes": "C2",
    "lee2026_gp_regimes": "C5",
    "lee2026_gp_colored_regimes": "C4",
    "doerfler2026_eone_regimes": "C6",
    "doerfler2026_ptr_regimes": "C1",
    "doerfler2026_hpn_regimes": "C2",
    "renault2023_piano_regimes": "C7",
}


def schedule_example(example: Mapping[str, np.ndarray], rate_hz: float) -> Figure:
    """Held-out event clock: observed collective speed/spread and ensemble intervals."""
    with paper_style():
        fig, axs = plt.subplots(
            2,
            4,
            figsize=(TEXT_IN, 2.75),
            sharex="col",
            gridspec_kw={"width_ratios": [1.2, 1, 1, 1]},
        )
        panels = (
            ("startup_standby", "Start-up and standby", "From motors-on (s)"),
            ("takeoff", "Takeoff", "From takeoff (s)"),
            ("cruise", "Cruise", "In cruise window (s)"),
            ("landing", "Landing", "From landing (s)"),
        )
        for col, (phase, label, xlabel) in enumerate(panels):
            truth = example[f"{phase}__real"]
            t = (np.arange(truth.shape[-1]) + float(example[f"{phase}__lo"])) / rate_hz
            for row, reduce in enumerate((np.mean, np.ptp)):
                ax = axs[row, col]
                for arm, name, color, style in (
                    ("ours", "Proposed", "C0", "-"),
                    ("phasevar4", "Phase VAR(4)", "C1", "--"),
                ):
                    values = reduce(example[f"{phase}__{arm}"], axis=1)
                    low, med, high = np.quantile(values, (0.1, 0.5, 0.9), axis=0)
                    ax.fill_between(t, low, high, color=color, alpha=0.17, linewidth=0)
                    ax.plot(t, med, color=color, ls=style, label=name)
                ax.plot(t, reduce(truth, axis=0), color="black", label="Measured", lw=0.9)
                ax.set_xlim(t[0], t[-1])
                if phase != "cruise" or row == 1:
                    ax.set_ylim(bottom=0)
                ax.grid(axis="y")
                ax.spines[["right", "top"]].set_visible(False)
            axs[0, col].set_title(label)
            axs[1, col].set_xlabel(xlabel)
        axs[0, 0].set_ylabel("Mean (rev/s)")
        axs[1, 0].set_ylabel("Range (rev/s)")
        fig.legend(*axs[0, 0].get_legend_handles_labels(), loc="upper center", ncol=3)
        fig.subplots_adjust(left=0.075, right=0.99, bottom=0.16, top=0.80, wspace=0.34, hspace=0.14)
        return fig


def treatment(method: str) -> str:
    """``"ours"``, ``"matched"`` (two-expert external) or ``"pooled"`` (native external fit)."""
    if method.startswith("stochastic"):
        return "ours"
    return "matched" if method.endswith(REGIME_SUFFIX) else "pooled"


def linestyle(method: str) -> str:
    """Dashed ablation, dash-dot two-expert external arms, solid otherwise."""
    if method == "stochastic_coherent":
        return "--"
    return MATCHED_STYLE if treatment(method) == "matched" else "-"


_PRINT_RC = {
    "font.size": 9.0,
    "axes.labelsize": 9.0,
    "axes.titlesize": 9.0,
    "xtick.labelsize": 8.0,
    "ytick.labelsize": 8.0,
    "legend.fontsize": 8.0,
    "lines.linewidth": 1.0,
}


def _welch_db(wave: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Microphone-mean Welch PSD in dB, 125–800 Hz; no rescaling."""
    f, p = welch(
        wave,
        fs=sr,
        window="hann",
        nperseg=16384,
        noverlap=12288,
        detrend=False,  # pyright: ignore[reportArgumentType]
        scaling="density",
        axis=-1,
    )
    keep = (f >= 125) & (f <= 800)
    return f[keep], 10 * np.log10(np.maximum(p.mean(axis=0), 1e-24))[keep]


def source_spectra(
    examples: Mapping[str, Mapping[str, np.ndarray]],
    *,
    sr: int = 16000,
    methods: Sequence[str],
) -> Figure:
    """One predeclared draw/window per regime; never rescale individual methods.

    One panel per regime overlays the recorded reference and the selected methods.
    """
    bad = [m for m in methods if m not in LABELS or m == "reference"]
    if bad or not methods:
        raise ValueError(f"methods must be non-empty labelled generated arms; bad: {bad}")
    with paper_style(), mpl.rc_context(_PRINT_RC):
        fig, axes = plt.subplots(
            1, len(examples), figsize=(6.5, 2.9), squeeze=False, layout="constrained"
        )
        for ax, (regime, waves) in zip(axes[0], examples.items(), strict=True):
            shown = []
            for name in ("reference", *methods):
                f, level = _welch_db(waves[name], sr)
                shown.append(level)
                ax.plot(
                    f,
                    level,
                    color=COLOURS[name],
                    linestyle=linestyle(name),
                    linewidth=1.1 if name == "reference" else 0.9,
                    label=LABELS[name],
                    alpha=0.95,
                )
            values = np.concatenate(shown)
            ax.set_ylim(float(np.percentile(values, 1)) - 3, float(values.max()) + 3)
            ax.set(xlabel="Frequency (Hz)", title=regime, xlim=(125, 800))
            ax.grid(alpha=0.25)
        axes[0, 0].set_ylabel("PSD (dB/Hz, digital units)")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="outside upper center", ncol=3)
        return fig


def control_spectral_map(
    frequencies: np.ndarray,
    psd: np.ndarray,
    methods: list[str],
    rotor_rps: np.ndarray,
    *,
    show: Sequence[str] | None = None,
) -> Figure:
    """Actual emitted peaks, with one shared colour scale across all methods.

    The colour limits always come from every method in ``psd``; ``show`` draws
    a subset of panels, so several figures cut from one probe share one scale.
    """
    f = np.asarray(frequencies)
    keep = (f >= 130) & (f <= 202)
    # Stored shape: method, condition, draw, frequency; average only the draws.
    levels = 10 * np.log10(np.maximum(np.asarray(psd).mean(axis=2)[..., keep], 1e-24))
    vmax = float(levels.max())
    vmin = max(float(np.percentile(levels, 5)), vmax - 45)
    shown = list(methods if show is None else show)
    missing = [m for m in shown if m not in methods]
    if missing or not shown:
        raise ValueError(
            f"show must be a non-empty subset of the probe methods; missing: {missing}"
        )
    with paper_style(), mpl.rc_context(_PRINT_RC):
        # From seven methods on, three columns keep the six-method figure height.
        ncols = min(2 if len(shown) <= 6 else 3, len(shown))
        nrows = (len(shown) + ncols - 1) // ncols
        fig, axes = plt.subplots(
            nrows,
            ncols,
            figsize=(6.5, 2.1 * nrows),
            sharex=True,
            sharey=True,
            squeeze=False,
            layout="constrained",
        )
        image = None
        for ax, name in zip(axes.flat, shown):
            i = methods.index(name)
            image = ax.imshow(
                levels[i],
                origin="lower",
                aspect="auto",
                interpolation="nearest",
                extent=(f[keep][0] / 2, f[keep][-1] / 2, 69, 75),
                cmap="inferno",
                vmin=vmin,
                vmax=vmax,
            )
            for fixed in rotor_rps[0, 1:]:
                ax.axvline(float(fixed), color="white", linestyle=":", linewidth=0.7, alpha=0.75)
            ax.plot(
                rotor_rps[:, 0],
                rotor_rps[:, 0],
                "x",
                color="white",
                markersize=5,
                markeredgewidth=0.9,
            )
            ax.set(
                title=LABELS[name].replace(", 2-expert", "\n2-expert"),
                yticks=rotor_rps[:, 0],
                xlim=(65, 101),
            )
        for ax in axes.flat[len(shown) :]:
            ax.set_visible(False)
        # Label the lowest visible panel of each column (the last row may be partial).
        for col in range(ncols):
            bottom = axes[(len(shown) - 1 - col) // ncols, col]
            bottom.set_xlabel("Frequency / 2 (rev/s)")
            bottom.tick_params(axis="x", labelbottom=True)
        fig.supylabel("Commanded rotor-0 speed (rev/s)", fontsize=9)
        if image is not None:
            fig.colorbar(image, ax=axes, label="Mean PSD (dB/Hz, digital units)", shrink=0.9)
        return fig


def tone_variation_spectrum(waves: Mapping[str, np.ndarray], *, sr: int = 16000) -> Figure:
    """A saved ablation example, with web-sized labels and no level matching."""
    labels = {
        "reference": "Recorded",
        "stochastic": "Our full model",
        "stochastic_coherent": "No added tone variation",
    }
    web_rc = {
        "font.family": "sans-serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "legend.fontsize": 11,
    }
    with paper_style(), mpl.rc_context(web_rc):
        fig, ax = plt.subplots(figsize=(7.4, 3.3), layout="constrained")
        for name, label in labels.items():
            f, level = _welch_db(waves[name], sr)
            ax.plot(
                f,
                level,
                label=label,
                color=COLOURS[name],
                linestyle="--" if name == "stochastic_coherent" else "-",
                linewidth=1.2,
                alpha=0.9,
            )
        ax.set(
            xlabel="Frequency (Hz)",
            ylabel="Spectrum (dB/Hz, digital units)",
            xlim=(125, 800),
        )
        ax.grid(alpha=0.2)
        fig.legend(loc="outside upper center", ncol=3, frameon=False)
        return fig


_WEB_RC = {
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.labelsize": 12,
    "axes.titlesize": 12,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
}
# Shared card geometry (inches) so every listening card and the rotor-speed
# figure put the time axis at identical horizontal positions when stacked.
_CARD_W, _CARD_H, _ROTOR_H = 7.0, 2.6, 2.2
_AX_LEFT, _AX_W, _CBAR_GAP, _CBAR_W = 0.95, 5.05, 0.12, 0.16
_AX_BOTTOM, _AX_TOP_PAD = 0.6, 0.38
_LISTEN_N_FFT, _LISTEN_HOP = 2048, 256
_COLOURBAR_LABEL = "STFT magnitude (dB)"
#: Colour-scale top above the anchor (recording) peak, in dB.
_COLOUR_HEADROOM_DB = 10.0


def _card_axes(fig: Figure, height: float) -> Axes:
    return fig.add_axes(
        (
            _AX_LEFT / _CARD_W,
            _AX_BOTTOM / height,
            _AX_W / _CARD_W,
            (height - _AX_BOTTOM - _AX_TOP_PAD) / height,
        )
    )


def listening_spectrograms(
    waves: Mapping[str, np.ndarray],
    rps: np.ndarray,
    *,
    sr: int = 16000,
    fmax: float = 4000.0,
    anchor: str = "reference",
) -> tuple[dict[str, Figure], Figure, dict[str, object]]:
    """Matched spectrogram cards for one listening case, plus its rotor-speed tracks.

    Waveforms are transformed exactly as given (no normalisation): each card
    shows ``20*log10(|STFT| + 1e-8)`` of the raw digital signal through the
    shared ``make_spectrogram_series`` + ``"audio_spectrogram"`` path, and all
    cards share one time/frequency extent and one colour range anchored on the
    ``anchor`` track (the recording). Colour limits clip the display only; a
    card whose peak exceeds the scale is labelled as display-saturated.
    """
    import tdseries as td

    from plots.timeframe.registry import TrackContext, get_renderer
    from plots.timeframe.renderers import ROTOR_COLORS, make_spectrogram_series

    names = list(waves)
    unknown = [n for n in names if n not in LABELS]
    if unknown:
        raise ValueError(f"no label for methods {unknown}; known: {list(LABELS)}")
    arrays = {n: np.asarray(waves[n]) for n in names}
    lengths = {a.shape for a in arrays.values()}
    if len(lengths) != 1 or next(iter(lengths)) == () or len(next(iter(lengths))) != 1:
        raise ValueError(f"expected mono 1-D waveforms of equal length, got shapes {lengths}")
    rotor = np.asarray(rps, dtype=float)
    if rotor.ndim != 2 or rotor.shape[0] != 4:
        raise ValueError(f"rps must have shape (4, n), got {rotor.shape}")
    n_samples = next(iter(lengths))[0]
    duration = n_samples / sr

    # torch.stft centres frame k on sample k*hop (center=True); the converter
    # assumes uncentred frames and adds n_fft/2 samples, so start the series
    # that much earlier to put frame centres at k*hop/sr.
    t_shift = -_LISTEN_N_FFT / (2 * sr)
    tracks = {
        n: make_spectrogram_series(
            td.uniform(a.astype(np.float32, copy=False), sr, dims=("time",), t_start=t_shift),
            n_fft=_LISTEN_N_FFT,
            hop_length=_LISTEN_HOP,
            fmax=fmax,
            log=True,
        )
        for n, a in arrays.items()
    }
    # One scale for every card, anchored on the recording so that a much louder
    # generated output cannot push the real audio (and every other card) into
    # the floor. Louder cards saturate on the display and are labelled.
    if anchor not in tracks:
        raise ValueError(f"colour anchor {anchor!r} is not among the waveforms {names}")
    anchor_db = tracks[anchor].series.data
    vmax = float(np.ceil(anchor_db.max())) + _COLOUR_HEADROOM_DB
    vmin = max(float(np.floor(np.percentile(anchor_db, 1))), vmax - 80.0)
    peaks = {n: float(t.series.data.max()) for n, t in tracks.items()}
    saturated = {
        n: {
            "excess_db": peaks[n] - vmax,
            "fraction_cells_above": float(np.mean(t.series.data > vmax)),
        }
        for n, t in tracks.items()
        if peaks[n] > vmax
    }
    freq_max_hz = float(next(iter(tracks.values())).hints["freq_max_hz"])
    n_bins = next(iter(tracks.values())).series.data.shape[0]
    bin_max_hz = (n_bins - 1) * sr / _LISTEN_N_FFT

    render = get_renderer("audio_spectrogram")
    cards: dict[str, Figure] = {}
    with paper_style(), mpl.rc_context(_WEB_RC):
        for name, track in tracks.items():
            fig = plt.figure(figsize=(_CARD_W, _CARD_H))
            ax = _card_axes(fig, _CARD_H)
            render(
                track.series,
                TrackContext(
                    ax=ax,
                    name=name,
                    t_start=0.0,
                    t_end=duration,
                    style={"_hints": {"freq_max_hz": bin_max_hz}},
                ),
            )
            mesh = ax.collections[-1]
            mesh.set_clim(vmin, vmax)
            ax.set(
                xlim=(0.0, duration),
                ylim=(0.0, freq_max_hz),
                xlabel="Time (s)",
                ylabel="Frequency (Hz)",
                title=LABELS[name],
            )
            cax = fig.add_axes(
                (
                    (_AX_LEFT + _AX_W + _CBAR_GAP) / _CARD_W,
                    _AX_BOTTOM / _CARD_H,
                    _CBAR_W / _CARD_W,
                    (_CARD_H - _AX_BOTTOM - _AX_TOP_PAD) / _CARD_H,
                )
            )
            excess = saturated.get(name)
            cbar = fig.colorbar(mesh, cax=cax, extend="both" if excess else "min")
            if excess:
                ax.text(
                    0.99,
                    0.97,
                    f"Display saturated: peak {excess['excess_db']:+.1f} dB above colour top",
                    transform=ax.transAxes,
                    ha="right",
                    va="top",
                    fontsize=10,
                    color="black",
                    bbox={"facecolor": "#ffd84d", "edgecolor": "black", "linewidth": 0.8, "pad": 3},
                )
            cbar.set_label(_COLOURBAR_LABEL, fontsize=9)
            cards[name] = fig

        rotor_fig = plt.figure(figsize=(_CARD_W, _ROTOR_H))
        ax = _card_axes(rotor_fig, _ROTOR_H)
        t = np.arange(rotor.shape[1]) * (duration / rotor.shape[1])
        for i, (track, colour) in enumerate(zip(rotor, ROTOR_COLORS, strict=True)):
            ax.plot(t, track, color=colour, linewidth=1.4, label=f"Rotor {i + 1}")
        ax.set(xlim=(0.0, duration), xlabel="Time (s)", ylabel="Rotor speed (rev/s)")
        ax.grid(alpha=0.25)
        ax.legend(
            loc="lower center",
            bbox_to_anchor=(0.5, 1.0),
            ncol=4,
            frameon=False,
            fontsize=10,
            borderaxespad=0.2,
        )

    metadata: dict[str, object] = {
        "methods": names,
        "labels": {n: LABELS[n] for n in names},
        "sample_rate_hz": sr,
        "duration_s": duration,
        "n_fft": _LISTEN_N_FFT,
        "hop_length": _LISTEN_HOP,
        "window": "periodic Hann (torch.hann_window), unnormalised torch.stft",
        "framing": "center=True, reflect padding; frame k centred at k*hop/sr",
        "frequency_resolution_hz": sr / _LISTEN_N_FFT,
        "frame_hop_s": _LISTEN_HOP / sr,
        "frequency_bins_shown": n_bins,
        "frequency_axis_max_hz": freq_max_hz,
        "last_bin_centre_hz": bin_max_hz,
        "frequency_coordinates": "FFT bin centres k*sr/n_fft; display extent includes the requested upper limit",
        "db_convention": (
            "20*log10(|X| + 1e-8), |X| the unnormalised STFT magnitude of the raw "
            "digital waveform (no rescaling); dB re 1 digital unit"
        ),
        "sinusoid_offset_db": float(20 * np.log10(_LISTEN_N_FFT / 4)),
        "sinusoid_note": (
            "a bin-centred sinusoid of peak amplitude A reads 20*log10(A) + sinusoid_offset_db"
        ),
        "vmin_db": vmin,
        "vmax_db": vmax,
        "colour_anchor": anchor,
        "colour_headroom_db": _COLOUR_HEADROOM_DB,
        "colour_limits_rule": (
            f"one scale for every card, anchored on the {anchor!r} track only: "
            f"vmax = ceil(anchor max) + {_COLOUR_HEADROOM_DB:g} dB; "
            "vmin = max(floor(anchor 1st percentile), vmax - 80); display clipping only"
        ),
        "peak_db": peaks,
        "display_saturated": saturated,
        "saturation_rule": (
            "a card is labelled 'display saturated' when its STFT peak exceeds vmax; values are "
            "{excess_db: peak - vmax, fraction_cells_above: share of shown cells above vmax}"
        ),
        "colourbar_label": _COLOURBAR_LABEL,
        "colormap": "magma",
        "rps_units": "rev/s",
        "rps_time_axis": "sample k of n at k*duration/n seconds (start-aligned)",
    }
    return cards, rotor_fig, metadata
