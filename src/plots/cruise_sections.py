"""Measured and independently generated cruise sections on the same rev/s axes."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from plotly.subplots import make_subplots

from plots.timeframe.renderers import ROTOR_COLORS

METHODS = ("Measured", "Proposed", "VAR–HSMM hybrid")
ROTORS = ("Front right", "Front left", "Rear left", "Rear right")
RIGS = {"michaels": "MD2", "vid_m100": "VID M100", "dregon": "DREGON"}
ORIGINS = ("early", "middle", "late")


def _ylim(x):
    lo, hi = float(x.min()), float(x.max())
    pad = max(0.5, 0.06 * (hi - lo))
    return [lo - pad, hi + pad]


def render(out: Path) -> None:
    with np.load(out / "cruise_draw0.npz") as data:
        waves, starts = data["waves"], data["starts_s"]
        keys = [tuple(k) for k in data["labels"].tolist()]
        t = np.arange(waves.shape[-1]) / float(data["fs"])
    summary = json.loads((out / "cruise_diagnosis.json").read_text())["summary"]

    def best(rig):
        gaps = summary["rigs"][rig]["gap_by_flight"]
        return max(gaps, key=gaps.__getitem__)

    vid = summary["rigs"]["vid_m100"]["gap_by_flight"]
    selected = [
        ("vid_m100", best("vid_m100")),
        ("vid_m100", min(vid, key=vid.__getitem__)),
        ("dregon", best("dregon")),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12.1, 7.55), sharex=True, sharey="row")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.80, bottom=0.087, hspace=0.78, wspace=0.07)
    for row, key in enumerate(selected):
        i = keys.index(key)
        rig, flight = key
        for col, ax in enumerate(axes[row]):
            y = waves[i, col, 1]
            for trace, name, color in zip(y, ROTORS, ROTOR_COLORS):
                ax.plot(t, trace, color=color, label=name, lw=0.88)
            ax.grid(axis="y", alpha=0.22)
            ax.spines[["top", "right"]].set_visible(False)
            ax.set_xlim(0, 16)
            ax.set_xticks([0, 4, 8, 12, 16])
            ax.tick_params(labelsize=9)
            if row == 0:
                ax.set_title(METHODS[col], fontsize=11, pad=3)
            if row == 2:
                ax.set_xlabel("Time in window (s)", fontsize=10)
        axes[row, 0].set_ylim(*_ylim(waves[i, :, 1]))
        axes[row, 0].set_ylabel("Speed (rev/s)", fontsize=10)
        gap = summary["rigs"][rig]["gap_by_flight"][flight]
        winner = "hybrid" if gap > 0 else "proposed"
        axes[row, 0].text(
            0,
            1.27 if not row else 1.12,
            f"{RIGS[rig]}  /  {flight}     ·     t = {starts[i, 1]:.1f} s     ·     {winner} wins this flight",
            transform=axes[row, 0].transAxes,
            fontsize=11,
            weight="bold",
            color="#243e50",
        )
    fig.suptitle(
        "Settled cruise: same schedule, independent motor-speed trajectories",
        fontsize=16,
        weight="bold",
        y=0.992,
    )
    fig.legend(
        *axes[0, 0].get_legend_handles_labels(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.947),
        ncol=4,
        frameon=False,
    )
    fig.savefig(out / "cruise_sections.png", dpi=180, facecolor="white")
    plt.close(fig)

    interactive = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        # plotly leaves the parameter unannotated; its docstring allows "all" (boolean or str).
        shared_yaxes="all",  # pyright: ignore[reportArgumentType]
        vertical_spacing=0.075,
        subplot_titles=METHODS,
    )
    for col in range(3):
        for motor, (name, color) in enumerate(zip(ROTORS, ROTOR_COLORS)):
            interactive.add_scatter(
                x=t.tolist(),
                y=waves[0, col, 0, motor].tolist(),
                mode="lines",
                name=name,
                line=dict(color=color, width=1.2),
                legendgroup=name,
                showlegend=col == 0,
                row=col + 1,
                col=1,
                hovertemplate="%{y:.2f} rev/s<extra>" + name + "</extra>",
            )

    def layout(i, j):
        rig, name = keys[i]
        lim = _ylim(waves[i, :, j])
        return {
            "title.text": f"<b>{RIGS[rig]} · {name}</b><br><sup>{ORIGINS[j].capitalize()} cruise window, "
            f"{starts[i, j]:.2f}–{starts[i, j] + 16:.2f} s from record start · generated draw 0 of 32</sup>",
            **{axis + ".range": lim for axis in ("yaxis", "yaxis2", "yaxis3")},
            **{axis + ".range": [0, 16] for axis in ("xaxis", "xaxis2", "xaxis3")},
        }

    buttons = []
    for i, (rig, flight) in enumerate(keys):
        for j, origin in enumerate(ORIGINS):
            ys = waves[i, :, j].reshape(12, -1).round(4).tolist()
            buttons.append(
                dict(
                    label=f"{RIGS[rig]} / {flight} / {origin}",
                    method="update",
                    args=[{"y": ys}, layout(i, j)],
                )
            )
    interactive.update_layout(
        template="plotly_white",
        height=820,
        margin=dict(l=68, r=30, t=138, b=66),
        title=dict(x=0.04, y=0.985, xanchor="left"),
        legend=dict(orientation="h", y=-0.085),
        updatemenus=[dict(buttons=buttons, x=0, xanchor="left", y=1.11, yanchor="top")],
        hovermode="x unified",
    )
    interactive.update_layout(layout(0, 0))
    interactive.update_yaxes(title_text="Speed (rev/s)")
    interactive.update_xaxes(title_text="Time in window (s)", row=3, col=1)
    interactive.write_html(
        out / "cruise_sections.html",
        include_plotlyjs=True,
        config=dict(displaylogo=False, scrollZoom=True),
        div_id="cruise-sections",
    )
