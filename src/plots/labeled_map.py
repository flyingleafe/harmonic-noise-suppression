"""A 2D map of a few labelled points that stays readable when points crowd.

One coordinate system, equal aspect. Every point gets a distinct colour and
marker, a legend entry, and a direct text label placed greedily (in display
pixels) to avoid other labels, markers and the axes edge, with a leader line
when pushed away. If some points crowd each other, the tightest group is
repeated in a zoomed inset of the SAME coordinates (``indicate_inset_zoom``)
and labelled there too. Rendered with the Agg canvas only (no pyplot state).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from scipy.spatial.distance import pdist, squareform

_MARKERS = "osD^vPX*hp<>"
_MARKER_SIZE = 90  # scatter area, pt^2
_CROWD_FRACTION = 0.08  # nearest neighbour closer than this x span -> crowded


def _overlap(a: Bbox, b: Bbox) -> float:
    w = min(a.x1, b.x1) - max(a.x0, b.x0)
    h = min(a.y1, b.y1) - max(a.y0, b.y0)
    return max(w, 0.0) * max(h, 0.0)


def _crowded_group(coords: np.ndarray) -> list[int]:
    """Indices of the largest group of mutually-near points (nearest origin on ties)."""
    n = coords.shape[0]
    if n < 2:
        return []
    span = float(np.ptp(coords, axis=0).max())
    if span <= 0:
        return list(range(n))
    near = squareform(pdist(coords)) < _CROWD_FRACTION * span
    seen, groups = set(), []
    for s in range(n):
        if s in seen:
            continue
        stack, comp = [s], []
        seen.add(s)
        while stack:
            i = stack.pop()
            comp.append(i)
            for j in np.flatnonzero(near[i]):
                if j not in seen:
                    seen.add(int(j))
                    stack.append(int(j))
        if len(comp) >= 2:
            groups.append(sorted(comp))
    if not groups:
        return []
    return max(groups, key=lambda g: (len(g), -float(np.linalg.norm(coords[g].mean(axis=0)))))


def _place_labels(ax, coords, labels, colors, idx, renderer, obstacles, fontsize):
    """Greedy collision-avoiding text labels for points ``idx`` on ``ax``."""
    fig = ax.figure
    px = fig.dpi / 72.0
    disp = ax.transData.transform(coords)
    half = 0.5 * np.sqrt(_MARKER_SIZE) * px + 2
    markers = [Bbox.from_extents(x - half, y - half, x + half, y + half) for x, y in disp[idx]]
    axbox = ax.get_window_extent(renderer)
    center = disp[idx].mean(axis=0)
    placed: list[Bbox] = []
    nn = squareform(pdist(disp[idx])) if len(idx) > 1 else np.zeros((1, 1))
    np.fill_diagonal(nn, np.inf)
    order = np.argsort(nn.min(axis=1))  # most crowded first
    angles = np.deg2rad(np.arange(0, 360, 22.5))
    pad = 3 * px
    for o in order:
        i = idx[o]
        out_dir = disp[i] - center
        base = np.arctan2(out_dir[1], out_dir[0]) if np.hypot(*out_dir) > 1 else np.pi / 4
        cand = sorted(angles, key=lambda a: abs(np.angle(np.exp(1j * (a - base)))))
        best = None
        for radius in (10, 18, 28, 40, 55, 75):
            for a in cand:
                c, s = np.cos(a), np.sin(a)
                ha = "left" if c > 0.3 else "right" if c < -0.3 else "center"
                va = "bottom" if s > 0.3 else "top" if s < -0.3 else "center"
                t = ax.annotate(
                    labels[i],
                    coords[i],
                    xytext=(radius * c, radius * s),
                    textcoords="offset points",
                    ha=ha,
                    va=va,
                    fontsize=fontsize,
                )
                bb = t.get_window_extent(renderer)
                bb = Bbox.from_extents(bb.x0 - pad, bb.y0 - pad, bb.x1 + pad, bb.y1 + pad)
                t.remove()
                area = bb.width * bb.height
                cost = sum(_overlap(bb, q) for q in placed + obstacles)
                cost += 2 * sum(_overlap(bb, q) for q in markers)
                cost += 5 * (area - _overlap(bb, axbox))
                if best is None or cost < best[0]:
                    best = (cost, radius, c, s, ha, va, bb)
            assert best is not None  # every radius tries at least one angle
            if best[0] == 0:
                break
        assert best is not None
        _, radius, c, s, ha, va, bb = best
        arrow = (
            dict(
                arrowstyle="-",
                color=colors[i],
                lw=0.9,
                shrinkA=0,
                shrinkB=0.5 * np.sqrt(_MARKER_SIZE),
            )
            if radius > 10
            else None
        )
        ax.annotate(
            labels[i],
            coords[i],
            xytext=(radius * c, radius * s),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=fontsize,
            zorder=5,
            arrowprops=arrow,
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=colors[i], lw=0.8, alpha=0.9),
        )
        placed.append(bb)


def _scatter(ax, coords, colors):
    for i, (x, y) in enumerate(coords):
        ax.scatter(
            x,
            y,
            s=_MARKER_SIZE,
            marker=_MARKERS[i % len(_MARKERS)],
            color=colors[i],
            edgecolor="black",
            linewidth=0.8,
            zorder=4,
        )


def plot_labeled_map(
    coords: np.ndarray,
    labels: list[str],
    path: Path,
    *,
    legend_labels: list[str] | None = None,
    info_lines: list[str] | None = None,
    title: str = "",
    axis_labels: tuple[str, str] = ("dim 1", "dim 2"),
    dpi: int = 200,
) -> Path:
    """Save a labelled 2D point map to ``path`` (PNG at ``dpi``) and return it."""
    coords = np.asarray(coords, dtype=float)
    n = coords.shape[0]
    colors = [matplotlib.colormaps["tab10"](i % 10) for i in range(n)]
    fig = Figure(figsize=(11, 7.5), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes((0.07, 0.09, 0.62, 0.83))
    ax.axhline(0, color="0.85", lw=0.8, zorder=0)
    ax.axvline(0, color="0.85", lw=0.8, zorder=0)
    _scatter(ax, coords, colors)
    lo, hi = coords.min(axis=0), coords.max(axis=0)
    pad = 0.22 * max(float((hi - lo).max()), 1e-12)
    ax.set_xlim(lo[0] - pad, hi[0] + pad)
    ax.set_ylim(lo[1] - pad, hi[1] + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(axis_labels[0])
    ax.set_ylabel(axis_labels[1])
    ax.set_title(title)
    ax.grid(alpha=0.25, lw=0.5)
    canvas.draw()
    renderer = canvas.get_renderer()

    obstacles: list[Bbox] = []
    group = _crowded_group(coords)
    if group:
        frac = ax.transAxes.inverted().transform(ax.transData.transform(coords))
        size = 0.4
        gbox = Bbox.from_extents(*frac[group].min(axis=0), *frac[group].max(axis=0))

        def corner_cost(x0, y0):
            box = Bbox.from_bounds(x0, y0, size, size)
            inside = sum(box.contains(*pt) for pt in frac)
            return 10 * inside + 100 * _overlap(box, gbox.expanded(1.3, 1.3)) / max(
                gbox.width * gbox.height, 1e-6
            )

        corners = [(0.58, 0.58), (0.02, 0.58), (0.58, 0.02), (0.02, 0.02)]
        x0, y0 = min(corners, key=lambda c: corner_cost(*c))
        axins = ax.inset_axes((x0, y0, size, size))
        axins.axhline(0, color="0.85", lw=0.8, zorder=0)
        axins.axvline(0, color="0.85", lw=0.8, zorder=0)
        _scatter(axins, coords, colors)
        c = coords[group].mean(axis=0)
        r = 0.5 * float(np.ptp(coords[group], axis=0).max()) * 1.9 + 1e-12
        axins.set_xlim(c[0] - r, c[0] + r)
        axins.set_ylim(c[1] - r, c[1] + r)
        axins.set_aspect("equal", adjustable="box")
        axins.tick_params(labelsize=7)
        axins.set_facecolor("#fbfbfb")
        axins.set_title("zoom (same axes)", fontsize=8, pad=3)
        ax.indicate_inset_zoom(axins, edgecolor="0.4")
        fig.canvas.draw()
        xl, yl = axins.get_xlim(), axins.get_ylim()
        visible = [
            i for i in range(n) if xl[0] <= coords[i, 0] <= xl[1] and yl[0] <= coords[i, 1] <= yl[1]
        ]
        _place_labels(axins, coords, labels, colors, visible, renderer, [], fontsize=8)
        inset_box = axins.get_tightbbox(renderer)
        assert inset_box is not None  # None only for an invisible axes
        obstacles.append(inset_box)

    _place_labels(ax, coords, labels, colors, list(range(n)), renderer, obstacles, fontsize=9)

    handles = [
        Line2D(
            [],
            [],
            marker=_MARKERS[i % len(_MARKERS)],
            linestyle="",
            markersize=9,
            markerfacecolor=colors[i],
            markeredgecolor="black",
        )
        for i in range(n)
    ]
    fig.legend(
        handles,
        legend_labels or labels,
        loc="upper left",
        bbox_to_anchor=(0.71, 0.92),
        fontsize=9,
        frameon=False,
    )
    if info_lines:
        fig.text(
            0.715,
            0.09,
            "\n".join(info_lines),
            fontsize=8.5,
            family="monospace",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", fc="#f6f6f6", ec="0.7"),
        )
    path = Path(path)
    fig.savefig(path, dpi=dpi)
    return path
