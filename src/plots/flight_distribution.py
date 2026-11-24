"""Interactive flight MDS, with two magnifications of the same coordinates."""

from itertools import combinations
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from scipy.spatial.distance import pdist


def _bounds(points: np.ndarray, margin: float) -> np.ndarray:
    pad = np.maximum(np.ptp(points, axis=0), np.ptp(points, axis=0).max() * 0.35) * margin + 1e-6
    return np.array([points.min(axis=0) - pad, points.max(axis=0) + pad])


def write_flight_map(records: list[dict], names: dict[str, str], path: Path) -> None:
    rigs = sorted({r["rig"] for r in records})
    colors = dict(
        zip(rigs, ("#176da3", "#db7108", "#23844b", "#bd3a46", "#8455aa", "#816047", "#cc4596"))
    )
    xy = np.array([(r["mds_1"], r["mds_2"]) for r in records])
    groups = {rig: [i for i, r in enumerate(records) if r["rig"] == rig] for rig in rigs}
    centers = np.array([xy[ix].mean(axis=0) for ix in groups.values()])
    near = min(combinations(range(len(rigs)), 3), key=lambda ix: pdist(centers[list(ix)]).max())
    # Keep a very tight pair readable even if its third neighbour is much farther away.
    pair = min(combinations(near, 2), key=lambda ix: pdist(centers[list(ix)])[0])
    if pdist(centers[list(pair)])[0] * 8 < pdist(centers[list(near)]).max():
        near = pair
    boxes = [_bounds(xy, 0.06), _bounds(centers, 0.24), _bounds(centers[list(near)], 0.65)]
    panels = [
        ("All flights", "", [0.015, 0.60], [0.08, 0.91]),
        ("Centroid region", "2", [0.68, 0.995], [0.55, 0.925]),
        ("Closest centroids", "3", [0.68, 0.995], [0.055, 0.425]),
    ]
    fig = go.Figure()
    for j, ((title, suffix, xd, yd), (lo, hi)) in enumerate(zip(panels, boxes)):
        xref, yref = "x" + suffix, "y" + suffix
        n = int(((xy >= lo) & (xy <= hi)).all(axis=1).sum())
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=xd[0],
            y=yd[1] + 0.025,
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            text=f"<b>{title}</b> · {n} flights",
            font=dict(size=15),
        )
        common = dict(
            showline=True,
            mirror=True,
            linecolor="#c6cdd1",
            gridcolor="#e6eaee",
            zeroline=False,
            constrain="domain",
            nticks=6,
        )
        fig.update_layout(
            {
                "xaxis" + suffix: dict(
                    common,
                    domain=xd,
                    anchor=yref,
                    range=[lo[0], hi[0]],
                    title="MDS 1" if not j else "",
                ),
                "yaxis" + suffix: dict(
                    common,
                    domain=yd,
                    anchor=xref,
                    range=[lo[1], hi[1]],
                    scaleanchor=xref,
                    scaleratio=1,
                    title="MDS 2" if not j else "",
                ),
            }
        )
        if j < 2:
            a, b = boxes[j + 1]
            fig.add_shape(
                type="rect",
                xref=xref,
                yref=yref,
                x0=a[0],
                x1=b[0],
                y0=a[1],
                y1=b[1],
                line=dict(color="#667988", width=1, dash="dot"),
                layer="below",
            )
        for k, rig in enumerate(rigs):
            ix = groups[rig]
            name = names[rig]
            color = colors[rig]
            meta = [
                [
                    records[i]["flight"],
                    records[i]["n_windows"],
                    records[i]["role"],
                    *(
                        records[i][p]
                        for p in ("front_right", "front_left", "rear_left", "rear_right")
                    ),
                ]
                for i in ix
            ]
            fig.add_scatter(
                x=xy[ix, 0].tolist(),
                y=xy[ix, 1].tolist(),
                xaxis=xref,
                yaxis=yref,
                mode="markers",
                name=f"{name} ({len(ix)})",
                legendgroup=rig,
                showlegend=j == 0,
                marker=dict(size=7, color=color, opacity=0.8, line=dict(color="white", width=0.4)),
                customdata=meta,
                hovertemplate=f"<b>{name}</b> · %{{customdata[0]}}<br>"
                "%{customdata[1]} cruise windows · %{customdata[2]}<br>"
                "Mean speed (rev/s): FR %{customdata[3]:.1f} · FL %{customdata[4]:.1f}"
                " · RL %{customdata[5]:.1f} · RR %{customdata[6]:.1f}<extra></extra>",
            )
            label = (j == 1 and k not in near) or (j == 2 and k in near)
            fig.add_scatter(
                x=[centers[k, 0]],
                y=[centers[k, 1]],
                xaxis=xref,
                yaxis=yref,
                mode="markers+text" if label else "markers",
                name=f"{name} centroid",
                text=[f"<b>{name}</b>"] if label else None,
                textposition="bottom center",
                textfont=dict(size=13, color=color, shadow="0 0 3px white"),
                legendgroup=rig,
                showlegend=False,
                hoverinfo="skip",
                marker=dict(symbol="diamond-open", size=15, color=color, line=dict(width=2)),
            )
    fig.update_layout(
        template="plotly_white",
        width=1260,
        height=835,
        margin=dict(l=65, r=25, t=105, b=80),
        title=dict(
            text="<b>Cruise operating speed and motor dynamics</b><br>"
            '<span style="font-size:15px">One dot per flight · Diamonds are drone centroids · '
            "Hover for mean rotor speeds; click for traces.</span>",
            x=0.05,
            y=0.96,
        ),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=-0.08, groupclick="togglegroup"),
        hovermode="closest",
        dragmode="zoom",
    )
    fig.write_html(
        path,
        div_id="flight-map",
        config=dict(
            displaylogo=False,
            scrollZoom=True,
            toImageButtonOptions=dict(format="svg", filename="flight_mds"),
        ),
    )
