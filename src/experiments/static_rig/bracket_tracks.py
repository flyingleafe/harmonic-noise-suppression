"""Fill an unobserved rank only between two complete, consistent rotor sets.

This is a fallback for track fragmentation, not another line detector. Both
ends must have real spectral evidence (not interpolation); no recorded switch-
off, extrapolation, rank crowding or contradictory observed point is bridged.
"""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np


def _rejoin_scale_fragments(track, t, rates, states, ids, err, orders):
    """Reattach a short octave-switched line between two knots of one known track.

    A recording-wide band can admit both 2x and 3x. Here both neighbors must
    belong to the *same* anchored track and the association must be unique
    in both directions. Corrected knots never become witnesses in this pass.
    """
    decisions = track["diagnostics"].get("scale_calibration", {}).get("tracks", ())
    if not decisions:
        return []
    known = {v["track_id"] for v in decisions if v["basis"] != "blind_no_scale_corroboration"}
    by_id = defaultdict(list)
    for i, j in zip(*np.nonzero(np.isfinite(rates)), strict=True):
        if ids[i, j] in known and states[i, j] != "interpolated":
            by_id[ids[i, j]].append((i, j))
    options = defaultdict(list)
    for tid, knots in by_id.items():
        for (a, u), (b, v) in zip(knots, knots[1:]):
            gap = t[b] - t[a]
            if b == a + 1 or gap > 8 or abs(rates[b, v] - rates[a, u]) > 1.0:
                continue
            if not any(
                start <= t[a] <= t[b] < end for start, end in track["diagnostics"]["active_spans"]
            ):
                continue
            for i in range(a + 1, b):
                if tid in ids[i]:
                    continue
                target = rates[a, u] + (rates[b, v] - rates[a, u]) * ((t[i] - t[a]) / gap)
                for j in np.flatnonzero(np.isfinite(rates[i])):
                    if ids[i, j] in known or states[i, j] == "interpolated":
                        continue
                    others = np.delete(rates[i], j)
                    if np.any(abs(others - target) < 0.5):
                        continue
                    for factor in (1.0, 2.0, 0.5, 3.0, 1 / 3):
                        rate = factor * rates[i, j]
                        if abs(rate - target) > 0.35:
                            continue
                        ks = [
                            round(k / factor)
                            for k in orders[i][j]
                            if k / factor >= 1 and abs(k / factor - round(k / factor)) < 1e-6
                        ]
                        if len(ks) >= 2:
                            options[i, j].append((tid, factor, ks, t[a], t[b]))
    occupied = defaultdict(int)
    for (i, _), opts in options.items():
        for tid, *_ in opts:
            occupied[i, tid] += 1
    result = []
    for (i, j), opts in options.items():
        if len(opts) != 1:
            continue
        tid, factor, ks, start, stop = opts[0]
        if occupied[i, tid] != 1:
            continue
        result.append(
            {
                "time_s": t[i],
                "slot": int(j),
                "previous_track_id": ids[i, j],
                "confirmed_track_id": tid,
                "factor": factor,
                "anchor_span_s": [start, stop],
            }
        )
        rates[i, j] *= factor
        states[i, j] = "track_supported"
        err[i, j] = max(0.15, err[i, j] * factor)
        ids[i, j] = tid
        orders[i][j] = ks
    return result


def fill_tracks(track: dict) -> dict:
    t = np.asarray(track["times_s"], float)
    rates = np.array(track["rates_rps"], float)
    err = np.array(track["uncertainty_rps"], float)
    ids = np.array(track["track_id"], dtype=object)
    states = np.array(track["provenance"], dtype=object)
    orders = [[list(cell) for cell in row] for row in track["order_support"]]
    rejoined = _rejoin_scale_fragments(track, t, rates, states, ids, err, orders)
    n = rates.shape[1]
    anchors = np.flatnonzero(np.all(np.isin(states, ["measured", "track_supported"]), axis=1))
    audit = []
    # Reuse a stable ID when present at both ends; a bridge between fragmented
    # IDs has its own anonymous ID. Never claim to preserve motor identity.
    for left, right in zip(anchors, anchors[1:]):
        if (
            right == left + 1
            or t[right] - t[left] > 16.0
            or np.any(states[left : right + 1] == "inactive")
        ):
            continue
        if not np.isnan(rates[left + 1 : right]).any():
            continue
        # Distinct activity spans can lie between adjacent grid cells. Only
        # interpolate if a single acoustic run covers both real endpoints.
        if not any(a <= t[left] <= t[right] < b for a, b in track["diagnostics"]["active_spans"]):
            continue
        li = np.argsort(rates[left])
        ri = np.argsort(rates[right])
        v0 = rates[left, li]
        v1 = rates[right, ri]
        if np.max(abs(v1 - v0)) > 1.0:
            continue
        # No rank if its observed frequency range can plausibly touch another.
        low = np.minimum(v0, v1) - 0.18
        hi = np.maximum(v0, v1) + 0.18
        isolated = np.array(
            [all(k == h or low[k] > hi[h] or low[h] > hi[k] for h in range(n)) for k in range(n)]
        )
        fills = []
        consistent = True
        for i in range(left + 1, right):
            pred = v0 + (v1 - v0) * (t[i] - t[left]) / (t[right] - t[left])
            ok = np.flatnonzero(np.isfinite(rates[i]))
            obs = np.sort(rates[i, ok])
            available = np.flatnonzero(~np.isfinite(rates[i]))
            if len(obs):
                choices = [
                    (max(abs(obs - pred[list(ix)])), ix) for ix in combinations(range(n), len(obs))
                ]
                best, _ = min(choices)
                if best > 0.65:
                    consistent = False
                    break
                # An ambiguous association is not evidence for the complementary rank.
                possible = {j for d, ix in choices if d <= min(0.65, best + 0.15) for j in ix}
            else:
                possible = set()
            gaps = [j for j in range(n) if isolated[j] and j not in possible]
            for slot, j in zip(available, gaps):
                a, b = ids[left, li[j]], ids[right, ri[j]]
                tid = a if a == b else f"bracket:{left}:{right}:{j}"
                if tid in ids[i, ok]:
                    continue
                uncertainty = max(
                    err[left, li[j]],
                    err[right, ri[j]],
                    0.12 + 0.035 * min(t[i] - t[left], t[right] - t[i]),
                )
                fills.append((i, slot, float(pred[j]), tid, uncertainty))
        if consistent and fills:
            for i, slot, value, tid, uncertainty in fills:
                rates[i, slot] = value
                states[i, slot] = "interpolated"
                ids[i, slot] = tid
                err[i, slot] = uncertainty
            audit.append(
                {"left_s": float(t[left]), "right_s": float(t[right]), "filled_cells": len(fills)}
            )
    missing = np.array(None, dtype=object)
    return dict(
        track,
        rates_rps=np.where(np.isfinite(rates), rates, missing).tolist(),
        uncertainty_rps=np.where(np.isfinite(err), err, missing).tolist(),
        track_id=ids.tolist(),
        provenance=states.tolist(),
        order_support=orders,
        diagnostics={
            **track["diagnostics"],
            "complete_set_bridges": audit,
            "scale_rejoins": rejoined,
        },
    )
