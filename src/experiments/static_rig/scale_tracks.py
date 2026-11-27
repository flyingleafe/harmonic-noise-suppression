"""Identify shaft order on a full acoustic trajectory from same-record speed anchors.

The waveform model supplies frequencies, time support, continuous track IDs and
harmonic orders. A single multiplier is selected per track ID, never opportunistically
per frame. Anchor use is explicit. Lacking scale evidence, keep the blind estimate
labelled as such. Conflicting harmonic copies cannot count as two distinct shafts.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np

FACTORS = np.asarray((1.0, 2.0, 0.5, 3.0, 1 / 3))
TOL = 0.65


def _ranks(x):
    return np.asarray(x, dtype=float)


def _count_distinct_blocks(events):
    return len(set(int(t // 2.0) for t in events))


def calibrate(track, observations, *, written_for, corroboration_rps=TOL):
    """Return a new track, retaining input time support. Inputs use relative native seconds.

    Direct one-to-one scale votes require a matching train anchor that does
    not also explain a different fitted track at a comparable distance.
    Whole-record scale inference is permitted only if at least three spaced
    anchors establish a compact speed band, **one** scale gives rates inside
    it, and its frequencies recur close to multiple training observations.
    Neither rule can add an audible component not found by the waveform model.
    """
    t = _ranks(track["times_s"])
    raw = _ranks(track["rates_rps"])
    err = _ranks(track["uncertainty_rps"])
    ids = np.asarray(track["track_id"], dtype=object)
    tags = np.asarray(track["provenance"], dtype=object).copy()
    new_ids = ids.copy()
    new = raw.copy()
    new_err = err.copy()
    orders = [[list(s) for s in row] for row in track["order_support"]]
    points = [(float(a["time_s"]), _ranks(a["rates_rps"])) for a in observations if a["rates_rps"]]
    finite = np.isfinite(raw)
    sequences = defaultdict(list)
    for i, j in zip(*np.nonzero(finite), strict=True):
        sequences[ids[i, j]].append((i, j))
    votes = defaultdict(lambda: [[] for _ in FACTORS])
    witnesses = defaultdict(list)
    for at, ref in points:
        i = int(np.argmin(abs(t - at)))
        if abs(t[i] - at) > 0.56:
            continue
        slots = np.flatnonzero(finite[i] & (tags[i] != "interpolated"))
        if not len(slots):
            continue
        delta = abs(raw[i, slots, None, None] * FACTORS[None, :, None] - ref[None, None, :])
        best = delta.min(axis=1)
        for h, s in enumerate(ref):
            rank = np.argsort(best[:, h], kind="stable")
            q = rank[0]
            k = int(delta[q, :, h].argmin())
            distance = delta[q, k, h]
            if distance > corroboration_rps or (
                len(rank) > 1 and best[rank[1], h] - distance < 0.1
            ):
                continue
            identifier = ids[i, slots[q]]
            votes[identifier][k].append(at)
            witnesses[identifier].append((at, float(s), k))
    if points:
        prior = np.sort(np.concatenate([v for _, v in points]))
        lo, hi = np.quantile(prior, [0.025, 0.975])
        pad = max(2.0, 0.06 * np.median(prior))
        bounds = (max(1.0, lo - pad), hi + pad)
        compact = _count_distinct_blocks(at for at, _ in points) >= 3 and hi / lo < 1.72
    else:
        prior = np.array([])
        bounds = None
        compact = False
    quality = {}
    decisions = []
    for identifier, where in sequences.items():
        ii, jj = np.asarray(where).T
        data = raw[ii, jj]
        scores = np.array([_count_distinct_blocks(events) for events in votes[identifier]])
        ix = np.argsort(-scores, kind="stable")
        pick = 0
        basis = "blind_no_scale_corroboration"
        selected = False
        if scores[ix[0]] >= 2 and scores[ix[0]] >= 2 * scores[ix[1]] + 1:
            pick = int(ix[0])
            basis = "track_anchor"
            selected = True
        elif compact and bounds is not None:
            a, b = np.quantile(data, [0.1, 0.9])
            allow = np.flatnonzero((a * FACTORS >= bounds[0]) & (b * FACTORS <= bounds[1]))
            if len(allow) == 1:
                candidate = int(allow[0])
                mid = float(np.median(data)) * FACTORS[candidate]
                width = max(1.0, (b - a) * FACTORS[candidate] + 0.5)
                repeats = [at for at, v in points if np.min(abs(v - mid)) <= width]
                if _count_distinct_blocks(repeats) >= 2 and not np.any(
                    np.delete(scores, candidate) > scores[candidate]
                ):
                    pick = candidate
                    basis = "recording_band"
                    selected = True
        gain = float(FACTORS[pick])
        new[ii, jj] = gain * data
        new_err[ii, jj] *= gain
        for i, j in where:
            if gain != 1.0:
                mapped = [
                    int(round(k / gain))
                    for k in orders[i][j]
                    if abs(k / gain - round(k / gain)) < 1e-7 and k / gain >= 1
                ]
                if tags[i, j] != "interpolated" and len(mapped) < 2:
                    new[i, j] = new_err[i, j] = np.nan
                    tags[i, j] = "unresolved"
                    new_ids[i, j] = None
                    orders[i][j] = []
                    continue
                orders[i][j] = mapped
                if tags[i, j] == "measured":
                    tags[i, j] = "track_supported"
                new_err[i, j] = max(new_err[i, j], 0.10 if basis == "track_anchor" else 0.25)
        quality[identifier] = (
            2 if basis == "track_anchor" else 1 if selected else 0,
            int(scores[pick]),
            len(where),
        )
        decisions.append(
            {
                "track_id": identifier,
                "scale_factor": gain,
                "basis": basis,
                "scale_votes": scores.tolist(),
                "witness_count": len(witnesses[identifier]),
            }
        )
    duplicates = 0
    for i in range(len(t)):
        kept = []
        for j in sorted(
            np.flatnonzero(np.isfinite(new[i])),
            key=lambda j: (quality[ids[i, j]], -new_err[i, j]),
            reverse=True,
        ):
            if any(
                abs(new[i, j] - new[i, q]) < max(0.12, min(0.35, new_err[i, j] + new_err[i, q]))
                for q in kept
            ):
                new[i, j] = new_err[i, j] = np.nan
                tags[i, j] = "unresolved"
                new_ids[i, j] = None
                orders[i][j] = []
                duplicates += 1
            else:
                kept.append(j)
    missing = np.array(None, dtype=object)
    calc_id = "same-source-harmonic-order/1"
    diag = dict(track["diagnostics"])
    diag["scale_calibration"] = {
        "method": calc_id,
        "anchor_mode": written_for,
        "n_input_anchors": len(points),
        "factors": FACTORS.tolist(),
        "recording_band_rps": bounds if compact else None,
        "tracks": decisions,
        "coincident_cells_removed": duplicates,
        "tolerance_rps": corroboration_rps,
    }
    return dict(
        track,
        rates_rps=np.where(np.isfinite(new), new, missing).tolist(),
        uncertainty_rps=np.where(np.isfinite(new_err), new_err, missing).tolist(),
        provenance=tags.tolist(),
        track_id=new_ids.tolist(),
        order_support=orders,
        diagnostics=diag,
    )
