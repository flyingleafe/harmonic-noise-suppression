"""Narrow band-power cross-check for anonymous acoustic tachometry.

The primary estimator works on detected, matched spectral peaks. This
verifier fits band-power combs without using any of those peak locations or
assignments. Per-microphone whitening and spike/slab line odds follow the AVQ
static-rig analysis. Search is restricted to each proposed neighborhood and
its ordinal aliases. Spectrum information never comes from telemetry.
"""

from __future__ import annotations

import numpy as np

from tracking.dsp import line_cell_spectrum

EVIDENCE_RECIPE = "avq-band-corroboration/2"


def compare_families(
    audio: np.ndarray,
    sr: float,
    f_max: float,
    rates: np.ndarray,
    tolerances: np.ndarray,
    blades: int | None,
    motor_orders: tuple[int, ...],
    motor_spacing: int,
) -> list[dict]:
    """Independent rate, exclusive orders and score margin for each proposal."""
    power, whitened, df = line_cell_spectrum(audio, sr, f_max)
    t = audio.shape[-1] / sr
    # Average one native bin (not four independent zero-padded ordinates).
    half = max(1, round(0.52 / (t * df)))
    saturated = np.minimum(whitened, 1e4)
    prefix = np.r_[0.0, np.cumsum(saturated, dtype=np.float64)]
    max_order = max(64, 4 * (blades or 1), max(motor_orders, default=0))
    orders = np.arange(motor_spacing, max_order + 1, motor_spacing)
    g = 100.0 / np.maximum(1, orders / max(1, motor_spacing)) ** 2
    if blades:
        g *= np.where(orders % blades == 0, blades**2, 1.0)
    slot = {int(k): i for i, k in enumerate(orders)}
    out = []
    for i, candidate in enumerate(rates):
        other = np.delete(rates, i)
        alternatives = (1.0, 0.5, 1 / 3, 0.25, 2.0, 3.0, 4.0)
        # Compare the two physically plausible orderings when the shaft's
        # documented blade number is not one of the standard aliases above.
        if blades and blades > 4:
            alternatives += (1.0 / blades, float(blades))
        modes = []
        for ratio in alternatives:
            seed = candidate * ratio
            if not 15 <= seed <= 600:
                continue
            width = min(0.55, max(0.10, tolerances[i] * ratio * 1.2))
            grid = seed + np.linspace(-width, width, 49)
            freq = grid[:, None] * orders
            centers = np.clip(np.rint(freq / df).astype(int), 0, power.size - 1)
            begins = np.maximum(0, centers - half)
            ends = np.minimum(power.size, centers + half + 1)
            z = (prefix[ends] - prefix[begins]) / (ends - begins) * np.log(2.0)
            valid = (freq < f_max - 1.0 / t) & (freq >= 15.0)
            # Count a line once. For nearby rotors, collision is decided in
            # measured hertz, not relative position in the zero-padded grid.
            for r in other:
                lattice = r * motor_spacing
                valid &= abs(freq - np.rint(freq / lattice) * lattice) >= 1.35 / t
            line = np.logaddexp(np.log(0.7), np.log(0.3) - np.log1p(g) + z * g / (1 + g))
            prom = np.where(valid, z, 0.0)
            cells = valid & (z >= 5.5) & (line > 0)
            # Cap each support contribution so a carrier cannot stand in for
            # an entire family. Downweight very high orders in the coarse
            # octave decision; their narrow lobes still refine the rate.
            contribution = np.where(cells, np.log1p(np.maximum(line, 0.0)), 0.0)
            raw_strength = np.where(cells, power[centers], 0.0)
            n = cells.sum(axis=1)
            low = (cells & (freq <= 12 * candidate)).sum(axis=1)
            scale_quality = np.zeros(len(grid))
            if blades is not None:
                # Both weak-BPF (large shaft imbalance) and BPF-dominated
                # profiles occur. The first and BPF teeth are evaluated on
                # different physical priors, never just nearest peak spacing.
                k_bpf = slot[blades]
                strongest = np.maximum(1e-24, raw_strength[:, : min(14, orders.size)].max(axis=1))
                bpf = raw_strength[:, k_bpf] / strongest
                shaft = raw_strength[:, 0] / strongest
                overtone = raw_strength[:, slot[2 * blades]] / strongest
                second = raw_strength[:, 1 : min(12, orders.size)].max(axis=1) / strongest
                if blades > 1:
                    odd = (orders % blades != 0) & (orders <= 16)
                    non_blade_lines = (cells[:, odd] & (prom[:, odd] >= 8)).sum(axis=1)
                else:
                    non_blade_lines = np.zeros(len(grid))
                # The AVQ BPF's first overtone can be attenuated by the
                # array, while non-blade orders remain visible. Count low
                # exclusive non-blade teeth as independent scale evidence.
                blade_model = 2.6 * np.sqrt(bpf) * ((overtone >= 0.025) | (non_blade_lines >= 2))
                imbalance_model = 2.25 * shaft * (second <= 0.13) * (second >= 0.0005)
                mixed_model = 1.5 * np.sqrt(shaft) * (bpf >= 0.07) * (non_blade_lines >= 2)
                scale_quality = np.maximum(np.maximum(blade_model, imbalance_model), mixed_model)
            elif motor_orders:
                which = [slot[k] for k in motor_orders]
                anchor_count = cells[:, which].sum(axis=1)
                scale_quality = 2.2 * (anchor_count >= min(2, len(which)))
                if motor_spacing > 1:
                    scale_quality *= cells[:, 0]
            total = np.sqrt(contribution).sum(axis=1) + 0.22 * low + 1.6 * scale_quality
            usable = (n >= 3) & (scale_quality > 0.7)
            if motor_spacing > 1:
                usable &= n >= 4
            if not np.any(usable):
                continue
            idx = int(np.argmax(np.where(usable, total, -np.inf)))
            modes.append(
                {
                    "ratio": ratio,
                    "rate_rev_s": float(grid[idx]),
                    "score": float(total[idx]),
                    "orders": orders[cells[idx]].tolist(),
                    "scale_score": float(scale_quality[idx]),
                }
            )
        primary = next((mode for mode in modes if mode["ratio"] == 1), None)
        alts = [mode for mode in modes if mode["ratio"] != 1]
        alternative = max(alts, key=lambda a: a["score"], default=None)
        margin = (primary["score"] - alternative["score"]) if primary and alternative else None
        out.append(
            {"primary": primary, "alternative": alternative, "margin": margin, "modes": modes}
        )
    return out
