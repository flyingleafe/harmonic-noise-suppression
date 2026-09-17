"""The round-1 noise-model-v2 fit driver's reductions over fit JSONs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import noise_v2_fit as NVF
import numpy as np
import pytest


def _fit(
    support: str,
    *,
    k_max: int,
    profile_db: float,
    loss: float,
    seed: int = 0,
    dyn: float = 1.0,
) -> dict[str, Any]:
    """A minimal ``noise-v2-fit/1`` payload: the fields the reductions read."""
    return dict(
        schema="noise-v2-fit/1",
        support=support,
        kind="bench",
        mode="bench",
        n_rotors=1,
        k_max=k_max,
        params=dict(
            sigma_nu=0.45 * dyn,
            lam=5.5 * dyn,
            sigma_eps_even=0.3 * dyn,
            sigma_eps_odd=0.3 * dyn,
            lam_eps_even=2.0 * dyn,
            lam_eps_odd=2.0 * dyn,
            carrier_rev_s=[70.0],
            profile=dict(
                profile_db=[[profile_db] * k_max],
                amp_exp=0.0,
                mic_line_gain_db=[[0.0]],
            ),
        ),
        objective=dict(whittle_nats=loss, n_cells=1000, per_band=dict(floor=0.0, comb=loss)),
        optimiser=dict(lbfgs_loss_after=loss, seed=seed, converged=False),
    )


def test_mean_comb_averages_each_order_over_the_fits_that_reach_it(tmp_path: Path):
    """Bench supports cap orders at their OWN Nyquist, so the frozen comb must
    survive fits of different widths — the DREGON floor fit died on exactly
    this (four Motor*_70 fits 115-118 orders wide)."""
    paths = []
    for i, (k_max, db) in enumerate(((4, -10.0), (6, -20.0), (5, -30.0))):
        p = tmp_path / f"m{i}.json"
        p.write_text(json.dumps(_fit(f"m{i}", k_max=k_max, profile_db=db, loss=-1.0)))
        paths.append(str(p))

    frozen, meta = NVF.mean_comb(paths)
    prof = np.asarray(frozen["profile_db"], dtype=np.float64)

    assert prof.shape == (1, 6), "the frozen comb is as wide as the widest fit"
    assert meta["profile_orders_per_fit"] == [4, 6, 5]
    # orders 1-4 are in all three, order 5 in two, order 6 in one
    assert prof[0, :4] == pytest.approx(-20.0)
    assert prof[0, 4] == pytest.approx(-25.0)
    assert prof[0, 5] == pytest.approx(-20.0)
    assert not np.isnan(prof).any()
    # a one-rotor comb frozen onto a four-rotor airframe keeps its width
    assert np.asarray(NVF.expand_frozen(frozen, n_rotors=4)["profile_db"]).shape == (4, 6)


def test_reduce_restarts_reports_the_best_start_and_the_spread(tmp_path: Path):
    """The reported fit is the best restart, and the block quantifies how far
    apart the restarts landed — the evidence that a support was sampled rather
    than fitted."""
    rdir = tmp_path / "restarts"
    rdir.mkdir()
    for seed, (loss, dyn) in enumerate(((-100.0, 1.0), (-160.0, 2.0), (-120.0, 4.0))):
        (rdir / f"s__bench__s{seed}.json").write_text(
            json.dumps(_fit("s", k_max=4, profile_db=-10.0, loss=loss, seed=seed, dyn=dyn))
        )

    rows = NVF.reduce_restarts(tmp_path)
    assert [r["support"] for r in rows] == ["s"]

    payload = json.loads((tmp_path / "s__bench.json").read_text())
    block = payload["restarts"]
    assert block["selected_seed"] == 1, "the lowest polished objective wins"
    assert payload["objective"]["whittle_nats"] == pytest.approx(-160.0)
    assert block["n_restarts"] == 3
    # median -120, best -160 over 1000 cells
    assert block["best_minus_median_per_cell"] == pytest.approx(0.04)
    assert block["best_minus_worst_per_cell"] == pytest.approx(0.06)
    assert block["params"]["lam"]["max_over_min"] == pytest.approx(4.0)
    assert block["params"]["lam"]["median"] == pytest.approx(11.0)
