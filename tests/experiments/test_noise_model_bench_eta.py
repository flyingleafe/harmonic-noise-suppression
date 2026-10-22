"""The saddle escape of the v4 bench fit (:func:`bench_eta.saddle_escape`) on a
tiny quadratic-plus-quartic objective with a planted saddle at the origin.

The toy stands in for :class:`bench_v4.RigModel` (``params``, ``pinned``,
``frozen``, ``whittle``, ``neg_log_prior``, ``objective``); its minima are at
``x1^2 = 0.5 / 0.41``, ``x0 = -0.3 x1^2``, ``x2 = 0``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from experiments.noise_model import bench_eta as E


class Toy:
    def __init__(self, x: list[float]):
        self.params = {"x": torch.tensor(x, dtype=torch.float64, requires_grad=True)}
        self.pinned: tuple[str, ...] = ()
        self.frozen = [None]

    def whittle(self, j: int) -> torch.Tensor:
        x0, x1, _ = self.params["x"]
        return 0.5 * x0**2 - 0.5 * x1**2 + 0.25 * x1**4 + 0.3 * x0 * x1**2

    def neg_log_prior(self) -> torch.Tensor:
        return 0.5 * self.params["x"][2] ** 2

    @torch.no_grad()
    def objective(self) -> tuple[float, list[float]]:
        parts = [float(self.whittle(0))]
        return float(self.neg_log_prior()) + sum(parts), parts


def lbfgs(model: Toy, budget_s: float | None) -> dict[str, Any]:
    x = model.params["x"]
    opt = torch.optim.LBFGS([x], max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        f = model.whittle(0) + model.neg_log_prior()
        f.backward()
        return f

    opt.step(closure)
    return dict(objective=model.objective()[0])


def test_escape_leaves_planted_saddle_for_a_minimum() -> None:
    model = Toy([0.0, 0.0, 0.0])
    # the origin is stationary: the convergence loop alone stays on the saddle
    assert lbfgs(model, None)["objective"] == 0.0
    rec, hp = E.saddle_escape(model, lbfgs, log=lambda *_: None)
    assert rec["stop"] == "at_minimum" and rec["at_minimum"] is True
    assert len(rec["rounds"]) == 1
    rnd = rec["rounds"][0]
    assert rnd["n_negative"] == 1 and np.isclose(rnd["min_eig"], -1.0)
    (d,) = rnd["directions"]
    assert d["param"] == "x" and d["index"] == 1 and abs(d["step"]) == 1.0
    assert np.isclose(d["gain"], 0.25)  # -0.5 t^2 + 0.25 t^4 at t = 1
    x = model.params["x"].detach().numpy()
    x1 = np.sqrt(0.5 / 0.41)
    assert np.allclose(np.abs(x), [0.3 * x1**2, x1, 0.0], atol=1e-6)
    # the returned Hessian is the one at the final parameters
    assert hp is not None
    w, _, _ = E.curvature(*hp)
    assert w.min() > 0.0 and rec["n_negative"] == 0


def test_line_search_without_descent_leaves_the_model() -> None:
    x1 = np.sqrt(0.5 / 0.41)
    model = Toy([-0.3 * x1**2, x1, 0.0])
    f0, _ = model.objective()
    before = model.params["x"].detach().clone()
    out = E.line_search(model, np.array([0.0, 0.6, 0.8]), f0)
    assert out["step"] == 0.0 and out["gain"] == 0.0
    assert torch.equal(model.params["x"].detach(), before)
