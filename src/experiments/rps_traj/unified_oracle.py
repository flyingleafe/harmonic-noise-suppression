"""Post-hoc *non-generative benchmark* diagnostic for failed free-running fits.

Reveal the observed recording only to the latent-anchor / hidden-action
posterior, then simulate the response WITHOUT teacher forcing and with fresh
innovations. This is not an admissible competitor on the fixed-schedule task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from numba import njit
from scipy.special import logsumexp

from experiments.rps_traj import unified_grammar as ug
from experiments.rps_traj import unified_response as ur
from experiments.rps_traj.schedule_score import score_phase
from experiments.rps_traj.unified_experiment import _metric
from tracking.rotors import MIXER


@njit(cache=True)
def _backdraw(alpha, trans, phase, u):
    """Sample an expanded-state path backwards from filtered log probabilities."""
    n, T = u.shape
    S = trans.shape[0]
    draws = np.empty((n, T), dtype=np.int64)
    weights = np.empty(S)
    for i in range(n):
        successor = 0
        for t in range(T - 1, -1, -1):
            reset = t == T - 1 or phase[t] != phase[t + 1]
            for k in range(S):
                weights[k] = alpha[t, k] + (0.0 if reset else trans[k, successor])
            largest = np.max(weights)
            if not np.isfinite(largest):
                raise ValueError("posterior cannot reach sampled successor")
            total = 0.0
            for k in range(S):
                total += np.exp(weights[k] - largest)
                weights[k] = total
            successor = np.searchsorted(
                weights, min(u[i, t] * total, np.nextafter(total, 0.0)), side="right"
            )
            draws[i, t] = successor
    return draws


def sample_diagnostic(model, speed, phases, seed, n):
    f = ur._prepare(speed, phases)
    c = model.grammar.compiled
    targets = model.params.targets(model.meta)
    la, ll = [], []
    for le in ur._emission_tables(f, model.params, targets):
        alpha = np.empty((len(phases), model.grammar.n_states))
        scale = np.empty(len(phases))
        ll.append(
            ug._forward(
                le,
                phases,
                c.state_label,
                c.row_ptr,
                c.col,
                c.log_prob,
                c.log_init,
                c.mask,
                alpha,
                scale,
                True,
            )
        )
        la.append(alpha)
    weight = model.params.log_pi + np.array(ll)
    prob = np.exp(weight - logsumexp(weight))
    rng = np.random.default_rng(np.random.SeedSequence(seed))
    j = rng.choice(len(prob), n, p=prob)
    uniforms = rng.random((n, len(phases)))
    eps = rng.standard_normal((n, len(phases), 4))
    ix = ur._response_index(model.params)
    F, K = np.atleast_2d(model.params.F)[ix], np.atleast_2d(model.params.K)[ix]
    x = np.empty_like(eps)
    # Use exactly the same forward graph uniforms and innovations as the
    # free run for anchor-only; the oracle path uses the same uniforms in reverse time.
    lab = np.empty(uniforms.shape, dtype=np.int64)
    ug._sample(phases, uniforms, c.state_label, c.row_ptr, c.col, c.prob, c.init, lab)
    ur._simulate(lab, j, targets, model.params.sigma, F, K, eps, x)
    anchor_only = np.einsum("rd,ntd->nrt", MIXER, x)
    trans = np.full((model.grammar.n_states, model.grammar.n_states), -np.inf)
    for k in range(model.grammar.n_states):
        a, b = c.row_ptr[k : k + 2]
        trans[k, c.col[a:b]] = c.log_prob[a:b]
    for k in range(len(prob)):
        mask = j == k
        if mask.any():
            lab[mask] = c.state_label[_backdraw(la[k], trans, phases, uniforms[mask])]
    ur._simulate(lab, j, targets, model.params.sigma, F, K, eps, x)
    return {"anchor_only": anchor_only, "anchor_and_actions": np.einsum("rd,ntd->nrt", MIXER, x)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", type=Path, default=Path("results/rps_traj/unified"))
    out = p.parse_args().out
    pack = out / "pack"
    manifest = json.loads((pack / "manifest.json").read_text())
    rows = []
    for key in manifest["folds"]:
        meta = json.loads((pack / f"{key}.json").read_text())
        fit = json.loads((out / "fits" / f"{key}__hierarchical.json").read_text())
        model = ur.UnifiedResponseModel.from_dict(fit["params"])
        metric = _metric(manifest, meta["rig"])
        with np.load(pack / f"{key}.eval.npz") as z:
            for target in meta["targets"]:
                f = target["flight"]
                arms = sample_diagnostic(model, z[f + "__x"], z[f + "__phase"], target["seed"], 32)
                for cell in target["cells"]:
                    phase = cell["phase"]
                    k = f"{f}__{phase}"
                    for arm, draws in arms.items():
                        scores = [
                            score_phase(
                                phase,
                                obs,
                                refs,
                                draws[:, :, i : i + cell["n"]],
                                cell["hover"],
                                metric,
                            )["score"]
                            for obs, refs, i in zip(
                                z[k + "__obs"], z[k + "__refs"], cell["starts"], strict=True
                            )
                        ]
                        rows.append(
                            dict(
                                arm=arm,
                                phase=phase,
                                rig=meta["rig"],
                                flight=f,
                                score=float(np.mean(scores)),
                            )
                        )
    balanced = {
        arm: {
            p: float(
                np.mean(
                    np.asarray(
                        [
                            np.mean(
                                [
                                    r["score"]
                                    for r in rows
                                    if (r["arm"], r["phase"], r["rig"]) == (arm, p, rig)
                                ]
                            )
                            for rig in manifest["protocol"]["rigs"]
                        ]
                    )
                )
            )
            for p in manifest["protocol"]["phases"]
        }
        for arm in ("anchor_only", "anchor_and_actions")
    }
    (out / "oracle_diagnostic.json").write_text(
        json.dumps(
            {"oracle_not_free_running": True, "phase_scores": balanced, "rows": rows},
            indent=1,
            allow_nan=False,
        )
        + "\n"
    )
    print(balanced)


if __name__ == "__main__":
    main()
