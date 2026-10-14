"""Probe of the trained HPPNet-L2 super-resolution head.

Source of the numbers in docs/experiments/paper-real-protocol.md, ladder
section ("what the old head does"). Checkpoint ``hppnet_l2_r2_s0`` /
``best.ckpt``, the frozen real split at mic 0, rotors-on frames
(rotor-mean target > 30 rev/s):

* PIT-MAE of the full model vs the model with the head's conv stack removed
  (the fixed interpolation W only, same CRF readout);
* trunk-only readout: argmax of each log-grid map converted to rev/s;
* peak shape of the trunk maps and of the head output (sigmoid peak,
  half-max width);
* cross-map dependence: shift of one rotor's output when the other three
  trunk maps are flattened to their time mean;
* impulse response: one log bin at +6, the rest at -6, through W and the net.

Run from the project root: ``python scripts/_probe_l2_head.py``.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from torch import nn

import zoo
from experiments import rps_bench as rb
from metrics._common import get_array
from models.salience_rps import FreqSuperResHead

EXPERIMENT, CKPT = "hppnet_l2_r2_s0", "best"
ON_REV_S = 30.0


def pit(pred: np.ndarray, ref: np.ndarray) -> np.ndarray:
    errs = []
    for t in range(ref.shape[1]):
        c = np.abs(pred[:, None, t] - ref[None, :, t])
        r, cc = linear_sum_assignment(c)
        errs.append(c[r, cc].mean())
    return np.asarray(errs)


def main() -> None:
    torch.set_grad_enabled(False)
    fm = zoo.load(EXPERIMENT, ckpt=CKPT)
    m = fm.model.eval()
    head = cast(FreqSuperResHead, m.superres_head)
    W, net = cast(torch.Tensor, head.W), head.net
    n_in, n_out = int(W.shape[1]), int(W.shape[0])
    logf = 27.5 * 2 ** (np.arange(n_in) / 48)
    linf = np.linspace(0, 150, n_out)
    frames = rb.part("real")
    cap: dict[str, torch.Tensor] = {}

    def _capture(_mod: nn.Module, inputs: tuple[torch.Tensor, ...], out: torch.Tensor) -> None:
        cap["x"], cap["y"] = inputs[0], out

    head.register_forward_hook(_capture)
    net_identity = nn.Sequential(nn.Identity())
    decode = cast(Any, m.decode_logits)

    full_err, interp_err, trunk_err = [], [], []
    wid_x, wid_y, peak_x, peak_y, floor_x, mix_shift = [], [], [], [], [], []
    for i in range(0, len(frames), 8):  # mic 0 of every clip
        f = frames[i]
        x = torch.as_tensor(np.asarray(get_array(f, "mixture"))[None, :], dtype=torch.float32)
        ref = np.asarray(f["rps"].data, float)
        n = x.shape[-1]

        head.net = net
        tr = decode(m(x), n)[0].numpy()
        xin, y = cap["x"][0], cap["y"][0]  # (4, F_in, T), (4, F_out, T)
        head.net = net_identity
        tr_int = decode(m(x), n)[0].numpy()
        head.net = net

        T = min(ref.shape[1], tr.shape[1])
        on = ref[:, :T].mean(0) > ON_REV_S
        if on.sum() == 0:
            continue
        full_err.append(pit(tr[:, :T], ref[:, :T])[on])
        interp_err.append(pit(tr_int[:, :T], ref[:, :T])[on])

        px = torch.sigmoid(xin)[:, :, :T]
        trunk_err.append(pit(logf[px.argmax(1).numpy()], ref[:, :T])[on])
        py = torch.sigmoid(y)[:, :, :T]
        for p, wl, pl in ((px, wid_x, peak_x), (py, wid_y, peak_y)):
            pk = p.max(1).values
            half = (p >= 0.5 * pk[:, None, :]).sum(1).float()
            wl.append(half[:, on].mean().item())
            pl.append(pk[:, on].mean().item())
        floor_x.append(px[:, :, on].median().item())

        base = torch.sigmoid(net(torch.einsum("oi,mit->mot", W, xin)[None]))[0].argmax(1)[:, :T]
        shifts = []
        for r in range(4):
            x2 = xin.clone()
            for q in range(4):
                if q != r:
                    x2[q] = xin[q].mean(-1, keepdim=True)
            am2 = torch.sigmoid(net(torch.einsum("oi,mit->mot", W, x2)[None]))[0].argmax(1)[r, :T]
            shifts.append((am2 - base[r]).abs().float()[on].mean().item() * (linf[1] - linf[0]))
        mix_shift.append(np.mean(shifts))

    print(
        f"PIT-MAE rotors on: full head {np.concatenate(full_err).mean():.2f} | "
        f"W only {np.concatenate(interp_err).mean():.2f} | "
        f"trunk argmax {np.concatenate(trunk_err).mean():.2f} rev/s"
    )
    print(
        f"trunk map: peak {np.mean(peak_x):.2f}, half-max width {np.mean(wid_x):.1f} log bins, "
        f"median {np.mean(floor_x):.3f} | head output: peak {np.mean(peak_y):.2f}, "
        f"width {np.mean(wid_y):.1f} bins ({linf[1] - linf[0]:.2f} rev/s)"
    )
    print(
        f"cross-map dependence: |argmax shift| with the other maps flattened {np.mean(mix_shift):.2f} rev/s"
    )
    for fb in (60, 74, 86):
        xi = torch.full((1, 4, n_in, 8), -6.0)
        xi[0, :, fb, :] = 6.0
        yi = torch.einsum("oi,bit->bot", W, xi.reshape(-1, n_in, 8)).reshape(1, 4, n_out, 8)
        pi = torch.sigmoid(yi)[0, 0, :, 4]
        po = torch.sigmoid(net(yi))[0, 0, :, 4]
        print(
            f"impulse at {logf[fb]:.1f} rev/s: W argmax {linf[pi.argmax()]:.1f} | net argmax "
            f"{linf[po.argmax()]:.1f}, peak {po.max():.2f}, width {(po >= 0.5 * po.max()).sum().item()} bins"
        )


if __name__ == "__main__":
    main()
