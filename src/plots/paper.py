"""House style of the paper figures (IEEEtran, two columns).

Every matplotlib figure that goes into a paper is drawn at its PRINT size
(``COLUMN_IN`` or ``TEXT_IN`` wide) and included at that width, so the font
sizes below are the sizes on the page. They match the TikZ figures of the
wrap-up paper as printed (measured 2026-10-26 with PyMuPDF on the built PDF:
5.8-7.7 pt, most text 6.6-7.5 pt) and their face (TeX Gyre Termes, the
newtxtext Times of the body).

    from plots.paper import COLUMN_IN, paper_style

    with paper_style():
        fig, ax = plt.subplots(figsize=(COLUMN_IN, 2.2), layout="constrained")
        ...
        fig.savefig("x.pdf")
"""

from __future__ import annotations

import contextlib
from collections.abc import Iterator

import matplotlib as mpl

__all__ = ["COLUMN_IN", "PAPER_RC", "TEXT_IN", "paper_style"]

#: IEEEtran column and text widths (inches).
COLUMN_IN = 3.5
TEXT_IN = 7.16

#: The rcParams of a paper figure.
PAPER_RC: dict[str, object] = {
    "font.family": "serif",
    "font.serif": [
        "TeX Gyre Termes",
        "Nimbus Roman",
        "Times New Roman",
        "STIXGeneral",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "stix",
    "font.size": 7.0,
    "axes.titlesize": 7.0,
    "axes.labelsize": 7.0,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "legend.fontsize": 6.5,
    "figure.titlesize": 7.0,
    "axes.linewidth": 0.5,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "lines.linewidth": 0.8,
    "grid.linewidth": 0.4,
    "grid.alpha": 0.3,
    "legend.frameon": False,
    "pdf.fonttype": 42,
    "savefig.dpi": 300,
}


@contextlib.contextmanager
def paper_style() -> Iterator[None]:
    """Draw inside :data:`PAPER_RC`; the caller's rcParams come back after."""
    with mpl.rc_context(PAPER_RC):
        yield
