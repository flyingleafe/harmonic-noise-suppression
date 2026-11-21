"""The partial scipy stub package under ``typings/`` stays a complete overlay.

pyright 1.1.407 overlays a partial stub package by redirecting EVERY ``.pyi``
path under the installed package to it, so each stub scipy ships must be linked
back from ``typings/scipy-stubs``, or pyright loses that module's types; and
the one stub this repository adds must declare exactly what scipy exports, or
a window function scipy gains becomes an unknown import. Why the package
exists at all: the header of ``typings/scipy-stubs/signal/windows/_windows.pyi``.
"""

from __future__ import annotations

import re
from pathlib import Path

import scipy
from scipy.signal.windows import _windows

STUBS = Path(__file__).resolve().parents[1] / "typings" / "scipy-stubs"
#: The stubs this repository writes; every other ``.pyi`` in the package is a link.
OWN = {"signal/windows/_windows.pyi"}


def test_stub_package_links_back_every_stub_scipy_ships() -> None:
    package = Path(scipy.__file__).resolve().parent
    shipped = {p.relative_to(package).as_posix() for p in package.rglob("*.pyi")}
    linked = {p.relative_to(STUBS).as_posix() for p in STUBS.rglob("*.pyi")} - OWN
    assert linked == shipped, (
        "typings/scipy-stubs must hold one link per stub of the installed scipy, at the same "
        "relative path, to .venv/lib/python3.12/site-packages/scipy/<path>"
    )
    for rel in sorted(shipped):
        assert (STUBS / rel).resolve() == package / rel, rel


def test_window_stub_declares_exactly_what_scipy_exports() -> None:
    text = (STUBS / "signal" / "windows" / "_windows.pyi").read_text()
    declared = set(re.findall(r"^def (\w+)\(", text, flags=re.MULTILINE))
    assert declared == set(_windows.__all__)
    listed = set(re.findall(r'^    "(\w+)",$', text, flags=re.MULTILINE))
    assert listed == declared
