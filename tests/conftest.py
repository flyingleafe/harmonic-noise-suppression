"""Repository-wide test configuration.

CPU cap (user rule, 2026-09-06): the CRF and comb-slot tests are CPU-bound
and torch takes every core by default. Leave half the cores free.
``HNS_TEST_THREADS`` overrides the cap.

``dload_tree``: real raw trees (``dload:DREGON``, ...) for tests that read
recorded data; the test skips when the tree cannot be resolved.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path

import pytest

_cap = int(os.environ.get("HNS_TEST_THREADS", max(1, (os.cpu_count() or 2) // 2)))
for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, str(_cap))
try:
    import torch

    torch.set_num_threads(_cap)
except Exception:  # torch absent in a docs-only environment
    pass


@pytest.fixture(scope="session")
def dload_tree() -> Callable[[str], Path]:
    """``resolve(name)`` -> the materialized ``dload:<name>`` tree.

    Resolution goes through ``streams.resolve_source`` (cache first, R2 pull on
    a miss), once per name per session. The calling test skips when it fails —
    no credentials, remote unreachable — with the resolver's error as reason.
    """
    resolved: dict[str, Path | Exception] = {}

    def resolve(name: str) -> Path:
        if name not in resolved:
            from data_processing.streams import resolve_source

            try:
                resolved[name] = resolve_source(f"dload:{name}")
            except Exception as exc:  # noqa: BLE001 — any failure means "unavailable"
                resolved[name] = exc
        hit = resolved[name]
        if isinstance(hit, Exception):
            pytest.skip(f"dload:{name} unavailable: {hit!r}")
        return hit

    return resolve
