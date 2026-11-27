"""The recording-condition release input manifest and source-specific classification dispatch."""

from __future__ import annotations

import json
from collections.abc import Callable
from functools import cache
from pathlib import Path
from typing import Any

from .schema import normalize

PARENT_FILE = Path(__file__).with_name("parents.json")
PARENT_VERSIONS: dict[str, str] = json.loads(PARENT_FILE.read_bytes())["parents"]
DATASETS = frozenset(PARENT_VERSIONS)

Classifier = Callable[[str, str, dict[str, Any], dict[str, Any]], dict[str, Any]]


@cache
def _classifiers() -> dict[str, Classifier]:
    from . import derived, legacy, native

    handlers: dict[str, Classifier] = {}
    for module in (derived, legacy, native):
        for name in module.DATASETS:
            if name in handlers:
                raise ValueError(f"two recording-condition rules claim {name!r}")
            handlers[name] = module.classify
    missing = DATASETS - handlers.keys()
    extra = handlers.keys() - DATASETS
    if missing or extra:
        raise ValueError(
            f"recording-condition rule scope mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
        )
    return handlers


def classify(dataset: str, key: str, meta: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """Classify the exact published sample. No waveform or remote operations in this path."""
    if dataset not in PARENT_VERSIONS:
        raise KeyError(f"{dataset!r} is outside the frozen recording inventory")
    if info.get("version") != PARENT_VERSIONS[dataset]:
        raise ValueError(f"{dataset}/{key}: source version does not match the annotation inventory")
    if not isinstance(info.get("has_audio"), bool):
        raise ValueError(f"{dataset}/{key}: has_audio must reflect the inspected source fields")
    facts = _classifiers()[dataset](dataset, key, meta, info)
    return normalize(facts, has_audio=info["has_audio"])
