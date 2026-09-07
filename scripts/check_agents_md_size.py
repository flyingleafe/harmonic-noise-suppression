#!/usr/bin/env python
"""Pre-commit hook: every AGENTS.md stays a map (<= LIMIT bytes).

Per-directory AGENTS.md files are loaded into agent context on every visit;
narrative, history, results tables and tutorials belong in docs/ (root
AGENTS.md rule 5). Fails listing each offending file and its size.
"""

from __future__ import annotations

import sys
from pathlib import Path

LIMIT = 10 * 1024


def main(paths: list[str]) -> int:
    over = [(p, Path(p).stat().st_size) for p in paths if Path(p).stat().st_size > LIMIT]
    for p, size in over:
        print(f"{p}: {size} bytes > {LIMIT} — AGENTS.md is a map; move narrative to docs/")
    return 1 if over else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
