"""``python3 -m mlops`` — the unified CLI entry point (milestone M3).

Runs ``mlops.cli.main`` so the charter's dispatch works verbatim::

    python3 -m mlops <train|evaluate|drift|retrain> [--seed N] [--batch PATH]

Exit codes are the dispatched module's API (see ``mlops/cli.py``);
usage errors exit 2.
"""

from __future__ import annotations

import sys

from mlops.cli import main

if __name__ == "__main__":
    sys.exit(main())
