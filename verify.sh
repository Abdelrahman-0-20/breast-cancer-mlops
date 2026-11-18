#!/usr/bin/env bash
# breast-cancer-mlops gates — exit 0 required at every milestone.
set -euo pipefail

echo "-- compileall"
python3 -m compileall -q mlops tests

if command -v ruff >/dev/null 2>&1; then
  echo "-- ruff"
  ruff check mlops tests
else
  echo "-- ruff: SKIPPED (not installed)"
fi

echo "-- pytest"
python3 -m pytest -q

echo "gates green"
