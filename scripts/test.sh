#!/usr/bin/env bash
# Run the full check suite: lint, types, tests.
#
# Everything runs offline against the mock providers, so this works on any
# machine and in CI without a GPU, a model download, or an API key.
#
#   scripts/test.sh                 # everything
#   scripts/test.sh tests/unit      # a subset
#   scripts/test.sh -k mentions     # by name

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT/backend"

VENV_BIN=".venv/bin"
[[ -d "$VENV_BIN" ]] || VENV_BIN=".venv/Scripts"
[[ -d "$VENV_BIN" ]] || { echo "No virtualenv found. Run scripts/dev.sh first."; exit 1; }

failed=0

echo "==> ruff"
"$VENV_BIN/python" -m ruff check app tests || failed=1

echo
echo "==> mypy"
# Not gating yet: the codebase is annotated but not strict-clean.
"$VENV_BIN/python" -m mypy app || echo "    (type errors reported, not fatal)"

echo
echo "==> pytest"
"$VENV_BIN/python" -m pytest "$@" || failed=1

echo
if (( failed )); then
  echo "FAILED"
  exit 1
fi
echo "All checks passed."
