#!/usr/bin/env bash
# pytest collects both unittest.TestCase and function-style tests.
set -euo pipefail
repo="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo"
exec python3 -B -m pytest "$@"
