#!/usr/bin/env bash
# Print or run the seed-expansion queue (Linux/Kaggle helper).
# Usage:
#   bash scripts/run_seed_expansion_queue.sh --dry-run
#   bash scripts/run_seed_expansion_queue.sh --only p0 --dry-run
#   bash scripts/run_seed_expansion_queue.sh --only p0 --execute   # GPU

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
exec python scripts/run_seed_expansion_queue.py "$@"
