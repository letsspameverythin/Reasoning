#!/usr/bin/env python
"""Print or run seed-expansion configs in priority order (P0 then P1).

Usage
-----
  python scripts/run_seed_expansion_queue.py --dry-run
  python scripts/run_seed_expansion_queue.py --only p0 --dry-run
  python scripts/run_seed_expansion_queue.py --only p1

Does not start GPU work unless ``--execute`` is passed. Prefer ``--dry-run``
locally; on Kaggle use the cells in ``notebooks/seed_expansion.md``.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUEUE = ROOT / "configs" / "seed_expansion" / "queue.json"


def load_queue(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise SystemExit(
            f"missing {path}; run: python scripts/generate_seed_expansion_configs.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def filter_runs(
    runs: List[Dict[str, Any]],
    only: Optional[str],
    stems: Optional[List[str]],
) -> List[Dict[str, Any]]:
    out = runs
    if only:
        out = [r for r in out if r["priority"] == only]
    if stems:
        want = set(stems)
        out = [r for r in out if r["stem"] in want]
    return out


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--queue",
        type=Path,
        default=DEFAULT_QUEUE,
        help="Path to queue.json (default: configs/seed_expansion/queue.json)",
    )
    p.add_argument(
        "--only",
        choices=("p0", "p1"),
        default=None,
        help="Restrict to P0 or P1 runs",
    )
    p.add_argument(
        "--stem",
        action="append",
        default=None,
        help="Run only this stem (repeatable), e.g. qwen1.5b__gsm_symbolic_main__c0__s1",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands only (default if --execute is not set)",
    )
    p.add_argument(
        "--execute",
        action="store_true",
        help="Actually run python -m src.runner for each config (GPU)",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the queue on the first non-zero exit (execute mode)",
    )
    args = p.parse_args()

    if not args.execute:
        args.dry_run = True

    payload = load_queue(args.queue)
    runs = filter_runs(payload["runs"], args.only, args.stem)
    if not runs:
        print("no runs matched filters", file=sys.stderr)
        return 1

    est = sum(float(r["est_gpu_hours"]) for r in runs)
    print(
        f"seed-expansion queue: {len(runs)} runs "
        f"(filter only={args.only or 'all'}) ~{est:.1f} GPU-h estimated"
    )
    print(f"source: {args.queue}")
    print()

    failures = 0
    for r in runs:
        cmd = r["command"]
        tag = f"[{r['priority'].upper()} #{r['order']:02d}]"
        print(f"{tag} ~{r['est_gpu_hours']:.2f}h  {cmd}")
        if args.execute:
            print(f"  -> executing ...", flush=True)
            proc = subprocess.run(cmd, shell=True, cwd=str(ROOT))
            if proc.returncode != 0:
                failures += 1
                print(f"  !! exit {proc.returncode}", file=sys.stderr)
                if args.stop_on_error:
                    return proc.returncode
    if args.dry_run and not args.execute:
        print()
        print("dry-run only; pass --execute to launch (requires GPU).")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
