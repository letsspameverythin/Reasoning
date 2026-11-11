#!/usr/bin/env python
"""Fast Option-B C3 table: pre-specified policy + multi-split + seed null."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.corpus import load_corpus
from src.analysis import selection as selection_mod
from src.utils import setup_logging


def _write(rows, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results_fp16")
    p.add_argument("--out", default="paper_artefacts/tab_selection_oracle_gap.csv")
    p.add_argument("--null-out", default="paper_artefacts/tab_seed_null_oracle.csv")
    p.add_argument("--n-bootstrap", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-multisplit", action="store_true")
    args = p.parse_args()
    setup_logging()
    corpus = load_corpus(args.results_dir)
    rows = selection_mod.evaluate_corpus(
        corpus,
        seed=args.seed,
        n_bootstrap=args.n_bootstrap,
        multisplit=not args.no_multisplit,
    )
    summary = selection_mod.selection_summary(rows)
    null_rows = selection_mod.seed_null_oracle_gap(corpus)
    _write(rows, Path(args.out))
    _write(null_rows, Path(args.null_out))
    print(f"wrote {len(rows)} selection rows -> {args.out}")
    print(f"wrote {len(null_rows)} seed-null rows -> {args.null_out}")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    if null_rows:
        gaps = [float(r["gap_seed_oracle_minus_seed0"]) for r in null_rows]
        print(
            f"  seed_null mean gap={100*sum(gaps)/len(gaps):.2f}pp "
            f"over {len(null_rows)} cells"
        )
    for r in rows:
        m = r["model"].split("/")[-1]
        print(
            f"  {r['dataset']}/{r.get('subset') or '-'} | {m}: "
            f"SC={100*float(r['acc_sc']):.1f} ora={100*float(r['acc_oracle_config']):.1f} "
            f"gap={100*float(r['gap_oracle_config_minus_sc']):.1f} | "
            f"primary={r.get('primary_policy')} "
            f"d={100*float(r.get('primary_delta') or 0):.1f}pp "
            f"frac={float(r.get('primary_frac_oracle_gap') or 0):.2f} "
            f"vs_val={100*float(r.get('primary_delta_vs_val_selected') or 0):+.1f}pp"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
