#!/usr/bin/env python
"""Run decisive C3 sample-count / seed-substitution controls."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.corpus import load_corpus
from src.analysis import c3_controls as ctrl
from src.utils import setup_logging


def _write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys()) if "contrast" in rows[0] else sorted({k for r in rows for k in r})
    # Prefer stable column order for the summary table.
    if "contrast" in rows[0]:
        keys = ["contrast", "n_cells", "mean_pp", "ci_low_pp", "ci_high_pp", "cells"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results_fp16")
    p.add_argument("--out", default="paper_artefacts/tab_c3_controls.csv")
    p.add_argument(
        "--summary-out", default="paper_artefacts/tab_c3_controls_summary.csv"
    )
    p.add_argument("--n-bootstrap", type=int, default=ctrl.DEFAULT_CELL_BOOTSTRAP)
    p.add_argument("--boot-seed", type=int, default=0)
    args = p.parse_args()
    setup_logging()
    corpus = load_corpus(args.results_dir)
    rows = ctrl.evaluate_controls_corpus(corpus)
    summary = ctrl.controls_summary(
        rows, n_bootstrap=args.n_bootstrap, seed=args.boot_seed
    )
    summary_table = ctrl.controls_summary_table(
        rows, n_bootstrap=args.n_bootstrap, seed=args.boot_seed
    )
    out = Path(args.out)
    _write_csv(out, rows)
    _write_csv(Path(args.summary_out), summary_table)
    print(f"wrote {len(rows)} rows -> {out}")
    print(f"wrote {len(summary_table)} contrasts -> {args.summary_out}")
    for k, v in summary.items():
        if isinstance(v, float):
            if k.endswith("_pp"):
                print(f"  {k}: {v:.4f}")
            elif "delta" in k or k.startswith("mean_delta"):
                print(f"  {k}: {100 * v:.2f}pp")
            else:
                print(f"  {k}: {v}")
        else:
            print(f"  {k}: {v}")
    print("--- per cell (pp) ---")
    for r in rows:
        m = r["model"].split("/")[-1]
        line = (
            f"  {r['dataset']}/{r.get('subset') or '-'} {m}: "
            f"cd_cfg={100*float(r.get('delta_cd_configs_minus_sc') or 0):+.1f} "
            f"mix={100*float(r.get('delta_mixture_minus_sc') or 0):+.1f} "
            f"seeds={r.get('c0_seeds', '')}"
        )
        if r.get("has_seed_control"):
            line += (
                f" | seed_cd={100*float(r.get('delta_cd_seeds_minus_sc') or 0):+.1f} "
                f"sc3n={100*float(r.get('delta_sc_3n_seeds_minus_sc') or 0):+.1f} "
                f"cfg-seed={100*float(r.get('delta_cd_configs_minus_cd_seeds') or 0):+.1f} "
                f"cfg-sc3n={100*float(r.get('delta_cd_configs_minus_sc_3n') or 0):+.1f}"
            )
        print(line)
    print("VERDICT:", summary.get("verdict"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
