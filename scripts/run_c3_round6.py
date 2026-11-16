#!/usr/bin/env python
"""Round-6 CPU analyses after the 14-cell seed draft (no GPU).

Writes:
  - tab_c3_controls.csv / tab_c3_controls_summary.csv (incl. seed_minus_sc3n)
  - tab_c3_per_cell_contrasts.csv
  - tab_c3_drop_c2_twoseed.csv (+ summary rows in same file or sibling)
  - tab_c3_gate_ablation.csv
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.corpus import load_corpus
from src.analysis import c3_controls as ctrl
from src.utils import setup_logging


def _write_csv(path: Path, rows: list, preferred: list | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    all_keys = sorted({k for r in rows for k in r})
    if preferred:
        # Keep preferred order, then any remaining keys from the union.
        keys = list(dict.fromkeys([*preferred, *all_keys]))
    elif "contrast" in all_keys:
        keys = list(
            dict.fromkeys(
                [
                    "contrast",
                    "n_cells",
                    "mean_pp",
                    "ci_low_pp",
                    "ci_high_pp",
                    *all_keys,
                ]
            )
        )
    else:
        keys = all_keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _pp(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:+.2f}"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results_fp16")
    p.add_argument("--out-dir", default="paper_artefacts")
    p.add_argument("--n-bootstrap", type=int, default=ctrl.DEFAULT_CELL_BOOTSTRAP)
    p.add_argument("--boot-seed", type=int, default=0)
    args = p.parse_args()
    setup_logging()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("loading corpus…")
    corpus = load_corpus(args.results_dir)

    print("=== C3 controls (14-cell seed grid) ===")
    rows = ctrl.evaluate_controls_corpus(corpus)
    summary = ctrl.controls_summary(
        rows, n_bootstrap=args.n_bootstrap, seed=args.boot_seed
    )
    summary_table = ctrl.controls_summary_table(
        rows, n_bootstrap=args.n_bootstrap, seed=args.boot_seed
    )
    per_cell = ctrl.per_cell_contrasts_table(rows)
    _write_csv(out_dir / "tab_c3_controls.csv", rows)
    _write_csv(
        out_dir / "tab_c3_controls_summary.csv",
        summary_table,
        preferred=["contrast", "n_cells", "mean_pp", "ci_low_pp", "ci_high_pp", "cells"],
    )
    _write_csv(
        out_dir / "tab_c3_per_cell_contrasts.csv",
        per_cell,
        preferred=[
            "cell",
            "dataset",
            "subset",
            "model",
            "has_seed_control",
            "n_c0_seeds",
            "c0_seeds",
            "cd_minus_sc_pp",
            "seed_minus_sc_pp",
            "sc3n_minus_sc_pp",
            "cd_minus_seed_pp",
            "cd_minus_sc3n_pp",
            "seed_minus_sc3n_pp",
        ],
    )
    print(f"wrote {len(rows)} cells -> tab_c3_controls.csv")
    print(f"wrote {len(summary_table)} contrasts -> tab_c3_controls_summary.csv")
    print(f"wrote {len(per_cell)} rows -> tab_c3_per_cell_contrasts.csv")
    for r in summary_table:
        print(
            f"  {r['contrast']}: {_pp(float(r['mean_pp']))} "
            f"[{_pp(float(r['ci_low_pp']))}, {_pp(float(r['ci_high_pp']))}] "
            f"(n={r['n_cells']})"
        )
    print("VERDICT:", summary.get("verdict"))

    print("=== gate ablation (ungated mode-of-modes vs gated CD) ===")
    gate = ctrl.gate_ablation_table(
        rows, n_bootstrap=args.n_bootstrap, seed=args.boot_seed
    )
    _write_csv(
        out_dir / "tab_c3_gate_ablation.csv",
        gate,
        preferred=[
            "contrast",
            "n_cells",
            "mean_pp",
            "ci_low_pp",
            "ci_high_pp",
            "note",
            "cells",
        ],
    )
    print(f"wrote {len(gate)} contrasts -> tab_c3_gate_ablation.csv")
    for r in gate:
        print(
            f"  {r['contrast']}: {_pp(float(r['mean_pp']))} "
            f"[{_pp(float(r['ci_low_pp']))}, {_pp(float(r['ci_high_pp']))}]"
        )

    print("=== drop-c2 / two-seed control (Claude item C; global drop) ===")
    drop_rows = ctrl.drop_c2_twoseed_corpus(corpus)
    drop_sum = ctrl.drop_c2_twoseed_summary_table(
        drop_rows, n_bootstrap=args.n_bootstrap, seed=args.boot_seed
    )
    # Combined artefact: summary contrasts first, then per-cell block marked by row_kind.
    combined = []
    for r in drop_sum:
        combined.append({"row_kind": "summary", **r})
    for r in drop_rows:
        combined.append(
            {
                "row_kind": "cell",
                "contrast": ctrl.short_cell_label(r),
                "n_cells": 1,
                "mean_pp": float("nan"),
                "ci_low_pp": float("nan"),
                "ci_high_pp": float("nan"),
                "drop_policy": "global",
                "cells": ctrl.short_cell_label(r),
                **{k: v for k, v in r.items() if k not in ("dataset", "subset", "model")},
                "dataset": r.get("dataset", ""),
                "subset": r.get("subset") or "",
                "model": r.get("model", ""),
            }
        )
    _write_csv(out_dir / "tab_c3_drop_c2_twoseed.csv", combined)
    print(f"wrote {len(drop_sum)} summary + {len(drop_rows)} cells -> tab_c3_drop_c2_twoseed.csv")
    for r in drop_sum:
        print(
            f"  {r['contrast']}: {_pp(float(r['mean_pp']))} "
            f"[{_pp(float(r['ci_low_pp']))}, {_pp(float(r['ci_high_pp']))}] "
            f"(n={r['n_cells']})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
