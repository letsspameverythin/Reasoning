#!/usr/bin/env python
"""CPU C3 bulletproofing: expected tokens, margin-split, quality-matched CDV."""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.corpus import load_corpus
from src.analysis import c3_compute as c3c
from src.utils import setup_logging


def _write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _pp(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{100.0 * x:+.2f}pp"


def _mean_key(rows: list, key: str) -> float:
    vals = [
        float(r[key])
        for r in rows
        if math.isfinite(float(r.get(key, float("nan"))))
    ]
    return sum(vals) / len(vals) if vals else float("nan")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results_fp16")
    p.add_argument("--out-dir", default="paper_artefacts")
    p.add_argument("--n-bootstrap", type=int, default=c3c.DEFAULT_CELL_BOOTSTRAP)
    p.add_argument("--boot-seed", type=int, default=0)
    args = p.parse_args()
    setup_logging()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus = load_corpus(args.results_dir)

    print("=== expected tokens (probe-then-commit) ===")
    tok_rows = c3c.expected_tokens_corpus(corpus)
    tok_path = out_dir / "tab_c3_expected_tokens.csv"
    _write_csv(tok_path, tok_rows)
    print(f"wrote {len(tok_rows)} rows -> {tok_path}")
    if tok_rows:
        print(
            f"  mean ratio probe/SC@N={_mean_key(tok_rows, 'ratio_probe_to_sc_n'):.3f}  "
            f"probe/always3={_mean_key(tok_rows, 'ratio_probe_to_always3'):.3f}  "
            f"frac_early_exit={_mean_key(tok_rows, 'frac_early_exit'):.3f}"
        )

    print("=== independent margin-split ===")
    ms_rows = c3c.margin_split_corpus(corpus)
    ms_path = out_dir / "tab_c3_margin_split.csv"
    _write_csv(ms_path, ms_rows)
    print(f"wrote {len(ms_rows)} rows -> {ms_path}")
    if ms_rows:
        d_ms = _mean_key(ms_rows, "delta_cd_margin_split_minus_sc")
        d_full = _mean_key(ms_rows, "delta_cd_full_minus_sc")
        print(
            f"  mean delta CD(margin-split)-SC={_pp(d_ms)}  "
            f"delta CD(full)-SC={_pp(d_full)}  "
            f"split-full={_pp(_mean_key(ms_rows, 'delta_margin_split_minus_full_cd'))}"
        )
        summ = c3c.summarise_delta_column(
            ms_rows,
            "delta_cd_margin_split_minus_sc",
            n_bootstrap=args.n_bootstrap,
            seed=args.boot_seed,
        )
        print(
            f"  cell-boot CI: {summ['mean_pp']:.2f} "
            f"[{summ['ci_low_pp']:.2f}, {summ['ci_high_pp']:.2f}] "
            f"(n={summ['n_cells']})"
        )

    print("=== quality-matched / weighted CDV ===")
    qm_rows = c3c.quality_matched_corpus(corpus)
    qm_path = out_dir / "tab_quality_matched_cdv.csv"
    _write_csv(qm_path, qm_rows)
    print(f"wrote {len(qm_rows)} rows -> {qm_path}")
    if qm_rows:
        print(
            f"  delta CD@3-SC={_pp(_mean_key(qm_rows, 'delta_cd_full3_minus_sc'))}  "
            f"delta CD{{c0,c1}}-SC={_pp(_mean_key(qm_rows, 'delta_cd_c0_c1_minus_sc'))}  "
            f"delta CD_w-SC={_pp(_mean_key(qm_rows, 'delta_cd_weighted_minus_sc'))}  "
            f"delta mix3-SC={_pp(_mean_key(qm_rows, 'delta_mixture_3cfg_minus_sc'))}"
        )
        seed_qm = [r for r in qm_rows if r.get("has_seed_control")]
        if seed_qm:
            print(
                f"  seed cells: delta mix3-SC@3N="
                f"{_pp(_mean_key(seed_qm, 'delta_mixture_3cfg_minus_sc_3n'))}  "
                f"delta CD@3-SC@3N={_pp(_mean_key(seed_qm, 'delta_cd_full3_minus_sc_3n'))}  "
                f"delta CD@3-mix={_pp(_mean_key(seed_qm, 'delta_cd_full3_minus_mixture'))}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
