#!/usr/bin/env python
"""Recompute all C3 selector contrasts under fixed vs fractional tie credit.

Writes:
  - paper_artefacts/tab_c3_fractional_credit.csv
  - paper_artefacts/tab_c3_fractional_vs_fixed.csv
  - docs/FRACTIONAL_CREDIT_VERDICT.md

Usage:
  PYTHONPATH=. python scripts/run_fractional_credit.py --results-dir results_fp16
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.corpus import load_corpus
from src.analysis import c3_ties as ties
from src.utils import setup_logging


def _write_csv(path: Path, rows: list, preferred: list | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    all_keys = sorted({k for r in rows for k in r})
    if preferred:
        keys = list(dict.fromkeys([*preferred, *all_keys]))
    else:
        keys = all_keys
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _fmt_pp(x: float) -> str:
    if not math.isfinite(x):
        return "nan"
    return f"{x:+.2f}"


def _fmt_ci(lo: float, hi: float) -> str:
    if not (math.isfinite(lo) and math.isfinite(hi)):
        return "[nan, nan]"
    return f"[{_fmt_pp(lo)}, {_fmt_pp(hi)}]"


def _render_verdict(
    side: list,
    verdict: dict,
    tie_summary: list,
    item_joint: list,
    item_meta: dict,
) -> str:
    by = {r["contrast"]: r for r in side}
    critical_names = [
        "CD_minus_SC_N",
        "SeedSub_minus_SC_N",
        "SC_3N_minus_SC_N",
        "CD_minus_SeedSub",
        "CD_minus_SC_3N",
        "SeedSub_minus_SC_3N",
        "Ungated_MoM_minus_SC_N",
        "Gated_minus_Ungated",
        "DropC2_CD_minus_SC_N",
        "DropC2_minus_TwoSeedCD",
        "TwoSeedCD_minus_SC_2N",
        "DropC2_minus_SC_2N",
        "MarginSplit_CD_minus_SC",
    ]

    lines = [
        "# Fractional credit verdict (Claude tie-artifact check)",
        "",
        "Recompute of C3 selector contrasts under **fixed** (sample: first-occurrence;",
        "mode policies: any-winner) vs **fractional** credit (`k`-way ties → credit",
        "`#correct / k`). Gated CD uses c0 sample plurality under the same rule.",
        "",
        "```bash",
        "PYTHONPATH=. python scripts/run_fractional_credit.py --results-dir results_fp16",
        "```",
        "",
        "## Recommendation",
        "",
        f"**{verdict['decision']}** the gating/aggregation claim.",
        "",
        verdict["reason"],
        "",
        "## Critical contrasts (cell-bootstrap, 14 cells)",
        "",
        "| Contrast | Fixed mean (pp) | Fixed 95% CI | Fractional mean (pp) | Fractional 95% CI | Δ (frac−fixed) |",
        "|----------|----------------:|-------------:|---------------------:|------------------:|---------------:|",
    ]
    for name in (
        "SeedSub_minus_SC_3N",
        "CD_minus_SC_N",
        "CD_minus_SC_3N",
        "Gated_minus_Ungated",
        "CD_minus_SeedSub",
    ):
        r = by.get(name, {})
        lines.append(
            f"| {name} | {_fmt_pp(float(r.get('fixed_mean_pp', float('nan'))))} | "
            f"{_fmt_ci(float(r.get('fixed_ci_low', float('nan'))), float(r.get('fixed_ci_high', float('nan'))))} | "
            f"{_fmt_pp(float(r.get('frac_mean_pp', float('nan'))))} | "
            f"{_fmt_ci(float(r.get('frac_ci_low', float('nan'))), float(r.get('frac_ci_high', float('nan'))))} | "
            f"{_fmt_pp(float(r.get('delta_frac_minus_fixed_pp', float('nan'))))} |"
        )

    lines += [
        "",
        "## All contrasts (1–13) + token frontier (14)",
        "",
        "| # | Contrast | Fixed | Frac | Δ |",
        "|--:|----------|------:|-----:|--:|",
    ]
    for name in critical_names:
        r = by.get(name, {})
        idx = r.get("contrast_index", "")
        lines.append(
            f"| {idx} | {name} | {_fmt_pp(float(r.get('fixed_mean_pp', float('nan'))))} | "
            f"{_fmt_pp(float(r.get('frac_mean_pp', float('nan'))))} | "
            f"{_fmt_pp(float(r.get('delta_frac_minus_fixed_pp', float('nan'))))} |"
        )

    # Token frontier accuracies (mean_pp here is 100*acc).
    lines += ["", "### Token frontier (~1.6×) accuracies", "", "| Point | Fixed acc | Frac acc |", "|-------|----------:|---------:|"]
    for name in (
        "token_frontier_CD",
        "token_frontier_SeedSub",
        "token_frontier_SC_1p6x",
    ):
        r = by.get(name, {})
        fa = float(r.get("fixed_mean_pp", float("nan")))
        ra = float(r.get("frac_mean_pp", float("nan")))
        lines.append(
            f"| {name} | "
            f"{(fa/100.0):.4f} | {(ra/100.0):.4f} |"
            if math.isfinite(fa) and math.isfinite(ra)
            else f"| {name} | nan | nan |"
        )

    lines += [
        "",
        "## Mean tie rates (policy layer, 14-cell mean)",
        "",
        "| Policy | Tie rate (fixed eval) | Tie rate (frac eval) |",
        "|--------|----------------------:|---------------------:|",
    ]
    for t in tie_summary:
        lines.append(
            f"| `{t['policy']}` | "
            f"{100*float(t['mean_tie_rate_fixed']):.2f}% | "
            f"{100*float(t['mean_tie_rate_frac']):.2f}% |"
            if math.isfinite(float(t.get("mean_tie_rate_fixed", float("nan"))))
            else f"| `{t['policy']}` | nan | nan |"
        )

    if item_joint:
        lines += [
            "",
            "## Item-joint bootstrap (critical contrasts)",
            "",
            f"B={item_meta.get('n_bootstrap', '?')}. Same approximation as "
            "`tab_c3_item_bootstrap.csv`, with fractional/fixed credit.",
            "",
            "| Contrast | Rule | Point (pp) | 95% CI |",
            "|----------|------|-----------:|-------:|",
        ]
        for r in item_joint:
            lines.append(
                f"| {r['contrast']} | {r['rule']} | "
                f"{_fmt_pp(float(r['mean_pp']))} | "
                f"{_fmt_ci(float(r['ci_low']), float(r['ci_high']))} |"
            )

    lines += [
        "",
        "## Artefacts",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `paper_artefacts/tab_c3_fractional_credit.csv` | Long format: contrast × rule |",
        "| `paper_artefacts/tab_c3_fractional_vs_fixed.csv` | Side-by-side fixed vs fractional |",
        "| `docs/FRACTIONAL_CREDIT_VERDICT.md` | This verdict |",
        "",
        "## Meta",
        "",
        "```json",
        json.dumps(
            {
                "decision": verdict["decision"],
                "critical": {
                    k: {
                        "fixed_mean_pp": v["fixed_mean_pp"],
                        "frac_mean_pp": v["frac_mean_pp"],
                        "frac_ci": list(v["frac_ci"]),
                        "delta_pp": v["delta_pp"],
                    }
                    for k, v in verdict["critical"].items()
                },
            },
            indent=2,
        ),
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results_fp16")
    p.add_argument("--out-dir", default="paper_artefacts")
    p.add_argument("--docs-out", default="docs/FRACTIONAL_CREDIT_VERDICT.md")
    p.add_argument("--n-bootstrap", type=int, default=ties.DEFAULT_CELL_BOOTSTRAP)
    p.add_argument("--item-bootstrap", type=int, default=ties.DEFAULT_ITEM_BOOTSTRAP)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--skip-item-joint", action="store_true")
    args = p.parse_args()
    setup_logging()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"loading corpus from {args.results_dir} …")
    corpus = load_corpus(args.results_dir)
    print(f"  {len(corpus.cells)} cells in {time.time()-t0:.1f}s")

    result = ties.run_fractional_credit_analysis(
        corpus,
        n_bootstrap=args.n_bootstrap,
        boot_seed=args.boot_seed,
        item_bootstrap=args.item_bootstrap,
        skip_item_joint=args.skip_item_joint,
    )

    long_path = out_dir / "tab_c3_fractional_credit.csv"
    _write_csv(
        long_path,
        result["long_rows"],
        preferred=[
            "contrast",
            "contrast_index",
            "rule",
            "mean_pp",
            "ci_low",
            "ci_high",
            "mean_tie_rate",
            "n_cells",
        ],
    )
    print(f"wrote {long_path} ({len(result['long_rows'])} rows)")

    side_path = out_dir / "tab_c3_fractional_vs_fixed.csv"
    _write_csv(
        side_path,
        result["side_by_side"],
        preferred=[
            "contrast",
            "contrast_index",
            "n_cells",
            "fixed_mean_pp",
            "fixed_ci_low",
            "fixed_ci_high",
            "frac_mean_pp",
            "frac_ci_low",
            "frac_ci_high",
            "delta_frac_minus_fixed_pp",
            "fixed_mean_tie_rate",
            "frac_mean_tie_rate",
        ],
    )
    print(f"wrote {side_path} ({len(result['side_by_side'])} rows)")

    if result["item_joint"]:
        ij_path = out_dir / "tab_c3_fractional_item_joint.csv"
        _write_csv(ij_path, result["item_joint"])
        print(f"wrote {ij_path}")

    verdict = ties.gating_verdict(result["side_by_side"])
    docs = _render_verdict(
        result["side_by_side"],
        verdict,
        result["tie_summary"],
        result["item_joint"],
        result["item_meta"],
    )
    docs_path = Path(args.docs_out)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(docs, encoding="utf-8")
    print(f"wrote {docs_path}")

    def _safe(s: str) -> str:
        return s.encode("ascii", "replace").decode("ascii")

    print()
    print("=" * 60)
    print(f"VERDICT: {verdict['decision']}")
    print(_safe(verdict["reason"]))
    print()
    for name in ("SeedSub_minus_SC_3N", "CD_minus_SC_N", "CD_minus_SC_3N"):
        r = next(x for x in result["side_by_side"] if x["contrast"] == name)
        print(
            _safe(
                f"  {name}: fixed={_fmt_pp(float(r['fixed_mean_pp']))} "
                f"{_fmt_ci(float(r['fixed_ci_low']), float(r['fixed_ci_high']))}  |  "
                f"frac={_fmt_pp(float(r['frac_mean_pp']))} "
                f"{_fmt_ci(float(r['frac_ci_low']), float(r['frac_ci_high']))}  "
                f"d={_fmt_pp(float(r['delta_frac_minus_fixed_pp']))}"
            )
        )
    print(f"done in {time.time()-t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
