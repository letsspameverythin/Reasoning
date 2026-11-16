#!/usr/bin/env python
"""Run Claude-requested measurement audits on graded results_fp16 (CPU only)."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.corpus import load_corpus
from src.analysis import measurement_audits as audits
from src.utils import setup_logging


def _write_csv(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r:
            if k not in seen:
                keys.append(k)
                seen.add(k)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--results-dir", default="results_fp16")
    p.add_argument("--out-dir", default="paper_artefacts")
    p.add_argument("--docs-out", default="docs/AUDIT_TIE_BOOTSTRAP.md")
    p.add_argument("--n-bootstrap", type=int, default=audits.DEFAULT_ITEM_BOOTSTRAP)
    p.add_argument("--boot-seed", type=int, default=0)
    p.add_argument("--p4-csv", default="paper_artefacts/tab_p4_seed_calibration.csv")
    p.add_argument("--skip-reorder", action="store_true")
    args = p.parse_args()
    setup_logging()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"loading corpus from {args.results_dir} ...")
    corpus = load_corpus(args.results_dir)
    print(f"  {len(corpus.cells)} cells in {time.time()-t0:.1f}s")

    # 1. Tie audit
    print("tie audit ...")
    t1 = time.time()
    tie_rows = audits.tie_audit_table(corpus)
    tie_summary = audits.tie_audit_method_summary(tie_rows)
    # Append method-summary rows tagged level=summary for a single CSV, plus
    # keep per-cell rows.
    for r in tie_rows:
        r["level"] = "cell"
    for r in tie_summary:
        r["level"] = "method_mean"
        r.setdefault("dataset", "")
        r.setdefault("subset", "")
        r.setdefault("model", "")
        r["frac_modal_tie"] = r.pop("mean_frac_modal_tie")
        r["acc_fixed_tiebreak"] = r.pop("mean_acc_fixed_tiebreak")
        r["acc_fractional"] = r.pop("mean_acc_fractional")
        r["n_items"] = r.pop("n_cells")
    tie_path = out_dir / "tab_tie_audit.csv"
    _write_csv(tie_path, tie_rows + tie_summary)
    print(f"  wrote {tie_path} ({len(tie_rows)} cell rows) in {time.time()-t1:.1f}s")
    for r in tie_summary:
        print(
            f"  {r['method']}: tie={100*float(r['frac_modal_tie']):.2f}% "
            f"fixed={100*float(r['acc_fixed_tiebreak']):.2f}% "
            f"frac={100*float(r['acc_fractional']):.2f}% "
            f"delta={float(r['mean_delta_frac_minus_fixed_pp']):+.3f}pp"
        )

    # 2. Item-joint bootstrap
    print(f"item-joint bootstrap B={args.n_bootstrap} ...")
    t2 = time.time()
    boot_rows, boot_meta = audits.item_joint_bootstrap_c3(
        corpus, n_bootstrap=args.n_bootstrap, seed=args.boot_seed
    )
    boot_path = out_dir / "tab_c3_item_bootstrap.csv"
    _write_csv(boot_path, boot_rows)
    print(f"  wrote {boot_path} in {time.time()-t2:.1f}s")
    for r in boot_rows:
        print(
            f"  {r['contrast']}: point={float(r['point_mean_pp']):+.3f}pp "
            f"CI=[{float(r['ci_low_pp']):+.3f}, {float(r['ci_high_pp']):+.3f}]"
        )

    # 3. Extraction / truncation
    print("extraction / truncation rates ...")
    t3 = time.time()
    ext_rows = audits.extraction_rates_table(corpus)
    ext_path = out_dir / "tab_extraction_rates.csv"
    _write_csv(ext_path, ext_rows)
    print(f"  wrote {ext_path} ({len(ext_rows)} rows) in {time.time()-t3:.1f}s")

    survivors = audits.p4_surviving_cells_from_csv(args.p4_csv)
    p4_rows = audits.p4_extraction_conditional(corpus, survivors)
    p4_path = out_dir / "tab_p4_extraction_conditional.csv"
    _write_csv(p4_path, p4_rows)
    print(f"  wrote {p4_path} ({len(p4_rows)} P4 survivors)")
    for r in p4_rows:
        print(
            f"  {r['dataset']}/{r['subset']} {r['model'].split('/')[-1]}: "
            f"range {float(r['config_range_pp_all']):.2f}→"
            f"{float(r['config_range_pp_excl_unparsed']):.2f}pp "
            f"persist={r['spread_persists']} "
            f"excl={r['n_items_excluded_unparsed_mode']}"
        )

    # 4. Seed reorder baseline
    reorder_rows: list = []
    if not args.skip_reorder:
        print("seed-to-seed reorder baseline ...")
        t4 = time.time()
        reorder_rows = audits.seed_reorder_baseline(corpus, n_bootstrap=50)
        re_path = out_dir / "tab_seed_reorder_baseline.csv"
        _write_csv(re_path, reorder_rows)
        print(f"  wrote {re_path} in {time.time()-t4:.1f}s")
        for r in reorder_rows:
            print(
                f"  {r['baseline']}: reorder={100*float(r['mean_reorder_rate']):.2f}% "
                f"corr={100*float(r['mean_corrective_rate']):.2f}% "
                f"dest={100*float(r['mean_destructive_rate']):.2f}%"
            )

    # 5. Docs
    docs = _render_docs(
        tie_summary=tie_summary,
        boot_rows=boot_rows,
        boot_meta=boot_meta,
        ext_rows=ext_rows,
        p4_rows=p4_rows,
        reorder_rows=reorder_rows,
        n_bootstrap=args.n_bootstrap,
    )
    docs_path = Path(args.docs_out)
    docs_path.parent.mkdir(parents=True, exist_ok=True)
    docs_path.write_text(docs, encoding="utf-8")
    print(f"wrote {docs_path}")
    print(f"done in {time.time()-t0:.1f}s")
    return 0


def _render_docs(
    *,
    tie_summary,
    boot_rows,
    boot_meta,
    ext_rows,
    p4_rows,
    reorder_rows,
    n_bootstrap,
) -> str:
    lines = [
        "# Measurement audits: ties, item-bootstrap, extraction, seed-reorder",
        "",
        "CPU-only audits on graded `results_fp16` (no GPU).",
        "",
        "```bash",
        "PYTHONPATH=. python scripts/run_measurement_audits.py --results-dir results_fp16",
        "```",
        "",
        "## Artefacts",
        "",
        "| File | Content |",
        "|------|---------|",
        "| `paper_artefacts/tab_tie_audit.csv` | Modal-tie fraction + fixed vs fractional credit |",
        "| `paper_artefacts/tab_c3_item_bootstrap.csv` | Item-joint bootstrap CIs for C3 contrasts |",
        "| `paper_artefacts/tab_extraction_rates.csv` | Extraction / truncation by model×dataset×config |",
        "| `paper_artefacts/tab_p4_extraction_conditional.csv` | P4 spreads excluding ⊥-mode items |",
        "| `paper_artefacts/tab_seed_reorder_baseline.csv` | Seed-pair vs cross-config reorder rates |",
        "",
        "## 1. Tie audit",
        "",
        "Methods: SC@N (`c0`), SC@3N (pooled `c0` seeds), conditional diversify,",
        "seed-substitution CD, mode-of-modes. Ties = top-1 count == top-2 count",
        "(`Cell.modal_tie` / class_counts). Fixed credit = current first-occurrence",
        "/ plurality rule. Fractional = expected accuracy under uniform choice among",
        "the `k` tied top classes/modes.",
        "",
        "| Method | Mean tie % | Acc fixed | Acc fractional | Δ (pp) |",
        "|--------|------------|-----------|----------------|--------|",
    ]
    for r in tie_summary:
        lines.append(
            f"| `{r['method']}` | {100*float(r['frac_modal_tie']):.2f} | "
            f"{100*float(r['acc_fixed_tiebreak']):.2f} | "
            f"{100*float(r['acc_fractional']):.2f} | "
            f"{float(r['mean_delta_frac_minus_fixed_pp']):+.3f} |"
        )
    lines += [
        "",
        "## 2. Item-joint bootstrap (C3)",
        "",
        f"B={n_bootstrap} replicates. **Approximation:** resample bootstrap units",
        "(template_id on GSM-Symbolic, else item_id) jointly within each",
        "`(dataset, subset)` from the intersection of units across models; rebuild",
        "each cell’s ItemView multiset; re-fit τ on the train fold; recompute",
        "CD / seed-CD / SC@3N; average over split seeds `{0..4}`; then mean across",
        "cells. Plug-in sample pools for SC@3N are not re-drawn.",
        "",
        "This differs from `tab_c3_controls_summary.csv`, which cell-bootstraps the",
        "14 point estimates and ignores shared items across models.",
        "",
        "| Contrast | Point (pp) | 95% CI (item-boot) |",
        "|----------|------------|---------------------|",
    ]
    for r in boot_rows:
        lines.append(
            f"| {r['contrast']} | {float(r['point_mean_pp']):+.3f} | "
            f"[{float(r['ci_low_pp']):+.3f}, {float(r['ci_high_pp']):+.3f}] |"
        )
    lines += [
        "",
        "## 3. Extraction / truncation",
        "",
        "Per model×dataset×config at seed 0 (`extraction_failure_rate`,",
        "`truncation_rate` from sample `finish_reason` / `truncated`).",
        "",
    ]
    # Headline: max extraction and truncation
    if ext_rows:
        by_ext = sorted(
            ext_rows,
            key=lambda r: float(r.get("extraction_failure_rate") or 0),
            reverse=True,
        )
        by_tr = sorted(
            ext_rows,
            key=lambda r: float(r.get("truncation_rate") or 0),
            reverse=True,
        )
        top_e = by_ext[0]
        top_t = by_tr[0]
        mean_e = sum(float(r["extraction_failure_rate"]) for r in ext_rows) / len(ext_rows)
        mean_t = sum(float(r["truncation_rate"]) for r in ext_rows) / len(ext_rows)
        lines += [
            f"- Mean extraction failure rate (seed0 primary+c3 cells): **{100*mean_e:.3f}%**",
            f"- Mean truncation rate: **{100*mean_t:.2f}%**",
            f"- Worst extraction: `{top_e['model']}` / {top_e['dataset']} / {top_e['config']} "
            f"= {100*float(top_e['extraction_failure_rate']):.3f}%",
            f"- Worst truncation: `{top_t['model']}` / {top_t['dataset']} / {top_t['config']} "
            f"= {100*float(top_t['truncation_rate']):.2f}%",
            "",
            "### P4 survivors conditional on no ⊥ mode",
            "",
            "Exclude items whose plug-in mode is `EXTRACTION_FAILURE_CLASS` under any of",
            "`c0`/`c1`/`c2`. Report whether config π_mode range still exceeds 3pp.",
            "",
            "| Cell | Range all (pp) | Range excl ⊥ (pp) | Persist >3pp | n excl |",
            "|------|----------------|-------------------|--------------|--------|",
        ]
        for r in p4_rows:
            short = (
                r["model"]
                .replace("Qwen/Qwen2.5-", "")
                .replace("-Instruct", "")
                .replace("meta-llama/Llama-3.2-", "Llama")
            )
            lines.append(
                f"| {short}\\|{r['dataset']}\\|{r['subset'] or 'main'} | "
                f"{float(r['config_range_pp_all']):.2f} | "
                f"{float(r['config_range_pp_excl_unparsed']):.2f} | "
                f"{r['spread_persists']} | {r['n_items_excluded_unparsed_mode']} |"
            )
    lines += ["", "## 4. Seed-to-seed reorder baseline", ""]
    if reorder_rows:
        lines += [
            "Treat `c0` seeds `{0,1,2}` as fake configs and compare mean reorder /",
            "corrective / destructive rates to cross-config pairs at seed 0.",
            "",
            "| Baseline | n pairs | Reorder | Corrective | Destructive | Net corr |",
            "|----------|---------|---------|------------|-------------|----------|",
        ]
        for r in reorder_rows:
            lines.append(
                f"| `{r['baseline']}` | {r['n_pairs']} | "
                f"{100*float(r['mean_reorder_rate']):.2f}% | "
                f"{100*float(r['mean_corrective_rate']):.2f}% | "
                f"{100*float(r['mean_destructive_rate']):.2f}% | "
                f"{100*float(r['net_corrective_rate']):+.2f}% |"
            )
    else:
        lines.append("_Skipped._")
    lines += [
        "",
        "## Meta",
        "",
        "```json",
        json.dumps(boot_meta, indent=2, default=str),
        "```",
        "",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
