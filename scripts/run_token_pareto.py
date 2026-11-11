"""Accuracy-vs-actual-tokens curves from tab5 (SC / ESC / certainty / CDV).

This answers the compute-matching suggestion for baselines that already exist
in the offline pool. Conditional-diversify expected tokens are a separate
analysis (needs margin-gated early exit over the multi-config pool).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


METHODS = ("sc", "esc", "certainty_vote", "cdv", "adaptive_cdv", "seed_pooled_sc")
LABELS = {
    "sc": "SC",
    "esc": "ESC",
    "certainty_vote": "Certainty",
    "cdv": "CDV",
    "adaptive_cdv": "Adaptive CDV",
    "seed_pooled_sc": "Seed-pooled SC",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab5", default="paper_artefacts/tab5_method_comparison.csv")
    ap.add_argument("--out-csv", default="paper_artefacts/tab_token_pareto.csv")
    ap.add_argument("--out-fig", default="paper/figures/fig_token_pareto.pdf")
    args = ap.parse_args()

    t = pd.read_csv(args.tab5)
    acc = t[(t.row_kind == "accuracy") & (t.method.isin(METHODS))].copy()
    # per-cell points
    acc["subset"] = acc["subset"].fillna("")
    cell = acc.groupby(
        ["dataset", "subset", "model", "method", "budget_multiple"],
        as_index=False,
        dropna=False,
    ).agg(
        accuracy=("accuracy", "mean"),
        mean_tokens=("mean_tokens_used", "mean"),
        mean_chains=("mean_chains_used", "mean"),
    )
    # pooled mean across cells at each method × budget
    pooled = (
        cell.groupby(["method", "budget_multiple"], as_index=False, dropna=False)
        .agg(
            accuracy=("accuracy", "mean"),
            mean_tokens=("mean_tokens", "mean"),
            mean_chains=("mean_chains", "mean"),
            n_cells=("accuracy", "count"),
        )
        .sort_values(["method", "budget_multiple"])
    )
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pooled.to_csv(args.out_csv, index=False)
    cell.to_csv(Path(args.out_csv).with_name("tab_token_pareto_cells.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5.2, 3.6))
    markers = {"sc": "o", "esc": "s", "certainty_vote": "^", "cdv": "D", "adaptive_cdv": "v", "seed_pooled_sc": "P"}
    for method in METHODS:
        sub = pooled[pooled.method == method]
        if sub.empty:
            continue
        ax.plot(
            sub.mean_tokens,
            100 * sub.accuracy,
            marker=markers.get(method, "o"),
            label=LABELS.get(method, method),
            linewidth=1.5,
        )
    ax.set_xlabel("Mean completion tokens / item")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs actual tokens (14-cell mean)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    fig.savefig(out_fig.with_suffix(".png"), dpi=150)
    print(pooled.to_string(index=False))
    print("wrote", args.out_csv, args.out_fig)


if __name__ == "__main__":
    main()
