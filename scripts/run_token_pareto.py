#!/usr/bin/env python
"""Accuracy-vs-tokens frontier with selector operating points at ~1.6× SC@N.

Curves come from ``tab5_method_comparison.csv`` (SC / ESC / certainty / CDV).
Selector points (probe-then-commit CD, seed-sub, SC matched to ~1.6×, ESC
nearest) use C3 control accuracies and expected-token ratios from
``tab_c3_*.csv`` (CPU-only; no GPU).
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


CURVE_METHODS = ("sc", "esc", "certainty_vote", "cdv", "adaptive_cdv", "seed_pooled_sc")
LABELS = {
    "sc": "SC",
    "esc": "ESC",
    "certainty_vote": "Certainty",
    "cdv": "CDV (budget grid)",
    "adaptive_cdv": "Adaptive CDV",
    "seed_pooled_sc": "Seed-pooled SC",
    "conditional_diversify": "CD probe-then-commit",
    "seed_substitution": "Seed-sub probe-then-commit",
    "sc_matched_1p6x": "SC @ matched ~1.6×",
    "esc_nearest_1p6x": "ESC @ nearest to ~1.6×",
}


def _interp_acc(token_mults: np.ndarray, accs: np.ndarray, target: float) -> float:
    """Linear interpolate accuracy vs token-multiple; clamp to endpoints."""
    order = np.argsort(token_mults)
    x = token_mults[order]
    y = accs[order]
    if target <= x[0]:
        return float(y[0])
    if target >= x[-1]:
        return float(y[-1])
    return float(np.interp(target, x, y))


def _mean_finite(vals) -> float:
    xs = [float(v) for v in vals if math.isfinite(float(v))]
    return sum(xs) / len(xs) if xs else float("nan")


def build_selector_points(
    controls: pd.DataFrame,
    expected: pd.DataFrame,
    tab5_acc: pd.DataFrame,
    target_mult: float,
) -> pd.DataFrame:
    """Build labeled selector points at / near ``target_mult`` × SC@N tokens."""
    ctrl = controls.copy()
    exp = expected.copy()
    for df in (ctrl, exp):
        df["subset"] = df["subset"].fillna("")
    merged = ctrl.merge(
        exp[
            [
                "dataset",
                "subset",
                "model",
                "ratio_probe_to_sc_n",
                "mean_tok_probe_then_commit",
                "mean_tok_sc_n",
                "mean_tok_sc_3n_seeds",
                "frac_early_exit",
            ]
        ],
        on=["dataset", "subset", "model"],
        how="left",
    )

    # Seed-sub probe-then-commit uses the same early-exit cost model on 3 seeds:
    # expected tokens ≈ same ratio as config probe (documented as shared cost model).
    probe_mult = float(_mean_finite(merged["ratio_probe_to_sc_n"]))
    if not math.isfinite(probe_mult):
        probe_mult = target_mult

    acc_cd = _mean_finite(merged["acc_conditional_diversify_configs"])
    acc_seed = _mean_finite(merged.get("acc_cd_seeds", []))
    acc_sc = _mean_finite(merged.get("acc_sc_n_joint", merged["acc_sc_n"]))
    acc_sc3n = _mean_finite(merged.get("acc_sc_3n_c0_seeds", []))

    # SC@3N token multiple from expected tokens (≈3 by construction).
    sc3n_mults = []
    for _, r in merged.iterrows():
        t0 = float(r.get("mean_tok_sc_n", float("nan")))
        t3 = float(r.get("mean_tok_sc_3n_seeds", float("nan")))
        if math.isfinite(t0) and t0 > 0 and math.isfinite(t3):
            sc3n_mults.append(t3 / t0)
    sc3n_mult = _mean_finite(sc3n_mults)
    if not math.isfinite(sc3n_mult) or sc3n_mult <= 1.0:
        sc3n_mult = 3.0

    # Interpolate SC between SC@N (1×) and SC@3N (~3×) to the probe multiple.
    acc_sc_matched = _interp_acc(
        np.array([1.0, sc3n_mult], dtype=float),
        np.array([acc_sc, acc_sc3n], dtype=float),
        probe_mult,
    )

    # ESC curve: token multiple relative to SC@N (= largest SC budget mean tokens).
    # ``tab5_acc`` may already be cell-aggregated with token_multiple filled in.
    tok_col = (
        "mean_tokens"
        if "mean_tokens" in tab5_acc.columns
        else "mean_tokens_used"
    )
    esc = tab5_acc[tab5_acc.method == "esc"].copy()
    esc["subset"] = esc["subset"].fillna("")
    if "token_multiple" not in esc.columns or esc["token_multiple"].isna().all():
        sc_full = tab5_acc[tab5_acc.method == "sc"].copy()
        max_b = float(sc_full["budget_multiple"].max())
        sc_ref = (
            sc_full[sc_full["budget_multiple"] == max_b]
            .groupby(["dataset", "subset", "model"], as_index=False, dropna=False)
            .agg(**{"sc_n_tokens": (tok_col, "mean")})
        )
        sc_ref["subset"] = sc_ref["subset"].fillna("")
        esc = esc.merge(sc_ref, on=["dataset", "subset", "model"], how="left")
        esc["token_multiple"] = esc[tok_col] / esc["sc_n_tokens"]
    esc_pool = (
        esc.groupby("budget_multiple", as_index=False)
        .agg(
            accuracy=("accuracy", "mean"),
            mean_tokens=(tok_col, "mean"),
            token_multiple=("token_multiple", "mean"),
            n_cells=("accuracy", "count"),
        )
        .sort_values("token_multiple")
    )
    if esc_pool.empty:
        esc_acc = float("nan")
        esc_mult = float("nan")
        esc_tok = float("nan")
        esc_budget = float("nan")
    else:
        # Nearest available ESC operating point to the probe multiple.
        idx = int((esc_pool["token_multiple"] - probe_mult).abs().idxmin())
        row = esc_pool.loc[idx]
        esc_acc = float(row["accuracy"])
        esc_mult = float(row["token_multiple"])
        esc_tok = float(row["mean_tokens"])
        esc_budget = float(row["budget_multiple"])

    mean_sc_tok = _mean_finite(merged["mean_tok_sc_n"])
    mean_probe_tok = _mean_finite(merged["mean_tok_probe_then_commit"])

    points = [
        {
            "method": "conditional_diversify",
            "point_kind": "selector",
            "budget_multiple": float("nan"),
            "accuracy": acc_cd,
            "mean_tokens": mean_probe_tok,
            "token_multiple": probe_mult,
            "mean_chains": float("nan"),
            "n_cells": int(merged.shape[0]),
            "label": LABELS["conditional_diversify"],
            "note": "probe-then-commit expected tokens from tab_c3_expected_tokens",
        },
        {
            "method": "seed_substitution",
            "point_kind": "selector",
            "budget_multiple": float("nan"),
            "accuracy": acc_seed,
            "mean_tokens": mean_probe_tok,  # shared probe-then-commit cost model
            "token_multiple": probe_mult,
            "mean_chains": float("nan"),
            "n_cells": int(
                sum(
                    1
                    for v in merged.get(
                        "has_seed_control", [True] * len(merged)
                    )
                    if bool(v)
                )
            ),
            "label": LABELS["seed_substitution"],
            "note": "same probe-then-commit cost model as CD (3 c0 seeds as fake configs)",
        },
        {
            "method": "sc_matched_1p6x",
            "point_kind": "selector",
            "budget_multiple": float("nan"),
            "accuracy": acc_sc_matched,
            "mean_tokens": mean_sc_tok * probe_mult
            if math.isfinite(mean_sc_tok)
            else float("nan"),
            "token_multiple": probe_mult,
            "mean_chains": float("nan"),
            "n_cells": int(merged.shape[0]),
            "label": LABELS["sc_matched_1p6x"],
            "note": (
                f"linear interp SC@N->SC@3N (sc3n_mult={sc3n_mult:.3f}) "
                f"to probe mult={probe_mult:.3f}"
            ),
        },
        {
            "method": "esc_nearest_1p6x",
            "point_kind": "selector",
            "budget_multiple": esc_budget,
            "accuracy": esc_acc,
            "mean_tokens": esc_tok,
            "token_multiple": esc_mult,
            "mean_chains": float("nan"),
            "n_cells": int(esc_pool["n_cells"].max()) if not esc_pool.empty else 0,
            "label": LABELS["esc_nearest_1p6x"],
            "note": (
                f"nearest ESC budget point to probe mult={probe_mult:.3f} "
                f"(ESC early-exits below full SC@N tokens)"
            ),
        },
    ]
    return pd.DataFrame(points)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tab5", default="paper_artefacts/tab5_method_comparison.csv")
    ap.add_argument(
        "--controls", default="paper_artefacts/tab_c3_controls.csv"
    )
    ap.add_argument(
        "--expected-tokens", default="paper_artefacts/tab_c3_expected_tokens.csv"
    )
    ap.add_argument("--out-csv", default="paper_artefacts/tab_token_pareto.csv")
    ap.add_argument("--out-fig", default="paper_artefacts/fig_token_pareto.pdf")
    ap.add_argument(
        "--target-multiple",
        type=float,
        default=1.6,
        help="Target token multiple of SC@N for matched selector points",
    )
    args = ap.parse_args()

    t = pd.read_csv(args.tab5)
    acc = t[(t.row_kind == "accuracy") & (t.method.isin(CURVE_METHODS))].copy()
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

    # Token multiple relative to each cell's full SC@N (largest SC budget).
    sc_full = (
        cell[cell.method == "sc"]
        .sort_values("budget_multiple")
        .groupby(["dataset", "subset", "model"], as_index=False)
        .tail(1)[["dataset", "subset", "model", "mean_tokens"]]
        .rename(columns={"mean_tokens": "sc_n_tokens"})
    )
    cell = cell.merge(sc_full, on=["dataset", "subset", "model"], how="left")
    cell["token_multiple"] = cell["mean_tokens"] / cell["sc_n_tokens"]

    pooled = (
        cell.groupby(["method", "budget_multiple"], as_index=False, dropna=False)
        .agg(
            accuracy=("accuracy", "mean"),
            mean_tokens=("mean_tokens", "mean"),
            mean_chains=("mean_chains", "mean"),
            token_multiple=("token_multiple", "mean"),
            n_cells=("accuracy", "count"),
        )
        .sort_values(["method", "budget_multiple"])
    )
    pooled["point_kind"] = "curve"
    pooled["label"] = pooled["method"].map(LABELS)
    pooled["note"] = "tab5 budget grid"

    controls = pd.read_csv(args.controls)
    expected = pd.read_csv(args.expected_tokens)
    selectors = build_selector_points(
        controls, expected, cell, target_mult=float(args.target_multiple)
    )

    out = pd.concat([pooled, selectors], ignore_index=True, sort=False)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    cell.to_csv(Path(args.out_csv).with_name("tab_token_pareto_cells.csv"), index=False)

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    markers = {
        "sc": "o",
        "esc": "s",
        "certainty_vote": "^",
        "cdv": "D",
        "adaptive_cdv": "v",
        "seed_pooled_sc": "P",
    }
    for method in CURVE_METHODS:
        sub = pooled[pooled.method == method]
        if sub.empty:
            continue
        ax.plot(
            sub.token_multiple,
            100 * sub.accuracy,
            marker=markers.get(method, "o"),
            label=LABELS.get(method, method),
            linewidth=1.5,
            alpha=0.85,
        )

    sel_style = {
        "conditional_diversify": ("*", "C3", 160),
        "seed_substitution": ("X", "C1", 110),
        "sc_matched_1p6x": ("P", "0.35", 120),
        "esc_nearest_1p6x": ("s", "C2", 90),
    }
    for _, r in selectors.iterrows():
        if not math.isfinite(float(r.accuracy)) or not math.isfinite(
            float(r.token_multiple)
        ):
            continue
        mk, color, size = sel_style.get(r.method, ("o", "k", 80))
        ax.scatter(
            float(r.token_multiple),
            100 * float(r.accuracy),
            marker=mk,
            s=size,
            color=color,
            zorder=5,
            label=r.label,
            edgecolors="k",
            linewidths=0.4,
        )

    target = float(
        selectors.loc[
            selectors.method == "conditional_diversify", "token_multiple"
        ].iloc[0]
    )
    if math.isfinite(target):
        ax.axvline(target, color="0.5", linestyle="--", linewidth=1.0, alpha=0.7)
        ax.text(
            target,
            ax.get_ylim()[0] if ax.get_ylim()[0] else 0,
            f"  {target:.2f}× SC@N",
            fontsize=8,
            color="0.35",
            va="bottom",
        )

    ax.set_xlabel("Token multiple of SC@N (completion tokens)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy vs tokens (14-cell mean; selector @ ~1.6×)")
    ax.legend(fontsize=7, frameon=False, loc="lower right")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out_fig = Path(args.out_fig)
    out_fig.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_fig)
    fig.savefig(out_fig.with_suffix(".png"), dpi=150)

    print(out.to_string(index=False))
    print("wrote", args.out_csv, args.out_fig)


if __name__ == "__main__":
    main()
