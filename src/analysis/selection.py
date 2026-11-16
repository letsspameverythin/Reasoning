"""Option B: label-free / val-selected configuration policies vs the oracle–SC gap.

Uniform CDV failing is the setup. The question is how much of
``oracle_best_single - SC`` (and the stronger per-item ``oracle_config - SC``)
is recoverable without test labels.

Design choices that survive ICML review:

* Policies are evaluated on a **held-out item split** (or template split for
  GSM-Symbolic). Thresholds / global config choice are fit on the train fold only.
* ``val_selected_global`` may use labels on the train fold (standard model
  selection). Purely label-free policies never see gold answers.
* Headline metric is **fraction of oracle gap recovered**, with item-clustered
  bootstrap CIs — not a bare accuracy delta.
* Uniform CDV / adaptive CDV / random config remain baselines, not the claim.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..elicitation import DEFAULT_CDV_CONFIG_IDS, REFERENCE_CONFIG_ID
from ..metrics import DEFAULT_ALPHA, DEFAULT_N_BOOTSTRAP, bootstrap_ci, equivalence_from_ci
from .corpus import Cell, Corpus

log = logging.getLogger(__name__)

DEFAULT_VAL_FRACTION = 0.4
DEFAULT_SPLIT_SEED = 0
#: Confirmatory label-free policy (pre-specified; NOT chosen per cell on test).
PRIMARY_POLICY = "conditional_diversify"
#: Split seeds for multi-split robustness (reviewer W3 / Claude winner-curse).
DEFAULT_SPLIT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
#: Margin threshold grid for conditional diversification (fit on train only).
MARGIN_THRESHOLDS: Tuple[float, ...] = (0.05, 0.10, 0.15, 0.25, 0.35, 0.50)
#: Practically meaningful lift over SC for policy superiority (1pp).
POLICY_DELTA = 0.01


@dataclass
class ItemView:
    """Per-item multi-config features usable without gold labels (except oracles)."""

    item_id: str
    unit: str
    modes: Dict[str, Optional[str]]
    correct: Dict[str, bool]
    margins: Dict[str, float]
    n: Dict[str, int]

    def available(self, configs: Sequence[str]) -> List[str]:
        return [c for c in configs if c in self.modes and self.n.get(c, 0) > 0]


def _item_views(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Sequence[str],
    seed: int = 0,
) -> List[ItemView]:
    columns: Dict[str, Dict[str, Cell]] = {c: {} for c in configs}
    for cell in corpus.cells.values():
        if (
            cell.dataset != dataset
            or cell.subset != subset
            or cell.model != model
            or cell.seed != seed
            or cell.config not in columns
            or cell.n == 0
        ):
            continue
        columns[cell.config][cell.item_id] = cell

    shared = None
    for col in columns.values():
        ids = set(col)
        shared = ids if shared is None else shared & ids
    if not shared:
        return []

    views: List[ItemView] = []
    for item_id in sorted(shared):
        modes: Dict[str, Optional[str]] = {}
        correct: Dict[str, bool] = {}
        margins: Dict[str, float] = {}
        n: Dict[str, int] = {}
        unit = None
        for c in configs:
            cell = columns[c][item_id]
            modes[c] = cell.modal_class()
            correct[c] = cell.modal_correct()
            margins[c] = cell.margin()
            n[c] = cell.n
            unit = cell.bootstrap_unit()
        views.append(
            ItemView(
                item_id=item_id,
                unit=str(unit or item_id),
                modes=modes,
                correct=correct,
                margins=margins,
                n=n,
            )
        )
    return views


def _split_views(
    views: Sequence[ItemView],
    val_fraction: float = DEFAULT_VAL_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
) -> Tuple[List[ItemView], List[ItemView]]:
    """Split by bootstrap unit so Symbolic templates do not leak across folds."""
    import numpy as np

    units = sorted({v.unit for v in views})
    if len(units) < 4:
        # Too small to split honestly: evaluate in-sample and flag.
        return list(views), list(views)

    rng = np.random.default_rng(split_seed)
    order = list(units)
    rng.shuffle(order)
    n_val = max(1, int(round(len(order) * val_fraction)))
    n_val = min(n_val, len(order) - 1)
    val_units = set(order[:n_val])
    train = [v for v in views if v.unit in val_units]
    test = [v for v in views if v.unit not in val_units]
    if not train or not test:
        return list(views), list(views)
    return train, test


def _accuracy(flags: Sequence[bool]) -> float:
    return sum(1 for f in flags if f) / len(flags) if flags else float("nan")


def _policy_sc(view: ItemView, reference: str) -> bool:
    return bool(view.correct.get(reference, False))


def _policy_oracle_config(view: ItemView, configs: Sequence[str]) -> bool:
    return any(view.correct.get(c, False) for c in view.available(configs))


def _policy_oracle_best_single(
    view: ItemView, chosen: str
) -> bool:
    return bool(view.correct.get(chosen, False))


def _policy_margin_max(view: ItemView, configs: Sequence[str], reference: str) -> bool:
    avail = view.available(configs)
    if not avail:
        return False
    best = max(avail, key=lambda c: (view.margins.get(c, -1.0), c == reference, c))
    return bool(view.correct.get(best, False))


def _policy_agreement_else_margin(
    view: ItemView, configs: Sequence[str], reference: str
) -> bool:
    avail = view.available(configs)
    if not avail:
        return False
    modes = [view.modes.get(c) for c in avail]
    if len(set(modes)) == 1 and modes[0] is not None:
        # All configs agree on the mode: take that verdict from any config.
        return bool(view.correct.get(avail[0], False))
    return _policy_margin_max(view, configs, reference)


def _mode_plurality_correct(view: ItemView, configs: Sequence[str]) -> bool:
    """One vote per config's plug-in mode; correct iff a winning mode is correct."""
    avail = view.available(configs)
    votes: Dict[str, int] = {}
    for c in avail:
        mode = view.modes.get(c)
        if mode is None:
            continue
        votes[mode] = votes.get(mode, 0) + 1
    if not votes:
        return False
    top = max(votes.values())
    winners = {m for m, k in votes.items() if k == top}
    for c in avail:
        mode = view.modes.get(c)
        if mode in winners and view.correct.get(c, False):
            return True
    return False


def _policy_conditional_diversify(
    view: ItemView,
    configs: Sequence[str],
    reference: str,
    threshold: float,
) -> bool:
    """Use reference when its margin is large; else plurality over config modes.

    ``threshold > 1`` forces always-diversify (margins lie in ``[0, 1]``).
    """
    avail = view.available(configs)
    if not avail:
        return False
    ref_margin = view.margins.get(reference, 0.0)
    if (
        threshold <= 1.0
        and math.isfinite(ref_margin)
        and ref_margin >= threshold
        and reference in avail
    ):
        return bool(view.correct.get(reference, False))
    return _mode_plurality_correct(view, configs)


def _fit_val_selected_config(
    train: Sequence[ItemView], configs: Sequence[str], reference: str
) -> str:
    best_c, best_acc = reference, -1.0
    for c in configs:
        flags = [v.correct.get(c, False) for v in train if c in v.correct]
        acc = _accuracy(flags)
        if math.isfinite(acc) and acc > best_acc:
            best_acc = acc
            best_c = c
    return best_c


def _fit_diversify_threshold(
    train: Sequence[ItemView],
    configs: Sequence[str],
    reference: str,
) -> float:
    best_t, best_acc = MARGIN_THRESHOLDS[0], -1.0
    for t in MARGIN_THRESHOLDS:
        flags = [
            _policy_conditional_diversify(v, configs, reference, t) for v in train
        ]
        acc = _accuracy(flags)
        if math.isfinite(acc) and acc > best_acc:
            best_acc = acc
            best_t = t
    return float(best_t)


def _cluster_accuracy_ci(
    views: Sequence[ItemView],
    flags: Sequence[bool],
    n_bootstrap: int,
    alpha: float,
    seed: int,
) -> Tuple[float, float, float]:
    """Accuracy point estimate + clustered bootstrap CI over bootstrap units."""
    if not views or len(views) != len(flags):
        return float("nan"), float("nan"), float("nan")
    clusters: Dict[str, List[float]] = {}
    for v, f in zip(views, flags):
        clusters.setdefault(v.unit, []).append(1.0 if f else 0.0)
    # Per-cluster means, then mean of clusters (approx); for CI resample clusters.
    cluster_means = [sum(xs) / len(xs) for xs in clusters.values()]
    point = sum(1 for f in flags if f) / len(flags)
    if n_bootstrap <= 0 or len(cluster_means) < 2:
        return point, point, point
    low, high, _ = bootstrap_ci(
        cluster_means, n_bootstrap=n_bootstrap, alpha=alpha, seed=seed
    )
    return point, float(low), float(high)


def evaluate_cell(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Held-out policy table for one (dataset, subset, model)."""
    configs = list(configs) if configs else list(DEFAULT_CDV_CONFIG_IDS)
    views = _item_views(corpus, dataset, subset, model, configs, seed=seed)
    if len(views) < 8:
        return {
            "dataset": dataset,
            "subset": subset or "",
            "model": model,
            "n_items": len(views),
            "error": "too_few_items",
        }

    train, test = _split_views(views, val_fraction=val_fraction, split_seed=split_seed)
    in_sample = train is test or (
        {v.item_id for v in train} == {v.item_id for v in test}
    )
    chosen_global = _fit_val_selected_config(train, configs, reference)
    thr = _fit_diversify_threshold(train, configs, reference)

    policies: Dict[str, Callable[[ItemView], bool]] = {
        "sc": lambda v: _policy_sc(v, reference),
        # Always pool config modes (uniform CDV analogue on plug-in modes).
        "cdv_mode_vote": lambda v: _policy_conditional_diversify(
            v, configs, reference, threshold=1.01
        ),
        "margin_max": lambda v: _policy_margin_max(v, configs, reference),
        "agreement_else_margin": lambda v: _policy_agreement_else_margin(
            v, configs, reference
        ),
        "conditional_diversify": lambda v: _policy_conditional_diversify(
            v, configs, reference, thr
        ),
        "val_selected_global": lambda v: _policy_oracle_best_single(v, chosen_global),
        "oracle_config": lambda v: _policy_oracle_config(v, configs),
    }
    # Honest global oracle for gap: config with best train π_mode, evaluated on test
    # is val_selected_global. True oracle_best_single uses hindsight on all data —
    # report separately as oracle_config (per-item) and val_selected as achievable.

    rows: Dict[str, Any] = {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "n_items_total": len(views),
        "n_train": len(train),
        "n_test": len(test),
        "in_sample_split": bool(in_sample),
        "reference": reference,
        "val_selected_config": chosen_global,
        "diversify_threshold": thr,
        "n_bootstrap": n_bootstrap,
        "split_seed": split_seed,
        "val_fraction": val_fraction,
    }

    accs: Dict[str, float] = {}
    for name, fn in policies.items():
        flags = [fn(v) for v in test]
        point, lo, hi = _cluster_accuracy_ci(
            test, flags, n_bootstrap=n_bootstrap, alpha=alpha, seed=split_seed
        )
        accs[name] = point
        rows[f"acc_{name}"] = point
        rows[f"acc_{name}_ci_low"] = lo
        rows[f"acc_{name}_ci_high"] = hi

    sc = accs.get("sc", float("nan"))
    # Gap vs per-item oracle (strongest) and vs val-selected (fair single-config).
    gap_oracle = accs.get("oracle_config", float("nan")) - sc
    gap_valbest = accs.get("val_selected_global", float("nan")) - sc
    rows["gap_oracle_config_minus_sc"] = gap_oracle
    rows["gap_val_selected_minus_sc"] = gap_valbest

    for name in (
        "cdv_mode_vote",
        "margin_max",
        "agreement_else_margin",
        "conditional_diversify",
        "val_selected_global",
    ):
        delta = accs.get(name, float("nan")) - sc
        rows[f"delta_{name}_minus_sc"] = delta
        if math.isfinite(gap_oracle) and gap_oracle > 1e-9 and math.isfinite(delta):
            rows[f"frac_oracle_gap_{name}"] = delta / gap_oracle
        else:
            rows[f"frac_oracle_gap_{name}"] = float("nan")
        if math.isfinite(gap_valbest) and abs(gap_valbest) > 1e-9 and math.isfinite(delta):
            # For val_selected itself this is 1; for others vs that weaker ceiling.
            rows[f"frac_valbest_gap_{name}"] = delta / gap_valbest
        else:
            rows[f"frac_valbest_gap_{name}"] = float("nan")

        # Equivalence / superiority vs SC using the policy CI on accuracy:
        # approximate delta CI as (acc_ci - sc) using policy CI bounds (conservative
        # when sc is treated as fixed plug-in on the same test fold).
        lo = rows.get(f"acc_{name}_ci_low", float("nan"))
        hi = rows.get(f"acc_{name}_ci_high", float("nan"))
        if math.isfinite(lo) and math.isfinite(hi) and math.isfinite(sc):
            equiv = equivalence_from_ci(
                lo - sc, hi - sc, delta=POLICY_DELTA, point=delta
            )
            rows[f"equiv_vs_sc_{name}"] = equiv["verdict"]
        else:
            rows[f"equiv_vs_sc_{name}"] = "undefined"

    # Confirmatory: pre-specified PRIMARY_POLICY only (no test-fold policy pick).
    # Exploratory: best-of label-free retained for appendix, not for C3 verdict.
    primary = PRIMARY_POLICY
    rows["primary_policy"] = primary
    rows["primary_frac_oracle_gap"] = rows.get(f"frac_oracle_gap_{primary}")
    rows["primary_delta"] = rows.get(f"delta_{primary}_minus_sc")
    rows["primary_delta_vs_val_selected"] = (
        (accs.get(primary, float("nan")) - accs.get("val_selected_global", float("nan")))
        if math.isfinite(accs.get(primary, float("nan")))
        and math.isfinite(accs.get("val_selected_global", float("nan")))
        else float("nan")
    )
    label_free = ["margin_max", "agreement_else_margin", "conditional_diversify"]
    best_lf = max(
        label_free,
        key=lambda n: (
            rows.get(f"frac_oracle_gap_{n}")
            if math.isfinite(rows.get(f"frac_oracle_gap_{n}", float("nan")))
            else -1.0
        ),
    )
    rows["best_label_free_policy"] = best_lf  # exploratory only
    rows["best_label_free_frac_oracle_gap"] = rows.get(f"frac_oracle_gap_{best_lf}")
    rows["best_label_free_delta"] = rows.get(f"delta_{best_lf}_minus_sc")
    # Compute note: these policies read modes/margins from the crossed corpus
    # (same generation as CDV). Live inference that materialises all three
    # configs costs ~3x single-config SC; fair matched baseline is CDV / mode-vote.
    rows["compute_note"] = (
        "offline aggregator over crossed corpus; live cost matched to CDV "
        "(all configs sampled), not to single-config SC"
    )
    rows["prediction"] = "C3-oracle-gap"
    rows["statement"] = (
        f"pre-specified {primary} recovers a positive fraction of the "
        "oracle_config - SC modal gap on held-out items"
    )
    frac = rows.get("primary_frac_oracle_gap")
    delta = rows.get("primary_delta")
    rows["supported"] = bool(
        math.isfinite(frac)
        and frac > 0
        and math.isfinite(delta)
        and delta > 0
    )
    return rows


def evaluate_cell_multisplit(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    split_seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
) -> Dict[str, Any]:
    """Average the confirmatory primary policy over multiple held-out splits."""
    per_split: List[Dict[str, Any]] = []
    for ss in split_seeds:
        row = evaluate_cell(
            corpus,
            dataset,
            subset,
            model,
            configs=configs,
            reference=reference,
            seed=seed,
            val_fraction=val_fraction,
            split_seed=int(ss),
            n_bootstrap=n_bootstrap,
            alpha=alpha,
        )
        if not row.get("error"):
            per_split.append(row)
    if not per_split:
        return {
            "dataset": dataset,
            "subset": subset or "",
            "model": model,
            "error": "no_valid_splits",
        }
    # Aggregate primary metrics across splits.
    def _mean(key: str) -> float:
        vals = [
            float(r[key])
            for r in per_split
            if r.get(key) is not None and math.isfinite(float(r.get(key, float("nan"))))
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    def _std(key: str) -> float:
        vals = [
            float(r[key])
            for r in per_split
            if r.get(key) is not None and math.isfinite(float(r.get(key, float("nan"))))
        ]
        if len(vals) < 2:
            return float("nan")
        mu = sum(vals) / len(vals)
        return math.sqrt(sum((v - mu) ** 2 for v in vals) / (len(vals) - 1))

    base = dict(per_split[0])
    base["n_splits"] = len(per_split)
    base["split_seeds"] = ",".join(str(s) for s in split_seeds[: len(per_split)])
    for key in (
        "acc_sc",
        "acc_oracle_config",
        "acc_conditional_diversify",
        "acc_margin_max",
        "acc_val_selected_global",
        "acc_cdv_mode_vote",
        "gap_oracle_config_minus_sc",
        "primary_delta",
        "primary_frac_oracle_gap",
        "primary_delta_vs_val_selected",
        "delta_conditional_diversify_minus_sc",
        "delta_margin_max_minus_sc",
        "delta_cdv_mode_vote_minus_sc",
        "delta_val_selected_global_minus_sc",
        "frac_oracle_gap_conditional_diversify",
        "frac_oracle_gap_margin_max",
    ):
        base[key] = _mean(key)
        base[f"{key}_std_across_splits"] = _std(key)
    base["primary_policy"] = PRIMARY_POLICY
    base["supported"] = bool(
        math.isfinite(base.get("primary_frac_oracle_gap", float("nan")))
        and base["primary_frac_oracle_gap"] > 0
        and math.isfinite(base.get("primary_delta", float("nan")))
        and base["primary_delta"] > 0
    )
    return base


def seed_null_oracle_gap(
    corpus: Corpus,
    reference: str = REFERENCE_CONFIG_ID,
    seeds: Sequence[int] = (0, 1, 2),
) -> List[Dict[str, Any]]:
    """Oracle gap when 'configs' are seed replicates at the reference prompt.

    If plug-in mode noise alone creates a large oracle−SC gap, the elicitation
    oracle gap is not identified. Claude/ChatGPT reviews both requested this null.
    """
    rows: List[Dict[str, Any]] = []
    seed_list = list(seeds)
    for dataset, subset in corpus.benchmarks():
        for model in corpus.models():
            # Build ItemView-like structure across seeds at reference config.
            columns: Dict[int, Dict[str, Cell]] = {s: {} for s in seed_list}
            for cell in corpus.cells.values():
                if (
                    cell.dataset != dataset
                    or cell.subset != subset
                    or cell.model != model
                    or cell.config != reference
                    or cell.seed not in columns
                    or cell.n == 0
                ):
                    continue
                columns[cell.seed][cell.item_id] = cell
            shared = None
            for col in columns.values():
                ids = set(col)
                shared = ids if shared is None else shared & ids
            if not shared or len(shared) < 8:
                continue
            if any(len(columns[s]) == 0 for s in seed_list):
                continue
            n = 0
            n_sc = 0  # seed 0 modal correct
            n_oracle = 0  # any seed modal correct
            for item in shared:
                flags = [columns[s][item].modal_correct() for s in seed_list]
                n += 1
                if flags[0]:
                    n_sc += 1
                if any(flags):
                    n_oracle += 1
            gap = (n_oracle - n_sc) / n if n else float("nan")
            rows.append(
                {
                    "dataset": dataset,
                    "subset": subset or "",
                    "model": model,
                    "n_items": n,
                    "acc_seed0": n_sc / n if n else float("nan"),
                    "acc_seed_oracle": n_oracle / n if n else float("nan"),
                    "gap_seed_oracle_minus_seed0": gap,
                    "n_seeds": len(seed_list),
                    "null_kind": "c0_seed_replicates_as_configs",
                }
            )
    return rows


def evaluate_corpus(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    alpha: float = DEFAULT_ALPHA,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    split_seed: int = DEFAULT_SPLIT_SEED,
    split_seeds: Optional[Sequence[int]] = None,
    multisplit: bool = True,
) -> List[Dict[str, Any]]:
    configs = list(configs) if configs else list(DEFAULT_CDV_CONFIG_IDS)
    seeds = list(split_seeds) if split_seeds is not None else list(DEFAULT_SPLIT_SEEDS)
    rows: List[Dict[str, Any]] = []
    for dataset, subset in corpus.benchmarks():
        for model in corpus.models():
            present = {
                c.config
                for c in corpus.cells.values()
                if c.dataset == dataset
                and c.subset == subset
                and c.model == model
                and c.seed == seed
                and c.config in configs
            }
            if len(present) < 2:
                continue
            cfgs = [c for c in configs if c in present]
            ref = reference if reference in present else sorted(present)[0]
            if multisplit:
                row = evaluate_cell_multisplit(
                    corpus,
                    dataset,
                    subset,
                    model,
                    configs=cfgs,
                    reference=ref,
                    seed=seed,
                    val_fraction=val_fraction,
                    split_seeds=seeds,
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                )
            else:
                row = evaluate_cell(
                    corpus,
                    dataset,
                    subset,
                    model,
                    configs=cfgs,
                    reference=ref,
                    seed=seed,
                    val_fraction=val_fraction,
                    split_seed=split_seed,
                    n_bootstrap=n_bootstrap,
                    alpha=alpha,
                )
            if row.get("error"):
                log.info(
                    "selection skip %s/%s/%s: %s",
                    dataset,
                    subset,
                    model,
                    row["error"],
                )
                continue
            rows.append(row)
    return rows


def selection_summary(rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Corpus-level C3 verdict for predictions.csv (primary policy only)."""
    usable = [
        r
        for r in rows
        if math.isfinite(r.get("primary_frac_oracle_gap", float("nan")))
    ]
    if not usable:
        # Fall back for older CSVs that only have best_label_free_*.
        usable = [
            r
            for r in rows
            if math.isfinite(r.get("best_label_free_frac_oracle_gap", float("nan")))
        ]
        if usable:
            fracs = [float(r["best_label_free_frac_oracle_gap"]) for r in usable]
            deltas = [
                float(r["best_label_free_delta"])
                for r in usable
                if math.isfinite(r.get("best_label_free_delta", float("nan")))
            ]
            mean_frac = sum(fracs) / len(fracs)
            mean_delta = sum(deltas) / len(deltas) if deltas else float("nan")
            return {
                "prediction": "C3-oracle-gap",
                "n_cells": len(usable),
                "primary_policy": "legacy_best_of_label_free",
                "mean_primary_frac_oracle_gap": mean_frac,
                "mean_primary_delta": mean_delta,
                "supported": False,
                "statement": "legacy best-of metric; re-run selection for confirmatory C3",
            }
        return {
            "prediction": "C3-oracle-gap",
            "n_cells": 0,
            "supported": None,
            "primary_policy": PRIMARY_POLICY,
            "statement": (
                f"pre-specified {PRIMARY_POLICY} recovers a positive fraction "
                "of the oracle_config - SC modal gap on held-out items"
            ),
        }
    fracs = [float(r["primary_frac_oracle_gap"]) for r in usable]
    deltas = [
        float(r["primary_delta"])
        for r in usable
        if math.isfinite(r.get("primary_delta", float("nan")))
    ]
    mean_frac = sum(fracs) / len(fracs)
    mean_delta = sum(deltas) / len(deltas) if deltas else float("nan")
    # Ratio of means (stable) vs mean of ratios (can differ).
    gaps = [
        float(r["gap_oracle_config_minus_sc"])
        for r in usable
        if math.isfinite(r.get("gap_oracle_config_minus_sc", float("nan")))
    ]
    mean_gap = sum(gaps) / len(gaps) if gaps else float("nan")
    ratio_of_means = (
        mean_delta / mean_gap
        if math.isfinite(mean_gap) and abs(mean_gap) > 1e-9 and math.isfinite(mean_delta)
        else float("nan")
    )
    n_positive = sum(1 for f in fracs if f > 0)
    return {
        "prediction": "C3-oracle-gap",
        "n_cells": len(usable),
        "primary_policy": PRIMARY_POLICY,
        "mean_primary_frac_oracle_gap": mean_frac,
        "mean_primary_delta": mean_delta,
        "mean_oracle_gap": mean_gap,
        "ratio_of_means_delta_over_gap": ratio_of_means,
        "n_cells_positive_frac": n_positive,
        "supported": bool(mean_frac > 0 and n_positive >= max(1, len(usable) // 2)),
        "statement": (
            f"pre-specified {PRIMARY_POLICY} recovers a positive fraction of "
            "the oracle_config - SC modal gap on held-out items (multi-split)"
        ),
        "falsification_hook": (
            "if mean_primary_frac_oracle_gap <= 0, C3 is unsupported; "
            "paper must lead with the CDV negative and oracle-gap measurement only"
        ),
    }
