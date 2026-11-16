"""CPU-side C3 bulletproofing that does not require new GPU inference.

Builds on the crossed ``results_fp16`` corpus already graded:

1. **Expected-token cost** of probe-then-commit ``conditional_diversify``
   (always sample ``c0``; sample ``c1``/``c2`` only when ``c0`` margin < τ).
2. **Independent margin-split**: estimate margin on half of ``c0`` samples,
   apply the diversify decision using the held-out half.
3. **Quality-matched / quality-weighted** pooling: drop weak ``c2``, weight
   configs by train accuracy, and compare sample-level mixture plurality.

All entry points take a :class:`~src.analysis.corpus.Corpus` from
``load_corpus``; nothing here launches generation.
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..elicitation import PRIMARY_CONFIG_IDS, REFERENCE_CONFIG_ID
from ..metrics import bootstrap_ci
from . import c3_controls as ctrl
from . import selection as sel
from .corpus import Corpus

log = logging.getLogger(__name__)

DEFAULT_SPLIT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_CELL_BOOTSTRAP = 20_000


def _mean(vals: Sequence[float]) -> float:
    finite = [float(v) for v in vals if math.isfinite(float(v))]
    return sum(finite) / len(finite) if finite else float("nan")


def _tokens(samples: Sequence[Any]) -> float:
    return float(sum(int(getattr(s, "tokens_completion", 0) or 0) for s in samples))


def _split_samples(
    samples: Sequence[Any], split_seed: int = 0
) -> Tuple[List[Any], List[Any]]:
    """Deterministic half-split of samples (A = margin half, B = decision half)."""
    import numpy as np

    if len(samples) < 2:
        return list(samples), list(samples)
    order = list(range(len(samples)))
    rng = np.random.default_rng(split_seed)
    rng.shuffle(order)
    mid = len(order) // 2
    # Keep both halves non-empty.
    mid = min(max(1, mid), len(order) - 1)
    a = [samples[i] for i in sorted(order[:mid])]
    b = [samples[i] for i in sorted(order[mid:])]
    return a, b


def _mode_correct_from_samples(samples: Sequence[Any]) -> Tuple[Optional[str], bool]:
    return ctrl._modal_from_samples(samples)


def _weighted_plurality_correct(
    view: sel.ItemView,
    configs: Sequence[str],
    weights: Dict[str, float],
) -> bool:
    avail = view.available(configs)
    votes: Dict[str, float] = {}
    for c in avail:
        mode = view.modes.get(c)
        if mode is None:
            continue
        votes[mode] = votes.get(mode, 0.0) + float(weights.get(c, 1.0))
    if not votes:
        return False
    top = max(votes.values())
    winners = {m for m, w in votes.items() if w == top}
    for c in avail:
        mode = view.modes.get(c)
        if mode in winners and view.correct.get(c, False):
            return True
    return False


def _policy_weighted_diversify(
    view: sel.ItemView,
    configs: Sequence[str],
    reference: str,
    threshold: float,
    weights: Dict[str, float],
) -> bool:
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
    return _weighted_plurality_correct(view, configs, weights)


def _fit_weighted_diversify_threshold(
    train: Sequence[sel.ItemView],
    configs: Sequence[str],
    reference: str,
    weights: Dict[str, float],
) -> float:
    best_t, best_acc = sel.MARGIN_THRESHOLDS[0], -1.0
    for t in sel.MARGIN_THRESHOLDS:
        flags = [
            _policy_weighted_diversify(v, configs, reference, t, weights) for v in train
        ]
        acc = ctrl._acc(flags)
        if math.isfinite(acc) and acc > best_acc:
            best_acc = acc
            best_t = t
    return float(best_t)


def _train_config_weights(
    train: Sequence[sel.ItemView], configs: Sequence[str]
) -> Dict[str, float]:
    weights: Dict[str, float] = {}
    for c in configs:
        flags = [bool(v.correct.get(c, False)) for v in train if c in v.correct]
        acc = ctrl._acc(flags)
        # Floor so a weak config still gets a tiny vote rather than vanishing.
        weights[c] = max(0.05, float(acc) if math.isfinite(acc) else 0.05)
    return weights


# --------------------------------------------------------------------------- expected tokens


def expected_tokens_for_cell(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
    split_seed: int = 0,
) -> Dict[str, Any]:
    """Mean completion tokens under probe-then-commit CDV vs matched baselines."""
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS)
    cfg_views, cfg_pool, per_cfg = ctrl._config_item_views_and_pool(
        corpus, dataset, subset, model, configs, seed=seed
    )
    if len(cfg_views) < 8:
        return {
            "dataset": dataset,
            "subset": subset or "",
            "model": model,
            "error": "too_few_config_items",
        }

    train, test = sel._split_views(
        cfg_views, val_fraction=val_fraction, split_seed=split_seed
    )
    thr = sel._fit_diversify_threshold(train, configs, reference)
    others = [c for c in configs if c != reference]

    seed_ids = ctrl.available_c0_seeds(corpus, dataset, subset, model)
    _, seed_pool = ctrl._seed_item_views(
        corpus, dataset, subset, model, reference=reference, seeds=seed_ids
    )

    rows_tok: List[Dict[str, float]] = []
    n_early = 0
    for v in test:
        c0 = per_cfg.get(reference, {}).get(v.item_id, [])
        tok_c0 = _tokens(c0)
        tok_others = sum(
            _tokens(per_cfg.get(c, {}).get(v.item_id, [])) for c in others
        )
        tok_all_cfg = tok_c0 + tok_others
        margin = float(v.margins.get(reference, float("nan")))
        early = bool(math.isfinite(margin) and margin >= thr)
        if early:
            n_early += 1
            tok_probe = tok_c0
        else:
            tok_probe = tok_all_cfg
        tok_sc3n_seeds = _tokens(seed_pool.get(v.item_id, [])) if v.item_id in seed_pool else float("nan")
        # SC@3N via configs ≈ always sample the three-way pool.
        rows_tok.append(
            {
                "tok_probe_then_commit": tok_probe,
                "tok_always_3cfg": tok_all_cfg,
                "tok_sc_n": tok_c0,
                "tok_sc_3n_configs": tok_all_cfg,
                "tok_sc_3n_seeds": tok_sc3n_seeds,
            }
        )

    n_test = len(test)
    out: Dict[str, Any] = {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "n_test": n_test,
        "n_train": len(train),
        "diversify_threshold": thr,
        "n_configs": len(configs),
        "n_c0_seeds": len(seed_ids),
        "frac_early_exit": (n_early / n_test) if n_test else float("nan"),
        "mean_tok_probe_then_commit": _mean([r["tok_probe_then_commit"] for r in rows_tok]),
        "mean_tok_always_3cfg": _mean([r["tok_always_3cfg"] for r in rows_tok]),
        "mean_tok_sc_n": _mean([r["tok_sc_n"] for r in rows_tok]),
        "mean_tok_sc_3n_configs": _mean([r["tok_sc_3n_configs"] for r in rows_tok]),
        "mean_tok_sc_3n_seeds": _mean([r["tok_sc_3n_seeds"] for r in rows_tok]),
    }
    sc_n = out["mean_tok_sc_n"]
    if math.isfinite(sc_n) and sc_n > 0:
        out["ratio_probe_to_sc_n"] = out["mean_tok_probe_then_commit"] / sc_n
        out["ratio_always3_to_sc_n"] = out["mean_tok_always_3cfg"] / sc_n
        out["ratio_probe_to_always3"] = (
            out["mean_tok_probe_then_commit"] / out["mean_tok_always_3cfg"]
            if out["mean_tok_always_3cfg"] > 0
            else float("nan")
        )
        if math.isfinite(out["mean_tok_sc_3n_seeds"]) and out["mean_tok_sc_3n_seeds"] > 0:
            out["ratio_probe_to_sc_3n_seeds"] = (
                out["mean_tok_probe_then_commit"] / out["mean_tok_sc_3n_seeds"]
            )
        else:
            out["ratio_probe_to_sc_3n_seeds"] = float("nan")
    return out


def expected_tokens_corpus(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    split_seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
) -> List[Dict[str, Any]]:
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS)
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
                and c.n > 0
            }
            if len(present) < 2 or reference not in present:
                continue
            use = [c for c in configs if c in present]
            per: List[Dict[str, Any]] = []
            for ss in split_seeds:
                r = expected_tokens_for_cell(
                    corpus,
                    dataset,
                    subset,
                    model,
                    configs=use,
                    reference=reference,
                    seed=seed,
                    val_fraction=val_fraction,
                    split_seed=int(ss),
                )
                if not r.get("error"):
                    per.append(r)
            if not per:
                continue
            rows.append(_average_numeric_rows(per, dataset, subset, model))
    return rows


# --------------------------------------------------------------------------- margin split


def margin_split_for_cell(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
    split_seed: int = 0,
    sample_split_seed: int = 0,
) -> Dict[str, Any]:
    """Margin from c0 half-A; CDV decision / SC from half-B (+ other configs full)."""
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS)
    # Need raw per-config samples.
    columns: Dict[str, Dict[str, Any]] = {c: {} for c in configs}
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
    if not shared or len(shared) < 8:
        return {
            "dataset": dataset,
            "subset": subset or "",
            "model": model,
            "error": "too_few_config_items",
        }

    views: List[sel.ItemView] = []
    for item_id in sorted(shared):
        modes: Dict[str, Optional[str]] = {}
        correct: Dict[str, bool] = {}
        margins: Dict[str, float] = {}
        n: Dict[str, int] = {}
        unit = None
        for c in configs:
            cell = columns[c][item_id]
            unit = cell.bootstrap_unit()
            if c == reference:
                # Stable per-item salt (avoid PYTHONHASHSEED nondeterminism).
                salt = sum((i + 1) * ord(ch) for i, ch in enumerate(item_id)) % 10_007
                half_a, half_b = _split_samples(
                    cell.samples, split_seed=sample_split_seed + salt
                )
                margins[c] = ctrl._margin_from_samples(half_a)
                mode_b, ok_b = _mode_correct_from_samples(half_b)
                modes[c] = mode_b
                correct[c] = ok_b
                n[c] = len(half_b)
            else:
                modes[c] = cell.modal_class()
                correct[c] = cell.modal_correct()
                margins[c] = cell.margin()
                n[c] = cell.n
        views.append(
            sel.ItemView(
                item_id=item_id,
                unit=str(unit or item_id),
                modes=modes,
                correct=correct,
                margins=margins,
                n=n,
            )
        )

    train, test = sel._split_views(
        views, val_fraction=val_fraction, split_seed=split_seed
    )
    thr = sel._fit_diversify_threshold(train, configs, reference)
    sc = [bool(v.correct.get(reference, False)) for v in test]
    cd = [
        sel._policy_conditional_diversify(v, configs, reference, thr) for v in test
    ]
    # Full-sample (non-split) baseline on the same test items for comparison.
    full_views, _, _ = ctrl._config_item_views_and_pool(
        corpus, dataset, subset, model, configs, seed=seed
    )
    full_by_id = {v.item_id: v for v in full_views}
    full_test = [full_by_id[v.item_id] for v in test if v.item_id in full_by_id]
    thr_full = sel._fit_diversify_threshold(
        [full_by_id[v.item_id] for v in train if v.item_id in full_by_id],
        configs,
        reference,
    ) if full_test else thr
    cd_full = [
        sel._policy_conditional_diversify(v, configs, reference, thr_full)
        for v in full_test
    ]
    sc_full = [bool(v.correct.get(reference, False)) for v in full_test]

    return {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "n_test": len(test),
        "n_train": len(train),
        "diversify_threshold_margin_split": thr,
        "diversify_threshold_full": thr_full,
        "acc_sc_half_b": ctrl._acc(sc),
        "acc_cd_margin_split": ctrl._acc(cd),
        "delta_cd_margin_split_minus_sc": ctrl._acc(cd) - ctrl._acc(sc),
        "acc_sc_full": ctrl._acc(sc_full),
        "acc_cd_full": ctrl._acc(cd_full),
        "delta_cd_full_minus_sc": ctrl._acc(cd_full) - ctrl._acc(sc_full),
        "delta_margin_split_minus_full_cd": ctrl._acc(cd) - ctrl._acc(cd_full),
    }


def margin_split_corpus(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    split_seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
) -> List[Dict[str, Any]]:
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS)
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
                and c.n > 0
            }
            if len(present) < 2 or reference not in present:
                continue
            use = [c for c in configs if c in present]
            per: List[Dict[str, Any]] = []
            for ss in split_seeds:
                r = margin_split_for_cell(
                    corpus,
                    dataset,
                    subset,
                    model,
                    configs=use,
                    reference=reference,
                    seed=seed,
                    val_fraction=val_fraction,
                    split_seed=int(ss),
                    sample_split_seed=int(ss),
                )
                if not r.get("error"):
                    per.append(r)
            if not per:
                continue
            rows.append(_average_numeric_rows(per, dataset, subset, model))
    return rows


# --------------------------------------------------------------------------- quality-matched


def quality_matched_for_cell(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
    split_seed: int = 0,
) -> Dict[str, Any]:
    """CDV on {c0,c1}, quality-weighted CDV, and sample-level mixture plurality."""
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS)
    cfg_views, cfg_pool, per_cfg = ctrl._config_item_views_and_pool(
        corpus, dataset, subset, model, configs, seed=seed
    )
    if len(cfg_views) < 8:
        return {
            "dataset": dataset,
            "subset": subset or "",
            "model": model,
            "error": "too_few_config_items",
        }

    train, test = sel._split_views(
        cfg_views, val_fraction=val_fraction, split_seed=split_seed
    )
    thr = sel._fit_diversify_threshold(train, configs, reference)
    pair = [c for c in (reference, "c1") if c in configs]
    if reference not in pair:
        pair = [configs[0]] + [c for c in configs[1:2]]
    thr_pair = (
        sel._fit_diversify_threshold(train, pair, reference)
        if len(pair) >= 2
        else thr
    )
    weights = _train_config_weights(train, configs)
    thr_w = _fit_weighted_diversify_threshold(train, configs, reference, weights)

    seed_ids = ctrl.available_c0_seeds(corpus, dataset, subset, model)
    seed_views, seed_pool = ctrl._seed_item_views(
        corpus, dataset, subset, model, reference=reference, seeds=seed_ids
    )
    seed_by_id = {v.item_id: v for v in seed_views}

    sc = [bool(v.correct.get(reference, False)) for v in test]
    cd_full = [
        sel._policy_conditional_diversify(v, configs, reference, thr) for v in test
    ]
    cd_pair = [
        sel._policy_conditional_diversify(v, pair, reference, thr_pair) for v in test
    ]
    cd_weighted = [
        _policy_weighted_diversify(v, configs, reference, thr_w, weights) for v in test
    ]
    mix_samples = []
    for v in test:
        _, ok = ctrl._modal_from_samples(cfg_pool.get(v.item_id, []))
        mix_samples.append(ok)

    # Pair-only sample mixture (c0+c1).
    mix_pair = []
    for v in test:
        samples: List[Any] = []
        for c in pair:
            samples.extend(per_cfg.get(c, {}).get(v.item_id, []))
        _, ok = ctrl._modal_from_samples(samples)
        mix_pair.append(ok)

    row: Dict[str, Any] = {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "n_test": len(test),
        "n_train": len(train),
        "diversify_threshold": thr,
        "diversify_threshold_c0_c1": thr_pair,
        "diversify_threshold_weighted": thr_w,
        "weight_c0": weights.get(reference, float("nan")),
        "weight_c1": weights.get("c1", float("nan")),
        "weight_c2": weights.get("c2", float("nan")),
        "acc_sc_n": ctrl._acc(sc),
        "acc_cd_full3": ctrl._acc(cd_full),
        "acc_cd_c0_c1": ctrl._acc(cd_pair),
        "acc_cd_quality_weighted": ctrl._acc(cd_weighted),
        "acc_mixture_3cfg_samples": ctrl._acc(mix_samples),
        "acc_mixture_c0_c1_samples": ctrl._acc(mix_pair),
        "delta_cd_full3_minus_sc": ctrl._acc(cd_full) - ctrl._acc(sc),
        "delta_cd_c0_c1_minus_sc": ctrl._acc(cd_pair) - ctrl._acc(sc),
        "delta_cd_weighted_minus_sc": ctrl._acc(cd_weighted) - ctrl._acc(sc),
        "delta_mixture_3cfg_minus_sc": ctrl._acc(mix_samples) - ctrl._acc(sc),
        "delta_mixture_c0_c1_minus_sc": ctrl._acc(mix_pair) - ctrl._acc(sc),
        "delta_cd_full3_minus_c0_c1": ctrl._acc(cd_full) - ctrl._acc(cd_pair),
        "delta_cd_full3_minus_weighted": ctrl._acc(cd_full) - ctrl._acc(cd_weighted),
        "has_seed_control": False,
        "n_c0_seeds": len(seed_ids),
    }

    joint = [v for v in test if v.item_id in seed_by_id]
    if len(joint) >= 4 and len(seed_ids) >= 2:
        sc_j = [bool(v.correct.get(reference, False)) for v in joint]
        mix_j = []
        for v in joint:
            _, ok = ctrl._modal_from_samples(cfg_pool.get(v.item_id, []))
            mix_j.append(ok)
        sc3n = []
        for v in joint:
            _, ok = ctrl._modal_from_samples(seed_pool.get(v.item_id, []))
            sc3n.append(ok)
        cd_j = [
            sel._policy_conditional_diversify(v, configs, reference, thr) for v in joint
        ]
        row.update(
            {
                "has_seed_control": True,
                "n_joint_seed_items": len(joint),
                "acc_sc_n_joint": ctrl._acc(sc_j),
                "acc_cd_full3_joint": ctrl._acc(cd_j),
                "acc_mixture_3cfg_joint": ctrl._acc(mix_j),
                "acc_sc_3n_c0_seeds": ctrl._acc(sc3n),
                "delta_mixture_3cfg_minus_sc_joint": ctrl._acc(mix_j) - ctrl._acc(sc_j),
                "delta_mixture_3cfg_minus_sc_3n": ctrl._acc(mix_j) - ctrl._acc(sc3n),
                "delta_cd_full3_minus_sc_3n": ctrl._acc(cd_j) - ctrl._acc(sc3n),
                "delta_cd_full3_minus_mixture": ctrl._acc(cd_j) - ctrl._acc(mix_j),
            }
        )
    return row


def quality_matched_corpus(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    split_seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
) -> List[Dict[str, Any]]:
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS)
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
                and c.n > 0
            }
            if len(present) < 2 or reference not in present:
                continue
            use = [c for c in configs if c in present]
            per: List[Dict[str, Any]] = []
            for ss in split_seeds:
                r = quality_matched_for_cell(
                    corpus,
                    dataset,
                    subset,
                    model,
                    configs=use,
                    reference=reference,
                    seed=seed,
                    val_fraction=val_fraction,
                    split_seed=int(ss),
                )
                if not r.get("error"):
                    per.append(r)
            if not per:
                continue
            out = _average_numeric_rows(per, dataset, subset, model)
            out["has_seed_control"] = any(r.get("has_seed_control") for r in per)
            rows.append(out)
    return rows


# --------------------------------------------------------------------------- helpers


def _average_numeric_rows(
    per: Sequence[Dict[str, Any]],
    dataset: str,
    subset: Optional[str],
    model: str,
) -> Dict[str, Any]:
    keys = set().union(*(r.keys() for r in per))
    out: Dict[str, Any] = {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "n_splits": len(per),
    }
    for k in keys:
        if k in out or k in ("dataset", "subset", "model", "error"):
            continue
        vals = []
        for r in per:
            v = r.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if math.isfinite(float(v)):
                    vals.append(float(v))
        if vals:
            out[k] = sum(vals) / len(vals)
    return out


def summarise_delta_column(
    rows: Sequence[Dict[str, Any]],
    key: str,
    n_bootstrap: int = DEFAULT_CELL_BOOTSTRAP,
    seed: int = 0,
    predicate: Optional[Any] = None,
) -> Dict[str, Any]:
    """Cell-bootstrap mean and CI (in percentage points) for one delta column."""
    selected = [
        r
        for r in rows
        if (predicate is None or predicate(r))
        and math.isfinite(float(r.get(key, float("nan"))))
    ]
    vals = [100.0 * float(r[key]) for r in selected]
    if not vals:
        return {
            "n_cells": 0,
            "mean_pp": float("nan"),
            "ci_low_pp": float("nan"),
            "ci_high_pp": float("nan"),
            "cells": "",
        }
    low, high, point = bootstrap_ci(
        vals, n_bootstrap=n_bootstrap, alpha=0.05, seed=seed
    )
    return {
        "n_cells": len(vals),
        "mean_pp": point,
        "ci_low_pp": low,
        "ci_high_pp": high,
        "cells": ";".join(ctrl.short_cell_label(r) for r in selected),
    }
