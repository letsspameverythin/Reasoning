"""Decisive C3 controls from Claude/ChatGPT Weak-Reject reviews.

1. Seed-substitution: run conditional_diversify treating c0 seeds as
   fake configs. If gains match real configs, C3 is mostly extra samples.
2. SC@3N: plurality over pooled c0 samples across available seeds.
3. Config mixture@3N: plurality over pooled samples from {c0,c1,c2} at seed 0.

Seed discovery is dynamic: cells with seeds {0,1,2} today and later
Symbolic/Llama or seeds {3,4} enter automatically once graded JSONL lands.

These answer: is the +2.4pp about elicitation, or about spending more tokens?
"""

from __future__ import annotations

import logging
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..answers import EXTRACTION_FAILURE_CLASS
from ..elicitation import PRIMARY_CONFIG_IDS, REFERENCE_CONFIG_ID
from ..metrics import bootstrap_ci
from .corpus import Cell, Corpus
from . import selection as sel

log = logging.getLogger(__name__)

#: Historical default seed set; discovery may return a superset (e.g. 0..4).
SEED_IDS: Tuple[int, ...] = (0, 1, 2)
DEFAULT_CELL_BOOTSTRAP = 20_000


def available_c0_seeds(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    reference: str = REFERENCE_CONFIG_ID,
    min_items: int = 8,
) -> Tuple[int, ...]:
    """Sorted c0 seeds with enough non-empty cells for this (dataset, model)."""
    counts: Dict[int, int] = {}
    for cell in corpus.cells.values():
        if (
            cell.dataset != dataset
            or cell.subset != subset
            or cell.model != model
            or cell.config != reference
            or cell.n == 0
        ):
            continue
        counts[int(cell.seed)] = counts.get(int(cell.seed), 0) + 1
    seeds = tuple(sorted(s for s, n in counts.items() if n >= min_items))
    return seeds if seeds else (0,)


def short_cell_label(row: Dict[str, Any]) -> str:
    """Compact label matching ``tab_c3_controls_summary.csv`` cell strings."""
    model = str(row.get("model", ""))
    short = (
        model.replace("Qwen/Qwen2.5-", "")
        .replace("-Instruct", "")
        .replace("meta-llama/Llama-3.2-", "Llama")
    )
    dataset = str(row.get("dataset", ""))
    subset = str(row.get("subset") or "main")
    if dataset == "math500" and not row.get("subset"):
        subset = "main"
    return f"{short}|{dataset}|{subset}"


def _modal_from_samples(samples: Sequence[Any]) -> Tuple[Optional[str], bool]:
    if not samples:
        return None, False
    counts: Dict[str, int] = {}
    for s in samples:
        counts[s.canonical_class] = counts.get(s.canonical_class, 0) + 1
    top = max(counts.values())
    mode = None
    for s in samples:
        if counts[s.canonical_class] == top:
            mode = s.canonical_class
            break
    if mode is None or mode == EXTRACTION_FAILURE_CLASS:
        return mode, False
    ok = any(s.is_correct for s in samples if s.canonical_class == mode)
    return mode, ok


def _margin_from_samples(samples: Sequence[Any]) -> float:
    if not samples:
        return float("nan")
    counts = sorted(
        (
            sum(1 for s in samples if s.canonical_class == cls)
            for cls in {s.canonical_class for s in samples}
        ),
        reverse=True,
    )
    if not counts:
        return float("nan")
    second = counts[1] if len(counts) > 1 else 0
    return (counts[0] - second) / len(samples)


def _seed_item_views(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    reference: str = REFERENCE_CONFIG_ID,
    seeds: Sequence[int] = SEED_IDS,
) -> Tuple[List[sel.ItemView], Dict[str, List[Any]]]:
    """Build ItemViews with fake configs seed0/seed1/seed2; also return pooled samples."""
    columns: Dict[int, Dict[str, Cell]] = {int(s): {} for s in seeds}
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
    if not shared:
        return [], {}

    fake_configs = [f"seed{s}" for s in seeds]
    views: List[sel.ItemView] = []
    pooled: Dict[str, List[Any]] = {}
    for item_id in sorted(shared):
        modes: Dict[str, Optional[str]] = {}
        correct: Dict[str, bool] = {}
        margins: Dict[str, float] = {}
        n: Dict[str, int] = {}
        unit = None
        all_samples: List[Any] = []
        for s in seeds:
            cell = columns[int(s)][item_id]
            key = f"seed{s}"
            modes[key] = cell.modal_class()
            correct[key] = cell.modal_correct()
            margins[key] = cell.margin()
            n[key] = cell.n
            unit = cell.bootstrap_unit()
            all_samples.extend(cell.samples)
        pooled[item_id] = all_samples
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
    return views, pooled


def _config_item_views_and_pool(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Sequence[str],
    seed: int = 0,
) -> Tuple[List[sel.ItemView], Dict[str, List[Any]], Dict[str, List[Any]]]:
    """Config ItemViews + per-config sample lists + pooled-across-config samples."""
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
        return [], {}, {}

    views: List[sel.ItemView] = []
    per_cfg: Dict[str, Dict[str, List[Any]]] = {c: {} for c in configs}
    pooled: Dict[str, List[Any]] = {}
    for item_id in sorted(shared):
        modes: Dict[str, Optional[str]] = {}
        correct: Dict[str, bool] = {}
        margins: Dict[str, float] = {}
        n: Dict[str, int] = {}
        unit = None
        all_samples: List[Any] = []
        for c in configs:
            cell = columns[c][item_id]
            modes[c] = cell.modal_class()
            correct[c] = cell.modal_correct()
            margins[c] = cell.margin()
            n[c] = cell.n
            unit = cell.bootstrap_unit()
            per_cfg[c][item_id] = list(cell.samples)
            all_samples.extend(cell.samples)
        pooled[item_id] = all_samples
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
    return views, pooled, {c: per_cfg[c] for c in configs}


def _acc(flags: Sequence[bool]) -> float:
    return sum(1 for f in flags if f) / len(flags) if flags else float("nan")


def evaluate_controls_for_cell(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
    split_seed: int = 0,
    c0_seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """One (dataset, model) control table on a held-out split."""
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS)
    seed_ids = tuple(c0_seeds) if c0_seeds is not None else available_c0_seeds(
        corpus, dataset, subset, model, reference=reference
    )
    cfg_views, cfg_pool, per_cfg = _config_item_views_and_pool(
        corpus, dataset, subset, model, configs, seed=seed
    )
    seed_views, seed_pool = _seed_item_views(
        corpus, dataset, subset, model, reference=reference, seeds=seed_ids
    )
    if len(cfg_views) < 8:
        return {
            "dataset": dataset,
            "subset": subset or "",
            "model": model,
            "error": "too_few_config_items",
        }

    # Fit τ on train using config views (same as main C3).
    train, test = sel._split_views(cfg_views, val_fraction=val_fraction, split_seed=split_seed)
    thr = sel._fit_diversify_threshold(train, configs, reference)
    fake_cfgs = [f"seed{s}" for s in seed_ids]
    ref_seed_name = f"seed{seed_ids[0]}" if seed_ids else "seed0"
    # For seed substitution, fit τ on seed-views train units aligned by item where possible.
    seed_by_id = {v.item_id: v for v in seed_views}
    seed_train = [seed_by_id[v.item_id] for v in train if v.item_id in seed_by_id]
    seed_test = [seed_by_id[v.item_id] for v in test if v.item_id in seed_by_id]
    thr_seed = (
        sel._fit_diversify_threshold(seed_train, fake_cfgs, ref_seed_name)
        if len(seed_train) >= 4 and len(seed_ids) >= 2
        else thr
    )

    test_ids = [v.item_id for v in test]
    has_seeds = (
        len(seed_ids) >= 2
        and len(seed_test) >= 4
        and bool(set(test_ids) & set(seed_by_id))
    )

    # --- flags on config test fold ---
    sc_n = [bool(v.correct.get(reference, False)) for v in test]
    cd_cfg = [
        sel._policy_conditional_diversify(v, configs, reference, thr) for v in test
    ]
    mode_vote_cfg = [
        sel._policy_conditional_diversify(v, configs, reference, 1.01) for v in test
    ]
    # SC@3N via config pool (all samples from c0+c1+c2) — extra samples + diversity
    sc_mix = []
    for v in test:
        _, ok = _modal_from_samples(cfg_pool.get(v.item_id, []))
        sc_mix.append(ok)
    # SC@3N via c0 only if we have 3× samples at c0 (deep N or seeds)
    sc_c0_deep = []
    for v in test:
        samples = per_cfg.get(reference, {}).get(v.item_id, [])
        _, ok = _modal_from_samples(samples)  # same as SC@N when only seed0
        sc_c0_deep.append(ok)

    row: Dict[str, Any] = {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "n_test": len(test),
        "n_train": len(train),
        "diversify_threshold": thr,
        "has_seed_control": bool(has_seeds),
        "n_c0_seeds": len(seed_ids),
        "c0_seeds": ",".join(str(s) for s in seed_ids),
        "acc_sc_n": _acc(sc_n),
        "acc_conditional_diversify_configs": _acc(cd_cfg),
        "acc_mode_vote_configs": _acc(mode_vote_cfg),
        "acc_mixture_all_configs": _acc(sc_mix),
        "delta_cd_configs_minus_sc": _acc(cd_cfg) - _acc(sc_n),
        "delta_mode_vote_minus_sc": _acc(mode_vote_cfg) - _acc(sc_n),
        "delta_mixture_minus_sc": _acc(sc_mix) - _acc(sc_n),
    }

    if has_seeds:
        # Restrict to items present in both
        joint = [v for v in test if v.item_id in seed_by_id]
        seed_joint = [seed_by_id[v.item_id] for v in joint]
        sc_n_j = [bool(v.correct.get(reference, False)) for v in joint]
        cd_cfg_j = [
            sel._policy_conditional_diversify(v, configs, reference, thr) for v in joint
        ]
        cd_seed = [
            sel._policy_conditional_diversify(v, fake_cfgs, ref_seed_name, thr_seed)
            for v in seed_joint
        ]
        mode_vote_seed = [
            sel._policy_conditional_diversify(v, fake_cfgs, ref_seed_name, 1.01)
            for v in seed_joint
        ]
        sc_3n_seed = []
        for v in seed_joint:
            _, ok = _modal_from_samples(seed_pool.get(v.item_id, []))
            sc_3n_seed.append(ok)

        row.update(
            {
                "n_joint_seed_items": len(joint),
                "diversify_threshold_seed": thr_seed,
                "acc_sc_n_joint": _acc(sc_n_j),
                "acc_cd_configs_joint": _acc(cd_cfg_j),
                "acc_cd_seeds": _acc(cd_seed),
                "acc_mode_vote_seeds": _acc(mode_vote_seed),
                "acc_sc_3n_c0_seeds": _acc(sc_3n_seed),
                "delta_cd_configs_joint_minus_sc": _acc(cd_cfg_j) - _acc(sc_n_j),
                "delta_cd_seeds_minus_sc": _acc(cd_seed) - _acc(sc_n_j),
                "delta_sc_3n_seeds_minus_sc": _acc(sc_3n_seed) - _acc(sc_n_j),
                "delta_cd_configs_minus_cd_seeds": _acc(cd_cfg_j) - _acc(cd_seed),
                "delta_cd_configs_minus_sc_3n": _acc(cd_cfg_j) - _acc(sc_3n_seed),
            }
        )
    return row


def evaluate_controls_corpus(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    split_seeds: Sequence[int] = (0, 1, 2, 3, 4),
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
            if len(present) < 2:
                continue
            c0_seeds = available_c0_seeds(
                corpus, dataset, subset, model, reference=reference
            )
            per: List[Dict[str, Any]] = []
            for ss in split_seeds:
                r = evaluate_controls_for_cell(
                    corpus,
                    dataset,
                    subset,
                    model,
                    configs=[c for c in configs if c in present],
                    reference=reference if reference in present else sorted(present)[0],
                    seed=seed,
                    val_fraction=val_fraction,
                    split_seed=int(ss),
                    c0_seeds=c0_seeds,
                )
                if not r.get("error"):
                    per.append(r)
            if not per:
                continue
            # Average numeric fields across splits
            keys = set().union(*(r.keys() for r in per))
            out: Dict[str, Any] = {
                "dataset": dataset,
                "subset": subset or "",
                "model": model,
                "n_splits": len(per),
                "has_seed_control": any(r.get("has_seed_control") for r in per),
                "n_c0_seeds": len(c0_seeds),
                "c0_seeds": ",".join(str(s) for s in c0_seeds),
            }
            for k in keys:
                if k in out or k in ("dataset", "subset", "model", "error", "c0_seeds"):
                    continue
                vals = []
                for r in per:
                    v = r.get(k)
                    if isinstance(v, (int, float)) and not isinstance(v, bool):
                        if math.isfinite(float(v)):
                            vals.append(float(v))
                if vals:
                    out[k] = sum(vals) / len(vals)
            rows.append(out)
    return rows


def _boot_mean_pp(
    rows: Sequence[Dict[str, Any]],
    key: str,
    n_bootstrap: int,
    seed: int,
) -> Tuple[float, float, float, int]:
    vals = [
        100.0 * float(r[key])
        for r in rows
        if math.isfinite(float(r.get(key, float("nan"))))
    ]
    if not vals:
        return float("nan"), float("nan"), float("nan"), 0
    low, high, point = bootstrap_ci(
        vals, n_bootstrap=n_bootstrap, alpha=0.05, seed=seed
    )
    return point, low, high, len(vals)


def controls_summary_table(
    rows: Sequence[Dict[str, Any]],
    n_bootstrap: int = DEFAULT_CELL_BOOTSTRAP,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Contrast table with cell-bootstrap CIs (pp), ready for CSV / paper table.

    Contrast names keep the historical ``seed6_*`` / ``all14`` prefixes used in
    ``tab_c3_controls_summary.csv``; ``n_cells`` is whatever the corpus has now
    (6 seed cells today, up to 14 once Symbolic/Llama seeds land).
    """
    all_rows = [
        r
        for r in rows
        if math.isfinite(float(r.get("delta_cd_configs_minus_sc", float("nan"))))
    ]
    seed_rows = [
        r
        for r in rows
        if r.get("has_seed_control")
        and math.isfinite(float(r.get("delta_cd_seeds_minus_sc", float("nan"))))
    ]
    noseed_rows = [
        r
        for r in all_rows
        if not r.get("has_seed_control")
    ]

    specs: List[Tuple[str, Sequence[Dict[str, Any]], str]] = [
        ("all14", all_rows, "delta_cd_configs_minus_sc"),
        ("seed6_cfg_full", seed_rows, "delta_cd_configs_minus_sc"),
        ("seed6_cfg_joint", seed_rows, "delta_cd_configs_joint_minus_sc"),
        ("seed6_seed_cd", seed_rows, "delta_cd_seeds_minus_sc"),
        ("seed6_sc3n", seed_rows, "delta_sc_3n_seeds_minus_sc"),
        ("seed6_cfg_minus_seed", seed_rows, "delta_cd_configs_minus_cd_seeds"),
        ("seed6_cfg_minus_sc3n", seed_rows, "delta_cd_configs_minus_sc_3n"),
        ("noseed8_cfg", noseed_rows, "delta_cd_configs_minus_sc"),
    ]
    out_rows: List[Dict[str, Any]] = []
    for i, (name, rs, key) in enumerate(specs):
        mean_pp, lo, hi, n = _boot_mean_pp(
            rs, key, n_bootstrap=n_bootstrap, seed=seed + i
        )
        out_rows.append(
            {
                "contrast": name,
                "n_cells": n,
                "mean_pp": mean_pp,
                "ci_low_pp": lo,
                "ci_high_pp": hi,
                "cells": ";".join(short_cell_label(r) for r in rs),
            }
        )
    return out_rows


def controls_summary(
    rows: Sequence[Dict[str, Any]],
    n_bootstrap: int = DEFAULT_CELL_BOOTSTRAP,
    seed: int = 0,
) -> Dict[str, Any]:
    """Headline: does config CD beat seed CD and SC@3N?"""
    all_rows = [
        r
        for r in rows
        if math.isfinite(float(r.get("delta_cd_configs_minus_sc", float("nan"))))
    ]
    seed_rows = [
        r
        for r in rows
        if r.get("has_seed_control")
        and math.isfinite(float(r.get("delta_cd_seeds_minus_sc", float("nan"))))
    ]

    def mean(key: str, rs: Sequence[Dict[str, Any]]) -> float:
        vals = [
            float(r[key])
            for r in rs
            if math.isfinite(float(r.get(key, float("nan"))))
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    out: Dict[str, Any] = {
        "n_cells": len(all_rows),
        "mean_delta_cd_configs_minus_sc": mean("delta_cd_configs_minus_sc", all_rows),
        "mean_delta_mode_vote_minus_sc": mean("delta_mode_vote_minus_sc", all_rows),
        "mean_delta_mixture_minus_sc": mean("delta_mixture_minus_sc", all_rows),
        "n_seed_cells": len(seed_rows),
        "mean_delta_cd_seeds_minus_sc": mean("delta_cd_seeds_minus_sc", seed_rows),
        "mean_delta_sc_3n_seeds_minus_sc": mean(
            "delta_sc_3n_seeds_minus_sc", seed_rows
        ),
        "mean_delta_cd_configs_minus_cd_seeds": mean(
            "delta_cd_configs_minus_cd_seeds", seed_rows
        ),
        "mean_delta_cd_configs_minus_sc_3n": mean(
            "delta_cd_configs_minus_sc_3n", seed_rows
        ),
    }

    # Cell-bootstrap CIs on the two reviewer-critical residuals (pp).
    for key, out_key in (
        ("delta_cd_configs_minus_cd_seeds", "cfg_minus_seed"),
        ("delta_cd_configs_minus_sc_3n", "cfg_minus_sc3n"),
        ("delta_cd_configs_minus_sc", "cfg_minus_sc_all"),
    ):
        rs = seed_rows if "seed" in out_key or "sc3n" in out_key else all_rows
        if out_key == "cfg_minus_sc_all":
            rs = all_rows
        mean_pp, lo, hi, n = _boot_mean_pp(
            rs, key, n_bootstrap=n_bootstrap, seed=seed
        )
        out[f"ci_{out_key}_mean_pp"] = mean_pp
        out[f"ci_{out_key}_low_pp"] = lo
        out[f"ci_{out_key}_high_pp"] = hi
        out[f"ci_{out_key}_n"] = n

    cfg = out["mean_delta_cd_configs_minus_sc"]
    seed_cd = out["mean_delta_cd_seeds_minus_sc"]
    residual = out["mean_delta_cd_configs_minus_cd_seeds"]
    vs_3n = out["mean_delta_cd_configs_minus_sc_3n"]
    out["configs_add_beyond_seed_cd"] = bool(
        math.isfinite(residual) and residual > 0.005  # >0.5pp
    )
    out["configs_beat_sc_3n"] = bool(math.isfinite(vs_3n) and vs_3n > 0.005)
    out["c3_likely_sample_count"] = bool(
        math.isfinite(seed_cd)
        and math.isfinite(cfg)
        and seed_cd >= cfg - 0.005  # seed CD ≈ config CD
    )
    out["verdict"] = (
        "sample_count_confound"
        if out["c3_likely_sample_count"] and not out["configs_beat_sc_3n"]
        else (
            "configs_add_signal"
            if out["configs_add_beyond_seed_cd"] or out["configs_beat_sc_3n"]
            else "inconclusive"
        )
    )
    return out
