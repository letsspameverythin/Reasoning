"""Recompute C3 selectors under fractional credit vs fixed (any-winner) ties.

Claude review concern: gating/aggregation residuals (esp. seed-sub − SC@3N,
CD − SC@N) may be an any-winner mode-plurality tie artifact.

Fractional credit
-----------------
* Sample plurality (SC@N / SC@3N / SC@2N): if ``k`` classes tie for top count,
  credit = (# of those classes that are correct) / ``k``.
* Mode-of-modes / CD diversify / seed-sub: if ``k`` modes tie for plurality,
  credit = (# of those modes that are correct) / ``k``.
* When CD gates to the reference column, use that column's *sample* plurality
  under the same rule (not the binary ``modal_correct`` flag alone).

Fixed credit
------------
* Sample plurality: first-occurrence deterministic tie-break (matches
  ``Cell.modal_correct`` / ``_modal_from_samples``).
* Mode policies: paper any-winner rule (``_mode_plurality_correct``).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..answers import EXTRACTION_FAILURE_CLASS
from ..elicitation import PRIMARY_CONFIG_IDS, REFERENCE_CONFIG_ID
from ..metrics import bootstrap_ci
from . import c3_compute as c3c
from . import c3_controls as ctrl
from . import selection as sel
from .corpus import Cell, Corpus

log = logging.getLogger(__name__)

DEFAULT_SPLIT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_CELL_BOOTSTRAP = 20_000
DEFAULT_ITEM_BOOTSTRAP = 1000

Rule = str  # "fixed" | "fractional"


# --------------------------------------------------------------------------- credit primitives


def _top_tied_classes(counts: Dict[str, int]) -> List[str]:
    if not counts:
        return []
    top = max(counts.values())
    return [c for c, n in counts.items() if n == top]


def _class_is_correct(samples: Sequence[Any], cls: str) -> bool:
    if cls is None or cls == EXTRACTION_FAILURE_CLASS:
        return False
    return any(s.is_correct for s in samples if s.canonical_class == cls)


def sample_plurality_credit(
    samples: Sequence[Any], rule: Rule = "fractional"
) -> Tuple[float, bool]:
    """Return (credit in [0,1], is_tie) for sample plurality."""
    if not samples:
        return 0.0, False
    counts: Dict[str, int] = {}
    for s in samples:
        counts[s.canonical_class] = counts.get(s.canonical_class, 0) + 1
    tied = _top_tied_classes(counts)
    is_tie = len(tied) > 1
    if rule == "fixed":
        _, ok = ctrl._modal_from_samples(samples)
        return float(ok), is_tie
    n_ok = sum(1 for c in tied if _class_is_correct(samples, c))
    return (n_ok / len(tied) if tied else 0.0), is_tie


def mode_plurality_credit(
    view: sel.ItemView,
    columns: Sequence[str],
    rule: Rule = "fractional",
) -> Tuple[float, bool]:
    """Plurality over column plug-in modes → (credit, is_tie)."""
    votes: Dict[str, int] = {}
    col_for_mode: Dict[str, List[str]] = defaultdict(list)
    for c in columns:
        if c not in view.modes or view.n.get(c, 0) <= 0:
            continue
        mode = view.modes.get(c)
        if mode is None:
            continue
        votes[mode] = votes.get(mode, 0) + 1
        col_for_mode[mode].append(c)
    if not votes:
        return 0.0, False
    top = max(votes.values())
    tied = [m for m, n in votes.items() if n == top]
    is_tie = len(tied) > 1
    winners = set(tied)
    if rule == "fixed":
        # Any-winner (paper rule).
        ok = any(
            view.modes.get(c) in winners and view.correct.get(c, False)
            for c in columns
            if c in view.modes and view.n.get(c, 0) > 0
        )
        return float(ok), is_tie
    mode_ok = {
        m: any(view.correct.get(c, False) for c in col_for_mode[m]) for m in tied
    }
    return sum(1 for m in tied if mode_ok[m]) / len(tied), is_tie


def cd_credit(
    view: sel.ItemView,
    columns: Sequence[str],
    reference: str,
    threshold: float,
    rule: Rule = "fractional",
    ref_samples: Optional[Sequence[Any]] = None,
) -> Tuple[float, bool]:
    """CD aggregator credit; gated → reference sample plurality."""
    avail = [c for c in columns if c in view.modes and view.n.get(c, 0) > 0]
    if not avail:
        return 0.0, False
    ref_margin = view.margins.get(reference, 0.0)
    if (
        threshold <= 1.0
        and math.isfinite(ref_margin)
        and ref_margin >= threshold
        and reference in avail
    ):
        if ref_samples is not None:
            return sample_plurality_credit(ref_samples, rule=rule)
        ok = bool(view.correct.get(reference, False))
        return float(ok), False
    return mode_plurality_credit(view, columns, rule=rule)


def _mean_credit(credits: Sequence[float]) -> float:
    return sum(credits) / len(credits) if credits else float("nan")


def _tie_rate(flags: Sequence[bool]) -> float:
    return sum(1 for t in flags if t) / len(flags) if flags else float("nan")


# --------------------------------------------------------------------------- sample maps


def _c0_seed_sample_map(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    reference: str,
    seeds: Sequence[int],
) -> Dict[int, Dict[str, List[Any]]]:
    out: Dict[int, Dict[str, List[Any]]] = {int(s): {} for s in seeds}
    for cell in corpus.cells.values():
        if (
            cell.dataset != dataset
            or cell.subset != subset
            or cell.model != model
            or cell.config != reference
            or cell.seed not in out
            or cell.n == 0
        ):
            continue
        out[int(cell.seed)][cell.item_id] = list(cell.samples)
    return out


# --------------------------------------------------------------------------- per-cell eval


def evaluate_cell_under_rule(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    rule: Rule,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
    split_seed: int = 0,
    c0_seeds: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """One (dataset, model) control table under ``rule`` (fixed|fractional)."""
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS)
    seed_ids = (
        tuple(c0_seeds)
        if c0_seeds is not None
        else ctrl.available_c0_seeds(
            corpus, dataset, subset, model, reference=reference
        )
    )
    cfg_views, _cfg_pool, per_cfg = ctrl._config_item_views_and_pool(
        corpus, dataset, subset, model, configs, seed=seed
    )
    seed_views, seed_pool = ctrl._seed_item_views(
        corpus, dataset, subset, model, reference=reference, seeds=seed_ids
    )
    seed_samples = _c0_seed_sample_map(
        corpus, dataset, subset, model, reference, seed_ids
    )
    if len(cfg_views) < 8:
        return {
            "dataset": dataset,
            "subset": subset or "",
            "model": model,
            "rule": rule,
            "error": "too_few_config_items",
        }

    train, test = sel._split_views(
        cfg_views, val_fraction=val_fraction, split_seed=split_seed
    )
    thr = sel._fit_diversify_threshold(train, configs, reference)
    fake_cfgs = [f"seed{s}" for s in seed_ids]
    ref_seed_name = f"seed{seed_ids[0]}" if seed_ids else "seed0"
    seed_by_id = {v.item_id: v for v in seed_views}
    seed_train = [seed_by_id[v.item_id] for v in train if v.item_id in seed_by_id]
    thr_seed = (
        sel._fit_diversify_threshold(seed_train, fake_cfgs, ref_seed_name)
        if len(seed_train) >= 4 and len(seed_ids) >= 2
        else thr
    )

    test_ids = [v.item_id for v in test]
    has_seeds = (
        len(seed_ids) >= 2
        and len([i for i in test_ids if i in seed_by_id]) >= 4
        and bool(set(test_ids) & set(seed_by_id))
    )

    def _ref_samps(item_id: str) -> Optional[List[Any]]:
        return per_cfg.get(reference, {}).get(item_id)

    sc_creds, sc_ties = [], []
    cd_creds, cd_ties = [], []
    mom_creds, mom_ties = [], []
    for v in test:
        c, t = sample_plurality_credit(_ref_samps(v.item_id) or [], rule=rule)
        sc_creds.append(c)
        sc_ties.append(t)
        c, t = cd_credit(
            v, configs, reference, thr, rule=rule, ref_samples=_ref_samps(v.item_id)
        )
        cd_creds.append(c)
        cd_ties.append(t)
        c, t = mode_plurality_credit(v, configs, rule=rule)
        mom_creds.append(c)
        mom_ties.append(t)

    row: Dict[str, Any] = {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "rule": rule,
        "n_test": len(test),
        "n_train": len(train),
        "diversify_threshold": thr,
        "has_seed_control": bool(has_seeds),
        "n_c0_seeds": len(seed_ids),
        "c0_seeds": ",".join(str(s) for s in seed_ids),
        "acc_sc_n": _mean_credit(sc_creds),
        "acc_conditional_diversify_configs": _mean_credit(cd_creds),
        "acc_mode_vote_configs": _mean_credit(mom_creds),
        "tie_rate_sc_n": _tie_rate(sc_ties),
        "tie_rate_cd": _tie_rate(cd_ties),
        "tie_rate_mode_of_modes": _tie_rate(mom_ties),
        "delta_cd_configs_minus_sc": _mean_credit(cd_creds) - _mean_credit(sc_creds),
        "delta_mode_vote_minus_sc": _mean_credit(mom_creds) - _mean_credit(sc_creds),
        "delta_cd_gated_minus_mode_vote_full": (
            _mean_credit(cd_creds) - _mean_credit(mom_creds)
        ),
    }

    if has_seeds:
        joint = [v for v in test if v.item_id in seed_by_id]
        seed_joint = [seed_by_id[v.item_id] for v in joint]
        ref_seed_idx = int(seed_ids[0])

        sc_j, sc_j_t = [], []
        cd_j, cd_j_t = [], []
        mom_j, mom_j_t = [], []
        seed_cd, seed_cd_t = [], []
        sc3, sc3_t = [], []
        for v, sv in zip(joint, seed_joint):
            c, t = sample_plurality_credit(_ref_samps(v.item_id) or [], rule=rule)
            sc_j.append(c)
            sc_j_t.append(t)
            c, t = cd_credit(
                v,
                configs,
                reference,
                thr,
                rule=rule,
                ref_samples=_ref_samps(v.item_id),
            )
            cd_j.append(c)
            cd_j_t.append(t)
            c, t = mode_plurality_credit(v, configs, rule=rule)
            mom_j.append(c)
            mom_j_t.append(t)
            ref_seed_samps = seed_samples.get(ref_seed_idx, {}).get(sv.item_id)
            c, t = cd_credit(
                sv,
                fake_cfgs,
                ref_seed_name,
                thr_seed,
                rule=rule,
                ref_samples=ref_seed_samps,
            )
            seed_cd.append(c)
            seed_cd_t.append(t)
            c, t = sample_plurality_credit(
                seed_pool.get(sv.item_id, []), rule=rule
            )
            sc3.append(c)
            sc3_t.append(t)

        row.update(
            {
                "n_joint_seed_items": len(joint),
                "diversify_threshold_seed": thr_seed,
                "acc_sc_n_joint": _mean_credit(sc_j),
                "acc_cd_configs_joint": _mean_credit(cd_j),
                "acc_cd_seeds": _mean_credit(seed_cd),
                "acc_mode_vote_configs_joint": _mean_credit(mom_j),
                "acc_sc_3n_c0_seeds": _mean_credit(sc3),
                "tie_rate_sc_n_joint": _tie_rate(sc_j_t),
                "tie_rate_cd_joint": _tie_rate(cd_j_t),
                "tie_rate_seed_cd": _tie_rate(seed_cd_t),
                "tie_rate_sc_3n": _tie_rate(sc3_t),
                "tie_rate_mode_of_modes_joint": _tie_rate(mom_j_t),
                "delta_cd_configs_joint_minus_sc": (
                    _mean_credit(cd_j) - _mean_credit(sc_j)
                ),
                "delta_cd_seeds_minus_sc": (
                    _mean_credit(seed_cd) - _mean_credit(sc_j)
                ),
                "delta_sc_3n_seeds_minus_sc": (
                    _mean_credit(sc3) - _mean_credit(sc_j)
                ),
                "delta_cd_configs_minus_cd_seeds": (
                    _mean_credit(cd_j) - _mean_credit(seed_cd)
                ),
                "delta_cd_configs_minus_sc_3n": (
                    _mean_credit(cd_j) - _mean_credit(sc3)
                ),
                "delta_cd_seeds_minus_sc_3n": (
                    _mean_credit(seed_cd) - _mean_credit(sc3)
                ),
                "delta_mode_vote_minus_sc_joint": (
                    _mean_credit(mom_j) - _mean_credit(sc_j)
                ),
                "delta_mode_vote_minus_sc_3n": (
                    _mean_credit(mom_j) - _mean_credit(sc3)
                ),
                "delta_cd_gated_minus_mode_vote": (
                    _mean_credit(cd_j) - _mean_credit(mom_j)
                ),
            }
        )
    return row


def drop_c2_cell_under_rule(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    rule: Rule,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
    split_seed: int = 0,
) -> Dict[str, Any]:
    """Global drop-c2 CD{c0,c1} vs two-seed CD / SC@2N under ``rule``."""
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS)
    pair = [c for c in (reference, "c1") if c in configs]
    if len(pair) < 2:
        return {
            "dataset": dataset,
            "subset": subset or "",
            "model": model,
            "rule": rule,
            "error": "need_c0_and_c1",
        }

    cfg_views, _, per_cfg = ctrl._config_item_views_and_pool(
        corpus, dataset, subset, model, configs, seed=seed
    )
    if len(cfg_views) < 8:
        return {
            "dataset": dataset,
            "subset": subset or "",
            "model": model,
            "rule": rule,
            "error": "too_few_config_items",
        }

    all_seeds = ctrl.available_c0_seeds(
        corpus, dataset, subset, model, reference=reference
    )
    two_seeds = tuple(s for s in (0, 1) if s in all_seeds)
    if len(two_seeds) < 2 and len(all_seeds) >= 2:
        two_seeds = tuple(all_seeds[:2])
    seed_views, seed_pool = ctrl._seed_item_views(
        corpus, dataset, subset, model, reference=reference, seeds=two_seeds
    )
    seed_samples = _c0_seed_sample_map(
        corpus, dataset, subset, model, reference, two_seeds
    )
    seed_by_id = {v.item_id: v for v in seed_views}

    train, test = sel._split_views(
        cfg_views, val_fraction=val_fraction, split_seed=split_seed
    )
    thr_pair = sel._fit_diversify_threshold(train, pair, reference)
    thr_full = sel._fit_diversify_threshold(train, configs, reference)
    fake_cfgs = [f"seed{s}" for s in two_seeds]
    ref_seed_name = f"seed{two_seeds[0]}" if two_seeds else "seed0"
    seed_train = [seed_by_id[v.item_id] for v in train if v.item_id in seed_by_id]
    thr_seed = (
        sel._fit_diversify_threshold(seed_train, fake_cfgs, ref_seed_name)
        if len(seed_train) >= 4 and len(two_seeds) >= 2
        else thr_pair
    )

    def _ref_samps(item_id: str) -> Optional[List[Any]]:
        return per_cfg.get(reference, {}).get(item_id)

    sc, sc_t = [], []
    cd_pair, cd_pair_t = [], []
    cd_full, cd_full_t = [], []
    for v in test:
        c, t = sample_plurality_credit(_ref_samps(v.item_id) or [], rule=rule)
        sc.append(c)
        sc_t.append(t)
        c, t = cd_credit(
            v, pair, reference, thr_pair, rule=rule, ref_samples=_ref_samps(v.item_id)
        )
        cd_pair.append(c)
        cd_pair_t.append(t)
        c, t = cd_credit(
            v,
            configs,
            reference,
            thr_full,
            rule=rule,
            ref_samples=_ref_samps(v.item_id),
        )
        cd_full.append(c)
        cd_full_t.append(t)

    row: Dict[str, Any] = {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "rule": rule,
        "drop_policy": "global",
        "n_test": len(test),
        "n_two_seeds": len(two_seeds),
        "two_seeds": ",".join(str(s) for s in two_seeds),
        "has_twoseed_control": False,
        "acc_sc_n": _mean_credit(sc),
        "acc_cd_c0_c1": _mean_credit(cd_pair),
        "acc_cd_full3": _mean_credit(cd_full),
        "tie_rate_cd_c0_c1": _tie_rate(cd_pair_t),
        "delta_cd_c0_c1_minus_sc": _mean_credit(cd_pair) - _mean_credit(sc),
        "delta_cd_full3_minus_sc": _mean_credit(cd_full) - _mean_credit(sc),
    }

    joint = [v for v in test if v.item_id in seed_by_id]
    if len(joint) >= 4 and len(two_seeds) >= 2:
        seed_joint = [seed_by_id[v.item_id] for v in joint]
        ref_seed_idx = int(two_seeds[0])
        sc_j, cd_pair_j, cd_two, sc2 = [], [], [], []
        cd_pair_j_t, cd_two_t, sc2_t = [], [], []
        for v, sv in zip(joint, seed_joint):
            c, _ = sample_plurality_credit(_ref_samps(v.item_id) or [], rule=rule)
            sc_j.append(c)
            c, t = cd_credit(
                v,
                pair,
                reference,
                thr_pair,
                rule=rule,
                ref_samples=_ref_samps(v.item_id),
            )
            cd_pair_j.append(c)
            cd_pair_j_t.append(t)
            c, t = cd_credit(
                sv,
                fake_cfgs,
                ref_seed_name,
                thr_seed,
                rule=rule,
                ref_samples=seed_samples.get(ref_seed_idx, {}).get(sv.item_id),
            )
            cd_two.append(c)
            cd_two_t.append(t)
            c, t = sample_plurality_credit(
                seed_pool.get(sv.item_id, []), rule=rule
            )
            sc2.append(c)
            sc2_t.append(t)
        row.update(
            {
                "has_twoseed_control": True,
                "n_joint_twoseed_items": len(joint),
                "acc_sc_n_joint": _mean_credit(sc_j),
                "acc_cd_c0_c1_joint": _mean_credit(cd_pair_j),
                "acc_cd_two_seed": _mean_credit(cd_two),
                "acc_sc_2n_c0_seeds": _mean_credit(sc2),
                "tie_rate_cd_c0_c1_joint": _tie_rate(cd_pair_j_t),
                "tie_rate_cd_two_seed": _tie_rate(cd_two_t),
                "tie_rate_sc_2n": _tie_rate(sc2_t),
                "delta_cd_c0_c1_joint_minus_sc": (
                    _mean_credit(cd_pair_j) - _mean_credit(sc_j)
                ),
                "delta_cd_two_seed_minus_sc": (
                    _mean_credit(cd_two) - _mean_credit(sc_j)
                ),
                "delta_sc_2n_minus_sc": _mean_credit(sc2) - _mean_credit(sc_j),
                "delta_cd_c0_c1_minus_two_seed": (
                    _mean_credit(cd_pair_j) - _mean_credit(cd_two)
                ),
                "delta_cd_c0_c1_minus_sc_2n": (
                    _mean_credit(cd_pair_j) - _mean_credit(sc2)
                ),
            }
        )
    return row


def margin_split_cell_under_rule(
    corpus: Corpus,
    dataset: str,
    subset: Optional[str],
    model: str,
    rule: Rule,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
    split_seed: int = 0,
    sample_split_seed: int = 0,
) -> Dict[str, Any]:
    """Margin-split CD vs same half-B SC baseline under ``rule``."""
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS)
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
    if not shared or len(shared) < 8:
        return {
            "dataset": dataset,
            "subset": subset or "",
            "model": model,
            "rule": rule,
            "error": "too_few_config_items",
        }

    views: List[sel.ItemView] = []
    half_b_samples: Dict[str, List[Any]] = {}
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
                salt = sum((i + 1) * ord(ch) for i, ch in enumerate(item_id)) % 10_007
                half_a, half_b = c3c._split_samples(
                    cell.samples, split_seed=sample_split_seed + salt
                )
                margins[c] = ctrl._margin_from_samples(half_a)
                mode_b, ok_b = ctrl._modal_from_samples(half_b)
                modes[c] = mode_b
                correct[c] = ok_b
                n[c] = len(half_b)
                half_b_samples[item_id] = list(half_b)
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

    sc, sc_t = [], []
    cd, cd_t = [], []
    for v in test:
        hb = half_b_samples.get(v.item_id, [])
        c, t = sample_plurality_credit(hb, rule=rule)
        sc.append(c)
        sc_t.append(t)
        c, t = cd_credit(
            v, configs, reference, thr, rule=rule, ref_samples=hb
        )
        cd.append(c)
        cd_t.append(t)

    return {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "rule": rule,
        "n_test": len(test),
        "diversify_threshold_margin_split": thr,
        "acc_sc_half_b": _mean_credit(sc),
        "acc_cd_margin_split": _mean_credit(cd),
        "tie_rate_sc_half_b": _tie_rate(sc_t),
        "tie_rate_cd_margin_split": _tie_rate(cd_t),
        "delta_cd_margin_split_minus_sc": _mean_credit(cd) - _mean_credit(sc),
    }


# --------------------------------------------------------------------------- corpus aggregators


def _average_numeric_rows(
    per: Sequence[Dict[str, Any]],
    dataset: str,
    subset: Optional[str],
    model: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    keys = set().union(*(r.keys() for r in per))
    out: Dict[str, Any] = {
        "dataset": dataset,
        "subset": subset or "",
        "model": model,
        "n_splits": len(per),
    }
    if extra:
        out.update(extra)
    skip = {
        "dataset",
        "subset",
        "model",
        "error",
        "c0_seeds",
        "two_seeds",
        "rule",
        "drop_policy",
    }
    for k in keys:
        if k in out or k in skip:
            continue
        vals = []
        for r in per:
            v = r.get(k)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                if math.isfinite(float(v)):
                    vals.append(float(v))
        if vals:
            out[k] = sum(vals) / len(vals)
    # Preserve bool flags / strings from first row when useful.
    for k in ("has_seed_control", "has_twoseed_control", "rule", "c0_seeds", "two_seeds"):
        if k not in out:
            for r in per:
                if k in r:
                    out[k] = r[k]
                    break
    if "has_seed_control" not in out:
        out["has_seed_control"] = any(r.get("has_seed_control") for r in per)
    if "has_twoseed_control" not in out:
        out["has_twoseed_control"] = any(r.get("has_twoseed_control") for r in per)
    return out


def evaluate_corpus_under_rule(
    corpus: Corpus,
    rule: Rule,
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
            if len(present) < 2:
                continue
            c0_seeds = ctrl.available_c0_seeds(
                corpus, dataset, subset, model, reference=reference
            )
            use = [c for c in configs if c in present]
            ref = reference if reference in present else use[0]
            per: List[Dict[str, Any]] = []
            for ss in split_seeds:
                r = evaluate_cell_under_rule(
                    corpus,
                    dataset,
                    subset,
                    model,
                    rule=rule,
                    configs=use,
                    reference=ref,
                    seed=seed,
                    val_fraction=val_fraction,
                    split_seed=int(ss),
                    c0_seeds=c0_seeds,
                )
                if not r.get("error"):
                    per.append(r)
            if not per:
                continue
            rows.append(
                _average_numeric_rows(
                    per,
                    dataset,
                    subset,
                    model,
                    extra={
                        "rule": rule,
                        "n_c0_seeds": len(c0_seeds),
                        "c0_seeds": ",".join(str(s) for s in c0_seeds),
                    },
                )
            )
    return rows


def drop_c2_corpus_under_rule(
    corpus: Corpus,
    rule: Rule,
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
            if reference not in present or "c1" not in present:
                continue
            use = [c for c in configs if c in present]
            per: List[Dict[str, Any]] = []
            for ss in split_seeds:
                r = drop_c2_cell_under_rule(
                    corpus,
                    dataset,
                    subset,
                    model,
                    rule=rule,
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
            rows.append(
                _average_numeric_rows(
                    per, dataset, subset, model, extra={"rule": rule}
                )
            )
    return rows


def margin_split_corpus_under_rule(
    corpus: Corpus,
    rule: Rule,
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
                r = margin_split_cell_under_rule(
                    corpus,
                    dataset,
                    subset,
                    model,
                    rule=rule,
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
            rows.append(
                _average_numeric_rows(
                    per, dataset, subset, model, extra={"rule": rule}
                )
            )
    return rows


# --------------------------------------------------------------------------- contrasts + bootstrap


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


def _mean_key(rows: Sequence[Dict[str, Any]], key: str) -> float:
    vals = [
        float(r[key])
        for r in rows
        if math.isfinite(float(r.get(key, float("nan"))))
    ]
    return sum(vals) / len(vals) if vals else float("nan")


#: Named contrasts matching the review ask (1–13) plus token frontier (14).
CONTRAST_SPECS: List[Dict[str, Any]] = [
    {
        "contrast": "CD_minus_SC_N",
        "family": "main",
        "key": "delta_cd_configs_minus_sc",
        "tie_key": "tie_rate_cd",
        "scope": "all",
        "index": 1,
    },
    {
        "contrast": "SeedSub_minus_SC_N",
        "family": "main",
        "key": "delta_cd_seeds_minus_sc",
        "tie_key": "tie_rate_seed_cd",
        "scope": "seed",
        "index": 2,
    },
    {
        "contrast": "SC_3N_minus_SC_N",
        "family": "main",
        "key": "delta_sc_3n_seeds_minus_sc",
        "tie_key": "tie_rate_sc_3n",
        "scope": "seed",
        "index": 3,
    },
    {
        "contrast": "CD_minus_SeedSub",
        "family": "main",
        "key": "delta_cd_configs_minus_cd_seeds",
        "tie_key": None,
        "scope": "seed",
        "index": 4,
    },
    {
        "contrast": "CD_minus_SC_3N",
        "family": "main",
        "key": "delta_cd_configs_minus_sc_3n",
        "tie_key": None,
        "scope": "seed",
        "index": 5,
    },
    {
        "contrast": "SeedSub_minus_SC_3N",
        "family": "main",
        "key": "delta_cd_seeds_minus_sc_3n",
        "tie_key": None,
        "scope": "seed",
        "index": 6,
    },
    {
        "contrast": "Ungated_MoM_minus_SC_N",
        "family": "main",
        "key": "delta_mode_vote_minus_sc",
        "tie_key": "tie_rate_mode_of_modes",
        "scope": "all",
        "index": 7,
    },
    {
        "contrast": "Gated_minus_Ungated",
        "family": "main",
        "key": "delta_cd_gated_minus_mode_vote_full",
        "tie_key": None,
        "scope": "all",
        "index": 8,
    },
    {
        "contrast": "DropC2_CD_minus_SC_N",
        "family": "drop_c2",
        "key": "delta_cd_c0_c1_minus_sc",
        "tie_key": "tie_rate_cd_c0_c1",
        "scope": "all",
        "index": 9,
    },
    {
        "contrast": "DropC2_minus_TwoSeedCD",
        "family": "drop_c2",
        "key": "delta_cd_c0_c1_minus_two_seed",
        "tie_key": None,
        "scope": "twoseed",
        "index": 10,
    },
    {
        "contrast": "TwoSeedCD_minus_SC_2N",
        "family": "drop_c2",
        "key": "delta_cd_two_seed_minus_sc",  # will rewrite relative to SC@2N below
        "tie_key": "tie_rate_cd_two_seed",
        "scope": "twoseed",
        "index": 11,
        "derived": "two_seed_minus_sc2n",
    },
    {
        "contrast": "DropC2_minus_SC_2N",
        "family": "drop_c2",
        "key": "delta_cd_c0_c1_minus_sc_2n",
        "tie_key": None,
        "scope": "twoseed",
        "index": 12,
    },
    {
        "contrast": "MarginSplit_CD_minus_SC",
        "family": "margin",
        "key": "delta_cd_margin_split_minus_sc",
        "tie_key": "tie_rate_cd_margin_split",
        "scope": "all",
        "index": 13,
    },
]


def _scope_rows(
    rows: Sequence[Dict[str, Any]], scope: str
) -> List[Dict[str, Any]]:
    if scope == "seed":
        return [
            r
            for r in rows
            if r.get("has_seed_control")
            and math.isfinite(float(r.get("delta_cd_seeds_minus_sc", float("nan"))))
        ]
    if scope == "twoseed":
        return [
            r
            for r in rows
            if r.get("has_twoseed_control")
            and math.isfinite(
                float(r.get("delta_cd_c0_c1_minus_two_seed", float("nan")))
            )
        ]
    return [
        r
        for r in rows
        if any(
            math.isfinite(float(r.get(k, float("nan"))))
            for k in (
                "delta_cd_configs_minus_sc",
                "delta_cd_c0_c1_minus_sc",
                "delta_cd_margin_split_minus_sc",
            )
        )
    ]


def _ensure_derived_deltas(rows: List[Dict[str, Any]]) -> None:
    """Add TwoSeedCD − SC@2N if missing."""
    for r in rows:
        if "delta_two_seed_minus_sc_2n" in r:
            continue
        a = r.get("acc_cd_two_seed", float("nan"))
        b = r.get("acc_sc_2n_c0_seeds", float("nan"))
        if (
            isinstance(a, (int, float))
            and isinstance(b, (int, float))
            and math.isfinite(float(a))
            and math.isfinite(float(b))
        ):
            r["delta_two_seed_minus_sc_2n"] = float(a) - float(b)


def contrast_table_for_rule(
    main_rows: Sequence[Dict[str, Any]],
    drop_rows: Sequence[Dict[str, Any]],
    margin_rows: Sequence[Dict[str, Any]],
    rule: Rule,
    n_bootstrap: int = DEFAULT_CELL_BOOTSTRAP,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Long-format contrast rows for one credit rule."""
    drop_list = list(drop_rows)
    _ensure_derived_deltas(drop_list)
    families = {
        "main": list(main_rows),
        "drop_c2": drop_list,
        "margin": list(margin_rows),
    }
    out: List[Dict[str, Any]] = []
    for i, spec in enumerate(CONTRAST_SPECS):
        fam = families[spec["family"]]
        rs = _scope_rows(fam, spec["scope"])
        key = (
            "delta_two_seed_minus_sc_2n"
            if spec.get("derived") == "two_seed_minus_sc2n"
            else spec["key"]
        )
        mean_pp, lo, hi, n = _boot_mean_pp(
            rs, key, n_bootstrap=n_bootstrap, seed=seed + 10 * i
        )
        tie_rate = float("nan")
        if spec.get("tie_key"):
            tie_rate = _mean_key(rs, spec["tie_key"])
        out.append(
            {
                "contrast": spec["contrast"],
                "contrast_index": spec["index"],
                "rule": rule,
                "n_cells": n,
                "mean_pp": mean_pp,
                "ci_low": lo,
                "ci_high": hi,
                "mean_tie_rate": tie_rate,
                "delta_key": key,
                "cells": ";".join(ctrl.short_cell_label(r) for r in rs),
            }
        )
    return out


def token_frontier_points(
    main_rows: Sequence[Dict[str, Any]],
    expected_rows: Optional[Sequence[Dict[str, Any]]] = None,
    rule: Rule = "fractional",
    target_mult: float = 1.601,
) -> List[Dict[str, Any]]:
    """CD / seed-sub / SC@~1.6× accuracies under ``rule`` (14-cell means)."""
    seed_rows = [
        r
        for r in main_rows
        if r.get("has_seed_control")
        and math.isfinite(float(r.get("acc_cd_seeds", float("nan"))))
    ]
    all_rows = [
        r
        for r in main_rows
        if math.isfinite(
            float(r.get("acc_conditional_diversify_configs", float("nan")))
        )
    ]

    def mean_acc(rs: Sequence[Dict[str, Any]], key: str) -> float:
        return _mean_key(rs, key)

    acc_cd = mean_acc(all_rows, "acc_conditional_diversify_configs")
    acc_seed = mean_acc(seed_rows, "acc_cd_seeds")
    acc_sc = mean_acc(seed_rows, "acc_sc_n_joint")
    if not math.isfinite(acc_sc):
        acc_sc = mean_acc(all_rows, "acc_sc_n")
    acc_sc3 = mean_acc(seed_rows, "acc_sc_3n_c0_seeds")

    # Token multiple from expected-token table when available.
    probe_mult = target_mult
    sc3n_mult = 3.0
    if expected_rows:
        ratios = [
            float(r["ratio_probe_to_sc_n"])
            for r in expected_rows
            if math.isfinite(float(r.get("ratio_probe_to_sc_n", float("nan"))))
        ]
        if ratios:
            probe_mult = sum(ratios) / len(ratios)
        sc3_ratios = []
        for r in expected_rows:
            t0 = float(r.get("mean_tok_sc_n", float("nan")))
            t3 = float(r.get("mean_tok_sc_3n_seeds", float("nan")))
            if math.isfinite(t0) and t0 > 0 and math.isfinite(t3):
                sc3_ratios.append(t3 / t0)
        if sc3_ratios:
            sc3n_mult = sum(sc3_ratios) / len(sc3_ratios)

    # Linear interp SC@N → SC@3N to probe multiple.
    if math.isfinite(acc_sc) and math.isfinite(acc_sc3) and sc3n_mult > 1.0:
        t = (probe_mult - 1.0) / (sc3n_mult - 1.0)
        t = max(0.0, min(1.0, t))
        acc_sc_matched = acc_sc + t * (acc_sc3 - acc_sc)
    else:
        acc_sc_matched = float("nan")

    return [
        {
            "contrast": "token_frontier_CD",
            "contrast_index": 14,
            "rule": rule,
            "n_cells": len(all_rows),
            "mean_pp": 100.0 * acc_cd if math.isfinite(acc_cd) else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "mean_tie_rate": _mean_key(all_rows, "tie_rate_cd"),
            "accuracy": acc_cd,
            "token_multiple": probe_mult,
            "note": "14-cell mean accuracy (not a delta)",
        },
        {
            "contrast": "token_frontier_SeedSub",
            "contrast_index": 14,
            "rule": rule,
            "n_cells": len(seed_rows),
            "mean_pp": 100.0 * acc_seed if math.isfinite(acc_seed) else float("nan"),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "mean_tie_rate": _mean_key(seed_rows, "tie_rate_seed_cd"),
            "accuracy": acc_seed,
            "token_multiple": probe_mult,
            "note": "14-cell mean accuracy (not a delta)",
        },
        {
            "contrast": "token_frontier_SC_1p6x",
            "contrast_index": 14,
            "rule": rule,
            "n_cells": len(seed_rows),
            "mean_pp": (
                100.0 * acc_sc_matched if math.isfinite(acc_sc_matched) else float("nan")
            ),
            "ci_low": float("nan"),
            "ci_high": float("nan"),
            "mean_tie_rate": float("nan"),
            "accuracy": acc_sc_matched,
            "token_multiple": probe_mult,
            "note": (
                f"interp SC@N→SC@3N (sc3n_mult={sc3n_mult:.3f}) "
                f"to probe_mult={probe_mult:.3f}"
            ),
        },
    ]


def side_by_side(
    fixed_rows: Sequence[Dict[str, Any]],
    frac_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Join fixed vs fractional contrast tables on contrast name."""
    by_f = {r["contrast"]: r for r in fixed_rows}
    by_r = {r["contrast"]: r for r in frac_rows}
    keys = list(dict.fromkeys([*by_f.keys(), *by_r.keys()]))
    out: List[Dict[str, Any]] = []
    for k in keys:
        f = by_f.get(k, {})
        r = by_r.get(k, {})
        mf = float(f.get("mean_pp", float("nan")))
        mr = float(r.get("mean_pp", float("nan")))
        out.append(
            {
                "contrast": k,
                "contrast_index": f.get("contrast_index", r.get("contrast_index")),
                "n_cells": f.get("n_cells", r.get("n_cells")),
                "fixed_mean_pp": mf,
                "fixed_ci_low": f.get("ci_low", float("nan")),
                "fixed_ci_high": f.get("ci_high", float("nan")),
                "fixed_mean_tie_rate": f.get("mean_tie_rate", float("nan")),
                "frac_mean_pp": mr,
                "frac_ci_low": r.get("ci_low", float("nan")),
                "frac_ci_high": r.get("ci_high", float("nan")),
                "frac_mean_tie_rate": r.get("mean_tie_rate", float("nan")),
                "delta_frac_minus_fixed_pp": (
                    mr - mf
                    if math.isfinite(mf) and math.isfinite(mr)
                    else float("nan")
                ),
                "fixed_ci_excludes_0": (
                    math.isfinite(float(f.get("ci_low", float("nan"))))
                    and math.isfinite(float(f.get("ci_high", float("nan"))))
                    and (
                        float(f["ci_low"]) > 0
                        or float(f["ci_high"]) < 0
                    )
                ),
                "frac_ci_excludes_0": (
                    math.isfinite(float(r.get("ci_low", float("nan"))))
                    and math.isfinite(float(r.get("ci_high", float("nan"))))
                    and (
                        float(r["ci_low"]) > 0
                        or float(r["ci_high"]) < 0
                    )
                ),
            }
        )
    return out


# --------------------------------------------------------------------------- optional item-joint (fractional)


def _eval_views_credit(
    cfg_views: Sequence[sel.ItemView],
    seed_by_id: Dict[str, sel.ItemView],
    seed_pool: Dict[str, List[Any]],
    per_cfg_ref: Dict[str, List[Any]],
    seed_ref_samples: Dict[str, List[Any]],
    configs: Sequence[str],
    reference: str,
    c0_seeds: Sequence[int],
    rule: Rule,
    val_fraction: float,
    split_seed: int,
) -> Dict[str, float]:
    if len(cfg_views) < 8:
        return {}
    train, test = sel._split_views(
        cfg_views, val_fraction=val_fraction, split_seed=split_seed
    )
    thr = sel._fit_diversify_threshold(train, configs, reference)
    fake = [f"seed{s}" for s in c0_seeds]
    ref_seed = f"seed{c0_seeds[0]}" if c0_seeds else "seed0"
    seed_train = [seed_by_id[v.item_id] for v in train if v.item_id in seed_by_id]
    thr_seed = (
        sel._fit_diversify_threshold(seed_train, fake, ref_seed)
        if len(seed_train) >= 4 and len(c0_seeds) >= 2
        else thr
    )
    joint = [v for v in test if v.item_id in seed_by_id]
    if len(c0_seeds) < 2 or len(joint) < 4:
        return {}
    seed_joint = [seed_by_id[v.item_id] for v in joint]
    cd, seed_cd, sc3, sc = [], [], [], []
    for v, sv in zip(joint, seed_joint):
        ref_s = per_cfg_ref.get(v.item_id)
        c, _ = cd_credit(
            v, configs, reference, thr, rule=rule, ref_samples=ref_s
        )
        cd.append(c)
        c, _ = cd_credit(
            sv,
            fake,
            ref_seed,
            thr_seed,
            rule=rule,
            ref_samples=seed_ref_samples.get(sv.item_id),
        )
        seed_cd.append(c)
        c, _ = sample_plurality_credit(seed_pool.get(sv.item_id, []), rule=rule)
        sc3.append(c)
        c, _ = sample_plurality_credit(ref_s or [], rule=rule)
        sc.append(c)
    acc_cd = _mean_credit(cd)
    acc_seed = _mean_credit(seed_cd)
    acc_sc3 = _mean_credit(sc3)
    acc_sc = _mean_credit(sc)
    return {
        "delta_cd_minus_sc": acc_cd - acc_sc,
        "delta_cd_minus_seed": acc_cd - acc_seed,
        "delta_cd_minus_sc3n": acc_cd - acc_sc3,
        "delta_seed_minus_sc3n": acc_seed - acc_sc3,
    }


def item_joint_bootstrap_fractional(
    corpus: Corpus,
    rule: Rule = "fractional",
    n_bootstrap: int = DEFAULT_ITEM_BOOTSTRAP,
    seed: int = 0,
    split_seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Item-joint bootstrap for critical C3 contrasts under ``rule``."""
    import random

    from . import measurement_audits as audits

    specs = audits._cell_specs(corpus)
    if not specs:
        return [], {"error": "no_cells"}

    # Attach reference sample maps (config seed0 + first c0 seed).
    for spec in specs:
        _, _, per_cfg = ctrl._config_item_views_and_pool(
            corpus,
            spec["dataset"],
            spec["subset"],
            spec["model"],
            spec["configs"],
            seed=0,
        )
        ref = spec["reference"]
        spec["per_cfg_ref"] = per_cfg.get(ref, {})
        seed_samps = _c0_seed_sample_map(
            corpus,
            spec["dataset"],
            spec["subset"],
            spec["model"],
            ref,
            spec["c0_seeds"],
        )
        first = int(spec["c0_seeds"][0]) if spec["c0_seeds"] else 0
        spec["seed_ref_samples"] = seed_samps.get(first, {})

    cell_unit_to_items: List[Dict[str, List[str]]] = []
    bench_unit_sets: Dict[Tuple[str, Optional[str]], List[set]] = defaultdict(list)
    for spec in specs:
        key = (spec["dataset"], spec["subset"])
        u2i: Dict[str, List[str]] = defaultdict(list)
        for v in spec["cfg_views"]:
            u2i[v.unit].append(v.item_id)
        cell_unit_to_items.append(dict(u2i))
        bench_unit_sets[key].append(set(u2i.keys()))

    bench_units: Dict[Tuple[str, Optional[str]], List[str]] = {}
    for key, sets in bench_unit_sets.items():
        inter = set.intersection(*sets) if sets else set()
        pool = inter if len(inter) >= 8 else set.union(*sets) if sets else set()
        bench_units[key] = sorted(pool)

    def _point_from_specs(active_specs: Sequence[Dict[str, Any]]) -> Dict[str, float]:
        cell_avgs: List[Dict[str, float]] = []
        for spec in active_specs:
            per = [
                _eval_views_credit(
                    spec["cfg_views"],
                    spec["seed_by_id"],
                    spec["seed_pool"],
                    spec["per_cfg_ref"],
                    spec["seed_ref_samples"],
                    spec["configs"],
                    spec["reference"],
                    spec["c0_seeds"],
                    rule,
                    val_fraction,
                    int(ss),
                )
                for ss in split_seeds
            ]
            per = [p for p in per if p]
            if not per:
                continue
            keys = set().union(*(p.keys() for p in per))
            avg = {k: sum(p[k] for p in per) / len(per) for k in keys}
            cell_avgs.append(avg)
        if not cell_avgs:
            return {}
        keys = set().union(*(c.keys() for c in cell_avgs))
        return {k: sum(c[k] for c in cell_avgs) / len(cell_avgs) for k in keys}

    point = _point_from_specs(specs)
    rng = random.Random(seed)
    repls: Dict[str, List[float]] = defaultdict(list)

    for _b in range(n_bootstrap):
        # Resample units per benchmark; rebuild each cell's view list.
        boot_specs: List[Dict[str, Any]] = []
        drawn: Dict[Tuple[str, Optional[str]], List[str]] = {}
        for key, units in bench_units.items():
            if not units:
                drawn[key] = []
                continue
            drawn[key] = [units[rng.randrange(len(units))] for _ in units]
        for i, spec in enumerate(specs):
            key = (spec["dataset"], spec["subset"])
            u2i = cell_unit_to_items[i]
            new_views: List[sel.ItemView] = []
            cfg_by = spec["cfg_by_id"]
            for u in drawn[key]:
                for item_id in u2i.get(u, []):
                    if item_id in cfg_by:
                        new_views.append(cfg_by[item_id])
            if len(new_views) < 8:
                continue
            boot_specs.append({**spec, "cfg_views": new_views})
        boot_point = _point_from_specs(boot_specs)
        for k, v in boot_point.items():
            if math.isfinite(v):
                repls[k].append(100.0 * v)

    contrast_map = {
        "delta_cd_minus_sc": "CD_minus_SC_N",
        "delta_cd_minus_seed": "CD_minus_SeedSub",
        "delta_cd_minus_sc3n": "CD_minus_SC_3N",
        "delta_seed_minus_sc3n": "SeedSub_minus_SC_3N",
    }
    rows: List[Dict[str, Any]] = []
    for key, name in contrast_map.items():
        vals = repls.get(key, [])
        p = 100.0 * float(point.get(key, float("nan")))
        if vals:
            import numpy as np

            lo = float(np.quantile(vals, 0.025))
            hi = float(np.quantile(vals, 0.975))
        else:
            lo = hi = float("nan")
        rows.append(
            {
                "contrast": name,
                "rule": rule,
                "ci_kind": "item_joint",
                "n_cells": len(specs),
                "mean_pp": p,
                "ci_low": lo,
                "ci_high": hi,
                "n_bootstrap": n_bootstrap,
            }
        )
    meta = {
        "n_bootstrap": n_bootstrap,
        "n_cells": len(specs),
        "rule": rule,
        "point_pp": {contrast_map[k]: 100.0 * v for k, v in point.items() if k in contrast_map},
    }
    return rows, meta


# --------------------------------------------------------------------------- orchestration


def run_fractional_credit_analysis(
    corpus: Corpus,
    n_bootstrap: int = DEFAULT_CELL_BOOTSTRAP,
    boot_seed: int = 0,
    item_bootstrap: int = DEFAULT_ITEM_BOOTSTRAP,
    skip_item_joint: bool = False,
) -> Dict[str, Any]:
    """Full fixed vs fractional recomputation for review artefacts."""
    log.info("evaluating main controls under fixed …")
    main_fixed = evaluate_corpus_under_rule(corpus, rule="fixed")
    log.info("evaluating main controls under fractional …")
    main_frac = evaluate_corpus_under_rule(corpus, rule="fractional")

    log.info("drop-c2 / two-seed under fixed …")
    drop_fixed = drop_c2_corpus_under_rule(corpus, rule="fixed")
    log.info("drop-c2 / two-seed under fractional …")
    drop_frac = drop_c2_corpus_under_rule(corpus, rule="fractional")

    log.info("margin-split under fixed …")
    margin_fixed = margin_split_corpus_under_rule(corpus, rule="fixed")
    log.info("margin-split under fractional …")
    margin_frac = margin_split_corpus_under_rule(corpus, rule="fractional")

    expected = c3c.expected_tokens_corpus(corpus)

    fixed_contrasts = contrast_table_for_rule(
        main_fixed, drop_fixed, margin_fixed, rule="fixed",
        n_bootstrap=n_bootstrap, seed=boot_seed,
    )
    frac_contrasts = contrast_table_for_rule(
        main_frac, drop_frac, margin_frac, rule="fractional",
        n_bootstrap=n_bootstrap, seed=boot_seed,
    )
    fixed_tok = token_frontier_points(main_fixed, expected, rule="fixed")
    frac_tok = token_frontier_points(main_frac, expected, rule="fractional")

    long_rows = fixed_contrasts + frac_contrasts + fixed_tok + frac_tok
    side = side_by_side(fixed_contrasts + fixed_tok, frac_contrasts + frac_tok)

    item_rows: List[Dict[str, Any]] = []
    item_meta: Dict[str, Any] = {}
    if not skip_item_joint and item_bootstrap > 0:
        log.info("item-joint bootstrap (fractional) B=%s …", item_bootstrap)
        item_rows, item_meta = item_joint_bootstrap_fractional(
            corpus, rule="fractional", n_bootstrap=item_bootstrap, seed=boot_seed
        )
        # Also fixed item-joint for side-by-side on critical contrasts.
        item_fixed, _ = item_joint_bootstrap_fractional(
            corpus, rule="fixed", n_bootstrap=item_bootstrap, seed=boot_seed
        )
        item_rows = item_fixed + item_rows

    # Policy-level mean tie rates (fractional table).
    tie_summary = []
    for method, key in (
        ("sc_n", "tie_rate_sc_n"),
        ("conditional_diversify", "tie_rate_cd"),
        ("mode_of_modes", "tie_rate_mode_of_modes"),
        ("seed_substitution_cd", "tie_rate_seed_cd"),
        ("sc_3n", "tie_rate_sc_3n"),
    ):
        tie_summary.append(
            {
                "policy": method,
                "mean_tie_rate_fixed": _mean_key(main_fixed, key),
                "mean_tie_rate_frac": _mean_key(main_frac, key),
            }
        )

    return {
        "long_rows": long_rows,
        "side_by_side": side,
        "main_fixed": main_fixed,
        "main_frac": main_frac,
        "item_joint": item_rows,
        "item_meta": item_meta,
        "tie_summary": tie_summary,
        "expected": expected,
    }


def gating_verdict(side_rows: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """KEEP vs DROP the gating/aggregation claim under fractional credit."""
    by = {r["contrast"]: r for r in side_rows}

    def pack(name: str) -> Dict[str, Any]:
        r = by.get(name, {})
        return {
            "contrast": name,
            "fixed_mean_pp": float(r.get("fixed_mean_pp", float("nan"))),
            "fixed_ci": (
                float(r.get("fixed_ci_low", float("nan"))),
                float(r.get("fixed_ci_high", float("nan"))),
            ),
            "frac_mean_pp": float(r.get("frac_mean_pp", float("nan"))),
            "frac_ci": (
                float(r.get("frac_ci_low", float("nan"))),
                float(r.get("frac_ci_high", float("nan"))),
            ),
            "frac_excludes_0": bool(r.get("frac_ci_excludes_0")),
            "delta_pp": float(r.get("delta_frac_minus_fixed_pp", float("nan"))),
        }

    critical = {
        "seed_sub_minus_sc3n": pack("SeedSub_minus_SC_3N"),
        "cd_minus_sc_n": pack("CD_minus_SC_N"),
        "cd_minus_sc3n": pack("CD_minus_SC_3N"),
        "gated_minus_ungated": pack("Gated_minus_Ungated"),
    }
    ss = critical["seed_sub_minus_sc3n"]
    cd = critical["cd_minus_sc_n"]
    cd3 = critical["cd_minus_sc3n"]

    # DROP if the residual that underwrote gating collapses under fractional:
    # seed-sub−SC@3N CI includes 0 (or point ≤ 0), and/or CD−SC@3N likewise,
    # while the drop vs fixed is material (>0.5pp shrinkage toward 0).
    seed_ok = ss["frac_excludes_0"] and ss["frac_mean_pp"] > 0
    cd3_ok = cd3["frac_excludes_0"] and cd3["frac_mean_pp"] > 0
    material_shrink = (
        (math.isfinite(ss["delta_pp"]) and ss["delta_pp"] < -0.5)
        or (math.isfinite(cd3["delta_pp"]) and cd3["delta_pp"] < -0.5)
    )
    if seed_ok and cd3_ok:
        decision = "KEEP"
        reason = (
            "Under fractional credit, SeedSub-SC@3N and CD-SC@3N still have "
            "cell-bootstrap CIs excluding 0; gating/aggregation residual is "
            "not solely an any-winner tie artifact."
        )
    elif (not seed_ok) and material_shrink:
        decision = "DROP"
        reason = (
            "Under fractional credit, the SeedSub-SC@3N residual that underwrote "
            "the gating/aggregation claim no longer clears 0 (and/or shrinks "
            ">0.5pp vs fixed any-winner). Treat the claim as a tie artifact."
        )
    elif not seed_ok or not cd3_ok:
        decision = "DROP"
        reason = (
            "Critical gating residual(s) lose CI-above-zero support under "
            "fractional credit; do not retain a strong gating/aggregation claim."
        )
    else:
        decision = "KEEP"
        reason = "Fractional credit does not overturn the residual pattern."

    return {
        "decision": decision,
        "reason": reason,
        "critical": critical,
        "cd_minus_sc_n_frac_pp": cd["frac_mean_pp"],
        "seed_sub_minus_sc3n_frac_pp": ss["frac_mean_pp"],
    }
