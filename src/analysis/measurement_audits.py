"""CPU-only measurement audits requested for review hardening.

1. Modal-tie rates + fixed vs fractional credit for SC@N / SC@3N / CD / seed-CD / MoM
2. Item-joint bootstrap CIs for key C3 contrasts (vs cell-bootstrap in tab_c3_*)
3. Extraction / truncation rates; P4 survivors conditional on no ⊥ mode
4. Seed-to-seed reorder baseline vs cross-config rates

All analyses read graded ``results_fp16`` cells — no GPU.
"""

from __future__ import annotations

import logging
import math
import random
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ..answers import EXTRACTION_FAILURE_CLASS
from ..elicitation import PRIMARY_CONFIG_IDS, REFERENCE_CONFIG_ID
from . import c3_controls as ctrl
from . import modes as modes_mod
from . import selection as sel
from .corpus import Cell, Corpus

log = logging.getLogger(__name__)

DEFAULT_ITEM_BOOTSTRAP = 1000
DEFAULT_SPLIT_SEEDS: Tuple[int, ...] = (0, 1, 2, 3, 4)


# --------------------------------------------------------------------------- ties
def _top_tied_classes(counts: Dict[str, int]) -> List[str]:
    if not counts:
        return []
    top = max(counts.values())
    return [c for c, n in counts.items() if n == top]


def _class_is_correct(samples: Sequence[Any], cls: str) -> bool:
    if cls is None or cls == EXTRACTION_FAILURE_CLASS:
        return False
    return any(s.is_correct for s in samples if s.canonical_class == cls)


def _fixed_mode_from_counts(
    counts: Dict[str, int], samples: Sequence[Any]
) -> Tuple[Optional[str], bool]:
    """First-occurrence tie-break matching ``Cell.modal_class`` / ``modal_correct``."""
    if not counts or not samples:
        return None, False
    top = max(counts.values())
    mode = None
    for s in samples:
        if counts.get(s.canonical_class) == top:
            mode = s.canonical_class
            break
    if mode is None or mode == EXTRACTION_FAILURE_CLASS:
        return mode, False
    return mode, _class_is_correct(samples, mode)


def _sample_plurality_tie_credit(
    samples: Sequence[Any],
) -> Tuple[bool, bool, float]:
    """Return (is_tie, fixed_correct, fractional_credit)."""
    if not samples:
        return False, False, 0.0
    counts: Dict[str, int] = {}
    for s in samples:
        counts[s.canonical_class] = counts.get(s.canonical_class, 0) + 1
    tied = _top_tied_classes(counts)
    is_tie = len(tied) > 1
    _, fixed_ok = _fixed_mode_from_counts(counts, samples)
    n_ok = sum(1 for c in tied if _class_is_correct(samples, c))
    frac = n_ok / len(tied) if tied else 0.0
    return is_tie, bool(fixed_ok), float(frac)


def _mode_vote_tie_credit(
    modes: Dict[str, Optional[str]],
    correct: Dict[str, bool],
    columns: Sequence[str],
) -> Tuple[bool, bool, float]:
    """Plurality over column modes (configs or fake seed configs).

    Fixed credit matches ``selection._mode_plurality_correct`` (True if any
    tied-winning mode has a correct column). Fractional credit is expected
    accuracy under uniform choice among the ``k`` tied top modes.
    """
    votes: Dict[str, int] = {}
    col_for_mode: Dict[str, List[str]] = defaultdict(list)
    for c in columns:
        mode = modes.get(c)
        if mode is None:
            continue
        votes[mode] = votes.get(mode, 0) + 1
        col_for_mode[mode].append(c)
    if not votes:
        return False, False, 0.0
    top = max(votes.values())
    tied = [m for m, n in votes.items() if n == top]
    is_tie = len(tied) > 1
    winners = set(tied)
    fixed_ok = any(
        modes.get(c) in winners and correct.get(c, False) for c in columns
    )
    mode_ok = {
        m: any(correct.get(c, False) for c in col_for_mode[m]) for m in tied
    }
    frac = sum(1 for m in tied if mode_ok[m]) / len(tied)
    return is_tie, bool(fixed_ok), float(frac)


def _cd_effective_tie_credit(
    view: sel.ItemView,
    columns: Sequence[str],
    reference: str,
    threshold: float,
) -> Tuple[bool, bool, float]:
    """CD aggregator: reference mode if margin ≥ τ, else mode plurality."""
    avail = view.available(columns)
    if not avail:
        return False, False, 0.0
    ref_margin = view.margins.get(reference, 0.0)
    if (
        threshold <= 1.0
        and math.isfinite(ref_margin)
        and ref_margin >= threshold
        and reference in avail
    ):
        # Single-config mode: tie state is not a multi-mode plurality tie.
        # Report sample-level tie for the reference column only if we had samples;
        # here we only have plug-in modes — treat as non-tie at the policy layer.
        ok = bool(view.correct.get(reference, False))
        return False, ok, float(ok)
    return _mode_vote_tie_credit(view.modes, view.correct, columns)


def _summarise_tie_rows(
    rows: Sequence[Tuple[bool, bool, float]],
) -> Dict[str, float]:
    if not rows:
        return {
            "n_items": 0,
            "frac_modal_tie": float("nan"),
            "acc_fixed_tiebreak": float("nan"),
            "acc_fractional": float("nan"),
            "acc_fixed_on_ties": float("nan"),
            "acc_fractional_on_ties": float("nan"),
            "n_ties": 0,
        }
    n = len(rows)
    n_ties = sum(1 for t, _, _ in rows if t)
    return {
        "n_items": n,
        "frac_modal_tie": n_ties / n,
        "acc_fixed_tiebreak": sum(1 for _, f, _ in rows if f) / n,
        "acc_fractional": sum(fr for _, _, fr in rows) / n,
        "acc_fixed_on_ties": (
            sum(1 for t, f, _ in rows if t and f) / n_ties if n_ties else float("nan")
        ),
        "acc_fractional_on_ties": (
            sum(fr for t, _, fr in rows if t) / n_ties if n_ties else float("nan")
        ),
        "n_ties": n_ties,
    }


def tie_audit_table(
    corpus: Corpus,
    configs: Optional[Sequence[str]] = None,
    reference: str = REFERENCE_CONFIG_ID,
    seed: int = 0,
    split_seed: int = 0,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
) -> List[Dict[str, Any]]:
    """Per-cell / per-method modal-tie audit."""
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS)
    out: List[Dict[str, Any]] = []

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
            use_cfgs = [c for c in configs if c in present]
            ref = reference if reference in present else use_cfgs[0]
            c0_seeds = ctrl.available_c0_seeds(
                corpus, dataset, subset, model, reference=ref
            )

            # --- SC@N (c0 seed0): item-level from cells ---
            c0_cells = {
                c.item_id: c
                for c in corpus.cells.values()
                if c.dataset == dataset
                and c.subset == subset
                and c.model == model
                and c.config == ref
                and c.seed == seed
                and c.n > 0
            }
            sc_rows = [
                _sample_plurality_tie_credit(c.samples) for c in c0_cells.values()
            ]
            out.append(
                {
                    "method": "sc_n_c0",
                    "dataset": dataset,
                    "subset": subset or "",
                    "model": model,
                    **_summarise_tie_rows(sc_rows),
                }
            )

            # --- SC@3N pooled c0 seeds ---
            seed_views, seed_pool = ctrl._seed_item_views(
                corpus, dataset, subset, model, reference=ref, seeds=c0_seeds
            )
            sc3_rows = [
                _sample_plurality_tie_credit(seed_pool[v.item_id])
                for v in seed_views
                if v.item_id in seed_pool
            ]
            out.append(
                {
                    "method": "sc_3n_c0_seeds",
                    "dataset": dataset,
                    "subset": subset or "",
                    "model": model,
                    "n_c0_seeds": len(c0_seeds),
                    **_summarise_tie_rows(sc3_rows),
                }
            )

            # --- config ItemViews + CD / mode-of-modes ---
            cfg_views, _, _ = ctrl._config_item_views_and_pool(
                corpus, dataset, subset, model, use_cfgs, seed=seed
            )
            if len(cfg_views) < 8:
                continue
            train, _test = sel._split_views(
                cfg_views, val_fraction=val_fraction, split_seed=split_seed
            )
            thr = sel._fit_diversify_threshold(train, use_cfgs, ref)

            mom_rows = [
                _mode_vote_tie_credit(v.modes, v.correct, use_cfgs) for v in cfg_views
            ]
            out.append(
                {
                    "method": "mode_of_modes",
                    "dataset": dataset,
                    "subset": subset or "",
                    "model": model,
                    "diversify_threshold": thr,
                    **_summarise_tie_rows(mom_rows),
                }
            )

            cd_rows = [
                _cd_effective_tie_credit(v, use_cfgs, ref, thr) for v in cfg_views
            ]
            out.append(
                {
                    "method": "conditional_diversify",
                    "dataset": dataset,
                    "subset": subset or "",
                    "model": model,
                    "diversify_threshold": thr,
                    **_summarise_tie_rows(cd_rows),
                }
            )

            # --- seed-substitution CD ---
            if len(c0_seeds) >= 2 and seed_views:
                fake = [f"seed{s}" for s in c0_seeds]
                ref_seed = f"seed{c0_seeds[0]}"
                seed_by_id = {v.item_id: v for v in seed_views}
                seed_train = [
                    seed_by_id[v.item_id] for v in train if v.item_id in seed_by_id
                ]
                thr_seed = (
                    sel._fit_diversify_threshold(seed_train, fake, ref_seed)
                    if len(seed_train) >= 4
                    else thr
                )
                joint = [seed_by_id[v.item_id] for v in cfg_views if v.item_id in seed_by_id]
                seed_cd_rows = [
                    _cd_effective_tie_credit(v, fake, ref_seed, thr_seed) for v in joint
                ]
                out.append(
                    {
                        "method": "seed_substitution_cd",
                        "dataset": dataset,
                        "subset": subset or "",
                        "model": model,
                        "n_c0_seeds": len(c0_seeds),
                        "diversify_threshold": thr_seed,
                        **_summarise_tie_rows(seed_cd_rows),
                    }
                )
    return out


def tie_audit_method_summary(rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Unweighted mean across cells per method (headline)."""
    by: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by[str(r["method"])].append(r)
    out: List[Dict[str, Any]] = []
    for method, rs in sorted(by.items()):
        def mean(key: str) -> float:
            vals = [
                float(r[key])
                for r in rs
                if math.isfinite(float(r.get(key, float("nan"))))
            ]
            return sum(vals) / len(vals) if vals else float("nan")

        out.append(
            {
                "method": method,
                "n_cells": len(rs),
                "mean_frac_modal_tie": mean("frac_modal_tie"),
                "mean_acc_fixed_tiebreak": mean("acc_fixed_tiebreak"),
                "mean_acc_fractional": mean("acc_fractional"),
                "mean_delta_frac_minus_fixed_pp": 100.0
                * (mean("acc_fractional") - mean("acc_fixed_tiebreak")),
            }
        )
    return out


# ------------------------------------------------------- item-joint bootstrap
def _percentile_ci(
    values: Sequence[float], alpha: float = 0.05
) -> Tuple[float, float, float]:
    import numpy as np

    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(arr.mean())
    if arr.size == 1:
        return point, point, point
    lo = float(np.quantile(arr, alpha / 2.0))
    hi = float(np.quantile(arr, 1.0 - alpha / 2.0))
    return lo, hi, point


def _acc_flags(flags: Sequence[bool]) -> float:
    return sum(1 for f in flags if f) / len(flags) if flags else float("nan")


def _cell_specs(corpus: Corpus) -> List[Dict[str, Any]]:
    """Build the 14 (dataset, subset, model) control cells with precomputed views."""
    configs = list(PRIMARY_CONFIG_IDS)
    specs: List[Dict[str, Any]] = []
    for dataset, subset in corpus.benchmarks():
        for model in corpus.models():
            present = {
                c.config
                for c in corpus.cells.values()
                if c.dataset == dataset
                and c.subset == subset
                and c.model == model
                and c.seed == 0
                and c.config in configs
                and c.n > 0
            }
            if len(present) < 2:
                continue
            use = [c for c in configs if c in present]
            ref = REFERENCE_CONFIG_ID if REFERENCE_CONFIG_ID in present else use[0]
            c0_seeds = ctrl.available_c0_seeds(
                corpus, dataset, subset, model, reference=ref
            )
            cfg_views, cfg_pool, per_cfg = ctrl._config_item_views_and_pool(
                corpus, dataset, subset, model, use, seed=0
            )
            seed_views, seed_pool = ctrl._seed_item_views(
                corpus, dataset, subset, model, reference=ref, seeds=c0_seeds
            )
            if len(cfg_views) < 8:
                continue
            specs.append(
                {
                    "dataset": dataset,
                    "subset": subset,
                    "model": model,
                    "configs": use,
                    "reference": ref,
                    "c0_seeds": c0_seeds,
                    "cfg_views": cfg_views,
                    "cfg_by_id": {v.item_id: v for v in cfg_views},
                    "seed_by_id": {v.item_id: v for v in seed_views},
                    "seed_pool": seed_pool,
                    "cfg_pool": cfg_pool,
                    "has_seed": len(c0_seeds) >= 2 and len(seed_views) >= 8,
                }
            )
    return specs


def _eval_on_views(
    cfg_views: Sequence[sel.ItemView],
    seed_by_id: Dict[str, sel.ItemView],
    seed_pool: Dict[str, List[Any]],
    configs: Sequence[str],
    reference: str,
    c0_seeds: Sequence[int],
    val_fraction: float,
    split_seed: int,
) -> Dict[str, float]:
    """Recompute CD / seed-CD / SC3N on a (possibly resampled) view list."""
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
    cd = [sel._policy_conditional_diversify(v, configs, reference, thr) for v in test]
    out: Dict[str, float] = {
        "acc_cd": _acc_flags(cd),
        "n_test": float(len(test)),
    }
    joint = [v for v in test if v.item_id in seed_by_id]
    if len(c0_seeds) >= 2 and len(joint) >= 4:
        seed_joint = [seed_by_id[v.item_id] for v in joint]
        cd_seed = [
            sel._policy_conditional_diversify(v, fake, ref_seed, thr_seed)
            for v in seed_joint
        ]
        sc3 = []
        for v in seed_joint:
            _, ok = ctrl._modal_from_samples(seed_pool.get(v.item_id, []))
            sc3.append(ok)
        acc_cd_j = _acc_flags(
            [
                sel._policy_conditional_diversify(v, configs, reference, thr)
                for v in joint
            ]
        )
        acc_seed = _acc_flags(cd_seed)
        acc_sc3 = _acc_flags(sc3)
        out.update(
            {
                "acc_cd_joint": acc_cd_j,
                "acc_cd_seeds": acc_seed,
                "acc_sc_3n": acc_sc3,
                "delta_cd_minus_seed": acc_cd_j - acc_seed,
                "delta_cd_minus_sc3n": acc_cd_j - acc_sc3,
                "delta_seed_minus_sc3n": acc_seed - acc_sc3,
            }
        )
    return out


def item_joint_bootstrap_c3(
    corpus: Corpus,
    n_bootstrap: int = DEFAULT_ITEM_BOOTSTRAP,
    seed: int = 0,
    split_seeds: Sequence[int] = DEFAULT_SPLIT_SEEDS,
    val_fraction: float = sel.DEFAULT_VAL_FRACTION,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Item-joint bootstrap over C3 contrasts.

    Approximation (documented): for each replicate,
    1. Within each ``(dataset, subset)``, resample *bootstrap units* (template_id
       on GSM-Symbolic, else item_id) with replacement from the union of units
       appearing in any model cell on that benchmark.
    2. Rebuild each cell's ItemView list by mapping resampled units → items
       (duplicating views when a unit is drawn multiple times).
    3. Re-fit τ on the train fold and recompute CD / seed-CD / SC@3N (same as
       ``evaluate_controls_for_cell``), average numeric deltas over ``split_seeds``,
       then mean across the 14 cells.

    This respects shared items within a benchmark better than cell-bootstrap over
    14 point estimates. Sample pools for SC@3N are *not* re-drawn (modes/margins
    stay plug-in); only the item multiset is resampled.
    """
    specs = _cell_specs(corpus)
    if not specs:
        return [], {"error": "no_cells"}

    # Per-cell unit → item_ids; per-benchmark unit pool = intersection across
    # models on that bench (fallback: union) so one draw serves all cells.
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

    rng = random.Random(seed)
    repl_cd_seed: List[float] = []
    repl_cd_sc3: List[float] = []
    repl_seed_sc3: List[float] = []
    # Point estimate on original (no resample), mean over splits then cells.
    point_cells: List[Dict[str, float]] = []
    for spec in specs:
        per_split = [
            _eval_on_views(
                spec["cfg_views"],
                spec["seed_by_id"],
                spec["seed_pool"],
                spec["configs"],
                spec["reference"],
                spec["c0_seeds"],
                val_fraction,
                int(ss),
            )
            for ss in split_seeds
        ]
        avg: Dict[str, float] = {}
        keys = set().union(*(p.keys() for p in per_split if p))
        for k in keys:
            vals = [float(p[k]) for p in per_split if k in p and math.isfinite(float(p[k]))]
            if vals:
                avg[k] = sum(vals) / len(vals)
        point_cells.append(avg)

    def _mean_key(cells: Sequence[Dict[str, float]], key: str) -> float:
        vals = [
            float(c[key])
            for c in cells
            if key in c and math.isfinite(float(c[key]))
        ]
        return sum(vals) / len(vals) if vals else float("nan")

    point = {
        "cd_minus_seed": _mean_key(point_cells, "delta_cd_minus_seed"),
        "cd_minus_sc3n": _mean_key(point_cells, "delta_cd_minus_sc3n"),
        "seed_minus_sc3n": _mean_key(point_cells, "delta_seed_minus_sc3n"),
        "n_cells": float(len(specs)),
        "n_seed_cells": float(
            sum(1 for c in point_cells if "delta_cd_minus_seed" in c)
        ),
    }

    for b in range(n_bootstrap):
        # Joint unit resample per benchmark.
        drawn: Dict[Tuple[str, Optional[str]], List[str]] = {}
        for bk, units in bench_units.items():
            if not units:
                drawn[bk] = []
                continue
            drawn[bk] = [units[rng.randrange(len(units))] for _ in units]

        boot_cells: List[Dict[str, float]] = []
        for i, spec in enumerate(specs):
            bk = (spec["dataset"], spec["subset"])
            u2i = cell_unit_to_items[i]
            boot_views: List[sel.ItemView] = []
            for u in drawn[bk]:
                for item_id in u2i.get(u, []):
                    # Duplicate the view object; item_id collision across draws is
                    # intentional for with-replacement — tag with draw index via
                    # a shallow copy of fields isn't needed because split uses
                    # unit; duplicate the same ItemView multiple times.
                    boot_views.append(spec["cfg_by_id"][item_id])
            if len(boot_views) < 8:
                continue
            # Unique-ify item_ids for seed lookup by using repeated views as-is;
            # _split_views groups by unit, so duplicates of the same unit inflate
            # that cluster — correct for unit bootstrap.
            per_split = [
                _eval_on_views(
                    boot_views,
                    spec["seed_by_id"],
                    spec["seed_pool"],
                    spec["configs"],
                    spec["reference"],
                    spec["c0_seeds"],
                    val_fraction,
                    int(ss),
                )
                for ss in split_seeds
            ]
            avg = {}
            keys = set().union(*(p.keys() for p in per_split if p))
            for k in keys:
                vals = [
                    float(p[k])
                    for p in per_split
                    if k in p and math.isfinite(float(p[k]))
                ]
                if vals:
                    avg[k] = sum(vals) / len(vals)
            boot_cells.append(avg)

        repl_cd_seed.append(100.0 * _mean_key(boot_cells, "delta_cd_minus_seed"))
        repl_cd_sc3.append(100.0 * _mean_key(boot_cells, "delta_cd_minus_sc3n"))
        repl_seed_sc3.append(100.0 * _mean_key(boot_cells, "delta_seed_minus_sc3n"))

    rows: List[Dict[str, Any]] = []
    for name, point_key, reps in (
        ("CD_minus_seed", "cd_minus_seed", repl_cd_seed),
        ("CD_minus_SC3N", "cd_minus_sc3n", repl_cd_sc3),
        ("seed_minus_SC3N", "seed_minus_sc3n", repl_seed_sc3),
    ):
        lo, hi, mean_boot = _percentile_ci(reps)
        rows.append(
            {
                "contrast": name,
                "n_bootstrap": n_bootstrap,
                "n_cells": int(point["n_cells"]),
                "n_seed_cells": int(point["n_seed_cells"]),
                "point_mean_pp": 100.0 * float(point[point_key])
                if math.isfinite(float(point[point_key]))
                else float("nan"),
                "boot_mean_pp": mean_boot,
                "ci_low_pp": lo,
                "ci_high_pp": hi,
                "approximation": (
                    "unit_resample_per_benchmark; refit_tau_on_train; "
                    "mean_over_split_seeds; plug-in sample pools unchanged"
                ),
            }
        )
    meta = {
        "n_bootstrap": n_bootstrap,
        "n_cells": int(point["n_cells"]),
        "n_seed_cells": int(point["n_seed_cells"]),
        "point_pp": {
            "cd_minus_seed": 100.0 * float(point["cd_minus_seed"]),
            "cd_minus_sc3n": 100.0 * float(point["cd_minus_sc3n"]),
            "seed_minus_sc3n": 100.0 * float(point["seed_minus_sc3n"]),
        },
    }
    return rows, meta


# ------------------------------------------------------- extraction / truncation
def extraction_rates_table(
    corpus: Corpus,
    seed: int = 0,
    configs: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """extraction_failure_rate (+ truncation) by model×dataset×config at seed0."""
    configs = list(configs) if configs else list(PRIMARY_CONFIG_IDS) + ["c3"]
    wanted = set(configs)
    grouped: Dict[Tuple[str, Optional[str], str, str], List[Cell]] = defaultdict(list)
    for cell in corpus.cells.values():
        if cell.seed != seed or cell.config not in wanted or cell.n == 0:
            continue
        grouped[(cell.dataset, cell.subset, cell.model, cell.config)].append(cell)

    rows: List[Dict[str, Any]] = []
    for (dataset, subset, model, config), cells in sorted(
        grouped.items(), key=lambda kv: tuple(str(x) for x in kv[0])
    ):
        total = sum(c.n for c in cells)
        fails = sum(c.n_extraction_failures for c in cells)
        trunc = sum(
            1
            for c in cells
            for s in c.samples
            if s.truncated
            or (s.finish_reason or "").lower() in ("length", "max_tokens")
        )
        unparsed_mode = sum(
            1 for c in cells if c.modal_class() == EXTRACTION_FAILURE_CLASS
        )
        rows.append(
            {
                "dataset": dataset,
                "subset": subset or "",
                "model": model,
                "config": config,
                "seed": seed,
                "n_items": len(cells),
                "n_samples": total,
                "extraction_failure_rate": fails / total if total else float("nan"),
                "truncation_rate": trunc / total if total else float("nan"),
                "unparsed_mode_rate": unparsed_mode / len(cells) if cells else float("nan"),
            }
        )
    return rows


def p4_surviving_cells_from_csv(path: Any) -> List[Tuple[str, str, str]]:
    """Read tab_p4_seed_calibration.csv survivors (p4_survives_seed_null_05)."""
    import csv
    from pathlib import Path

    p = Path(path)
    if not p.exists():
        return []
    out: List[Tuple[str, str, str]] = []
    with p.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            flag = str(row.get("p4_survives_seed_null_05", "")).lower()
            if flag in ("true", "1", "yes"):
                out.append(
                    (
                        row["dataset"],
                        row.get("subset") or "",
                        row["model"],
                    )
                )
    return out


def p4_extraction_conditional(
    corpus: Corpus,
    survivors: Sequence[Tuple[str, str, str]],
    configs: Sequence[str] = PRIMARY_CONFIG_IDS,
    seed: int = 0,
) -> List[Dict[str, Any]]:
    """Config π_mode spread with/without items whose any-config mode is ⊥."""
    rows: List[Dict[str, Any]] = []
    for dataset, subset_s, model in survivors:
        subset: Optional[str] = subset_s if subset_s else None
        # math500 uses subset None stored as ""
        if dataset == "math500" and subset_s in ("", "main"):
            subset = None
        columns: Dict[str, Dict[str, Cell]] = {c: {} for c in configs}
        for cell in corpus.cells.values():
            if (
                cell.dataset != dataset
                or (cell.subset or "") != (subset or "")
                or cell.model != model
                or cell.seed != seed
                or cell.config not in columns
                or cell.n == 0
            ):
                continue
            columns[cell.config][cell.item_id] = cell
        if any(not columns[c] for c in configs):
            continue
        shared = set.intersection(*(set(columns[c]) for c in configs))
        if not shared:
            continue

        def spread(item_ids: Iterable[str]) -> Tuple[float, Dict[str, float]]:
            pis: Dict[str, float] = {}
            for c in configs:
                flags = [
                    columns[c][i].modal_correct()
                    for i in item_ids
                    if i in columns[c]
                ]
                pis[c] = sum(flags) / len(flags) if flags else float("nan")
            vals = [v for v in pis.values() if math.isfinite(v)]
            sp = (max(vals) - min(vals)) if vals else float("nan")
            return sp, pis

        # Items whose mode is EXTRACTION_FAILURE under any of c0/c1/c2
        bad = {
            i
            for i in shared
            for c in configs
            if columns[c][i].modal_class() == EXTRACTION_FAILURE_CLASS
        }
        kept = sorted(shared - bad)
        sp_all, pis_all = spread(shared)
        sp_kept, pis_kept = spread(kept)
        rows.append(
            {
                "dataset": dataset,
                "subset": subset_s,
                "model": model,
                "n_items_all": len(shared),
                "n_items_excluded_unparsed_mode": len(bad),
                "n_items_kept": len(kept),
                "config_range_pp_all": 100.0 * sp_all if math.isfinite(sp_all) else float("nan"),
                "config_range_pp_excl_unparsed": 100.0 * sp_kept
                if math.isfinite(sp_kept)
                else float("nan"),
                "p4_gate_3pp_all": bool(math.isfinite(sp_all) and 100.0 * sp_all > 3.0),
                "p4_gate_3pp_excl_unparsed": bool(
                    math.isfinite(sp_kept) and 100.0 * sp_kept > 3.0
                ),
                "spread_persists": bool(
                    math.isfinite(sp_kept) and 100.0 * sp_kept > 3.0
                ),
                **{f"pi_{c}_all": pis_all.get(c, float("nan")) for c in configs},
                **{f"pi_{c}_excl": pis_kept.get(c, float("nan")) for c in configs},
            }
        )
    return rows


# ------------------------------------------------------- seed reorder baseline
def seed_reorder_baseline(
    corpus: Corpus,
    reference: str = REFERENCE_CONFIG_ID,
    seeds: Sequence[int] = (0, 1, 2),
    configs: Sequence[str] = PRIMARY_CONFIG_IDS,
    n_bootstrap: int = 200,
) -> List[Dict[str, Any]]:
    """Mean reorder / corrective / destructive: seed pairs as fake configs vs real."""
    from itertools import combinations

    # Cross-config at seed 0 (existing machinery).
    cfg_rows = modes_mod.transition_rows(
        corpus, configs=list(configs), seed=0, n_bootstrap=n_bootstrap
    )
    cfg_summary = modes_mod.transition_summary(cfg_rows)

    # Seed-as-config: build synthetic columns named seed0/seed1/seed2 by temporarily
    # rewriting cell.config — avoid mutating corpus; hand-roll transition records.
    seed_records_all: List[Dict[str, Any]] = []
    pair_rows: List[modes_mod.TransitionRow] = []

    columns: Dict[Tuple[str, Optional[str], str, int], Dict[str, Cell]] = {}
    wanted = set(int(s) for s in seeds)
    for cell in corpus.cells.values():
        if cell.config != reference or cell.n == 0 or int(cell.seed) not in wanted:
            continue
        columns.setdefault(
            (cell.dataset, cell.subset, cell.model, int(cell.seed)), {}
        )[cell.item_id] = cell

    keys = sorted({(d, s, m) for (d, s, m, _seed) in columns})
    for dataset, subset, model in keys:
        for sa, sb in combinations(sorted(wanted), 2):
            col_a = columns.get((dataset, subset, model, sa))
            col_b = columns.get((dataset, subset, model, sb))
            if not col_a or not col_b:
                continue
            items = sorted(set(col_a) & set(col_b))
            if len(items) < 4:
                continue
            records: List[Dict[str, Any]] = []
            for item in items:
                ca, cb = col_a[item], col_b[item]
                verdict = modes_mod.classify_transition(
                    ca.modal_class(),
                    ca.modal_correct(),
                    cb.modal_class(),
                    cb.modal_correct(),
                )
                verdict["item_id"] = item
                verdict["margin"] = ca.margin()
                verdict["margin_b"] = cb.margin()
                verdict["bootstrap_unit"] = ca.bootstrap_unit()
                records.append(verdict)
            seed_records_all.extend(records)
            pair_rows.append(
                modes_mod._transition_row(
                    dataset,
                    subset,
                    model,
                    f"seed{sa}",
                    f"seed{sb}",
                    records,
                    n_bootstrap,
                    0.05,
                    0,
                    modes_mod.SMALL_MARGIN_THRESHOLD,
                )
            )

    seed_summary = modes_mod.transition_summary(pair_rows) if pair_rows else {}

    def _pack(label: str, summary: Dict[str, Any], n_pairs: int) -> Dict[str, Any]:
        return {
            "baseline": label,
            "n_pairs": n_pairs,
            "mean_reorder_rate": summary.get("mean_reorder_rate", float("nan")),
            "mean_corrective_rate": summary.get("mean_corrective_rate", float("nan")),
            "mean_destructive_rate": summary.get("mean_destructive_rate", float("nan")),
            "net_corrective_rate": (
                float(summary.get("mean_corrective_rate", float("nan")))
                - float(summary.get("mean_destructive_rate", float("nan")))
                if summary
                else float("nan")
            ),
            "small_margin_reorder_rate": summary.get(
                "small_margin_reorder_rate", float("nan")
            ),
            "large_margin_reorder_rate": summary.get(
                "large_margin_reorder_rate", float("nan")
            ),
        }

    return [
        _pack("cross_config_c0c1c2_seed0", cfg_summary, len(cfg_rows)),
        _pack("seed_to_seed_c0_as_fake_configs", seed_summary, len(pair_rows)),
    ]
