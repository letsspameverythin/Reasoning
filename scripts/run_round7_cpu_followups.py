#!/usr/bin/env python
"""Round-7 CPU follow-ups: P4 combined test, truncation-by-config, seed-reorder asymmetry.

No GPU. Reads graded ``results_fp16`` + ``tab_p4_seed_calibration.csv``.
Does not touch ``main.tex``.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import defaultdict
from itertools import combinations, product
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.analysis.corpus import Cell, Corpus, load_corpus
from src.analysis import modes as modes_mod
from src.analysis.measurement_audits import p4_surviving_cells_from_csv
from src.answers import EXTRACTION_FAILURE_CLASS
from src.elicitation import PRIMARY_CONFIG_IDS, REFERENCE_CONFIG_ID
from src.utils import setup_logging


CellKey = Tuple[str, str, str]  # dataset, subset_str, model


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
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


def _subset_norm(dataset: str, subset: Optional[str]) -> str:
    if dataset == "math500":
        return ""
    return (subset or "") if subset is not None else ""


def _short_model(m: str) -> str:
    return (
        m.replace("Qwen/Qwen2.5-", "")
        .replace("-Instruct", "")
        .replace("meta-llama/Llama-3.2-", "Llama")
    )


def _exact_seed_null_p(config_range: float, seed_vec: np.ndarray) -> float:
    """Exact with-replacement null over |S|^3 assignments of seed π to 3 slots.

    With |S|=3 this is 27 atoms. Cells with config_range > observed seed_range
    get p=0. Round-7 prose sometimes says \"20-permutation\"; the exact catalogue
    for three seeds is 27 (min positive atom mass 1/27 ≈ 0.037).
    """
    n = len(seed_vec)
    if n < 2 or not math.isfinite(config_range):
        return float("nan")
    ge = 0
    total = 0
    for idx in product(range(n), repeat=3):
        sample = seed_vec[list(idx)]
        r = float(sample.max() - sample.min())
        total += 1
        if r + 1e-15 >= config_range:
            ge += 1
    return ge / total if total else float("nan")


def _binom_sf_ge(k: int, n: int, p: float) -> float:
    from math import comb

    return sum(comb(n, i) * (p**i) * ((1 - p) ** (n - i)) for i in range(k, n + 1))


def _modal_correct_notrunc(cell: Cell) -> bool:
    samples = [
        s
        for s in cell.samples
        if not (
            s.truncated
            or (s.finish_reason or "").lower() in ("length", "max_tokens")
        )
    ]
    if not samples:
        return False
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
        return False
    return any(s.is_correct for s in samples if s.canonical_class == mode)


def _item_is_trunc(cell: Cell) -> bool:
    return any(
        s.truncated or (s.finish_reason or "").lower() in ("length", "max_tokens")
        for s in cell.samples
    )


# --------------------------------------------------------------------------- corpus index
def _index_cells(
    corpus: Corpus,
) -> Dict[CellKey, Dict[str, Dict[int, Dict[str, Cell]]]]:
    store: Dict[CellKey, Dict[str, Dict[int, Dict[str, Cell]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(dict))
    )
    for cell in corpus.cells.values():
        if cell.config not in PRIMARY_CONFIG_IDS or cell.n == 0:
            continue
        key = (cell.dataset, _subset_norm(cell.dataset, cell.subset), cell.model)
        store[key][cell.config][int(cell.seed)][cell.item_id] = cell
    return store


def _cell_item_flags(
    by_cfg: Dict[str, Dict[int, Dict[str, Cell]]],
) -> Optional[Dict[str, Any]]:
    seed_maps = by_cfg.get(REFERENCE_CONFIG_ID, {})
    seeds = sorted(seed_maps)
    if len(seeds) < 2:
        return None
    conf_maps = {}
    for cfg in PRIMARY_CONFIG_IDS:
        m = by_cfg.get(cfg, {}).get(0)
        if not m:
            return None
        conf_maps[cfg] = m

    shared_seed = set.intersection(*(set(seed_maps[s]) for s in seeds))
    shared_conf = set.intersection(*(set(conf_maps[c]) for c in PRIMARY_CONFIG_IDS))
    shared = sorted(shared_seed & shared_conf)
    if len(shared) < 4:
        return None

    unit_to_items: Dict[str, List[str]] = defaultdict(list)
    for iid in shared:
        u = seed_maps[seeds[0]][iid].bootstrap_unit()
        unit_to_items[u].append(iid)
    unique_units = sorted(unit_to_items)

    seed_ok = {
        s: np.array([bool(seed_maps[s][i].modal_correct()) for i in shared], dtype=bool)
        for s in seeds
    }
    conf_ok = {
        c: np.array([bool(conf_maps[c][i].modal_correct()) for i in shared], dtype=bool)
        for c in PRIMARY_CONFIG_IDS
    }
    conf_ok_notrunc = {
        c: np.array(
            [_modal_correct_notrunc(conf_maps[c][i]) for i in shared], dtype=bool
        )
        for c in PRIMARY_CONFIG_IDS
    }
    trunc_any = np.array(
        [
            any(_item_is_trunc(conf_maps[c][iid]) for c in PRIMARY_CONFIG_IDS)
            for iid in shared
        ],
        dtype=bool,
    )
    trunc_rate = np.array(
        [
            sum(
                1
                for c in PRIMARY_CONFIG_IDS
                for s in conf_maps[c][iid].samples
                if s.truncated
                or (s.finish_reason or "").lower() in ("length", "max_tokens")
            )
            / max(
                1,
                sum(len(conf_maps[c][iid].samples) for c in PRIMARY_CONFIG_IDS),
            )
            for iid in shared
        ],
        dtype=float,
    )

    return {
        "item_ids": shared,
        "seeds": seeds,
        "seed_ok": seed_ok,
        "conf_ok": conf_ok,
        "conf_ok_notrunc": conf_ok_notrunc,
        "trunc_any": trunc_any,
        "trunc_rate": trunc_rate,
        "unique_units": unique_units,
        "unit_to_items": dict(unit_to_items),
        "item_index": {iid: j for j, iid in enumerate(shared)},
        "conf_maps": conf_maps,
    }


def _ranges_from_flags(
    seed_ok: Dict[int, np.ndarray],
    conf_ok: Dict[str, np.ndarray],
    mask: Optional[np.ndarray] = None,
) -> Tuple[float, float, float]:
    def mean_ok(arr: np.ndarray) -> float:
        if mask is not None:
            arr = arr[mask]
        return float(arr.mean()) if len(arr) else float("nan")

    seed_pis = np.array([mean_ok(seed_ok[s]) for s in sorted(seed_ok)], dtype=float)
    conf_pis = np.array([mean_ok(conf_ok[c]) for c in PRIMARY_CONFIG_IDS], dtype=float)
    seed_range = float(seed_pis.max() - seed_pis.min())
    config_range = float(conf_pis.max() - conf_pis.min())
    return config_range, seed_range, config_range - seed_range


def _survives(config_range: float, p_null: float, alpha: float = 0.05) -> bool:
    return bool(config_range > 0.03 + 1e-15 and p_null < alpha)


# --------------------------------------------------------------------------- P4 combined
def run_p4_combined(
    store: Dict[CellKey, Dict[str, Dict[int, Dict[str, Cell]]]],
    calib_path: Path,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any], str]:
    """Point estimates from tab_p4_seed_calibration; item-bootstrap from corpus."""
    calib = pd.read_csv(calib_path)
    cell_rows: List[Dict[str, Any]] = []
    packed: List[Dict[str, Any]] = []

    for _, row in calib.iterrows():
        dataset = str(row["dataset"])
        subset = "" if dataset == "math500" else str(row.get("subset") or "")
        if subset == "nan":
            subset = ""
        model = str(row["model"])
        key = (dataset, subset, model)

        seed_cols = [
            c for c in ("pi_c0_seed0", "pi_c0_seed1", "pi_c0_seed2") if c in row.index
        ]
        seed_vec = np.array([float(row[c]) for c in seed_cols], dtype=float)
        conf_vec = np.array(
            [float(row[f"pi_{c}_seed0"]) for c in PRIMARY_CONFIG_IDS], dtype=float
        )
        # Prefer calibration's published ranges when present (matches 5/14 gate).
        cfg_r = float(row["config_range_pp"]) / 100.0
        seed_r = float(row["seed_range_pp"]) / 100.0
        # sanity vs vectors
        cfg_r_v = float(conf_vec.max() - conf_vec.min())
        seed_r_v = float(seed_vec.max() - seed_vec.min())
        if abs(cfg_r - cfg_r_v) > 1e-6:
            cfg_r = cfg_r_v
        if abs(seed_r - seed_r_v) > 1e-6:
            seed_r = seed_r_v

        p_exact = _exact_seed_null_p(cfg_r, seed_vec)
        # Official survivor flag from calibration CSV (authoritative 5/14).
        surv_flag = str(row.get("p4_survives_seed_null_05", "")).lower() in (
            "true",
            "1",
            "yes",
        )
        cell_rows.append(
            {
                "dataset": dataset,
                "subset": subset,
                "model": model,
                "n_seeds": int(row.get("n_seeds", len(seed_vec))),
                "config_range_pp": 100.0 * cfg_r,
                "seed_range_pp": 100.0 * seed_r,
                "config_minus_seed_pp": 100.0 * (cfg_r - seed_r),
                "config_exceeds_seed_range": bool(cfg_r > seed_r + 1e-15),
                "p_exact_seed_null_27": p_exact,
                "p_calib_bootstrap": float(row.get("p_config_range_le_seed_null", float("nan"))),
                "p4_gate_3pp": bool(cfg_r > 0.03 + 1e-15),
                "p4_survives_seed_null_05": surv_flag,
                "n_null_atoms": len(seed_vec) ** 3,
            }
        )
        flags = _cell_item_flags(store[key]) if key in store else None
        if flags is not None:
            packed.append({"key": key, "flags": flags, "row": cell_rows[-1]})

    n_cells = len(cell_rows)
    n_surv = sum(1 for r in cell_rows if r["p4_survives_seed_null_05"])
    n_cfg_gt_seed = sum(1 for r in cell_rows if r["config_exceeds_seed_range"])
    mean_diff_pp = float(np.mean([r["config_minus_seed_pp"] for r in cell_rows]))
    pr_ge = _binom_sf_ge(n_surv, n_cells, alpha) if n_cells else float("nan")

    boot_surv: List[int] = []
    boot_mean_diff: List[float] = []
    boot_n_gt: List[int] = []
    for _ in range(n_boot):
        surv_b = 0
        diffs: List[float] = []
        gt = 0
        for p in packed:
            flags = p["flags"]
            units = flags["unique_units"]
            chosen_units = [units[i] for i in rng.integers(0, len(units), size=len(units))]
            idxs: List[int] = []
            item_index = flags["item_index"]
            for u in chosen_units:
                for iid in flags["unit_to_items"][u]:
                    idxs.append(item_index[iid])
            if len(idxs) < 4:
                continue
            idx = np.array(idxs, dtype=int)
            seed_ok_b = {s: flags["seed_ok"][s][idx] for s in flags["seeds"]}
            conf_ok_b = {c: flags["conf_ok"][c][idx] for c in PRIMARY_CONFIG_IDS}
            cfg_r, seed_r, diff = _ranges_from_flags(seed_ok_b, conf_ok_b)
            seed_vec = np.array(
                [float(seed_ok_b[s].mean()) for s in flags["seeds"]], dtype=float
            )
            p_exact = _exact_seed_null_p(cfg_r, seed_vec)
            if _survives(cfg_r, p_exact, alpha=alpha):
                surv_b += 1
            if cfg_r > seed_r + 1e-15:
                gt += 1
            diffs.append(100.0 * diff)
        boot_surv.append(surv_b)
        boot_mean_diff.append(float(np.mean(diffs)) if diffs else float("nan"))
        boot_n_gt.append(gt)

    boot_surv_a = np.asarray(boot_surv, dtype=float)
    boot_diff_a = np.asarray(boot_mean_diff, dtype=float)
    boot_gt_a = np.asarray(boot_n_gt, dtype=float)

    def _ci(a: np.ndarray) -> Tuple[float, float]:
        return float(np.quantile(a, 0.025)), float(np.quantile(a, 0.975))

    surv_lo, surv_hi = _ci(boot_surv_a)
    diff_lo, diff_hi = _ci(boot_diff_a)
    gt_lo, gt_hi = _ci(boot_gt_a)

    ps = np.array([max(float(r["p_exact_seed_null_27"]), 1e-12) for r in cell_rows])
    fisher_stat = float(-2.0 * np.log(ps).sum())
    try:
        from scipy.stats import chi2

        fisher_p = float(chi2.sf(fisher_stat, df=2 * n_cells))
    except Exception:
        fisher_p = float("nan")

    summary = {
        "n_cells": n_cells,
        "n_survivors": n_surv,
        "n_config_gt_seed": n_cfg_gt_seed,
        "mean_config_minus_seed_pp": mean_diff_pp,
        "alpha": alpha,
        "pr_ge_survivors_indep_binom": pr_ge,
        "item_boot_B": n_boot,
        "boot_mean_n_survivors": float(boot_surv_a.mean()),
        "boot_n_survivors_ci_low": surv_lo,
        "boot_n_survivors_ci_high": surv_hi,
        "boot_mean_config_minus_seed_pp": float(np.nanmean(boot_diff_a)),
        "boot_mean_diff_ci_low": diff_lo,
        "boot_mean_diff_ci_high": diff_hi,
        "boot_mean_n_config_gt_seed": float(boot_gt_a.mean()),
        "boot_n_config_gt_seed_ci_low": gt_lo,
        "boot_n_config_gt_seed_ci_high": gt_hi,
        "fisher_stat": fisher_stat,
        "fisher_p": fisher_p,
    }

    out_rows: List[Dict[str, Any]] = [{**r, "level": "cell"} for r in cell_rows]
    out_rows.append(
        {
            "level": "global_summary",
            "dataset": "",
            "subset": "",
            "model": "",
            "n_seeds": "",
            "config_range_pp": "",
            "seed_range_pp": "",
            "config_minus_seed_pp": mean_diff_pp,
            "config_exceeds_seed_range": "",
            "p_exact_seed_null_27": "",
            "p_calib_bootstrap": "",
            "p4_gate_3pp": "",
            "p4_survives_seed_null_05": n_surv,
            "n_null_atoms": "",
            "n_cells": n_cells,
            "n_survivors": n_surv,
            "n_config_gt_seed": n_cfg_gt_seed,
            "pr_ge_survivors_indep_binom_alpha05": pr_ge,
            "item_boot_B": n_boot,
            "boot_mean_n_survivors": summary["boot_mean_n_survivors"],
            "boot_n_survivors_ci95_low": surv_lo,
            "boot_n_survivors_ci95_high": surv_hi,
            "boot_mean_config_minus_seed_pp": summary["boot_mean_config_minus_seed_pp"],
            "boot_mean_diff_ci95_low": diff_lo,
            "boot_mean_diff_ci95_high": diff_hi,
            "boot_mean_n_config_gt_seed": summary["boot_mean_n_config_gt_seed"],
            "boot_n_config_gt_seed_ci95_low": gt_lo,
            "boot_n_config_gt_seed_ci95_high": gt_hi,
            "fisher_combined_stat": fisher_stat,
            "fisher_combined_p": fisher_p,
        }
    )
    return out_rows, summary, _render_p4_combined_md(cell_rows, summary)


def _render_p4_combined_md(
    cell_rows: List[Dict[str, Any]], summary: Dict[str, Any]
) -> str:
    n = summary["n_cells"]
    k = summary["n_survivors"]
    lines = [
        "# P4 combined test (seed-calibrated, CPU)",
        "",
        "Global follow-up on the 14 multi-seed cells. Per-cell ranges and the",
        "official survivor flag come from `tab_p4_seed_calibration.csv`.",
        "Per-cell null reported here: **exact with-replacement** assignment of the",
        "3 observed c0 seed π_mode values to 3 fake-config slots (3³ = **27** atoms).",
        "Survivors require config range > 3pp and one-sided p < 0.05 under the",
        "calibration seed-null. With only 3 seeds, `config_range > seed_range`",
        "implies exact p = 0 (no null atom exceeds the observed seed range).",
        "Round-7 text sometimes says \"20-permutation\"; the exact catalogue for",
        "|S|=3 is 27 (atom mass 1/27 ≈ 0.037).",
        "",
        "```bash",
        "PYTHONPATH=. python scripts/run_round7_cpu_followups.py --results-dir results_fp16",
        "```",
        "",
        "## Headline",
        "",
        f"- Survivors: **{k}/{n}** at α=0.05 (calibration gate).",
        f"- Cells with `config_range > seed_range`: **{summary['n_config_gt_seed']}/{n}**.",
        f"- Mean(`config_range − seed_range`): **{summary['mean_config_minus_seed_pp']:+.2f} pp**.",
        f"- Reference under independent Bernoulli null: "
        f"Pr(≥{k} survivors | Bin({n}, α=0.05)) ≈ **{summary['pr_ge_survivors_indep_binom']:.4g}**.",
        f"- Item-bootstrap (B={summary['item_boot_B']}, template/item units): "
        f"n_survivors mean **{summary['boot_mean_n_survivors']:.2f}** "
        f"95% CI [{summary['boot_n_survivors_ci_low']:.0f}, {summary['boot_n_survivors_ci_high']:.0f}]; "
        f"mean(config−seed) **{summary['boot_mean_config_minus_seed_pp']:+.2f}** pp "
        f"[{summary['boot_mean_diff_ci_low']:+.2f}, {summary['boot_mean_diff_ci_high']:+.2f}].",
        f"- Fisher combination of the 14 exact one-sided p-values (zeros clamped to 1e-12): "
        f"stat={summary['fisher_stat']:.1f}, p≈{summary['fisher_p']:.3g} "
        "(sensitivity only; cells are not independent).",
        "",
        "## Per-cell survivors",
        "",
        "| Cell | cfg pp | seed pp | Δ pp | p_exact27 |",
        "|------|-------:|--------:|-----:|----------:|",
    ]
    for r in cell_rows:
        if not r["p4_survives_seed_null_05"]:
            continue
        lines.append(
            f"| {_short_model(r['model'])}|{r['dataset']}|{r['subset'] or 'main'} | "
            f"{r['config_range_pp']:.2f} | {r['seed_range_pp']:.2f} | "
            f"{r['config_minus_seed_pp']:+.2f} | {r['p_exact_seed_null_27']:.3f} |"
        )
    lines += [
        "",
        "## Exchangeability (recommended language)",
        "",
        "Treating c0 seeds as exchangeable \"fake configs\" is a **calibration**",
        "device, not a proof that decoding configs are exchangeable with seeds.",
        "Seeds share the same prompt/decoding family; configs deliberately change",
        "elicitation. If that distinction makes the null dubious, report P4 as a",
        "**descriptive seed-calibrated comparison**: config π_mode range vs the",
        "observed c0 seed-to-seed range (and the fraction of cells where",
        "config_range > seed_range), rather than as a formal p-value rejection of",
        "an exchangeability null. The binomial Pr(≥k) line is a **reference** under",
        "independent α-level tests — not a joint claim about the 14 cells.",
        "",
        "## Artefact",
        "",
        "- `paper_artefacts/tab_p4_combined.csv`",
        "",
    ]
    return "\n".join(lines)


# --------------------------------------------------------------------------- truncation
def truncation_by_config(corpus: Corpus, seed: int = 0) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str, str, str], List[Cell]] = defaultdict(list)
    for cell in corpus.cells.values():
        if cell.seed != seed or cell.config not in PRIMARY_CONFIG_IDS or cell.n == 0:
            continue
        key = (
            cell.dataset,
            _subset_norm(cell.dataset, cell.subset),
            cell.model,
            cell.config,
        )
        grouped[key].append(cell)

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
        rows.append(
            {
                "dataset": dataset,
                "subset": subset,
                "model": model,
                "config": config,
                "seed": seed,
                "n_items": len(cells),
                "n_samples": total,
                "extraction_failure_rate": fails / total if total else float("nan"),
                "truncation_rate": trunc / total if total else float("nan"),
            }
        )
    return rows


def p4_truncation_conditional(
    store: Dict[CellKey, Dict[str, Dict[int, Dict[str, Cell]]]],
    survivors: Sequence[CellKey],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for key in survivors:
        if key not in store:
            continue
        flags = _cell_item_flags(store[key])
        if flags is None:
            continue
        dataset, subset, model = key
        n_all = len(flags["item_ids"])
        keep_mask = ~flags["trunc_any"]
        n_excl = int(flags["trunc_any"].sum())
        n_kept = int(keep_mask.sum())

        cfg_all, _, _ = _ranges_from_flags(flags["seed_ok"], flags["conf_ok"])
        # modes from non-truncated samples only (best-effort when any-trunc is vacuous)
        conf_nt = flags["conf_ok_notrunc"]
        pis_nt = {c: float(conf_nt[c].mean()) for c in PRIMARY_CONFIG_IDS}
        cfg_nt = float(max(pis_nt.values()) - min(pis_nt.values()))

        if n_kept >= 4:
            cfg_kept, _, _ = _ranges_from_flags(
                flags["seed_ok"], flags["conf_ok"], mask=keep_mask
            )
            pis_kept = {
                c: float(flags["conf_ok"][c][keep_mask].mean())
                for c in PRIMARY_CONFIG_IDS
            }
        else:
            cfg_kept = float("nan")
            pis_kept = {c: float("nan") for c in PRIMARY_CONFIG_IDS}

        # truncation-heavy: above-median item truncation rate (usually still mild)
        med = float(np.median(flags["trunc_rate"]))
        heavy = flags["trunc_rate"] >= med - 1e-15
        n_heavy = int(heavy.sum())
        if n_heavy >= 4:
            pis_heavy = {
                c: float(flags["conf_ok"][c][heavy].mean()) for c in PRIMARY_CONFIG_IDS
            }
            heavy_range = float(max(pis_heavy.values()) - min(pis_heavy.values()))
        else:
            pis_heavy = {c: float("nan") for c in PRIMARY_CONFIG_IDS}
            heavy_range = float("nan")

        pis_all = {c: float(flags["conf_ok"][c].mean()) for c in PRIMARY_CONFIG_IDS}

        rows.append(
            {
                "dataset": dataset,
                "subset": subset,
                "model": model,
                "n_items_all": n_all,
                "n_items_excluded_any_trunc": n_excl,
                "n_items_kept_no_any_trunc": n_kept,
                "frac_items_any_trunc": n_excl / n_all if n_all else float("nan"),
                "any_trunc_filter_vacuous": bool(n_kept < 4),
                "config_range_pp_all": 100.0 * cfg_all,
                "config_range_pp_excl_any_trunc": 100.0 * cfg_kept
                if math.isfinite(cfg_kept)
                else float("nan"),
                "config_range_pp_mode_notrunc_samples": 100.0 * cfg_nt,
                "p4_gate_3pp_all": bool(cfg_all > 0.03),
                "p4_gate_3pp_excl_any_trunc": bool(
                    math.isfinite(cfg_kept) and cfg_kept > 0.03
                ),
                "p4_gate_3pp_mode_notrunc_samples": bool(cfg_nt > 0.03),
                "spread_persists_mode_notrunc": bool(cfg_nt > 0.03),
                "config_range_pp_trunc_heavy_ge_median": 100.0 * heavy_range
                if math.isfinite(heavy_range)
                else float("nan"),
                "n_trunc_heavy_ge_median": n_heavy,
                "median_item_trunc_rate": med,
                **{f"pi_{c}_all": pis_all[c] for c in PRIMARY_CONFIG_IDS},
                **{f"pi_{c}_excl_any_trunc": pis_kept[c] for c in PRIMARY_CONFIG_IDS},
                **{f"pi_{c}_mode_notrunc": pis_nt[c] for c in PRIMARY_CONFIG_IDS},
                **{f"pi_{c}_trunc_heavy": pis_heavy[c] for c in PRIMARY_CONFIG_IDS},
            }
        )
    return rows


# --------------------------------------------------------------------------- seed reorder asymmetry
def diagnose_seed_reorder_asymmetry(
    corpus: Corpus,
    seeds: Sequence[int] = (0, 1, 2),
) -> Tuple[Dict[str, Any], str]:
    wanted = sorted(int(s) for s in seeds)
    columns: Dict[Tuple[str, Optional[str], str, int], Dict[str, Cell]] = {}
    for cell in corpus.cells.values():
        if (
            cell.config != REFERENCE_CONFIG_ID
            or cell.n == 0
            or int(cell.seed) not in wanted
        ):
            continue
        columns.setdefault(
            (cell.dataset, cell.subset, cell.model, int(cell.seed)), {}
        )[cell.item_id] = cell

    keys = sorted({(d, s, m) for (d, s, m, _seed) in columns})

    seed_accs: Dict[int, List[float]] = {s: [] for s in wanted}
    for dataset, subset, model in keys:
        for s in wanted:
            col = columns.get((dataset, subset, model, s))
            if not col:
                continue
            flags = [c.modal_correct() for c in col.values()]
            seed_accs[s].append(sum(flags) / len(flags) if flags else float("nan"))
    mean_acc = {s: float(np.nanmean(seed_accs[s])) for s in wanted}

    def _pair_stats(orient: str) -> Dict[str, Any]:
        n_items = 0
        n_corr = 0
        n_dest = 0
        n_reorder = 0
        n_pairs = 0
        dest_from0 = corr_from0 = items_from0 = 0
        dest_not0 = corr_not0 = items_not0 = 0

        for dataset, subset, model in keys:
            for sa, sb in combinations(wanted, 2):
                col_a = columns.get((dataset, subset, model, sa))
                col_b = columns.get((dataset, subset, model, sb))
                if not col_a or not col_b:
                    continue
                items = sorted(set(col_a) & set(col_b))
                if len(items) < 4:
                    continue

                def one_dir(src: int, dst: int) -> None:
                    nonlocal n_items, n_corr, n_dest, n_reorder, n_pairs
                    nonlocal dest_from0, corr_from0, items_from0
                    nonlocal dest_not0, corr_not0, items_not0
                    csrc = columns[(dataset, subset, model, src)]
                    cdst = columns[(dataset, subset, model, dst)]
                    n_pairs += 1
                    for item in items:
                        ca, cb = csrc[item], cdst[item]
                        v = modes_mod.classify_transition(
                            ca.modal_class(),
                            ca.modal_correct(),
                            cb.modal_class(),
                            cb.modal_correct(),
                        )
                        n_items += 1
                        if v["reordered"]:
                            n_reorder += 1
                        if v["kind"] == "corrective":
                            n_corr += 1
                        elif v["kind"] == "destructive":
                            n_dest += 1
                        if src == 0:
                            items_from0 += 1
                            if v["kind"] == "destructive":
                                dest_from0 += 1
                            elif v["kind"] == "corrective":
                                corr_from0 += 1
                        else:
                            items_not0 += 1
                            if v["kind"] == "destructive":
                                dest_not0 += 1
                            elif v["kind"] == "corrective":
                                corr_not0 += 1

                if orient == "lex":
                    one_dir(sa, sb)
                elif orient == "flipped":
                    one_dir(sb, sa)
                else:
                    one_dir(sa, sb)
                    one_dir(sb, sa)

        return {
            "orient": orient,
            "n_pair_directions": n_pairs,
            "n_item_rows": n_items,
            "reorder_rate": n_reorder / n_items if n_items else float("nan"),
            "corrective_rate": n_corr / n_items if n_items else float("nan"),
            "destructive_rate": n_dest / n_items if n_items else float("nan"),
            "net_corrective_rate": (n_corr - n_dest) / n_items if n_items else float("nan"),
            "dest_rate_source_seed0": dest_from0 / items_from0 if items_from0 else float("nan"),
            "corr_rate_source_seed0": corr_from0 / items_from0 if items_from0 else float("nan"),
            "dest_rate_source_not0": dest_not0 / items_not0 if items_not0 else float("nan"),
            "corr_rate_source_not0": corr_not0 / items_not0 if items_not0 else float("nan"),
        }

    lex = _pair_stats("lex")
    flipped = _pair_stats("flipped")
    sym = _pair_stats("symmetric")
    summary = {
        "mean_pi_mode_by_seed": mean_acc,
        "lex": lex,
        "flipped": flipped,
        "symmetric": sym,
    }
    return summary, _render_asymmetry_md(summary)


def _render_asymmetry_md(summary: Dict[str, Any]) -> str:
    mean_acc = summary["mean_pi_mode_by_seed"]
    lex = summary["lex"]
    flipped = summary["flipped"]
    sym = summary["symmetric"]
    lines = [
        "# Seed-pair reorder asymmetry diagnosis",
        "",
        "From `tab_seed_reorder_baseline.csv`, seed-to-seed pairs show",
        f"destructive ≈ {100 * lex['destructive_rate']:.1f}% vs corrective ≈ "
        f"{100 * lex['corrective_rate']:.1f}% under the lex-oriented baseline.",
        "This note checks whether that gap is an artefact of pair orientation.",
        "",
        "## Mechanism in the baseline code",
        "",
        "`seed_reorder_baseline` iterates `combinations(sorted(seeds), 2)`, so every",
        "pair is oriented **low seed index → high** (`seed0→seed1`, `seed0→seed2`,",
        "`seed1→seed2`). `classify_transition` labels",
        "**destructive** = correct on source, wrong on target, and",
        "**corrective** = wrong on source, right on target.",
        "Modal ties break by **first occurrence within a cell's sample list**",
        "(not across seeds); the cross-seed asymmetry is from **pair orientation**,",
        "not the within-cell tie-break.",
        "",
        "## Mean c0 π_mode by seed (14 cells)",
        "",
    ]
    for s, a in sorted(mean_acc.items()):
        lines.append(f"- seed {s}: **{100 * a:.2f}%**")
    ordered = sorted(mean_acc.items(), key=lambda kv: -kv[1])
    lines += [
        "",
        f"Best mean seed: **{ordered[0][0]}** ({100 * ordered[0][1]:.2f}%). "
        "If seed0 is better on average, lex orientation (seed0 as source on 2/3 of",
        "pairs) inflates destructive relative to corrective.",
        "",
        "## Oriented rates (pooled item×pair rows)",
        "",
        "| Orientation | Destructive | Corrective | Net corr |",
        "|-------------|------------:|-----------:|---------:|",
        f"| Lex (baseline, low→high) | {100 * lex['destructive_rate']:.2f}% | "
        f"{100 * lex['corrective_rate']:.2f}% | {100 * lex['net_corrective_rate']:+.2f}% |",
        f"| Flipped (high→low) | {100 * flipped['destructive_rate']:.2f}% | "
        f"{100 * flipped['corrective_rate']:.2f}% | {100 * flipped['net_corrective_rate']:+.2f}% |",
        f"| Symmetric (both dirs) | {100 * sym['destructive_rate']:.2f}% | "
        f"{100 * sym['corrective_rate']:.2f}% | {100 * sym['net_corrective_rate']:+.2f}% |",
        "",
        "### Source = seed0 vs not (lex orientation)",
        "",
        f"- Pairs with source seed0 (`0→1`, `0→2`): dest "
        f"{100 * lex['dest_rate_source_seed0']:.2f}% / corr "
        f"{100 * lex['corr_rate_source_seed0']:.2f}%",
        f"- Pair `1→2` only: dest {100 * lex['dest_rate_source_not0']:.2f}% / corr "
        f"{100 * lex['corr_rate_source_not0']:.2f}%",
        "",
        "## Verdict",
        "",
    ]
    gap_lex = lex["destructive_rate"] - lex["corrective_rate"]
    gap_sym = abs(sym["destructive_rate"] - sym["corrective_rate"])
    if gap_lex > 0.005 and gap_sym < 0.005:
        verdict = (
            "The dest>corr gap is **mostly an orientation artefact**: it flips under "
            "reversed pairs and collapses under symmetric averaging. Prefer reporting "
            "symmetric (or absolute) transition rates for seed-seed baselines, or "
            "state explicitly that pairs are oriented low→high seed index."
        )
    elif gap_lex > 0.005 and lex["dest_rate_source_seed0"] > lex[
        "corr_rate_source_seed0"
    ] + 0.005:
        verdict = (
            "Asymmetry is concentrated on **seed0-as-source** pairs and tracks "
            "seed0's mean accuracy edge. Treat the ~4.6% vs ~3.4% gap as "
            "**lex-order / seed0-as-source bias**, not evidence that seed noise is "
            "systematically destructive."
        )
    else:
        verdict = (
            "Orientation contributes, but a residual gap may remain after "
            "symmetrizing; report both lex and symmetric rates."
        )
    lines += [verdict, ""]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_fp16")
    ap.add_argument("--out-dir", default="paper_artefacts")
    ap.add_argument("--p4-csv", default="paper_artefacts/tab_p4_seed_calibration.csv")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    setup_logging()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    t0 = time.time()
    print(f"loading corpus from {args.results_dir} ...")
    corpus = load_corpus(args.results_dir)
    print(f"  {len(corpus.cells)} cells in {time.time() - t0:.1f}s")
    store = _index_cells(corpus)

    print("P4 combined test ...")
    t1 = time.time()
    p4_rows, p4_summary, p4_md = run_p4_combined(
        store, Path(args.p4_csv), n_boot=args.n_boot, rng=rng
    )
    p4_path = out_dir / "tab_p4_combined.csv"
    _write_csv(p4_path, p4_rows)
    docs_p4 = Path("docs/P4_COMBINED.md")
    docs_p4.write_text(p4_md, encoding="utf-8")
    print(f"  wrote {p4_path} and {docs_p4} in {time.time() - t1:.1f}s")
    print(
        f"  survivors={p4_summary['n_survivors']}/{p4_summary['n_cells']}  "
        f"Pr(>=k|Bin)={p4_summary['pr_ge_survivors_indep_binom']:.4g}  "
        f"mean_delta={p4_summary['mean_config_minus_seed_pp']:+.2f}pp  "
        f"boot_surv_CI=[{p4_summary['boot_n_survivors_ci_low']:.0f},"
        f"{p4_summary['boot_n_survivors_ci_high']:.0f}]"
    )

    print("truncation by config ...")
    t2 = time.time()
    trunc_rows = truncation_by_config(corpus)
    trunc_path = out_dir / "tab_truncation_by_config.csv"
    _write_csv(trunc_path, trunc_rows)

    survivors_list = p4_surviving_cells_from_csv(args.p4_csv)
    survivors: List[CellKey] = [
        (d, "" if d == "math500" else (s or ""), m) for (d, s, m) in survivors_list
    ]
    cond_rows = p4_truncation_conditional(store, survivors)
    cond_path = out_dir / "tab_p4_truncation_conditional.csv"
    _write_csv(cond_path, cond_rows)
    print(f"  wrote {trunc_path} ({len(trunc_rows)} rows) in {time.time() - t2:.1f}s")
    print(f"  wrote {cond_path} ({len(cond_rows)} survivors)")
    for r in cond_rows:
        print(
            f"  {_short_model(r['model'])}|{r['dataset']}|{r['subset'] or 'main'}: "
            f"all={r['config_range_pp_all']:.2f} "
            f"notrunc_mode={r['config_range_pp_mode_notrunc_samples']:.2f}pp "
            f"persist={r['spread_persists_mode_notrunc']} "
            f"any_trunc_frac={100 * float(r['frac_items_any_trunc']):.0f}%"
        )

    print("seed-reorder asymmetry diagnosis ...")
    t3 = time.time()
    asym_summary, asym_md = diagnose_seed_reorder_asymmetry(corpus)
    asym_path = Path("docs/SEED_REORDER_ASYMMETRY.md")
    asym_path.write_text(asym_md, encoding="utf-8")
    print(f"  wrote {asym_path} in {time.time() - t3:.1f}s")
    lex = asym_summary["lex"]
    sym = asym_summary["symmetric"]
    print(
        f"  lex dest/corr={100 * lex['destructive_rate']:.2f}/"
        f"{100 * lex['corrective_rate']:.2f}%  "
        f"sym dest/corr={100 * sym['destructive_rate']:.2f}/"
        f"{100 * sym['corrective_rate']:.2f}%"
    )
    print(f"done in {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
