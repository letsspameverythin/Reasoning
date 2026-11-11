"""P4 calibration: config π_mode range vs c0 seed-to-seed range.

For each multi-seed cell, compare observed config range (c0/c1/c2 at seed 0)
to the null distribution of ranges obtained by treating c0 seeds as if they
were configs (permutation / resampling over the three seed plug-ins).

Also emits a simple Monte Carlo: sample 3 seed accuracies with replacement
from the empirical seed vector and record max-min.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis.corpus import load_corpus
from src.answers import EXTRACTION_FAILURE_CLASS


def _pi_mode_for_group(cells) -> float:
    if not cells:
        return float("nan")
    oks = []
    for cell in cells:
        mode = cell.modal_class()
        if mode is None or mode == EXTRACTION_FAILURE_CLASS:
            oks.append(False)
        else:
            oks.append(cell.modal_correct())
    return float(sum(oks) / len(oks))


def short_model(m: str) -> str:
    return (
        m.replace("Qwen/Qwen2.5-", "")
        .replace("-Instruct", "")
        .replace("meta-llama/Llama-3.2-", "Llama")
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results_fp16")
    ap.add_argument("--out", default="paper_artefacts/tab_p4_seed_calibration.csv")
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    corpus = load_corpus(args.results_dir)
    # (dataset, subset, model) -> config -> seed -> [cells]
    store: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for cell in corpus.cells.values():
        if cell.config not in ("c0", "c1", "c2") or cell.n == 0:
            continue
        key = (cell.dataset, cell.subset or "", cell.model)
        store[key][cell.config][int(cell.seed)].append(cell)

    rng = np.random.default_rng(args.seed)
    rows = []
    for key, by_cfg in sorted(store.items()):
        dataset, subset, model = key
        seed_pis = {}
        for s, cells in by_cfg.get("c0", {}).items():
            seed_pis[s] = _pi_mode_for_group(cells)
        if len(seed_pis) < 2:
            continue
        seeds_sorted = sorted(seed_pis)
        seed_vec = np.array([seed_pis[s] for s in seeds_sorted], dtype=float)
        seed_range = float(seed_vec.max() - seed_vec.min())

        conf_pis = {}
        for cfg in ("c0", "c1", "c2"):
            cells = by_cfg.get(cfg, {}).get(0, [])
            if cells:
                conf_pis[cfg] = _pi_mode_for_group(cells)
        if len(conf_pis) < 2:
            continue
        conf_vals = np.array(list(conf_pis.values()), dtype=float)
        config_range = float(conf_vals.max() - conf_vals.min())

        # Null: resample 3 values from seed_vec with replacement → range
        # (also exact permutation of assigning seed plug-ins to fake configs)
        n = len(seed_vec)
        boot = []
        for _ in range(args.n_boot):
            sample = seed_vec[rng.integers(0, n, size=3)]
            boot.append(float(sample.max() - sample.min()))
        boot = np.asarray(boot)
        # one-sided: P(seed_null_range >= config_range)
        p_ge = float((boot >= config_range).mean())
        # also report whether config_range exceeds observed seed_range
        exceeds_obs = bool(config_range > seed_range + 1e-12)

        row = {
            "dataset": dataset,
            "subset": subset,
            "model": model,
            "n_seeds": n,
            "config_range_pp": 100 * config_range,
            "seed_range_pp": 100 * seed_range,
            "config_minus_seed_pp": 100 * (config_range - seed_range),
            "p_config_range_le_seed_null": p_ge,
            "config_exceeds_obs_seed_range": exceeds_obs,
            "p4_gate_3pp": bool(config_range > 0.03),
            "p4_survives_seed_null_05": bool(config_range > 0.03 and p_ge < 0.05),
            **{f"pi_c0_seed{s}": seed_pis[s] for s in seeds_sorted},
            **{f"pi_{cfg}_seed0": conf_pis.get(cfg, float("nan")) for cfg in ("c0", "c1", "c2")},
        }
        rows.append(row)
        print(
            f"{short_model(model)}|{dataset}|{subset or 'main'}: "
            f"cfg={100*config_range:.1f} seed={100*seed_range:.1f} "
            f"p_null={p_ge:.3f} survives05={row['p4_survives_seed_null_05']}"
        )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    if len(df):
        print(
            f"\nn={len(df)}  mean cfg={df.config_range_pp.mean():.2f}  "
            f"mean seed={df.seed_range_pp.mean():.2f}  "
            f"P4>3pp={df.p4_gate_3pp.sum()}  "
            f"P4 survives seed-null@0.05={df.p4_survives_seed_null_05.sum()}"
        )
    print("wrote", out)


if __name__ == "__main__":
    main()
