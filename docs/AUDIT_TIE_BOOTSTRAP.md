# Measurement audits: ties, item-bootstrap, extraction, seed-reorder

CPU-only audits on graded `results_fp16` (no GPU).

```bash
PYTHONPATH=. python scripts/run_measurement_audits.py --results-dir results_fp16
```

## Artefacts

| File | Content |
|------|---------|
| `paper_artefacts/tab_tie_audit.csv` | Modal-tie fraction + fixed vs fractional credit |
| `paper_artefacts/tab_c3_item_bootstrap.csv` | Item-joint bootstrap CIs for C3 contrasts |
| `paper_artefacts/tab_extraction_rates.csv` | Extraction / truncation by model×dataset×config |
| `paper_artefacts/tab_p4_extraction_conditional.csv` | P4 spreads excluding ⊥-mode items |
| `paper_artefacts/tab_seed_reorder_baseline.csv` | Seed-pair vs cross-config reorder rates |

## 1. Tie audit

Methods: SC@N (`c0`), SC@3N (pooled `c0` seeds), conditional diversify,
seed-substitution CD, mode-of-modes. Ties = top-1 count == top-2 count
(`Cell.modal_tie` / class_counts). Fixed credit = first-occurrence for sample
pluralities; for mode-of-modes / CD fallback, the paper’s plurality rule
(credit 1 if *any* tied-winning mode is correct). Fractional = expected
accuracy under uniform choice among the `k` tied top classes/modes.

SC@N / SC@3N ties move accuracy by ≪1pp. Mode-plurality policies look ~1.7–2.0pp
more optimistic under the any-winner fixed rule than under fractional credit.

| Method | Mean tie % | Acc fixed | Acc fractional | delta (pp) |
|--------|------------|-----------|----------------|------------|
| `conditional_diversify` | 7.71 | 76.65 | 74.87 | -1.778 |
| `mode_of_modes` | 8.36 | 76.12 | 74.10 | -2.012 |
| `sc_3n_c0_seeds` | 3.38 | 75.24 | 75.19 | -0.050 |
| `sc_n_c0` | 7.07 | 74.12 | 73.89 | -0.224 |
| `seed_substitution_cd` | 6.84 | 76.77 | 75.07 | -1.708 |

## 2. Item-joint bootstrap (C3)

B=1000 replicates. **Approximation:** resample bootstrap units
(template_id on GSM-Symbolic, else item_id) jointly within each
`(dataset, subset)` from the intersection of units across models; rebuild
each cell’s ItemView multiset; re-fit τ on the train fold; recompute
CD / seed-CD / SC@3N; average over split seeds `{0..4}`; then mean across
cells. Plug-in sample pools for SC@3N are not re-drawn.

This differs from `tab_c3_controls_summary.csv`, which cell-bootstraps the
14 point estimates and ignores shared items across models.

| Contrast | Point (pp) | 95% CI (item-boot) |
|----------|------------|---------------------|
| CD_minus_seed | -0.150 | [-1.075, +0.655] |
| CD_minus_SC3N | +0.892 | [-0.046, +2.095] |
| seed_minus_SC3N | +1.041 | [+0.386, +2.102] |

**Read-out:** CD−seed still crosses zero (configs ≉ distinguishable from seeds).
CD−SC3N point stays ~+0.9pp but the item-joint CI *grazes* zero (cell-bootstrap
in `tab_c3_controls_summary` was barely above zero). Seed-CD−SC3N is clearly
positive — most of the lift over SC@3N is available from seeds alone.

## 3. Extraction / truncation

Per model×dataset×config at seed 0 (`extraction_failure_rate`,
`truncation_rate` from sample `finish_reason` / `truncated`).

- Mean extraction failure rate (seed0 primary+c3 cells): **0.001%**
- Mean truncation rate: **6.82%**
- Worst extraction: `meta-llama/Llama-3.2-3B-Instruct` / math500 / c1 = 0.021%
- Worst truncation: `meta-llama/Llama-3.2-3B-Instruct` / math500 / c1 = 16.21%

### P4 survivors conditional on no ⊥ mode

Exclude items whose plug-in mode is `EXTRACTION_FAILURE_CLASS` under any of
`c0`/`c1`/`c2`. Report whether config π_mode range still exceeds 3pp.

| Cell | Range all (pp) | Range excl ⊥ (pp) | Persist >3pp | n excl |
|------|----------------|-------------------|--------------|--------|
| 1.5B\|gsm8k\|main | 6.50 | 6.50 | True | 0 |
| 7B\|gsm8k\|main | 6.50 | 6.50 | True | 0 |
| 3B\|gsm_symbolic\|main | 10.67 | 10.67 | True | 0 |
| 7B\|gsm_symbolic\|main | 13.33 | 13.33 | True | 0 |
| 7B\|gsm_symbolic\|p2 | 6.00 | 6.00 | True | 0 |

## 4. Seed-to-seed reorder baseline

Treat `c0` seeds `{0,1,2}` as fake configs and compare mean reorder /
corrective / destructive rates to cross-config pairs at seed 0.

| Baseline | n pairs | Reorder | Corrective | Destructive | Net corr |
|----------|---------|---------|------------|-------------|----------|
| `cross_config_c0c1c2_seed0` | 42 | 21.64% | 3.97% | 6.42% | -2.44% |
| `seed_to_seed_c0_as_fake_configs` | 42 | 17.70% | 3.76% | 4.60% | -0.84% |

## Meta

```json
{
  "n_bootstrap": 1000,
  "n_cells": 14,
  "n_seed_cells": 14,
  "point_pp": {
    "cd_minus_seed": -0.150,
    "cd_minus_sc3n": 0.892,
    "seed_minus_sc3n": 1.041
  }
}
```

