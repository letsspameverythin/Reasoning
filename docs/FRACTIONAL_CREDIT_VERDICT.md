# Fractional credit verdict (Claude tie-artifact check)

Recompute of C3 selector contrasts under **fixed** (sample: first-occurrence;
mode policies: any-winner) vs **fractional** credit (`k`-way ties → credit
`#correct / k`). Gated CD uses c0 sample plurality under the same rule.

```bash
PYTHONPATH=. python scripts/run_fractional_credit.py --results-dir results_fp16
```

## Recommendation

**DROP** the gating/aggregation claim.

Under fractional credit, the SeedSub-SC@3N residual that underwrote the gating/aggregation claim no longer clears 0 (and/or shrinks >0.5pp vs fixed any-winner). Treat the claim as a tie artifact.

## Critical contrasts (cell-bootstrap, 14 cells)

| Contrast | Fixed mean (pp) | Fixed 95% CI | Fractional mean (pp) | Fractional 95% CI | Δ (frac−fixed) |
|----------|----------------:|-------------:|---------------------:|------------------:|---------------:|
| SeedSub_minus_SC_3N | +1.04 | [+0.17, +2.08] | -0.50 | [-0.95, -0.09] | -1.54 |
| CD_minus_SC_N | +2.29 | [+1.07, +3.70] | +0.65 | [+0.18, +1.19] | -1.64 |
| CD_minus_SC_3N | +0.89 | [+0.02, +1.78] | -0.80 | [-1.40, -0.29] | -1.69 |
| Gated_minus_Ungated | +0.57 | [-0.37, +1.78] | +0.87 | [-0.04, +2.09] | +0.30 |
| CD_minus_SeedSub | -0.15 | [-0.86, +0.63] | -0.29 | [-0.72, +0.14] | -0.14 |

## All contrasts (1–13) + token frontier (14)

| # | Contrast | Fixed | Frac | Δ |
|--:|----------|------:|-----:|--:|
| 1 | CD_minus_SC_N | +2.29 | +0.65 | -1.64 |
| 2 | SeedSub_minus_SC_N | +2.44 | +0.94 | -1.50 |
| 3 | SC_3N_minus_SC_N | +1.40 | +1.44 | +0.04 |
| 4 | CD_minus_SeedSub | -0.15 | -0.29 | -0.14 |
| 5 | CD_minus_SC_3N | +0.89 | -0.80 | -1.69 |
| 6 | SeedSub_minus_SC_3N | +1.04 | -0.50 | -1.54 |
| 7 | Ungated_MoM_minus_SC_N | +1.72 | -0.22 | -1.95 |
| 8 | Gated_minus_Ungated | +0.57 | +0.87 | +0.30 |
| 9 | DropC2_CD_minus_SC_N | +3.34 | -0.05 | -3.38 |
| 10 | DropC2_minus_TwoSeedCD | -0.22 | -0.25 | -0.03 |
| 11 | TwoSeedCD_minus_SC_2N | +2.72 | -0.78 | -3.50 |
| 12 | DropC2_minus_SC_2N | +2.50 | -1.03 | -3.54 |
| 13 | MarginSplit_CD_minus_SC | +3.22 | +1.28 | -1.94 |

### Token frontier (~1.6×) accuracies

| Point | Fixed acc | Frac acc |
|-------|----------:|---------:|
| token_frontier_CD | 0.7615 | 0.7437 |
| token_frontier_SeedSub | 0.7636 | 0.7472 |
| token_frontier_SC_1p6x | 0.7436 | 0.7423 |

## Mean tie rates (policy layer, 14-cell mean)

| Policy | Tie rate (fixed eval) | Tie rate (frac eval) |
|--------|----------------------:|---------------------:|
| `sc_n` | 7.20% | 7.20% |
| `conditional_diversify` | 7.68% | 7.68% |
| `mode_of_modes` | 8.49% | 8.49% |
| `seed_substitution_cd` | 6.82% | 6.82% |
| `sc_3n` | 3.20% | 3.20% |

## Item-joint bootstrap (critical contrasts)

B=1000. Same approximation as `tab_c3_item_bootstrap.csv`, with fractional/fixed credit.

| Contrast | Rule | Point (pp) | 95% CI |
|----------|------|-----------:|-------:|
| CD_minus_SC_N | fixed | +2.29 | [+1.26, +3.05] |
| CD_minus_SeedSub | fixed | -0.15 | [-1.07, +0.66] |
| CD_minus_SC_3N | fixed | +0.89 | [-0.05, +2.10] |
| SeedSub_minus_SC_3N | fixed | +1.04 | [+0.39, +2.10] |
| CD_minus_SC_N | fractional | +0.65 | [+0.05, +1.60] |
| CD_minus_SeedSub | fractional | -0.29 | [-0.84, +0.59] |
| CD_minus_SC_3N | fractional | -0.80 | [-1.46, +0.44] |
| SeedSub_minus_SC_3N | fractional | -0.50 | [-1.09, +0.38] |

## Artefacts

| File | Content |
|------|---------|
| `paper_artefacts/tab_c3_fractional_credit.csv` | Long format: contrast × rule |
| `paper_artefacts/tab_c3_fractional_vs_fixed.csv` | Side-by-side fixed vs fractional |
| `docs/FRACTIONAL_CREDIT_VERDICT.md` | This verdict |

## Meta

```json
{
  "decision": "DROP",
  "critical": {
    "seed_sub_minus_sc3n": {
      "fixed_mean_pp": 1.0414747195524223,
      "frac_mean_pp": -0.5031156463928387,
      "frac_ci": [
        -0.9453414350901505,
        -0.08853196264004137
      ],
      "delta_pp": -1.5445903659452611
    },
    "cd_minus_sc_n": {
      "fixed_mean_pp": 2.2909936831503486,
      "frac_mean_pp": 0.646302972217177,
      "frac_ci": [
        0.17716972178758622,
        1.1917561304001891
      ],
      "delta_pp": -1.6446907109331717
    },
    "cd_minus_sc3n": {
      "fixed_mean_pp": 0.8918125122811159,
      "frac_mean_pp": -0.7970562629975014,
      "frac_ci": [
        -1.3958998873632422,
        -0.28558254873557287
      ],
      "delta_pp": -1.6888687752786173
    },
    "gated_minus_ungated": {
      "fixed_mean_pp": 0.5692854483171441,
      "frac_mean_pp": 0.8697508670440959,
      "frac_ci": [
        -0.03832959263973384,
        2.085018083759132
      ],
      "delta_pp": 0.30046541872695176
    }
  }
}
```
