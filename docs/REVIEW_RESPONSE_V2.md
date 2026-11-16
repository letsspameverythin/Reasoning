# Response after P0 seed expansion (14/14 controlled)

## What changed

All 16 P0 runs merged from `seed_expansion_results_fp16/` into `results_fp16/`.
Every C3 evaluation cell now has $c_0$ seeds `{0,1,2}`.

## Locked numbers (identical 14-cell set)

| Contrast | Mean (pp) | 95% CI |
|----------|----------:|--------|
| CD − SC@N | +2.29 | [+1.07, +3.70] |
| Seed-sub − SC@N | +2.44 | [+1.24, +3.97] |
| SC@3N − SC@N | +1.40 | [+0.63, +2.23] |
| **CD − seed-sub** | **−0.15** | **[−0.86, +0.63]** |
| CD − SC@3N | +0.89 | [+0.02, +1.80] |
| Seed-null oracle | +5.6 | — |
| P4 survivors (>3pp ∧ seed-null p<0.05) | **5/14** | — |

## Interpretation (honest)

- **Elicitation control (seed-sub):** still null — configs ≈ seeds.
- **Sample-count control (SC@3N):** small +0.9 pp residual (CI just above 0) —
  consistent with gating/aggregation, not prompt elicitation recovery.
- **C1:** 5 cells survive seed-calibrated P4, including Symbolic 3B/7B main.
- **C2:** unchanged negative.

Claude/ChatGPT’s main blocker (controls only on 6 low-spread cells) is closed.
Optional P1 (seeds `{3,4}`) would further stabilize the nulls.
