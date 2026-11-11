# Response to third-round reviews (bulletproofing)

## Reviewer asks → status

| Ask | Status |
|-----|--------|
| Seed controls on all 14 cells | **Queued** P0 (16 runs, ~45 GPU-h); no local GPU |
| More seeds (3→5) | **Queued** P1 (28 runs, ~105 GPU-h) |
| CIs on CD−seed / CD−SC@3N | **Done** — both cross 0 |
| Independent margin-split | **Done** — lift survives (+3.2 pp) |
| Token Pareto + ESC/certainty | **Done** — `fig_token_pareto.pdf` |
| Probe-then-commit / expected tokens | **Done** — 1.60× SC@N |
| Quality-matched pooling | **Done** — drop-c2 helps; mixture ≱ SC@3N |
| P4 vs seed-range permutation | **Done** — 2/6 survive |
| Inline tables | **Done** |

## Locked numbers (current corpus)

- CD−SC (14): **+2.29** [+1.07, +3.70]
- CD−seed (6): **−0.42** [−1.20, +0.28]
- CD−SC@3N (6): **+0.39** [−0.11, +0.89]
- Probe tokens: **1.60×** SC@N; **0.53×** always-3
- P4 seed-calibrated survivors: **2/6**

## To finish bulletproofing

Run on Kaggle (see `notebooks/seed_expansion.md`):

```bash
python scripts/run_seed_expansion_queue.py --only p0 --execute
# after P0 grades, optionally:
python scripts/run_seed_expansion_queue.py --only p1 --execute
```

Then re-run analysis scripts; `has_seed_control` expands automatically.
