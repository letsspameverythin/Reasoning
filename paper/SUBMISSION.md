# ICML 2027 — fractional-credit rewrite (DROP gating claim)

## Verdict from Claude's tie audit

Under **fractional credit**, the any-winner “gating/aggregation” claim **dies**:

| Contrast | Fixed | Fractional |
|----------|------:|-----------:|
| CD − SC@N | +2.29 | **+0.65** |
| Seed-sub − SC@3N | +1.04 | **−0.50** (flips) |
| CD − SC@3N | +0.89 | **−0.80** (flips) |
| CD − seed-sub | −0.15 | −0.29 (still null) |
| Drop-c2 − SC@N | +3.34 | **−0.05** |
| Token@1.6× CD/seed/SC | 0.762/0.764/0.744 | **0.744/0.747/0.742** |

Paper now leads with: CDV negative + tie-aware protocol warning.

## Optional P1 seeds (deeper; ChatGPT/Claude say can wait)

```python
!python scripts/run_seed_expansion_queue.py --only p1 --dry-run
!python scripts/run_seed_expansion_queue.py --only p1 --execute --stop-on-error
```

28 runs ≈105 GPU-h. Zip `results_fp16` after each session as before.

## Bundle
`paper/main.pdf`, `paper/ICML2027_submission_bundle.zip`
