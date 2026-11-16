# Option B — locked scientific plan (ICML 2027)

## One-sentence claim

Elicitation moves the self-consistency ceiling; **pooling** configurations at equal budget does not raise it; **selecting** among them recovers a large fraction of the remaining oracle–SC gap without test labels.

## Empirical snapshot (fp16 corpus, held-out 40% units, B=400 draft)

| Metric | Value |
|--------|------:|
| Multi-config cells | 14 |
| Mean oracle_config − SC | ~6.0 pp |
| Mean best label-free Δ vs SC | **+2.9 pp** |
| Mean fraction of oracle gap | **0.38 (38%)** |
| Cells with positive fraction | **13 / 14** |
| C3 supported | **True** |

Worst cell: GSM-Symbolic main × Qwen2.5-3B (negative frac) — report honestly.

## Avoiding the prior Reject

See `docs/REVIEWER_GATE.md`. Non-negotiable: camera-ready GLMM + bootstrap before PDF upload; CIs + δ=1pp TOST; held-out C3; scoped claims.

## Code

- `src/analysis/selection.py` — policies + gap metric
- `scripts/run_selection_only.py` — fast C3 table
- `src/analysis/paper.py` — writes `tab_selection_oracle_gap.csv` + predictions row
