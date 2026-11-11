# ICML 2027 — bulletproofing status

## Done (CPU; in `main.pdf`)

| Control | Result |
|---------|--------|
| Seed-sub / SC@3N (6 cells) | CD−seed **−0.42** [−1.20, +0.28]; CD−SC@3N **+0.39** [−0.11, +0.89] |
| Margin-split | CD−SC **+3.22** [+1.37, +5.46] — lift survives |
| Probe-then-commit tokens | **1.60×** SC@N; **0.53×** always-3; 74% early-exit |
| Drop-c2 / quality-weighted | Drop-c2 **+3.34** pp; weighted **+0.66** pp |
| Token Pareto (SC/ESC/certainty/CDV) | `fig_token_pareto.pdf` |
| P4 seed-permutation | **2/6** cells survive >3pp ∧ p<0.05 |

## GPU blocker (no local NVIDIA)

| Queue | Runs | Est. GPU-h | Command |
|-------|-----:|-----------:|---------|
| **P0** Symbolic+Llama seeds `{1,2}` | 16 | ~45 | `python scripts/run_seed_expansion_queue.py --only p0 --execute` |
| **P1** seeds `{3,4}` on 14 cells | 28 | ~105 | `--only p1 --execute` |

Kaggle: `notebooks/seed_expansion.md`. After grading:

```bash
PYTHONPATH=. python -m src.analysis.paper --results-dir results_fp16
PYTHONPATH=. python scripts/run_c3_controls.py
PYTHONPATH=. python scripts/run_c3_compute.py
PYTHONPATH=. python scripts/run_p4_seed_calibration.py
PYTHONPATH=. python scripts/run_token_pareto.py
```

Analysis auto-expands to 14 seed cells when data lands.

## Bundle

- `paper/main.pdf`
- `paper/ICML2027_submission_bundle.zip`
- Docs: `docs/C3_CONTROLS_BULLETPROOF.md`, `docs/SEED_EXPANSION_PLAN.md`
