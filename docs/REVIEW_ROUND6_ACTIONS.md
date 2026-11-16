# Review Round 6 — CPU follow-ups (14/14 seed cells)

Recomputed from graded `results_fp16` via `src.analysis.corpus.load_corpus`.
**No GPU.** Primary config pool `{c0,c1,c2}`; drop-c2 control is **global**
(always exclude `c2`, never quality-conditional).

```bash
PYTHONPATH=. python scripts/run_c3_round6.py --results-dir results_fp16
PYTHONPATH=. python scripts/run_token_pareto.py
```

## Artefacts

| File | Content |
|------|---------|
| `paper_artefacts/tab_c3_controls_summary.csv` | Cell-bootstrap CIs; now includes **`seed_minus_sc3n`** |
| `paper_artefacts/tab_c3_per_cell_contrasts.csv` | Per-cell CD−SC, seed−SC, SC3N−SC, CD−seed, CD−SC3N, seed−SC3N |
| `paper_artefacts/tab_c3_drop_c2_twoseed.csv` | Global drop-c2 CD{c0,c1} vs two-seed CD / SC@2N |
| `paper_artefacts/tab_c3_gate_ablation.csv` | Ungated mode-of-modes (τ=1.01) vs gated CD vs SC@N / SC@3N |
| `paper_artefacts/tab_token_pareto.csv` | Budget-grid curves + labeled selector points @ ~1.6× |
| `paper_artefacts/fig_token_pareto.pdf` | Accuracy vs token-multiple of SC@N |

## Headline numbers (14 cells, cell-bootstrap 95% CIs)

### 1. Seed-sub − SC@3N (new residual)

| Contrast | Mean (pp) | 95% CI |
|----------|----------:|--------|
| CD − SC@N | +2.29 | [+1.07, +3.70] |
| Seed-sub − SC@N | +2.44 | [+1.24, +3.97] |
| SC@3N − SC@N | +1.40 | [+0.63, +2.23] |
| CD − seed-sub | −0.15 | [−0.86, +0.63] |
| CD − SC@3N | +0.89 | [+0.02, +1.80] |
| **Seed-sub − SC@3N** | **+1.04** | **[+0.18, +2.10]** |

Seed-sub beats sample-count SC@3N by ~1 pp (CI above 0) — gating/aggregation
adds something beyond raw plurality over 3N c0 samples, but that something is
**not** config-specific (CD − seed-sub still crosses 0).

### 2. Drop-c2 / two-seed (Claude item C; global drop)

| Contrast | Mean (pp) | 95% CI |
|----------|----------:|--------|
| CD{c0,c1} − SC@N | **+3.34** | [+1.85, +4.93] |
| CD{c0,c1,c2} − SC@N | +2.29 | [+1.06, +3.70] |
| Two-seed CD{s0,s1} − SC@N | +3.56 | [+2.04, +5.22] |
| SC@2N − SC@N | +0.84 | [−0.02, +1.65] |
| **CD{c0,c1} − two-seed-CD** | **−0.22** | **[−1.09, +0.70]** |
| CD{c0,c1} − SC@2N | +2.50 | [+1.01, +4.08] |

**Drop policy:** global (always drop `c2`). Dropping weak `c2` *raises* the CD
lift (+3.34 vs +2.29). The config−seed residual on the matched two-seed control
still crosses zero (−0.22 pp) — same elicitation-null story as the three-way
comparison. CD{c0,c1} does beat SC@2N (+2.50), consistent with gating helping
beyond raw 2N plurality.

### 3. Token frontier @ actual expected tokens (~1.6×)

Mean probe-then-commit cost: **1.601×** SC@N tokens (frac early-exit ≈ 0.74).

| Method point | Acc (14-cell) | Token × SC@N |
|--------------|--------------:|-------------:|
| CD probe-then-commit | 0.762 | 1.601 |
| Seed-sub (same cost model) | 0.764 | 1.601 |
| SC matched (~interp SC@N→SC@3N) | 0.744 | 1.601 |
| ESC nearest available | 0.729 | 0.564 |

At matched ~1.6× tokens, CD ≈ seed-sub and both sit ~1.8 pp above
token-matched SC. ESC never reaches 1.6× of full SC@N (early-exit caps tokens
at ~0.56×); nearest reported.

### 4. Gate ablation (ungated mode-of-modes vs gated CD)

| Contrast | Mean (pp) | 95% CI |
|----------|----------:|--------|
| Ungated (τ=1.01) − SC@N | +1.72 | [−0.28, +3.67] |
| Gated CD − SC@N | +2.29 | [+1.08, +3.67] |
| Gated − ungated | +0.57 | [−0.38, +1.78] |
| Ungated − SC@3N | +0.32 | [−1.42, +1.91] |
| Gated CD − SC@3N | +0.89 | [+0.01, +1.78] |

Gating helps on point estimate (~0.6 pp) but the gated−ungated CI crosses zero.
Ungated alone does **not** beat SC@3N; gated CD’s residual vs SC@3N remains the
small positive that just clears zero.

## Interpretation (honest)

1. **Elicitation null holds on 14/14:** CD − seed-sub and CD{c0,c1} − two-seed-CD
   both cross zero.
2. **Gating / aggregation** is the remaining non-null story: seed-sub − SC@3N
   (+1.04) and gated CD − SC@3N (+0.89) sit just above zero; ungated does not.
3. **Quality confound:** global drop-c2 *improves* CD vs SC — `c2` hurts
   unweighted plurality; not free diversity.
4. **Token-matched:** at the actual ~1.6× probe cost, CD does not separate from
   seed-sub; both edge token-matched SC.

## Not done (GPU / out of scope)

- Live probe-then-commit wall-clock (offline token estimates only).
- Optional P1 seeds `{3,4}` for a stabler seed-null.
- `main.tex` left untouched (per request).
