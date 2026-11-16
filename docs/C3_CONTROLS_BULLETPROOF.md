# C3 controls bulletproofing (CPU-only)

Recomputed from graded `results_fp16` via `src.analysis.corpus.load_corpus`.
**No GPU inference.** Config pool is primary `{c0,c1,c2}` (`PRIMARY_CONFIG_IDS`).

```bash
PYTHONPATH=. python scripts/run_c3_controls.py --results-dir results_fp16
PYTHONPATH=. python scripts/run_c3_compute.py  --results-dir results_fp16
```

## Artefacts

| File | Content |
|------|---------|
| `paper_artefacts/tab_c3_controls.csv` | Per-cell seed-sub / SC@3N / mixture deltas (14 cells) |
| `paper_artefacts/tab_c3_controls_summary.csv` | Cell-bootstrap CIs over contrasts |
| `paper_artefacts/tab_c3_expected_tokens.csv` | Probe-then-commit token cost vs SC@N / always-3 / SC@3N |
| `paper_artefacts/tab_c3_margin_split.csv` | Independent c0 half-A margin / half-B decision |
| `paper_artefacts/tab_quality_matched_cdv.csv` | Drop-c2 CDV, quality-weighted CDV, sample mixture |

## Headline numbers (current corpus)

### Seed-matched controls (still 6 cells)

| Contrast | n | Mean (pp) | 95% CI |
|----------|---|-----------|--------|
| CD − SC (all cells) | 14 | **+2.29** | [+1.07, +3.70] |
| CD − SC (seed 6) | 6 | +1.58 | [+0.42, +2.89] |
| CD − SC (no-seed 8) | 8 | +2.82 | [+0.94, +4.93] |
| Seed-sub − SC | 6 | +2.01 | [+0.61, +3.52] |
| SC@3N − SC | 6 | +1.20 | [+0.22, +2.18] |
| **CD − seed-sub** | 6 | **−0.42** | **[−1.20, +0.28]** |
| **CD − SC@3N** | 6 | **+0.39** | **[−0.11, +0.89]** |

Verdict unchanged: both key residuals cross zero → on controlled cells we
**cannot distinguish configs from seeds**; sample count explains the resolvable lift.
(`c3_controls.controls_summary` → `sample_count_confound`.)

> Note: means differ slightly from an earlier draft that also pooled `c3` on
> MATH-500 3B (where `c3` exists). Primary pool is `{c0,c1,c2}` to match the
> paper’s stated design.

### Expected tokens (probe-then-commit)

Always sample `c0`; sample `c1`/`c2` only when `margin(c0) < τ` (τ fit on train).

| Quantity | Mean over 14 cells |
|----------|--------------------|
| Fraction early-exit (`margin ≥ τ`) | **0.74** |
| Tokens(probe) / Tokens(SC@N) | **1.60×** |
| Tokens(probe) / Tokens(always-3-cfg) | **0.53×** |
| Tokens(always-3) / Tokens(SC@N) | ≈3.0× (by construction) |

So conditional diversify is **not** a free lunch vs SC@N (~1.6× tokens), but it
is roughly **half** the cost of always materialising three configs. High-accuracy
cells (7B GSM8K) early-exit ~99% of the time (~1.02× SC); weak cells (1.5B
Symbolic) early-exit ~25% and approach full 3-config cost.

### Independent margin-split

Margin from half of `c0` samples; SC / diversify decision on the held-out half
(other configs keep full samples). Same train/test item splits as main C3.

| Quantity | Mean (14 cells) | Cell-boot CI |
|----------|-----------------|--------------|
| CD(margin-split) − SC(half-B) | **+3.22 pp** | [+1.37, +5.46] |
| CD(full) − SC(full) | +2.29 pp | (matches controls) |
| Acc CD(split) − Acc CD(full) | ≈ −1.5 pp | — |

Lift over SC **survives** the independent margin estimate (not an artefact of
reusing the same samples for margin and decision). Absolute CD accuracy drops
~1–3 pp vs the full-sample aggregator, as expected from noisier half-B modes.

### Quality-matched / weighted pooling

| Method | Mean Δ vs SC@N (14 cells) |
|--------|---------------------------|
| CD over `{c0,c1,c2}` | +2.29 pp |
| CD over `{c0,c1}` only (drop `c2`) | **+3.34 pp** |
| Quality-weighted CD (train π weights) | +0.66 pp |
| Sample-level mixture plurality (3-cfg pool) | +0.58 pp |

On the **6 seed cells**:

| Contrast | Mean |
|----------|------|
| Mixture@3cfg − SC@3N(c0 seeds) | **−0.28 pp** |
| CD@3 − SC@3N | +0.39 pp |
| CD@3 − mixture | +0.67 pp |

Dropping weak `c2` **increases** the CD lift → `c2` is a quality confound for
unweighted plurality, not a free diversity source. Unconditional sample mixture
does **not** beat SC@3N on seed-matched cells; gated CD still edges mixture.

## Ready for 14 seed cells

`available_c0_seeds()` discovers whatever `c0` seeds exist with enough items.
When Symbolic / Llama seeds `{1,2}` (and later `{3,4}`) land in `results_fp16`,
re-running the two scripts above expands `has_seed_control` without code changes.
Contrast names in the summary CSV keep the historical `seed6_*` / `all14`
prefixes; `n_cells` reflects the live count.

## Blockers (GPU)

- Seed replicates still missing for Llama × {GSM8K, MATH-500} and all
  GSM-Symbolic cells (8 cells) — largest C1 spreads remain uncontrolled.
- Optional: `c0` seeds `{3,4}` on the existing 6 cells for a stabler seed-null.
- Live probe-then-commit inference (actual early-exit wall-clock) is not measured;
  token estimates assume offline corpus costs.
