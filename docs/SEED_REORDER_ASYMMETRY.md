# Seed-pair reorder asymmetry diagnosis

From `tab_seed_reorder_baseline.csv`, seed-to-seed pairs show
destructive ≈ 4.4% vs corrective ≈ 3.6% under the lex-oriented baseline.
This note checks whether that gap is an artefact of pair orientation.

## Mechanism in the baseline code

`seed_reorder_baseline` iterates `combinations(sorted(seeds), 2)`, so every
pair is oriented **low seed index → high** (`seed0→seed1`, `seed0→seed2`,
`seed1→seed2`). `classify_transition` labels
**destructive** = correct on source, wrong on target, and
**corrective** = wrong on source, right on target.
Modal ties break by **first occurrence within a cell's sample list**
(not across seeds); the cross-seed asymmetry is from **pair orientation**,
not the within-cell tie-break.

## Mean c0 π_mode by seed (14 cells)

- seed 0: **74.12%**
- seed 1: **73.51%**
- seed 2: **72.85%**

Best mean seed: **0** (74.12%). If seed0 is better on average, lex orientation (seed0 as source on 2/3 of
pairs) inflates destructive relative to corrective.

## Oriented rates (pooled item×pair rows)

| Orientation | Destructive | Corrective | Net corr |
|-------------|------------:|-----------:|---------:|
| Lex (baseline, low→high) | 4.42% | 3.64% | -0.77% |
| Flipped (high→low) | 3.64% | 4.42% | +0.77% |
| Symmetric (both dirs) | 4.03% | 4.03% | +0.00% |

### Source = seed0 vs not (lex orientation)

- Pairs with source seed0 (`0→1`, `0→2`): dest 4.50% / corr 3.64%
- Pair `1→2` only: dest 4.24% / corr 3.64%

## Verdict

The dest>corr gap is **mostly an orientation artefact**: it flips under reversed pairs and collapses under symmetric averaging. Prefer reporting symmetric (or absolute) transition rates for seed-seed baselines, or state explicitly that pairs are oriented low→high seed index.
