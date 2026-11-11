# Adversarial ICML review — triage and revision plan

**Verdict on the review:** Fair Reject on the *current PDF* (draft artefacts). Not a “wrong paper” Reject. The research question and owned-negative frame are salvageable **if and only if Priority 1 lands before submission**.

**Do not pivot to pure theory.** The reviewer correctly says Props 1–2 are elementary. We added Prop.~3 (margin gate) as *interpretation*, not a theory paper. Theory-only would still get Reject without new math.

---

## What the reviewer got right (must fix)

| # | Issue | Our status | Action |
|---|--------|------------|--------|
| 3.1 | GLMM + bootstrap skipped | True for checked-in artefacts | **Priority 1:** camera-ready `paper` run |
| 3.2 | Plug-in modes noisy at N=24 | True | Depth-stability + O2 persistence (**no new GPU** if N=64 cells exist) |
| 3.3 | “No lift” ≠ equivalence | True | CI-TOST at δ=1pp wired into `compare_to_sc` / `cdv_summary` |
| 3.4 | Novelty rests on empirics | True | Narrow claims; keep Formal Picture interpretive |
| 3.5 | Transition denominators unclear | Fair | Documented: mean over ordered pairs of n_kind/n_items |
| 3.6 | Variance needs GLMM | Same as 3.1 | Camera-ready |
| 3.9 | Submission unfinished | True until P1 | Gate submission on camera-ready artefacts |

## What is overstated / already addressable in writing

| # | Issue | Response |
|---|--------|----------|
| 3.3 / Q2 | “CDV generally ineffective” | **Narrowed:** equal-budget CDV over {c0,c1,c2} only |
| 3.7 | Semantics / format shift | Log extraction failures; Limitations; full prompt texts already in elicitation module — cite appendix if space |
| 3.8 | External validity | Explicit scope to tested models/benchmarks (done in Results lead + Conclusion) |
| Q5 | Theory contribution | Prop.~3 margin gate; explicitly *not* a theory paper |
| Token matching | Actual vs intended | Method now states completion-token / offline allocator |

## Priority order (execution)

### Priority 1 — blocks acceptance (CPU only, hours)

```bash
python scripts/split_precision.py --src results_bundle
python -u -m src.analysis.paper \
  --results-dir results_fp16 \
  --precision-results-dir results_4bit \
  --out-dir paper_artefacts \
  --n-bootstrap 1000 \
  --n-repeats 50 \
  --no-png
# NO --skip-glmm
cp paper_artefacts/fig*.pdf paper/figures/
```

**Must appear in the submitted PDF:**

- Bootstrap CIs on every primary CDV−SC contrast (`tab5` comparison rows).
- `predictions.csv`: `cdv_null_status` ∈ {`equivalence_supported`, `no_lift_beyond_delta`, …}.
- VB-GLMM shares + CIs next to / instead of moment-only F1 language.
- 4-bit `ρ_disatt` if computable → resolve P7; else keep P7 unsupported.

**Gate:** do **not** upload a PDF that still confesses `--skip-glmm --n-bootstrap 0` as the analysis used.

### Priority 2 — mechanism credibility (CPU; uses existing chains)

Already implemented in analysis; filled by the same camera-ready run:

- `tab_depth_stability.csv` — maj@N / mode agreement vs full-N for N∈{8,16,24,48,64}.
- `tab_transition_persistence.csv` — corrective/destructive labels vs depth on O2 (N≥64).

**Paper:** cite these in Results §reordering + Limitations (already wired in `main.tex`).

Optional GPU (only if P1+P2 look good and time remains): deeper N on a Symbolic subset — *not* required to answer the review if O2 persistence is clean.

### Priority 3 — strengthen contribution (defer unless time)

| Ask | Cost | Decision |
|-----|------|----------|
| Validation-selected config baseline | Needs val split + analysis | Nice; not blocking if claims stay policy-narrow |
| Nonuniform CDV allocations | Offline reweight of existing pools | Medium; do after P1 |
| Reasoning-specialised / larger model | GPU heavy | Out of scope unless chasing Accept hard |
| Held-out margin→transition predictor | Analysis only | Good rebuttal figure; after P1 |
| Matched control prompts | New GPU | Defer; audit extraction failures first |

---

## Mapping reviewer Q1–Q7 → our answers after P1

| Q | Plan |
|---|------|
| Q1 CIs / GLMM | Camera-ready artefacts; report CIs in table notes / appendix |
| Q2 CDV null informative? | Equivalence δ=1pp; narrow to tested policy |
| Q3 Transitions vs depth | Persistence + depth tables |
| Q4 Predictive mechanism | Margin concentration already; held-out predictor = P3 |
| Q5 Theory | Prop.~3 + “interpretive, not theory paper” |
| Q6 Elicitation ops | Failure rates + Limitations; control prompts = P3 |
| Q7 Generalization | Explicit scope narrowing (done) |

---

## Honest acceptance outlook

| State | Likely outcome |
|-------|----------------|
| Submit current draft PDF | **Reject** (this review) |
| P1 + claim narrowing only | **Borderline / weak Accept shot** — still small models, selective C1 |
| P1 + P2 + clean equivalence on CDV | **Credible empirical negative** — competitive as careful negative+mechanism |
| P1–P3 + extra model | Stronger, not required for January if P1 slips |

**Bottom line:** The reviewer’s Reject is about *incomplete inference on a near-null*, not about owning the negative. Finish the pipeline; keep the owned frame; do not rewrite as theory.
