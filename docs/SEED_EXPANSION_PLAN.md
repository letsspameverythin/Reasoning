# Seed expansion plan (review P0)

## Why only 6 cells today

Multi-seed $c_0$ was run only as a **noise floor** for the Qwen ladder on
GSM8K + MATH-500 (`seeds {0,1,2}`). Llama and all GSM-Symbolic cells were
seed-0 only. Seed-substitution / SC@3N therefore cannot cover the 8 cells
where C1 spreads are largest.

## What to run (priority order)

| Priority | Cells | Seeds | Why |
|----------|-------|-------|-----|
| P0 | Qwen × GSM-Symbolic main+p2 (6 cells) × $c_0$ | `{1,2}` (0 exists) | Unlocks controls where config spreads are 3–13 pp |
| P0 | Llama-3.2-3B × GSM8K + MATH-500 × $c_0$ | `{1,2}` | Completes the 14-cell C3 control grid |
| P1 | All 14 controlled cells × $c_0$ | add `{3,4}` → 5 seeds | Stabilizes seed-null / permutation null |
| P2 | Optional $c_1$/$c_2$ seeds on 1–2 high-spread cells | `{1,2}` | Quality-matched diversity check |

Each new seed = one `configured_sampling` run at $N{=}24$ on the cell’s item
subsample (150 Symbolic / 200 MATH/GSM8K). Inference-only; no new prompts.

## Exact launch package (this repo)

Generated artefacts (do **not** hand-edit; regenerate):

```bash
python scripts/generate_seed_expansion_configs.py
python scripts/run_seed_expansion_queue.py --dry-run
python scripts/run_seed_expansion_queue.py --only p0 --dry-run
```

| Bucket | Runs | Est. GPU-h | Notes |
|--------|-----:|-----------:|-------|
| **P0** | **16** | **~44.9** | 6 Symbolic cells × seeds {1,2} + Llama × {gsm8k, math500} × {1,2} |
| **P1** | **28** | **~105.2** | 14 control cells × seeds {3,4} |
| **Total** | **44** | **~150.1** | configs under `configs/seed_expansion/` |

Estimates = mean `gpu_hours` (from `elapsed_seconds`) on matching existing
`results_fp16` c0 manifests per (model, dataset); MATH-500 / Qwen-3B uses the
**median** to down-weight one anomalous short s0. Queue detail:
`configs/seed_expansion/queue.json`.

### P0 stem list (16)

- `qwen{1.5b,3b,7b}__gsm_symbolic_{main,p2}__c0__s{1,2}` (12)
- `llama3b__{gsm8k,math500}__c0__s{1,2}` (4)

### P1 stem list (28)

All 14 cells × `s{3,4}`:

- Qwen{1.5B,3B,7B} × {gsm8k, math500, gsm_symbolic main, gsm_symbolic p2}
- Llama-3.2-3B × {gsm8k, math500}

### Conventions (hash-matched to existing c0)

- `strategy: configured_sampling` with `n_samples: 24`, `batch_calls: true`
- `elicitation.id: c0` (decode T=0.8, top_p=0.95 via elicitation overrides)
- YAML `generation.temperature: 0.8`, `max_new_tokens: 1024` (top_p left at
  base 1.0 so `config_hash` matches existing seed-0 dirs)
- `subsample_seed: 1234`; subsample 150 (Symbolic) / 200 (GSM8K, MATH-500)
- `runtime.out_dir: results_fp16`
- Defaults inheritance: `defaults: ../base.yaml` (configs live in a subfolder)

Kaggle copy-paste: [`notebooks/seed_expansion.md`](../notebooks/seed_expansion.md).

## Resume instructions

1. Unpack any prior `results_fp16` zip into the same `out_dir` the YAML uses
   (`results_fp16` or `/kaggle/working/results_fp16`).
2. Re-run the **identical** command:
   `python -m src.runner --config configs/seed_expansion/<stem>.yaml`
3. The runner skips finished example uids; wall-clock stops cleanly under
   `time_budget_hours: 11.0` before the Kaggle 12 h kill.
4. Zip `results_fp16` at the end of every session; attach the zip next session.
5. Prefer `python scripts/run_seed_expansion_queue.py --only p0 --dry-run` to
   list remaining work; pass `--execute` only on a GPU session.

Do **not** change semantic fields (model, subsample, N, elicitation) mid-run —
that creates a new `config_hash` and a new directory.

## After grading

```bash
python -m src.analysis.paper --results-dir results_fp16 --out-dir paper_artefacts
python scripts/run_c3_controls.py --results-dir results_fp16
python scripts/run_p4_seed_calibration.py
python scripts/run_token_pareto.py
```

Re-lock Table `tab:c3-controls` on **identical 14-cell sets**, and report P4
as “survives seed-null permutation” not just “>3 pp.”

## Parallel CPU (already runnable)

- `scripts/run_p4_seed_calibration.py` — config range vs seed-null
- `scripts/run_token_pareto.py` — SC / ESC / certainty / CDV vs actual tokens
