# P4 combined test (seed-calibrated, CPU)

Global follow-up on the 14 multi-seed cells. Per-cell ranges and the
official survivor flag come from `tab_p4_seed_calibration.csv`.
Per-cell null reported here: **exact with-replacement** assignment of the
3 observed c0 seed π_mode values to 3 fake-config slots (3³ = **27** atoms).
Survivors require config range > 3pp and one-sided p < 0.05 under the
calibration seed-null. With only 3 seeds, `config_range > seed_range`
implies exact p = 0 (no null atom exceeds the observed seed range).
Round-7 text sometimes says "20-permutation"; the exact catalogue for
|S|=3 is 27 (atom mass 1/27 ≈ 0.037).

```bash
PYTHONPATH=. python scripts/run_round7_cpu_followups.py --results-dir results_fp16
```

## Headline

- Survivors: **5/14** at α=0.05 (calibration gate).
- Cells with `config_range > seed_range`: **8/14**.
- Mean(`config_range − seed_range`): **+1.98 pp**.
- Reference under independent Bernoulli null: Pr(≥5 survivors | Bin(14, α=0.05)) ≈ **0.0004274**.
- Item-bootstrap (B=1000, template/item units): n_survivors mean **7.05** 95% CI [4, 10]; mean(config−seed) **+1.99** pp [+0.50, +3.43].
- Fisher combination of the 14 exact one-sided p-values (zeros clamped to 1e-12): stat=445.2, p≈1.17e-76 (sensitivity only; cells are not independent).

## Per-cell survivors

| Cell | cfg pp | seed pp | Δ pp | p_exact27 |
|------|-------:|--------:|-----:|----------:|
| 1.5B|gsm8k|main | 6.50 | 4.13 | +2.37 | 0.000 |
| 7B|gsm8k|main | 6.50 | 0.50 | +6.00 | 0.000 |
| 3B|gsm_symbolic|main | 10.67 | 3.33 | +7.33 | 0.000 |
| 7B|gsm_symbolic|main | 13.33 | 0.67 | +12.67 | 0.000 |
| 7B|gsm_symbolic|p2 | 6.00 | 1.33 | +4.67 | 0.000 |

## Exchangeability (recommended language)

Treating c0 seeds as exchangeable "fake configs" is a **calibration**
device, not a proof that decoding configs are exchangeable with seeds.
Seeds share the same prompt/decoding family; configs deliberately change
elicitation. If that distinction makes the null dubious, report P4 as a
**descriptive seed-calibrated comparison**: config π_mode range vs the
observed c0 seed-to-seed range (and the fraction of cells where
config_range > seed_range), rather than as a formal p-value rejection of
an exchangeability null. The binomial Pr(≥k) line is a **reference** under
independent α-level tests — not a joint claim about the 14 cells.

## Artefact

- `paper_artefacts/tab_p4_combined.csv`
