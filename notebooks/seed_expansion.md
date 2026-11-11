# Seed expansion on Kaggle (c0 multi-seed)

Copy-paste cells for launching the missing multi-seed $c_0$ runs. Keep the
accelerator off until the dry-run queue looks right.

**Queue size:** 44 runs total — **P0 = 16** (~45 GPU-h), **P1 = 28** (~105 GPU-h).
Estimates come from `elapsed_seconds` / `gpu_hours` on matching existing
`results_fp16` c0 manifests.

Configs live under `configs/seed_expansion/` (each uses `defaults: ../base.yaml`;
regenerate with `python scripts/generate_seed_expansion_configs.py`).

---

## Cell A — locate harness + set paths

```python
import glob, os, shutil, subprocess, sys
from pathlib import Path

WORKDIR = Path("/kaggle/working") if Path("/kaggle/working").is_dir() else Path.cwd()
RESULTS = WORKDIR / "results_fp16"
RESULTS.mkdir(parents=True, exist_ok=True)

HARNESS = None
for cand in (Path.cwd(), WORKDIR / "harness"):
    if (cand / "src" / "runner.py").exists():
        HARNESS = cand
        break
if HARNESS is None:
    hits = sorted(glob.glob("/kaggle/input/*/src/runner.py")) + sorted(
        glob.glob("/kaggle/input/*/*/src/runner.py")
    )
    if hits:
        src_root = Path(hits[0]).parents[1]
        HARNESS = WORKDIR / "harness"
        shutil.copytree(src_root, HARNESS, dirs_exist_ok=True)

assert HARNESS is not None, "attach the repo as a Kaggle Dataset or clone it"
os.chdir(HARNESS)
sys.path.insert(0, str(HARNESS))
print("HARNESS", HARNESS)
print("RESULTS", RESULTS)
print(subprocess.run(["nvidia-smi"], capture_output=True, text=True).stdout or "NO GPU")
```

---

## Cell B — install deps (once per session)

```python
%pip install -q -r requirements.txt
```

---

## Cell C — regenerate configs + dry-run queue (CPU)

```python
!python scripts/generate_seed_expansion_configs.py
!python scripts/run_seed_expansion_queue.py --only p0 --dry-run
# !python scripts/run_seed_expansion_queue.py --only p1 --dry-run
```

Override `runtime.out_dir` for Kaggle working disk if needed (configs default to
`results_fp16` relative to the harness cwd). To force `/kaggle/working/results_fp16`:

```python
# optional: rewrite out_dir in every seed-expansion YAML for this session
from pathlib import Path
for p in Path("configs/seed_expansion").glob("*.yaml"):
    text = p.read_text(encoding="utf-8")
    if "out_dir: results_fp16" in text:
        p.write_text(
            text.replace("out_dir: results_fp16", "out_dir: /kaggle/working/results_fp16"),
            encoding="utf-8",
        )
print("rewrote out_dir -> /kaggle/working/results_fp16")
```

---

## Cell D — run P0 (GPU) one session at a time

Kaggle sessions die at ~12 h. Prefer running **one or two configs** per session
and zipping results. Re-running the **same** command resumes automatically.

```python
# Example: first P0 Symbolic main 1.5B seed 1
!python -m src.runner --config configs/seed_expansion/qwen1.5b__gsm_symbolic_main__c0__s1.yaml

# Or walk the P0 queue until the wall-clock budget trips (each run has
# time_budget_hours: 11.0). Stop between runs and zip if disk is tight.
# !python scripts/run_seed_expansion_queue.py --only p0 --execute --stop-on-error
```

---

## Cell E — after P0 completes, run P1 seeds {3,4}

```python
!python scripts/run_seed_expansion_queue.py --only p1 --dry-run
# !python scripts/run_seed_expansion_queue.py --only p1 --execute --stop-on-error
```

---

## Cell F — zip + carry forward (resume across sessions)

```python
import shutil
from pathlib import Path

src = Path("/kaggle/working/results_fp16")
if not src.is_dir():
    src = Path("results_fp16")
out = Path("/kaggle/working") / "seed_expansion_results_fp16"
shutil.make_archive(str(out), "zip", root_dir=src)
print("wrote", out.with_suffix(".zip"), "bytes", out.with_suffix(".zip").stat().st_size)
```

**Resume:** attach the previous zip as a Dataset (or unpack into
`/kaggle/working/results_fp16`), then re-run the **identical**
`python -m src.runner --config ...` command. Finished example uids are skipped.

---

## Cell G — post-grade analysis (after all seeds land)

```bash
python -m src.analysis.paper --results-dir results_fp16 --out-dir paper_artefacts
python scripts/run_c3_controls.py --results-dir results_fp16
python scripts/run_p4_seed_calibration.py
python scripts/run_token_pareto.py
```
